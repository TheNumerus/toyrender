use crate::args::Args;
use crate::err::AppError;
use crate::image::ImageResource;
use crate::import;
use crate::import::ImportedScene;
use crate::input::InputMapper;
use crate::renderer::{FrameContext, ResourceSubsystem, VulkanContext, VulkanMcPathTracer, VulkanRenderer};
use crate::scene::{Scene, SkyVariant};
use image::DynamicImage;
use imgui::Ui;
use log::info;
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use sdl2::mouse::MouseButton;
use sdl2::video::Window;
use sdl2::{EventPump, Sdl};
use std::cell::RefCell;
use std::cmp::PartialEq;
use std::fs::File;
use std::path::Path;
use std::rc::Rc;
use std::str::FromStr;
use std::time::Instant;
use zip::ZipArchive;

pub(crate) mod shader_loader;
use shader_loader::ShaderLoader;

pub(crate) mod frame_stats;
use crate::app::frame_stats::StatStorage;
use frame_stats::FrameStats;

pub struct App {
    pub vulkan_context: Rc<VulkanContext>,
    pub renderer: VulkanRenderer,
    pub reference_renderer: VulkanMcPathTracer,
    pub resource_subsystem: ResourceSubsystem,
    pub sdl_context: Sdl,
    pub window: Window,
    pub event_pump: EventPump,
    pub input_mapper: InputMapper<InputAxes>,
    pub scene: Scene,
    pub imgui: RefCell<imgui::Context>,
}

impl App {
    pub fn create() -> Result<Self, AppError> {
        let sdl_context = sdl2::init().expect("cannot init sdl2");

        let video_subsystem = sdl_context.video().expect("cannot init video");

        let mut window = video_subsystem
            .window("Vulkan Demo", 1920, 1080)
            .allow_highdpi()
            .resizable()
            .position_centered()
            .vulkan()
            .build()
            .expect("cannot build window");

        window.set_minimum_size(120, 40).expect("cannot set min size");

        let event_pump = sdl_context.event_pump().expect("cannot get event pump");

        let mut imgui = imgui::Context::create();
        Self::set_ui_style(imgui.style_mut());

        let shader_loader = ShaderLoader::from_zip(open_shader_zip("shaders.zip")?)?;

        let vulkan_context = Rc::new(VulkanContext::init(&window, &mut imgui, shader_loader)?);
        let resource_subsystem = ResourceSubsystem::new(vulkan_context.clone());
        let renderer = VulkanRenderer::init(vulkan_context.clone())?;
        let reference_renderer = VulkanMcPathTracer::init(vulkan_context.clone())?;

        let input_mapper = Self::setup_input_mapper();

        let scene = Scene::new();

        Ok(Self {
            sdl_context,
            window,
            event_pump,
            renderer,
            reference_renderer,
            resource_subsystem,
            vulkan_context,
            input_mapper,
            scene,
            imgui: RefCell::new(imgui),
        })
    }

    pub fn run(mut self, args: Args) -> Result<(), AppError> {
        if let Some(path) = args.file_to_open {
            info!("loading file `{:?}`", path);
            let start = Instant::now();

            let file = std::fs::read(&path).map_err(|e| {
                let msg = format!("file {} cannot be read: {e}", path.to_string_lossy());

                AppError::Import(msg)
            })?;

            let ImportedScene {
                resources: _resources,
                instances,
                camera,
            } = import::extract_scene(&file)?;
            self.scene.meshes.extend(instances);

            let end = Instant::now();

            info!("Loaded in {} s", (end - start).as_secs_f32());

            if let Some(camera) = camera {
                self.scene.camera.fov = camera.fov;
                self.scene.camera.position = camera.position;
                self.scene.camera.rotation = camera.rotation;
            }
        }

        info!(
            "{} pipelines created",
            self.vulkan_context.pipeline_builder.borrow().get_pipeline_count()
        );

        if args.benchmark {
            self.benchmark(300)?;
            return Ok(());
        }

        let start = Instant::now();
        let mut frame_end = Instant::now();

        let mut state = AppState::new();

        let mouse_sens = 0.002;
        let scroll_sens = 2.5;
        let movement_speed = 16.0;
        let mut focused = true;
        let mut taa_enable = true;
        let mut culling = true;

        let mut sel_render = 1;
        let mut renderer_changed = false;

        let mut frame = 1;

        let mut platform = imgui_sdl2_support::SdlPlatform::new(&mut self.imgui.borrow_mut());

        let mut frame_stats = FrameStats::new(20);

        'running: loop {
            let mut resized = false;

            let frame_start = Instant::now();
            let delta = frame_start.duration_since(frame_end).as_secs_f32();

            let mut mouse = (0, 0);
            let mut mouse_scroll = 0.0;
            let mut dragging;
            let mut bounce_adjust: i32 = 0;
            let mut exposure_adjust = 0.0;
            let mut clear_taa = false;
            let mut debug_mode_flip = false;
            let mut movement = false;

            for event in self.event_pump.poll_iter() {
                platform.handle_event(&mut self.imgui.borrow_mut(), &event);
                match event {
                    Event::Quit { .. } => {
                        self.renderer.device.wait_idle()?;
                        break 'running;
                    }
                    Event::Window {
                        win_event: WindowEvent::Resized(_, _),
                        ..
                    } => {
                        resized = true;
                    }
                    Event::Window {
                        win_event: WindowEvent::FocusGained,
                        ..
                    } => {
                        focused = true;
                    }
                    Event::Window {
                        win_event: WindowEvent::FocusLost,
                        ..
                    } => {
                        focused = false;
                    }
                    Event::DropFile { filename, .. } => {
                        let start = Instant::now();

                        let action = Self::on_file_drop(filename)?;

                        info!("Loaded in {} s", start.elapsed().as_secs_f32());

                        match action {
                            FileDroppedAction::LoadScene(ls) => {
                                self.scene.meshes.extend(ls.instances);
                            }
                            FileDroppedAction::LoadImage(i) => {
                                self.scene.env.sky.variant = SkyVariant::Textured(ImageResource::new(i, "sky texture"));
                            }
                        }
                    }
                    Event::MouseWheel { y, .. } => {
                        mouse_scroll = y as f32 * scroll_sens;
                        movement = true;
                    }
                    Event::MouseMotion {
                        xrel, yrel, mousestate, ..
                    } => {
                        dragging = mousestate.is_mouse_button_pressed(MouseButton::Right);
                        self.sdl_context.mouse().set_relative_mouse_mode(dragging);

                        if dragging {
                            mouse.0 += xrel;
                            mouse.1 += yrel;
                            movement = true;
                        } else {
                            self.sdl_context.mouse().show_cursor(true);
                        }
                    }
                    Event::KeyDown { keycode, .. } => match keycode {
                        Some(Keycode::LeftBracket) => bounce_adjust = -1,
                        Some(Keycode::RightBracket) => bounce_adjust = 1,
                        Some(Keycode::R) => {
                            debug_mode_flip = true;
                            clear_taa = true;
                        }
                        Some(Keycode::KpPlus) => {
                            exposure_adjust += 0.5;
                        }
                        Some(Keycode::KpMinus) => {
                            exposure_adjust -= 0.5;
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }

            self.input_mapper.update(self.event_pump.keyboard_state());

            let directions = self.scene.camera.directions();

            self.scene.camera.fov += mouse_scroll;
            self.scene.camera.position += (self.input_mapper.get_value(InputAxes::Up) * directions.up
                + self.input_mapper.get_value(InputAxes::Forward) * directions.forward
                + self.input_mapper.get_value(InputAxes::Right) * directions.right)
                * delta
                * movement_speed;

            self.scene.camera.rotation.z -= mouse.0 as f32 * mouse_sens;
            self.scene.camera.rotation.x -= mouse.1 as f32 * mouse_sens;

            if self.input_mapper.inner_state.values().any(|a| *a != 0.0) {
                movement = true;
            }

            if state.selected_renderer == SelectedRenderer::Reference && movement {
                clear_taa = true;
            }

            if clear_taa {
                frame = 0;
            }

            let mut context = FrameContext {
                delta_time: delta,
                total_time: frame_end.duration_since(start).as_secs_f32(),
                clear_taa: resized || clear_taa || frame == 0 || !taa_enable,
                frame_index: frame,
                culling,
            };

            if bounce_adjust != 0 {
                let new_bounces = (self.renderer.quality.pt_bounces as i32 + bounce_adjust).max(0) as u32;
                self.renderer.quality.pt_bounces = new_bounces;
                info!("new bounce count: {new_bounces}");
            }

            self.scene.env.exposure = (self.scene.env.exposure + exposure_adjust).clamp(-32.0, 32.0);

            if debug_mode_flip {
                self.renderer.debug_mode = self.renderer.debug_mode.next();
                self.reference_renderer.debug_mode = self.renderer.debug_mode;
                eprintln!("debug mode: {:?}", self.renderer.debug_mode);
            }

            if resized {
                self.renderer.resize(self.window.drawable_size())?;
                self.reference_renderer.resize(self.window.drawable_size())?;
            }

            frame_end = Instant::now();

            platform.prepare_frame(&mut self.imgui.borrow_mut(), &self.window, &self.event_pump);

            let mut imgui_ref = self.imgui.borrow_mut();
            let ui = imgui_ref.new_frame();
            let mut window = ui
                .window("toyrender controls")
                .size([300.0, 100.0], imgui::Condition::FirstUseEver)
                .begin();
            if window.is_some() {
                if ui.combo("Renderer", &mut sel_render, &["Realtime", "Reference"], |a| {
                    std::borrow::Cow::Borrowed(a)
                }) {
                    state.selected_renderer = match sel_render {
                        0 => SelectedRenderer::Realtime,
                        1 => SelectedRenderer::Reference,
                        _ => unreachable!(),
                    };
                    renderer_changed = true;
                } else {
                    renderer_changed = false;
                }

                if ui.collapsing_header("RT Settings", imgui::TreeNodeFlags::DEFAULT_OPEN) {
                    if ui.slider(
                        "Direct trace distance",
                        0.0,
                        500.0,
                        &mut self.renderer.quality.rt_direct_trace_distance,
                    ) {
                        self.reference_renderer.quality.rt_direct_trace_distance =
                            self.renderer.quality.rt_direct_trace_distance;
                    }
                    if ui.slider(
                        "Indirect trace distance",
                        0.0,
                        500.0,
                        &mut self.renderer.quality.rt_indirect_trace_distance,
                    ) {
                        self.reference_renderer.quality.rt_indirect_trace_distance =
                            self.renderer.quality.rt_indirect_trace_distance;
                    }
                    if ui.slider("Bounce count", 0, 10, &mut self.renderer.quality.pt_bounces) {
                        self.reference_renderer.quality.pt_bounces = self.renderer.quality.pt_bounces;
                    }

                    if ui.slider(
                        "Indirect intensity clamp",
                        0.0,
                        100.0,
                        &mut self.renderer.quality.indirect_light_clamp,
                    ) {
                        self.reference_renderer.quality.indirect_light_clamp =
                            self.renderer.quality.indirect_light_clamp;
                    }

                    ui.checkbox("Temporal accumulation", &mut taa_enable);
                    if let SelectedRenderer::Realtime = state.selected_renderer {
                        ui.checkbox("Spatial denoise", &mut self.renderer.quality.use_spatial_denoise);
                        ui.checkbox("Culling", &mut culling);
                    }

                    ui.input_float3("Camera position", self.scene.camera.position.as_mut())
                        .build();
                    ui.input_float3("Camera rotation", self.scene.camera.rotation.as_mut())
                        .build();
                    ui.slider("Camera FoV", 1.0, 174.0, &mut self.scene.camera.fov);
                    if ui.slider("Render scale", 0.01, 1.0, &mut self.renderer.render_scale) {
                        context.clear_taa = true;
                        self.reference_renderer.render_scale = self.renderer.render_scale;
                    }
                }
                if ui.collapsing_header("Environment", imgui::TreeNodeFlags::DEFAULT_OPEN) {
                    ui.slider("Exposure", -10.0, 10.0, &mut self.scene.env.exposure);
                    ui.slider("Sun intensity", 0.0, 10.0, &mut self.scene.env.sun_intensity);
                    ui.slider("Sky intensity", 0.0, 10.0, &mut self.scene.env.sky.intensity);
                    ui.input_float3("Sun direction", self.scene.env.sun_direction.as_mut())
                        .build();
                    ui.slider(
                        "Sun angle",
                        0.0,
                        std::f32::consts::FRAC_PI_2,
                        &mut self.scene.env.sun_angle,
                    );
                    ui.color_edit3("Sun color", self.scene.env.sun_color.as_mut());
                }
                if ui.collapsing_header("Stats", imgui::TreeNodeFlags::DEFAULT_OPEN) {
                    self.stats_tab(ui, &frame_stats, delta);
                }
                if ui.collapsing_header("Scene", imgui::TreeNodeFlags::empty()) {
                    for (index, mesh) in &mut self.scene.meshes.iter_mut().enumerate() {
                        ui.text(format!("'{}'", mesh.resource.name));
                        ui.same_line();
                        ui.checkbox(format!("{}, visible", index), &mut mesh.visible);
                        ui.same_line();
                        ui.text(format!(
                            "'{:?}'",
                            mesh.resource.culling_info.bb_max - mesh.resource.culling_info.bb_min
                        ));
                    }
                }
            }

            if window.is_some() {
                window.take().unwrap().end();
            }
            drop(window);
            drop(imgui_ref);

            let mut imgui_ref = self.imgui.borrow_mut();
            let draw_data = imgui_ref.render();

            if renderer_changed {
                self.vulkan_context.device.wait_idle()?;

                context.clear_taa = true;
                context.frame_index = 0;
                frame = 0;
            }

            let report = match state.selected_renderer {
                SelectedRenderer::Realtime => self.renderer.render_frame(
                    &self.scene,
                    &mut self.resource_subsystem,
                    self.window.drawable_size(),
                    &context,
                    Some(draw_data),
                )?,
                SelectedRenderer::Reference => self.reference_renderer.render_frame(
                    &self.scene,
                    &mut self.resource_subsystem,
                    self.window.drawable_size(),
                    &context,
                    Some(draw_data),
                )?,
            };

            frame_stats.update(report);

            if !focused {
                let frametime_target = 1.0 / 30.0;

                if delta < frametime_target {
                    //std::thread::sleep(Duration::from_secs_f32(frametime_target - delta));
                }
            };
            frame += 1;
        }
        eprintln!();

        Ok(())
    }

    fn stats_tab(&self, ui: &Ui, frame_stats: &FrameStats, delta: f32) {
        let stats = frame_stats.compute();

        ui.text(format!("FPS: {:>8.3} ms", 1.0 / delta));
        ui.text(format!("Frame time: {:>8.3} ms", delta * 1000.0));

        for (desc, value) in stats.iter() {
            match value {
                StatStorage::Int(i) => {
                    ui.text(format!("{}: {}", desc, i.latest));
                }
                StatStorage::Float(f) => {
                    ui.text(format!("{}: {:>8.3}", desc, f.avg));
                }
            }
        }
    }

    fn on_file_drop(filename: String) -> Result<FileDroppedAction, AppError> {
        info!("loading file `{filename}`");

        let file = std::fs::read(&filename).map_err(|e| {
            let msg = format!("file {} cannot be read: {e}", filename);

            AppError::Import(msg)
        })?;
        let path = std::path::PathBuf::from_str(&filename).unwrap();

        match path.extension().map(|ext| ext.to_str().unwrap()) {
            Some("glb") => Ok(FileDroppedAction::LoadScene(import::extract_scene(&file)?)),
            Some("exr") | Some("hdr") => Ok(FileDroppedAction::LoadImage(image::load_from_memory(&file)?)),
            _ => Err(AppError::Import("Unknown file format".to_owned())),
        }
    }

    pub fn setup_input_mapper() -> InputMapper<InputAxes> {
        use sdl2::keyboard::Scancode;

        InputMapper::with_configuration([
            (Scancode::W, vec![(InputAxes::Forward, -1.0)]),
            (Scancode::S, vec![(InputAxes::Forward, 1.0)]),
            (Scancode::A, vec![(InputAxes::Right, -1.0)]),
            (Scancode::D, vec![(InputAxes::Right, 1.0)]),
            (Scancode::Q, vec![(InputAxes::Up, -1.0)]),
            (Scancode::E, vec![(InputAxes::Up, 1.0)]),
        ])
    }

    fn benchmark(&mut self, frames: usize) -> Result<(), AppError> {
        let start = Instant::now();
        let mut bench_start = Instant::now();

        // skip first few frames, for increased precision
        for frame in 0..(frames + 100) {
            for _event in self.event_pump.poll_iter() {}

            let frame_start = Instant::now();

            if frame == 100 {
                bench_start = Instant::now();
            }

            let context = FrameContext {
                delta_time: 0.016,
                total_time: frame_start.duration_since(start).as_secs_f32(),
                clear_taa: false,
                frame_index: frame as u32,
                culling: true,
            };

            self.renderer.render_frame(
                &self.scene,
                &mut self.resource_subsystem,
                self.window.drawable_size(),
                &context,
                None,
            )?;
        }

        self.renderer.device.wait_idle()?;

        let end = Instant::now();

        let total_time = end.duration_since(bench_start).as_secs_f32();
        let avg = total_time * 1000.0 / (frames as f32);

        println!("BENCHMARK RESULT\n");
        println!("Total frames: {frames}");
        println!("Total time (s): {:.3}", total_time);
        println!("Average time per frame (ms): {avg}");

        Ok(())
    }

    fn set_ui_style(style: &mut imgui::Style) {
        style.use_dark_colors();
        style.window_rounding = 4.0;
        style.tab_rounding = 4.0;
        style.frame_rounding = 4.0;

        style.colors[imgui::StyleColor::WindowBg as usize] = [0.02, 0.02, 0.02, 0.9];
    }
}

fn open_shader_zip(path: impl AsRef<Path>) -> Result<ZipArchive<File>, AppError> {
    let mut base =
        std::env::current_exe().map_err(|_| AppError::Other("Cannot get current path to executable".into()))?;

    base.pop();
    base.push(path);

    info!("Loading shaders from {:?}", base);

    let file = File::open(&base).map_err(|e| AppError::Other(format!("Cannot open shaders library: {e}")))?;
    let arch = ZipArchive::new(file).map_err(|e| AppError::Other(format!("Cannot read shaders library: {e}")))?;

    Ok(arch)
}

pub struct AppState {
    selected_renderer: SelectedRenderer,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            selected_renderer: SelectedRenderer::Reference,
        }
    }
}

#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub enum InputAxes {
    Forward,
    Right,
    Up,
}

#[derive(PartialEq)]
enum SelectedRenderer {
    Realtime,
    Reference,
}

enum FileDroppedAction {
    LoadScene(ImportedScene),
    LoadImage(DynamicImage),
}
