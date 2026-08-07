use crate::args::Args;
use crate::debug;
use crate::err::AppError;
use crate::image::ImageResource;
use crate::import;
use crate::import::ImportedScene;
use crate::input::InputMapper;
use crate::renderer::{
    ConvolutionPassState, FrameContext, ResourceSubsystem, VulkanContext, VulkanMcPathTracer, VulkanRenderer,
};
use crate::scene::{Scene, SkyVariant};

use image::DynamicImage;

use log::info;

use nalgebra_glm::{Vec3, vec3};

use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use sdl2::mouse::MouseButton;
use sdl2::video::Window;
use sdl2::{EventPump, Sdl};

use std::cmp::PartialEq;
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::fs::File;
use std::path::Path;
use std::rc::Rc;
use std::slice::Iter;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::thread;
use std::time::Instant;
use uuid::Uuid;

use zip::ZipArchive;

pub(crate) mod gui;

pub(crate) mod shader_loader;
use shader_loader::ShaderLoader;

pub(crate) mod frame_stats;
use crate::app::frame_stats::StatStorage;
use frame_stats::FrameStats;

static FONT: &[u8] = include_bytes!("../assets/Inter-Regular.ttf");

static LOADING: AtomicBool = AtomicBool::new(false);

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
    pub imgui: imgui::Context,
}

impl App {
    pub fn create() -> Result<Self, AppError> {
        let sdl_context = sdl2::init().expect("cannot init sdl2");

        let video_subsystem = sdl_context.video().expect("cannot init video");

        let mut window = video_subsystem
            .window("Toyrender", 1920, 1080)
            .allow_highdpi()
            .resizable()
            .position_centered()
            .vulkan()
            .build()
            .expect("cannot build window");

        window.set_minimum_size(120, 40).expect("cannot set min size");

        let event_pump = sdl_context.event_pump().expect("cannot get event pump");

        let mut imgui = imgui::Context::create();
        imgui.io_mut().config_flags |= imgui::ConfigFlags::DOCKING_ENABLE;
        imgui.fonts().add_font(&[imgui::FontSource::TtfData {
            data: FONT,
            config: None,
            size_pixels: 14.0,
        }]);
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
            imgui,
        })
    }

    pub fn run(mut self, args: Args) -> Result<(), AppError> {
        // Channels for non-blocking tasks
        let (tx, rx) = std::sync::mpsc::channel();
        let (tx_reply, rx_reply) = std::sync::mpsc::channel();
        let handle = thread::spawn(move || {
            loader_thread(rx, tx_reply);
        });

        let mut textures: Vec<ImageResource> = Vec::new();

        let mut gizmo_scene = import::extract_scene(debug::GIZMO_SUN_SCENE)?;
        let sun_gizmo = gizmo_scene.resources.pop().unwrap();

        let mut gizmo_scene = import::extract_scene(debug::GIZMO_ARROW_SCENE)?;
        let arrow_gizmo = gizmo_scene.resources.pop().unwrap();

        //TODO move this elsewhere
        self.resource_subsystem
            .init_gizmo_meshes(&self.reference_renderer.tlas_prepare_cmd_buf, &[sun_gizmo, arrow_gizmo])?;

        if let Some(path) = args.file_to_open {
            tx.send(ThreadAction::OpenScene {
                filename: path.to_string_lossy().to_string(),
            })
            .unwrap();
        }

        info!(
            "{} pipelines created in {:.3} s",
            self.vulkan_context.pipeline_builder.borrow().get_pipeline_count(),
            self.vulkan_context.pipeline_builder.borrow().total_time_compiling
        );

        if args.benchmark {
            self.benchmark(300)?;
            return Ok(());
        }

        let mut camera_set = false;

        let start = Instant::now();
        let mut frame_end = Instant::now();

        let mut state = AppState::new();

        let mouse_sens = 0.002;
        let scroll_sens = 2.5;
        let movement_speed = 16.0;
        let mut focused = true;
        let mut taa_enable = true;

        let mut sel_sky = 0;

        let mut frame = 1;

        let mut platform = imgui_sdl2_support::SdlPlatform::new(&mut self.imgui);

        let mut frame_stats = FrameStats::new(20);

        let mut sky_textures: HashMap<Uuid, Rc<ImageResource>> = HashMap::new();
        let mut selected_texture = None;

        let mut values = VecDeque::with_capacity(100);

        let mut messages = BTreeSet::new();

        let mut last_sun_sum = vec3(0.0, 0.0, 0.0);
        let mut last_sum = vec3(0.0, 0.0, 0.0);

        // need to do this because of borrowing
        let Self {
            mut imgui,
            mut scene,
            mut input_mapper,
            mut window,
            sdl_context,
            mut resource_subsystem,
            mut event_pump,
            mut reference_renderer,
            mut renderer,
            vulkan_context,
        } = self;

        'running: loop {
            let mut resized = false;

            let frame_start = Instant::now();
            let delta = frame_start.duration_since(frame_end).as_secs_f32();

            let mut mouse = (0, 0);
            let mut mouse_scroll = 0.0;
            let mut dragging;
            let mut clear_taa = false;
            let mut debug_mode_flip = false;
            let mut ui_focused = false;

            messages.clear();

            for event in event_pump.poll_iter() {
                if state.ui_visible {
                    platform.handle_event(&mut imgui, &event);
                } else {
                    ui_focused = false;
                }
                match event {
                    Event::Quit { .. } => {
                        renderer.device.wait_idle()?;
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
                        let action = Self::on_file_drop(filename)?;

                        match action {
                            FileDroppedAction::LoadScene { filename } => {
                                tx.send(ThreadAction::OpenScene { filename }).unwrap();
                            }
                            FileDroppedAction::LoadImage { filename } => {
                                tx.send(ThreadAction::DecodeImage { filename }).unwrap();
                            }
                        }
                    }
                    Event::MouseWheel { y, .. } => {
                        mouse_scroll = y as f32 * scroll_sens;
                        messages.insert(UiMessage::ReferenceRenderReset);
                    }
                    Event::MouseMotion {
                        xrel, yrel, mousestate, ..
                    } => {
                        dragging = mousestate.is_mouse_button_pressed(MouseButton::Right);
                        sdl_context.mouse().set_relative_mouse_mode(dragging);

                        if dragging {
                            mouse.0 += xrel;
                            mouse.1 += yrel;
                            messages.insert(UiMessage::ReferenceRenderReset);
                        } else {
                            sdl_context.mouse().show_cursor(true);
                        }
                    }
                    Event::KeyDown { keycode, .. } => match keycode {
                        Some(Keycode::R) => {
                            debug_mode_flip = true;
                            messages.insert(UiMessage::CompleteRenderReset);
                        }
                        Some(Keycode::H) => {
                            state.ui_visible = !state.ui_visible;
                        }
                        Some(Keycode::F10) => {
                            messages.insert(UiMessage::SaveScreen);
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }

            if let Ok(a) = rx_reply.try_recv() {
                match a {
                    ThreadReply::Err(e) => {
                        return Err(e);
                    }
                    ThreadReply::DecodeImage { data, name } => {
                        let resource = Rc::new(ImageResource::new(data, name));
                        sky_textures.insert(resource.id, resource.clone());
                        selected_texture = Some(resource.id);
                        scene.env.sky.variant = SkyVariant::Textured(resource, 0.0);
                        sel_sky = 2;

                        messages.insert(UiMessage::CompleteRenderReset);
                        messages.insert(UiMessage::ConvReset);

                        renderer.passes.conv.state = ConvolutionPassState::Init;
                    }
                    ThreadReply::OpenScene { data: is, name } => {
                        vulkan_context.device.wait_idle()?;

                        messages.insert(UiMessage::ReferenceRenderReset);

                        scene.meshes.extend(is.instances);
                        textures.extend(is.textures);

                        if !camera_set {
                            camera_set = true;
                            if let Some(camera) = is.camera {
                                scene.camera.fov = camera.fov;
                                scene.camera.position = camera.position;
                                scene.camera.rotation = camera.rotation;
                            }
                        }

                        window.set_title(&format!("Toyrender - [{}]", name)).unwrap();
                    }
                }
            }

            input_mapper.update(event_pump.keyboard_state());
            if input_mapper.inner_state.values().any(|a| *a != 0.0) {
                messages.insert(UiMessage::ReferenceRenderReset);
            }

            if resized {
                renderer.resize(window.drawable_size())?;
                reference_renderer.resize(window.drawable_size())?;
                messages.insert(UiMessage::CompleteRenderReset);
            }

            frame_end = Instant::now();

            platform.prepare_frame(&mut imgui, &window, &event_pump);
            let ui = imgui.new_frame();
            ui.dockspace_over_main_viewport();

            let width = window.drawable_size().0 as f32;
            if LOADING.load(Ordering::SeqCst)
                && let Some(loading_popup) = ui
                    .window("LOADING")
                    .size([100.0, 30.0], imgui::Condition::Always)
                    .position([width - 120.0, 20.0], imgui::Condition::Always)
                    .flags(imgui::WindowFlags::NO_DECORATION | imgui::WindowFlags::NO_INPUTS)
                    .begin()
            {
                ui.text("Loading...");

                loading_popup.end();
            }

            let window_builder = ui
                .window("toyrender controls")
                .size([300.0, 100.0], imgui::Condition::FirstUseEver);

            if let Some(iw) = window_builder.begin() {
                ui_focused = ui.is_window_focused() && state.ui_visible;

                if let Some(cb) = ui.begin_combo("Renderer", format!("{:?}", state.selected_renderer)) {
                    for cur in SelectedRenderer::iterator() {
                        if state.selected_renderer == *cur {
                            ui.set_item_default_focus();
                        }

                        let clicked = ui
                            .selectable_config(format!("{:?}", cur))
                            .selected(state.selected_renderer == *cur)
                            .build();

                        if clicked {
                            state.selected_renderer = *cur;
                            messages.insert(UiMessage::CompleteRenderReset);
                        }
                    }
                    cb.end();
                }

                if ui.collapsing_header("RT Settings", imgui::TreeNodeFlags::DEFAULT_OPEN) {
                    if ui.slider(
                        "Direct trace distance",
                        0.0,
                        500.0,
                        &mut renderer.quality.rt_direct_trace_distance,
                    ) {
                        reference_renderer.quality.rt_direct_trace_distance = renderer.quality.rt_direct_trace_distance;
                        messages.insert(UiMessage::ReferenceRenderReset);
                    }
                    if ui.slider(
                        "Indirect trace distance",
                        0.0,
                        500.0,
                        &mut renderer.quality.rt_indirect_trace_distance,
                    ) {
                        reference_renderer.quality.rt_indirect_trace_distance =
                            renderer.quality.rt_indirect_trace_distance;
                        messages.insert(UiMessage::ReferenceRenderReset);
                    }
                    if ui.slider("Bounce count", 0, 10, &mut renderer.quality.pt_bounces) {
                        reference_renderer.quality.pt_bounces = renderer.quality.pt_bounces;
                        messages.insert(UiMessage::ReferenceRenderReset);
                    }

                    if ui.slider(
                        "Indirect intensity clamp",
                        0.0,
                        100.0,
                        &mut renderer.quality.indirect_light_clamp,
                    ) {
                        reference_renderer.quality.indirect_light_clamp = renderer.quality.indirect_light_clamp;
                        messages.insert(UiMessage::ReferenceRenderReset);
                    }

                    if ui.checkbox("Temporal accumulation", &mut taa_enable) {
                        frame = 0;
                    }

                    ui.checkbox("Fixed sample", &mut state.fixed_sample);

                    match state.selected_renderer {
                        SelectedRenderer::Realtime => {
                            ui.checkbox("Spatial denoise", &mut renderer.quality.use_spatial_denoise);
                            ui.checkbox("Culling", &mut renderer.quality.culling);

                            if ui.button("Debug Probes") {
                                renderer.passes.conv.state = ConvolutionPassState::Debug;
                            }

                            if ui.button("Reset probes") {
                                messages.insert(UiMessage::ConvReset);
                            }
                        }
                        SelectedRenderer::Reference => {
                            if ui.checkbox(
                                "Importance Sampling",
                                &mut reference_renderer.quality.importance_sampling,
                            ) {
                                messages.insert(UiMessage::ReferenceRenderReset);
                            }
                            if ui.checkbox("Russian roulette", &mut reference_renderer.quality.russian_roulette) {
                                messages.insert(UiMessage::ReferenceRenderReset);
                            }
                            if ui.checkbox("Disable materials", &mut reference_renderer.quality.disable_materials) {
                                messages.insert(UiMessage::ReferenceRenderReset);
                            }
                        }
                    }

                    if ui
                        .input_float3("Camera position", scene.camera.position.as_mut())
                        .build()
                    {
                        messages.insert(UiMessage::ReferenceRenderReset);
                    }
                    if ui
                        .input_float3("Camera rotation", scene.camera.rotation.as_mut())
                        .build()
                    {
                        messages.insert(UiMessage::ReferenceRenderReset);
                    }
                    if ui.slider("Camera FoV", 1.0, 174.0, &mut scene.camera.fov) {
                        messages.insert(UiMessage::ReferenceRenderReset);
                    }
                    if ui.slider("Render scale", 0.01, 1.0, &mut renderer.render_scale) {
                        clear_taa = true;
                        reference_renderer.render_scale = renderer.render_scale;
                        messages.insert(UiMessage::ReferenceRenderReset);
                    }
                }
                if ui.collapsing_header("Environment", imgui::TreeNodeFlags::DEFAULT_OPEN) {
                    let variants = if sky_textures.is_empty() {
                        vec!["Shader", "SingleColor"]
                    } else {
                        vec!["Shader", "SingleColor", "Textured"]
                    };

                    if ui.combo("Sky", &mut sel_sky, &variants, |a| std::borrow::Cow::Borrowed(a)) {
                        scene.env.sky.variant = match sel_sky {
                            0 => SkyVariant::Shader,
                            1 => SkyVariant::SingleColor(Vec3::from_element(1.0)),
                            2 => {
                                let curr_rot = match &scene.env.sky.variant {
                                    SkyVariant::Textured(_, r) => *r,
                                    _ => 0.0,
                                };

                                SkyVariant::Textured(sky_textures[&selected_texture.unwrap()].clone(), curr_rot)
                            }
                            _ => unreachable!(),
                        };
                        messages.insert(UiMessage::ReferenceRenderReset);
                        messages.insert(UiMessage::ConvReset);
                    }

                    match &mut scene.env.sky.variant {
                        SkyVariant::Textured(ir, r) => {
                            if let Some(combo) = ui.begin_combo("Texture", &ir.name) {
                                for (k, v) in &sky_textures {
                                    if ui.selectable_config(v.name.clone()).build() {
                                        messages.insert(UiMessage::ConvReset);
                                        selected_texture = Some(*k);
                                        *ir = v.clone();
                                    }
                                }
                                combo.end();
                            }

                            if ui.slider("Sky rotation", 0.0, 1.0, r) {
                                messages.insert(UiMessage::ReferenceRenderReset);
                            }
                        }
                        SkyVariant::SingleColor(color) => {
                            if ui.color_edit3("Sky color", color.as_mut()) {
                                messages.insert(UiMessage::ReferenceRenderReset);
                                messages.insert(UiMessage::ConvReset);
                            }
                        }
                        _ => {}
                    }

                    if ui.slider("Sky intensity", 0.0, 10.0, &mut scene.env.sky.intensity) {
                        messages.insert(UiMessage::ReferenceRenderReset);
                    }

                    ui.separator();

                    ui.slider("Exposure", -10.0, 10.0, &mut scene.env.exposure);

                    ui.separator();

                    if ui.slider("Sun intensity", 0.0, 10.0, &mut scene.env.sun_intensity) {
                        messages.insert(UiMessage::ReferenceRenderReset);
                    }

                    if ui
                        .input_float3("Sun direction", scene.env.sun_direction.as_mut())
                        .build()
                    {
                        messages.insert(UiMessage::ReferenceRenderReset);
                        if let SkyVariant::Shader = scene.env.sky.variant {
                            messages.insert(UiMessage::ConvReset);
                        }
                    }

                    if ui.slider("Sun angle", 0.0, std::f32::consts::FRAC_PI_2, &mut scene.env.sun_angle) {
                        messages.insert(UiMessage::ReferenceRenderReset);
                    }

                    if ui.color_edit3("Sun color", scene.env.sun_color.as_mut()) {
                        messages.insert(UiMessage::ReferenceRenderReset);
                    }
                }
                if ui.collapsing_header("Stats", imgui::TreeNodeFlags::DEFAULT_OPEN) {
                    gui::stats_tab(ui, &frame_stats, delta);
                }
                if ui.collapsing_header("Lights", imgui::TreeNodeFlags::DEFAULT_OPEN) {
                    gui::lights_tab(ui, &mut scene);
                }
                if ui.collapsing_header("Scene", imgui::TreeNodeFlags::empty()) {
                    gui::scene_tab(ui, scene.meshes.iter_mut(), &mut messages);
                }
                if ui.collapsing_header("Textures", imgui::TreeNodeFlags::empty()) {
                    gui::textures_tab(ui, &textures);
                }

                iw.end();
            }

            let draw_data = imgui.render();

            if !ui_focused {
                let directions = scene.camera.directions();

                scene.camera.fov += mouse_scroll;
                scene.camera.position += (input_mapper.get_value(InputAxes::Up) * directions.up
                    + input_mapper.get_value(InputAxes::Forward) * directions.forward
                    + input_mapper.get_value(InputAxes::Right) * directions.right)
                    * delta
                    * movement_speed;

                scene.camera.rotation.z -= mouse.0 as f32 * mouse_sens;
                scene.camera.rotation.x -= mouse.1 as f32 * mouse_sens;

                if debug_mode_flip {
                    renderer.debug_mode = renderer.debug_mode.next();
                    reference_renderer.debug_mode = renderer.debug_mode;
                    eprintln!("debug mode: {:?}", renderer.debug_mode);
                }
            }

            if state.fixed_sample {
                frame = 0;
                messages.insert(UiMessage::ConvReset);
            }

            for msg in &messages {
                match msg {
                    UiMessage::CompleteRenderReset => {
                        clear_taa = true;
                        frame = 0;
                    }
                    UiMessage::ReferenceRenderReset if state.selected_renderer == SelectedRenderer::Reference => {
                        clear_taa = true;
                        frame = 0;
                    }
                    UiMessage::SaveScreen => Self::on_save_image(
                        state.selected_renderer,
                        &mut renderer,
                        &mut reference_renderer,
                        window.drawable_size(),
                    )?,
                    UiMessage::ConvReset => {
                        values.clear();
                        renderer.passes.conv.state = ConvolutionPassState::Init;
                    }
                    _ => {}
                }
            }

            let skip_primary_render = match state.selected_renderer {
                SelectedRenderer::Reference => false,
                SelectedRenderer::Realtime => {
                    (renderer.passes.conv.state != ConvolutionPassState::Finished
                        && renderer.passes.conv.state != ConvolutionPassState::Debug)
                        && !state.fixed_sample
                        && frame % 15 != 0
                }
            };

            let context = FrameContext {
                delta_time: delta,
                total_time: frame_end.duration_since(start).as_secs_f32(),
                clear_taa: resized || clear_taa || frame == 0 || !taa_enable,
                frame_index: frame as u32,
                skip_primary_render,
            };

            let draw_data = match state.ui_visible {
                true => Some(draw_data),
                false => None,
            };

            let report = match state.selected_renderer {
                SelectedRenderer::Realtime => renderer.render_frame(
                    &scene,
                    &mut resource_subsystem,
                    &textures,
                    window.drawable_size(),
                    &context,
                    draw_data,
                )?,
                SelectedRenderer::Reference => reference_renderer.render_frame(
                    &scene,
                    &mut resource_subsystem,
                    &textures,
                    window.drawable_size(),
                    &context,
                    draw_data,
                )?,
            };

            if state.selected_renderer == SelectedRenderer::Realtime {
                let state = &mut renderer.passes.conv.state;

                if *state != ConvolutionPassState::Finished {
                    vulkan_context.device.wait_idle()?;
                }

                let mem = renderer.output_buf_cpu.read_host();
                let s = unsafe { std::slice::from_raw_parts(mem.as_ptr() as *const f32, mem.len() / 4) };

                match *state {
                    ConvolutionPassState::SunProbe(_) | ConvolutionPassState::SunlessProbe(_) => {
                        if values.len() >= 100 {
                            values.pop_front();
                        }

                        let luma = s[0] * 0.2126 + s[1] * 0.7152 + s[2] * 0.0722;

                        values.push_back(luma);
                    }
                    _ => {}
                }

                let avg = if values.len() > 10 {
                    let mut diffs = VecDeque::with_capacity(100);

                    for x in values.iter().collect::<Vec<_>>().windows(2) {
                        diffs.push_back((x[0] - x[1]).abs());
                    }

                    // ignore last diff, it might have bad data from the last iteration
                    diffs.pop_front();

                    let avg = diffs.iter().fold(0.0, |acc, x| acc + x) / diffs.len() as f32;

                    Some(-avg.ln())
                } else {
                    None
                };

                *state = match *state {
                    ConvolutionPassState::Init => {
                        values.clear();
                        ConvolutionPassState::SunProbe(0)
                    }
                    ConvolutionPassState::SunProbe(i) => {
                        match avg {
                            Some(a) if a > 11.0 => {
                                // good enough quality for sun probe
                                info!("Sun conv finished in {} iterations", i + 1);
                                values.clear();
                                last_sun_sum = vec3(s[0], s[1], s[2]);
                                ConvolutionPassState::FindingMax
                            }
                            Some(a) if i % 500 == 0 => {
                                info!("Sun probe avg log step {}: {}/11", i, a);
                                ConvolutionPassState::SunProbe(i + 1)
                            }
                            _ => ConvolutionPassState::SunProbe(i + 1),
                        }
                    }
                    ConvolutionPassState::FindingMax => {
                        info!("Sun pos: {},{}", s[4], s[5]);

                        let z = 1.0 - s[5] * 2.0;
                        let amplitude = (z * std::f32::consts::PI * 0.5).cos();
                        if let SkyVariant::Textured(_, r) = scene.env.sky.variant {
                            let rot = s[4] + r;
                            scene.env.sun_direction.x = -(rot * std::f32::consts::PI * 2.0).cos() * amplitude;
                            scene.env.sun_direction.y = (rot * std::f32::consts::PI * 2.0).sin() * amplitude;
                            scene.env.sun_direction.z = (z * std::f32::consts::PI * 0.5).sin();
                        };

                        ConvolutionPassState::SunlessProbe(0)
                    }
                    ConvolutionPassState::SunlessProbe(i) => {
                        match avg {
                            Some(a) if a > 14.0 => {
                                // better quality for sunless probe
                                info!("Conv finished in {} iterations", i + 1);
                                values.clear();
                                last_sum = vec3(s[0], s[1], s[2]);

                                dbg!(last_sum, last_sun_sum);

                                if let SkyVariant::Textured(_, r) = scene.env.sky.variant {
                                    let sun_only = last_sun_sum - last_sum;
                                    let color_adjust = sun_only.max();

                                    let sun_color = sun_only / color_adjust;

                                    scene.env.sun_color = sun_color;
                                    scene.env.sun_intensity = color_adjust * std::f32::consts::PI * 2.0;
                                };

                                ConvolutionPassState::Finished
                            }
                            Some(a) if i % 500 == 0 => {
                                info!("Sunless probe avg log step {}: {}/14", i, a);
                                ConvolutionPassState::SunlessProbe(i + 1)
                            }
                            _ => ConvolutionPassState::SunlessProbe(i + 1),
                        }
                    }
                    ConvolutionPassState::Debug => {
                        let z = 1.0 - s[5] * 2.0;
                        if let SkyVariant::Textured(_, r) = scene.env.sky.variant {
                            let rot = s[4] + r;
                            scene.env.sun_direction.x = -(rot * std::f32::consts::PI * 2.0).cos() * (1.0 - z);
                            scene.env.sun_direction.y = (rot * std::f32::consts::PI * 2.0).sin() * (1.0 - z);
                            scene.env.sun_direction.z = z;
                        };

                        ConvolutionPassState::Debug
                    }
                    a => a,
                };
            }

            frame_stats.update(report);

            if !focused {
                let frametime_target = 1.0 / 30.0;

                if delta < frametime_target {
                    //std::thread::sleep(std::time::Duration::from_secs_f32(frametime_target - delta));
                }
            };
            frame += 1;
        }

        // drop sending handle right before waiting for worker thread to avoid deadlock
        drop(tx);
        handle.join().unwrap();

        Ok(())
    }

    fn on_file_drop(filename: String) -> Result<FileDroppedAction, AppError> {
        let path = std::path::PathBuf::from_str(&filename).unwrap();

        match path.extension().map(|ext| ext.to_str().unwrap()) {
            Some("glb") => Ok(FileDroppedAction::LoadScene { filename }),
            Some("exr") | Some("hdr") => Ok(FileDroppedAction::LoadImage { filename }),
            _ => Err(AppError::Import("Unknown file format".to_owned())),
        }
    }

    fn on_save_image(
        selected_renderer: SelectedRenderer,
        renderer: &mut VulkanRenderer,
        reference_renderer: &mut VulkanMcPathTracer,
        size: (u32, u32),
    ) -> Result<(), AppError> {
        if selected_renderer == SelectedRenderer::Reference {
            let data = reference_renderer.save_image(size)?;

            image::save_buffer(
                format! {"{}.exr", chrono::Utc::now().format("%Y_%m_%d_%H_%M_%S")},
                &data,
                size.0,
                size.1,
                image::ExtendedColorType::Rgba32F,
            )?;
        }

        Ok(())
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
                skip_primary_render: false,
            };

            self.renderer.render_frame(
                &self.scene,
                &mut self.resource_subsystem,
                &[],
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

fn loader_thread(rx: Receiver<ThreadAction>, tx_reply: Sender<ThreadReply>) {
    for task in rx.iter() {
        let start = Instant::now();
        LOADING.store(true, Ordering::SeqCst);
        match task {
            ThreadAction::DecodeImage { filename } => {
                info!("Decoding image {filename}");

                let buf = match std::fs::read(&filename) {
                    Ok(d) => d,
                    Err(e) => {
                        let msg = format!("file {} cannot be read: {e}", filename);

                        tx_reply.send(ThreadReply::Err(AppError::Import(msg))).unwrap();
                        LOADING.store(false, Ordering::SeqCst);
                        continue;
                    }
                };

                let data = match image::load_from_memory(&buf) {
                    Ok(d) => d,
                    Err(e) => {
                        tx_reply.send(ThreadReply::Err(e.into())).unwrap();
                        LOADING.store(false, Ordering::SeqCst);
                        continue;
                    }
                };

                let path = std::path::PathBuf::from_str(&filename).unwrap();
                let name = path.file_name().unwrap().to_string_lossy().into_owned();

                tx_reply.send(ThreadReply::DecodeImage { name, data }).unwrap();

                info!("Loaded (in thread) in {} s", start.elapsed().as_secs_f32());
            }
            ThreadAction::OpenScene { filename } => {
                info!("Opening scene {filename}");

                let buf = match std::fs::read(&filename) {
                    Ok(d) => d,
                    Err(e) => {
                        let msg = format!("file {} cannot be read: {e}", filename);

                        tx_reply.send(ThreadReply::Err(AppError::Import(msg))).unwrap();
                        LOADING.store(false, Ordering::SeqCst);
                        continue;
                    }
                };

                let path = std::path::PathBuf::from_str(&filename).unwrap();
                let name = path.file_name().unwrap().to_string_lossy().into_owned();

                match import::extract_scene(&buf) {
                    Ok(data) => tx_reply.send(ThreadReply::OpenScene { data, name }).unwrap(),
                    Err(e) => tx_reply.send(ThreadReply::Err(e)).unwrap(),
                }

                info!("Loaded (in thread) in {} s", start.elapsed().as_secs_f32());
            }
        }
        LOADING.store(false, Ordering::SeqCst);
    }
}

pub struct AppState {
    selected_renderer: SelectedRenderer,
    ui_visible: bool,
    fixed_sample: bool,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            selected_renderer: SelectedRenderer::Reference,
            ui_visible: true,
            fixed_sample: false,
        }
    }
}

#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub enum InputAxes {
    Forward,
    Right,
    Up,
}

#[derive(PartialEq, Debug, Copy, Clone)]
enum SelectedRenderer {
    Realtime,
    Reference,
}

impl SelectedRenderer {
    pub fn iterator() -> Iter<'static, Self> {
        static CASES: [SelectedRenderer; 2] = [SelectedRenderer::Realtime, SelectedRenderer::Reference];
        CASES.iter()
    }
}

enum FileDroppedAction {
    LoadScene { filename: String },
    LoadImage { filename: String },
}

enum ThreadAction {
    DecodeImage { filename: String },
    OpenScene { filename: String },
}

enum ThreadReply {
    DecodeImage { data: DynamicImage, name: String },
    OpenScene { data: ImportedScene, name: String },
    Err(AppError),
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum UiMessage {
    CompleteRenderReset,
    ReferenceRenderReset,
    SaveScreen,
    ConvReset,
}
