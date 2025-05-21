use crate::args::Args;
use crate::err::AppError;
use crate::import;
use crate::import::ImportedScene;
use crate::input::InputMapper;
use crate::renderer::{FrameContext, VulkanRenderer};
use crate::scene::Scene;
use log::{error, info};
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use sdl2::mouse::MouseButton;
use sdl2::video::Window;
use sdl2::{EventPump, Sdl};
use std::time::Instant;

pub struct App {
    pub renderer: VulkanRenderer,
    pub sdl_context: Sdl,
    pub window: Window,
    pub event_pump: EventPump,
    pub input_mapper: InputMapper<InputAxes>,
    pub scene: Scene,
}

impl App {
    pub fn create() -> Result<Self, AppError> {
        let sdl_context = sdl2::init().expect("cannot init sdl2");

        let video_subsystem = sdl_context.video().expect("cannot init video");

        let mut window = video_subsystem
            .window("Vulkan Demo", 1600, 900)
            .allow_highdpi()
            .resizable()
            .position_centered()
            .vulkan()
            .build()
            .expect("cannot build window");

        window.set_minimum_size(120, 40).expect("cannot set min size");

        let event_pump = sdl_context.event_pump().expect("cannot get event pump");

        let renderer = VulkanRenderer::init(&window)?;

        let input_mapper = Self::setup_input_mapper();

        let scene = Scene::new();

        Ok(Self {
            sdl_context,
            window,
            event_pump,
            renderer,
            input_mapper,
            scene,
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

        let start = Instant::now();
        let mut frame_end = Instant::now();

        let mouse_sens = 0.002;
        let scroll_sens = 0.5;
        let movement_speed = 4.0;
        let mut focused = true;
        let mut taa_enable = true;

        if args.benchmark {
            self.benchmark(300)?;
            return Ok(());
        }

        let mut frame = 0;

        'running: loop {
            let mut resized = false;

            let frame_start = Instant::now();
            let delta = frame_start.duration_since(frame_end).as_secs_f32();

            let mut mouse = (0, 0);
            let mut mouse_scroll = 0.0;
            let mut dragging;
            let mut sample_adjust = 0;
            let mut dist_adjust = 0.0;
            let mut exposure_adjust = 0.0;
            let mut flip_half_res = false;
            let mut clear_taa = false;
            let mut debug_mode_flip = false;

            for event in self.event_pump.poll_iter() {
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
                        Self::on_file_drop(filename, &mut self.scene)?;
                    }
                    Event::MouseWheel { y, .. } => {
                        mouse_scroll = y as f32 * scroll_sens;
                    }
                    Event::MouseMotion {
                        xrel, yrel, mousestate, ..
                    } => {
                        dragging = mousestate.is_mouse_button_pressed(MouseButton::Right);
                        self.sdl_context.mouse().set_relative_mouse_mode(dragging);

                        if dragging {
                            mouse = (xrel, yrel);
                        } else {
                            self.sdl_context.mouse().show_cursor(true);
                        }
                    }
                    Event::KeyDown { keycode, .. } => match keycode {
                        Some(Keycode::LeftBracket) => sample_adjust = -1,
                        Some(Keycode::RightBracket) => sample_adjust = 1,
                        Some(Keycode::O) => dist_adjust = -5.0,
                        Some(Keycode::P) => dist_adjust = 5.0,
                        Some(Keycode::R) => {
                            debug_mode_flip = true;
                            clear_taa = true;
                        }
                        Some(Keycode::H) => {
                            flip_half_res = true;
                            clear_taa = true;
                        }
                        Some(Keycode::T) => {
                            taa_enable = !taa_enable;
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

            frame_end = Instant::now();

            let context = FrameContext {
                delta_time: delta,
                total_time: frame_end.duration_since(start).as_secs_f32(),
                clear_taa: resized || clear_taa || frame == 0 || !taa_enable,
                frame_index: frame,
            };

            if sample_adjust != 0 {
                let new_samples = (self.renderer.quality.rtao_samples + sample_adjust).max(0);
                self.renderer.quality.rtao_samples = new_samples;
                info!("new sample count: {new_samples}");
            }

            self.scene.env.exposure = (self.scene.env.exposure + exposure_adjust).clamp(-32.0, 32.0);
            self.renderer.quality.rt_trace_disance += dist_adjust;

            if flip_half_res {
                self.renderer.quality.half_res = !self.renderer.quality.half_res;
            }

            if debug_mode_flip {
                self.renderer.debug_mode = self.renderer.debug_mode.next();
                eprintln!("debug mode: {:?}", self.renderer.debug_mode);
            }

            if resized {
                self.renderer.resize(self.window.drawable_size())?;
            }

            let cpu_time = self
                .renderer
                .render_frame(&self.scene, self.window.drawable_size(), &context)?;

            if !focused {
                let frametime_target = 1.0 / 30.0;

                if delta < frametime_target {
                    //std::thread::sleep(Duration::from_secs_f32(frametime_target - delta));
                }
            };

            eprint!("{} FPS, CPU: {} ms\r", 1.0 / delta, cpu_time * 1000.0);
            frame += 1;
        }
        eprintln!();

        Ok(())
    }

    fn on_file_drop(filename: String, scene: &mut Scene) -> Result<(), AppError> {
        info!("loading file `{filename}`");
        let start = Instant::now();

        let file = std::fs::read(&filename).map_err(|e| {
            let msg = format!("file {} cannot be read: {e}", filename);

            AppError::Import(msg)
        });

        let file = match file {
            Ok(f) => f,
            Err(e) => {
                error!("{e}");
                return Ok(());
            }
        };

        let ImportedScene { instances, .. } = import::extract_scene(&file)?;
        scene.meshes.extend(instances);

        let end = Instant::now();

        info!("Loaded in {} s", (end - start).as_secs_f32());

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
                frame_index: 0,
            };

            self.renderer
                .render_frame(&self.scene, self.window.drawable_size(), &context)?;
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
}

#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub enum InputAxes {
    Forward,
    Right,
    Up,
}
