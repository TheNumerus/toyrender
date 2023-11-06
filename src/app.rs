use crate::args::Args;
use crate::err::AppError;
use crate::import;
use crate::input::InputMapper;
use crate::renderer::{FrameContext, VulkanRenderer};
use crate::scene::Scene;
use log::{error, info};
use sdl2::event::{Event, WindowEvent};
use sdl2::mouse::MouseButton;
use sdl2::video::Window;
use sdl2::{EventPump, Sdl};
use std::time::Instant;

pub struct App {
    pub sdl_context: Sdl,
    pub window: Window,
    pub event_pump: EventPump,
    pub renderer: VulkanRenderer,
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

    pub fn run(self, args: Args) -> Result<(), AppError> {
        let App {
            sdl_context,
            window,
            mut event_pump,
            mut renderer,
            mut input_mapper,
            mut scene,
        } = self;

        if let Some(path) = args.file_to_open {
            let file = std::fs::read(&path).map_err(|e| {
                let msg = format!("file {} cannot be read: {e}", path.to_string_lossy());

                AppError::Import(msg)
            })?;

            let (_meshes, instances) = import::extract_scene(renderer.device.clone(), &renderer.command_pool, &file)?;
            scene.meshes.extend(instances);
        }

        let start = Instant::now();
        let mut frame_end = Instant::now();

        let mouse_sens = 0.002;
        let movement_speed = 4.0;

        'running: loop {
            let mut resized = false;

            let frame_start = Instant::now();
            let delta = frame_start.duration_since(frame_end).as_secs_f32();

            let mut mouse = (0, 0);
            let mut dragging;

            for event in event_pump.poll_iter() {
                match event {
                    Event::Quit { .. } => {
                        break 'running;
                    }
                    Event::Window {
                        win_event: WindowEvent::Resized(_, _),
                        ..
                    } => {
                        resized = true;
                    }
                    Event::DropFile { filename, .. } => {
                        Self::on_file_drop(filename, &renderer, &mut scene)?;
                    }
                    Event::MouseMotion {
                        xrel, yrel, mousestate, ..
                    } => {
                        dragging = mousestate.is_mouse_button_pressed(MouseButton::Right);
                        sdl_context.mouse().set_relative_mouse_mode(dragging);

                        if dragging {
                            mouse = (xrel, yrel);
                        } else {
                            sdl_context.mouse().show_cursor(true);
                        }
                    }
                    _ => {}
                }
            }

            input_mapper.update(event_pump.keyboard_state());

            let directions = scene.camera.directions();

            scene.camera.position += (input_mapper.get_value(InputAxes::Up) * directions.up
                + input_mapper.get_value(InputAxes::Forward) * directions.forward
                + input_mapper.get_value(InputAxes::Right) * directions.right)
                * delta
                * movement_speed;

            scene.camera.rotation.z -= mouse.0 as f32 * mouse_sens;
            scene.camera.rotation.x -= mouse.1 as f32 * mouse_sens;

            frame_end = Instant::now();

            let context = FrameContext {
                delta_time: delta,
                total_time: frame_end.duration_since(start).as_secs_f32(),
            };

            renderer.render_frame(&scene, window.drawable_size(), &context)?;
            eprint!("{} FPS\r", 1.0 / delta);

            if resized {
                renderer.resize(window.drawable_size())?;
            }
        }
        eprintln!();

        Ok(())
    }

    fn on_file_drop(filename: String, renderer: &VulkanRenderer, scene: &mut Scene) -> Result<(), AppError> {
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

        let (_meshes, instances) = import::extract_scene(renderer.device.clone(), &renderer.command_pool, &file)?;
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
}

#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub enum InputAxes {
    Forward,
    Right,
    Up,
}
