use log::info;
use sdl2::event::{Event, WindowEvent};
use std::time::Instant;

mod app;
mod camera;
mod err;
mod input;
mod mesh;
mod renderer;
mod scene;
mod vulkan;

use crate::input::InputMapper;
use err::AppError;
use renderer::FrameContext;
use scene::Scene;

fn main() -> Result<(), AppError> {
    env_logger::init();

    let mut app = app::App::create();
    let mut renderer = renderer::VulkanRenderer::init(&app)?;

    let (shape, indices) = mesh::square();
    let mesh = mesh::Mesh::new(renderer.device.clone(), &renderer.command_pool, &shape, &indices)?;

    let mut input_mapper = setup_input_mapper();

    let mut scene = Scene::new();
    scene
        .camera
        .transform
        .append_translation_mut(&nalgebra_glm::Vec3::new(0.0, 0.0, -1.0));
    scene.meshes.push(mesh);

    let start = Instant::now();
    let mut frame_end = Instant::now();

    'running: loop {
        let mut resized = false;

        let frame_start = Instant::now();
        let delta = frame_start.duration_since(frame_end).as_secs_f32();

        for event in app.event_pump.poll_iter() {
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
                _ => {}
            }
        }

        input_mapper.update(app.event_pump.keyboard_state());

        scene.camera.transform.append_translation_mut(
            &(nalgebra_glm::Vec3::new(
                input_mapper.get_value(InputAxes::Right),
                input_mapper.get_value(InputAxes::Up),
                input_mapper.get_value(InputAxes::Forward),
            ) * delta),
        );

        frame_end = Instant::now();

        let context = FrameContext {
            delta_time: delta,
            total_time: frame_end.duration_since(start).as_secs_f32(),
        };

        renderer.render_frame(&app, &scene, &context)?;
        eprint!("{} FPS\r", 1.0 / delta);

        if resized {
            renderer.resize(&app)?;
        }
    }
    eprintln!();

    info!("Quitting app...");

    Ok(())
}

pub fn setup_input_mapper() -> InputMapper<InputAxes> {
    use sdl2::keyboard::Scancode;

    InputMapper::with_configuration([
        (Scancode::W, vec![(InputAxes::Forward, 1.0)]),
        (Scancode::S, vec![(InputAxes::Forward, -1.0)]),
        (Scancode::A, vec![(InputAxes::Right, 1.0)]),
        (Scancode::D, vec![(InputAxes::Right, -1.0)]),
        (Scancode::Q, vec![(InputAxes::Up, 1.0)]),
        (Scancode::E, vec![(InputAxes::Up, -1.0)]),
    ])
}

#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub enum InputAxes {
    Forward,
    Right,
    Up,
}
