use log::info;
use sdl2::event::{Event, WindowEvent};
use sdl2::mouse::MouseButton;
use std::time::Instant;

mod app;
mod camera;
mod err;
mod import;
mod input;
mod mesh;
mod renderer;
mod scene;
mod vulkan;

use crate::input::InputMapper;
use crate::mesh::MeshInstance;
use err::AppError;
use renderer::FrameContext;
use scene::Scene;

fn main() -> Result<(), AppError> {
    env_logger::init();

    let mut app = app::App::create();
    let mut renderer = renderer::VulkanRenderer::init(&app)?;

    let sphere = include_bytes!("../meshes/sphere.glb");
    let axes = include_bytes!("../meshes/axes.glb");

    let (shape, indices) = import::extract_mesh(axes)?;
    let mesh = mesh::Mesh::new(renderer.device.clone(), &renderer.command_pool, &shape, &indices)?;
    let mesh = std::rc::Rc::new(mesh);

    let (shape, indices) = import::extract_mesh(sphere)?;
    let mesh_2 = mesh::Mesh::new(renderer.device.clone(), &renderer.command_pool, &shape, &indices)?;
    let mesh_2 = std::rc::Rc::new(mesh_2);

    let instance = MeshInstance::new(mesh);
    let mut instance_2 = MeshInstance::new(mesh_2);

    let mut input_mapper = setup_input_mapper();

    let mut scene = Scene::new();
    scene.camera.position = nalgebra_glm::Vec3::new(2.2, -5.0, 3.1);

    scene.camera.rotation.x = 1.15;
    scene.camera.rotation.z = 0.3;
    scene.meshes.push(instance);

    instance_2
        .transform
        .append_translation_mut(&nalgebra_glm::Vec3::new(2.0, 2.0, 0.0));
    scene.meshes.push(instance_2);

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
                Event::MouseMotion {
                    xrel, yrel, mousestate, ..
                } => {
                    dragging = mousestate.is_mouse_button_pressed(MouseButton::Right);
                    app.sdl_context.mouse().set_relative_mouse_mode(dragging);

                    if dragging {
                        mouse = (xrel, yrel);
                    } else {
                        app.sdl_context.mouse().show_cursor(true);
                    }
                }
                _ => {}
            }
        }

        input_mapper.update(app.event_pump.keyboard_state());

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
        (Scancode::W, vec![(InputAxes::Forward, -1.0)]),
        (Scancode::S, vec![(InputAxes::Forward, 1.0)]),
        (Scancode::A, vec![(InputAxes::Right, -1.0)]),
        (Scancode::D, vec![(InputAxes::Right, 1.0)]),
        (Scancode::Q, vec![(InputAxes::Up, -1.0)]),
        (Scancode::E, vec![(InputAxes::Up, 1.0)]),
    ])
}

#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub enum InputAxes {
    Forward,
    Right,
    Up,
}
