use ash::vk;
use log::info;
use sdl2::event::{Event, WindowEvent};

mod app;
mod err;
mod mesh;
mod renderer;
mod vulkan;

use err::AppError;
use vulkan::VulkanError;

fn main() -> Result<(), AppError> {
    env_logger::init();

    let mut app = app::App::create();
    let mut renderer = renderer::VulkanRenderer::init(&app)?;

    let (shape, indices) = mesh::square();
    let mesh = mesh::Mesh::new(renderer.device.clone(), &renderer.command_pool, &shape, &indices)?;

    let start = std::time::Instant::now();

    'running: loop {
        let mut resized = false;

        for event in app.event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => {
                    break 'running;
                }
                Event::Window { win_event, .. } => match win_event {
                    WindowEvent::Resized(_, _) => {
                        resized = true;
                    }
                    _ => {}
                },
                Event::KeyDown { keycode, .. } => match keycode {
                    Some(sdl2::keyboard::Keycode::W) => {
                        dbg!("forward");
                    }
                    _ => {}
                },
                _ => {}
            }
        }

        let end = std::time::Instant::now();

        let push_constants = end.duration_since(start).as_secs_f32();

        renderer.render_frame(&app, |renderer, cb, fb| {
            record_command_buffer(renderer, cb, fb, &mesh, &push_constants)
        })?;

        if resized {
            renderer.resize(&app)?;
        }
    }

    info!("Quitting app...");

    Ok(())
}

fn record_command_buffer(
    renderer: &renderer::VulkanRenderer,
    command_buffer: &vulkan::CommandBuffer,
    framebuffer: &vulkan::SwapChainFramebuffer,
    mesh: &mesh::Mesh,
    push_constants: &f32,
) -> Result<(), VulkanError> {
    command_buffer.begin()?;

    let clear_color = vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 1.0],
        },
    };

    let render_pass_info = vk::RenderPassBeginInfo {
        render_pass: renderer.render_pass.inner,
        framebuffer: framebuffer.inner,
        render_area: vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: renderer.swap_chain.extent,
        },
        clear_value_count: 1,
        p_clear_values: &clear_color,
        ..Default::default()
    };

    command_buffer.begin_render_pass(&render_pass_info, vk::SubpassContents::INLINE);

    command_buffer.bind_pipeline(&renderer.pipeline, vk::PipelineBindPoint::GRAPHICS);
    command_buffer.set_viewport(renderer.pipeline.viewport);
    command_buffer.set_scissor(renderer.pipeline.scissor);
    command_buffer.bind_vertex_buffers(&[&mesh.buf], &[0]);

    unsafe {
        renderer.device.inner.cmd_push_constants(
            command_buffer.inner,
            renderer.pipeline.layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            &(push_constants).to_le_bytes(),
        );

        renderer.device.inner.cmd_bind_index_buffer(
            command_buffer.inner,
            mesh.buf.inner.inner,
            mesh.indices_offset,
            vk::IndexType::UINT32,
        );

        renderer
            .device
            .inner
            .cmd_draw_indexed(command_buffer.inner, mesh.index_count as u32, 1, 0, 0, 0);
    }

    command_buffer.end_render_pass();
    command_buffer.end()
}
