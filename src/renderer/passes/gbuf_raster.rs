use crate::err::AppError;
use crate::renderer::debug::DebugMode;
use crate::renderer::render_target::{MultipleRenderTarget, MultipleRenderTargetBuilder, RenderTargets};
use crate::renderer::{RenderPasses, VulkanRenderer};
use crate::scene::Scene;
use crate::vulkan::{CommandBuffer, RenderPass};
use ash::vk;
use std::cell::RefCell;
use std::rc::Rc;

pub struct GbufRaster {
    render_pass: Rc<RenderPass>,
    gbuffer: Rc<RefCell<MultipleRenderTarget>>,
}

impl GbufRaster {
    pub fn init(render_targets: &mut RenderTargets, render_passes: &RenderPasses) -> Result<Self, AppError> {
        let render_pass = render_passes.passes.get("gb").unwrap().clone();

        let gbuffer = render_targets.add_mrt(
            MultipleRenderTargetBuilder::new("gbuffer", render_pass.clone())
                .add_target(
                    vk::Format::A2B10G10R10_UNORM_PACK32,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                    vk::ImageAspectFlags::COLOR,
                )
                .add_target(
                    vk::Format::A2B10G10R10_UNORM_PACK32,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                    vk::ImageAspectFlags::COLOR,
                )
                .add_target(
                    vk::Format::D32_SFLOAT,
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                        | vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::TRANSFER_SRC,
                    vk::ImageAspectFlags::DEPTH,
                ),
        )?;

        Ok(Self { gbuffer, render_pass })
    }

    pub fn draw(&self, renderer: &VulkanRenderer, command_buffer: &CommandBuffer, scene: &Scene) {
        renderer.device.begin_label("GBuffer", command_buffer);

        let clears = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                },
            },
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.5, 0.5, 1.0, 1.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 },
            },
        ];

        let render_pass_info = vk::RenderPassBeginInfo {
            render_pass: self.render_pass.inner,
            framebuffer: self.gbuffer.borrow().framebuffer.inner,
            render_area: vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: renderer.swap_chain.extent,
            },
            clear_value_count: clears.len() as u32,
            p_clear_values: clears.as_ptr(),
            ..Default::default()
        };

        let viewport = vk::Viewport {
            width: renderer.swap_chain.extent.width as f32,
            height: renderer.swap_chain.extent.height as f32,
            max_depth: 1.0,
            ..Default::default()
        };

        command_buffer.begin_render_pass(&render_pass_info, vk::SubpassContents::INLINE);

        let pipeline = renderer.pipeline_builder.get_graphics("deferred").unwrap();

        command_buffer.bind_graphics_pipeline(pipeline);
        command_buffer.set_viewport(viewport);
        command_buffer.set_scissor(render_pass_info.render_area);

        unsafe {
            renderer.device.inner.cmd_bind_descriptor_sets(
                command_buffer.inner,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.layout,
                0,
                &[
                    renderer.descriptors.global_sets[renderer.current_frame].inner.inner,
                    renderer.descriptors.image_sets[renderer.current_frame].inner.inner,
                ],
                &[],
            );
        }

        for instance in &scene.meshes {
            let mesh = &instance.resource;
            let mesh_data = &renderer.meshes[&mesh.id];

            command_buffer.bind_vertex_buffers(&[&mesh_data.buf], &[0]);

            unsafe {
                let constants = std::slice::from_raw_parts(instance.transform.as_ptr() as *const u8, 64);

                renderer.device.inner.cmd_push_constants(
                    command_buffer.inner,
                    pipeline.layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    constants,
                );

                renderer.device.inner.cmd_bind_index_buffer(
                    command_buffer.inner,
                    mesh_data.buf.inner.inner,
                    mesh_data.indices_offset,
                    vk::IndexType::UINT32,
                );

                renderer
                    .device
                    .inner
                    .cmd_draw_indexed(command_buffer.inner, mesh_data.index_count as u32, 1, 0, 0, 0);
            }
        }

        command_buffer.end_render_pass();

        renderer.device.end_label(command_buffer);
    }

    pub fn compatible_debug_modes() -> Vec<DebugMode> {
        vec![DebugMode::BaseColor, DebugMode::Depth, DebugMode::Normal]
    }
}
