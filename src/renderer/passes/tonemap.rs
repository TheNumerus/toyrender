use crate::renderer::descriptors::DescLayout;
use crate::renderer::render_target::{RenderTarget, RenderTargetBuilder};
use crate::renderer::VulkanRenderer;
use crate::vulkan::{CommandBuffer, Device, VulkanError};
use ash::vk;
use std::cell::RefCell;
use std::rc::Rc;

pub(crate) struct TonemapPass {
    pub device: Rc<Device>,
    pub render_target: Rc<RefCell<RenderTarget>>,
}

impl TonemapPass {
    pub const TARGET_FORMATS: [vk::Format; 1] = [vk::Format::A2B10G10R10_UNORM_PACK32];
    pub const DESC_LAYOUTS: [DescLayout; 2] = [DescLayout::Global, DescLayout::Image];

    pub fn render_target_def() -> RenderTargetBuilder {
        RenderTargetBuilder::new("tonemap_test")
            .with_color_attachment()
            .with_storage()
            .with_transfer()
            .with_format(Self::TARGET_FORMATS[0])
            .with_sampled()
    }

    pub fn record(&self, command_buffer: &CommandBuffer, renderer: &VulkanRenderer) -> Result<(), VulkanError> {
        self.device.begin_label("PostProcessing", command_buffer);

        let attachments = [vk::RenderingAttachmentInfo {
            image_view: self.render_target.borrow().view.inner,
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            load_op: vk::AttachmentLoadOp::DONT_CARE,
            store_op: vk::AttachmentStoreOp::STORE,
            ..Default::default()
        }];

        let rendering_info = vk::RenderingInfo {
            render_area: vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: renderer.swap_chain.extent,
            },
            layer_count: 1,
            color_attachment_count: attachments.len() as u32,
            p_color_attachments: attachments.as_ptr(),
            ..Default::default()
        };

        unsafe {
            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier {
                    src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ,
                    dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    image: self.render_target.borrow().image.inner,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                }],
            );
        }

        let pipeline = renderer.pipeline_builder.get_graphics("tonemap").unwrap();

        command_buffer.begin_rendering(&rendering_info);

        command_buffer.bind_graphics_pipeline(pipeline);

        command_buffer.bind_vertex_buffers(&[&renderer.fs_quad.buf], &[0]);

        unsafe {
            self.device.inner.cmd_bind_index_buffer(
                command_buffer.inner,
                renderer.fs_quad.buf.inner.inner,
                renderer.fs_quad.indices_offset,
                vk::IndexType::UINT32,
            );

            self.device.inner.cmd_bind_descriptor_sets(
                command_buffer.inner,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.layout,
                0,
                &[
                    renderer.descriptors.borrow().global_sets[renderer.current_frame].inner,
                    renderer.descriptors.borrow().image_sets[renderer.current_frame].inner,
                ],
                &[],
            );

            let constants = (*renderer.descriptors.borrow().samplers.get("taa_target").unwrap() as u32).to_le_bytes();

            self.device.inner.cmd_push_constants(
                command_buffer.inner,
                pipeline.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                &constants,
            );

            self.device
                .inner
                .cmd_draw_indexed(command_buffer.inner, renderer.fs_quad.index_count as u32, 1, 0, 0, 0);
        }

        command_buffer.end_rendering();

        unsafe {
            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier {
                    src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    dst_access_mask: vk::AccessFlags::SHADER_READ,
                    old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image: self.render_target.borrow().image.inner,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                }],
            );
        }

        self.device.end_label(command_buffer);

        Ok(())
    }
}
