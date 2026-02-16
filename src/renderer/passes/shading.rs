use crate::renderer::VulkanRenderer;
use crate::renderer::descriptors::DescLayout;
use crate::renderer::push_const::PushConstBuilder;
use crate::renderer::render_target::{RenderTarget, RenderTargetBuilder};
use crate::vulkan::{CommandBuffer, Device, VulkanError};
use ash::vk;
use std::cell::RefCell;
use std::rc::Rc;

pub(crate) struct ShadingPass {
    pub device: Rc<Device>,
    pub render_target: Rc<RefCell<RenderTarget>>,
}

impl ShadingPass {
    pub const TARGET_FORMATS: [vk::Format; 1] = [vk::Format::R16G16B16A16_SFLOAT];
    pub const DESC_LAYOUTS: [DescLayout; 2] = [DescLayout::Global, DescLayout::Compute];

    pub fn render_target_def() -> RenderTargetBuilder {
        RenderTargetBuilder::new("tonemap")
            .with_color_attachment()
            .with_storage()
            .with_transfer()
            .with_format(Self::TARGET_FORMATS[0])
            .with_sampled()
    }

    pub fn record(
        &self,
        command_buffer: &CommandBuffer,
        renderer: &VulkanRenderer,
        viewport: (u32, u32),
    ) -> Result<(), VulkanError> {
        self.device.begin_label("Lighting", command_buffer);

        unsafe {
            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier {
                    src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ,
                    dst_access_mask: vk::AccessFlags::SHADER_WRITE,
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::GENERAL,
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

        let pipeline = renderer.pipeline_builder.get_compute("light").unwrap();

        command_buffer.bind_compute_pipeline(pipeline);

        unsafe {
            self.device.inner.cmd_bind_descriptor_sets(
                command_buffer.inner,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                0,
                &[
                    renderer.descriptors.borrow().global_sets[renderer.current_frame].inner,
                    renderer.descriptors.borrow().compute_sets[renderer.current_frame].inner,
                ],
                &[],
            );

            let pc = PushConstBuilder::new()
                .add_u32(*renderer.descriptors.borrow().storages.get("tonemap").unwrap() as u32)
                .add_u32(*renderer.descriptors.borrow().samplers.get("gbuffer_color").unwrap() as u32)
                .add_u32(*renderer.descriptors.borrow().samplers.get("gbuffer_normal").unwrap() as u32)
                .add_u32(*renderer.descriptors.borrow().samplers.get("gbuffer_depth").unwrap() as u32)
                .add_u32(
                    *renderer
                        .descriptors
                        .borrow()
                        .samplers
                        .get("denoise_direct_out")
                        .unwrap() as u32,
                )
                .add_u32(
                    *renderer
                        .descriptors
                        .borrow()
                        .samplers
                        .get("denoise_indirect_out")
                        .unwrap() as u32,
                )
                .add_u32(*renderer.descriptors.borrow().samplers.get("sky").unwrap() as u32)
                .build();

            self.device.inner.cmd_push_constants(
                command_buffer.inner,
                pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &pc,
            );

            let x = (viewport.0 / 16) + 1;
            let y = (viewport.1 / 16) + 1;

            self.device.inner.cmd_dispatch(command_buffer.inner, x, y, 1);
        }

        unsafe {
            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier {
                    src_access_mask: vk::AccessFlags::SHADER_WRITE,
                    dst_access_mask: vk::AccessFlags::SHADER_READ,
                    old_layout: vk::ImageLayout::GENERAL,
                    new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
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
