use crate::renderer::descriptors::DescLayout;
use crate::renderer::render_target::{RenderTarget, RenderTargetBuilder, RenderTargetSize};
use crate::renderer::{PushConstBuilder, VulkanRenderer};
use crate::vulkan::{CommandBuffer, Device, VulkanError};
use ash::vk;
use std::cell::RefCell;
use std::rc::Rc;

pub(crate) struct SkyPass {
    pub device: Rc<Device>,
    pub render_target: Rc<RefCell<RenderTarget>>,
    pub is_init: RefCell<bool>,
}

impl SkyPass {
    pub const SKY_SIZE: [u32; 2] = [256, 128];
    pub const TARGET_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
    pub const DESC_LAYOUTS: [DescLayout; 2] = [DescLayout::Global, DescLayout::Compute];

    pub fn render_target_def() -> RenderTargetBuilder {
        RenderTargetBuilder::new("sky")
            .with_storage()
            .with_format(Self::TARGET_FORMAT)
            .with_size(RenderTargetSize::Custom(Self::SKY_SIZE[0], Self::SKY_SIZE[1]))
    }

    pub fn record(&self, command_buffer: &CommandBuffer, renderer: &VulkanRenderer) -> Result<(), VulkanError> {
        self.device.begin_label("Sky", command_buffer);

        let pipeline = renderer.pipeline_builder.get_compute("sky").unwrap();

        if !*self.is_init.borrow() {
            let barrier = vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::SHADER_READ,
                dst_access_mask: vk::AccessFlags::SHADER_WRITE,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::GENERAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: self.render_target.borrow().image.inner,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            };

            unsafe {
                self.device.inner.cmd_pipeline_barrier(
                    command_buffer.inner,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }

            *self.is_init.borrow_mut() = true;
        }

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
                .add_u32(*renderer.descriptors.borrow().storages.get("sky").unwrap() as u32)
                .build();

            self.device.inner.cmd_push_constants(
                command_buffer.inner,
                pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &pc,
            );

            let x = (Self::SKY_SIZE[0] / 16);
            let y = (Self::SKY_SIZE[1] / 16);

            self.device.inner.cmd_dispatch(command_buffer.inner, x, y, 1);

            let barrier = vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::SHADER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                old_layout: vk::ImageLayout::GENERAL,
                new_layout: vk::ImageLayout::GENERAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: self.render_target.borrow().image.inner,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            };

            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

            let image_color_res = vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            };
            let image_depth_res = vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                ..image_color_res
            };
        }

        self.device.end_label(command_buffer);

        Ok(())
    }
}
