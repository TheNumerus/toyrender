use crate::renderer::descriptors::DescLayout;
use crate::renderer::render_target::{RenderTarget, RenderTargetBuilder};
use crate::renderer::{PushConstBuilder, VulkanRenderer};
use crate::vulkan::{CommandBuffer, Device, VulkanError};
use ash::vk;
use std::cell::RefCell;
use std::rc::Rc;

pub(crate) struct TaaPass {
    pub device: Rc<Device>,
    pub render_target: Rc<RefCell<RenderTarget>>,
    pub render_target_history: Rc<RefCell<RenderTarget>>,
}

impl TaaPass {
    pub const TARGET_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
    pub const DESC_LAYOUTS: [DescLayout; 2] = [DescLayout::Global, DescLayout::Compute];

    pub fn render_target_defs() -> [RenderTargetBuilder; 2] {
        [
            RenderTargetBuilder::new("taa_target")
                .with_transfer()
                .with_storage()
                .with_color_attachment()
                .with_sampled()
                .with_format(Self::TARGET_FORMAT),
            RenderTargetBuilder::new("taa_history_target")
                .with_transfer()
                .with_storage()
                .with_color_attachment()
                .with_sampled()
                .with_format(Self::TARGET_FORMAT),
        ]
    }

    pub fn record(
        &self,
        command_buffer: &CommandBuffer,
        renderer: &VulkanRenderer,
        clear: bool,
    ) -> Result<(), VulkanError> {
        self.device.begin_label("TAA Resolve", command_buffer);

        let pipeline = renderer.pipeline_builder.get_compute("taa").unwrap();

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

            let clear = if clear { 1 } else { 0 };

            let pc = PushConstBuilder::new()
                .add_u32(clear as u32)
                .add_u32(*renderer.descriptors.borrow().storages.get("taa_target").unwrap() as u32)
                .add_u32(*renderer.descriptors.borrow().samplers.get("tonemap").unwrap() as u32)
                .add_u32(
                    *renderer
                        .descriptors
                        .borrow()
                        .samplers
                        .get("taa_history_target")
                        .unwrap() as u32,
                )
                .add_u32(*renderer.descriptors.borrow().samplers.get("gbuffer_depth").unwrap() as u32)
                .add_u32(*renderer.descriptors.borrow().samplers.get("last_depth").unwrap() as u32)
                .build();

            self.device.inner.cmd_push_constants(
                command_buffer.inner,
                pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &pc,
            );

            let x = (renderer.swap_chain.extent.width / 16) + 1;
            let y = (renderer.swap_chain.extent.height / 16) + 1;

            self.device.inner.cmd_dispatch(command_buffer.inner, x, y, 1);

            let extent_3d = vk::Extent3D {
                width: renderer.swap_chain.extent.width,
                height: renderer.swap_chain.extent.height,
                depth: 1,
            };

            let barriers = [renderer.render_targets.get_ref("last_depth").unwrap().image.inner].map(|image| {
                vk::ImageMemoryBarrier {
                    src_access_mask: vk::AccessFlags::MEMORY_WRITE,
                    dst_access_mask: vk::AccessFlags::MEMORY_READ,
                    old_layout: vk::ImageLayout::GENERAL,
                    new_layout: vk::ImageLayout::GENERAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::DEPTH,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                }
            });

            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
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

            self.device.inner.cmd_copy_image(
                command_buffer.inner,
                renderer
                    .render_targets
                    .get("gbuffer_depth")
                    .unwrap()
                    .borrow()
                    .image
                    .inner,
                vk::ImageLayout::GENERAL,
                renderer.render_targets.get("last_depth").unwrap().borrow().image.inner,
                vk::ImageLayout::GENERAL,
                &[vk::ImageCopy {
                    extent: extent_3d,
                    dst_subresource: image_depth_res,
                    src_subresource: image_depth_res,
                    ..Default::default()
                }],
            );

            self.device.inner.cmd_copy_image(
                command_buffer.inner,
                renderer
                    .render_targets
                    .get_ref("denoise_direct_out")
                    .unwrap()
                    .image
                    .inner,
                vk::ImageLayout::GENERAL,
                renderer
                    .render_targets
                    .get_ref("denoise_direct_history")
                    .unwrap()
                    .image
                    .inner,
                vk::ImageLayout::GENERAL,
                &[vk::ImageCopy {
                    extent: extent_3d,
                    dst_subresource: image_color_res,
                    src_subresource: image_color_res,
                    ..Default::default()
                }],
            );
            self.device.inner.cmd_copy_image(
                command_buffer.inner,
                renderer
                    .render_targets
                    .get_ref("denoise_indirect_out")
                    .unwrap()
                    .image
                    .inner,
                vk::ImageLayout::GENERAL,
                renderer
                    .render_targets
                    .get_ref("denoise_indirect_history")
                    .unwrap()
                    .image
                    .inner,
                vk::ImageLayout::GENERAL,
                &[vk::ImageCopy {
                    extent: extent_3d,
                    dst_subresource: image_color_res,
                    src_subresource: image_color_res,
                    ..Default::default()
                }],
            );

            self.device.inner.cmd_copy_image(
                command_buffer.inner,
                self.render_target.borrow().image.inner,
                vk::ImageLayout::GENERAL,
                self.render_target_history.borrow().image.inner,
                vk::ImageLayout::GENERAL,
                &[vk::ImageCopy {
                    extent: extent_3d,
                    dst_subresource: image_color_res,
                    src_subresource: image_color_res,
                    ..Default::default()
                }],
            );
        }

        self.device.end_label(command_buffer);

        Ok(())
    }
}
