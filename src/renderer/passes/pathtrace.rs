use crate::renderer::descriptors::DescLayout;
use crate::renderer::render_target::{RenderTarget, RenderTargetBuilder};
use crate::renderer::{PushConstBuilder, VulkanRenderer};
use crate::scene::Scene;
use crate::vulkan::{CommandBuffer, Device, VulkanError};
use ash::vk;
use std::cell::RefCell;
use std::rc::Rc;

pub(crate) struct PathTracePass {
    pub device: Rc<Device>,
    pub direct_render_target: Rc<RefCell<RenderTarget>>,
    pub indirect_render_target: Rc<RefCell<RenderTarget>>,
}

impl PathTracePass {
    pub const TARGET_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

    pub fn render_target_defs() -> [RenderTargetBuilder; 2] {
        [
            RenderTargetBuilder::new("rt_direct")
                .with_storage()
                .with_transfer()
                .with_format(Self::TARGET_FORMAT),
            RenderTargetBuilder::new("rt_indirect")
                .with_storage()
                .with_transfer()
                .with_format(Self::TARGET_FORMAT),
        ]
    }

    pub const DESC_LAYOUTS: [DescLayout; 2] = [DescLayout::Global, DescLayout::Compute];

    pub fn record(
        &self,
        command_buffer: &CommandBuffer,
        renderer: &VulkanRenderer,
        scene: &Scene,
    ) -> Result<(), VulkanError> {
        self.device.begin_label("Path Tracing", command_buffer);

        let pipeline = renderer.pipeline_builder.get_rt("pt").unwrap();

        command_buffer.bind_rt_pipeline(pipeline);

        unsafe {
            self.device.inner.cmd_bind_descriptor_sets(
                command_buffer.inner,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                pipeline.layout,
                0,
                &[
                    renderer.descriptors.borrow().global_sets[renderer.current_frame].inner,
                    renderer.descriptors.borrow().compute_sets[renderer.current_frame].inner,
                ],
                &[],
            );

            let pc = PushConstBuilder::new()
                .add_u32(renderer.quality.pt_bounces as u32)
                .add_u32(*renderer.descriptors.borrow().samplers.get("gbuffer_depth").unwrap() as u32)
                .add_u32(*renderer.descriptors.borrow().samplers.get("gbuffer_normal").unwrap() as u32)
                .add_u32(*renderer.descriptors.borrow().storages.get("rt_direct").unwrap() as u32)
                .add_u32(*renderer.descriptors.borrow().storages.get("rt_indirect").unwrap() as u32)
                .add_u32(*renderer.descriptors.borrow().samplers.get("sky").unwrap() as u32)
                .add_f32(renderer.quality.rt_direct_trace_disance)
                .add_f32(renderer.quality.rt_indirect_trace_disance)
                .build();

            self.device.inner.cmd_push_constants(
                command_buffer.inner,
                pipeline.layout,
                vk::ShaderStageFlags::RAYGEN_KHR,
                0,
                &pc,
            );

            let (width, height) = if renderer.quality.half_res {
                (
                    renderer.swap_chain.extent.width / 2,
                    renderer.swap_chain.extent.height / 2,
                )
            } else {
                (renderer.swap_chain.extent.width, renderer.swap_chain.extent.height)
            };

            renderer.rt_pipeline_ext.loader.cmd_trace_rays(
                command_buffer.inner,
                &renderer.shader_binding_table.raygen_region,
                &renderer.shader_binding_table.miss_region,
                &renderer.shader_binding_table.hit_region,
                &renderer.shader_binding_table.call_region,
                width,
                height,
                1,
            );

            let barriers = [
                self.direct_render_target.borrow().image.inner,
                self.indirect_render_target.borrow().image.inner,
            ]
            .map(|image| vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::MEMORY_WRITE,
                dst_access_mask: vk::AccessFlags::MEMORY_READ,
                old_layout: vk::ImageLayout::GENERAL,
                new_layout: vk::ImageLayout::GENERAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            });

            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );
        }

        self.device.end_label(command_buffer);

        Ok(())
    }
}
