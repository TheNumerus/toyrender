use crate::renderer::pipeline_builder::PipelineHandle;
use crate::renderer::render_target::RenderTargetBuilder;
use crate::renderer::{FrameContext, PushConstBuilder, VulkanContext, VulkanMcPathTracer};
use crate::vulkan::{CommandBuffer, Device, VulkanError};
use ash::vk;
use std::rc::Rc;

pub struct ReferencePathtracePass {
    pub context: Rc<VulkanContext>,
    pub pipeline_handle: PipelineHandle,
}

impl ReferencePathtracePass {
    pub const TARGET_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

    pub fn render_target_def() -> RenderTargetBuilder {
        RenderTargetBuilder::new("rt_out")
            .with_storage()
            .with_transfer()
            .with_format(Self::TARGET_FORMAT)
    }

    pub fn record_pt(
        &self,
        command_buffer: &CommandBuffer,
        renderer: &VulkanMcPathTracer,
        context: &FrameContext,
        viewport: (u32, u32),
        fov: f32,
    ) -> Result<(), VulkanError> {
        self.context.device.begin_label("Path Tracing", command_buffer);

        let pipeline = renderer.pipeline_builder.get_rt(&self.pipeline_handle).unwrap();

        command_buffer.bind_rt_pipeline(pipeline);

        unsafe {
            let barriers = [renderer.render_targets.get("sky").unwrap().borrow().image.inner].map(|image| {
                vk::ImageMemoryBarrier {
                    src_access_mask: vk::AccessFlags::SHADER_WRITE,
                    dst_access_mask: vk::AccessFlags::SHADER_READ,
                    old_layout: vk::ImageLayout::GENERAL,
                    new_layout: vk::ImageLayout::GENERAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image,
                    subresource_range: crate::vulkan::Image::single_color_layer_range(),
                    ..Default::default()
                }
            });

            self.context.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );

            command_buffer.bind_descriptor_sets(
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                pipeline.layout,
                [
                    renderer.descriptors.borrow().global_sets[renderer.current_frame].inner,
                    renderer.descriptors.borrow().compute_sets[renderer.current_frame].inner,
                ],
            );

            let pc = PushConstBuilder::with_capacity(7 * size_of::<u32>())
                .add_u32(context.frame_index)
                .add_u32(renderer.quality.pt_bounces as u32)
                .add_u32(*renderer.descriptors.borrow().storages.get("rt_out").unwrap() as u32)
                .add_u32(*renderer.descriptors.borrow().samplers.get("sky").unwrap() as u32)
                .add_f32(renderer.quality.rt_direct_trace_distance)
                .add_f32(renderer.quality.rt_indirect_trace_distance)
                .add_f32(fov)
                .build();

            command_buffer.push_constants(
                vk::ShaderStageFlags::RAYGEN_KHR
                    | vk::ShaderStageFlags::MISS_KHR
                    | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                pipeline.layout,
                &pc,
            );

            self.context.rt_pipeline_ext.loader.cmd_trace_rays(
                command_buffer.inner,
                &renderer.shader_binding_table.raygen_region,
                &renderer.shader_binding_table.miss_region,
                &renderer.shader_binding_table.hit_region,
                &renderer.shader_binding_table.call_region,
                viewport.0,
                viewport.1,
                1,
            );

            let barriers = [renderer.render_targets.get("rt_out").unwrap().borrow().image.inner].map(|image| {
                vk::ImageMemoryBarrier {
                    src_access_mask: vk::AccessFlags::SHADER_WRITE,
                    dst_access_mask: vk::AccessFlags::SHADER_READ,
                    old_layout: vk::ImageLayout::GENERAL,
                    new_layout: vk::ImageLayout::GENERAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image,
                    subresource_range: crate::vulkan::Image::single_color_layer_range(),
                    ..Default::default()
                }
            });

            self.context.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );
        }

        self.context.device.end_label(command_buffer);

        Ok(())
    }
}

struct PtPassInput {}
