use crate::err::AppError;
use crate::math;
use crate::renderer::VulkanRenderer;
use crate::renderer::descriptors::RendererDescriptors;
use crate::renderer::pipeline_builder::PipelineBuilder;
use crate::renderer::push_const::PushConstBuilder;
use crate::vulkan::{CommandBuffer, Compute, Device, Pipeline, VulkanError};
use ash::vk;
use std::cell::Ref;
use std::rc::Rc;

pub(crate) struct DepthDebugPass {
    device: Rc<Device>,
    pipeline_handle: Rc<Pipeline<Compute>>,
}

impl DepthDebugPass {
    pub fn create(
        device: Rc<Device>,
        pipeline_builder: &mut PipelineBuilder,
        descriptors: Ref<RendererDescriptors>,
    ) -> Result<Self, AppError> {
        let pipeline = pipeline_builder.build_compute("depth_convert", "depth_convert|main", descriptors)?;

        Ok(Self {
            device,
            pipeline_handle: pipeline,
        })
    }

    pub fn record(
        &self,
        command_buffer: &CommandBuffer,
        renderer: &VulkanRenderer,
        viewport: (u32, u32),
    ) -> Result<(), VulkanError> {
        self.device.begin_label("Depth Convert", command_buffer);

        let pipeline = &self.pipeline_handle;

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
        }

        let pc = PushConstBuilder::new()
            .add_u32(*renderer.descriptors.borrow().storages.get("tonemap").unwrap() as u32)
            .add_u32(*renderer.descriptors.borrow().samplers.get("gbuffer_depth").unwrap() as u32)
            .build();

        command_buffer.push_constants(vk::ShaderStageFlags::COMPUTE, pipeline.layout, pc.as_ref());

        let x = math::workgroup_saturate(viewport.0, pipeline.reflect_data.workgroup_size.0);
        let y = math::workgroup_saturate(viewport.1, pipeline.reflect_data.workgroup_size.1);

        command_buffer.dispatch(x, y, 1);

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
                    new_layout: vk::ImageLayout::GENERAL,
                    image: renderer.passes.shading.render_target.borrow().image.inner,
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
