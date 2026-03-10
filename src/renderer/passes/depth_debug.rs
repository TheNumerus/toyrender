use crate::err::AppError;
use crate::math;
use crate::renderer::descriptors::{DescriptorLayouts, RendererDescriptors};
use crate::renderer::pipeline_builder::PipelineBuilder;
use crate::renderer::push_const::PushConstBuilder;
use crate::renderer::render_target::RenderTarget;
use crate::vulkan::{CommandBuffer, Compute, Device, Pipeline, VulkanError};
use ash::vk;
use std::cell::RefCell;
use std::rc::Rc;

pub(crate) struct DepthDebugPass {
    device: Rc<Device>,
    pipeline_handle: Rc<Pipeline<Compute>>,
}

impl DepthDebugPass {
    pub fn create(
        device: Rc<Device>,
        pipeline_builder: &mut PipelineBuilder,
        descriptor_layouts: &DescriptorLayouts,
    ) -> Result<Self, AppError> {
        let pipeline = pipeline_builder.build_compute("depth_convert", "depth_convert|main", descriptor_layouts)?;

        Ok(Self {
            device,
            pipeline_handle: pipeline,
        })
    }

    pub fn record(
        &self,
        command_buffer: &CommandBuffer,
        descriptors: &RendererDescriptors,
        render_target: Rc<RefCell<RenderTarget>>,
        depth_render_target: Rc<RefCell<RenderTarget>>,
        viewport: (u32, u32),
    ) -> Result<(), VulkanError> {
        self.device.begin_label("Depth Convert", command_buffer);

        let pipeline = &self.pipeline_handle;

        command_buffer.bind_compute_pipeline(pipeline);

        command_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            [descriptors.global_set.inner, descriptors.compute_set.inner],
        );

        let pc = PushConstBuilder::new()
            .add_u32(render_target.borrow().storage_index.unwrap())
            .add_u32(depth_render_target.borrow().sampler_index.unwrap())
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
                    image: render_target.borrow().image.inner,
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
