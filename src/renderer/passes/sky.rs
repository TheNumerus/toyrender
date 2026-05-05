use crate::err::AppError;
use crate::math;
use crate::renderer::PushConstBuilder;
use crate::renderer::descriptors::{DescriptorLayouts, RendererDescriptors};
use crate::renderer::pipeline_builder::PipelineBuilder;
use crate::renderer::render_target::{
    RenderTarget, RenderTargetBuilder, RenderTargetSampler, RenderTargetSize, RenderTargets,
};
use crate::vulkan::{CommandBuffer, Compute, Device, Pipeline, VulkanError};
use ash::vk;
use std::cell::RefCell;
use std::rc::Rc;

pub(crate) mod importance_map;

pub(crate) struct SkyPass {
    device: Rc<Device>,
    pub render_target: Rc<RefCell<RenderTarget>>,
    pipeline_handle: Rc<Pipeline<Compute>>,
}

impl SkyPass {
    pub const SKY_SIZE: [u32; 2] = [256, 128];

    pub fn create(
        device: Rc<Device>,
        render_targets: &mut RenderTargets,
        pipeline_builder: &mut PipelineBuilder,
        descriptor_layouts: &DescriptorLayouts,
    ) -> Result<Self, AppError> {
        let pipeline = pipeline_builder.build_compute("sky", "sky|main", descriptor_layouts)?;

        Ok(Self {
            device,
            render_target: render_targets.add(Self::render_target_def())?,
            pipeline_handle: pipeline,
        })
    }

    fn render_target_def() -> RenderTargetBuilder {
        RenderTargetBuilder::new("sky")
            .with_storage()
            .with_sampled()
            .with_sampler(RenderTargetSampler::RepeatXOnly)
            .with_format(vk::Format::R16G16B16A16_SFLOAT)
            .with_size(RenderTargetSize::Custom(Self::SKY_SIZE[0], Self::SKY_SIZE[1]))
    }

    pub fn record(&self, command_buffer: &CommandBuffer, descriptors: &RendererDescriptors) -> Result<(), VulkanError> {
        self.device.begin_label("Sky", command_buffer);

        let pipeline = &self.pipeline_handle;

        command_buffer.bind_compute_pipeline(pipeline);
        command_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            [descriptors.global_set.inner, descriptors.compute_set.inner],
        );

        let pc = PushConstBuilder::with_capacity(size_of::<u32>())
            .add_u32(self.render_target.borrow().storage_index.unwrap())
            .build();

        command_buffer.push_constants(vk::ShaderStageFlags::COMPUTE, pipeline.layout, &pc);

        let x = math::workgroup_saturate(Self::SKY_SIZE[0], pipeline.reflect_data.workgroup_size.0);
        let y = math::workgroup_saturate(Self::SKY_SIZE[1], pipeline.reflect_data.workgroup_size.1);

        command_buffer.dispatch(x, y, 1);

        self.device.end_label(command_buffer);

        Ok(())
    }
}
