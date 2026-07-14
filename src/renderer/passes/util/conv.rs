use crate::err::AppError;
use crate::renderer::descriptors::{DescriptorLayouts, RendererDescriptors};
use crate::renderer::pipeline_builder::PipelineBuilder;
use crate::renderer::push_const::PushConstBuilder;
use crate::renderer::render_target::{
    RenderTarget, RenderTargetBuilder, RenderTargetSampler, RenderTargetSize, RenderTargets,
};
use crate::vulkan::{CommandBuffer, Compute, Device, Pipeline, VulkanError};
use ash::vk;
use std::cell::RefCell;
use std::rc::Rc;

/// This pass creates a convoluted sky map. It can be used as a building block for more complicated tasks.
///
/// Because of this, it does not issue its own barriers or have its own render target.
pub(crate) struct ConvolutionPass {
    device: Rc<Device>,
    pub conv_render_target: Rc<RefCell<RenderTarget>>,
    conv_pipeline: Rc<Pipeline<Compute>>,
}

impl ConvolutionPass {
    pub const OCTA_SIZE: [u32; 2] = [32, 32];

    pub fn create(
        device: Rc<Device>,
        render_targets: &mut RenderTargets,
        pipeline_builder: &mut PipelineBuilder,
        descriptor_layouts: &DescriptorLayouts,
    ) -> Result<Self, AppError> {
        let conv_pipeline = pipeline_builder.build_compute("conv", "conv|main", descriptor_layouts)?;

        Ok(Self {
            device,
            conv_render_target: render_targets.add(Self::octa_render_target_def())?,
            conv_pipeline,
        })
    }

    fn octa_render_target_def() -> RenderTargetBuilder {
        RenderTargetBuilder::new("conv_sky")
            .with_storage()
            .with_sampled()
            .with_format(vk::Format::R16G16B16A16_SFLOAT)
            .with_size(RenderTargetSize::Custom(Self::OCTA_SIZE[0], Self::OCTA_SIZE[1]))
            .with_sampler(RenderTargetSampler::Clamped)
    }

    pub fn record(
        &self,
        command_buffer: &CommandBuffer,
        descriptors: &RendererDescriptors,
        inputs: ConvolutionInputs,
    ) -> Result<(), VulkanError> {
        self.device.begin_label("Convolution", command_buffer);

        let pipeline = &self.conv_pipeline;

        command_buffer.bind_compute_pipeline(pipeline);
        command_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            [descriptors.global_set.inner, descriptors.compute_set.inner],
        );

        let pc = PushConstBuilder::with_capacity(3 * size_of::<f32>())
            .add_u32(inputs.src_sampler)
            .add_u32(self.conv_render_target.borrow().storage_index.unwrap())
            .add_f32(inputs.clamp)
            .build();

        command_buffer.push_constants(vk::ShaderStageFlags::COMPUTE, pipeline.layout, pc.as_ref());

        command_buffer.dispatch(Self::OCTA_SIZE[0], Self::OCTA_SIZE[1], 1);

        self.device.end_label(command_buffer);

        Ok(())
    }
}

pub struct ConvolutionInputs {
    pub src_sampler: u32,
    pub target_sampler: u32,
    pub clamp: f32,
}
