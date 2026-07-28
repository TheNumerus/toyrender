use crate::err::AppError;
use crate::renderer::descriptors::{DescriptorLayouts, RendererDescriptors};
use crate::renderer::pipeline_builder::PipelineBuilder;
use crate::renderer::push_const::PushConstBuilder;
use crate::renderer::render_target::{
    RenderTarget, RenderTargetBuilder, RenderTargetSampler, RenderTargetSize, RenderTargets,
};
use crate::vulkan::{Buffer, CommandBuffer, Compute, Device, Pipeline, VulkanError};
use ash::vk;
use std::cell::RefCell;
use std::rc::Rc;

/// This pass creates a convoluted sky map. It can be used as a building block for more complicated tasks.
///
/// Because of this, it does not issue its own barriers or have its own render target.
pub(crate) struct ConvolutionPass {
    device: Rc<Device>,
    pub conv_render_target: Rc<RefCell<RenderTarget>>,
    init: RefCell<bool>,
    pub run: RefCell<bool>,
    conv_pipeline: Rc<Pipeline<Compute>>,
    sum_pipeline: Rc<Pipeline<Compute>>,
}

impl ConvolutionPass {
    /// number of pixels representing quarter of the sphere diameter
    pub const QUARTER_DIAMETER: u32 = 16;
    pub const OCTA_SIZE: [u32; 2] = [Self::QUARTER_DIAMETER * 4 + 1, Self::QUARTER_DIAMETER * 4 + 1];

    pub fn create(
        device: Rc<Device>,
        render_targets: &mut RenderTargets,
        pipeline_builder: &mut PipelineBuilder,
        descriptor_layouts: &DescriptorLayouts,
    ) -> Result<Self, AppError> {
        let conv_pipeline = pipeline_builder.build_compute("conv", "conv|main", descriptor_layouts)?;
        let sum_pipeline = pipeline_builder.build_compute("image_diff", "image_diff|main", descriptor_layouts)?;

        Ok(Self {
            device,
            conv_render_target: render_targets.add(Self::octa_render_target_def())?,
            init: RefCell::new(false),
            run: RefCell::new(true),
            conv_pipeline,
            sum_pipeline,
        })
    }

    fn octa_render_target_def() -> RenderTargetBuilder {
        RenderTargetBuilder::new("conv_sky")
            .with_storage()
            .with_sampled()
            .with_format(vk::Format::R32G32B32A32_SFLOAT)
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

        if !*self.init.borrow() {
            let image_color_res = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };

            unsafe {
                self.device.inner.cmd_pipeline_barrier(
                    command_buffer.inner,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier {
                        src_access_mask: vk::AccessFlags::NONE,
                        dst_access_mask: vk::AccessFlags::SHADER_WRITE,
                        old_layout: vk::ImageLayout::UNDEFINED,
                        new_layout: vk::ImageLayout::GENERAL,
                        image: self.conv_render_target.borrow().image.inner,
                        subresource_range: image_color_res,
                        ..Default::default()
                    }],
                );
            }

            self.init.replace(true);
        }

        if !*self.run.borrow() {
            return Ok(());
        }

        command_buffer.bind_compute_pipeline(pipeline);
        command_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            [descriptors.global_set.inner, descriptors.compute_set.inner],
        );

        let pc = PushConstBuilder::with_capacity(4 * size_of::<f32>())
            .add_u32(inputs.src_sampler)
            .add_u32(self.conv_render_target.borrow().storage_index.unwrap())
            .add_f32(inputs.clamp)
            .add_u32(inputs.iteration)
            .build();

        command_buffer.push_constants(vk::ShaderStageFlags::COMPUTE, pipeline.layout, pc.as_ref());

        command_buffer.dispatch(Self::OCTA_SIZE[0], Self::OCTA_SIZE[1], 1);

        let image_color_res = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

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
                    image: self.conv_render_target.borrow().image.inner,
                    subresource_range: image_color_res,
                    ..Default::default()
                }],
            );
        }

        let pipeline = &self.sum_pipeline;

        command_buffer.bind_compute_pipeline(pipeline);
        command_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            [descriptors.global_set.inner, descriptors.compute_set.inner],
        );

        let pc = PushConstBuilder::with_capacity(2 * size_of::<f32>())
            .add_u32(self.conv_render_target.borrow().storage_index.unwrap())
            .add_u32(self.conv_render_target.borrow().storage_index.unwrap())
            .build();

        command_buffer.push_constants(vk::ShaderStageFlags::COMPUTE, pipeline.layout, pc.as_ref());

        command_buffer.dispatch(1, 1, 1);

        unsafe {
            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[vk::BufferMemoryBarrier {
                    src_access_mask: vk::AccessFlags::SHADER_WRITE,
                    dst_access_mask: vk::AccessFlags::TRANSFER_READ,
                    buffer: inputs.buf_src.inner,
                    size: inputs.buf_src.size,
                    ..Default::default()
                }],
                &[],
            );

            self.device.inner.cmd_copy_buffer(
                command_buffer.inner,
                inputs.buf_src.inner,
                inputs.buf_dst.inner,
                &[vk::BufferCopy {
                    size: 64,
                    dst_offset: 0,
                    src_offset: 0,
                }],
            );

            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[vk::BufferMemoryBarrier {
                    src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    dst_access_mask: vk::AccessFlags::SHADER_READ,
                    buffer: inputs.buf_dst.inner,
                    size: inputs.buf_dst.size,
                    ..Default::default()
                }],
                &[],
            );
        }

        self.device.end_label(command_buffer);

        Ok(())
    }
}

pub struct ConvolutionInputs<'a> {
    pub src_sampler: u32,
    pub target_sampler: u32,
    pub clamp: f32,
    pub iteration: u32,
    pub buf_src: &'a Buffer,
    pub buf_dst: &'a Buffer,
}
