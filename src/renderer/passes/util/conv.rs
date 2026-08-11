use crate::err::AppError;
use crate::renderer::descriptors::{DescriptorLayouts, RendererDescriptors};
use crate::renderer::pipeline_builder::{PipelineBuilder, SpecConsts};
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
    pub conv_sun_render_target: Rc<RefCell<RenderTarget>>,
    init: RefCell<bool>,
    pub state: ConvolutionPassState,
    conv_pipeline: Rc<Pipeline<Compute>>,
    conv_sunless_pipeline: Rc<Pipeline<Compute>>,
    sum_pipeline: Rc<Pipeline<Compute>>,
    max_init_pipeline: Rc<Pipeline<Compute>>,
    max_pipeline: Rc<Pipeline<Compute>>,
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
        let conv_pipeline =
            pipeline_builder.build_compute("conv", "conv|main", descriptor_layouts, Some(SpecConsts::new().push(0)))?;
        let conv_sunless_pipeline =
            pipeline_builder.build_compute("conv", "conv|main", descriptor_layouts, Some(SpecConsts::new().push(1)))?;
        let sum_pipeline = pipeline_builder.build_compute("image_avg", "image_avg|main", descriptor_layouts, None)?;
        let max_init_pipeline = pipeline_builder.build_compute(
            "find_bright_spot_init",
            "find_bright_spot|mainInit",
            descriptor_layouts,
            None,
        )?;
        let max_pipeline = pipeline_builder.build_compute(
            "find_bright_spot",
            "find_bright_spot|mainSuccessive",
            descriptor_layouts,
            None,
        )?;

        Ok(Self {
            device,
            conv_render_target: render_targets.add(Self::octa_render_target_def())?,
            conv_sun_render_target: render_targets.add(Self::octa_render_target_def().duplicate("conv_sun_sky"))?,
            init: RefCell::new(false),
            state: ConvolutionPassState::Init,
            conv_pipeline,
            conv_sunless_pipeline,
            sum_pipeline,
            max_pipeline,
            max_init_pipeline,
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

        if self.state == ConvolutionPassState::Finished {
            self.device.end_label(command_buffer);
            return Ok(());
        }

        let sun_storage = self.conv_sun_render_target.borrow().storage_index.unwrap();
        let sunless_storage = self.conv_render_target.borrow().storage_index.unwrap();

        match self.state {
            ConvolutionPassState::Init => {}
            ConvolutionPassState::FindingMax => {
                self.find_max(command_buffer, descriptors, &inputs);
                self.copy_buf(command_buffer, &inputs);
            }
            ConvolutionPassState::SunProbe(i) => {
                self.conv(
                    command_buffer,
                    descriptors,
                    &inputs,
                    i,
                    &self.conv_pipeline,
                    sun_storage,
                );
                self.sum(command_buffer, descriptors, sun_storage);
                self.copy_buf(command_buffer, &inputs);
            }
            ConvolutionPassState::SunlessProbe(i) => {
                self.conv(
                    command_buffer,
                    descriptors,
                    &inputs,
                    i,
                    &self.conv_sunless_pipeline,
                    sunless_storage,
                );
                self.sum(command_buffer, descriptors, sunless_storage);
                self.copy_buf(command_buffer, &inputs);
            }
            ConvolutionPassState::Finished => {}
            ConvolutionPassState::Debug => {
                self.conv(
                    command_buffer,
                    descriptors,
                    &inputs,
                    0,
                    &self.conv_pipeline,
                    sun_storage,
                );
                self.sum(command_buffer, descriptors, sun_storage);
                self.find_max(command_buffer, descriptors, &inputs);
                self.conv(
                    command_buffer,
                    descriptors,
                    &inputs,
                    0,
                    &self.conv_sunless_pipeline,
                    sunless_storage,
                );
                self.sum(command_buffer, descriptors, sunless_storage);
                self.copy_buf(command_buffer, &inputs);
            }
        }

        self.device.end_label(command_buffer);

        Ok(())
    }

    fn conv(
        &self,
        command_buffer: &CommandBuffer,
        descriptors: &RendererDescriptors,
        inputs: &ConvolutionInputs,
        iteration: u32,
        pipeline: &Rc<Pipeline<Compute>>,
        dst_storage: u32,
    ) {
        command_buffer.bind_compute_pipeline(pipeline);
        command_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            [descriptors.global_set.inner, descriptors.compute_set.inner],
        );

        let pc = PushConstBuilder::with_capacity(4 * size_of::<f32>())
            .add_u32(inputs.src_sampler)
            .add_u32(dst_storage)
            .add_u32(iteration + 1)
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
    }

    fn sum(&self, command_buffer: &CommandBuffer, descriptors: &RendererDescriptors, src_storage: u32) {
        let pipeline = &self.sum_pipeline;

        command_buffer.bind_compute_pipeline(pipeline);
        command_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            [descriptors.global_set.inner, descriptors.compute_set.inner],
        );

        let pc = PushConstBuilder::with_capacity(size_of::<f32>())
            .add_u32(src_storage)
            .build();

        command_buffer.push_constants(vk::ShaderStageFlags::COMPUTE, pipeline.layout, pc.as_ref());

        command_buffer.dispatch(1, 1, 1);
    }

    fn find_max(&self, command_buffer: &CommandBuffer, descriptors: &RendererDescriptors, inputs: &ConvolutionInputs) {
        let pipeline = &self.max_init_pipeline;

        let mut count = inputs.src_res.0 * inputs.src_res.1;

        command_buffer.bind_compute_pipeline(pipeline);
        command_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            [descriptors.global_set.inner, descriptors.compute_set.inner],
        );

        let pc = PushConstBuilder::with_capacity(2 * size_of::<f32>())
            .add_u32(inputs.src_storage)
            .add_u32(count)
            .build();

        command_buffer.push_constants(vk::ShaderStageFlags::COMPUTE, pipeline.layout, pc.as_ref());

        count /= 1024;
        count = count.max(1);

        command_buffer.dispatch(1, count, 1);

        unsafe {
            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[
                    vk::BufferMemoryBarrier {
                        src_access_mask: vk::AccessFlags::SHADER_WRITE,
                        dst_access_mask: vk::AccessFlags::SHADER_READ,
                        buffer: inputs.buf_src.inner,
                        size: inputs.buf_src.size,
                        ..Default::default()
                    },
                    vk::BufferMemoryBarrier {
                        src_access_mask: vk::AccessFlags::SHADER_WRITE,
                        dst_access_mask: vk::AccessFlags::SHADER_READ,
                        buffer: inputs.buf_scratch.inner,
                        size: inputs.buf_scratch.size,
                        ..Default::default()
                    },
                ],
                &[],
            );
        }

        loop {
            if count == 1 {
                break;
            }

            let pipeline = &self.max_pipeline;

            command_buffer.bind_compute_pipeline(pipeline);
            command_buffer.bind_descriptor_sets(
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                [descriptors.global_set.inner, descriptors.compute_set.inner],
            );

            let pc = PushConstBuilder::with_capacity(2 * size_of::<f32>())
                .add_u32(inputs.src_storage)
                .add_u32(count)
                .build();

            command_buffer.push_constants(vk::ShaderStageFlags::COMPUTE, pipeline.layout, pc.as_ref());

            count /= 1024;
            count = count.max(1);

            command_buffer.dispatch(1, count, 1);

            unsafe {
                self.device.inner.cmd_pipeline_barrier(
                    command_buffer.inner,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[
                        vk::BufferMemoryBarrier {
                            src_access_mask: vk::AccessFlags::SHADER_WRITE,
                            dst_access_mask: vk::AccessFlags::SHADER_READ,
                            buffer: inputs.buf_src.inner,
                            size: inputs.buf_src.size,
                            ..Default::default()
                        },
                        vk::BufferMemoryBarrier {
                            src_access_mask: vk::AccessFlags::SHADER_WRITE,
                            dst_access_mask: vk::AccessFlags::SHADER_READ,
                            buffer: inputs.buf_scratch.inner,
                            size: inputs.buf_scratch.size,
                            ..Default::default()
                        },
                    ],
                    &[],
                );
            }
        }
    }

    fn copy_buf(&self, command_buffer: &CommandBuffer, inputs: &ConvolutionInputs) {
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
                    size: 128,
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
    }
}

pub struct ConvolutionInputs<'a> {
    pub src_sampler: u32,
    pub src_storage: u32,
    pub src_res: (u32, u32),
    pub target_sampler: u32,
    pub clamp: f32,
    pub buf_src: &'a Buffer,
    pub buf_dst: &'a Buffer,
    pub buf_scratch: &'a Buffer,
}

#[derive(PartialEq, Copy, Clone, Debug)]
pub(crate) enum ConvolutionPassState {
    /// Resets the convolution
    Init,
    FindingMax,
    SunProbe(u32),
    SunlessProbe(u32),
    Finished,
    Debug,
}
