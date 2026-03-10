use crate::err::AppError;
use crate::math;
use crate::renderer::descriptors::{DescriptorLayouts, RendererDescriptors};
use crate::renderer::pipeline_builder::PipelineBuilder;
use crate::renderer::push_const::PushConstBuilder;
use crate::renderer::render_target::{RenderTarget, RenderTargetBuilder, RenderTargets};
use crate::vulkan::{CommandBuffer, Compute, Device, Pipeline, VulkanError};
use ash::vk;
use std::cell::RefCell;
use std::rc::Rc;

pub(crate) struct ShadingPass {
    device: Rc<Device>,
    pub render_target: Rc<RefCell<RenderTarget>>,
    pipeline_handle: Rc<Pipeline<Compute>>,
}

impl ShadingPass {
    pub fn create(
        device: Rc<Device>,
        render_targets: &mut RenderTargets,
        pipeline_builder: &mut PipelineBuilder,
        descriptor_layouts: &DescriptorLayouts,
    ) -> Result<Self, AppError> {
        let pipeline = pipeline_builder.build_compute("light", "light|main", descriptor_layouts)?;

        Ok(Self {
            device,
            render_target: render_targets.add(Self::render_target_def())?,
            pipeline_handle: pipeline,
        })
    }

    fn render_target_def() -> RenderTargetBuilder {
        RenderTargetBuilder::new("tonemap")
            .with_color_attachment()
            .with_storage()
            .with_transfer()
            .with_format(vk::Format::R16G16B16A16_SFLOAT)
            .with_sampled()
    }

    pub fn record(
        &self,
        command_buffer: &CommandBuffer,
        descriptors: &RendererDescriptors,
        inputs: ShadingInputs,
        viewport: (u32, u32),
    ) -> Result<(), VulkanError> {
        self.device.begin_label("Lighting", command_buffer);

        unsafe {
            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier {
                    src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ,
                    dst_access_mask: vk::AccessFlags::SHADER_WRITE,
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::GENERAL,
                    image: self.render_target.borrow().image.inner,
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

        let pipeline = &self.pipeline_handle;

        command_buffer.bind_compute_pipeline(pipeline);

        command_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            [descriptors.global_set.inner, descriptors.compute_set.inner],
        );

        let pc = PushConstBuilder::new()
            .add_u32(self.render_target.borrow().storage_index.unwrap())
            .add_u32(inputs.color.sampler_index.unwrap())
            .add_u32(inputs.normal.sampler_index.unwrap())
            .add_u32(inputs.depth.sampler_index.unwrap())
            .add_u32(inputs.direct.sampler_index.unwrap())
            .add_u32(inputs.indirect.sampler_index.unwrap())
            .add_u32(inputs.sky_sampler)
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
                    image: self.render_target.borrow().image.inner,
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

pub struct ShadingInputs<'a> {
    pub color: &'a RenderTarget,
    pub depth: &'a RenderTarget,
    pub normal: &'a RenderTarget,
    pub direct: &'a RenderTarget,
    pub indirect: &'a RenderTarget,
    pub sky_sampler: u32,
}
