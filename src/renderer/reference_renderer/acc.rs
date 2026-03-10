use crate::err::AppError;
use crate::math;
use crate::renderer::PushConstBuilder;
use crate::renderer::descriptors::{DescriptorLayouts, RendererDescriptors};
use crate::renderer::pipeline_builder::PipelineBuilder;
use crate::renderer::render_target::{RenderTarget, RenderTargetBuilder, RenderTargets};
use crate::vulkan::{CommandBuffer, Compute, Device, Pipeline, VulkanError};
use ash::vk;
use std::cell::RefCell;
use std::rc::Rc;

pub struct AccumulatePass {
    device: Rc<Device>,
    pub render_target: Rc<RefCell<RenderTarget>>,
    pub pipeline_handle: Rc<Pipeline<Compute>>,
}

impl AccumulatePass {
    pub fn create(
        device: Rc<Device>,
        render_targets: &mut RenderTargets,
        pipeline_builder: &mut PipelineBuilder,
        descriptor_layouts: &DescriptorLayouts,
    ) -> Result<Self, AppError> {
        let render_target = render_targets.add(Self::render_target_def())?;

        let pipeline_handle = pipeline_builder.build_compute("accumulator", "accumulator|main", descriptor_layouts)?;

        Ok(Self {
            device,
            render_target,
            pipeline_handle,
        })
    }

    fn render_target_def() -> RenderTargetBuilder {
        // need full fat buffer for increased precision
        RenderTargetBuilder::new("acc_out")
            .with_storage()
            .with_transfer()
            .with_format(vk::Format::R32G32B32A32_SFLOAT)
    }

    pub fn record(
        &self,
        command_buffer: &CommandBuffer,
        descriptors: &RendererDescriptors,
        inputs: AccumulateInputs,
        viewport: (u32, u32),
    ) -> Result<(), VulkanError> {
        let pipeline = &self.pipeline_handle;

        command_buffer.bind_compute_pipeline(pipeline);
        command_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            [descriptors.global_set.inner, descriptors.compute_set.inner],
        );

        let pc = PushConstBuilder::with_capacity(4 * size_of::<u32>())
            .add_u32(inputs.frame_index)
            .add_u32(if inputs.clear { 1 } else { 0 })
            .add_u32(inputs.input.storage_index.unwrap())
            .add_u32(self.render_target.borrow().storage_index.unwrap());

        command_buffer.push_constants(vk::ShaderStageFlags::COMPUTE, pipeline.layout, &pc.build());

        let x = math::workgroup_saturate(viewport.0, pipeline.reflect_data.workgroup_size.0);
        let y = math::workgroup_saturate(viewport.1, pipeline.reflect_data.workgroup_size.1);

        command_buffer.dispatch(x, y, 1);

        let barrier = vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::SHADER_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            old_layout: vk::ImageLayout::GENERAL,
            new_layout: vk::ImageLayout::GENERAL,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: self.render_target.borrow().image.inner,
            subresource_range: crate::vulkan::Image::single_color_layer_range(),
            ..Default::default()
        };

        unsafe {
            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }

        Ok(())
    }
}

pub struct AccumulateInputs<'a> {
    pub input: &'a RenderTarget,
    pub frame_index: u32,
    pub clear: bool,
}
