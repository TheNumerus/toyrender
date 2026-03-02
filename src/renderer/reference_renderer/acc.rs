use crate::err::AppError;
use crate::renderer::descriptors::RendererDescriptors;
use crate::renderer::pipeline_builder::PipelineBuilder;
use crate::renderer::render_target::{RenderTarget, RenderTargetBuilder, RenderTargets};
use crate::renderer::{FrameContext, PushConstBuilder, VulkanMcPathTracer};
use crate::vulkan::{CommandBuffer, Compute, Device, Pipeline, VulkanError};
use ash::vk;
use std::cell::{Ref, RefCell};
use std::rc::Rc;

pub struct AccumulatePass {
    device: Rc<Device>,
    render_target: Rc<RefCell<RenderTarget>>,
    pub pipeline_handle: Rc<Pipeline<Compute>>,
}

impl AccumulatePass {
    pub fn create(
        device: Rc<Device>,
        render_targets: &mut RenderTargets,
        pipeline_builder: &mut PipelineBuilder,
        descriptors: Ref<RendererDescriptors>,
    ) -> Result<Self, AppError> {
        let render_target = render_targets.add(Self::render_target_def())?;

        let pipeline_handle = pipeline_builder.build_compute("accumulator", "accumulator|main", descriptors)?;

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
        renderer: &VulkanMcPathTracer,
        context: &FrameContext,
        viewport: (u32, u32),
    ) -> Result<(), VulkanError> {
        let pipeline = &self.pipeline_handle;

        command_buffer.bind_compute_pipeline(pipeline);
        command_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            [
                renderer.descriptors.borrow().global_sets[renderer.current_frame].inner,
                renderer.descriptors.borrow().compute_sets[renderer.current_frame].inner,
            ],
        );

        let pc = PushConstBuilder::with_capacity(4 * size_of::<u32>())
            .add_u32(context.frame_index)
            .add_u32(if context.clear_taa { 1 } else { 0 })
            .add_u32(*renderer.descriptors.borrow().storages.get("rt_out").unwrap() as u32)
            .add_u32(*renderer.descriptors.borrow().storages.get("acc_out").unwrap() as u32);

        command_buffer.push_constants(vk::ShaderStageFlags::COMPUTE, pipeline.layout, &pc.build());

        let x = viewport.0 / pipeline.reflect_data.workgroup_size.0 + 1;
        let y = viewport.1 / pipeline.reflect_data.workgroup_size.1 + 1;

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
