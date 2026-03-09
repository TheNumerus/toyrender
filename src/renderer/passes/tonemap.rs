use crate::err::AppError;
use crate::renderer::descriptors::RendererDescriptors;
use crate::renderer::pipeline_builder::PipelineBuilder;
use crate::renderer::push_const::PushConstBuilder;
use crate::renderer::render_target::{RenderTarget, RenderTargetBuilder, RenderTargets};
use crate::renderer::{VulkanMcPathTracer, VulkanRenderer};
use crate::vulkan::{CommandBuffer, Compute, Device, Pipeline, VulkanError};
use ash::vk;
use std::cell::{Ref, RefCell};
use std::rc::Rc;

pub(crate) struct TonemapPass {
    device: Rc<Device>,
    pub render_target: Rc<RefCell<RenderTarget>>,
    pipeline_handle: Rc<Pipeline<Compute>>,
}

impl TonemapPass {
    pub fn create(
        device: Rc<Device>,
        render_targets: &mut RenderTargets,
        pipeline_builder: &mut PipelineBuilder,
        descriptors: Ref<RendererDescriptors>,
    ) -> Result<Self, AppError> {
        let render_target = render_targets.add(Self::render_target_def())?;

        let pipeline_handle = pipeline_builder.build_compute("tonemap", "tonemap|main", descriptors)?;

        Ok(Self {
            device,
            render_target,
            pipeline_handle,
        })
    }

    fn render_target_def() -> RenderTargetBuilder {
        RenderTargetBuilder::new("tonemap_out")
            .with_storage()
            .with_transfer()
            .with_format(vk::Format::R16G16B16A16_SFLOAT)
    }

    pub fn record(
        &self,
        command_buffer: &CommandBuffer,
        renderer: &VulkanRenderer,
        viewport: (u32, u32),
    ) -> Result<(), VulkanError> {
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

        self.device.begin_label("PostProcessing", command_buffer);

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

        let pc = PushConstBuilder::with_capacity(3 * size_of::<u32>())
            .add_u32(*renderer.descriptors.borrow().storages.get("taa_target").unwrap() as u32)
            .add_u32(*renderer.descriptors.borrow().storages.get("tonemap_out").unwrap() as u32)
            .add_u32(0)
            .build();

        unsafe {
            self.device.inner.cmd_push_constants(
                command_buffer.inner,
                pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &pc,
            );
        }

        let x = viewport.0 / pipeline.reflect_data.workgroup_size.0 + 1;
        let y = viewport.1 / pipeline.reflect_data.workgroup_size.1 + 1;

        command_buffer.dispatch(x, y, 1);

        unsafe {
            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier {
                    src_access_mask: vk::AccessFlags::SHADER_WRITE,
                    dst_access_mask: vk::AccessFlags::MEMORY_READ,
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

    pub fn record_reference(
        &self,
        command_buffer: &CommandBuffer,
        renderer: &VulkanMcPathTracer,
        viewport: (u32, u32),
    ) -> Result<(), VulkanError> {
        self.device.begin_label("PostProcessing", command_buffer);

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

        let pc = PushConstBuilder::with_capacity(3 * size_of::<u32>())
            .add_u32(*renderer.descriptors.borrow().storages.get("acc_out").unwrap() as u32)
            .add_u32(*renderer.descriptors.borrow().storages.get("tonemap_out").unwrap() as u32)
            .add_u32(1)
            .build();

        command_buffer.push_constants(vk::ShaderStageFlags::COMPUTE, pipeline.layout, &pc);

        let x = viewport.0 / pipeline.reflect_data.workgroup_size.0 + 1;
        let y = viewport.1 / pipeline.reflect_data.workgroup_size.1 + 1;

        command_buffer.dispatch(x, y, 1);

        unsafe {
            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier {
                    src_access_mask: vk::AccessFlags::SHADER_WRITE,
                    dst_access_mask: vk::AccessFlags::TRANSFER_READ,
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
