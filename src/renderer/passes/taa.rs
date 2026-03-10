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

pub(crate) struct TaaPass {
    pub device: Rc<Device>,
    pub render_target: Rc<RefCell<RenderTarget>>,
    pub render_target_history: Rc<RefCell<RenderTarget>>,
    pipeline_handle: Rc<Pipeline<Compute>>,
}

impl TaaPass {
    pub const TARGET_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

    pub fn create(
        device: Rc<Device>,
        render_targets: &mut RenderTargets,
        pipeline_builder: &mut PipelineBuilder,
        descriptor_layouts: &DescriptorLayouts,
    ) -> Result<Self, AppError> {
        let [a, b] = Self::render_target_defs();
        let render_target = render_targets.add(a)?;
        let render_target_history = render_targets.add(b)?;

        let pipeline_handle = pipeline_builder.build_compute("taa", "taa|main", descriptor_layouts)?;

        Ok(Self {
            device,
            render_target,
            render_target_history,
            pipeline_handle,
        })
    }

    fn render_target_defs() -> [RenderTargetBuilder; 2] {
        [
            RenderTargetBuilder::new("taa_target")
                .with_transfer()
                .with_storage()
                .with_color_attachment()
                .with_sampled()
                .with_format(Self::TARGET_FORMAT),
            RenderTargetBuilder::new("taa_history_target")
                .with_transfer()
                .with_storage()
                .with_color_attachment()
                .with_sampled()
                .with_format(Self::TARGET_FORMAT),
        ]
    }

    pub fn record(
        &self,
        command_buffer: &CommandBuffer,
        descriptors: &RendererDescriptors,
        inputs: TaaInputs,
        viewport: (u32, u32),
    ) -> Result<(), VulkanError> {
        self.device.begin_label("TAA Resolve", command_buffer);

        let pipeline = &self.pipeline_handle;

        command_buffer.bind_compute_pipeline(pipeline);

        command_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            [descriptors.global_set.inner, descriptors.compute_set.inner],
        );

        let clear = if inputs.clear { 1 } else { 0 };

        let pc = PushConstBuilder::new()
            .add_u32(clear as u32)
            .add_u32(self.render_target.borrow().storage_index.unwrap())
            .add_u32(inputs.src.sampler_index.unwrap())
            .add_u32(self.render_target_history.borrow().sampler_index.unwrap())
            .add_u32(inputs.depth.sampler_index.unwrap())
            .add_u32(inputs.last_depth.sampler_index.unwrap())
            .build();

        command_buffer.push_constants(vk::ShaderStageFlags::COMPUTE, pipeline.layout, &pc);

        let x = math::workgroup_saturate(viewport.0, pipeline.reflect_data.workgroup_size.0);
        let y = math::workgroup_saturate(viewport.1, pipeline.reflect_data.workgroup_size.1);

        command_buffer.dispatch(x, y, 1);

        let extent_3d = vk::Extent3D {
            width: viewport.0,
            height: viewport.1,
            depth: 1,
        };

        unsafe {
            let barriers = [self.render_target.borrow().image.inner].map(|image| vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::SHADER_WRITE,
                dst_access_mask: vk::AccessFlags::MEMORY_READ,
                old_layout: vk::ImageLayout::GENERAL,
                new_layout: vk::ImageLayout::GENERAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            });

            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );

            let image_color_res = vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            };
            let image_depth_res = vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                ..image_color_res
            };

            self.device.inner.cmd_copy_image(
                command_buffer.inner,
                inputs.depth.image.inner,
                vk::ImageLayout::GENERAL,
                inputs.last_depth.image.inner,
                vk::ImageLayout::GENERAL,
                &[vk::ImageCopy {
                    extent: extent_3d,
                    dst_subresource: image_depth_res,
                    src_subresource: image_depth_res,
                    ..Default::default()
                }],
            );

            self.device.inner.cmd_copy_image(
                command_buffer.inner,
                self.render_target.borrow().image.inner,
                vk::ImageLayout::GENERAL,
                self.render_target_history.borrow().image.inner,
                vk::ImageLayout::GENERAL,
                &[vk::ImageCopy {
                    extent: extent_3d,
                    dst_subresource: image_color_res,
                    src_subresource: image_color_res,
                    ..Default::default()
                }],
            );
        }

        self.device.end_label(command_buffer);

        Ok(())
    }
}

pub struct TaaInputs<'a> {
    pub depth: &'a RenderTarget,
    pub last_depth: &'a RenderTarget,
    pub src: &'a RenderTarget,
    pub clear: bool,
}
