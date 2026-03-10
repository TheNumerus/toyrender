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

pub(crate) struct DenoisePass {
    pub device: Rc<Device>,
    pub direct_render_target: Rc<RefCell<RenderTarget>>,
    pub direct_render_target_acc: Rc<RefCell<RenderTarget>>,
    pub direct_render_target_history: Rc<RefCell<RenderTarget>>,
    pub indirect_render_target: Rc<RefCell<RenderTarget>>,
    pub indirect_render_target_acc: Rc<RefCell<RenderTarget>>,
    pub indirect_render_target_history: Rc<RefCell<RenderTarget>>,
    pub moments_direct_render_target: Rc<RefCell<RenderTarget>>,
    pub moments_indirect_render_target: Rc<RefCell<RenderTarget>>,
    temporal_pipeline: Rc<Pipeline<Compute>>,
    spatial_pipeline: Rc<Pipeline<Compute>>,
    variance_estimate_pipeline: Rc<Pipeline<Compute>>,
}

impl DenoisePass {
    pub fn create(
        device: Rc<Device>,
        render_targets: &mut RenderTargets,
        pipeline_builder: &mut PipelineBuilder,
        descriptor_layouts: &DescriptorLayouts,
    ) -> Result<Self, AppError> {
        let denoise_pass_target = Self::render_target_def();

        let spatial_pipeline = pipeline_builder.build_compute("atrous", "atrous|main", descriptor_layouts)?;
        let temporal_pipeline =
            pipeline_builder.build_compute("denoise_temporal", "denoise_temporal|main", descriptor_layouts)?;

        let variance_estimate_pipeline =
            pipeline_builder.build_compute("variance", "variance|main", descriptor_layouts)?;

        Ok(Self {
            device,
            direct_render_target: render_targets.add(denoise_pass_target.duplicate("denoise_direct_out"))?,
            direct_render_target_acc: render_targets.add(denoise_pass_target.duplicate("denoise_direct_acc"))?,
            direct_render_target_history: render_targets
                .add(denoise_pass_target.duplicate("denoise_direct_history"))?,
            indirect_render_target: render_targets.add(denoise_pass_target.duplicate("denoise_indirect_out"))?,
            indirect_render_target_acc: render_targets.add(denoise_pass_target.duplicate("denoise_indirect_acc"))?,
            indirect_render_target_history: render_targets
                .add(denoise_pass_target.duplicate("denoise_indirect_history"))?,
            moments_direct_render_target: render_targets.add(Self::moment_render_target_def())?,
            moments_indirect_render_target: render_targets
                .add(Self::moment_render_target_def().duplicate("denoise_indirect_moments"))?,
            spatial_pipeline,
            temporal_pipeline,
            variance_estimate_pipeline,
        })
    }

    fn render_target_def() -> RenderTargetBuilder {
        RenderTargetBuilder::new("denoise_direct_out")
            .with_format(vk::Format::R16G16B16A16_SFLOAT)
            .with_transfer()
            .with_storage()
            .with_sampled()
    }

    fn moment_render_target_def() -> RenderTargetBuilder {
        RenderTargetBuilder::new("denoise_direct_moments")
            .with_format(vk::Format::R16G16B16A16_SFLOAT)
            .with_transfer()
            .with_storage()
    }

    pub fn record(
        &self,
        command_buffer: &CommandBuffer,
        descriptors: &RendererDescriptors,
        inputs: DenoiseInputs,
        viewport: (u32, u32),
    ) -> Result<(), VulkanError> {
        self.device.begin_label("RT Denoise Temporal", command_buffer);

        let pipeline = &self.temporal_pipeline;

        command_buffer.bind_compute_pipeline(pipeline);
        command_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            [descriptors.global_set.inner, descriptors.compute_set.inner],
        );

        let clear = if inputs.clear { 1 } else { 0 };

        let pc = PushConstBuilder::with_capacity(7 * size_of::<u32>())
            .add_u32(clear as u32)
            .add_u32(self.direct_render_target.borrow().storage_index.unwrap())
            .add_u32(inputs.depth.sampler_index.unwrap())
            .add_u32(inputs.last_depth.sampler_index.unwrap())
            .add_u32(inputs.rt_direct.storage_index.unwrap())
            .add_u32(self.direct_render_target_history.borrow().sampler_index.unwrap())
            .add_u32(inputs.normal.sampler_index.unwrap())
            .add_u32(self.moments_direct_render_target.borrow().storage_index.unwrap());

        command_buffer.push_constants(vk::ShaderStageFlags::COMPUTE, pipeline.layout, pc.as_ref());

        let x = math::workgroup_saturate(viewport.0, pipeline.reflect_data.workgroup_size.0);
        let y = math::workgroup_saturate(viewport.1, pipeline.reflect_data.workgroup_size.1);

        command_buffer.dispatch(x, y, 1);

        let pc = pc
            .update_u32(self.indirect_render_target.borrow().storage_index.unwrap(), 4)
            .update_u32(inputs.rt_indirect.storage_index.unwrap(), 16)
            .update_u32(self.indirect_render_target_history.borrow().sampler_index.unwrap(), 20)
            .update_u32(self.moments_indirect_render_target.borrow().storage_index.unwrap(), 28);

        command_buffer.push_constants(vk::ShaderStageFlags::COMPUTE, pipeline.layout, pc.as_ref());

        command_buffer.dispatch(x, y, 1);

        unsafe {
            let barriers = [
                self.direct_render_target.borrow().image.inner,
                self.indirect_render_target.borrow().image.inner,
                self.moments_direct_render_target.borrow().image.inner,
                self.moments_indirect_render_target.borrow().image.inner,
            ]
            .map(|image| vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::SHADER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                old_layout: vk::ImageLayout::GENERAL,
                new_layout: vk::ImageLayout::GENERAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image,
                subresource_range: crate::vulkan::Image::single_color_layer_range(),
                ..Default::default()
            });

            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );
        }

        let pipeline = &self.variance_estimate_pipeline;

        command_buffer.bind_compute_pipeline(pipeline);
        command_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            [descriptors.global_set.inner, descriptors.compute_set.inner],
        );

        let pc = PushConstBuilder::with_capacity(5 * size_of::<u32>())
            .add_u32(self.direct_render_target.borrow().storage_index.unwrap())
            .add_u32(self.direct_render_target_acc.borrow().storage_index.unwrap())
            .add_u32(inputs.depth.sampler_index.unwrap())
            .add_u32(inputs.normal.sampler_index.unwrap())
            .add_u32(self.moments_direct_render_target.borrow().storage_index.unwrap());

        command_buffer.push_constants(vk::ShaderStageFlags::COMPUTE, pipeline.layout, &pc.clone().build());

        let x = math::workgroup_saturate(viewport.0, pipeline.reflect_data.workgroup_size.0);
        let y = math::workgroup_saturate(viewport.1, pipeline.reflect_data.workgroup_size.1);

        command_buffer.dispatch(x, y, 1);

        let pc = pc
            .update_u32(self.indirect_render_target.borrow().storage_index.unwrap(), 0)
            .update_u32(self.indirect_render_target_acc.borrow().storage_index.unwrap(), 4)
            .update_u32(self.moments_indirect_render_target.borrow().storage_index.unwrap(), 16);

        command_buffer.push_constants(vk::ShaderStageFlags::COMPUTE, pipeline.layout, pc.as_ref());

        command_buffer.dispatch(x, y, 1);

        unsafe {
            let barriers = [
                self.direct_render_target_acc.borrow().image.inner,
                self.indirect_render_target_acc.borrow().image.inner,
            ]
            .map(|image| vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::SHADER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                old_layout: vk::ImageLayout::GENERAL,
                new_layout: vk::ImageLayout::GENERAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image,
                subresource_range: crate::vulkan::Image::single_color_layer_range(),
                ..Default::default()
            });

            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );
        }

        self.device.end_label(command_buffer);

        if inputs.use_spatial_denoise {
            self.device.begin_label("RT Denoise", command_buffer);

            let pipeline = &self.spatial_pipeline;

            command_buffer.bind_compute_pipeline(pipeline);

            let x = math::workgroup_saturate(viewport.0, pipeline.reflect_data.workgroup_size.0);
            let y = math::workgroup_saturate(viewport.1, pipeline.reflect_data.workgroup_size.1);

            command_buffer.bind_descriptor_sets(
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                [descriptors.global_set.inner, descriptors.compute_set.inner],
            );

            for level in 0..4_u32 {
                let mut src_idx = self.direct_render_target.borrow().storage_index.unwrap();
                let mut out_idx = self.direct_render_target_acc.borrow().storage_index.unwrap();

                if level % 2 == 0 {
                    (out_idx, src_idx) = (src_idx, out_idx);
                };

                let pc = PushConstBuilder::with_capacity(5 * size_of::<u32>())
                    .add_u32(level)
                    .add_u32(inputs.normal.sampler_index.unwrap())
                    .add_u32(inputs.depth.sampler_index.unwrap())
                    .add_u32(src_idx)
                    .add_u32(out_idx);

                command_buffer.push_constants(vk::ShaderStageFlags::COMPUTE, pipeline.layout, pc.as_ref());

                command_buffer.dispatch(x, y, 1);

                let mut src_idx = self.indirect_render_target.borrow().storage_index.unwrap();
                let mut out_idx = self.indirect_render_target_acc.borrow().storage_index.unwrap();

                if level % 2 == 0 {
                    (out_idx, src_idx) = (src_idx, out_idx);
                };

                let pc = pc.update_u32(src_idx, 12).update_u32(out_idx, 16);

                command_buffer.push_constants(vk::ShaderStageFlags::COMPUTE, pipeline.layout, pc.as_ref());

                command_buffer.dispatch(x, y, 1);

                let barriers = if level % 2 == 0 {
                    [
                        self.direct_render_target_acc.borrow().image.inner,
                        self.indirect_render_target_acc.borrow().image.inner,
                    ]
                } else {
                    [
                        self.direct_render_target.borrow().image.inner,
                        self.indirect_render_target.borrow().image.inner,
                    ]
                };

                unsafe {
                    let barriers = barriers.map(|image| vk::ImageMemoryBarrier {
                        src_access_mask: vk::AccessFlags::MEMORY_WRITE,
                        dst_access_mask: vk::AccessFlags::MEMORY_READ,
                        old_layout: vk::ImageLayout::GENERAL,
                        new_layout: vk::ImageLayout::GENERAL,
                        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        image,
                        subresource_range: crate::vulkan::Image::single_color_layer_range(),
                        ..Default::default()
                    });

                    self.device.inner.cmd_pipeline_barrier(
                        command_buffer.inner,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &barriers,
                    );
                }

                // after first level copy denoised result to history buffer
                if level == 0 {
                    unsafe {
                        let extent_3d = vk::Extent3D {
                            width: viewport.0,
                            height: viewport.1,
                            depth: 1,
                        };

                        let image_color_res = vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        };

                        self.device.inner.cmd_copy_image(
                            command_buffer.inner,
                            self.direct_render_target_acc.borrow().image.inner,
                            vk::ImageLayout::GENERAL,
                            self.direct_render_target_history.borrow().image.inner,
                            vk::ImageLayout::GENERAL,
                            &[vk::ImageCopy {
                                extent: extent_3d,
                                dst_subresource: image_color_res,
                                src_subresource: image_color_res,
                                ..Default::default()
                            }],
                        );
                        self.device.inner.cmd_copy_image(
                            command_buffer.inner,
                            self.indirect_render_target_acc.borrow().image.inner,
                            vk::ImageLayout::GENERAL,
                            self.indirect_render_target_history.borrow().image.inner,
                            vk::ImageLayout::GENERAL,
                            &[vk::ImageCopy {
                                extent: extent_3d,
                                dst_subresource: image_color_res,
                                src_subresource: image_color_res,
                                ..Default::default()
                            }],
                        );

                        let barriers = [
                            self.direct_render_target_history.borrow().image.inner,
                            self.indirect_render_target_history.borrow().image.inner,
                        ]
                        .map(|image| vk::ImageMemoryBarrier {
                            src_access_mask: vk::AccessFlags::MEMORY_WRITE,
                            dst_access_mask: vk::AccessFlags::MEMORY_READ,
                            old_layout: vk::ImageLayout::GENERAL,
                            new_layout: vk::ImageLayout::GENERAL,
                            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                            image,
                            subresource_range: crate::vulkan::Image::single_color_layer_range(),
                            ..Default::default()
                        });

                        self.device.inner.cmd_pipeline_barrier(
                            command_buffer.inner,
                            vk::PipelineStageFlags::TRANSFER,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::DependencyFlags::empty(),
                            &[],
                            &[],
                            &barriers,
                        );
                    }
                }
            }

            self.device.end_label(command_buffer);
        } else {
            let extent_3d = vk::Extent3D {
                width: viewport.0,
                height: viewport.1,
                depth: 1,
            };

            let image_color_res = vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            };

            unsafe {
                self.device.inner.cmd_copy_image(
                    command_buffer.inner,
                    self.direct_render_target.borrow().image.inner,
                    vk::ImageLayout::GENERAL,
                    self.direct_render_target_history.borrow().image.inner,
                    vk::ImageLayout::GENERAL,
                    &[vk::ImageCopy {
                        extent: extent_3d,
                        dst_subresource: image_color_res,
                        src_subresource: image_color_res,
                        ..Default::default()
                    }],
                );
                self.device.inner.cmd_copy_image(
                    command_buffer.inner,
                    self.indirect_render_target.borrow().image.inner,
                    vk::ImageLayout::GENERAL,
                    self.indirect_render_target_history.borrow().image.inner,
                    vk::ImageLayout::GENERAL,
                    &[vk::ImageCopy {
                        extent: extent_3d,
                        dst_subresource: image_color_res,
                        src_subresource: image_color_res,
                        ..Default::default()
                    }],
                );
            }
        }

        Ok(())
    }
}

pub struct DenoiseInputs<'a> {
    pub depth: &'a RenderTarget,
    pub last_depth: &'a RenderTarget,
    pub normal: &'a RenderTarget,
    pub rt_direct: &'a RenderTarget,
    pub rt_indirect: &'a RenderTarget,
    pub clear: bool,
    pub use_spatial_denoise: bool,
}
