use crate::err::AppError;
use crate::renderer::descriptors::RendererDescriptors;
use crate::renderer::pipeline_builder::PipelineBuilder;
use crate::renderer::render_target::{RenderTarget, RenderTargetBuilder, RenderTargets};
use crate::renderer::{PushConstBuilder, VulkanRenderer};
use crate::vulkan::{CommandBuffer, Compute, Device, Pipeline, VulkanError};
use ash::vk;
use std::cell::{Ref, RefCell};
use std::rc::Rc;

pub(crate) struct DenoisePass {
    pub device: Rc<Device>,
    pub direct_render_target: Rc<RefCell<RenderTarget>>,
    pub direct_render_target_acc: Rc<RefCell<RenderTarget>>,
    pub direct_render_target_history: Rc<RefCell<RenderTarget>>,
    pub indirect_render_target: Rc<RefCell<RenderTarget>>,
    pub indirect_render_target_acc: Rc<RefCell<RenderTarget>>,
    pub indirect_render_target_history: Rc<RefCell<RenderTarget>>,
    temporal_pipeline: Rc<Pipeline<Compute>>,
    spatial_pipeline: Rc<Pipeline<Compute>>,
}

impl DenoisePass {
    pub fn create(
        device: Rc<Device>,
        render_targets: &mut RenderTargets,
        pipeline_builder: &mut PipelineBuilder,
        descriptors: Ref<RendererDescriptors>,
    ) -> Result<Self, AppError> {
        let denoise_pass_target = Self::render_target_def();

        let spatial_pipeline = pipeline_builder.build_compute("atrous", "atrous|main", Ref::clone(&descriptors))?;
        let temporal_pipeline =
            pipeline_builder.build_compute("denoise_temporal", "denoise_temporal|main", descriptors)?;

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
            spatial_pipeline,
            temporal_pipeline,
        })
    }

    fn render_target_def() -> RenderTargetBuilder {
        RenderTargetBuilder::new("denoise_direct_out")
            .with_format(vk::Format::R16G16B16A16_SFLOAT)
            .with_transfer()
            .with_storage()
            .with_sampled()
    }

    pub fn record(
        &self,
        command_buffer: &CommandBuffer,
        renderer: &VulkanRenderer,
        clear: bool,
        viewport: (u32, u32),
    ) -> Result<(), VulkanError> {
        self.device.begin_label("RT Denoise Temporal", command_buffer);

        let pipeline = &self.temporal_pipeline;

        let desc_ref = renderer.descriptors.borrow();

        command_buffer.bind_compute_pipeline(pipeline);
        command_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            [
                desc_ref.global_sets[renderer.current_frame].inner,
                desc_ref.compute_sets[renderer.current_frame].inner,
            ],
        );

        let clear = if clear { 1 } else { 0 };

        let mut pc = PushConstBuilder::with_capacity(7 * size_of::<u32>())
            .add_u32(clear as u32)
            .add_u32(*desc_ref.storages.get("denoise_direct_acc").unwrap() as u32)
            .add_u32(*desc_ref.samplers.get("gbuffer_depth").unwrap() as u32)
            .add_u32(*desc_ref.samplers.get("last_depth").unwrap() as u32)
            .add_u32(*desc_ref.storages.get("rt_direct").unwrap() as u32)
            .add_u32(*desc_ref.samplers.get("denoise_direct_history").unwrap() as u32)
            .add_u32(*desc_ref.samplers.get("gbuffer_normal").unwrap() as u32)
            .build();

        unsafe {
            self.device.inner.cmd_push_constants(
                command_buffer.inner,
                pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &pc,
            );

            let x = (viewport.0 / 16) + 1;
            let y = (viewport.1 / 16) + 1;

            self.device.inner.cmd_dispatch(command_buffer.inner, x, y, 1);

            pc[4..8].copy_from_slice(&(*desc_ref.storages.get("denoise_indirect_acc").unwrap() as u32).to_le_bytes());
            pc[16..20].copy_from_slice(&(*desc_ref.storages.get("rt_indirect").unwrap() as u32).to_le_bytes());
            pc[20..24]
                .copy_from_slice(&(*desc_ref.samplers.get("denoise_indirect_history").unwrap() as u32).to_le_bytes());

            self.device.inner.cmd_push_constants(
                command_buffer.inner,
                pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &pc,
            );

            command_buffer.dispatch(x, y, 1);

            let barriers = [
                self.direct_render_target_acc.borrow().image.inner,
                self.indirect_render_target_acc.borrow().image.inner,
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
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );
        }

        self.device.end_label(command_buffer);

        if renderer.quality.use_spatial_denoise {
            self.device.begin_label("RT Denoise", command_buffer);

            let pipeline = &self.spatial_pipeline;

            command_buffer.bind_compute_pipeline(pipeline);

            let x = (viewport.0 / 16) + 1;
            let y = (viewport.1 / 16) + 1;

            unsafe {
                for level in 0..4_i32 {
                    let level = level.to_le_bytes();

                    let mut pc = [0_u8; 5 * size_of::<f32>()];
                    pc[0..4].copy_from_slice(&level);
                    pc[4..8].copy_from_slice(&(*desc_ref.samplers.get("gbuffer_normal").unwrap() as u32).to_le_bytes());
                    pc[8..12].copy_from_slice(&(*desc_ref.samplers.get("gbuffer_depth").unwrap() as u32).to_le_bytes());
                    pc[12..16]
                        .copy_from_slice(&(*desc_ref.storages.get("denoise_direct_out").unwrap() as u32).to_le_bytes());
                    pc[16..20]
                        .copy_from_slice(&(*desc_ref.storages.get("denoise_direct_acc").unwrap() as u32).to_le_bytes());

                    command_buffer.bind_descriptor_sets(
                        vk::PipelineBindPoint::COMPUTE,
                        pipeline.layout,
                        [
                            desc_ref.global_sets[renderer.current_frame].inner,
                            desc_ref.compute_sets[renderer.current_frame].inner,
                        ],
                    );

                    self.device.inner.cmd_push_constants(
                        command_buffer.inner,
                        pipeline.layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        &pc,
                    );

                    command_buffer.dispatch(x, y, 1);

                    pc[12..16].copy_from_slice(
                        &(*desc_ref.storages.get("denoise_indirect_out").unwrap() as u32).to_le_bytes(),
                    );
                    pc[16..20].copy_from_slice(
                        &(*desc_ref.storages.get("denoise_indirect_acc").unwrap() as u32).to_le_bytes(),
                    );

                    self.device.inner.cmd_push_constants(
                        command_buffer.inner,
                        pipeline.layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        &pc,
                    );

                    command_buffer.dispatch(x, y, 1);

                    let barriers = [
                        renderer
                            .render_targets
                            .get_ref("denoise_direct_out")
                            .unwrap()
                            .image
                            .inner,
                        renderer
                            .render_targets
                            .get_ref("denoise_indirect_out")
                            .unwrap()
                            .image
                            .inner,
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
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &barriers,
                    );
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
                    self.direct_render_target_acc.borrow().image.inner,
                    vk::ImageLayout::GENERAL,
                    self.direct_render_target.borrow().image.inner,
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
                    self.indirect_render_target.borrow().image.inner,
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
