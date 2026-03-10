use crate::err::AppError;
use crate::renderer::ResourceSubsystem;
use crate::renderer::descriptors::{DescriptorLayouts, RendererDescriptors};
use crate::renderer::mesh_collector::DrawData;
use crate::renderer::pipeline_builder::PipelineBuilder;
use crate::renderer::render_target::{RenderTarget, RenderTargetBuilder, RenderTargets};
use crate::vulkan::{CommandBuffer, Device, Graphics, Pipeline, VulkanError};
use ash::vk;
use std::cell::{Ref, RefCell};
use std::rc::Rc;

pub(crate) struct GBufferPass {
    pub device: Rc<Device>,
    pub render_target_color: Rc<RefCell<RenderTarget>>,
    pub render_target_normal: Rc<RefCell<RenderTarget>>,
    pub render_target_depth: Rc<RefCell<RenderTarget>>,
    pub pipeline_handle: Rc<Pipeline<Graphics>>,
}

impl GBufferPass {
    pub const TARGET_FORMATS: [vk::Format; 3] = [
        vk::Format::A2B10G10R10_UNORM_PACK32,
        vk::Format::A2B10G10R10_UNORM_PACK32,
        vk::Format::D32_SFLOAT,
    ];
    pub const PIPELINE_TARGET_FORMATS: [vk::Format; 2] = [
        vk::Format::A2B10G10R10_UNORM_PACK32,
        vk::Format::A2B10G10R10_UNORM_PACK32,
    ];

    pub fn create(
        device: Rc<Device>,
        render_targets: &mut RenderTargets,
        pipeline_builder: &mut PipelineBuilder,
        descriptor_layouts: &DescriptorLayouts,
    ) -> Result<Self, AppError> {
        let pipeline_handle = pipeline_builder.build_graphics(
            "deferred",
            "deferred|vertMain",
            "deferred|fragMain",
            descriptor_layouts,
            &Self::PIPELINE_TARGET_FORMATS,
            true,
        )?;

        let [a, b, c] = Self::render_target_defs();
        let render_target_color = render_targets.add(a)?;
        let render_target_normal = render_targets.add(b)?;
        let render_target_depth = render_targets.add(c)?;

        Ok(Self {
            device,
            render_target_color,
            render_target_normal,
            render_target_depth,
            pipeline_handle,
        })
    }

    pub fn render_target_defs() -> [RenderTargetBuilder; 3] {
        [
            RenderTargetBuilder::new("gbuffer_color")
                .with_color_attachment()
                .with_sampled()
                .with_storage()
                .with_transfer()
                .with_format(Self::TARGET_FORMATS[0]),
            RenderTargetBuilder::new("gbuffer_normal")
                .with_color_attachment()
                .with_sampled()
                .with_storage()
                .with_transfer()
                .with_format(Self::TARGET_FORMATS[1]),
            RenderTargetBuilder::new_depth("gbuffer_depth")
                .with_sampled()
                .with_storage()
                .with_transfer()
                .with_format(Self::TARGET_FORMATS[2]),
        ]
    }

    pub fn record(
        &self,
        command_buffer: &CommandBuffer,
        descriptors: Ref<RendererDescriptors>,
        resource_subsystem: &ResourceSubsystem,
        draw_data: &Vec<DrawData>,
        viewport: (u32, u32),
    ) -> Result<(), VulkanError> {
        self.device.begin_label("GBuffer", command_buffer);

        let color_rt = self.render_target_color.borrow();
        let normal_rt = self.render_target_normal.borrow();
        let depth_rt = self.render_target_depth.borrow();

        let attachments = [
            vk::RenderingAttachmentInfo {
                image_view: color_rt.view.inner,
                image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                load_op: vk::AttachmentLoadOp::DONT_CARE,
                store_op: vk::AttachmentStoreOp::STORE,
                ..Default::default()
            },
            vk::RenderingAttachmentInfo {
                image_view: normal_rt.view.inner,
                image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                load_op: vk::AttachmentLoadOp::DONT_CARE,
                store_op: vk::AttachmentStoreOp::STORE,
                ..Default::default()
            },
        ];

        let depth_attachment = vk::RenderingAttachmentInfo {
            image_view: depth_rt.view.inner,
            image_layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            load_op: vk::AttachmentLoadOp::DONT_CARE,
            store_op: vk::AttachmentStoreOp::STORE,
            ..Default::default()
        };

        let extent = vk::Extent2D {
            width: viewport.0,
            height: viewport.1,
        };

        let rendering_info = vk::RenderingInfo {
            render_area: vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent,
            },
            layer_count: 1,
            color_attachment_count: attachments.len() as u32,
            p_color_attachments: attachments.as_ptr(),
            p_depth_attachment: std::ptr::addr_of!(depth_attachment),
            ..Default::default()
        };

        let image_color_res = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let image_depth_res = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            ..image_color_res
        };

        unsafe {
            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::ALL_GRAPHICS,
                vk::PipelineStageFlags::ALL_GRAPHICS | vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[
                    vk::ImageMemoryBarrier {
                        src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ,
                        dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        old_layout: vk::ImageLayout::UNDEFINED,
                        new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        image: color_rt.image.inner,
                        subresource_range: image_color_res,
                        ..Default::default()
                    },
                    vk::ImageMemoryBarrier {
                        src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ,
                        dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        old_layout: vk::ImageLayout::UNDEFINED,
                        new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        image: normal_rt.image.inner,
                        subresource_range: image_color_res,
                        ..Default::default()
                    },
                    vk::ImageMemoryBarrier {
                        src_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
                        dst_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        old_layout: vk::ImageLayout::UNDEFINED,
                        new_layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                        image: depth_rt.image.inner,
                        subresource_range: image_depth_res,
                        ..Default::default()
                    },
                ],
            );
        }

        command_buffer.begin_rendering(&rendering_info);

        let viewport = vk::Viewport {
            width: viewport.0 as f32,
            height: viewport.1 as f32,
            max_depth: 1.0,
            ..Default::default()
        };

        let pipeline = &self.pipeline_handle;

        command_buffer.bind_graphics_pipeline(pipeline);
        command_buffer.set_viewport(viewport);
        command_buffer.set_scissor(vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent,
        });

        unsafe {
            self.device
                .inner
                .cmd_set_cull_mode(command_buffer.inner, vk::CullModeFlags::NONE);

            let rect = vk::ClearRect {
                layer_count: 1,
                base_array_layer: 0,
                rect: vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent,
                },
            };

            self.device.inner.cmd_clear_attachments(
                command_buffer.inner,
                &[
                    vk::ClearAttachment {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        color_attachment: 0,
                        clear_value: Default::default(),
                    },
                    vk::ClearAttachment {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        color_attachment: 1,
                        clear_value: Default::default(),
                    },
                    vk::ClearAttachment {
                        aspect_mask: vk::ImageAspectFlags::DEPTH,
                        color_attachment: 0,
                        clear_value: Default::default(),
                    },
                ],
                &[rect],
            );
        }

        command_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.layout,
            [descriptors.global_set.inner],
        );

        for draw in draw_data {
            let mesh_data = &resource_subsystem.meshes[&draw.mesh_id];

            command_buffer.bind_vertex_buffers(&[&mesh_data.buf], &[0]);

            unsafe {
                // Call raw function to skip the need for recasting a single push constant in a loop
                (self.device.inner.fp_v1_0().cmd_push_constants)(
                    command_buffer.inner,
                    pipeline.layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    size_of::<u32>() as u32,
                    &draw.offset as *const u32 as *const _,
                );

                self.device.inner.cmd_bind_index_buffer(
                    command_buffer.inner,
                    mesh_data.buf.inner.inner,
                    mesh_data.indices_offset,
                    vk::IndexType::UINT32,
                );

                self.device.inner.cmd_draw_indexed(
                    command_buffer.inner,
                    mesh_data.index_count as u32,
                    draw.count,
                    0,
                    0,
                    0,
                );
            }
        }

        let viewport = vk::Viewport {
            width: extent.width as f32,
            height: extent.height as f32,
            max_depth: 1.0,
            ..Default::default()
        };
        command_buffer.set_viewport(viewport);

        command_buffer.end_rendering();

        unsafe {
            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[
                    vk::ImageMemoryBarrier {
                        src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        dst_access_mask: vk::AccessFlags::SHADER_READ,
                        old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        new_layout: vk::ImageLayout::GENERAL,
                        image: color_rt.image.inner,
                        subresource_range: image_color_res,
                        ..Default::default()
                    },
                    vk::ImageMemoryBarrier {
                        src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        dst_access_mask: vk::AccessFlags::SHADER_READ,
                        old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        new_layout: vk::ImageLayout::GENERAL,
                        image: normal_rt.image.inner,
                        subresource_range: image_color_res,
                        ..Default::default()
                    },
                    vk::ImageMemoryBarrier {
                        src_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        dst_access_mask: vk::AccessFlags::SHADER_READ,
                        old_layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                        new_layout: vk::ImageLayout::GENERAL,
                        image: depth_rt.image.inner,
                        subresource_range: image_depth_res,
                        ..Default::default()
                    },
                ],
            );
        }

        self.device.end_label(command_buffer);
        Ok(())
    }
}
