use crate::renderer::VulkanRenderer;
use crate::renderer::descriptors::DescLayout;
use crate::renderer::render_target::{RenderTarget, RenderTargetBuilder};
use crate::scene::Scene;
use crate::vulkan::{CommandBuffer, Device, VulkanError};
use ash::vk;
use std::cell::RefCell;
use std::rc::Rc;

pub(crate) struct GBufferPass {
    pub device: Rc<Device>,
    pub render_target_color: Rc<RefCell<RenderTarget>>,
    pub render_target_normal: Rc<RefCell<RenderTarget>>,
    pub render_target_depth: Rc<RefCell<RenderTarget>>,
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
    pub const DESC_LAYOUTS: [DescLayout; 2] = [DescLayout::Global, DescLayout::Image];

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
        renderer: &VulkanRenderer,
        scene: &Scene,
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

        let rendering_info = vk::RenderingInfo {
            render_area: vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: renderer.swap_chain.extent,
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
                vk::PipelineStageFlags::ALL_GRAPHICS,
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
            width: renderer.swap_chain.extent.width as f32,
            height: renderer.swap_chain.extent.height as f32,
            max_depth: 1.0,
            ..Default::default()
        };

        let pipeline = renderer.pipeline_builder.get_graphics("deferred").unwrap();

        command_buffer.bind_graphics_pipeline(pipeline);
        command_buffer.set_viewport(viewport);
        command_buffer.set_scissor(vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: renderer.swap_chain.extent,
        });

        unsafe {
            let rect = vk::ClearRect {
                layer_count: 1,
                base_array_layer: 0,
                rect: vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent: renderer.swap_chain.extent,
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

            self.device.inner.cmd_bind_descriptor_sets(
                command_buffer.inner,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.layout,
                0,
                &[renderer.descriptors.borrow().global_sets[renderer.current_frame].inner],
                &[],
            );
        }

        for instance in &scene.meshes {
            let mesh = &instance.resource;
            let mesh_data = &renderer.meshes[&mesh.id];

            command_buffer.bind_vertex_buffers(&[&mesh_data.buf], &[0]);

            unsafe {
                let constants = std::slice::from_raw_parts(instance.transform.as_ptr() as *const u8, 64);

                self.device.inner.cmd_push_constants(
                    command_buffer.inner,
                    pipeline.layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    constants,
                );

                self.device.inner.cmd_bind_index_buffer(
                    command_buffer.inner,
                    mesh_data.buf.inner.inner,
                    mesh_data.indices_offset,
                    vk::IndexType::UINT32,
                );

                self.device
                    .inner
                    .cmd_draw_indexed(command_buffer.inner, mesh_data.index_count as u32, 1, 0, 0, 0);
            }
        }

        command_buffer.end_rendering();

        unsafe {
            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::ALL_GRAPHICS,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[
                    vk::ImageMemoryBarrier {
                        src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        dst_access_mask: vk::AccessFlags::SHADER_READ,
                        old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        image: color_rt.image.inner,
                        subresource_range: image_color_res,
                        ..Default::default()
                    },
                    vk::ImageMemoryBarrier {
                        src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        dst_access_mask: vk::AccessFlags::SHADER_READ,
                        old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        image: normal_rt.image.inner,
                        subresource_range: image_color_res,
                        ..Default::default()
                    },
                    vk::ImageMemoryBarrier {
                        src_access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        dst_access_mask: vk::AccessFlags::SHADER_READ,
                        old_layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
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
