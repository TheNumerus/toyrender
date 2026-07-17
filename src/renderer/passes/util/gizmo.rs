use crate::err::AppError;
use crate::renderer::descriptors::{DescriptorLayouts, RendererDescriptors};
use crate::renderer::pipeline_builder::PipelineBuilder;
use crate::renderer::render_target::RenderTarget;
use crate::renderer::{PushConstBuilder, ResourceSubsystem};
use crate::vulkan::{CommandBuffer, Device, Graphics, Pipeline, VulkanError};
use ash::vk;
use nalgebra_glm::Mat4;
use std::rc::Rc;

pub(crate) struct GizmoPass {
    device: Rc<Device>,
    pipeline: Rc<Pipeline<Graphics>>,
    pipeline_arrow: Rc<Pipeline<Graphics>>,
    sun_gizmo_id: u64,
    arrow_gizmo_id: u64,
}

impl GizmoPass {
    pub fn create(
        device: Rc<Device>,
        pipeline_builder: &mut PipelineBuilder,
        descriptor_layouts: &DescriptorLayouts,
        sun_gizmo_id: u64,
        arrow_gizmo_id: u64,
    ) -> Result<Self, AppError> {
        let pipeline = pipeline_builder.build_graphics(
            "gizmo",
            "gizmo|vert",
            "gizmo|frag",
            descriptor_layouts,
            &[vk::Format::R16G16B16A16_SFLOAT],
            false,
        )?;

        let pipeline_arrow = pipeline_builder.build_graphics(
            "gizmo",
            "gizmo|vertArrow",
            "gizmo|frag",
            descriptor_layouts,
            &[vk::Format::R16G16B16A16_SFLOAT],
            false,
        )?;

        Ok(Self {
            device,
            pipeline,
            pipeline_arrow,
            sun_gizmo_id,
            arrow_gizmo_id,
        })
    }

    pub fn record<'a>(
        &self,
        command_buffer: &'a CommandBuffer,
        descriptors: &RendererDescriptors,
        resource_subsystem: &ResourceSubsystem,
        inputs: GizmoInputs<'a>,
    ) -> Result<(), VulkanError> {
        self.device.begin_label("Gizmos", command_buffer);

        if inputs.draw_sky_gizmo {
            let attachments = [vk::RenderingAttachmentInfo {
                image_view: inputs.target.view.inner,
                image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                load_op: vk::AttachmentLoadOp::DONT_CARE,
                store_op: vk::AttachmentStoreOp::STORE,
                ..Default::default()
            }];

            let extent = vk::Extent2D {
                width: inputs.viewport.0,
                height: inputs.viewport.1,
            };

            let rendering_info = vk::RenderingInfo {
                render_area: vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent,
                },
                layer_count: 1,
                color_attachment_count: attachments.len() as u32,
                p_color_attachments: attachments.as_ptr(),
                p_depth_attachment: std::ptr::null(),
                ..Default::default()
            };

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
                    vk::PipelineStageFlags::ALL_GRAPHICS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier {
                        src_access_mask: vk::AccessFlags::SHADER_WRITE,
                        dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        old_layout: vk::ImageLayout::GENERAL,
                        new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        image: inputs.target.image.inner,
                        subresource_range: image_color_res,
                        ..Default::default()
                    }],
                );
            }

            command_buffer.begin_rendering(&rendering_info);

            let viewport = vk::Viewport {
                width: inputs.viewport.0 as f32,
                height: inputs.viewport.1 as f32,
                max_depth: 1.0,
                ..Default::default()
            };

            command_buffer.bind_graphics_pipeline(&self.pipeline);
            command_buffer.set_viewport(viewport);
            command_buffer.set_scissor(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent,
            });

            command_buffer.bind_descriptor_sets(
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.layout,
                [descriptors.global_set.inner],
            );

            let mesh_data = &resource_subsystem.meshes[&self.sun_gizmo_id];

            command_buffer.bind_vertex_buffers(&[&mesh_data.buf], &[0]);

            unsafe {
                self.device
                    .inner
                    .cmd_set_cull_mode(command_buffer.inner, vk::CullModeFlags::NONE);

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

            command_buffer.bind_graphics_pipeline(&self.pipeline_arrow);

            let mesh_data = &resource_subsystem.meshes[&self.arrow_gizmo_id];

            command_buffer.bind_vertex_buffers(&[&mesh_data.buf], &[0]);

            unsafe {
                let pc = PushConstBuilder::new().add_mat(inputs.arrow_rot).build();

                self.device.inner.cmd_push_constants(
                    command_buffer.inner,
                    self.pipeline.layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    &pc,
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
                    vk::PipelineStageFlags::ALL_GRAPHICS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier {
                        src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                        dst_access_mask: vk::AccessFlags::SHADER_READ,
                        old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        new_layout: vk::ImageLayout::GENERAL,
                        image: inputs.target.image.inner,
                        subresource_range: image_color_res,
                        ..Default::default()
                    }],
                );
            }
        }

        self.device.end_label(command_buffer);

        Ok(())
    }
}

pub struct GizmoInputs<'a> {
    pub target: &'a RenderTarget,
    pub draw_sky_gizmo: bool,
    pub viewport: (u32, u32),
    pub arrow_rot: Mat4,
}
