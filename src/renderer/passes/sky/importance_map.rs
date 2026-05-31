use crate::err::AppError;
use crate::math;
use crate::renderer::descriptors::{DescriptorLayouts, RendererDescriptors};
use crate::renderer::pipeline_builder::PipelineBuilder;
use crate::renderer::push_const::PushConstBuilder;
use crate::renderer::render_target::{RenderTarget, RenderTargetBuilder, RenderTargetSize, RenderTargets};
use crate::vulkan::{CommandBuffer, Compute, Device, Pipeline, VulkanError};
use ash::vk;
use std::cell::RefCell;
use std::rc::Rc;

/**
This pass creates a sky importance map in a way heavily inspired by PBR Book (https://pbr-book.org/4ed/Light_Sources/Infinite_Area_Lights).

First, the sky is read from a loaded equirectangular texture and mapped to 1024x1024 octahedral format.
This is Blender's default size, and it should work fine with any sun in the image.
Code for octa mapping is taken from the Ray Tracing Gems II book (Chapter 3.8).

This is done because it minimizes streching. The pass also converts color to luminance info, because colors aren't needed.

The octa map is then in two passes converted into rolling sums. First per line, then sum of lines. This should create proto-CDF in another 1025x1024 texture (extra line for sums).
Annother pass then does normalization. This could be improved by using hierarchical conversion, as described in Ray Tracing Gems I book (Chapter 16.4.2.3).

Final pass does the inversion so the CDF can be sampled directly.
 */
pub(crate) struct ImportanceMapPass {
    device: Rc<Device>,
    pub octa_render_target: Rc<RefCell<RenderTarget>>,
    pub cdf_render_target: Rc<RefCell<RenderTarget>>,
    pub cdf_temp_render_target: Rc<RefCell<RenderTarget>>,
    octa_pipeline_handle: Rc<Pipeline<Compute>>,
    cdf_horizontal_pipeline_handle: Rc<Pipeline<Compute>>,
    cdf_vertical_pipeline_handle: Rc<Pipeline<Compute>>,
    pdf_normalize_pipeline_handle: Rc<Pipeline<Compute>>,
    cdf_normalize_pipeline_handle: Rc<Pipeline<Compute>>,
    cdf_invert_pipeline_handle: Rc<Pipeline<Compute>>,
}

impl ImportanceMapPass {
    pub const OCTA_SIZE: [u32; 2] = [1024, 1024];
    pub const CDF_SIZE: [u32; 2] = [1025, 1024];

    pub fn create(
        device: Rc<Device>,
        render_targets: &mut RenderTargets,
        pipeline_builder: &mut PipelineBuilder,
        descriptor_layouts: &DescriptorLayouts,
    ) -> Result<Self, AppError> {
        let octa_pipeline_handle = pipeline_builder.build_compute("octa_map", "octa_map|main", descriptor_layouts)?;
        let cdf_horizontal_pipeline_handle =
            pipeline_builder.build_compute("cdf_horizontal_map", "cdf_map|horizontal", descriptor_layouts)?;
        let cdf_vertical_pipeline_handle =
            pipeline_builder.build_compute("cdf_vertical_map", "cdf_map|vertical", descriptor_layouts)?;
        let pdf_normalize_pipeline_handle =
            pipeline_builder.build_compute("pdf_normalize_map", "cdf_map|pdfNormalize", descriptor_layouts)?;
        let cdf_normalize_pipeline_handle =
            pipeline_builder.build_compute("cdf_normalize_map", "cdf_map|cdfNormalize", descriptor_layouts)?;
        let cdf_invert_pipeline_handle =
            pipeline_builder.build_compute("cdf_invert_map", "cdf_invert|main", descriptor_layouts)?;

        Ok(Self {
            device,
            octa_render_target: render_targets.add(Self::octa_render_target_def())?,
            cdf_render_target: render_targets.add(Self::render_target_def())?,
            cdf_temp_render_target: render_targets.add(Self::render_target_def().duplicate("sky_importance_temp"))?,
            octa_pipeline_handle,
            cdf_horizontal_pipeline_handle,
            cdf_vertical_pipeline_handle,
            pdf_normalize_pipeline_handle,
            cdf_normalize_pipeline_handle,
            cdf_invert_pipeline_handle,
        })
    }

    fn octa_render_target_def() -> RenderTargetBuilder {
        RenderTargetBuilder::new("sky_octa_map")
            .with_storage()
            .with_sampled()
            .with_format(vk::Format::R32_SFLOAT)
            .with_size(RenderTargetSize::Custom(Self::OCTA_SIZE[0], Self::OCTA_SIZE[1]))
    }

    fn render_target_def() -> RenderTargetBuilder {
        RenderTargetBuilder::new("sky_importance_map")
            .with_storage()
            .with_format(vk::Format::R32_SFLOAT)
            .with_size(RenderTargetSize::Custom(Self::CDF_SIZE[0], Self::CDF_SIZE[1]))
    }

    pub fn record(
        &self,
        command_buffer: &CommandBuffer,
        descriptors: &RendererDescriptors,
        inputs: ImportanceMapInputs,
    ) -> Result<(), VulkanError> {
        self.device.begin_label("Sky Importance Map", command_buffer);

        let pipeline = &self.octa_pipeline_handle;

        command_buffer.bind_compute_pipeline(pipeline);
        command_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            [descriptors.global_set.inner, descriptors.compute_set.inner],
        );

        let pc = PushConstBuilder::new()
            .add_u32(inputs.src_sampler)
            .add_u32(self.octa_render_target.borrow().storage_index.unwrap())
            .build();

        command_buffer.push_constants(vk::ShaderStageFlags::COMPUTE, pipeline.layout, pc.as_ref());

        let x = math::workgroup_saturate(Self::OCTA_SIZE[0], pipeline.reflect_data.workgroup_size.0);
        let y = math::workgroup_saturate(Self::OCTA_SIZE[1], pipeline.reflect_data.workgroup_size.1);

        command_buffer.dispatch(x, y, 1);

        self.intra_barrier(command_buffer, &self.octa_render_target);

        let pipeline = &self.cdf_horizontal_pipeline_handle;

        command_buffer.bind_compute_pipeline(pipeline);
        command_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            [descriptors.global_set.inner, descriptors.compute_set.inner],
        );

        let pc = PushConstBuilder::new()
            .add_u32(self.octa_render_target.borrow().storage_index.unwrap())
            .add_u32(self.cdf_render_target.borrow().storage_index.unwrap())
            .build();

        command_buffer.push_constants(vk::ShaderStageFlags::COMPUTE, pipeline.layout, pc.as_ref());

        command_buffer.dispatch(1, 1024, 1);

        self.intra_barrier(command_buffer, &self.cdf_render_target);

        let pipeline = &self.cdf_vertical_pipeline_handle;

        command_buffer.bind_compute_pipeline(pipeline);
        command_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            [descriptors.global_set.inner, descriptors.compute_set.inner],
        );

        let pc = PushConstBuilder::new()
            .add_u32(self.cdf_render_target.borrow().storage_index.unwrap())
            .add_u32(self.cdf_temp_render_target.borrow().storage_index.unwrap())
            .build();

        command_buffer.push_constants(vk::ShaderStageFlags::COMPUTE, pipeline.layout, pc.as_ref());

        command_buffer.dispatch(1, 1, 1);

        self.intra_barrier(command_buffer, &self.cdf_render_target);

        let pipeline = &self.pdf_normalize_pipeline_handle;

        command_buffer.bind_compute_pipeline(pipeline);
        command_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            [descriptors.global_set.inner, descriptors.compute_set.inner],
        );

        let pc = PushConstBuilder::new()
            .add_u32(self.cdf_render_target.borrow().storage_index.unwrap())
            .add_u32(self.octa_render_target.borrow().storage_index.unwrap())
            .build();

        command_buffer.push_constants(vk::ShaderStageFlags::COMPUTE, pipeline.layout, pc.as_ref());

        let x = math::workgroup_saturate(Self::OCTA_SIZE[0], pipeline.reflect_data.workgroup_size.0);
        let y = math::workgroup_saturate(Self::OCTA_SIZE[1], pipeline.reflect_data.workgroup_size.1);

        command_buffer.dispatch(x, y, 1);

        self.intra_barrier(command_buffer, &self.cdf_render_target);

        let pipeline = &self.cdf_normalize_pipeline_handle;

        command_buffer.bind_compute_pipeline(pipeline);
        command_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            [descriptors.global_set.inner, descriptors.compute_set.inner],
        );

        let pc = PushConstBuilder::new()
            .add_u32(self.cdf_render_target.borrow().storage_index.unwrap())
            .add_u32(self.cdf_temp_render_target.borrow().storage_index.unwrap())
            .build();

        command_buffer.push_constants(vk::ShaderStageFlags::COMPUTE, pipeline.layout, pc.as_ref());

        let x = math::workgroup_saturate(Self::OCTA_SIZE[0], pipeline.reflect_data.workgroup_size.0);
        let y = math::workgroup_saturate(Self::OCTA_SIZE[1], pipeline.reflect_data.workgroup_size.1);

        command_buffer.dispatch(x, y, 1);

        self.intra_barrier(command_buffer, &self.cdf_temp_render_target);

        let pipeline = &self.cdf_invert_pipeline_handle;

        command_buffer.bind_compute_pipeline(pipeline);
        command_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            [descriptors.global_set.inner, descriptors.compute_set.inner],
        );

        let pc = PushConstBuilder::new()
            .add_u32(self.cdf_temp_render_target.borrow().storage_index.unwrap())
            .add_u32(self.cdf_render_target.borrow().storage_index.unwrap())
            .build();

        command_buffer.push_constants(vk::ShaderStageFlags::COMPUTE, pipeline.layout, pc.as_ref());

        // this way the result is 1024x1025
        command_buffer.dispatch(1, 1025, 1);

        self.device.end_label(command_buffer);

        Ok(())
    }

    fn intra_barrier(&self, command_buffer: &CommandBuffer, render_target: &Rc<RefCell<RenderTarget>>) {
        unsafe {
            let barriers = [render_target.borrow().image.inner].map(|image| vk::ImageMemoryBarrier {
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
    }
}

pub struct ImportanceMapInputs {
    pub src_sampler: u32,
}
