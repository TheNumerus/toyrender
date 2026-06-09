use crate::err::AppError;
use crate::renderer::descriptors::{DescriptorLayouts, RendererDescriptors};
use crate::renderer::pipeline_builder::{PipelineBuilder, SpecConsts};
use crate::renderer::render_target::{RenderTarget, RenderTargetBuilder, RenderTargets};
use crate::renderer::{FrameContext, PushConstBuilder, VulkanContext};
use crate::scene::SkyVariant;
use crate::vulkan::{CommandBuffer, Pipeline, Rt, ShaderBindingTable, VulkanError};
use ash::vk;
use std::cell::RefCell;
use std::rc::Rc;

pub struct ReferencePathtracePass {
    context: Rc<VulkanContext>,
    pub render_target: Rc<RefCell<RenderTarget>>,
    pub shader_pipeline_handle: Rc<Pipeline<Rt>>,
    pub solid_sky_pipeline_handle: Rc<Pipeline<Rt>>,
    pub texture_sky_pipeline_handle: Rc<Pipeline<Rt>>,
}

impl ReferencePathtracePass {
    pub const TARGET_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

    pub fn create(
        context: Rc<VulkanContext>,
        render_targets: &mut RenderTargets,
        pipeline_builder: &mut PipelineBuilder,
        descriptor_layouts: &DescriptorLayouts,
    ) -> Result<Self, AppError> {
        let render_target = render_targets.add(Self::render_target_def())?;

        let shader_pipeline_handle = pipeline_builder.build_rt(
            "pt|shader",
            "pt_reference|raygen",
            &["pt_reference|miss", "pt_reference|missEmpty"],
            &["pt_reference|chit", "pt_reference|chitEmpty"],
            descriptor_layouts,
            Some(SpecConsts::new().push(0)),
        )?;

        let solid_sky_pipeline_handle = pipeline_builder.build_rt(
            "pt|solid",
            "pt_reference|raygen",
            &["pt_reference|miss", "pt_reference|missEmpty"],
            &["pt_reference|chit", "pt_reference|chitEmpty"],
            descriptor_layouts,
            Some(SpecConsts::new().push(1)),
        )?;

        let texture_sky_pipeline_handle = pipeline_builder.build_rt(
            "pt|texture",
            "pt_reference|raygen",
            &["pt_reference|miss", "pt_reference|missEmpty"],
            &["pt_reference|chit", "pt_reference|chitEmpty"],
            descriptor_layouts,
            Some(SpecConsts::new().push(2)),
        )?;

        Ok(Self {
            context,
            render_target,
            shader_pipeline_handle,
            solid_sky_pipeline_handle,
            texture_sky_pipeline_handle,
        })
    }

    fn render_target_def() -> RenderTargetBuilder {
        RenderTargetBuilder::new("rt_out")
            .with_storage()
            .with_transfer()
            .with_format(Self::TARGET_FORMAT)
    }

    pub fn record_pt(
        &self,
        command_buffer: &CommandBuffer,
        descriptors: &RendererDescriptors,
        inputs: ReferencePathTraceInputs,
        context: &FrameContext,
        viewport: (u32, u32),
    ) -> Result<(), VulkanError> {
        self.context.device.begin_label("Path Tracing", command_buffer);

        let pipeline = match inputs.sky {
            SkyVariant::SingleColor(_) => &self.solid_sky_pipeline_handle,
            SkyVariant::Textured(_, _) => &self.texture_sky_pipeline_handle,
            _ => &self.shader_pipeline_handle,
        };

        command_buffer.bind_rt_pipeline(pipeline);

        unsafe {
            command_buffer.bind_descriptor_sets(
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                pipeline.layout,
                [descriptors.global_set.inner, descriptors.compute_set.inner],
            );

            let pc = PushConstBuilder::with_capacity(11 * size_of::<u32>())
                .add_u32(context.frame_index)
                .add_u32(inputs.bounces)
                .add_u32(self.render_target.borrow().storage_index.unwrap())
                .add_u32(inputs.sky_pdf.sampler_index.unwrap())
                .add_u32(inputs.sky_importance_map.storage_index.unwrap())
                .add_u32(if context.importance_sampling { 1 } else { 0 })
                .add_u32(inputs.sky_sampler)
                .add_f32(inputs.direct_trace_distance)
                .add_f32(inputs.indirect_trace_distance)
                .add_f32(inputs.fov)
                .add_f32(inputs.indirect_intensity_clamp)
                .build();

            command_buffer.push_constants(
                vk::ShaderStageFlags::RAYGEN_KHR
                    | vk::ShaderStageFlags::MISS_KHR
                    | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                pipeline.layout,
                &pc,
            );

            self.context.rt_pipeline_ext.loader.cmd_trace_rays(
                command_buffer.inner,
                &inputs.sbt.raygen_region,
                &inputs.sbt.miss_region,
                &inputs.sbt.hit_region,
                &inputs.sbt.call_region,
                viewport.0,
                viewport.1,
                1,
            );

            let barriers = [self.render_target.borrow().image.inner].map(|image| vk::ImageMemoryBarrier {
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

            self.context.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );
        }

        self.context.device.end_label(command_buffer);

        Ok(())
    }

    pub fn get_active_pipeline(&self, sky: &SkyVariant) -> &Pipeline<Rt> {
        match sky {
            SkyVariant::SingleColor(_) => &self.solid_sky_pipeline_handle,
            SkyVariant::Textured(_, _) => &self.texture_sky_pipeline_handle,
            _ => &self.shader_pipeline_handle,
        }
    }
}

pub struct ReferencePathTraceInputs<'a> {
    pub sky_sampler: u32,
    pub sky: &'a SkyVariant,
    pub sky_pdf: &'a RenderTarget,
    pub sky_importance_map: &'a RenderTarget,
    pub sbt: &'a ShaderBindingTable,
    pub bounces: u32,
    pub direct_trace_distance: f32,
    pub indirect_trace_distance: f32,
    pub fov: f32,
    pub indirect_intensity_clamp: f32,
}
