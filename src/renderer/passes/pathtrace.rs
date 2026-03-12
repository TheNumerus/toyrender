use crate::err::AppError;
use crate::renderer::descriptors::{DescriptorLayouts, RendererDescriptors};
use crate::renderer::pipeline_builder::PipelineBuilder;
use crate::renderer::render_target::{RenderTarget, RenderTargetBuilder, RenderTargets};
use crate::renderer::{PushConstBuilder, VulkanContext};
use crate::vulkan::{CommandBuffer, Pipeline, Rt, ShaderBindingTable};
use ash::vk;
use std::cell::RefCell;
use std::rc::Rc;

pub(crate) struct PathTracePass {
    context: Rc<VulkanContext>,
    pub direct_render_target: Rc<RefCell<RenderTarget>>,
    pub indirect_render_target: Rc<RefCell<RenderTarget>>,
    pub pipeline_handle: Rc<Pipeline<Rt>>,
}

impl PathTracePass {
    pub const TARGET_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

    pub fn create(
        context: Rc<VulkanContext>,
        render_targets: &mut RenderTargets,
        pipeline_builder: &mut PipelineBuilder,
        descriptor_layouts: &DescriptorLayouts,
    ) -> Result<Self, AppError> {
        let [a, b] = Self::render_target_defs();
        let direct_render_target = render_targets.add(a)?;
        let indirect_render_target = render_targets.add(b)?;

        let pipeline_handle =
            pipeline_builder.build_rt("pt_rt", "pt_rt|raygen", "pt_rt|miss", "pt_rt|chit", descriptor_layouts)?;

        Ok(Self {
            context,
            direct_render_target,
            indirect_render_target,
            pipeline_handle,
        })
    }

    pub fn render_target_defs() -> [RenderTargetBuilder; 2] {
        [
            RenderTargetBuilder::new("rt_direct")
                .with_storage()
                .with_transfer()
                .with_format(Self::TARGET_FORMAT),
            RenderTargetBuilder::new("rt_indirect")
                .with_storage()
                .with_transfer()
                .with_format(Self::TARGET_FORMAT),
        ]
    }

    pub fn record(
        &self,
        command_buffer: &CommandBuffer,
        descriptors: &RendererDescriptors,
        inputs: PathTraceInputs,
        viewport: (u32, u32),
    ) -> Result<(), AppError> {
        self.context.device.begin_label("Path Tracing", command_buffer);

        let pipeline = &self.pipeline_handle;

        command_buffer.bind_rt_pipeline(pipeline);
        command_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            pipeline.layout,
            [descriptors.global_set.inner, descriptors.compute_set.inner],
        );

        let pc = PushConstBuilder::with_capacity(10 * size_of::<u32>())
            .add_u32(inputs.bounces)
            .add_u32(inputs.color.storage_index.unwrap())
            .add_u32(inputs.depth.storage_index.unwrap())
            .add_u32(inputs.normal.storage_index.unwrap())
            .add_u32(self.direct_render_target.borrow().storage_index.unwrap())
            .add_u32(self.indirect_render_target.borrow().storage_index.unwrap())
            .add_u32(inputs.sky_sampler)
            .add_f32(inputs.direct_trace_distance)
            .add_f32(inputs.indirect_trace_distance)
            .add_f32(inputs.indirect_intensity_clamp)
            .build();

        unsafe {
            self.context.device.inner.cmd_push_constants(
                command_buffer.inner,
                pipeline.layout,
                vk::ShaderStageFlags::RAYGEN_KHR
                    | vk::ShaderStageFlags::CLOSEST_HIT_KHR
                    | vk::ShaderStageFlags::MISS_KHR,
                0,
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

            let barriers = [
                self.direct_render_target.borrow().image.inner,
                self.indirect_render_target.borrow().image.inner,
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
}

pub struct PathTraceInputs<'a> {
    pub color: &'a RenderTarget,
    pub depth: &'a RenderTarget,
    pub normal: &'a RenderTarget,
    pub sky_sampler: u32,
    pub sbt: &'a ShaderBindingTable,
    pub bounces: u32,
    pub direct_trace_distance: f32,
    pub indirect_trace_distance: f32,
    pub indirect_intensity_clamp: f32,
}
