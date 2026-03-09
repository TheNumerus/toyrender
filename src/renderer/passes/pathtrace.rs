use crate::err::AppError;
use crate::renderer::descriptors::RendererDescriptors;
use crate::renderer::pipeline_builder::PipelineBuilder;
use crate::renderer::render_target::{RenderTarget, RenderTargetBuilder, RenderTargets};
use crate::renderer::{PushConstBuilder, VulkanRenderer};
use crate::vulkan::{CommandBuffer, Device, Pipeline, Rt};
use ash::vk;
use std::cell::{Ref, RefCell};
use std::rc::Rc;

pub(crate) struct PathTracePass {
    device: Rc<Device>,
    pub direct_render_target: Rc<RefCell<RenderTarget>>,
    pub indirect_render_target: Rc<RefCell<RenderTarget>>,
    pub pipeline_handle: Rc<Pipeline<Rt>>,
}

impl PathTracePass {
    pub const TARGET_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

    pub fn create(
        device: Rc<Device>,
        render_targets: &mut RenderTargets,
        pipeline_builder: &mut PipelineBuilder,
        descriptors: Ref<RendererDescriptors>,
    ) -> Result<Self, AppError> {
        let [a, b] = Self::render_target_defs();
        let direct_render_target = render_targets.add(a)?;
        let indirect_render_target = render_targets.add(b)?;

        let pipeline_handle =
            pipeline_builder.build_rt("pt_rt", "pt_rt|raygen", "pt_rt|miss", "pt_rt|chit", descriptors)?;

        Ok(Self {
            device,
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
        renderer: &VulkanRenderer,
        viewport: (u32, u32),
    ) -> Result<(), AppError> {
        self.device.begin_label("Path Tracing", command_buffer);

        let pipeline = &self.pipeline_handle;

        command_buffer.bind_rt_pipeline(pipeline);
        command_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::RAY_TRACING_KHR,
            pipeline.layout,
            [
                renderer.descriptors.borrow().global_sets[renderer.current_frame].inner,
                renderer.descriptors.borrow().compute_sets[renderer.current_frame].inner,
            ],
        );

        let pc = PushConstBuilder::with_capacity(9 * size_of::<u32>())
            .add_u32(renderer.quality.pt_bounces as u32)
            .add_u32(*renderer.descriptors.borrow().samplers.get("gbuffer_color").unwrap() as u32)
            .add_u32(*renderer.descriptors.borrow().samplers.get("gbuffer_depth").unwrap() as u32)
            .add_u32(*renderer.descriptors.borrow().samplers.get("gbuffer_normal").unwrap() as u32)
            .add_u32(*renderer.descriptors.borrow().storages.get("rt_direct").unwrap() as u32)
            .add_u32(*renderer.descriptors.borrow().storages.get("rt_indirect").unwrap() as u32)
            .add_u32(*renderer.descriptors.borrow().samplers.get("sky").unwrap() as u32)
            .add_f32(renderer.quality.rt_direct_trace_distance)
            .add_f32(renderer.quality.rt_indirect_trace_distance)
            .build();

        unsafe {
            self.device.inner.cmd_push_constants(
                command_buffer.inner,
                pipeline.layout,
                vk::ShaderStageFlags::RAYGEN_KHR
                    | vk::ShaderStageFlags::CLOSEST_HIT_KHR
                    | vk::ShaderStageFlags::MISS_KHR,
                0,
                &pc,
            );

            renderer.context.rt_pipeline_ext.loader.cmd_trace_rays(
                command_buffer.inner,
                &renderer.shader_binding_table.raygen_region,
                &renderer.shader_binding_table.miss_region,
                &renderer.shader_binding_table.hit_region,
                &renderer.shader_binding_table.call_region,
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

            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );
        }

        self.device.end_label(command_buffer);

        Ok(())
    }
}
