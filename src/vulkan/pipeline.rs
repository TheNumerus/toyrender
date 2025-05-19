use crate::vulkan::{Device, IntoVulkanError, RayTracingPipeline, RenderPass, Vertex, VulkanError};
use ash::vk;
use ash::vk::{Pipeline as RawPipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use std::marker::PhantomData;
use std::rc::Rc;

pub struct Graphics;
pub struct Rt;
pub struct Compute;

pub struct Pipeline<T> {
    pub inner: RawPipeline,
    pub layout: PipelineLayout,
    device: Rc<Device>,
    _marker: PhantomData<T>,
}

impl Pipeline<Graphics> {
    pub fn new_graphics(
        device: Rc<Device>,
        render_pass: Option<&RenderPass>,
        stages: &[PipelineShaderStageCreateInfo],
        descriptor_layouts: &[vk::DescriptorSetLayout],
        color_attachments: u32,
    ) -> Result<Self, VulkanError> {
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        let dynamic_state = vk::PipelineDynamicStateCreateInfo {
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };

        let vertex_binds = Vertex::binding_description();
        let vertex_atts = Vertex::attribute_description();

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo {
            vertex_binding_description_count: 1,
            vertex_attribute_description_count: vertex_atts.len() as u32,
            p_vertex_binding_descriptions: &vertex_binds,
            p_vertex_attribute_descriptions: vertex_atts.as_ptr(),
            ..Default::default()
        };

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: vk::FALSE,
            ..Default::default()
        };

        let viewport_state = vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            scissor_count: 1,
            ..Default::default()
        };

        let rasterizer = vk::PipelineRasterizationStateCreateInfo {
            depth_clamp_enable: vk::FALSE,
            rasterizer_discard_enable: vk::FALSE,
            polygon_mode: vk::PolygonMode::FILL,
            line_width: 1.0,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            depth_bias_enable: vk::FALSE,
            ..Default::default()
        };

        let multisampling = vk::PipelineMultisampleStateCreateInfo {
            sample_shading_enable: vk::FALSE,
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState {
            color_write_mask: vk::ColorComponentFlags::RGBA,
            blend_enable: vk::FALSE,
            ..Default::default()
        };

        let color_attachments = vec![color_blend_attachment; color_attachments as usize];

        let color_blending = vk::PipelineColorBlendStateCreateInfo {
            logic_op_enable: vk::FALSE,
            logic_op: vk::LogicOp::COPY,
            attachment_count: color_attachments.len() as u32,
            p_attachments: color_attachments.as_ptr(),
            ..Default::default()
        };

        let ranges = [vk::PushConstantRange {
            offset: 0,
            size: (std::mem::size_of::<nalgebra_glm::Mat4>()) as u32,
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        }];

        let layout = create_layout(&device, &ranges, descriptor_layouts)?;

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: vk::TRUE,
            depth_write_enable: vk::TRUE,
            depth_compare_op: vk::CompareOp::GREATER,
            depth_bounds_test_enable: vk::FALSE,
            min_depth_bounds: 0.0,
            max_depth_bounds: 1.0,
            stencil_test_enable: vk::FALSE,
            ..Default::default()
        };

        let mut pipeline_info = vk::GraphicsPipelineCreateInfo {
            stage_count: stages.len() as u32,
            p_stages: stages.as_ptr(),
            p_vertex_input_state: &vertex_input_info,
            p_input_assembly_state: &input_assembly,
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterizer,
            p_multisample_state: &multisampling,
            p_color_blend_state: &color_blending,
            p_dynamic_state: &dynamic_state,
            p_depth_stencil_state: &depth_stencil,
            layout,
            render_pass: render_pass.map(|a| a.inner).unwrap_or(vk::RenderPass::null()),
            subpass: 0,
            ..Default::default()
        };

        let formats = [vk::Format::R16G16B16A16_SFLOAT];

        let rendering_info = vk::PipelineRenderingCreateInfo {
            color_attachment_count: color_attachments.len() as u32,
            p_color_attachment_formats: formats.as_ptr(),
            ..Default::default()
        };

        if render_pass.is_none() {
            pipeline_info.p_next = std::ptr::addr_of!(rendering_info) as *const _;
        }

        let pipeline = unsafe {
            device
                .inner
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, code)| VulkanError {
                    code,
                    msg: "Cannot create pipeline".into(),
                })?
        }[0];

        Ok(Self {
            device,
            inner: pipeline,
            layout,
            _marker: PhantomData,
        })
    }
}

impl Pipeline<Rt> {
    pub fn new_rt(
        device: Rc<Device>,
        rtp: &RayTracingPipeline,
        stages: &[PipelineShaderStageCreateInfo],
        descriptor_layouts: &[vk::DescriptorSetLayout],
    ) -> Result<Self, VulkanError> {
        let gen_group_create_info = vk::RayTracingShaderGroupCreateInfoKHR {
            ty: vk::RayTracingShaderGroupTypeKHR::GENERAL,
            general_shader: 0,
            closest_hit_shader: vk::SHADER_UNUSED_KHR,
            any_hit_shader: vk::SHADER_UNUSED_KHR,
            intersection_shader: vk::SHADER_UNUSED_KHR,
            ..Default::default()
        };

        let miss_group_create_info = vk::RayTracingShaderGroupCreateInfoKHR {
            ty: vk::RayTracingShaderGroupTypeKHR::GENERAL,
            general_shader: 1,
            closest_hit_shader: vk::SHADER_UNUSED_KHR,
            any_hit_shader: vk::SHADER_UNUSED_KHR,
            intersection_shader: vk::SHADER_UNUSED_KHR,
            ..Default::default()
        };

        let hit_group_create_info = vk::RayTracingShaderGroupCreateInfoKHR {
            ty: vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP,
            general_shader: vk::SHADER_UNUSED_KHR,
            closest_hit_shader: 2,
            any_hit_shader: vk::SHADER_UNUSED_KHR,
            intersection_shader: vk::SHADER_UNUSED_KHR,
            ..Default::default()
        };

        let dynamic_state = vk::PipelineDynamicStateCreateInfo {
            dynamic_state_count: 0,
            p_dynamic_states: [].as_ptr(),
            ..Default::default()
        };

        let ranges = [vk::PushConstantRange {
            offset: 0,
            size: std::mem::size_of::<f32>() as u32,
            stage_flags: vk::ShaderStageFlags::RAYGEN_KHR,
        }];

        let layout = create_layout(&device, &ranges, descriptor_layouts)?;

        let groups = [gen_group_create_info, miss_group_create_info, hit_group_create_info];

        let pipeline_info = vk::RayTracingPipelineCreateInfoKHR {
            stage_count: stages.len() as u32,
            p_stages: stages.as_ptr(),
            group_count: groups.len() as u32,
            p_groups: groups.as_ptr(),
            max_pipeline_ray_recursion_depth: device.rt_properties.max_recursion,
            p_dynamic_state: &dynamic_state,
            layout,
            ..Default::default()
        };

        let pipeline = unsafe {
            rtp.loader
                .create_ray_tracing_pipelines(
                    vk::DeferredOperationKHR::null(),
                    vk::PipelineCache::null(),
                    &[pipeline_info],
                    None,
                )
                .map_err(|code| VulkanError {
                    code,
                    msg: "Cannot create pipeline".into(),
                })?
        }[0];

        Ok(Self {
            inner: pipeline,
            layout,
            device,
            _marker: PhantomData,
        })
    }
}

impl Pipeline<Compute> {
    pub fn new_compute(
        device: Rc<Device>,
        stage: PipelineShaderStageCreateInfo,
        descriptor_layouts: &[vk::DescriptorSetLayout],
    ) -> Result<Self, VulkanError> {
        let ranges = [vk::PushConstantRange {
            offset: 0,
            size: std::mem::size_of::<f32>() as u32,
            stage_flags: vk::ShaderStageFlags::COMPUTE,
        }];

        let layout = create_layout(&device, &ranges, descriptor_layouts)?;

        let create_info = vk::ComputePipelineCreateInfo {
            base_pipeline_index: 0,
            stage,
            layout,
            ..Default::default()
        };

        let pipeline = unsafe {
            device
                .inner
                .create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None)
                .map_err(|(_, code)| VulkanError {
                    code,
                    msg: "Cannot create compute pipeline".into(),
                })?
        }[0];

        Ok(Self {
            inner: pipeline,
            layout,
            device,
            _marker: PhantomData,
        })
    }
}

impl<T> Drop for Pipeline<T> {
    fn drop(&mut self) {
        unsafe {
            self.device.inner.destroy_pipeline_layout(self.layout, None);
            self.device.inner.destroy_pipeline(self.inner, None);
        }
    }
}

fn create_layout(
    device: &Rc<Device>,
    ranges: &[vk::PushConstantRange],
    descriptor_layouts: &[vk::DescriptorSetLayout],
) -> Result<PipelineLayout, VulkanError> {
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
        p_push_constant_ranges: ranges.as_ptr(),
        push_constant_range_count: ranges.len() as u32,
        p_set_layouts: descriptor_layouts.as_ptr(),
        set_layout_count: descriptor_layouts.len() as u32,
        ..Default::default()
    };

    unsafe {
        device
            .inner
            .create_pipeline_layout(&pipeline_layout_info, None)
            .map_to_err("Cannot create pipeline layout")
    }
}
