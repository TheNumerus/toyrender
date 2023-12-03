use crate::vulkan::{Device, IntoVulkanError, RayTracingPipeline, RenderPass, Vertex, VulkanError};
use ash::vk;
use ash::vk::{Extent2D, Pipeline as RawPipeline, PipelineLayout, PipelineShaderStageCreateInfo, Rect2D, Viewport};
use std::rc::Rc;

pub struct Pipeline {
    pub inner: RawPipeline,
    pub layout: PipelineLayout,
    pub viewport: Viewport,
    pub scissor: Rect2D,
    device: Rc<Device>,
}
impl Pipeline {
    pub fn new(
        device: Rc<Device>,
        default_extent: Extent2D,
        render_pass: &RenderPass,
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

        let viewport = Viewport {
            x: 0.0,
            y: 0.0,
            width: default_extent.width as f32,
            height: default_extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let scissor = Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: default_extent,
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
            size: (std::mem::size_of::<f32>() + std::mem::size_of::<nalgebra_glm::Mat4>()) as u32,
            stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        }];

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            p_push_constant_ranges: ranges.as_ptr(),
            push_constant_range_count: ranges.len() as u32,
            p_set_layouts: descriptor_layouts.as_ptr(),
            set_layout_count: descriptor_layouts.len() as u32,
            ..Default::default()
        };

        let layout = unsafe {
            device
                .inner
                .create_pipeline_layout(&pipeline_layout_info, None)
                .map_to_err("Cannot create pipeline layout")?
        };

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: vk::TRUE,
            depth_write_enable: vk::TRUE,
            depth_compare_op: vk::CompareOp::LESS,
            depth_bounds_test_enable: vk::FALSE,
            min_depth_bounds: 0.0,
            max_depth_bounds: 1.0,
            stencil_test_enable: vk::FALSE,
            ..Default::default()
        };

        let pipeline_info = vk::GraphicsPipelineCreateInfo {
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
            render_pass: render_pass.inner,
            subpass: 0,
            ..Default::default()
        };

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
            viewport,
            scissor,
        })
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.inner.destroy_pipeline_layout(self.layout, None);
            self.device.inner.destroy_pipeline(self.inner, None);
        }
    }
}

pub struct RtPipeline {
    pub inner: RawPipeline,
    pub layout: PipelineLayout,
    device: Rc<Device>,
}

impl RtPipeline {
    pub fn new(
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

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            p_push_constant_ranges: ranges.as_ptr(),
            push_constant_range_count: ranges.len() as u32,
            p_set_layouts: descriptor_layouts.as_ptr(),
            set_layout_count: descriptor_layouts.len() as u32,
            ..Default::default()
        };

        let layout = unsafe {
            device
                .inner
                .create_pipeline_layout(&pipeline_layout_info, None)
                .map_to_err("Cannot create pipeline layout")?
        };

        let groups = [gen_group_create_info, miss_group_create_info];

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
        })
    }
}

impl Drop for RtPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.inner.destroy_pipeline(self.inner, None);
            self.device.inner.destroy_pipeline_layout(self.layout, None);
        }
    }
}
