use crate::vulkan::{Device, IntoVulkanError, RenderPass, SwapChain, VulkanError};
use ash::vk;
use ash::vk::{Pipeline as RawPipeline, PipelineLayout, PipelineShaderStageCreateInfo, Rect2D, Viewport};
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
        swapchain: &SwapChain,
        render_pass: &RenderPass,
        stages: &[PipelineShaderStageCreateInfo],
    ) -> Result<Self, VulkanError> {
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        let dynamic_state = vk::PipelineDynamicStateCreateInfo {
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo {
            vertex_binding_description_count: 0,
            vertex_attribute_description_count: 0,
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
            width: swapchain.extent.width as f32,
            height: swapchain.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let scissor = Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain.extent,
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
            front_face: vk::FrontFace::CLOCKWISE,
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

        let color_blending = vk::PipelineColorBlendStateCreateInfo {
            logic_op_enable: vk::FALSE,
            logic_op: vk::LogicOp::COPY,
            attachment_count: 1,
            p_attachments: &color_blend_attachment,
            ..Default::default()
        };

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default();

        let layout = unsafe {
            device
                .inner
                .create_pipeline_layout(&pipeline_layout_info, None)
                .map_to_err("Cannot create pipeline layout")?
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
