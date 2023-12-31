use crate::vulkan::{Device, IntoVulkanError, Pipeline, RtPipeline, VertexIndexBuffer, VulkanError};
use ash::vk;
use ash::vk::{CommandBuffer as RawCommandBuffer, Rect2D, Viewport};
use std::rc::Rc;

pub struct CommandBuffer {
    pub inner: RawCommandBuffer,
    device: Rc<Device>,
}

impl CommandBuffer {
    pub fn new(device: Rc<Device>, inner: RawCommandBuffer) -> Self {
        Self { device, inner }
    }

    pub fn reset(&self) -> Result<(), VulkanError> {
        unsafe {
            self.device
                .inner
                .reset_command_buffer(self.inner, vk::CommandBufferResetFlags::empty())
                .map_to_err("Cannot reset command buffer")
        }
    }

    pub fn bind_vertex_buffers(&self, buffers: &[&VertexIndexBuffer], offsets: &[u64]) {
        let buffers = buffers.iter().map(|vb| vb.inner.inner).collect::<Vec<_>>();

        unsafe {
            self.device
                .inner
                .cmd_bind_vertex_buffers(self.inner, 0, &buffers, offsets);
        }
    }

    pub fn begin_render_pass(&self, render_pass_info: &vk::RenderPassBeginInfo, contents: vk::SubpassContents) {
        unsafe {
            self.device
                .inner
                .cmd_begin_render_pass(self.inner, render_pass_info, contents)
        }
    }

    pub fn bind_pipeline(&self, pipeline: &Pipeline, bind_point: vk::PipelineBindPoint) {
        unsafe {
            self.device
                .inner
                .cmd_bind_pipeline(self.inner, bind_point, pipeline.inner)
        }
    }

    pub fn bind_rt_pipeline(&self, pipeline: &RtPipeline) {
        unsafe {
            self.device
                .inner
                .cmd_bind_pipeline(self.inner, vk::PipelineBindPoint::RAY_TRACING_KHR, pipeline.inner)
        }
    }

    pub fn set_viewport(&self, viewport: Viewport) {
        unsafe { self.device.inner.cmd_set_viewport(self.inner, 0, &[viewport]) }
    }

    pub fn set_scissor(&self, scissor: Rect2D) {
        unsafe { self.device.inner.cmd_set_scissor(self.inner, 0, &[scissor]) }
    }

    pub fn end_render_pass(&self) {
        unsafe { self.device.inner.cmd_end_render_pass(self.inner) }
    }

    pub fn begin(&self) -> Result<(), VulkanError> {
        let begin_info = vk::CommandBufferBeginInfo::default();

        unsafe {
            self.device
                .inner
                .begin_command_buffer(self.inner, &begin_info)
                .map_to_err("cannot begin recording")
        }
    }

    pub fn begin_one_time(&self) -> Result<(), VulkanError> {
        let begin_info = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };

        unsafe {
            self.device
                .inner
                .begin_command_buffer(self.inner, &begin_info)
                .map_to_err("cannot begin recording")
        }
    }

    pub fn end(&self) -> Result<(), VulkanError> {
        unsafe {
            self.device
                .inner
                .end_command_buffer(self.inner)
                .map_to_err("cannot end command buffer")
        }
    }
}
