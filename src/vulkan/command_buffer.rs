use crate::vulkan::{
    Compute, DebugMarker, Device, Graphics, IntoVulkanError, Pipeline, Rt, VertexIndexBuffer, VulkanError,
};
use ash::vk;
use ash::vk::{CommandBuffer as RawCommandBuffer, Handle, Rect2D, Viewport};
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

    pub fn begin_rendering(&self, info: &vk::RenderingInfo) {
        unsafe { self.device.inner.cmd_begin_rendering(self.inner, info) }
    }

    pub fn bind_graphics_pipeline(&self, pipeline: &Pipeline<Graphics>) {
        unsafe {
            self.device
                .inner
                .cmd_bind_pipeline(self.inner, vk::PipelineBindPoint::GRAPHICS, pipeline.inner)
        }
    }

    pub fn bind_rt_pipeline(&self, pipeline: &Pipeline<Rt>) {
        unsafe {
            self.device
                .inner
                .cmd_bind_pipeline(self.inner, vk::PipelineBindPoint::RAY_TRACING_KHR, pipeline.inner)
        }
    }

    pub fn bind_compute_pipeline(&self, pipeline: &Pipeline<Compute>) {
        unsafe {
            self.device
                .inner
                .cmd_bind_pipeline(self.inner, vk::PipelineBindPoint::COMPUTE, pipeline.inner)
        }
    }

    pub fn set_viewport(&self, viewport: Viewport) {
        unsafe { self.device.inner.cmd_set_viewport(self.inner, 0, &[viewport]) }
    }

    pub fn set_scissor(&self, scissor: Rect2D) {
        unsafe { self.device.inner.cmd_set_scissor(self.inner, 0, &[scissor]) }
    }

    pub fn end_rendering(&self) {
        unsafe { self.device.inner.cmd_end_rendering(self.inner) }
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

    pub fn bind_descriptor_sets(
        &self,
        bind_point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        sets: impl AsRef<[vk::DescriptorSet]>,
    ) {
        unsafe {
            self.device
                .inner
                .cmd_bind_descriptor_sets(self.inner, bind_point, layout, 0, sets.as_ref(), &[]);
        }
    }

    pub fn dispatch(&self, groups_x: u32, groups_y: u32, groups_z: u32) {
        unsafe {
            self.device.inner.cmd_dispatch(self.inner, groups_x, groups_y, groups_z);
        }
    }
}

impl DebugMarker for CommandBuffer {
    fn device(&self) -> &Rc<Device> {
        &self.device
    }

    fn object_type(&self) -> vk::ObjectType {
        vk::ObjectType::COMMAND_BUFFER
    }

    fn handle(&self) -> u64 {
        self.inner.as_raw()
    }
}
