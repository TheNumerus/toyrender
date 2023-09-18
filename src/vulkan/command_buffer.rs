use crate::vulkan::{Device, IntoVulkanError, VulkanError};
use ash::vk;
use ash::vk::CommandBuffer as RawCommandBuffer;
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

    pub fn begin(&self) -> Result<(), VulkanError> {
        let begin_info = vk::CommandBufferBeginInfo::default();

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
