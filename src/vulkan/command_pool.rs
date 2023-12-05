use crate::vulkan::command_buffer::CommandBuffer;
use crate::vulkan::{Device, IntoVulkanError, VulkanError};
use ash::vk;
use ash::vk::CommandPool as RawCommandPool;
use std::rc::Rc;

pub struct CommandPool {
    pub inner: RawCommandPool,
    device: Rc<Device>,
}

impl CommandPool {
    pub fn new_graphics(device: Rc<Device>) -> Result<Self, VulkanError> {
        let command_pool = vk::CommandPoolCreateInfo {
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index: device.graphics_queue_family as u32,
            ..Default::default()
        };

        let inner = unsafe {
            device
                .inner
                .create_command_pool(&command_pool, None)
                .map_to_err("Cannot create command pool")?
        };

        Ok(Self { device, inner })
    }

    pub fn new_compute(device: Rc<Device>) -> Result<Self, VulkanError> {
        let command_pool = vk::CommandPoolCreateInfo {
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index: device.compute_queue_family as u32,
            ..Default::default()
        };

        let inner = unsafe {
            device
                .inner
                .create_command_pool(&command_pool, None)
                .map_to_err("Cannot create command pool")?
        };

        Ok(Self { device, inner })
    }

    pub fn allocate_cmd_buffers(&self, count: u32) -> Result<Vec<CommandBuffer>, VulkanError> {
        let alloc_info = vk::CommandBufferAllocateInfo {
            command_pool: self.inner,
            level: vk::CommandBufferLevel::PRIMARY,
            command_buffer_count: count,
            ..Default::default()
        };

        let command_buffers = unsafe {
            self.device
                .inner
                .allocate_command_buffers(&alloc_info)
                .map_to_err("Cannot allocate command buffer")?
        };

        Ok(command_buffers
            .into_iter()
            .map(|inner| CommandBuffer::new(self.device.clone(), inner))
            .collect::<Vec<_>>())
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe {
            self.device.inner.destroy_command_pool(self.inner, None);
        }
    }
}
