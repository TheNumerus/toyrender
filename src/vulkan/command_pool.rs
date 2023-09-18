use crate::vulkan::{Device, VulkanError};
use ash::vk;
use ash::vk::CommandPool as RawCommandPool;
use std::rc::Rc;

pub struct CommandPool {
    pub inner: RawCommandPool,
    device: Rc<Device>,
}

impl CommandPool {
    pub fn new(device: Rc<Device>) -> Result<Self, VulkanError> {
        let command_pool = vk::CommandPoolCreateInfo {
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            queue_family_index: device.graphics_queue_family as u32,
            ..Default::default()
        };

        let inner = unsafe {
            device
                .inner
                .create_command_pool(&command_pool, None)
                .map_err(|code| VulkanError {
                    code,
                    msg: "Cannot create command pool".into(),
                })?
        };

        Ok(Self { device, inner })
    }
}

impl Drop for CommandPool {
    fn drop(&mut self) {
        unsafe {
            self.device.inner.destroy_command_pool(self.inner, None);
        }
    }
}
