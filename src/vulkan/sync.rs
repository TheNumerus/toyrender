use crate::vulkan::{Device, IntoVulkanError, VulkanError};
use ash::vk;
use ash::vk::Fence as RawFence;
use ash::vk::Semaphore as RawSemaphore;

pub struct Semaphore {
    pub inner: RawSemaphore,
    device: std::rc::Rc<Device>,
}

impl Semaphore {
    pub fn new(device: std::rc::Rc<Device>) -> Result<Self, VulkanError> {
        let info = vk::SemaphoreCreateInfo::default();

        let inner = unsafe {
            device
                .inner
                .create_semaphore(&info, None)
                .map_to_err("Cannot create semaphore")?
        };

        Ok(Self { inner, device })
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe { self.device.inner.destroy_semaphore(self.inner, None) }
    }
}

pub struct Fence {
    pub inner: RawFence,
    device: std::rc::Rc<Device>,
}

impl Fence {
    pub fn new(device: std::rc::Rc<Device>) -> Result<Self, VulkanError> {
        let info = vk::FenceCreateInfo {
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };

        let inner = unsafe {
            device
                .inner
                .create_fence(&info, None)
                .map_to_err("Cannot create fence")?
        };

        Ok(Self { inner, device })
    }

    pub fn wait(&self) -> Result<(), VulkanError> {
        unsafe {
            self.device
                .inner
                .wait_for_fences(&[self.inner], true, u64::MAX)
                .map_to_err("failed to wait for fence")
        }
    }

    pub fn reset(&self) -> Result<(), VulkanError> {
        unsafe {
            self.device
                .inner
                .reset_fences(&[self.inner])
                .map_to_err("failed to reset fence")
        }
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        unsafe { self.device.inner.destroy_fence(self.inner, None) }
    }
}
