use crate::vulkan::{Device, IntoVulkanError, VulkanError};
use ash::vk;
use std::rc::Rc;

pub struct DescriptorPool {
    pub inner: vk::DescriptorPool,
    device: Rc<Device>,
}

impl DescriptorPool {
    pub fn new(device: Rc<Device>, pool_size: u32) -> Result<DescriptorPool, VulkanError> {
        let pool_size = vk::DescriptorPoolSize {
            descriptor_count: pool_size,
            ty: vk::DescriptorType::UNIFORM_BUFFER,
        };

        let pool_info = vk::DescriptorPoolCreateInfo {
            pool_size_count: 1,
            p_pool_sizes: &pool_size,
            max_sets: pool_size.descriptor_count,
            ..Default::default()
        };

        let inner = unsafe {
            device
                .inner
                .create_descriptor_pool(&pool_info, None)
                .map_to_err("Cannot create descriptor pool")?
        };

        Ok(Self { inner, device })
    }

    pub fn allocate_sets(&self, layouts: &[vk::DescriptorSetLayout]) -> Result<Vec<DescriptorSet>, VulkanError> {
        let alloc_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool: self.inner,
            descriptor_set_count: layouts.len() as u32,
            p_set_layouts: layouts.as_ptr(),
            ..Default::default()
        };

        unsafe {
            let raw_sets = self
                .device
                .inner
                .allocate_descriptor_sets(&alloc_info)
                .map_to_err("Cannot allocate descriptor sets")?;

            Ok(raw_sets.iter().map(|r| DescriptorSet { inner: *r }).collect())
        }
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe {
            self.device.inner.destroy_descriptor_pool(self.inner, None);
        }
    }
}

pub struct DescriptorSet {
    pub inner: vk::DescriptorSet,
}
