use crate::vulkan::{Device, IntoVulkanError, VulkanError};
use ash::vk;
use std::rc::Rc;

pub struct DescriptorPool {
    pub inner: vk::DescriptorPool,
    device: Rc<Device>,
}

impl DescriptorPool {
    pub fn new(
        device: Rc<Device>,
        pool_sizes: &[vk::DescriptorPoolSize],
        max_sets: u32,
    ) -> Result<DescriptorPool, VulkanError> {
        let pool_info = vk::DescriptorPoolCreateInfo {
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
            max_sets,
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

    pub fn allocate_sets(
        &self,
        count: u32,
        layout: vk::DescriptorSetLayout,
    ) -> Result<Vec<DescriptorSet>, VulkanError> {
        let layouts = vec![layout; count as usize];

        let alloc_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool: self.inner,
            descriptor_set_count: count,
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

pub struct DescriptorSetLayout {
    pub inner: vk::DescriptorSetLayout,
    device: Rc<Device>,
}

impl DescriptorSetLayout {
    pub fn new(device: Rc<Device>, bindings: &[vk::DescriptorSetLayoutBinding]) -> Result<Self, VulkanError> {
        let create_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
            ..Default::default()
        };

        let inner = unsafe {
            device
                .inner
                .create_descriptor_set_layout(&create_info, None)
                .map_to_err("Cannot create descriptor set layout")?
        };

        Ok(Self { inner, device })
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe { self.device.inner.destroy_descriptor_set_layout(self.inner, None) }
    }
}
