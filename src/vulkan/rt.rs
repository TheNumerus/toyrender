use crate::vulkan::{Device, IntoVulkanError, VulkanError};
use ash::vk;
use std::rc::Rc;

#[derive(Clone)]
pub struct RtDescriptorSetLayout {
    pub inner: vk::DescriptorSetLayout,
    device: Rc<Device>,
}

impl RtDescriptorSetLayout {
    pub fn new(device: Rc<Device>, binding: u32, stages: vk::ShaderStageFlags) -> Result<Self, VulkanError> {
        let binding = vk::DescriptorSetLayoutBinding {
            binding,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: stages,
            ..Default::default()
        };

        let create_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: 1,
            p_bindings: &binding,
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

impl Drop for RtDescriptorSetLayout {
    fn drop(&mut self) {
        unsafe { self.device.inner.destroy_descriptor_set_layout(self.inner, None) }
    }
}
