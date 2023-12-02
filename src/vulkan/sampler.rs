use crate::vulkan::{Device, IntoVulkanError, VulkanError};
use ash::vk;
use ash::vk::Sampler as RawSampler;
use std::rc::Rc;

pub struct Sampler {
    pub inner: RawSampler,
    device: Rc<Device>,
}

impl Sampler {
    pub fn new(device: Rc<Device>) -> Result<Self, VulkanError> {
        let sampler_info = vk::SamplerCreateInfo {
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::NEAREST,
            address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_w: vk::SamplerAddressMode::REPEAT,
            anisotropy_enable: vk::FALSE,
            compare_enable: vk::FALSE,
            compare_op: vk::CompareOp::ALWAYS,
            unnormalized_coordinates: vk::FALSE,
            ..Default::default()
        };

        let inner = unsafe {
            device
                .inner
                .create_sampler(&sampler_info, None)
                .map_to_err("Cannot create sampler")?
        };

        Ok(Self { inner, device })
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe { self.device.inner.destroy_sampler(self.inner, None) }
    }
}
