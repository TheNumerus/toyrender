use crate::vulkan::{Device, IntoVulkanError, VulkanError};
use ash::vk;
use std::rc::Rc;

pub struct ShaderModule {
    pub inner: vk::ShaderModule,
    device: Rc<Device>,
}

impl ShaderModule {
    pub fn new(bytecode: &[u8], device: Rc<Device>) -> Result<Self, VulkanError> {
        let create_info = vk::ShaderModuleCreateInfo {
            code_size: bytecode.len(),
            p_code: bytecode.as_ptr() as *const u32,
            ..Default::default()
        };

        let inner = unsafe {
            device
                .inner
                .create_shader_module(&create_info, None)
                .map_to_err("cannot create shader module")?
        };

        Ok(Self { inner, device })
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe { self.device.inner.destroy_shader_module(self.inner, None) }
    }
}
