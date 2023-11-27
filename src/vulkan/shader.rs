use crate::vulkan::{Device, IntoVulkanError, VulkanError};
use ash::vk;
use std::ffi::CString;
use std::rc::Rc;

pub struct ShaderModule {
    pub inner: vk::ShaderModule,
    pub stage: ShaderStage,
    device: Rc<Device>,
    entry: CString,
}

impl ShaderModule {
    pub fn new(
        bytecode: &[u8],
        device: Rc<Device>,
        shader_stage: ShaderStage,
        entry: Option<String>,
    ) -> Result<Self, VulkanError> {
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

        let entry = match entry {
            Some(e) => CString::new(e).expect("Unexpected zero byte in entry name"),
            None => CString::new("main").unwrap(),
        };

        Ok(Self {
            inner,
            stage: shader_stage,
            device,
            entry,
        })
    }

    pub fn stage_info(&self) -> vk::PipelineShaderStageCreateInfo {
        let stage = match self.stage {
            ShaderStage::Fragment => vk::ShaderStageFlags::FRAGMENT,
            ShaderStage::Vertex => vk::ShaderStageFlags::VERTEX,
        };

        vk::PipelineShaderStageCreateInfo {
            stage,
            module: self.inner,
            p_name: self.entry.as_ptr(),
            ..Default::default()
        }
    }
}

impl Drop for ShaderModule {
    fn drop(&mut self) {
        unsafe { self.device.inner.destroy_shader_module(self.inner, None) }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum ShaderStage {
    Fragment,
    Vertex,
}
