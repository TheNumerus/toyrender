use crate::vulkan::{Device, IntoVulkanError, VulkanError};
use ash::vk;
use ash::vk::Handle;
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
        name: Option<String>,
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

        if let Some(name) = name {
            let name_ptr = CString::new(name).unwrap();
            let name_info = vk::DebugUtilsObjectNameInfoEXT {
                object_type: vk::ObjectType::SHADER_MODULE,
                object_handle: inner.as_raw(),
                p_object_name: name_ptr.as_ptr(),
                ..Default::default()
            };

            device.name_object(name_info)?;
        }

        let entry = CString::new("main").unwrap();

        Ok(Self {
            inner,
            stage: shader_stage,
            device,
            entry,
        })
    }

    pub fn stage_info(&self) -> vk::PipelineShaderStageCreateInfo {
        vk::PipelineShaderStageCreateInfo {
            stage: self.stage.into(),
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
    RayMiss,
    RayGen,
    RayClosestHit,
    Compute,
}

impl From<ShaderStage> for vk::ShaderStageFlags {
    fn from(value: ShaderStage) -> Self {
        match value {
            ShaderStage::Fragment => Self::FRAGMENT,
            ShaderStage::Vertex => Self::VERTEX,
            ShaderStage::Compute => Self::COMPUTE,
            ShaderStage::RayGen => Self::RAYGEN_KHR,
            ShaderStage::RayMiss => Self::MISS_KHR,
            ShaderStage::RayClosestHit => Self::CLOSEST_HIT_KHR,
        }
    }
}
