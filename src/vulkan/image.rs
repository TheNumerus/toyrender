use crate::vulkan::{DebugMarker, Device, IntoVulkanError, VulkanError};
use ash::vk;
use ash::vk::{Handle, Image as RawImage};
use std::rc::Rc;

pub struct Image {
    pub inner: RawImage,
    pub memory: vk::DeviceMemory,
    device: Rc<Device>,
}

impl Image {
    pub fn new(
        device: Rc<Device>,
        format: vk::Format,
        extent: vk::Extent3D,
        usage: vk::ImageUsageFlags,
    ) -> Result<Self, VulkanError> {
        let create_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage,
            initial_layout: vk::ImageLayout::UNDEFINED,
            ..Default::default()
        };

        let inner = unsafe {
            device
                .inner
                .create_image(&create_info, None)
                .map_to_err("Cannot create vertex buffer")?
        };

        let mem_req = unsafe { device.inner.get_image_memory_requirements(inner) };

        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size: mem_req.size,
            memory_type_index: device
                .find_memory_type_index(mem_req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
                .ok_or(VulkanError {
                    msg: "Cannot find memory type".into(),
                    code: vk::Result::ERROR_UNKNOWN,
                })?,
            ..Default::default()
        };

        let memory = unsafe {
            device
                .inner
                .allocate_memory(&alloc_info, None)
                .map_to_err("Cannot allocate memory")?
        };

        unsafe {
            device
                .inner
                .bind_image_memory(inner, memory, 0)
                .map_to_err("Cannot bind memory to buffer")
        }?;

        Ok(Self { inner, memory, device })
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            self.device.inner.destroy_image(self.inner, None);
            self.device.inner.free_memory(self.memory, None);
        }
    }
}

impl DebugMarker for Image {
    fn device(&self) -> &Rc<Device> {
        &self.device
    }

    fn object_type(&self) -> vk::ObjectType {
        vk::ObjectType::IMAGE
    }

    fn handle(&self) -> u64 {
        self.inner.as_raw()
    }
}
