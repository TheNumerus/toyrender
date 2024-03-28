use crate::vulkan::{Device, IntoVulkanError, VulkanError};
use ash::vk;
use ash::vk::{Handle, Image as RawImage};
use std::ops::Not;
use std::rc::Rc;

pub struct Image {
    pub inner: RawImage,
    pub memory: vk::DeviceMemory,
    pub current_layout: vk::ImageLayout,
    pub usage: vk::ImageUsageFlags,
    device: Rc<Device>,
}

impl Image {
    pub fn new(
        device: Rc<Device>,
        name: &str,
        format: vk::Format,
        extent: vk::Extent3D,
        usage: vk::ImageUsageFlags,
    ) -> Result<Self, VulkanError> {
        let current_layout = vk::ImageLayout::UNDEFINED;

        let create_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            tiling: vk::ImageTiling::OPTIMAL,
            usage,
            initial_layout: current_layout,
            ..Default::default()
        };

        let inner = unsafe {
            device
                .inner
                .create_image(&create_info, None)
                .map_to_err("Cannot create image")?
        };

        device.name_object(name, inner.as_raw(), vk::ObjectType::IMAGE)?;

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

        Ok(Self {
            inner,
            memory,
            device,
            usage,
            current_layout,
        })
    }

    pub fn init_layout(&self) -> vk::ImageMemoryBarrier {
        let aspect = if self.usage.contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT) {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };

        vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::MEMORY_WRITE,
            old_layout: self.current_layout,
            new_layout: vk::ImageLayout::GENERAL,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: self.inner,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..Default::default()
        }
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
