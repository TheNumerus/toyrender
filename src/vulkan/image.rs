use crate::err::AppError;
use crate::err::AppError::VulkanAllocatorError;
use crate::vulkan::{DebugMarker, Device, IntoVulkanError};
use ash::vk;
use ash::vk::{Handle, Image as RawImage};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

pub struct Image {
    pub inner: RawImage,
    allocation: Option<Allocation>,
    device: Rc<Device>,
    pub allocator: Arc<Mutex<Allocator>>,
}

impl Image {
    pub fn new(
        device: Rc<Device>,
        allocator: Arc<Mutex<Allocator>>,
        format: vk::Format,
        extent: vk::Extent3D,
        usage: vk::ImageUsageFlags,
    ) -> Result<Self, AppError> {
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

        let requirements = unsafe { device.inner.get_image_memory_requirements(inner) };

        let allocation = allocator
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name: "Image",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(VulkanAllocatorError)?;

        unsafe {
            device
                .inner
                .bind_image_memory(inner, allocation.memory(), allocation.offset())
                .map_to_err("Cannot bind memory to buffer")
        }?;

        Ok(Self {
            inner,
            allocation: Some(allocation),
            device,
            allocator,
        })
    }

    pub fn single_color_layer_range() -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        }
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            self.device.inner.destroy_image(self.inner, None);
            self.allocator
                .lock()
                .unwrap()
                .free(self.allocation.take().unwrap())
                .unwrap();
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
