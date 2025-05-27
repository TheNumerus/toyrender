use crate::err::AppError;
use crate::err::AppError::VulkanAllocatorError;
use crate::vulkan::{Device, IntoVulkanError, VulkanError};
use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

pub struct Buffer {
    pub inner: vk::Buffer,
    allocation: Option<Allocation>,
    device: Rc<Device>,
    allocator: Arc<Mutex<Allocator>>,
}

impl Buffer {
    pub fn new(
        device: Rc<Device>,
        allocator: Arc<Mutex<Allocator>>,
        location: MemoryLocation,
        usage: vk::BufferUsageFlags,
        size: u64,
    ) -> Result<Self, AppError> {
        let info = vk::BufferCreateInfo {
            size,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let inner = unsafe {
            device
                .inner
                .create_buffer(&info, None)
                .map_to_err("Cannot create vertex buffer")?
        };

        let requirements = unsafe { device.inner.get_buffer_memory_requirements(inner) };

        let allocation = allocator
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name: "TEST",
                requirements,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(VulkanAllocatorError)?;

        unsafe {
            device
                .inner
                .bind_buffer_memory(inner, allocation.memory(), allocation.offset())
                .map_to_err("Cannot bind memory to buffer")
        }?;

        Ok(Self {
            inner,
            device,
            allocation: Some(allocation),
            allocator,
        })
    }

    pub fn new_with_alignment(
        device: Rc<Device>,
        allocator: Arc<Mutex<Allocator>>,
        location: MemoryLocation,
        usage: vk::BufferUsageFlags,
        size: u64,
        alignment: u64,
    ) -> Result<Self, AppError> {
        let info = vk::BufferCreateInfo {
            size,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let inner = unsafe {
            device
                .inner
                .create_buffer(&info, None)
                .map_to_err("Cannot create vertex buffer")?
        };

        let mut requirements = unsafe { device.inner.get_buffer_memory_requirements(inner) };
        requirements.alignment = alignment;

        let allocation = allocator
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name: "TEST",
                requirements,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(VulkanAllocatorError)?;

        unsafe {
            device
                .inner
                .bind_buffer_memory(inner, allocation.memory(), allocation.offset())
                .map_to_err("Cannot bind memory to buffer")
        }?;

        Ok(Self {
            inner,
            device,
            allocation: Some(allocation),
            allocator,
        })
    }

    pub fn fill_host(&mut self, data: &[u8]) -> Result<(), VulkanError> {
        let slice = self.allocation.as_mut().unwrap().mapped_slice_mut().unwrap();

        let (left, _) = slice.split_at_mut(data.len());
        left.copy_from_slice(data);

        Ok(())
    }

    pub fn get_device_addr(&self) -> vk::DeviceAddress {
        let addr_info = vk::BufferDeviceAddressInfo {
            buffer: self.inner,
            ..Default::default()
        };

        unsafe { self.device.inner.get_buffer_device_address(&addr_info) }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        self.allocator
            .lock()
            .unwrap()
            .free(self.allocation.take().unwrap())
            .unwrap();
        unsafe { self.device.inner.destroy_buffer(self.inner, None) }
    }
}
