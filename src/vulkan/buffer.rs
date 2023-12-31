use crate::vulkan::{Device, IntoVulkanError, VulkanError};
use ash::vk;
use std::ffi::c_void;
use std::rc::Rc;

pub struct Buffer {
    pub inner: vk::Buffer,
    pub memory: vk::DeviceMemory,
    persistent_ptr: Option<*mut c_void>,
    get_address: bool,
    device: Rc<Device>,
}

impl Buffer {
    pub fn new(
        device: Rc<Device>,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
        size: u64,
        is_persistent: bool,
        get_address: bool,
    ) -> Result<Self, VulkanError> {
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

        let mem_req = unsafe { device.inner.get_buffer_memory_requirements(inner) };

        let flags = vk::MemoryAllocateFlagsInfo {
            flags: vk::MemoryAllocateFlags::DEVICE_ADDRESS,
            ..Default::default()
        };

        let next = if get_address {
            std::ptr::addr_of!(flags)
        } else {
            std::ptr::null()
        };

        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size: mem_req.size,
            memory_type_index: device
                .find_memory_type_index(mem_req.memory_type_bits, properties)
                .ok_or(VulkanError {
                    msg: "Cannot find memory type".into(),
                    code: vk::Result::ERROR_UNKNOWN,
                })?,
            p_next: next as *const c_void,
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
                .bind_buffer_memory(inner, memory, 0)
                .map_to_err("Cannot bind memory to buffer")
        }?;

        let persistent_ptr = if is_persistent {
            unsafe {
                Some(
                    device
                        .inner
                        .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
                        .map_to_err("Cannot map memory")?,
                )
            }
        } else {
            None
        };

        Ok(Self {
            inner,
            device,
            memory,
            persistent_ptr,
            get_address,
        })
    }

    pub unsafe fn fill_host(&self, data: &[u8]) -> Result<(), VulkanError> {
        let ptr = match self.persistent_ptr {
            Some(p) => p,
            None => self
                .device
                .inner
                .map_memory(self.memory, 0, data.len() as u64, vk::MemoryMapFlags::empty())
                .map_to_err("Cannot map memory")?,
        };

        ptr.copy_from(data.as_ptr() as *const c_void, data.len());

        if self.persistent_ptr.is_none() {
            self.device.inner.unmap_memory(self.memory);
        }

        Ok(())
    }

    pub fn get_device_addr(&self) -> Option<vk::DeviceAddress> {
        if self.get_address {
            let addr_info = vk::BufferDeviceAddressInfo {
                buffer: self.inner,
                ..Default::default()
            };

            let addr = unsafe { self.device.inner.get_buffer_device_address(&addr_info) };
            Some(addr)
        } else {
            None
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if self.persistent_ptr.is_some() {
            unsafe { self.device.inner.unmap_memory(self.memory) }
        }
        unsafe { self.device.inner.destroy_buffer(self.inner, None) }
        unsafe { self.device.inner.free_memory(self.memory, None) }
    }
}
