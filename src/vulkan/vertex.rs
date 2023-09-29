use crate::vulkan::{Device, IntoVulkanError, VulkanError};
use ash::vk;
use nalgebra_glm::{Vec2, Vec3, Vec4};
use std::ffi::c_void;
use std::rc::Rc;

#[repr(C)]
pub struct Vertex {
    pub pos: Vec3,
    pub color: Vec4,
    pub uv: Vec2,
}

impl Vertex {
    pub fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: std::mem::size_of::<Self>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }
    }

    pub fn attribute_description() -> [vk::VertexInputAttributeDescription; 3] {
        [
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 3 * std::mem::size_of::<f32>() as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 2,
                format: vk::Format::R32G32_SFLOAT,
                offset: 7 * std::mem::size_of::<f32>() as u32,
            },
        ]
    }
}

pub struct VertexBuffer {
    pub inner: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub vertices: usize,
    device: Rc<Device>,
}

impl VertexBuffer {
    pub fn new(device: Rc<Device>, vertices: usize) -> Result<Self, VulkanError> {
        let info = vk::BufferCreateInfo {
            size: vertices as u64 * std::mem::size_of::<Vertex>() as u64,
            usage: vk::BufferUsageFlags::VERTEX_BUFFER,
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

        let alloc_info = vk::MemoryAllocateInfo {
            allocation_size: mem_req.size,
            memory_type_index: device
                .find_memory_type_index(
                    mem_req.memory_type_bits,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
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
                .bind_buffer_memory(inner, memory, 0)
                .map_to_err("Cannot bind memory to buffer")
        }?;

        Ok(Self {
            inner,
            vertices,
            device,
            memory,
        })
    }

    pub fn fill(&self, data: &[Vertex]) -> Result<(), VulkanError> {
        unsafe {
            let size = self.vertices as u64 * std::mem::size_of::<Vertex>() as u64;

            let ptr = self
                .device
                .inner
                .map_memory(self.memory, 0, size, vk::MemoryMapFlags::empty())
                .map_to_err("Cannot map memory")?;

            ptr.copy_from(data.as_ptr() as *const c_void, size as usize);

            self.device.inner.unmap_memory(self.memory);
        }

        Ok(())
    }
}

impl Drop for VertexBuffer {
    fn drop(&mut self) {
        unsafe { self.device.inner.destroy_buffer(self.inner, None) }
        unsafe { self.device.inner.free_memory(self.memory, None) }
    }
}
