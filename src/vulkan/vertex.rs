use crate::vulkan::{Device, IntoVulkanError, VulkanError};
use ash::vk;
use ash::vk::Buffer;
use nalgebra_glm::{vec3, vec4, Vec2, Vec3, Vec4};
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
                offset: 3,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 2,
                format: vk::Format::R32G32_SFLOAT,
                offset: 7,
            },
        ]
    }

    pub fn triangle() -> [Self; 3] {
        [
            Self {
                pos: vec3(0.0, -0.5, 0.0),
                color: vec4(1.0, 0.0, 0.0, 0.0),
                uv: Default::default(),
            },
            Self {
                pos: vec3(0.5, 0.5, 0.0),
                color: vec4(0.0, 1.0, 0.0, 0.0),
                uv: Default::default(),
            },
            Self {
                pos: vec3(-0.5, 0.5, 0.0),
                color: vec4(0.0, 0.0, 1.0, 0.0),
                uv: Default::default(),
            },
        ]
    }
}

pub struct VertexBuffer {
    pub inner: Buffer,
    device: Rc<Device>,
}

impl VertexBuffer {
    pub fn new(device: Rc<Device>, size: usize) -> Result<Self, VulkanError> {
        let info = vk::BufferCreateInfo {
            size: size as u64,
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

        Ok(Self { inner, device })
    }

    pub fn get_memory_requirements(&self) -> vk::MemoryRequirements {
        unsafe { self.device.inner.get_buffer_memory_requirements(self.inner) }
    }
}

impl Drop for VertexBuffer {
    fn drop(&mut self) {
        unsafe { self.device.inner.destroy_buffer(self.inner, None) }
    }
}
