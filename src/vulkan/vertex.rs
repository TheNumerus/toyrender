use crate::vulkan::{Buffer, CommandPool, Device, IntoVulkanError, VulkanError};
use ash::vk;
use nalgebra_glm::{Vec2, Vec3, Vec4};
use std::rc::Rc;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Vertex {
    pub pos: Vec3,
    pub normal: Vec3,
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

    pub fn attribute_description() -> [vk::VertexInputAttributeDescription; 4] {
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
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 3 * std::mem::size_of::<f32>() as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 2,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 6 * std::mem::size_of::<f32>() as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 3,
                format: vk::Format::R32G32_SFLOAT,
                offset: 10 * std::mem::size_of::<f32>() as u32,
            },
        ]
    }
}

pub struct VertexIndexBuffer {
    pub inner: Buffer,
}

impl VertexIndexBuffer {
    pub fn new(device: Rc<Device>, cmd_pool: &CommandPool, data: &[u8]) -> Result<Self, VulkanError> {
        let staging = Buffer::new(
            device.clone(),
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            data.len() as u64,
            false,
            false,
        )?;

        unsafe {
            staging.fill_host(data)?;
        }

        let inner = Buffer::new(
            device.clone(),
            vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            data.len() as u64,
            false,
            false,
        )?;

        let cmd_buf = cmd_pool.allocate_cmd_buffers(1)?.pop().unwrap();
        cmd_buf.begin_one_time()?;

        unsafe {
            let region = vk::BufferCopy {
                size: data.len() as u64,
                src_offset: 0,
                dst_offset: 0,
            };

            device
                .inner
                .cmd_copy_buffer(cmd_buf.inner, staging.inner, inner.inner, &[region]);
        }

        cmd_buf.end()?;

        let submit_info = vk::SubmitInfo {
            command_buffer_count: 1,
            p_command_buffers: &cmd_buf.inner,
            ..Default::default()
        };

        unsafe {
            device
                .inner
                .queue_submit(device.graphics_queue, &[submit_info], vk::Fence::null())
                .map_to_err("Cannot submit queue")?;
            device
                .inner
                .queue_wait_idle(device.graphics_queue)
                .map_to_err("Cannot wait idle")?;
        }

        Ok(Self { inner })
    }
}
