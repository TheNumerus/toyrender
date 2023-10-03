use crate::vulkan::{CommandPool, Device, Vertex, VertexIndexBuffer, VulkanError};
use nalgebra_glm::{vec3, vec4};
use std::rc::Rc;

pub struct Mesh {
    pub buf: VertexIndexBuffer,
    pub indices_offset: u64,
    pub index_count: usize,
}

impl Mesh {
    pub fn new(
        device: Rc<Device>,
        cmd_pool: &CommandPool,
        vertices: &[Vertex],
        indices: &[u32],
    ) -> Result<Self, VulkanError> {
        let total_size = vertices.len() * std::mem::size_of::<Vertex>() + indices.len() * std::mem::size_of::<u32>();

        let mut data = Vec::with_capacity(total_size);

        data.resize(total_size, 0);

        let indices_offset = vertices.len() as u64 * std::mem::size_of::<Vertex>() as u64;

        data[0..indices_offset as usize].copy_from_slice(unsafe {
            std::slice::from_raw_parts(
                vertices.as_ptr() as *const u8,
                vertices.len() * std::mem::size_of::<Vertex>(),
            )
        });
        data[indices_offset as usize..].copy_from_slice(unsafe {
            std::slice::from_raw_parts(
                indices.as_ptr() as *const u8,
                indices.len() * std::mem::size_of::<u32>(),
            )
        });

        let buf = VertexIndexBuffer::new(device, cmd_pool, &data)?;

        Ok(Self {
            buf,
            indices_offset,
            index_count: indices.len(),
        })
    }
}

#[allow(dead_code)]
pub fn triangle() -> ([Vertex; 3], [u32; 3]) {
    (
        [
            Vertex {
                pos: vec3(0.0, -0.5, 0.0),
                color: vec4(1.0, 0.0, 0.0, 0.0),
                uv: Default::default(),
            },
            Vertex {
                pos: vec3(0.5, 0.5, 0.0),
                color: vec4(0.0, 1.0, 0.0, 0.0),
                uv: Default::default(),
            },
            Vertex {
                pos: vec3(-0.5, 0.5, 0.0),
                color: vec4(0.0, 0.0, 1.0, 0.0),
                uv: Default::default(),
            },
        ],
        [0, 1, 2],
    )
}

#[allow(dead_code)]
pub fn square() -> ([Vertex; 4], [u32; 6]) {
    (
        [
            Vertex {
                pos: vec3(-0.5, -0.5, 0.0),
                color: vec4(1.0, 0.0, 0.0, 0.0),
                uv: Default::default(),
            },
            Vertex {
                pos: vec3(0.5, 0.5, 0.0),
                color: vec4(0.0, 1.0, 0.0, 0.0),
                uv: Default::default(),
            },
            Vertex {
                pos: vec3(-0.5, 0.5, 0.0),
                color: vec4(0.0, 0.0, 1.0, 0.0),
                uv: Default::default(),
            },
            Vertex {
                pos: vec3(0.5, -0.5, 0.0),
                color: vec4(1.0, 1.0, 0.0, 0.0),
                uv: Default::default(),
            },
        ],
        [0, 1, 2, 1, 0, 3],
    )
}
