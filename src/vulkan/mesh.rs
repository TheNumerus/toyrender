use crate::err::AppError;
use crate::mesh::MeshResource;
use crate::vulkan::{CommandPool, Device, Vertex, VertexIndexBuffer};
use gpu_allocator::vulkan::Allocator;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

pub struct VulkanMesh {
    pub buf: VertexIndexBuffer,
    pub indices_offset: u64,
    pub index_count: usize,
}

impl VulkanMesh {
    pub fn new(
        device: Rc<Device>,
        allocator: Arc<Mutex<Allocator>>,
        cmd_pool: &CommandPool,
        mesh: &MeshResource,
    ) -> Result<Self, AppError> {
        let vertices = &mesh.vertices;
        let indices = mesh.indices.to_vec_u32();

        let total_size = std::mem::size_of_val(vertices.as_slice()) + std::mem::size_of_val(indices.as_slice());

        let mut data = vec![0; total_size];

        let indices_offset = vertices.len() as u64 * std::mem::size_of::<Vertex>() as u64;

        data[0..indices_offset as usize].copy_from_slice(unsafe {
            std::slice::from_raw_parts(
                vertices.as_ptr() as *const u8,
                std::mem::size_of_val(vertices.as_slice()),
            )
        });
        data[indices_offset as usize..].copy_from_slice(unsafe {
            std::slice::from_raw_parts(indices.as_ptr() as *const u8, std::mem::size_of_val(indices.as_slice()))
        });

        let buf = VertexIndexBuffer::new(device, allocator, cmd_pool, &data)?;

        Ok(Self {
            buf,
            indices_offset,
            index_count: indices.len(),
        })
    }
}
