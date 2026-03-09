use crate::err::AppError;
use crate::renderer::{TlasIndex, VulkanContext};
use crate::scene::Scene;
use crate::vulkan::{BottomLevelAs, CommandBuffer, IntoVulkanError, Vertex, VulkanMesh};
use ash::vk;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

pub struct MeshSubsystem {
    pub blases: BTreeMap<u64, BottomLevelAs>,
    pub meshes: BTreeMap<u64, VulkanMesh>,
    context: Rc<VulkanContext>,
}

impl MeshSubsystem {
    pub fn new(context: Rc<VulkanContext>) -> Self {
        Self {
            blases: BTreeMap::new(),
            meshes: BTreeMap::new(),
            context,
        }
    }

    pub fn prepare_meshes(&mut self, scene: &Scene, command_buffer: &CommandBuffer) -> Result<bool, AppError> {
        let mut changed = false;

        for instance in &scene.meshes {
            if let std::collections::btree_map::Entry::Vacant(e) = self.meshes.entry(instance.resource.id) {
                // only run command if needed
                if !changed {
                    command_buffer.reset()?;
                    command_buffer.begin_one_time()?;
                    changed = true;
                }

                let mesh = VulkanMesh::new_nonblocking(
                    self.context.device.clone(),
                    self.context.allocator.clone(),
                    command_buffer,
                    &instance.resource,
                )?;
                e.insert(mesh);
            }
        }

        if changed {
            command_buffer.end()?;

            let submit_info = vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: &command_buffer.inner,
                ..Default::default()
            };

            unsafe {
                self.context
                    .device
                    .inner
                    .queue_submit(self.context.device.compute_queue, &[submit_info], vk::Fence::null())
                    .map_to_err("Cannot submit queue")?;
                self.context
                    .device
                    .inner
                    .queue_wait_idle(self.context.device.compute_queue)
                    .map_to_err("Cannot wait idle")?;
            }

            for mesh in self.meshes.values_mut() {
                mesh.buf.finalize();
            }

            let mut geos = BTreeMap::new();
            let mut ranges = BTreeMap::new();

            let mut processed = BTreeSet::new();

            for mesh in &scene.meshes {
                if processed.contains(&mesh.resource.id) {
                    continue;
                }

                let res = self.meshes.get(&mesh.resource.id).unwrap();

                let addr = res.buf.inner.get_device_addr();
                let max_prim_count = res.index_count / 3;

                let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR {
                    vertex_format: vk::Format::R32G32B32_SFLOAT,
                    vertex_data: vk::DeviceOrHostAddressConstKHR { device_address: addr },
                    vertex_stride: std::mem::size_of::<Vertex>() as vk::DeviceSize,
                    max_vertex: mesh.resource.vertices.len() as u32 - 1,
                    index_type: vk::IndexType::UINT32,
                    index_data: vk::DeviceOrHostAddressConstKHR {
                        device_address: addr + (res.indices_offset),
                    },
                    ..Default::default()
                };

                let geo = vk::AccelerationStructureGeometryKHR {
                    geometry_type: vk::GeometryTypeKHR::TRIANGLES,
                    geometry: vk::AccelerationStructureGeometryDataKHR { triangles },
                    flags: vk::GeometryFlagsKHR::OPAQUE,
                    ..Default::default()
                };
                geos.insert(mesh.resource.id, geo);

                let range = vk::AccelerationStructureBuildRangeInfoKHR {
                    primitive_count: max_prim_count as u32,
                    primitive_offset: 0,
                    first_vertex: 0,
                    transform_offset: 0,
                };
                ranges.insert(mesh.resource.id, range);
                processed.insert(mesh.resource.id);
            }

            let batch = BottomLevelAs::batch_bottom_build(
                self.context.device.clone(),
                self.context.allocator.clone(),
                self.context.rt_acc_struct_ext.clone(),
                &self.context.compute_command_pool,
                ranges,
                geos,
                vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                    | vk::BuildAccelerationStructureFlagsKHR::ALLOW_DATA_ACCESS,
            )?;

            self.blases = batch;
        }

        Ok(changed)
    }

    pub(crate) fn build_tlas_index(&self, scene: &Scene) -> TlasIndex {
        let mut index = Vec::with_capacity(scene.meshes.len());

        for mesh in &scene.meshes {
            let transform = mesh.transform;
            let id = mesh.resource.id;

            if !mesh.visible {
                continue;
            }

            index.push((id, transform));
        }

        TlasIndex { index }
    }
}
