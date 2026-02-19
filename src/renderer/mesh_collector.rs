use crate::scene::Scene;
use nalgebra_glm::{Mat4, vec4};
use std::collections::BTreeMap;

pub struct MeshCollector {}

pub struct DrawData {
    pub mesh_id: u64,
    pub count: u32,
    pub offset: u32,
}

pub struct CollectedResult {
    pub data: Vec<u8>,
    pub draws: Vec<DrawData>,
}

impl MeshCollector {
    pub fn collect_transforms(scene: &Scene, culling: bool, view: &Mat4, proj: &Mat4) -> CollectedResult {
        let mut transforms = BTreeMap::new();

        for mesh in scene.meshes.iter() {
            let id = mesh.resource.id;

            if culling {
                let viewmodel = view * mesh.transform;

                let min_view_pos = viewmodel
                    * vec4(
                        mesh.resource.culling_info.bb_min.x,
                        mesh.resource.culling_info.bb_min.y,
                        mesh.resource.culling_info.bb_min.z,
                        1.0,
                    );

                let max_view_pos = viewmodel
                    * vec4(
                        mesh.resource.culling_info.bb_max.x,
                        mesh.resource.culling_info.bb_max.y,
                        mesh.resource.culling_info.bb_max.z,
                        1.0,
                    );

                let center = (max_view_pos + min_view_pos) * 0.5;
                let radius = (max_view_pos - min_view_pos).magnitude();

                if center.z > radius {
                    continue;
                }
            }

            let entry = transforms.entry(id).or_insert_with(|| Vec::with_capacity(1));

            entry.push((mesh.transform, mesh.inverse));
        }

        let count = transforms.values().map(|v| v.len()).sum();

        let mut data = Vec::with_capacity(count * 2 * size_of::<Mat4>());

        let mut index = 0;
        let mut draws = Vec::with_capacity(count);
        for (key, value) in transforms.iter() {
            for (transform, inverse) in value {
                data.extend_from_slice(unsafe {
                    std::slice::from_raw_parts(transform as *const Mat4 as *const u8, size_of::<Mat4>())
                });
                data.extend_from_slice(unsafe {
                    std::slice::from_raw_parts(inverse as *const Mat4 as *const u8, size_of::<Mat4>())
                });
            }

            draws.push(DrawData {
                mesh_id: *key,
                count: value.len() as u32,
                offset: index as u32,
            });
            index += value.len();
        }

        CollectedResult { data, draws }
    }
}
