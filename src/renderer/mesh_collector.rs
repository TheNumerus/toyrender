use crate::app::frame_stats::FrameReport;
use crate::renderer::stats;
use crate::scene::Scene;
use nalgebra_glm::{Mat4, Vec3, vec3, vec4};
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
    pub fn collect_transforms(
        scene: &Scene,
        culling: bool,
        view: &Mat4,
        proj_inverse: &Mat4,
        report: &mut FrameReport,
    ) -> CollectedResult {
        let mut transforms = BTreeMap::new();

        let frustum_normals = Self::compute_frustum_normals(proj_inverse);
        let mut total = 0;
        let mut visible = 0;

        'mesh: for mesh in scene.meshes.iter() {
            let id = mesh.resource.id;

            if !mesh.visible {
                continue 'mesh;
            }

            total += 1;

            if culling {
                let viewmodel = view * mesh.transform;

                let min_view_pos = (viewmodel * mesh.resource.culling_info.bb_min.insert_row(3, 1.0)).xyz();
                let max_view_pos = (viewmodel * mesh.resource.culling_info.bb_max.insert_row(3, 1.0)).xyz();

                let center = (max_view_pos + min_view_pos) * 0.5;
                let radius = (max_view_pos - min_view_pos).magnitude();

                for norm in &frustum_normals {
                    let dist = norm.dot(&center);

                    if dist > radius {
                        continue 'mesh;
                    }
                }
            }

            let entry = transforms.entry(id).or_insert_with(|| Vec::with_capacity(1));

            visible += 1;
            entry.push((mesh.transform, mesh.inverse));
        }

        let count = transforms.values().map(|v| v.len()).sum();

        report.log::<stats::CullPercentageStat>((1.0 - (visible as f32 / total as f32)) * 100.0);
        report.log::<stats::InstanceCountStat>(visible as u32);

        let mut data = Vec::with_capacity(count * (2 * size_of::<Mat4>() + size_of::<i32>() * 4));

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

                let is_flipped = if transform.view((0, 0), (3, 3)).determinant() > 0.0 {
                    0_i32
                } else {
                    1_i32
                };

                data.extend_from_slice(&is_flipped.to_le_bytes());

                // padding
                data.extend_from_slice(&[0; 12]);
            }

            draws.push(DrawData {
                mesh_id: *key,
                count: value.len() as u32,
                offset: index as u32,
            });
            index += value.len();
        }

        report.log::<stats::DrawCallStat>(transforms.len() as u32);

        CollectedResult { data, draws }
    }

    fn compute_frustum_normals(proj_inverse: &Mat4) -> [Vec3; 4] {
        let corner = vec4(-1.0, -1.0, 1.0, 1.0);
        let corner_mapped = proj_inverse * corner;
        let corner_mapped = corner_mapped.xyz() * corner_mapped.w;

        // corners are symmetrical in view space, this way only one matrix multiply needs to be done
        let mapped_corners = [
            corner_mapped,
            vec3(-corner_mapped.x, corner_mapped.y, corner_mapped.z),
            vec3(-corner_mapped.x, -corner_mapped.y, corner_mapped.z),
            vec3(corner_mapped.x, -corner_mapped.y, corner_mapped.z),
        ];

        [
            mapped_corners[1].cross(&mapped_corners[0]).normalize(),
            mapped_corners[2].cross(&mapped_corners[1]).normalize(),
            mapped_corners[3].cross(&mapped_corners[2]).normalize(),
            mapped_corners[0].cross(&mapped_corners[3]).normalize(),
        ]
    }
}
