use crate::app::frame_stats::FrameReport;
use crate::renderer::stats;
use crate::scene::Scene;
use nalgebra_glm::{Mat4, Vec3, vec3, vec4};
use std::collections::BTreeMap;

pub struct MeshCollector {}

pub struct DrawData {
    pub primitive_id: u64,
    pub mesh_id: u64,
    pub count: u32,
    pub offset: u32,
}

#[repr(C)]
pub struct RasterMeshInstanceDataGPU {
    pub model: Mat4,
    pub inverse: Mat4,
    pub base_color: [f32; 3],
    pub roughness: f32,
    pub is_flipped: i32,
    pub _pad_0: [i32; 3],
}

pub struct CollectedResult {
    pub data: Vec<RasterMeshInstanceDataGPU>,
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

            visible += 1;

            for primitive in &mesh.resource.primitives {
                let id = primitive.id;
                let (_, entry) = transforms
                    .entry(id)
                    .or_insert_with(|| (mesh.resource.id, Vec::with_capacity(1)));
                entry.push((mesh.transform, mesh.inverse, primitive.material));
            }
        }

        let count = transforms.values().map(|(_, v)| v.len()).sum();

        report.log::<stats::CullPercentageStat>((1.0 - (visible as f32 / total as f32)) * 100.0);
        report.log::<stats::InstanceCountStat>(visible as u32);

        let mut data = Vec::with_capacity(count);

        let mut index = 0;
        let mut draws = Vec::with_capacity(count);
        for (key, (mesh_id, value)) in transforms.iter() {
            for (transform, inverse, mat) in value {
                let is_flipped = if transform.view((0, 0), (3, 3)).determinant() > 0.0 {
                    0_i32
                } else {
                    1_i32
                };

                let instance_data = RasterMeshInstanceDataGPU {
                    model: *transform,
                    inverse: *inverse,
                    base_color: mat.base_color.data.0[0],
                    roughness: mat.roughness,
                    is_flipped,
                    _pad_0: [0; 3],
                };
                data.push(instance_data);
            }

            draws.push(DrawData {
                primitive_id: *key,
                mesh_id: *mesh_id,
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
