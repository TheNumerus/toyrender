use crate::err::AppError;
use crate::mesh::{Mesh, MeshInstance};
use crate::vulkan::{CommandPool, Device, Vertex};
use gltf::buffer::Data;
use gltf::mesh::Mode;
use gltf::{Accessor, Primitive, Semantic};
use nalgebra::Quaternion;
use nalgebra_glm::{inverse, quat_cast, vec3, Mat4, Vec2, Vec3, Vec4};
use std::rc::Rc;

pub fn extract_mesh_internal(mesh: gltf::Mesh, buffers: &[Data]) -> Result<(Vec<Vertex>, Vec<u32>), AppError> {
    let gltf_convert_matrix = Mat4::from_euler_angles(std::f32::consts::FRAC_PI_2, 0.0, 0.0);
    let gltf_normal_convert_matrix = Mat4::from_euler_angles(std::f32::consts::FRAC_PI_2, 0.0, 0.0);

    let mut vertices_total = Vec::new();
    let mut indices_total = Vec::new();
    let mut offset = 0;

    for primitive in mesh.primitives().filter(|p| p.mode() == Mode::Triangles) {
        let accessors = Accessors::new(&primitive, buffers)?;

        let mut vertices = Vec::with_capacity(accessors.len);

        for i in 0..accessors.len {
            let pos = (gltf_convert_matrix
                * Vec4::new(
                    accessors.pos[3 * i],
                    accessors.pos[3 * i + 1],
                    accessors.pos[3 * i + 2],
                    1.0,
                ))
            .xyz();
            let normal = (gltf_normal_convert_matrix
                * Vec4::new(
                    accessors.normal[3 * i],
                    accessors.normal[3 * i + 1],
                    accessors.normal[3 * i + 2],
                    0.0,
                ))
            .xyz();
            let color = match accessors.color {
                Some(ac) => Vec4::new(
                    ac[4 * i] as f32 / u16::MAX as f32,
                    ac[4 * i + 1] as f32 / u16::MAX as f32,
                    ac[4 * i + 2] as f32 / u16::MAX as f32,
                    ac[4 * i + 3] as f32 / u16::MAX as f32,
                ),
                None => Vec4::from_element(0.0),
            };
            let uv = match accessors.uv {
                Some(ac) => Vec2::new(ac[2 * i], ac[2 * i + 1]),
                None => Vec2::from_element(0.0),
            };

            let vertex = Vertex { pos, normal, color, uv };

            vertices.push(vertex);
        }

        let indices = accessors.indices.iter().map(|&a| a as u32 + offset).collect::<Vec<_>>();

        let len = vertices.len();

        vertices_total.extend(vertices);
        indices_total.extend(indices);

        offset += len as u32;
    }

    Ok((vertices_total, indices_total))
}

pub fn extract_scene(
    device: Rc<Device>,
    cmd_pool: &CommandPool,
    slice: &[u8],
) -> Result<(Vec<Rc<Mesh>>, Vec<MeshInstance>), AppError> {
    let (document, buffers, _images) = gltf::import_slice(slice)?;

    let mut meshes = Vec::new();
    let mut instances = Vec::new();

    let gltf_convert_matrix = Mat4::from_euler_angles(std::f32::consts::FRAC_PI_2, 0.0, 0.0);

    for mesh in document.meshes() {
        let (v, i) = extract_mesh_internal(mesh, &buffers)?;

        let m = Rc::new(Mesh::new(device.clone(), cmd_pool, &v, &i)?);

        meshes.push(m);
    }

    for instance in document.scenes().next().unwrap().nodes() {
        let mut ins = MeshInstance::new(meshes[instance.mesh().unwrap().index()].clone());

        let (pos, rot, scale) = instance.transform().decomposed();

        let quat = Quaternion::new(rot[3], rot[0], rot[1], rot[2]);

        ins.transform = Mat4::new_translation(&(Vec3::from(pos).xzy().component_mul(&vec3(1.0, -1.0, 1.0))))
            * gltf_convert_matrix
            * quat_cast(&quat)
            * inverse(&gltf_convert_matrix)
            * Mat4::new_nonuniform_scaling(&Vec3::from(scale));

        instances.push(ins);
    }

    Ok((meshes, instances))
}

struct Accessors<'a> {
    pos: &'a [f32],
    normal: &'a [f32],
    indices: &'a [u16],
    uv: Option<&'a [f32]>,
    color: Option<&'a [u16]>,
    len: usize,
}

impl<'a> Accessors<'a> {
    pub fn new(primitive: &'a Primitive, data: &[Data]) -> Result<Self, AppError> {
        let pos = primitive.get(&Semantic::Positions);
        let normal = primitive.get(&Semantic::Normals);
        let uv = primitive.get(&Semantic::TexCoords(0));
        let color = primitive.get(&Semantic::Colors(0));
        let indices = primitive.indices();

        if pos.is_none() || normal.is_none() || indices.is_none() {
            return Err(AppError::Import("No position/normals/indices found".into()));
        }

        let pos = Self::accessor_to_slice(pos.unwrap(), data);
        let normal = Self::accessor_to_slice(normal.unwrap(), data);
        let indices = Self::accessor_to_slice(indices.unwrap(), data);

        let uv = uv.map(|uv| Self::accessor_to_slice(uv, data));
        let color = color.map(|color| Self::accessor_to_slice(color, data));

        let len = pos.len() / 3;

        let mut lengths = vec![("normal", normal.len() / 3)];
        if let Some(a) = uv {
            lengths.push(("uv", a.len() / 2));
        }
        if let Some(a) = color {
            lengths.push(("color", a.len() / 4));
        }

        for (name, l) in lengths {
            if l != len {
                return Err(AppError::Import(format!(
                    "{name} input data length mismatch, expected {len}, got {l}"
                )));
            }
        }

        Ok(Self {
            pos,
            normal,
            indices,
            uv,
            color,
            len,
        })
    }

    fn accessor_to_slice<T>(accessor: Accessor<'a>, data: &[Data]) -> &'a [T] {
        let count = accessor.count();
        let size = accessor.size();
        let data_type = accessor.data_type();
        let accessor_offset = accessor.offset();

        let buffer_view = accessor.view().unwrap();
        let view_offset = buffer_view.offset();
        let pos = data.get(buffer_view.buffer().index()).unwrap().as_ptr() as *const T;
        let slice = unsafe {
            std::slice::from_raw_parts(
                pos.add((view_offset + accessor_offset) / data_type.size()),
                count * size / data_type.size(),
            )
        };

        slice
    }
}
