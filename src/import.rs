use crate::err::AppError;
use crate::math;
use crate::mesh::{Indices, MeshCullingInfo, MeshInstance, MeshResource};
use crate::vulkan::Vertex;
use gltf::buffer::Data;
use gltf::json::accessor::ComponentType;
use gltf::mesh::Mode;
use gltf::{Accessor, Gltf, Primitive, Semantic};
use log::info;
use nalgebra::{Point3, Quaternion, Rotation3};
use nalgebra_glm::{Mat4, Vec2, Vec3, Vec4, inverse, quat_cast, vec3};
use std::rc::Rc;

fn get_mesh_indices_type(mesh: &gltf::Mesh, buffers: &[Data]) -> Result<IndexType, AppError> {
    let mut total = 0;

    for primitive in mesh.primitives().filter(|p| p.mode() == Mode::Triangles) {
        let accessors = Accessors::new(&primitive, buffers)?;

        match accessors.indices {
            IndicesAccessor::U16(i) => {
                total += i.len();
            }
            IndicesAccessor::U32(_) => {
                return Ok(IndexType::U32);
            }
        }
    }

    if total >= u16::MAX as usize {
        return Ok(IndexType::U32);
    }

    Ok(IndexType::U16)
}

fn extract_mesh(mesh: gltf::Mesh, buffers: &[Data]) -> Result<MeshResource, AppError> {
    let gltf_convert_matrix = Mat4::from_euler_angles(std::f32::consts::FRAC_PI_2, 0.0, 0.0);
    let gltf_normal_convert_matrix = Mat4::from_euler_angles(std::f32::consts::FRAC_PI_2, 0.0, 0.0);

    let mut vertices_total = Vec::new();
    let indices_type = get_mesh_indices_type(&mesh, buffers)?;
    let mut indices_total = match indices_type {
        IndexType::U16 => Indices::U16(Vec::new()),
        IndexType::U32 => Indices::U32(Vec::new()),
    };

    let mut culling_info = MeshCullingInfo {
        bb_min: vec3(0.0, 0.0, 0.0),
        bb_max: vec3(0.0, 0.0, 0.0),
    };

    let mut min_pos: Vec3 = vec3(f32::MAX, f32::MAX, f32::MAX);
    let mut max_pos: Vec3 = vec3(f32::MIN, f32::MIN, f32::MIN);

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
            min_pos.x = min_pos.x.min(pos.x);
            min_pos.y = min_pos.y.min(pos.y);
            min_pos.z = min_pos.z.min(pos.z);
            max_pos.x = max_pos.x.max(pos.x);
            max_pos.y = max_pos.y.max(pos.y);
            max_pos.z = max_pos.z.max(pos.z);
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

        match accessors.indices {
            IndicesAccessor::U16(i) => match &mut indices_total {
                Indices::U16(t) => t.extend(i.iter().map(|&a| a + offset as u16)),
                Indices::U32(t) => t.extend(i.iter().map(|&a| a as u32 + offset)),
            },
            IndicesAccessor::U32(i) => match &mut indices_total {
                Indices::U16(t) => t.extend(i.iter().map(|&a| a as u16 + offset as u16)),
                Indices::U32(t) => t.extend(i.iter().map(|&a| a + offset)),
            },
        };

        let len = vertices.len();

        vertices_total.extend(vertices);

        offset += len as u32;
    }

    culling_info.bb_min = min_pos;
    culling_info.bb_max = max_pos;

    Ok(MeshResource::new(vertices_total, indices_total, culling_info))
}

pub fn extract_scene(slice: &[u8]) -> Result<ImportedScene, AppError> {
    let gltf_convert_matrix = Mat4::from_euler_angles(std::f32::consts::FRAC_PI_2, 0.0, 0.0);

    let gltf = Gltf::from_slice(slice)?;

    let mut camera = None;

    for cam in gltf.cameras() {
        if let gltf::camera::Projection::Perspective(per) = cam.projection() {
            let index = cam.index();
            let mut camera_node = None;

            for node in gltf.nodes() {
                if let Some(c) = node.camera() {
                    if c.index() == index {
                        camera_node = Some(node);
                    }
                    break;
                }
            }

            let camera_node = match camera_node {
                Some(a) => a,
                None => return Err(AppError::Import("Error importing camera".into())),
            };

            let transform = transform_from_node(&camera_node, &gltf_convert_matrix);

            let rot = camera_node.transform().decomposed().1;
            let quat = Quaternion::new(rot[3], rot[0], rot[1], rot[2]);
            let rot = gltf_convert_matrix * quat_cast(&quat);
            let mat3 = nalgebra_glm::mat4_to_mat3(&rot);
            let rot = Rotation3::from_matrix(&mat3);
            let angles = rot.euler_angles();

            let stub = ImportedCamera {
                fov: math::rad_to_deg(math::fovy_to_fovx(per.yfov(), per.aspect_ratio().unwrap_or(16.0 / 9.0))),
                position: Vec4::from(transform.transform_point(&Point3::new(0.0, 0.0, 0.0))).xyz(),
                rotation: Vec3::new(angles.0, angles.1, angles.2),
            };

            camera = Some(stub);
            break;
        }
    }

    let buffers = gltf::import_buffers(&gltf.document, None, gltf.blob)?;

    let mut meshes = Vec::new();
    let mut instances = Vec::new();

    for mesh in gltf.document.meshes() {
        let m = Rc::new(extract_mesh(mesh, &buffers)?);

        meshes.push(m);
    }

    extract_instances(
        gltf_convert_matrix,
        Mat4::identity(),
        gltf.document.scenes().next().unwrap().nodes(),
        &mut meshes,
        &mut instances,
    );

    info!("Loaded {} meshes in {} instances", meshes.len(), instances.len());

    Ok(ImportedScene {
        resources: meshes,
        instances,
        camera,
    })
}

fn extract_instances<'a, T: ExactSizeIterator<Item = gltf::Node<'a>>>(
    convert_mat: Mat4,
    root_transform: Mat4,
    nodes: T,
    meshes: &mut Vec<Rc<MeshResource>>,
    instances: &mut Vec<MeshInstance>,
) {
    for instance in nodes {
        let transform = transform_from_node(&instance, &convert_mat);
        extract_instances(
            convert_mat,
            root_transform * transform,
            instance.children(),
            meshes,
            instances,
        );

        let mesh = instance.mesh();

        let mut ins = match mesh {
            Some(m) => MeshInstance::new(meshes[m.index()].clone()),
            None => continue,
        };

        ins.transform = root_transform * transform;
        ins.inverse = ins.transform.try_inverse().unwrap();

        instances.push(ins);
    }
}

pub struct ImportedScene {
    pub resources: Vec<Rc<MeshResource>>,
    pub instances: Vec<MeshInstance>,
    pub camera: Option<ImportedCamera>,
}

#[derive(Debug)]
pub struct ImportedCamera {
    pub fov: f32,
    pub position: Vec3,
    pub rotation: Vec3,
}

enum IndexType {
    U16,
    U32,
}

enum IndicesAccessor<'a> {
    U16(&'a [u16]),
    U32(&'a [u32]),
}

struct Accessors<'a> {
    pos: &'a [f32],
    normal: &'a [f32],
    indices: IndicesAccessor<'a>,
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
        let indices = Self::accessor_to_indices(indices.unwrap(), data);

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

    fn accessor_to_indices(accessor: Accessor<'a>, data: &[Data]) -> IndicesAccessor<'a> {
        match accessor.data_type() {
            ComponentType::U16 => IndicesAccessor::U16(Self::accessor_to_slice(accessor, data)),
            ComponentType::U32 => IndicesAccessor::U32(Self::accessor_to_slice(accessor, data)),
            _ => panic!("invalid indices format in glTF: {}", {
                accessor.data_type().as_gl_enum()
            }),
        }
    }
}

fn transform_from_node(node: &gltf::Node, convert_matrix: &Mat4) -> Mat4 {
    let (pos, rot, scale) = node.transform().decomposed();
    let scale = [scale[0], scale[2], scale[1]];

    let quat = Quaternion::new(rot[3], rot[0], rot[1], rot[2]);

    Mat4::new_translation(&(Vec3::from(pos).xzy().component_mul(&vec3(1.0, -1.0, 1.0))))
        * convert_matrix
        * quat_cast(&quat)
        * inverse(convert_matrix)
        * Mat4::new_nonuniform_scaling(&Vec3::from(scale))
}
