use crate::vulkan::Vertex;
use nalgebra_glm::{vec2, vec3, vec4, Mat4};
use std::borrow::Cow;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

pub static MESH_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

pub struct MeshResource {
    pub id: u64,
    pub vertices: Vec<Vertex>,
    pub indices: Indices,
}

impl MeshResource {
    pub fn new(vertices: Vec<Vertex>, indices: Indices) -> Self {
        Self {
            id: MESH_ID_COUNTER.fetch_add(1, Ordering::SeqCst),
            vertices,
            indices,
        }
    }
}

pub enum Indices {
    U16(Vec<u16>),
    U32(Vec<u32>),
}

impl Indices {
    pub fn to_vec_u32(&self) -> Cow<Vec<u32>> {
        match &self {
            Indices::U16(v) => Cow::Owned(v.iter().map(|&r| r as u32).collect()),
            Indices::U32(v) => Cow::Borrowed(v),
        }
    }
}

pub struct MeshInstance {
    pub resource: Rc<MeshResource>,
    pub transform: Mat4,
}

impl MeshInstance {
    pub fn new(resource: Rc<MeshResource>) -> Self {
        Self {
            resource,
            transform: Mat4::identity(),
        }
    }
}

#[allow(dead_code)]
pub fn triangle() -> ([Vertex; 3], [u32; 3]) {
    (
        [
            Vertex {
                pos: vec3(0.0, -0.5, 0.0),
                normal: vec3(0.0, 0.0, 1.0),
                color: vec4(1.0, 0.0, 0.0, 0.0),
                uv: Default::default(),
            },
            Vertex {
                pos: vec3(0.5, 0.5, 0.0),
                normal: vec3(0.0, 0.0, 1.0),
                color: vec4(0.0, 1.0, 0.0, 0.0),
                uv: Default::default(),
            },
            Vertex {
                pos: vec3(-0.5, 0.5, 0.0),
                normal: vec3(0.0, 0.0, 1.0),
                color: vec4(0.0, 0.0, 1.0, 0.0),
                uv: Default::default(),
            },
        ],
        [0, 1, 2],
    )
}
#[allow(dead_code)]
pub fn fs_triangle() -> ([Vertex; 3], [u32; 3]) {
    (
        [
            Vertex {
                pos: vec3(-1.0, -3.0, 0.0),
                normal: vec3(0.0, 0.0, 1.0),
                color: vec4(1.0, 1.0, 1.0, 1.0),
                uv: vec2(0.0, -1.0),
            },
            Vertex {
                pos: vec3(3.0, 1.0, 0.0),
                normal: vec3(0.0, 0.0, 1.0),
                color: vec4(1.0, 1.0, 1.0, 1.0),
                uv: vec2(2.0, 1.0),
            },
            Vertex {
                pos: vec3(-1.0, 1.0, 0.0),
                normal: vec3(0.0, 0.0, 1.0),
                color: vec4(1.0, 1.0, 1.0, 1.0),
                uv: vec2(0.0, 1.0),
            },
        ],
        [0, 2, 1],
    )
}

#[allow(dead_code)]
pub fn square() -> ([Vertex; 4], [u32; 6]) {
    (
        [
            Vertex {
                pos: vec3(-0.5, -0.5, 0.0),
                normal: vec3(0.0, 0.0, 1.0),
                color: vec4(1.0, 0.0, 0.0, 0.0),
                uv: Default::default(),
            },
            Vertex {
                pos: vec3(0.5, 0.5, 0.0),
                normal: vec3(0.0, 0.0, 1.0),
                color: vec4(0.0, 1.0, 0.0, 0.0),
                uv: Default::default(),
            },
            Vertex {
                pos: vec3(-0.5, 0.5, 0.0),
                normal: vec3(0.0, 0.0, 1.0),
                color: vec4(0.0, 0.0, 1.0, 0.0),
                uv: Default::default(),
            },
            Vertex {
                pos: vec3(0.5, -0.5, 0.0),
                normal: vec3(0.0, 0.0, 1.0),
                color: vec4(1.0, 1.0, 0.0, 0.0),
                uv: Default::default(),
            },
        ],
        [0, 1, 2, 1, 0, 3],
    )
}
