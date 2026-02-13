use crate::camera::PerspectiveCamera;
use crate::mesh::MeshInstance;
use nalgebra_glm::{Mat4, Vec3, vec3};

pub struct PointLight {
    pub color: Vec3,
    pub intensity: f32,
    pub radius: f32,
}

pub enum Component {
    Transform(Mat4),
    MeshInstance,
    PointLight(PointLight),
    Camera,
    Environment,
}

pub struct Node {
    pub children: Vec<Node>,
    pub components: Vec<Component>,
}

pub struct Scene {
    pub nodes: Vec<Node>,
    pub camera: PerspectiveCamera,
    pub meshes: Vec<MeshInstance>,
    pub env: Environment,
}

impl Scene {
    pub fn new() -> Self {
        Scene {
            nodes: Vec::new(),
            camera: PerspectiveCamera::new(),
            meshes: Vec::new(),
            env: Environment::default(),
        }
    }
}

pub struct Environment {
    pub sun_direction: Vec3,
    pub sun_color: Vec3,
    pub sun_angle: f32,
    pub exposure: f32,
    pub sky_intensity: f32,
    pub sun_intensity: f32,
}

impl Default for Environment {
    fn default() -> Self {
        Self {
            sun_direction: vec3(0.1, -0.8, 1.00).normalize(),
            sun_color: vec3(0.9, 0.8, 0.7),
            sun_angle: 0.54 / (180.0 * std::f32::consts::PI),
            exposure: 0.0,
            sky_intensity: 1.0,
            sun_intensity: 1.0,
        }
    }
}
