use crate::camera::PerspectiveCamera;
use crate::mesh::MeshInstance;
use nalgebra_glm::{vec3, Vec3};

pub struct Scene {
    pub camera: PerspectiveCamera,
    pub meshes: Vec<MeshInstance>,
    pub env: Environment,
}

impl Scene {
    pub fn new() -> Self {
        Scene {
            camera: PerspectiveCamera::new(),
            meshes: Vec::new(),
            env: Environment::default(),
        }
    }
}

pub struct Environment {
    pub sun_direction: Vec3,
    pub sun_color: Vec3,
    pub exposure: f32,
    pub sky_only: bool,
}

impl Default for Environment {
    fn default() -> Self {
        Self {
            sun_direction: vec3(0.1, -0.8, 1.00).normalize(),
            sun_color: vec3(0.9, 0.8, 0.7),
            exposure: 0.0,
            sky_only: false,
        }
    }
}
