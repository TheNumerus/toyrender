use crate::camera::PerspectiveCamera;
use crate::mesh::MeshInstance;
use nalgebra_glm::{Vec3, vec3};

pub struct PointLight {
    pub position: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub radius: f32,
}

pub struct Scene {
    pub camera: PerspectiveCamera,
    pub meshes: Vec<MeshInstance>,
    pub env: Environment,
    pub lights: Vec<PointLight>,
}

impl Scene {
    pub fn new() -> Self {
        Scene {
            camera: PerspectiveCamera::new(),
            meshes: Vec::new(),
            env: Environment::default(),
            lights: Vec::new(),
        }
    }
}

pub struct Environment {
    pub sun_direction: Vec3,
    pub sun_color: Vec3,
    pub sun_angle: f32,
    pub exposure: f32,
    pub sky_intensity: f32,
    pub sky_only: bool,
}

impl Default for Environment {
    fn default() -> Self {
        Self {
            sun_direction: vec3(0.1, -0.8, 1.00).normalize(),
            sun_color: vec3(0.9, 0.8, 0.7),
            sun_angle: 0.54 / (180.0 * std::f32::consts::PI),
            exposure: 0.0,
            sky_intensity: 1.0,
            sky_only: false,
        }
    }
}
