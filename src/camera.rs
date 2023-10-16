use nalgebra_glm::{Mat4, Vec3};

#[derive(Clone)]
pub struct PerspectiveCamera {
    pub fov: f32,
    pub transform: Mat4,
}

impl PerspectiveCamera {
    pub fn new() -> Self {
        Self {
            fov: 80.0,
            transform: Mat4::new_translation(&Vec3::from_element(0.0)),
        }
    }
}

impl Default for PerspectiveCamera {
    fn default() -> Self {
        Self::new()
    }
}
