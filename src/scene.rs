use crate::camera::PerspectiveCamera;
use crate::mesh::Mesh;

pub struct Scene {
    pub camera: PerspectiveCamera,
    pub meshes: Vec<Mesh>,
}

impl Scene {
    pub fn new() -> Self {
        Scene {
            camera: PerspectiveCamera::new(),
            meshes: Vec::new(),
        }
    }
}
