use crate::camera::PerspectiveCamera;
use crate::mesh::MeshInstance;

pub struct Scene {
    pub camera: PerspectiveCamera,
    pub meshes: Vec<MeshInstance>,
}

impl Scene {
    pub fn new() -> Self {
        Scene {
            camera: PerspectiveCamera::new(),
            meshes: Vec::new(),
        }
    }
}
