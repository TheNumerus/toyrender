use nalgebra_glm::{Mat4, Vec3};

#[derive(Clone)]
pub struct PerspectiveCamera {
    pub fov: f32,
    pub transform: Mat4,
    pub position: Vec3,
    pub rotation: Vec3,
}

impl PerspectiveCamera {
    pub fn new() -> Self {
        Self {
            fov: 70.0,
            transform: Mat4::new_translation(&Vec3::from_element(0.0)),
            position: Vec3::from_element(0.0),
            rotation: Vec3::from_element(0.0),
        }
    }

    pub fn directions(&self) -> Directions {
        let view = nalgebra_glm::inverse(&self.view());

        Directions {
            up: nalgebra_glm::normalize(&view.column(1).xyz()),
            forward: nalgebra_glm::normalize(&view.column(2).xyz()),
            right: nalgebra_glm::normalize(&view.column(0).xyz()),
        }
    }

    pub fn view(&self) -> Mat4 {
        let pitch = nalgebra_glm::rotation(-self.rotation.x, &Vec3::new(1.0, 0.0, 0.0));
        let yaw = nalgebra_glm::rotation(-self.rotation.z, &Vec3::new(0.0, 0.0, 1.0));
        let eye = nalgebra_glm::translation(&-self.position);

        pitch * yaw * eye
    }
}

impl Default for PerspectiveCamera {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Directions {
    pub up: Vec3,
    pub right: Vec3,
    pub forward: Vec3,
}
