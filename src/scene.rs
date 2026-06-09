use crate::camera::PerspectiveCamera;
use crate::image::ImageResource;
use crate::mesh::MeshInstance;
use nalgebra_glm::{Mat4, Vec3, vec3};
use std::any::Any;
use std::rc::Rc;

pub trait Component: 'static + Clone + Copy + Sized {}

#[derive(Clone, Copy)]
pub struct PointLight {
    pub color: Vec3,
    pub intensity: f32,
    pub radius: f32,
}

impl Component for PointLight {}

#[derive(Clone, Copy)]
pub struct Transform(pub Mat4);
impl Component for Transform {}

pub struct Node {
    pub children: Vec<Node>,
    pub components: Vec<Box<dyn Any>>,
}

impl Node {
    pub fn new() -> Self {
        Self {
            components: Vec::new(),
            children: Vec::new(),
        }
    }

    pub fn add_component(mut self, component: impl Component) -> Self {
        self.components.push(Box::new(component) as Box<dyn Any>);
        self
    }

    pub fn get_component<T: Component>(&self) -> Option<&T> {
        for component in &self.components {
            match component.downcast_ref::<T>() {
                Some(component) => return Some(component),
                None => continue,
            }
        }
        None
    }

    pub fn get_component_mut<T: Component>(&mut self) -> Option<&mut T> {
        for component in &mut self.components {
            match component.downcast_mut::<T>() {
                Some(component) => return Some(component),
                None => continue,
            }
        }
        None
    }
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

#[derive(Debug)]
pub enum SkyVariant {
    Shader,
    SingleColor(Vec3),
    Textured(Rc<ImageResource>, f32),
}

pub struct Sky {
    pub variant: SkyVariant,
    pub intensity: f32,
}

pub struct Environment {
    pub sun_direction: Vec3,
    pub sun_color: Vec3,
    pub sun_angle: f32,
    pub exposure: f32,
    pub sun_intensity: f32,
    pub sky: Sky,
}

impl Default for Environment {
    fn default() -> Self {
        Self {
            sun_direction: vec3(0.1, -0.8, 1.00).normalize(),
            sun_color: vec3(0.9, 0.8, 0.7),
            sun_angle: 0.54 / (180.0 * std::f32::consts::PI),
            exposure: 0.0,
            sun_intensity: 1.0,
            sky: Sky {
                variant: SkyVariant::Shader,
                intensity: 1.0,
            },
        }
    }
}
