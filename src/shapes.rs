use crate::vulkan::Vertex;
use nalgebra_glm::{vec3, vec4};

#[allow(dead_code)]
pub fn triangle() -> [Vertex; 3] {
    [
        Vertex {
            pos: vec3(0.0, -0.5, 0.0),
            color: vec4(1.0, 0.0, 0.0, 0.0),
            uv: Default::default(),
        },
        Vertex {
            pos: vec3(0.5, 0.5, 0.0),
            color: vec4(0.0, 1.0, 0.0, 0.0),
            uv: Default::default(),
        },
        Vertex {
            pos: vec3(-0.5, 0.5, 0.0),
            color: vec4(0.0, 0.0, 1.0, 0.0),
            uv: Default::default(),
        },
    ]
}

#[allow(dead_code)]
pub fn square() -> [Vertex; 6] {
    [
        Vertex {
            pos: vec3(-0.5, -0.5, 0.0),
            color: vec4(1.0, 0.0, 0.0, 0.0),
            uv: Default::default(),
        },
        Vertex {
            pos: vec3(0.5, 0.5, 0.0),
            color: vec4(0.0, 1.0, 0.0, 0.0),
            uv: Default::default(),
        },
        Vertex {
            pos: vec3(-0.5, 0.5, 0.0),
            color: vec4(0.0, 0.0, 1.0, 0.0),
            uv: Default::default(),
        },
        Vertex {
            pos: vec3(0.5, -0.5, 0.0),
            color: vec4(1.0, 1.0, 0.0, 0.0),
            uv: Default::default(),
        },
        Vertex {
            pos: vec3(0.5, 0.5, 0.0),
            color: vec4(0.0, 1.0, 0.0, 0.0),
            uv: Default::default(),
        },
        Vertex {
            pos: vec3(-0.5, -0.5, 0.0),
            color: vec4(1.0, 0.0, 0.0, 0.0),
            uv: Default::default(),
        },
    ]
}
