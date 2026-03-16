use crate::scene::{Environment, SkyVariant};
use nalgebra_glm::Vec3;

#[repr(C)]
/// float3 followed by float are turned into float 4
pub struct GPUEnv {
    sun_dir: [f32; 3],
    sun_angle: f32,
    sun_color: [f32; 3],
    sun_intensity: f32,
    sky_color: [f32; 3],
    sky_intensity: f32,
    sky_mode: i32,
    sky_rotation: f32,
    exposure: f32,
    do_distortion: i32,
}

impl Environment {
    pub fn to_bytes(&self) -> Vec<u8> {
        let size = size_of::<GPUEnv>();
        let gpu = GPUEnv {
            sun_dir: self.sun_direction.normalize().into(),
            sun_angle: self.sun_angle,
            sun_color: self.sun_color.into(),
            sun_intensity: self.sun_intensity,
            sky_color: match self.sky.variant {
                SkyVariant::Shader => Vec3::from_element(1.0).into(),
                SkyVariant::SingleColor(a) => a.into(),
                SkyVariant::Textured(_, _) => Vec3::from_element(1.0).into(),
            },
            sky_intensity: self.sky.intensity,
            sky_mode: match self.sky.variant {
                SkyVariant::Shader => 0,
                SkyVariant::SingleColor(_) => 1,
                SkyVariant::Textured(_, _) => 2,
            },
            sky_rotation: match self.sky.variant {
                SkyVariant::Textured(_, r) => r,
                _ => 0.0,
            },
            exposure: self.exposure,
            do_distortion: match self.sky.variant {
                SkyVariant::Shader => 1,
                _ => 0,
            },
        };

        unsafe { core::slice::from_raw_parts(&gpu as *const GPUEnv as *const u8, size).to_owned() }
    }
}
