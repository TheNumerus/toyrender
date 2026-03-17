mod env;
pub use env::GPUEnv;

mod global;
pub use global::Globals;

mod view_proj;
pub use view_proj::ViewProj;

#[derive(Debug)]
pub struct PointLightGpu {
    pub color: [f32; 3],
    pub intensity: f32,
    pub pos: [f32; 3],
    pub radius: f32,
}

impl PointLightGpu {
    pub fn to_bytes(&self) -> Vec<u8> {
        let size = size_of::<Self>();
        unsafe { core::slice::from_raw_parts(self as *const Self as *const u8, size).to_owned() }
    }
}
