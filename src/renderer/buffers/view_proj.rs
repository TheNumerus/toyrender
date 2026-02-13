use nalgebra_glm::Mat4;

#[derive(Default)]
pub struct ViewProj {
    pub view: Mat4,
    pub projection: Mat4,
    pub view_prev: Mat4,
    pub projection_prev: Mat4,
    pub view_inverse: Mat4,
    pub projection_inverse: Mat4,
}

#[repr(C)]
struct ViewProjGPU {
    pub view: Mat4,
    pub view_prev: Mat4,
    pub projection: Mat4,
    pub projection_prev: Mat4,
    pub view_inverse: Mat4,
    pub projection_inverse: Mat4,
}

impl ViewProj {
    pub const BUF_SIZE: usize = std::mem::size_of::<ViewProjGPU>();

    pub fn to_bytes(&self) -> Vec<u8> {
        let gpu = ViewProjGPU {
            view: self.view,
            projection: self.projection,
            view_prev: self.view_prev,
            projection_prev: self.projection_prev,
            view_inverse: self.view_inverse,
            projection_inverse: self.projection_inverse,
        };

        let size = size_of::<ViewProjGPU>();
        unsafe { core::slice::from_raw_parts(&gpu as *const ViewProjGPU as *const u8, size).to_owned() }
    }
}
