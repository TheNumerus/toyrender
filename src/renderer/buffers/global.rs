#[repr(C)]
pub struct Globals {
    pub exposure: f32,
    pub debug_mode: i32,
    pub res_x: f32,
    pub res_y: f32,
    pub draw_res_x: f32,
    pub draw_res_y: f32,
    pub time: f32,
    pub frame_index: u32,
    pub current_jitter: (f32, f32),
    pub prev_jitter: (f32, f32),
}

impl Globals {
    pub fn to_bytes(&self) -> Vec<u8> {
        let size = size_of::<Self>();
        unsafe { core::slice::from_raw_parts(self as *const Self as *const u8, size).to_owned() }
    }
}
