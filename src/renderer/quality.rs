pub struct QualitySettings {
    pub pt_bounces: i32,
    pub rt_direct_trace_disance: f32,
    pub rt_indirect_trace_disance: f32,
    pub half_res: bool,
    pub use_spatial_denoise: bool,
}

impl QualitySettings {
    pub fn new() -> Self {
        Self {
            pt_bounces: 3,
            rt_direct_trace_disance: 100.0,
            rt_indirect_trace_disance: 100.0,
            half_res: false,
            use_spatial_denoise: true,
        }
    }
}

impl Default for QualitySettings {
    fn default() -> Self {
        Self::new()
    }
}
