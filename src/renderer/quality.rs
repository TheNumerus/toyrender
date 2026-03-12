pub struct QualitySettings {
    pub pt_bounces: u32,
    pub rt_direct_trace_distance: f32,
    pub rt_indirect_trace_distance: f32,
    pub indirect_light_clamp: f32,
    pub use_spatial_denoise: bool,
}

impl QualitySettings {
    pub fn new() -> Self {
        Self {
            pt_bounces: 3,
            rt_direct_trace_distance: 100.0,
            rt_indirect_trace_distance: 100.0,
            use_spatial_denoise: true,
            indirect_light_clamp: 0.0,
        }
    }
}

impl Default for QualitySettings {
    fn default() -> Self {
        Self::new()
    }
}
