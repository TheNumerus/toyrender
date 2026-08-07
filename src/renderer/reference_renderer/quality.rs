pub struct QualitySettings {
    pub pt_bounces: u32,
    pub rt_direct_trace_distance: f32,
    pub rt_indirect_trace_distance: f32,
    pub indirect_light_clamp: f32,
    pub importance_sampling: bool,
    pub russian_roulette: bool,
    pub disable_materials: bool,
}

impl QualitySettings {
    pub fn new() -> Self {
        Self {
            pt_bounces: 3,
            rt_direct_trace_distance: 100.0,
            rt_indirect_trace_distance: 100.0,
            indirect_light_clamp: 0.0,
            importance_sampling: true,
            russian_roulette: true,
            disable_materials: false,
        }
    }
}

impl Default for QualitySettings {
    fn default() -> Self {
        Self::new()
    }
}
