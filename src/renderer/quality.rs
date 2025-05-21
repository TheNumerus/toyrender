pub struct QualitySettings {
    pub rtao_samples: i32,
    pub rt_trace_disance: f32,
    pub half_res: bool,
}

impl QualitySettings {
    pub fn new() -> Self {
        Self {
            rtao_samples: 3,
            rt_trace_disance: 100.0,
            half_res: false,
        }
    }
}

impl Default for QualitySettings {
    fn default() -> Self {
        Self::new()
    }
}
