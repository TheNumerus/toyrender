pub struct QualitySettings {
    pub rtao_samples: i32,
    pub half_res: bool,
}

impl QualitySettings {
    pub fn new() -> Self {
        Self {
            rtao_samples: 1,
            half_res: false,
        }
    }
}

impl Default for QualitySettings {
    fn default() -> Self {
        Self::new()
    }
}
