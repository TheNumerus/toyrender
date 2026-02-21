use nalgebra_glm::Mat4;

pub struct PushConstBuilder {
    storage: Vec<u8>,
}

impl PushConstBuilder {
    pub fn new() -> Self {
        Self { storage: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            storage: Vec::with_capacity(capacity),
        }
    }

    pub fn add_u32(mut self, value: u32) -> Self {
        self.storage.extend(value.to_le_bytes());
        self
    }

    pub fn update_u32(mut self, value: u32, position: usize) -> Self {
        self.storage.as_mut_slice()[position..position + 4].copy_from_slice(&value.to_le_bytes());
        self
    }

    pub fn add_f32(mut self, value: f32) -> Self {
        self.storage.extend(value.to_le_bytes());
        self
    }

    pub fn update_f32(mut self, value: f32, position: usize) -> Self {
        self.storage.as_mut_slice()[position..position + 4].copy_from_slice(&value.to_le_bytes());
        self
    }

    pub fn add_mat(mut self, mat4: Mat4) -> Self {
        let slice = mat4.data.as_slice();

        let slice_cast = unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 4) };

        self.storage.extend(slice_cast);
        self
    }

    pub fn build(self) -> Box<[u8]> {
        self.storage.into_boxed_slice()
    }
}
