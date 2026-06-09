use image::DynamicImage;
use std::fmt::Debug;

pub struct ImageResource {
    pub id: uuid::Uuid,
    pub data: DynamicImage,
    pub name: String,
}

impl ImageResource {
    pub fn new(data: DynamicImage, name: impl AsRef<str>) -> Self {
        Self {
            id: uuid::Uuid::new_v4(),
            data,
            name: name.as_ref().to_owned(),
        }
    }
}

impl Debug for ImageResource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageResource")
            .field("id", &self.id)
            .field("name", &self.name)
            .finish()
    }
}
