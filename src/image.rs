use image::DynamicImage;

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
