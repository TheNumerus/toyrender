use crate::vulkan::{Device, IntoVulkanError, VulkanError};
use ash::vk;
use ash::vk::{Image, ImageView as RawImageView};
use std::rc::Rc;

pub struct ImageView {
    pub inner: RawImageView,
    device: Rc<Device>,
}

impl ImageView {
    pub fn new(
        device: Rc<Device>,
        image: Image,
        format: vk::Format,
        aspect_flags: vk::ImageAspectFlags,
    ) -> Result<Self, VulkanError> {
        let create_info = vk::ImageViewCreateInfo {
            image,
            view_type: vk::ImageViewType::TYPE_2D,
            format,
            components: vk::ComponentMapping::default(),
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: aspect_flags,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..Default::default()
        };

        let inner = unsafe {
            device
                .inner
                .create_image_view(&create_info, None)
                .map_to_err("Cannot create image view")?
        };

        Ok(Self { inner, device })
    }
}

impl Drop for ImageView {
    fn drop(&mut self) {
        unsafe { self.device.inner.destroy_image_view(self.inner, None) }
    }
}
