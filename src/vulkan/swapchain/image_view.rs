use crate::vulkan::{Device, SwapChain, VulkanError};
use ash::vk;
use ash::vk::{Image, ImageView};
use std::rc::Rc;

pub struct SwapChainImageView {
    pub inner: ImageView,
    device: Rc<Device>,
}

impl SwapChainImageView {
    pub fn new(device: Rc<Device>, image: Image, swapchain: &SwapChain) -> Result<Self, VulkanError> {
        let create_info = vk::ImageViewCreateInfo {
            image,
            view_type: vk::ImageViewType::TYPE_2D,
            format: swapchain.format.format,
            components: vk::ComponentMapping::default(),
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
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
                .map_err(|code| VulkanError {
                    code,
                    msg: "Cannot create image view".into(),
                })?
        };

        Ok(Self { inner, device })
    }
}

impl Drop for SwapChainImageView {
    fn drop(&mut self) {
        unsafe { self.device.inner.destroy_image_view(self.inner, None) }
    }
}
