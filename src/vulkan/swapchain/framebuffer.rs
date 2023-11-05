use crate::vulkan::{Device, ImageView, IntoVulkanError, RenderPass, SwapChain, VulkanError};
use ash::vk;
use ash::vk::Framebuffer;
use std::rc::Rc;

pub struct SwapChainFramebuffer {
    pub inner: Framebuffer,
    device: Rc<Device>,
}

impl SwapChainFramebuffer {
    pub fn new(
        device: Rc<Device>,
        render_pass: &RenderPass,
        swapchain: &SwapChain,
        image_views: &[&ImageView],
    ) -> Result<Self, VulkanError> {
        let attachments = image_views.iter().map(|&w| w.inner).collect::<Vec<_>>();

        let framebuffer_info = vk::FramebufferCreateInfo {
            render_pass: render_pass.inner,
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            width: swapchain.extent.width,
            height: swapchain.extent.height,
            layers: 1,
            ..Default::default()
        };

        let inner = unsafe {
            device
                .inner
                .create_framebuffer(&framebuffer_info, None)
                .map_to_err("cannot create framebuffer")?
        };

        Ok(Self { device, inner })
    }
}

impl Drop for SwapChainFramebuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.inner.destroy_framebuffer(self.inner, None);
        }
    }
}
