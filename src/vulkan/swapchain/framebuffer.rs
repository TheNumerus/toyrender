use crate::vulkan::{Device, RenderPass, SwapChain, SwapChainImageView, VulkanError};
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
        image_view: &SwapChainImageView,
    ) -> Result<Self, VulkanError> {
        let attachments = [image_view.inner];

        let framebuffer_info = vk::FramebufferCreateInfo {
            render_pass: render_pass.inner,
            attachment_count: 1,
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
                .map_err(|code| VulkanError {
                    code,
                    msg: "cannot create framebuffer".into(),
                })?
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
