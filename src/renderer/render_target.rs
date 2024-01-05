use crate::vulkan::{Device, Framebuffer, Image, ImageView, RenderPass, Sampler, VulkanError};
use ash::vk;
use ash::vk::{Extent2D, Extent3D};
use std::rc::Rc;

pub struct RenderTarget {
    pub framebuffer: Option<Framebuffer>,
    pub image: Image,
    pub view: ImageView,
    pub sampler: Sampler,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    aspect: vk::ImageAspectFlags,
    device: Rc<Device>,
}

impl RenderTarget {
    pub fn new(
        device: Rc<Device>,
        extent: Extent3D,
        render_pass: Option<&RenderPass>,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        aspect: vk::ImageAspectFlags,
    ) -> Result<Self, VulkanError> {
        let image = Image::new(device.clone(), format, extent, usage)?;

        let view = ImageView::new(device.clone(), image.inner, format, aspect)?;

        let framebuffer = match render_pass {
            None => None,
            Some(render_pass) => Some(Framebuffer::new(
                device.clone(),
                render_pass,
                Extent2D {
                    width: extent.width,
                    height: extent.height,
                },
                &[&view],
            )?),
        };

        let sampler = Sampler::new(device.clone())?;

        Ok(Self {
            image,
            view,
            framebuffer,
            sampler,
            device,
            format,
            usage,
            aspect,
        })
    }

    pub fn resize(&mut self, extent: Extent3D, render_pass: Option<&RenderPass>) -> Result<(), VulkanError> {
        self.image = Image::new(self.device.clone(), self.format, extent, self.usage)?;

        self.view = ImageView::new(self.device.clone(), self.image.inner, self.format, self.aspect)?;

        if let Some(render_pass) = render_pass {
            self.framebuffer = Some(Framebuffer::new(
                self.device.clone(),
                render_pass,
                Extent2D {
                    width: extent.width,
                    height: extent.height,
                },
                &[&self.view],
            )?)
        }

        Ok(())
    }

    pub fn descriptor_image_info(&self, image_layout: vk::ImageLayout) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo {
            sampler: self.sampler.inner,
            image_view: self.view.inner,
            image_layout,
        }
    }
}
