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

pub struct GBuffer {
    pub framebuffer: Framebuffer,
    pub images: Vec<Image>,
    pub views: Vec<ImageView>,
    pub sampler: Sampler,
    device: Rc<Device>,
}

impl GBuffer {
    pub fn attrs() -> [(vk::Format, vk::ImageUsageFlags, vk::ImageAspectFlags); 3] {
        [
            (
                vk::Format::A2B10G10R10_UNORM_PACK32,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                vk::ImageAspectFlags::COLOR,
            ),
            (
                vk::Format::A2B10G10R10_UNORM_PACK32,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                vk::ImageAspectFlags::COLOR,
            ),
            (
                vk::Format::D32_SFLOAT,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                vk::ImageAspectFlags::DEPTH,
            ),
        ]
    }

    pub fn new(device: Rc<Device>, extent: Extent3D, render_pass: &RenderPass) -> Result<Self, VulkanError> {
        let images = Self::attrs()
            .into_iter()
            .map(|(format, usage, _)| Image::new(device.clone(), format, extent, usage))
            .collect::<Result<Vec<_>, _>>()?;

        let views = images
            .iter()
            .zip(Self::attrs().into_iter())
            .map(|(image, (format, _, aspect))| ImageView::new(device.clone(), image.inner, format, aspect))
            .collect::<Result<Vec<_>, _>>()?;

        let framebuffer = Framebuffer::new(
            device.clone(),
            render_pass,
            Extent2D {
                width: extent.width,
                height: extent.height,
            },
            &[&views[0], &views[1], &views[2]],
        )?;

        let sampler = Sampler::new(device.clone())?;

        Ok(Self {
            images,
            views,
            framebuffer,
            sampler,
            device,
        })
    }

    pub fn resize(&mut self, extent: Extent3D, render_pass: &RenderPass) -> Result<(), VulkanError> {
        let attrs = Self::attrs();

        for (idx, image) in &mut self.images.iter_mut().enumerate() {
            *image = Image::new(self.device.clone(), attrs[idx].0, extent, attrs[idx].1)?;
        }

        for (idx, image) in &mut self.views.iter_mut().enumerate() {
            *image = ImageView::new(self.device.clone(), self.images[idx].inner, attrs[idx].0, attrs[idx].2)?;
        }

        self.framebuffer = Framebuffer::new(
            self.device.clone(),
            render_pass,
            Extent2D {
                width: extent.width,
                height: extent.height,
            },
            &[&self.views[0], &self.views[1], &self.views[2]],
        )?;

        Ok(())
    }
}
