use crate::err::AppError;
use crate::vulkan::{Device, Framebuffer, Image, ImageView, RenderPass, Sampler, VulkanError};
use ash::vk;
use ash::vk::{Extent2D, Extent3D, Handle};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::CString;
use std::rc::Rc;

pub struct RenderTarget {
    pub image: Image,
    pub view: ImageView,
    pub sampler: Sampler,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    aspect: vk::ImageAspectFlags,
    device: Rc<Device>,
    name: Option<String>,
}

impl RenderTarget {
    pub fn new(
        device: Rc<Device>,
        extent: Extent3D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        aspect: vk::ImageAspectFlags,
    ) -> Result<Self, VulkanError> {
        let image = Image::new(device.clone(), format, extent, usage)?;
        let view = ImageView::new(device.clone(), image.inner, format, aspect)?;
        let sampler = Sampler::new(device.clone())?;

        Ok(Self {
            image,
            view,
            sampler,
            device,
            format,
            usage,
            aspect,
            name: None,
        })
    }

    pub fn set_names<S: AsRef<str>>(&mut self, name: S) -> Result<(), VulkanError> {
        let name = name.as_ref().to_owned();

        self.name = Some(name.clone());
        let name_ptr = CString::new(name.clone() + "_image").unwrap();
        let name_info = vk::DebugUtilsObjectNameInfoEXT {
            object_type: vk::ObjectType::IMAGE,
            object_handle: self.image.inner.as_raw(),
            p_object_name: name_ptr.as_ptr(),
            ..Default::default()
        };

        self.device.name_object(name_info)?;

        let name_ptr = CString::new(name.clone() + "_imageview").unwrap();
        let name_info = vk::DebugUtilsObjectNameInfoEXT {
            object_type: vk::ObjectType::IMAGE_VIEW,
            object_handle: self.view.inner.as_raw(),
            p_object_name: name_ptr.as_ptr(),
            ..Default::default()
        };

        self.device.name_object(name_info)?;
        let name_ptr = CString::new(name + "_sampler").unwrap();
        let name_info = vk::DebugUtilsObjectNameInfoEXT {
            object_type: vk::ObjectType::SAMPLER,
            object_handle: self.sampler.inner.as_raw(),
            p_object_name: name_ptr.as_ptr(),
            ..Default::default()
        };

        self.device.name_object(name_info)
    }

    pub fn resize(&mut self, extent: Extent3D) -> Result<(), VulkanError> {
        self.image = Image::new(self.device.clone(), self.format, extent, self.usage)?;
        self.view = ImageView::new(self.device.clone(), self.image.inner, self.format, self.aspect)?;

        if let Some(name) = &self.name {
            self.set_names(name.to_owned())?
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

pub struct MultipleRenderTargetBuilder {
    name: String,
    render_pass: Rc<RenderPass>,
    attributes: Vec<(vk::Format, vk::ImageUsageFlags, vk::ImageAspectFlags)>,
}

impl MultipleRenderTargetBuilder {
    pub fn new(name: impl Into<String>, render_pass: Rc<RenderPass>) -> Self {
        Self {
            name: name.into(),
            attributes: Vec::new(),
            render_pass,
        }
    }

    pub fn add_target(mut self, format: vk::Format, usage: vk::ImageUsageFlags, aspect: vk::ImageAspectFlags) -> Self {
        self.attributes.push((format, usage, aspect));

        self
    }
}

pub struct MultipleRenderTarget {
    pub framebuffer: Framebuffer,
    pub images: Vec<Image>,
    pub views: Vec<ImageView>,
    pub sampler: Sampler,
    attributes: Vec<(vk::Format, vk::ImageUsageFlags, vk::ImageAspectFlags)>,
    device: Rc<Device>,
    render_pass: Rc<RenderPass>,
}

impl MultipleRenderTarget {
    fn new(
        device: Rc<Device>,
        extent: Extent3D,
        render_pass: Rc<RenderPass>,
        attributes: Vec<(vk::Format, vk::ImageUsageFlags, vk::ImageAspectFlags)>,
    ) -> Result<Self, VulkanError> {
        let images = attributes
            .iter()
            .map(|(format, usage, _)| Image::new(device.clone(), *format, extent, *usage))
            .collect::<Result<Vec<_>, _>>()?;

        let views = images
            .iter()
            .zip(attributes.iter())
            .map(|(image, (format, _, aspect))| ImageView::new(device.clone(), image.inner, *format, *aspect))
            .collect::<Result<Vec<_>, _>>()?;

        let views_slice = views.iter().collect::<Vec<_>>();

        let framebuffer = Framebuffer::new(
            device.clone(),
            &render_pass,
            Extent2D {
                width: extent.width,
                height: extent.height,
            },
            &views_slice,
        )?;

        let sampler = Sampler::new(device.clone())?;

        Ok(Self {
            images,
            views,
            framebuffer,
            sampler,
            attributes,
            device,
            render_pass,
        })
    }

    pub fn resize(&mut self, extent: Extent3D) -> Result<(), VulkanError> {
        for (idx, image) in &mut self.images.iter_mut().enumerate() {
            *image = Image::new(
                self.device.clone(),
                self.attributes[idx].0,
                extent,
                self.attributes[idx].1,
            )?;
        }

        for (idx, image) in &mut self.views.iter_mut().enumerate() {
            *image = ImageView::new(
                self.device.clone(),
                self.images[idx].inner,
                self.attributes[idx].0,
                self.attributes[idx].2,
            )?;
        }

        let views_slice = self.views.iter().collect::<Vec<_>>();

        self.framebuffer = Framebuffer::new(
            self.device.clone(),
            &self.render_pass,
            Extent2D {
                width: extent.width,
                height: extent.height,
            },
            &views_slice,
        )?;

        Ok(())
    }

    pub fn descriptor_image_info(&self, index: usize, image_layout: vk::ImageLayout) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo {
            sampler: self.sampler.inner,
            image_view: self.views[index].inner,
            image_layout,
        }
    }
}

pub struct DenoiseRenderTargets {
    pub direct_out: RenderTarget,
    pub indirect_out: RenderTarget,
    pub direct_history: RenderTarget,
    pub indirect_history: RenderTarget,
    pub direct_acc: RenderTarget,
    pub indirect_acc: RenderTarget,
}

impl DenoiseRenderTargets {
    pub fn new(device: Rc<Device>, extent: Extent3D) -> Result<Self, VulkanError> {
        let get_rt = || {
            RenderTarget::new(
                device.clone(),
                extent,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
                vk::ImageAspectFlags::COLOR,
            )
        };

        Ok(Self {
            direct_out: get_rt()?,
            indirect_out: get_rt()?,
            direct_acc: get_rt()?,
            indirect_acc: get_rt()?,
            direct_history: get_rt()?,
            indirect_history: get_rt()?,
        })
    }
}

#[derive(Copy, Clone)]
pub enum RenderTargetSize {
    Window,
    Scaled(f32),
    Custom(u32, u32),
}

pub struct RenderTargetBuilder {
    name: String,
    size: RenderTargetSize,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
    usage: vk::ImageUsageFlags,
}

impl RenderTargetBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            size: RenderTargetSize::Window,
            format: vk::Format::R16G16B16A16_SFLOAT,
            aspect: vk::ImageAspectFlags::COLOR,
            usage: vk::ImageUsageFlags::SAMPLED,
        }
    }

    pub fn new_depth(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            size: RenderTargetSize::Window,
            format: vk::Format::D32_SFLOAT,
            aspect: vk::ImageAspectFlags::DEPTH,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        }
    }

    pub fn duplicate(&self, name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..*self
        }
    }

    pub fn with_size(mut self, size: RenderTargetSize) -> Self {
        self.size = size;
        self
    }

    pub fn with_format(mut self, format: vk::Format) -> Self {
        self.format = format;
        self
    }

    pub fn with_aspect(mut self, aspect: vk::ImageAspectFlags) -> Self {
        self.aspect = aspect;
        self
    }

    pub fn with_transfer(mut self) -> Self {
        self.usage |= vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        self
    }

    pub fn with_storage(mut self) -> Self {
        self.usage |= vk::ImageUsageFlags::STORAGE;
        self
    }

    pub fn with_color_attachment(mut self) -> Self {
        self.usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        self
    }
}

struct RenderTargetItem {
    value: Rc<RefCell<RenderTarget>>,
    size: RenderTargetSize,
}

pub struct RenderTargets {
    targets: HashMap<String, RenderTargetItem>,
    mr_targets: HashMap<String, Rc<RefCell<MultipleRenderTarget>>>,
    device: Rc<Device>,
    default_extent: Extent3D,
}

impl RenderTargets {
    pub fn new(device: Rc<Device>, default_extent: Extent3D) -> Self {
        Self {
            targets: HashMap::new(),
            mr_targets: HashMap::new(),
            device,
            default_extent,
        }
    }

    pub fn set_extent(&mut self, new_extent: Extent3D) {
        self.default_extent = new_extent;
    }

    pub fn add(&mut self, builder: RenderTargetBuilder) -> Result<Rc<RefCell<RenderTarget>>, AppError> {
        if self.targets.contains_key(&builder.name) {
            return Err(AppError::Other(format!(
                "Render target with name '{}' already exists.",
                &builder.name
            )));
        }

        let key = builder.name.clone();
        let size = builder.size;
        let rt = self.build(builder)?;
        let rt_ptr = Rc::new(RefCell::new(rt));

        self.targets.insert(
            key,
            RenderTargetItem {
                value: rt_ptr.clone(),
                size,
            },
        );

        Ok(rt_ptr)
    }

    pub fn add_mrt(
        &mut self,
        builder: MultipleRenderTargetBuilder,
    ) -> Result<Rc<RefCell<MultipleRenderTarget>>, AppError> {
        if self.mr_targets.contains_key(&builder.name) {
            return Err(AppError::Other(format!(
                "Multiple render target with name '{}' already exists.",
                &builder.name
            )));
        }

        let key = builder.name.clone();
        let mrt = MultipleRenderTarget::new(
            self.device.clone(),
            self.default_extent,
            builder.render_pass,
            builder.attributes,
        )?;
        let mrt_ptr = Rc::new(RefCell::new(mrt));

        self.mr_targets.insert(key, mrt_ptr.clone());

        Ok(mrt_ptr)
    }

    pub fn get(&self, key: &str) -> Option<Rc<RefCell<RenderTarget>>> {
        self.targets.get(key).map(|item| item.value.clone())
    }

    pub fn get_mrt(&self, key: &str) -> Option<Rc<RefCell<MultipleRenderTarget>>> {
        self.mr_targets.get(key).cloned()
    }

    pub fn resize(&mut self) -> Result<(), VulkanError> {
        for (_name, target) in &mut self.targets {
            let extent = match target.size {
                RenderTargetSize::Window => self.default_extent,
                RenderTargetSize::Scaled(scale_factor) => Extent3D {
                    width: (self.default_extent.width as f32 * scale_factor) as u32,
                    height: (self.default_extent.height as f32 * scale_factor) as u32,
                    depth: 1,
                },
                _ => continue,
            };

            target.value.borrow_mut().resize(extent)?;
        }

        for (_name, target) in &mut self.mr_targets {
            target.borrow_mut().resize(self.default_extent)?;
        }

        Ok(())
    }

    fn build(&self, builder: RenderTargetBuilder) -> Result<RenderTarget, AppError> {
        let extent = match builder.size {
            RenderTargetSize::Window => self.default_extent,
            RenderTargetSize::Scaled(scale_factor) => {
                if scale_factor <= 0.0 {
                    return Err(AppError::Other(format!(
                        "Invalid scale factor for render target: '{scale_factor}'"
                    )));
                }
                Extent3D {
                    width: (self.default_extent.width as f32 * scale_factor) as u32,
                    height: (self.default_extent.height as f32 * scale_factor) as u32,
                    depth: 1,
                }
            }
            RenderTargetSize::Custom(x, y) => Extent3D {
                width: x,
                height: y,
                depth: 1,
            },
        };

        RenderTarget::new(
            self.device.clone(),
            extent,
            builder.format,
            builder.usage,
            builder.aspect,
        )
        .map_err(AppError::VulkanError)
    }
}
