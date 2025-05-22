use crate::err::AppError;
use crate::vulkan::{Device, Image, ImageView, Sampler, VulkanError};
use ash::vk;
use ash::vk::{Extent3D, Handle};
use std::cell::{Ref, RefCell};
use std::collections::HashMap;
use std::ffi::CString;
use std::rc::Rc;

pub struct RenderTarget {
    pub image: Image,
    pub view: ImageView,
    sampler: Rc<Sampler>,
    format: vk::Format,
    pub usage: vk::ImageUsageFlags,
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
        sampler: Rc<Sampler>,
    ) -> Result<Self, VulkanError> {
        let image = Image::new(device.clone(), format, extent, usage)?;
        let view = ImageView::new(device.clone(), image.inner, format, aspect)?;

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

    pub fn with_sampled(mut self) -> Self {
        self.usage |= vk::ImageUsageFlags::SAMPLED;
        self
    }

    pub fn with_color_attachment(mut self) -> Self {
        self.usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        self
    }
}

pub(crate) struct RenderTargetItem {
    pub value: Rc<RefCell<RenderTarget>>,
    pub size: RenderTargetSize,
}

pub struct RenderTargets {
    pub targets: HashMap<String, RenderTargetItem>,
    device: Rc<Device>,
    default_extent: Extent3D,
    default_sampler: Rc<Sampler>,
}

impl RenderTargets {
    pub fn new(device: Rc<Device>, default_extent: Extent3D, default_sampler: Rc<Sampler>) -> Self {
        Self {
            targets: HashMap::new(),
            device,
            default_extent,
            default_sampler,
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

    pub fn get(&self, key: &str) -> Option<Rc<RefCell<RenderTarget>>> {
        self.targets.get(key).map(|item| item.value.clone())
    }

    pub fn get_ref(&self, key: &str) -> Option<Ref<RenderTarget>> {
        self.targets.get(key).map(|item| item.value.borrow())
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

        let mut rt = RenderTarget::new(
            self.device.clone(),
            extent,
            builder.format,
            builder.usage,
            builder.aspect,
            self.default_sampler.clone(),
        )
        .map_err(AppError::VulkanError)?;

        rt.set_names(&builder.name)?;

        Ok(rt)
    }
}
