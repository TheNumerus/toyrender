use ash::khr::swapchain::Device as SwapchainLoader;
use ash::vk;
use ash::vk::{Extent2D, Image, SurfaceFormatKHR, SwapchainKHR};
use std::rc::Rc;
use vk::PresentModeKHR;

use super::{ImageView, Instance, VulkanError};
use crate::vulkan::{Device, IntoVulkanError, Semaphore, Surface, SwapChainSupport};

pub struct Swapchain {
    pub swapchain: SwapchainKHR,
    pub format: SurfaceFormatKHR,
    pub extent: Extent2D,
    pub loader: SwapchainLoader,
    pub images: Vec<Image>,
    device: Rc<Device>,
}

impl Swapchain {
    pub fn new(
        device: Rc<Device>,
        instance: &Instance,
        drawable_size: (u32, u32),
        surface: &Surface,
    ) -> Result<Self, VulkanError> {
        let swapchain_support = Device::query_swapchain_support(device.physical_device, surface)?;

        let swap_format = Self::choose_swap_surface_format(&swapchain_support);
        let swap_present_mode = Self::choose_swap_present_mode(&swapchain_support);
        let swap_extent = Self::choose_swap_extent(&swapchain_support, drawable_size);

        let image_count = swapchain_support.capabilities.min_image_count + 1;

        let indices = [device.graphics_queue_family as u32, device.present_queue_family as u32];

        let (sharing_mode, index_count, indices_ptr) = if device.present_queue_family != device.graphics_queue_family {
            (vk::SharingMode::CONCURRENT, 2, indices.as_ptr())
        } else {
            (vk::SharingMode::EXCLUSIVE, 0, std::ptr::null())
        };

        let swapchain_create_info = vk::SwapchainCreateInfoKHR {
            min_image_count: image_count,
            image_format: swap_format.format,
            image_color_space: swap_format.color_space,
            image_extent: swap_extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: sharing_mode,
            queue_family_index_count: index_count,
            p_queue_family_indices: indices_ptr,
            pre_transform: swapchain_support.capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode: swap_present_mode,
            clipped: vk::TRUE,
            old_swapchain: SwapchainKHR::null(),
            surface: surface.surface,
            ..Default::default()
        };

        let loader = SwapchainLoader::new(&instance.inner, &device.inner);

        let swapchain = unsafe {
            loader
                .create_swapchain(&swapchain_create_info, None)
                .map_to_err("Cannot create swapchain")?
        };

        let images = unsafe {
            loader
                .get_swapchain_images(swapchain)
                .map_to_err("Cannot get swapchain images")?
        };

        Ok(Self {
            swapchain,
            format: swap_format,
            extent: swap_extent,
            loader,
            images,
            device,
        })
    }

    pub fn recreate(
        &mut self,
        device: Rc<Device>,
        drawable_size: (u32, u32),
        surface: &Surface,
    ) -> Result<(), VulkanError> {
        self.images.clear();

        let swapchain_support = Device::query_swapchain_support(device.physical_device, surface)?;

        let swap_present_mode = Self::choose_swap_present_mode(&swapchain_support);
        self.extent = Self::choose_swap_extent(&swapchain_support, drawable_size);

        let image_count = swapchain_support.capabilities.min_image_count + 1;

        let indices = [device.graphics_queue_family as u32, device.present_queue_family as u32];

        let (sharing_mode, index_count, indices_ptr) = if device.present_queue_family != device.graphics_queue_family {
            (vk::SharingMode::CONCURRENT, 2, indices.as_ptr())
        } else {
            (vk::SharingMode::EXCLUSIVE, 0, std::ptr::null())
        };

        let old_swapchain = self.swapchain;

        let swapchain_create_info = vk::SwapchainCreateInfoKHR {
            min_image_count: image_count,
            image_format: self.format.format,
            image_color_space: self.format.color_space,
            image_extent: self.extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::TRANSFER_DST,
            image_sharing_mode: sharing_mode,
            queue_family_index_count: index_count,
            p_queue_family_indices: indices_ptr,
            pre_transform: swapchain_support.capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode: swap_present_mode,
            clipped: vk::TRUE,
            old_swapchain,
            surface: surface.surface,
            ..Default::default()
        };

        self.swapchain = unsafe {
            self.loader
                .create_swapchain(&swapchain_create_info, None)
                .map_to_err("Cannot create swapchain")?
        };

        unsafe { self.loader.destroy_swapchain(old_swapchain, None) };

        self.images = unsafe {
            self.loader
                .get_swapchain_images(self.swapchain)
                .map_to_err("Cannot get swapchain images")?
        };

        Ok(())
    }

    pub fn acquire_next_image(&self, semaphore: &Semaphore) -> Result<(u32, bool), VulkanError> {
        unsafe {
            self.loader
                .acquire_next_image(self.swapchain, u64::MAX, semaphore.inner, vk::Fence::null())
                .map_to_err("cannot acquire image")
        }
    }

    fn choose_swap_surface_format(swapchain_support: &SwapChainSupport) -> SurfaceFormatKHR {
        for format in &swapchain_support.formats {
            if format.format == vk::Format::B8G8R8A8_SRGB && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR {
                return *format;
            }
        }

        swapchain_support.formats[0]
    }

    fn choose_swap_present_mode(swapchain_support: &SwapChainSupport) -> PresentModeKHR {
        for present_mode in &swapchain_support.present_modes {
            if *present_mode == PresentModeKHR::IMMEDIATE {
                return *present_mode;
            }
        }

        PresentModeKHR::IMMEDIATE
    }

    fn choose_swap_extent(swapchain_support: &SwapChainSupport, drawable_size: (u32, u32)) -> Extent2D {
        if swapchain_support.capabilities.current_extent.width != u32::MAX {
            swapchain_support.capabilities.current_extent
        } else {
            let (width, height) = drawable_size;

            Extent2D {
                width: width.clamp(
                    swapchain_support.capabilities.min_image_extent.width,
                    swapchain_support.capabilities.max_image_extent.width,
                ),
                height: height.clamp(
                    swapchain_support.capabilities.min_image_extent.height,
                    swapchain_support.capabilities.max_image_extent.height,
                ),
            }
        }
    }

    pub fn create_image_views(&self) -> Result<Vec<ImageView>, VulkanError> {
        self.images
            .iter()
            .map(|&image| {
                ImageView::new(
                    self.device.clone(),
                    image,
                    self.format.format,
                    vk::ImageAspectFlags::COLOR,
                )
            })
            .collect::<Result<Vec<_>, _>>()
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_swapchain(self.swapchain, None) };
    }
}
