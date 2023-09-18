use crate::vulkan::{Instance, VulkanError};
use ash::extensions::khr::Surface as SurfaceLoader;
use ash::vk::{Handle, SurfaceKHR};
use ash::Entry;
use sdl2::video::VkSurfaceKHR;

pub struct Surface {
    pub loader: SurfaceLoader,
    pub surface: SurfaceKHR,
}

impl Surface {
    pub fn new(instance: &Instance, entry: &Entry, raw: VkSurfaceKHR) -> Result<Self, VulkanError> {
        let surface = SurfaceKHR::from_raw(raw);
        let loader = SurfaceLoader::new(&entry, &instance.instance);

        Ok(Self { surface, loader })
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_surface(self.surface, None) }
    }
}
