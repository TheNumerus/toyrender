use crate::vulkan::{Instance, Surface, VulkanError, SWAPCHAIN_EXTENSION};
use ash::vk;
use ash::vk::{PhysicalDevice, PresentModeKHR, Queue, SurfaceCapabilitiesKHR, SurfaceFormatKHR};
use ash::Device as RawDevice;
use std::collections::HashSet;
use std::ffi::CStr;

pub struct Device {
    pub inner: RawDevice,
    pub graphics_queue: Queue,
    pub graphics_queue_family: usize,
    pub present_queue: Queue,
    pub present_queue_family: usize,
    pub physical_device: PhysicalDevice,
}

impl Device {
    pub fn query_applicable(instance: &Instance, surface: &Surface) -> DeviceQueryResult {
        let devices = unsafe { instance.instance.enumerate_physical_devices() };

        let devices = match devices {
            Ok(d) => d,
            Err(code) => {
                return DeviceQueryResult::VulkanError(VulkanError {
                    code,
                    msg: "Cannot enumerate physical devices".into(),
                })
            }
        };

        if devices.is_empty() {
            return DeviceQueryResult::NoDevice;
        }

        let mut applicable_devices = Vec::new();

        for device in devices {
            let extensions = unsafe {
                instance
                    .instance
                    .enumerate_device_extension_properties(device)
                    .expect("cannot get device extensions")
            };

            let has_swapchain = extensions
                .iter()
                .filter(|ext| {
                    let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };

                    name == SWAPCHAIN_EXTENSION
                })
                .count()
                != 0;

            let queue_families = Self::find_device_queue_families(device, instance, surface);
            let queue_families = match queue_families {
                Ok(q) => q,
                Err(e) => return DeviceQueryResult::VulkanError(e),
            };

            let swapchain_support = Self::query_swapchain_support(device, surface);

            if queue_families.graphics.is_some()
                && queue_families.present.is_some()
                && has_swapchain
                && !swapchain_support.present_modes.is_empty()
                && !swapchain_support.formats.is_empty()
            {
                applicable_devices.push(device);
            }
        }

        if !applicable_devices.is_empty() {
            DeviceQueryResult::ApplicableDevices(applicable_devices)
        } else {
            DeviceQueryResult::NoApplicableDevice
        }
    }

    pub fn new(instance: &Instance, physical_device: PhysicalDevice, surface: &Surface) -> Result<Self, VulkanError> {
        let queue_families = Self::find_device_queue_families(physical_device, instance, surface)?;

        let unique_queue_families = HashSet::from([queue_families.graphics.unwrap(), queue_families.present.unwrap()]);

        let queue_create_infos = unique_queue_families
            .iter()
            .map(|&index| vk::DeviceQueueCreateInfo {
                queue_family_index: index as u32,
                queue_count: 1,
                p_queue_priorities: &1.0,
                ..Default::default()
            })
            .collect::<Vec<_>>();

        let device_extensions = [SWAPCHAIN_EXTENSION];
        let device_extensions_ptr = device_extensions.iter().map(|c| (*c).as_ptr()).collect::<Vec<_>>();

        let create_info = vk::DeviceCreateInfo {
            p_queue_create_infos: queue_create_infos.as_ptr(),
            queue_create_info_count: queue_create_infos.len() as u32,
            pp_enabled_extension_names: device_extensions_ptr.as_ptr(),
            enabled_extension_count: device_extensions.len() as u32,
            p_enabled_features: &vk::PhysicalDeviceFeatures::default(),
            ..Default::default()
        };

        let inner = unsafe {
            instance
                .instance
                .create_device(physical_device, &create_info, None)
                .expect("cannot create logical device")
        };

        let graphics_queue = unsafe { inner.get_device_queue(queue_families.graphics.unwrap() as u32, 0) };
        let present_queue = unsafe { inner.get_device_queue(queue_families.present.unwrap() as u32, 0) };

        Ok(Self {
            inner,
            graphics_queue,
            graphics_queue_family: queue_families.graphics.unwrap(),
            present_queue,
            present_queue_family: queue_families.present.unwrap(),
            physical_device,
        })
    }

    pub fn query_swapchain_support(physical_device: PhysicalDevice, surface: &Surface) -> SwapChainSupport {
        let capabilities = unsafe {
            surface
                .loader
                .get_physical_device_surface_capabilities(physical_device, surface.surface)
                .expect("cannot get surface capabilities")
        };

        let formats = unsafe {
            surface
                .loader
                .get_physical_device_surface_formats(physical_device, surface.surface)
                .expect("cannot get device surface formats")
        };

        let present_modes = unsafe {
            surface
                .loader
                .get_physical_device_surface_present_modes(physical_device, surface.surface)
                .expect("cannot get device surface formats")
        };

        SwapChainSupport {
            capabilities,
            formats,
            present_modes,
        }
    }

    pub fn wait_idle(&self) -> Result<(), VulkanError> {
        unsafe {
            self.inner.device_wait_idle().map_err(|code| VulkanError {
                code,
                msg: "Cannot wait for device idle".into(),
            })
        }
    }

    fn find_device_queue_families(
        physical_device: PhysicalDevice,
        instance: &Instance,
        surface: &Surface,
    ) -> Result<QueueFamilyIndices, VulkanError> {
        let queue_families = unsafe {
            instance
                .instance
                .get_physical_device_queue_family_properties(physical_device)
        };

        let mut graphics_family = None;
        let mut present_family = None;

        for (i, queue) in queue_families.iter().enumerate() {
            if queue.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                graphics_family = Some(i);
            }

            let has_present_support = unsafe {
                surface
                    .loader
                    .get_physical_device_surface_support(physical_device, i as u32, surface.surface)
                    .map_err(|code| VulkanError {
                        code,
                        msg: "error getting present support".into(),
                    })?
            };

            if has_present_support {
                present_family = Some(i);
            }
        }

        Ok(QueueFamilyIndices {
            graphics: graphics_family,
            present: present_family,
        })
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { self.inner.destroy_device(None) };
    }
}

pub enum DeviceQueryResult {
    NoDevice,
    NoApplicableDevice,
    ApplicableDevices(Vec<PhysicalDevice>),
    VulkanError(VulkanError),
}

struct QueueFamilyIndices {
    graphics: Option<usize>,
    present: Option<usize>,
}

pub struct SwapChainSupport {
    pub capabilities: SurfaceCapabilitiesKHR,
    pub formats: Vec<SurfaceFormatKHR>,
    pub present_modes: Vec<PresentModeKHR>,
}
