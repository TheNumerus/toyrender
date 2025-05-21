use crate::vulkan::{
    CommandBuffer, Instance, IntoVulkanError, Surface, VulkanError, DEFERRED_HOST_OPS_EXTENSION, DYN_RENDER_EXTENSION,
    RT_ACCELERATION_EXTENSION, RT_PIPELINE_EXTENSION, RT_POSITION_FETCH_EXTENSION, SHADER_CLOCK_EXTENSION,
    SWAPCHAIN_EXTENSION,
};
use ash::vk;
use ash::vk::{ExtensionProperties, PhysicalDevice, PresentModeKHR, Queue, SurfaceCapabilitiesKHR, SurfaceFormatKHR};
use ash::Device as RawDevice;
use log::info;
use std::collections::HashSet;
use std::ffi::{CStr, CString};
use std::os::raw::c_void;
use std::rc::Rc;

pub struct Device {
    pub inner: RawDevice,
    pub graphics_queue: Queue,
    pub graphics_queue_family: usize,
    pub present_queue: Queue,
    pub present_queue_family: usize,
    pub compute_queue: Queue,
    pub compute_queue_family: usize,
    pub physical_device: PhysicalDevice,
    pub rt_properties: RtProperties,
    instance: Rc<Instance>,
}

impl Device {
    pub fn query_applicable(instance: &Instance, surface: &Surface) -> Result<Vec<DeviceQueryResult>, VulkanError> {
        let devices = unsafe {
            instance
                .inner
                .enumerate_physical_devices()
                .map_to_err("Cannot enumerate physical devices")?
        };

        let needed_extensions = [
            SWAPCHAIN_EXTENSION,
            RT_PIPELINE_EXTENSION,
            DEFERRED_HOST_OPS_EXTENSION,
            RT_ACCELERATION_EXTENSION,
            RT_POSITION_FETCH_EXTENSION,
            SHADER_CLOCK_EXTENSION,
            DYN_RENDER_EXTENSION,
        ];

        let mut res_devices = Vec::new();

        for device in devices {
            let extensions = unsafe {
                instance
                    .inner
                    .enumerate_device_extension_properties(device)
                    .map_to_err("cannot get device extensions")?
            };

            let mut missing_extensions = Vec::new();

            for ext in needed_extensions {
                if !Self::has_extension(&extensions, ext) {
                    missing_extensions.push(ext.to_string_lossy().to_string());
                }
            }

            let queue_families = Self::find_device_queue_families(device, instance, surface)?;
            let swapchain_support = Self::query_swapchain_support(device, surface)?;
            let (properties, _rt_properties, _bvh_properties) = Self::get_device_properties(instance, device);

            let name = unsafe {
                CStr::from_ptr(properties.properties.device_name.as_ptr())
                    .to_str()
                    .unwrap_or("'no device name'")
                    .to_owned()
            };

            if queue_families.graphics.is_some()
                && queue_families.present.is_some()
                && queue_families.compute.is_some()
                && missing_extensions.is_empty()
                && !swapchain_support.present_modes.is_empty()
                && !swapchain_support.formats.is_empty()
            {
                res_devices.push(DeviceQueryResult::Applicable(ApplicableDevice { name, device }))
            } else {
                res_devices.push(DeviceQueryResult::NotApplicable(NotApplicableDevice {
                    name,
                    missing_extensions,
                }))
            }
        }

        Ok(res_devices)
    }

    pub fn new(instance: Rc<Instance>, device: ApplicableDevice, surface: &Surface) -> Result<Self, VulkanError> {
        let queue_families = Self::find_device_queue_families(device.device, &instance, surface)?;

        let unique_queue_families = HashSet::from([
            queue_families.graphics.unwrap(),
            queue_families.present.unwrap(),
            queue_families.compute.unwrap(),
        ]);

        let queue_create_infos = unique_queue_families
            .iter()
            .map(|&index| vk::DeviceQueueCreateInfo {
                queue_family_index: index as u32,
                queue_count: 1,
                p_queue_priorities: &1.0,
                ..Default::default()
            })
            .collect::<Vec<_>>();

        let device_extensions = [
            SWAPCHAIN_EXTENSION,
            RT_ACCELERATION_EXTENSION,
            RT_PIPELINE_EXTENSION,
            DEFERRED_HOST_OPS_EXTENSION,
            RT_POSITION_FETCH_EXTENSION,
            SHADER_CLOCK_EXTENSION,
            DYN_RENDER_EXTENSION,
        ];

        let device_extensions_ptr = device_extensions.iter().map(|c| (*c).as_ptr()).collect::<Vec<_>>();

        let clock_info = vk::PhysicalDeviceShaderClockFeaturesKHR {
            shader_device_clock: 1,
            ..Default::default()
        };

        let rt_fetch_info = vk::PhysicalDeviceRayTracingPositionFetchFeaturesKHR {
            ray_tracing_position_fetch: 1,
            p_next: std::ptr::addr_of!(clock_info) as *mut c_void,
            ..Default::default()
        };

        let rt_acc_create_info = vk::PhysicalDeviceAccelerationStructureFeaturesKHR {
            acceleration_structure: 1,
            p_next: std::ptr::addr_of!(rt_fetch_info) as *mut c_void,
            ..Default::default()
        };

        let rt_create_info = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR {
            ray_tracing_pipeline: 1,
            p_next: std::ptr::addr_of!(rt_acc_create_info) as *mut c_void,
            ..Default::default()
        };

        let dyn_create_info = vk::PhysicalDeviceDynamicRenderingFeatures {
            dynamic_rendering: 1,
            p_next: std::ptr::addr_of!(rt_create_info) as *mut c_void,
            ..Default::default()
        };

        let vk_12_info = vk::PhysicalDeviceVulkan12Features {
            runtime_descriptor_array: 1,
            buffer_device_address: 1,
            descriptor_binding_partially_bound: 1,
            p_next: std::ptr::addr_of!(dyn_create_info) as *mut c_void,
            ..Default::default()
        };

        let create_info = vk::DeviceCreateInfo {
            p_queue_create_infos: queue_create_infos.as_ptr(),
            queue_create_info_count: queue_create_infos.len() as u32,
            pp_enabled_extension_names: device_extensions_ptr.as_ptr(),
            enabled_extension_count: device_extensions.len() as u32,
            p_enabled_features: &vk::PhysicalDeviceFeatures::default(),
            p_next: std::ptr::addr_of!(vk_12_info) as *const c_void,
            ..Default::default()
        };

        let inner = unsafe {
            instance
                .inner
                .create_device(device.device, &create_info, None)
                .map_to_err("cannot create logical device")?
        };

        let graphics_queue = unsafe { inner.get_device_queue(queue_families.graphics.unwrap() as u32, 0) };
        let present_queue = unsafe { inner.get_device_queue(queue_families.present.unwrap() as u32, 0) };
        let compute_queue = unsafe { inner.get_device_queue(queue_families.compute.unwrap() as u32, 0) };

        let (properties, rt_properties, bvh_properties) = Self::get_device_properties(&instance, device.device);

        let rt_properties = RtProperties {
            shader_group_base_alignment: rt_properties.shader_group_base_alignment,
            shader_group_handle_alignment: rt_properties.shader_group_handle_alignment,
            shader_group_handle_size: rt_properties.shader_group_handle_size,
            max_recursion: rt_properties.max_ray_recursion_depth,
            min_bvh_scratch_alignment: bvh_properties.min_acceleration_structure_scratch_offset_alignment,
        };

        info!(
            "{} - Vulkan {}.{}.{}",
            device.name,
            vk::api_version_major(properties.properties.api_version),
            vk::api_version_minor(properties.properties.api_version),
            vk::api_version_patch(properties.properties.api_version)
        );

        Ok(Self {
            inner,
            graphics_queue,
            graphics_queue_family: queue_families.graphics.unwrap(),
            present_queue,
            present_queue_family: queue_families.present.unwrap(),
            compute_queue,
            compute_queue_family: queue_families.compute.unwrap(),
            physical_device: device.device,
            rt_properties,
            instance,
        })
    }

    pub fn query_swapchain_support(
        physical_device: PhysicalDevice,
        surface: &Surface,
    ) -> Result<SwapChainSupport, VulkanError> {
        let capabilities = unsafe {
            surface
                .loader
                .get_physical_device_surface_capabilities(physical_device, surface.surface)
                .map_to_err("cannot get surface capabilities")?
        };

        let formats = unsafe {
            surface
                .loader
                .get_physical_device_surface_formats(physical_device, surface.surface)
                .map_to_err("cannot get device surface formats")?
        };

        let present_modes = unsafe {
            surface
                .loader
                .get_physical_device_surface_present_modes(physical_device, surface.surface)
                .map_to_err("cannot get device surface formats")?
        };

        Ok(SwapChainSupport {
            capabilities,
            formats,
            present_modes,
        })
    }

    pub fn find_memory_type_index(&self, type_filter: u32, property_flags: vk::MemoryPropertyFlags) -> Option<u32> {
        let memory_properties = Self::get_memory_properties(&self.instance, self.physical_device);

        for i in 0..memory_properties.memory_type_count {
            let property = memory_properties.memory_types[i as usize];

            let passes_filter = (type_filter & (1 << i)) != 0;
            let acceptable = property.property_flags.contains(property_flags);

            if passes_filter && acceptable {
                return Some(i);
            }
        }

        None
    }

    pub fn wait_idle(&self) -> Result<(), VulkanError> {
        unsafe { self.inner.device_wait_idle().map_to_err("Cannot wait for device idle") }
    }

    fn get_memory_properties(
        instance: &Instance,
        physical_device: PhysicalDevice,
    ) -> vk::PhysicalDeviceMemoryProperties {
        unsafe { instance.inner.get_physical_device_memory_properties(physical_device) }
    }

    fn get_device_properties(
        instance: &Instance,
        physical_device: PhysicalDevice,
    ) -> (
        vk::PhysicalDeviceProperties2,
        vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
        vk::PhysicalDeviceAccelerationStructurePropertiesKHR,
    ) {
        unsafe {
            let mut next_2 = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
            let mut next = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR {
                p_next: std::ptr::addr_of_mut!(next_2) as *mut c_void,
                ..Default::default()
            };
            let mut props = vk::PhysicalDeviceProperties2 {
                p_next: std::ptr::addr_of_mut!(next) as *mut c_void,
                ..Default::default()
            };
            instance
                .inner
                .get_physical_device_properties2(physical_device, &mut props);
            (props, next, next_2)
        }
    }

    fn find_device_queue_families(
        physical_device: PhysicalDevice,
        instance: &Instance,
        surface: &Surface,
    ) -> Result<QueueFamilyIndices, VulkanError> {
        let queue_families = unsafe {
            instance
                .inner
                .get_physical_device_queue_family_properties(physical_device)
        };

        let mut graphics_family = None;
        let mut present_family = None;
        let mut compute_family = None;

        for (i, queue) in queue_families.iter().enumerate() {
            if queue.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                graphics_family = Some(i);
            }

            if queue.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                compute_family = Some(i);
            }

            let has_present_support = unsafe {
                surface
                    .loader
                    .get_physical_device_surface_support(physical_device, i as u32, surface.surface)
                    .map_to_err("error getting present support")?
            };

            if has_present_support && present_family.is_none() {
                present_family = Some(i);
            }
        }

        Ok(QueueFamilyIndices {
            graphics: graphics_family,
            present: present_family,
            compute: compute_family,
        })
    }

    fn has_extension(extensions: &[ExtensionProperties], name: &CStr) -> bool {
        extensions.iter().any(|ext| {
            let ext_name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };
            name == ext_name
        })
    }

    pub fn name_object(&self, name_info: vk::DebugUtilsObjectNameInfoEXT) -> Result<(), VulkanError> {
        if !self.instance.markers_active() {
            return Ok(());
        }

        unsafe {
            self.instance
                .debug_utils
                .as_ref()
                .unwrap()
                .set_debug_utils_object_name(self.inner.handle(), &name_info)
                .map_to_err("Cannot name object")
        }
    }

    pub fn begin_label(&self, label: &str, command_buffer: &CommandBuffer) {
        if !self.instance.markers_active() {
            return;
        }

        let label = CString::new(label).expect("Invalid label name");

        let label_info = vk::DebugUtilsLabelEXT {
            p_label_name: label.as_ptr(),
            color: [1.0, 1.0, 1.0, 1.0],
            ..Default::default()
        };

        unsafe {
            self.instance
                .debug_utils
                .as_ref()
                .unwrap()
                .cmd_begin_debug_utils_label(command_buffer.inner, &label_info);
        }
    }

    pub fn end_label(&self, command_buffer: &CommandBuffer) {
        if !self.instance.markers_active() {
            return;
        }

        unsafe {
            self.instance
                .debug_utils
                .as_ref()
                .unwrap()
                .cmd_end_debug_utils_label(command_buffer.inner);
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { self.inner.destroy_device(None) };
    }
}

pub struct ApplicableDevice {
    pub name: String,
    pub device: PhysicalDevice,
}

pub struct NotApplicableDevice {
    pub name: String,
    pub missing_extensions: Vec<String>,
}

pub enum DeviceQueryResult {
    Applicable(ApplicableDevice),
    NotApplicable(NotApplicableDevice),
}

impl DeviceQueryResult {
    pub fn is_applicable(&self) -> bool {
        match self {
            DeviceQueryResult::Applicable(_) => true,
            DeviceQueryResult::NotApplicable { .. } => false,
        }
    }
}

struct QueueFamilyIndices {
    graphics: Option<usize>,
    present: Option<usize>,
    compute: Option<usize>,
}

pub struct SwapChainSupport {
    pub capabilities: SurfaceCapabilitiesKHR,
    pub formats: Vec<SurfaceFormatKHR>,
    pub present_modes: Vec<PresentModeKHR>,
}

pub struct RtProperties {
    pub shader_group_handle_size: u32,
    pub shader_group_handle_alignment: u32,
    pub shader_group_base_alignment: u32,
    pub max_recursion: u32,
    pub min_bvh_scratch_alignment: u32,
}
