use crate::vulkan::{DEBUG_UTILS_EXTENSION, IntoVulkanError};
use ash::ext::debug_utils::Instance as DebugUtils;
use ash::{Entry, Instance as RawInstance, vk};
use log::warn;
use std::ffi::{CStr, CString};

use super::{VALIDATION_LAYER, VulkanError};

pub struct Instance {
    pub inner: RawInstance,
    pub debug_utils: Option<DebugUtils>,
}

impl Instance {
    pub fn new(entry: &Entry, required_extensions: &[impl AsRef<str>]) -> Result<Self, VulkanError> {
        let app_name = CString::new("Toy Renderer").unwrap();
        let engine_name = CString::new("Jaky engine LMAO").unwrap();
        let app_info = vk::ApplicationInfo {
            p_application_name: app_name.as_ptr(),
            application_version: vk::make_api_version(0, 0, 1, 0),
            p_engine_name: engine_name.as_ptr(),
            engine_version: vk::make_api_version(0, 0, 1, 0),
            api_version: vk::API_VERSION_1_3,
            ..Default::default()
        };

        let mut required_instance_extensions = required_extensions
            .iter()
            .map(|e| CString::new(e.as_ref()).unwrap())
            .collect::<Vec<_>>();

        let instance_layers = get_instance_layers(entry)?;
        let instance_extensions = get_instance_extensions(entry)?;
        let markers_active = !instance_extensions.is_empty();
        required_instance_extensions.extend(instance_extensions);

        let instance_ext_ptrs = required_instance_extensions
            .iter()
            .map(|c| (*c).as_ptr())
            .collect::<Vec<_>>();
        let instance_layers_ptrs = instance_layers.iter().map(|c| (*c).as_ptr()).collect::<Vec<_>>();

        let create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            enabled_extension_count: instance_ext_ptrs.len() as u32,
            pp_enabled_extension_names: instance_ext_ptrs.as_ptr(),
            enabled_layer_count: instance_layers_ptrs.len() as u32,
            pp_enabled_layer_names: instance_layers_ptrs.as_ptr(),
            ..Default::default()
        };

        let instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .map_to_err("Cannot create instance")?
        };

        let debug_utils = if markers_active {
            Some(DebugUtils::new(entry, &instance))
        } else {
            None
        };

        Ok(Self {
            inner: instance,
            debug_utils,
        })
    }

    pub fn markers_active(&self) -> bool {
        self.debug_utils.is_some()
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe { self.inner.destroy_instance(None) };
    }
}

fn get_instance_layers(entry: &Entry) -> Result<Vec<CString>, VulkanError> {
    if cfg!(not(debug_assertions)) {
        return Ok(vec![]);
    }

    let layers = unsafe {
        entry
            .enumerate_instance_layer_properties()
            .map_to_err("cannot get possible layers")?
    };

    let has_validation = layers
        .iter()
        .filter(|l| {
            let name = unsafe { CStr::from_ptr(l.layer_name.as_ptr()) };

            name.to_str().unwrap() == VALIDATION_LAYER
        })
        .count()
        != 0;

    if has_validation {
        Ok(vec![CString::new(VALIDATION_LAYER).unwrap()])
    } else {
        warn!("{} not found", VALIDATION_LAYER);
        Ok(vec![])
    }
}

fn get_instance_extensions(entry: &Entry) -> Result<Vec<CString>, VulkanError> {
    if cfg!(not(debug_assertions)) && std::env::var("TOYRENDER_DEBUG").is_err() {
        return Ok(vec![]);
    }

    let layers = unsafe {
        entry
            .enumerate_instance_extension_properties(None)
            .map_to_err("cannot get possible layers")?
    };

    let has_markers = layers
        .iter()
        .filter(|l| {
            let name = unsafe { CStr::from_ptr(l.extension_name.as_ptr()) };

            name == DEBUG_UTILS_EXTENSION
        })
        .count()
        != 0;

    if has_markers {
        Ok(vec![CString::from(DEBUG_UTILS_EXTENSION)])
    } else {
        warn!("{} not found", DEBUG_UTILS_EXTENSION.to_string_lossy());
        Ok(vec![])
    }
}
