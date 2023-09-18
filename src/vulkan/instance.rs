use ash::{vk, Entry, Instance as RawInstance};
use std::ffi::{CStr, CString};

use super::{VulkanError, VALIDATION_LAYER};

pub struct Instance {
    pub instance: RawInstance,
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
            api_version: vk::API_VERSION_1_2,
            ..Default::default()
        };

        let instance_extensions = required_extensions
            .iter()
            .map(|e| CString::new(e.as_ref()).unwrap())
            .collect::<Vec<_>>();

        let instance_layers = get_instance_layers(entry);

        let instance_ext_ptrs = instance_extensions.iter().map(|c| (*c).as_ptr()).collect::<Vec<_>>();
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
            entry.create_instance(&create_info, None).map_err(|code| VulkanError {
                msg: "Cannot create instance".into(),
                code,
            })?
        };

        Ok(Self { instance })
    }

    #[allow(dead_code)]
    pub fn list_instance_extensions(entry: &Entry) {
        let extensions = entry
            .enumerate_instance_extension_properties(None)
            .expect("cannot get possible extensions");

        println!("Listing available instance extensions...");
        for ext in extensions {
            let name = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) };

            println!("{}", name.to_str().unwrap());
        }
    }

    #[allow(dead_code)]
    pub fn list_instance_layers(entry: &Entry) {
        let layers = entry
            .enumerate_instance_layer_properties()
            .expect("cannot get possible layers");

        println!("Listing available instance layers...");
        for layer in layers {
            let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
            let desc = unsafe { CStr::from_ptr(layer.description.as_ptr()) };

            println!("{} - {}", name.to_str().unwrap(), desc.to_str().unwrap());
        }
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe { self.instance.destroy_instance(None) };
    }
}

fn get_instance_layers(entry: &Entry) -> Vec<CString> {
    #[cfg(not(debug_assertions))]
    {
        return Vec::new();
    }

    let layers = entry
        .enumerate_instance_layer_properties()
        .expect("cannot get possible layers");

    let has_validation = layers
        .iter()
        .filter(|l| {
            let name = unsafe { CStr::from_ptr(l.layer_name.as_ptr()) };

            name.to_str().unwrap() == VALIDATION_LAYER
        })
        .count()
        != 0;

    if has_validation {
        vec![CString::new(VALIDATION_LAYER).unwrap()]
    } else {
        eprintln!("{} not found", VALIDATION_LAYER);
        vec![]
    }
}
