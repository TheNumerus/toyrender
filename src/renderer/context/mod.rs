use crate::err::AppError;
use crate::renderer::MAX_FRAMES_IN_FLIGHT;
use crate::vulkan::{
    CommandPool, Device, DeviceQueryResult, ImageView, Instance, RayTracingAs, RayTracingPipeline, Surface, Swapchain,
};
use ash::Entry;
use ash::vk::Handle;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use sdl2::video::Window;
use std::cell::RefCell;
use std::fmt::Write;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

pub struct VulkanContext {
    pub instance: Rc<Instance>,
    pub device: Rc<Device>,
    pub allocator: Arc<Mutex<Allocator>>,
    pub swap_chain: RefCell<Swapchain>,
    pub swap_chain_image_views: RefCell<Vec<ImageView>>,
    pub surface: Surface,
    pub rt_pipeline_ext: Rc<RayTracingPipeline>,
    pub rt_acc_struct_ext: Rc<RayTracingAs>,
    pub graphics_command_pool: CommandPool,
    pub compute_command_pool: CommandPool,
    pub imgui_renderer: RefCell<imgui_rs_vulkan_renderer::Renderer>,
}

impl VulkanContext {
    pub fn init(window: &Window, imgui: &mut imgui::Context) -> Result<Self, AppError> {
        let entry = unsafe { Entry::load().expect("cannot load vulkan entry") };

        let instance = Rc::new(Instance::new(&entry, &window.vulkan_instance_extensions().unwrap())?);

        let surface = window
            .vulkan_create_surface(instance.inner.handle().as_raw() as usize)
            .map_err(AppError::Other)?;
        let surface = Surface::new(&instance, &entry, surface)?;

        let device = Self::init_device(instance.clone(), &surface)?;

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.inner.clone(),
            device: device.inner.clone(),
            physical_device: device.physical_device,
            debug_settings: Default::default(),
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })?;

        let allocator = Arc::new(Mutex::new(allocator));

        let rt_pipeline_ext = Rc::new(RayTracingPipeline::new(&instance, &device)?);
        let rt_acc_struct_ext = Rc::new(RayTracingAs::new(&instance, &device)?);

        let swap_chain = RefCell::new(Swapchain::new(
            device.clone(),
            &instance,
            window.drawable_size(),
            &surface,
        )?);
        let swap_chain_image_views = RefCell::new(swap_chain.borrow_mut().create_image_views()?);

        let graphics_command_pool = CommandPool::new_graphics(device.clone())?;
        let compute_command_pool = CommandPool::new_compute(device.clone())?;

        let imgui_renderer = imgui_rs_vulkan_renderer::Renderer::with_gpu_allocator(
            allocator.clone(),
            device.inner.clone(),
            device.graphics_queue,
            graphics_command_pool.inner,
            imgui_rs_vulkan_renderer::DynamicRendering {
                color_attachment_format: swap_chain.borrow().format.format,
                depth_attachment_format: None,
            },
            imgui,
            Some(imgui_rs_vulkan_renderer::Options {
                in_flight_frames: MAX_FRAMES_IN_FLIGHT,
                ..Default::default()
            }),
        )
        .map_err(|e| AppError::Other(format!("Failed to create imgui renderer: {}", e)))?;

        Ok(Self {
            instance,
            device: device.clone(),
            allocator,
            surface,
            rt_pipeline_ext,
            rt_acc_struct_ext,
            swap_chain,
            swap_chain_image_views,
            graphics_command_pool,
            compute_command_pool,
            imgui_renderer: RefCell::new(imgui_renderer),
        })
    }

    fn init_device(instance: Rc<Instance>, surface: &Surface) -> Result<Rc<Device>, AppError> {
        let devices = Device::query_applicable(&instance, surface)?;
        if devices.is_empty() {
            return Err(AppError::Other("No GPUs with Vulkan support found".into()));
        }

        let no_applicable = devices.iter().all(|d| !d.is_applicable());

        if no_applicable {
            let mut message = String::from("No applicable device found: \n");
            for device in devices {
                if let DeviceQueryResult::NotApplicable(device) = device {
                    let extensions = device.missing_extensions.iter().fold(String::new(), |mut out, ext| {
                        let _ = write!(out, "\t\t - {}", ext);
                        out
                    });

                    let device_msg = format!("\t{}\n\t\tMissing extensions:\n{}", device.name, &extensions);
                    message.push_str(&device_msg);
                }
            }

            Err(AppError::Other(message))
        } else {
            for device in devices {
                if let DeviceQueryResult::Applicable(device) = device {
                    return Ok(Rc::new(Device::new(instance, device, surface)?));
                }
            }
            unreachable!()
        }
    }

    pub fn recreate_swapchain(&self, drawable_size: (u32, u32)) -> Result<(), AppError> {
        self.device.wait_idle()?;

        // need to drop before creating new ones
        self.swap_chain_image_views.borrow_mut().clear();

        self.swap_chain
            .borrow_mut()
            .recreate(self.device.clone(), drawable_size, &self.surface)?;

        self.swap_chain_image_views
            .replace(self.swap_chain.borrow_mut().create_image_views()?);

        Ok(())
    }
}
