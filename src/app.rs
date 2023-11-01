use crate::err::AppError;
use crate::vulkan::Instance;
use ash::vk::Handle;
use sdl2::video::{VkSurfaceKHR, Window};
use sdl2::{EventPump, Sdl};

pub struct App {
    pub sdl_context: Sdl,
    pub window: Window,
    pub event_pump: EventPump,
}

impl App {
    pub fn create() -> Self {
        let sdl_context = sdl2::init().expect("cannot init sdl2");

        let video_subsystem = sdl_context.video().expect("cannot init video");

        let mut window = video_subsystem
            .window("Vulkan Demo", 1600, 900)
            .allow_highdpi()
            .resizable()
            .position_centered()
            .vulkan()
            .build()
            .expect("cannot build window");

        window.set_minimum_size(120, 40).expect("cannot set min size");

        let event_pump = sdl_context.event_pump().expect("cannot get event pump");

        Self {
            sdl_context,
            window,
            event_pump,
        }
    }

    pub fn create_vulkan_surface(&self, instance: &Instance) -> Result<VkSurfaceKHR, AppError> {
        self.window
            .vulkan_create_surface(instance.inner.handle().as_raw() as usize)
            .map_err(AppError::Other)
    }
}
