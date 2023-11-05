use ash::vk;
use std::borrow::Cow;
use std::ffi::CStr;
use thiserror::Error;

mod buffer;
mod command_buffer;
mod command_pool;
mod descriptor;
mod device;
mod image;
mod image_view;
mod instance;
mod pipeline;
mod render_pass;
mod shader;
mod surface;
mod swapchain;
mod sync;
mod uniform;
mod vertex;

pub use buffer::Buffer;
pub use command_buffer::CommandBuffer;
pub use command_pool::CommandPool;
pub use descriptor::{DescriptorPool, DescriptorSet};
pub use device::{Device, DeviceQueryResult, SwapChainSupport};
pub use image::Image;
pub use image_view::ImageView;
pub use instance::Instance;
pub use pipeline::Pipeline;
pub use render_pass::RenderPass;
pub use shader::{ShaderModule, ShaderStage};
pub use surface::Surface;
pub use swapchain::{framebuffer::SwapChainFramebuffer, SwapChain};
pub use sync::{Fence, Semaphore};
pub use uniform::UniformDescriptorSetLayout;
pub use vertex::{Vertex, VertexIndexBuffer};

pub const VALIDATION_LAYER: &str = "VK_LAYER_KHRONOS_validation";
pub const SWAPCHAIN_EXTENSION: &CStr = ash::extensions::khr::Swapchain::name();

#[derive(Error, Debug)]
#[error("{msg}: {code}")]
pub struct VulkanError {
    msg: Cow<'static, str>,
    code: vk::Result,
}

pub trait IntoVulkanError<T> {
    fn map_to_err(self, msg: impl Into<Cow<'static, str>>) -> Result<T, VulkanError>;
}

impl<T> IntoVulkanError<T> for ash::prelude::VkResult<T> {
    fn map_to_err(self, msg: impl Into<Cow<'static, str>>) -> Result<T, VulkanError> {
        self.map_err(|code| VulkanError { code, msg: msg.into() })
    }
}
