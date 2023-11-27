use ash::vk;
use std::borrow::Cow;
use std::ffi::CStr;
use thiserror::Error;

mod buffer;
mod command_buffer;
mod command_pool;
mod descriptor;
mod device;
mod framebuffer;
mod image;
mod image_view;
mod instance;
mod mesh;
mod pipeline;
mod render_pass;
mod rt;
mod sampler;
mod shader;
mod surface;
mod swapchain;
mod sync;
mod vertex;

pub use buffer::Buffer;
pub use command_buffer::CommandBuffer;
pub use command_pool::CommandPool;
pub use descriptor::{DescriptorPool, DescriptorSet, DescriptorSetLayout};
pub use device::{Device, DeviceQueryResult, SwapChainSupport};
pub use framebuffer::Framebuffer;
pub use image::Image;
pub use image_view::ImageView;
pub use instance::Instance;
pub use mesh::VulkanMesh;
pub use pipeline::Pipeline;
pub use render_pass::RenderPass;
pub use sampler::Sampler;
pub use shader::{ShaderModule, ShaderStage};
pub use surface::Surface;
pub use swapchain::SwapChain;
pub use sync::{Fence, Semaphore};
pub use vertex::{Vertex, VertexIndexBuffer};

pub const VALIDATION_LAYER: &str = "VK_LAYER_KHRONOS_validation";
pub const SWAPCHAIN_EXTENSION: &CStr = ash::extensions::khr::Swapchain::name();
pub const RT_ACCELERATION_EXTENSION: &CStr = ash::extensions::khr::AccelerationStructure::name();
pub const RT_PIPELINE_EXTENSION: &CStr = ash::extensions::khr::RayTracingPipeline::name();
pub const DEFERRED_HOST_OPS_EXTENSION: &CStr = ash::extensions::khr::DeferredHostOperations::name();

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
