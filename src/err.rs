use crate::renderer::ShaderLoaderError;
use crate::vulkan::VulkanError;
use gpu_allocator::AllocationError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error(transparent)]
    VulkanError(#[from] VulkanError),
    #[error("{0}")]
    VulkanAllocatorError(#[from] AllocationError),
    #[error(transparent)]
    ShaderLoader(#[from] ShaderLoaderError),
    #[error("{0}")]
    Import(String),
    #[error("{0}")]
    Other(String),
}

impl From<gltf::Error> for AppError {
    fn from(value: gltf::Error) -> Self {
        Self::Import(value.to_string())
    }
}
