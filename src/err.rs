use crate::vulkan::VulkanError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("{0}")]
    VulkanError(VulkanError),
    #[error("{0}")]
    Import(String),
    #[error("{0}")]
    Other(String),
}

impl From<VulkanError> for AppError {
    fn from(value: VulkanError) -> Self {
        Self::VulkanError(value)
    }
}

impl From<gltf::Error> for AppError {
    fn from(value: gltf::Error) -> Self {
        Self::Import(value.to_string())
    }
}
