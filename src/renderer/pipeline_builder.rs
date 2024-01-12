use crate::renderer::shader_loader::ShaderLoader;
use crate::vulkan::Device;
use std::rc::Rc;

pub struct PipelineBuilder {}

impl PipelineBuilder {
    pub fn build(shader_loader: ShaderLoader, device: Rc<Device>) -> Self {
        Self {}
    }
}
