use crate::err::AppError;
use crate::renderer::shader_loader::{PipelineStub, ShaderLoader};
use crate::vulkan::{Compute, Device, Graphics, Pipeline, RayTracingPipeline, Rt, ShaderModule, ShaderStage};
use ash::vk;
use std::collections::HashMap;
use std::rc::Rc;

pub struct PipelineBuilder {
    shader_loader: ShaderLoader,
    device: Rc<Device>,
    rt_pipeline_ext: Rc<RayTracingPipeline>,
    graphic_pipelines: HashMap<String, Pipeline<Graphics>>,
    rt_pipelines: HashMap<String, Pipeline<Rt>>,
    compute_pipelines: HashMap<String, Pipeline<Compute>>,
}

impl PipelineBuilder {
    pub fn new(shader_loader: ShaderLoader, device: Rc<Device>, rt_pipeline_ext: Rc<RayTracingPipeline>) -> Self {
        Self {
            shader_loader,
            device,
            rt_pipeline_ext,
            graphic_pipelines: Default::default(),
            rt_pipelines: Default::default(),
            compute_pipelines: Default::default(),
        }
    }

    pub fn build_graphics(
        &mut self,
        name: impl AsRef<str>,
        desc_layouts: &[vk::DescriptorSetLayout],
        attachment_formats: &[vk::Format],
        push_consts_size: u32,
    ) -> Result<(), AppError> {
        match self.shader_loader.manifest.stubs.get(name.as_ref()) {
            Some(stub) => {
                if let PipelineStub::Graphics {
                    vertex,
                    fragment,
                    use_depth,
                } = stub
                {
                    let vertex_module = ShaderModule::new(
                        self.shader_loader.shaders.get(vertex).unwrap().as_ref(),
                        self.device.clone(),
                        ShaderStage::Vertex,
                    )?;
                    vertex_module.set_name(vertex.to_owned())?;
                    let fragment_module = ShaderModule::new(
                        self.shader_loader.shaders.get(fragment).unwrap().as_ref(),
                        self.device.clone(),
                        ShaderStage::Fragment,
                    )?;
                    fragment_module.set_name(fragment.to_owned())?;

                    let stages = [vertex_module.stage_info(), fragment_module.stage_info()];

                    let pipeline = Pipeline::new_graphics(
                        self.device.clone(),
                        &stages,
                        desc_layouts,
                        attachment_formats,
                        *use_depth,
                        push_consts_size,
                    )?;
                    pipeline.set_name(name.as_ref().to_owned())?;

                    self.graphic_pipelines.insert(name.as_ref().to_owned(), pipeline);

                    Ok(())
                } else {
                    Err(AppError::Other(format!(
                        "Wrong type of pipeline stub: {}",
                        name.as_ref()
                    )))
                }
            }
            None => Err(AppError::Other(format!("Missing pipeline stub: {}", name.as_ref()))),
        }
    }

    pub fn build_compute(
        &mut self,
        name: impl AsRef<str>,
        desc_layouts: &[vk::DescriptorSetLayout],
        push_consts_size: u32,
    ) -> Result<(), AppError> {
        match self.shader_loader.manifest.stubs.get(name.as_ref()) {
            Some(stub) => {
                if let PipelineStub::Compute { compute } = stub {
                    let module = ShaderModule::new(
                        self.shader_loader.shaders.get(compute).unwrap().as_ref(),
                        self.device.clone(),
                        ShaderStage::Compute,
                    )?;
                    module.set_name(compute.to_owned())?;

                    let pipeline = Pipeline::new_compute(
                        self.device.clone(),
                        module.stage_info(),
                        desc_layouts,
                        push_consts_size,
                    )?;
                    pipeline.set_name(name.as_ref().to_owned())?;

                    self.compute_pipelines.insert(name.as_ref().to_owned(), pipeline);

                    Ok(())
                } else {
                    Err(AppError::Other(format!(
                        "Wrong type of pipeline stub: {}",
                        name.as_ref()
                    )))
                }
            }
            None => Err(AppError::Other(format!("Missing pipeline stub: {}", name.as_ref()))),
        }
    }

    pub fn build_rt(
        &mut self,
        name: impl AsRef<str>,
        desc_layouts: &[vk::DescriptorSetLayout],
        push_consts_size: u32,
    ) -> Result<(), AppError> {
        match self.shader_loader.manifest.stubs.get(name.as_ref()) {
            Some(stub) => {
                if let PipelineStub::Rt { raygen, closest, miss } = stub {
                    let raygen_module = ShaderModule::new(
                        self.shader_loader.shaders.get(raygen).unwrap().as_ref(),
                        self.device.clone(),
                        ShaderStage::RayGen,
                    )?;
                    raygen_module.set_name(raygen.to_owned())?;

                    let miss_module = ShaderModule::new(
                        self.shader_loader.shaders.get(miss).unwrap().as_ref(),
                        self.device.clone(),
                        ShaderStage::RayMiss,
                    )?;
                    miss_module.set_name(miss.to_owned())?;

                    let hit_module = ShaderModule::new(
                        self.shader_loader.shaders.get(closest).unwrap().as_ref(),
                        self.device.clone(),
                        ShaderStage::RayClosestHit,
                    )?;
                    hit_module.set_name(closest.to_owned())?;

                    let rt_stages = [
                        raygen_module.stage_info(),
                        miss_module.stage_info(),
                        hit_module.stage_info(),
                    ];

                    let pipeline = Pipeline::new_rt(
                        self.device.clone(),
                        &*self.rt_pipeline_ext,
                        &rt_stages,
                        desc_layouts,
                        push_consts_size,
                    )?;
                    pipeline.set_name(name.as_ref().to_owned())?;

                    self.rt_pipelines.insert(name.as_ref().to_owned(), pipeline);
                    Ok(())
                } else {
                    Err(AppError::Other(format!(
                        "Wrong type of pipeline stub: {}",
                        name.as_ref()
                    )))
                }
            }
            None => Err(AppError::Other(format!("Missing pipeline stub: {}", name.as_ref()))),
        }
    }

    pub fn get_graphics(&self, key: &str) -> Option<&Pipeline<Graphics>> {
        self.graphic_pipelines.get(key)
    }

    pub fn get_compute(&self, key: &str) -> Option<&Pipeline<Compute>> {
        self.compute_pipelines.get(key)
    }

    pub fn get_rt(&self, key: &str) -> Option<&Pipeline<Rt>> {
        self.rt_pipelines.get(key)
    }
}
