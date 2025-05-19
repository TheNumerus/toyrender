use crate::err::AppError;
use crate::renderer::descriptors::{DescLayout, RendererDescriptors};
use crate::renderer::shader_loader::{PipelineStub, ShaderLoader, SpirVFile};
use crate::renderer::RenderPasses;
use crate::vulkan::{
    Compute, Device, Graphics, Pipeline, RayTracingPipeline, Rt, ShaderModule, ShaderStage, VulkanError,
};
use std::collections::HashMap;
use std::rc::Rc;

pub struct PipelineBuilder {
    graphic_pipelines: HashMap<String, Pipeline<Graphics>>,
    rt_pipelines: HashMap<String, Pipeline<Rt>>,
    compute_pipelines: HashMap<String, Pipeline<Compute>>,
}

impl PipelineBuilder {
    pub fn build(
        shader_loader: ShaderLoader,
        device: Rc<Device>,
        rt_pipeline_ext: &RayTracingPipeline,
        descriptors: &RendererDescriptors,
        render_passes: &RenderPasses,
    ) -> Result<Self, AppError> {
        let mut modules: HashMap<String, Rc<ShaderModule>> = HashMap::new();

        let mut graphic_pipelines = HashMap::new();
        let mut rt_pipelines = HashMap::new();
        let mut compute_pipelines = HashMap::new();

        for (name, stub) in shader_loader.manifest.stubs {
            match stub {
                PipelineStub::Graphics {
                    vertex,
                    fragment,
                    render_pass,
                } => {
                    let render_pass_handle = match &render_pass {
                        None => None,
                        Some(key) => render_passes.passes.get(key),
                    };

                    let attachments = match &render_pass {
                        Some(s) if s == "gb" => 2,
                        _ => 1,
                    };

                    let descriptor_layouts = match &render_pass {
                        Some(s) if s == "gb" => vec![descriptors.get_layout(DescLayout::Global).inner],
                        _ => vec![
                            descriptors.get_layout(DescLayout::Global).inner,
                            descriptors.get_layout(DescLayout::Image).inner,
                        ],
                    };

                    let vertex_module = Self::get_or_compile(
                        &mut modules,
                        &shader_loader.shaders,
                        device.clone(),
                        vertex,
                        ShaderStage::Vertex,
                    )?;

                    let fragment_module = Self::get_or_compile(
                        &mut modules,
                        &shader_loader.shaders,
                        device.clone(),
                        fragment,
                        ShaderStage::Fragment,
                    )?;

                    let stages = [vertex_module.stage_info(), fragment_module.stage_info()];

                    let pipeline = Pipeline::new_graphics(
                        device.clone(),
                        render_pass_handle.map(|a| a.as_ref()),
                        &stages,
                        &descriptor_layouts,
                        attachments,
                    )?;

                    graphic_pipelines.insert(name, pipeline);
                }
                PipelineStub::Rt { raygen, closest, miss } => {
                    let raygen_module = Self::get_or_compile(
                        &mut modules,
                        &shader_loader.shaders,
                        device.clone(),
                        raygen,
                        ShaderStage::RayGen,
                    )?;

                    let miss_module = Self::get_or_compile(
                        &mut modules,
                        &shader_loader.shaders,
                        device.clone(),
                        miss,
                        ShaderStage::RayMiss,
                    )?;

                    let hit_module = Self::get_or_compile(
                        &mut modules,
                        &shader_loader.shaders,
                        device.clone(),
                        closest,
                        ShaderStage::RayClosestHit,
                    )?;

                    let rt_stages = [
                        raygen_module.stage_info(),
                        miss_module.stage_info(),
                        hit_module.stage_info(),
                    ];

                    let pipeline = Pipeline::new_rt(
                        device.clone(),
                        rt_pipeline_ext,
                        &rt_stages,
                        &[
                            descriptors.get_layout(DescLayout::Global).inner,
                            descriptors.get_layout(DescLayout::Compute).inner,
                        ],
                    )?;

                    rt_pipelines.insert(name, pipeline);
                }
                PipelineStub::Compute { compute } => {
                    let compute = Self::get_or_compile(
                        &mut modules,
                        &shader_loader.shaders,
                        device.clone(),
                        compute,
                        ShaderStage::Compute,
                    )?;

                    let pipeline = Pipeline::new_compute(
                        device.clone(),
                        compute.stage_info(),
                        &[
                            descriptors.get_layout(DescLayout::Global).inner,
                            descriptors.get_layout(DescLayout::Compute).inner,
                        ],
                    )?;

                    compute_pipelines.insert(name, pipeline);
                }
            }
        }

        Ok(Self {
            graphic_pipelines,
            rt_pipelines,
            compute_pipelines,
        })
    }

    fn get_or_compile(
        modules: &mut HashMap<String, Rc<ShaderModule>>,
        shaders: &HashMap<String, SpirVFile>,
        device: Rc<Device>,
        key: String,
        stage: ShaderStage,
    ) -> Result<Rc<ShaderModule>, VulkanError> {
        if modules.contains_key(&key) {
            Ok(modules.get(&key).unwrap().clone())
        } else {
            let module = ShaderModule::new(shaders.get(&key).unwrap().as_ref(), device, stage)?;
            module.set_name(key.to_owned())?;

            modules.insert(key.clone(), Rc::new(module));

            Ok(modules.get(&key).unwrap().clone())
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
