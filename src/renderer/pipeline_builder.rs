use crate::err::AppError;
use crate::renderer::shader_loader::ShaderLoader;
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
        vert_name: impl AsRef<str>,
        frag_name: impl AsRef<str>,
        desc_layouts: &[vk::DescriptorSetLayout],
        attachment_formats: &[vk::Format],
        push_consts_size: u32,
        use_depth: bool,
    ) -> Result<(), AppError> {
        let vert_name = vert_name.as_ref();
        let frag_name = frag_name.as_ref();

        let vertex_module = self.get_shader(vert_name, ShaderStage::Vertex)?;
        vertex_module.set_name(vert_name.to_owned())?;

        let fragment_module = self.get_shader(frag_name, ShaderStage::Fragment)?;
        fragment_module.set_name(frag_name.to_owned())?;

        let stages = [vertex_module.stage_info(), fragment_module.stage_info()];

        let pipeline = Pipeline::new_graphics(
            self.device.clone(),
            &stages,
            desc_layouts,
            attachment_formats,
            use_depth,
            push_consts_size,
        )?;
        pipeline.set_name(name.as_ref().to_owned())?;

        self.graphic_pipelines.insert(name.as_ref().to_owned(), pipeline);

        Ok(())
    }

    pub fn build_compute(
        &mut self,
        name: impl AsRef<str>,
        compute_name: impl AsRef<str>,
        desc_layouts: &[vk::DescriptorSetLayout],
        push_consts_size: u32,
    ) -> Result<(), AppError> {
        let compute_name = compute_name.as_ref();
        let module = self.get_shader(compute_name, ShaderStage::Compute)?;
        module.set_name(compute_name.to_owned())?;

        let pipeline = Pipeline::new_compute(self.device.clone(), module.stage_info(), desc_layouts, push_consts_size)?;
        pipeline.set_name(name.as_ref().to_owned())?;

        self.compute_pipelines.insert(name.as_ref().to_owned(), pipeline);

        Ok(())
    }

    pub fn build_rt(
        &mut self,
        name: impl AsRef<str>,
        name_raygen: impl AsRef<str>,
        name_miss: impl AsRef<str>,
        name_hit: impl AsRef<str>,
        desc_layouts: &[vk::DescriptorSetLayout],
        push_consts_size: u32,
    ) -> Result<(), AppError> {
        let raygen_name = name_raygen.as_ref();
        let raygen_module = self.get_shader(raygen_name, ShaderStage::RayGen)?;
        raygen_module.set_name(raygen_name.to_owned())?;

        let miss_name = name_miss.as_ref();
        let miss_module = self.get_shader(miss_name, ShaderStage::RayMiss)?;
        miss_module.set_name(miss_name.to_owned())?;

        let hit_name = name_hit.as_ref();
        let hit_module = self.get_shader(hit_name, ShaderStage::RayClosestHit)?;
        hit_module.set_name(hit_name.to_owned())?;

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

    fn get_shader(&self, shader_name: &str, shader_stage: ShaderStage) -> Result<ShaderModule, AppError> {
        let module = ShaderModule::new(
            self.shader_loader
                .shaders
                .get(shader_name)
                .ok_or_else(|| AppError::Other(format!("Missing shader stub: {}", shader_name)))?
                .as_ref(),
            self.device.clone(),
            shader_stage,
        )?;
        Ok(module)
    }
}
