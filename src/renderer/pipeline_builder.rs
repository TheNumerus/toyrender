use crate::err::AppError;
use crate::renderer::descriptors::RendererDescriptors;
use crate::renderer::shader_loader::ShaderLoader;
use crate::vulkan::{
    Compute, DebugMarker, Device, Graphics, Pipeline, RayTracingPipeline, Rt, ShaderModule, ShaderStage,
};
use ash::vk;
use rspirv_reflect::{DescriptorInfo, Reflection};
use std::cell::Ref;
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

pub static PIPELINE_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(PartialEq, Hash, Debug, Clone, Copy, Eq)]
pub struct PipelineHandle(u64);

pub struct PipelineBuilder {
    shader_loader: ShaderLoader,
    device: Rc<Device>,
    rt_pipeline_ext: Rc<RayTracingPipeline>,
    graphic_pipelines: HashMap<String, Pipeline<Graphics>>,
    rt_pipelines: HashMap<PipelineHandle, Pipeline<Rt>>,
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
        descriptors: Ref<RendererDescriptors>,
        attachment_formats: &[vk::Format],
        use_depth: bool,
    ) -> Result<(), AppError> {
        let vert_name = vert_name.as_ref();
        let frag_name = frag_name.as_ref();

        // TODO both stages can have different usage of descriptor sets
        // Right now the fragment stage does not use anything, so reflection returns nothing

        let (vertex_module, refl) = self.get_shader(vert_name, ShaderStage::Vertex)?;
        vertex_module.set_name(vert_name.to_owned())?;

        let (fragment_module, _) = self.get_shader(frag_name, ShaderStage::Fragment)?;
        fragment_module.set_name(frag_name.to_owned())?;

        let stages = [vertex_module.stage_info(), fragment_module.stage_info()];

        let push_consts_size = refl
            .get_push_constant_range()
            .map_err(|e| AppError::Import(e.to_string()))?
            .unwrap_or(rspirv_reflect::PushConstantInfo { offset: 0, size: 0 })
            .size;

        let desc_sets_info = refl
            .get_descriptor_sets()
            .map_err(|e| AppError::Import(format!("Could not get descriptor sets: {}", e)))?;

        if desc_sets_info.is_empty() {
            return Err(AppError::Import(format!(
                "No descriptor sets found in reflection for '{}'",
                name.as_ref()
            )));
        }

        let desc_layouts = Self::get_desc_layouts(descriptors, desc_sets_info)
            .map_err(|e| AppError::Import(format!("Error getting descriptor sets for '{}': {}", name.as_ref(), e)))?;

        let pipeline = Pipeline::new_graphics(
            self.device.clone(),
            &stages,
            &desc_layouts,
            attachment_formats,
            use_depth,
            push_consts_size,
        )?;
        pipeline.name(name.as_ref())?;

        self.graphic_pipelines.insert(name.as_ref().to_owned(), pipeline);

        Ok(())
    }

    pub fn build_compute(
        &mut self,
        name: impl AsRef<str>,
        compute_name: impl AsRef<str>,
        descriptors: Ref<RendererDescriptors>,
    ) -> Result<(), AppError> {
        let compute_name = compute_name.as_ref();
        let (module, refl) = self.get_shader(compute_name, ShaderStage::Compute)?;
        module.set_name(compute_name.to_owned())?;

        let push_consts_size = refl
            .get_push_constant_range()
            .map_err(|e| AppError::Import(e.to_string()))?
            .unwrap_or(rspirv_reflect::PushConstantInfo { offset: 0, size: 0 })
            .size;

        let desc_sets_info = refl
            .get_descriptor_sets()
            .map_err(|e| AppError::Import(format!("Could not get descriptor sets: {}", e)))?;

        if desc_sets_info.is_empty() {
            return Err(AppError::Import(format!(
                "No descriptor sets found in reflection for '{}'",
                name.as_ref()
            )));
        }

        let desc_layouts = Self::get_desc_layouts(descriptors, desc_sets_info)
            .map_err(|e| AppError::Import(format!("Error getting descriptor sets for '{}': {}", name.as_ref(), e)))?;

        let pipeline = Pipeline::new_compute(
            self.device.clone(),
            module.stage_info(),
            &desc_layouts,
            push_consts_size,
        )?;
        pipeline.name(name.as_ref())?;

        self.compute_pipelines.insert(name.as_ref().to_owned(), pipeline);

        Ok(())
    }

    pub fn build_rt(
        &mut self,
        name: impl AsRef<str>,
        name_raygen: impl AsRef<str>,
        name_miss: impl AsRef<str>,
        name_hit: impl AsRef<str>,
        descriptors: Ref<RendererDescriptors>,
    ) -> Result<PipelineHandle, AppError> {
        let raygen_name = name_raygen.as_ref();
        let (raygen_module, refl) = self.get_shader(raygen_name, ShaderStage::RayGen)?;
        raygen_module.set_name(raygen_name.to_owned())?;

        let miss_name = name_miss.as_ref();
        let (miss_module, _) = self.get_shader(miss_name, ShaderStage::RayMiss)?;
        miss_module.set_name(miss_name.to_owned())?;

        let hit_name = name_hit.as_ref();
        let (hit_module, _) = self.get_shader(hit_name, ShaderStage::RayClosestHit)?;
        hit_module.set_name(hit_name.to_owned())?;

        let rt_stages = [
            raygen_module.stage_info(),
            miss_module.stage_info(),
            hit_module.stage_info(),
        ];

        let push_consts_size = refl
            .get_push_constant_range()
            .map_err(|e| AppError::Import(format!("Could not get push const range: {}", e)))?
            .unwrap_or(rspirv_reflect::PushConstantInfo { offset: 0, size: 0 })
            .size;

        let desc_sets_info = refl
            .get_descriptor_sets()
            .map_err(|e| AppError::Import(format!("Could not get descriptor sets: {}", e)))?;

        if desc_sets_info.is_empty() {
            return Err(AppError::Import(format!(
                "No descriptor sets found in reflection for '{}'",
                name.as_ref()
            )));
        }

        let desc_layouts = Self::get_desc_layouts(descriptors, desc_sets_info)
            .map_err(|e| AppError::Import(format!("Error getting descriptor sets for '{}': {}", name.as_ref(), e)))?;

        let pipeline = Pipeline::new_rt(
            self.device.clone(),
            &self.rt_pipeline_ext,
            &rt_stages,
            &desc_layouts,
            push_consts_size,
        )?;
        pipeline.name(name.as_ref())?;

        let id = PIPELINE_ID_COUNTER.fetch_add(1, Ordering::SeqCst);

        self.rt_pipelines.insert(PipelineHandle(id), pipeline);
        Ok(PipelineHandle(id))
    }

    pub fn get_graphics(&self, key: &str) -> Option<&Pipeline<Graphics>> {
        self.graphic_pipelines.get(key)
    }

    pub fn get_compute(&self, key: &str) -> Option<&Pipeline<Compute>> {
        self.compute_pipelines.get(key)
    }

    pub fn get_rt(&self, key: &PipelineHandle) -> Result<&Pipeline<Rt>, AppError> {
        self.rt_pipelines
            .get(key)
            .ok_or_else(|| AppError::InvalidKey(format!("RT pipeline not found: {:?}", key)))
    }

    fn get_shader(&self, shader_name: &str, shader_stage: ShaderStage) -> Result<(ShaderModule, Reflection), AppError> {
        let stub = self
            .shader_loader
            .shaders
            .get(shader_name)
            .ok_or_else(|| AppError::Other(format!("Missing shader stub: {}", shader_name)))?
            .as_ref();

        let module = ShaderModule::new(stub, self.device.clone(), shader_stage)?;

        let reflect_data = Reflection::new_from_spirv(stub).map_err(|e| AppError::Other(e.to_string()))?;

        Ok((module, reflect_data))
    }

    fn get_desc_layouts(
        descriptors: Ref<RendererDescriptors>,
        desc_info: BTreeMap<u32, BTreeMap<u32, DescriptorInfo>>,
    ) -> Result<Vec<vk::DescriptorSetLayout>, AppError> {
        let mut sets = Vec::with_capacity(desc_info.len());

        // hope that no index is skipped in shader definition
        for set in desc_info.values() {
            sets.push(descriptors.guess_layout_from_reflection(set)?);
        }

        Ok(sets)
    }
}
