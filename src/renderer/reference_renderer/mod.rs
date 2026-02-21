use crate::err::AppError;
use crate::math;
use crate::renderer::buffers::global::Globals;
use crate::renderer::buffers::view_proj::ViewProj;
use crate::renderer::descriptors::{DescriptorWrite, DescriptorWriter, RendererDescriptors};
use crate::renderer::passes::{PathTracePass, SkyPass, TonemapPass};
use crate::renderer::pipeline_builder::PipelineBuilder;
use crate::renderer::push_const::PushConstBuilder;
use crate::renderer::quality::QualitySettings;
use crate::renderer::render_target::{RenderTargetBuilder, RenderTargets};
use crate::renderer::shader_loader::ShaderLoader;
use crate::renderer::{
    FrameContext, FrameStats, GPUEnv, TlasIndex, VulkanContext, create_buffer_update, env_to_buffer, open_shader_zip,
};
use crate::scene::Scene;
use crate::vulkan::{
    AccelerationStructure, Buffer, CommandBuffer, DebugMarker, DescriptorPool, Device, Fence, Sampler, Semaphore,
    ShaderBindingTable, TopLevelAs, Vertex, VulkanError, VulkanMesh,
};
use ash::vk;
use gpu_allocator::MemoryLocation;
use log::info;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;
use std::time::Instant;

const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct VulkanMcPathTracer {
    pub context: Rc<VulkanContext>,
    pub device: Rc<Device>,
    pub render_targets: RenderTargets,
    pub pipeline_builder: PipelineBuilder,
    pub shader_binding_table: ShaderBindingTable,
    pub tlases: Vec<TopLevelAs>,
    pub blases: BTreeMap<u64, AccelerationStructure>,
    pub descriptor_pool: DescriptorPool,
    pub command_buffers: Vec<CommandBuffer>,
    pub descriptors: Rc<RefCell<RendererDescriptors>>,
    pub uniform_buffers: Vec<Buffer>,
    pub uniform_buffers_globals: Vec<Buffer>,
    pub env_uniforms: Vec<Buffer>,
    pub mesh_bufs: Vec<Buffer>,
    pub current_frame: usize,
    pub quality: QualitySettings,
    pub render_scale: f32,
    meshes: BTreeMap<u64, VulkanMesh>,
    img_available: Vec<Semaphore>,
    render_finished: Vec<Semaphore>,
    in_flight: Vec<Fence>,
    prev_jitter: (f32, f32),
    frames_in_flight: usize,
    sky_pass: SkyPass,
    tonemap_pass: TonemapPass,
}

impl VulkanMcPathTracer {
    pub fn init(context: Rc<VulkanContext>) -> Result<Self, AppError> {
        let device = context.device.clone();

        let shader_loader = ShaderLoader::from_zip(open_shader_zip("shaders.zip")?)?;

        let default_sampler = Rc::new(Sampler::new(device.clone())?);
        let repeat_sampler = Rc::new(Sampler::new_repeat(device.clone())?);
        let repeat_x_only_sampler = Rc::new(Sampler::new_repeat_x_only(device.clone())?);

        let descriptor_pool = DescriptorPool::new(
            device.clone(),
            &[
                vk::DescriptorPoolSize {
                    descriptor_count: 40 * MAX_FRAMES_IN_FLIGHT as u32,
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                },
                vk::DescriptorPoolSize {
                    descriptor_count: 40 * MAX_FRAMES_IN_FLIGHT as u32,
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                },
                vk::DescriptorPoolSize {
                    descriptor_count: 40 * MAX_FRAMES_IN_FLIGHT as u32,
                    ty: vk::DescriptorType::STORAGE_IMAGE,
                },
                vk::DescriptorPoolSize {
                    descriptor_count: MAX_FRAMES_IN_FLIGHT as u32,
                    ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                },
                vk::DescriptorPoolSize {
                    descriptor_count: MAX_FRAMES_IN_FLIGHT as u32,
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                },
            ],
            200 * MAX_FRAMES_IN_FLIGHT as u32,
        )?;

        let descriptors = Rc::new(RefCell::new(RendererDescriptors::build(
            device.clone(),
            &descriptor_pool,
            MAX_FRAMES_IN_FLIGHT as u32,
        )?));

        let extent_3d = vk::Extent3D {
            width: context.swap_chain.borrow().extent.width,
            height: context.swap_chain.borrow().extent.height,
            depth: 1,
        };

        let mut render_targets = RenderTargets::new(
            device.clone(),
            extent_3d,
            default_sampler,
            repeat_sampler,
            repeat_x_only_sampler,
        );
        let mut pipeline_builder = PipelineBuilder::new(shader_loader, device.clone(), context.rt_pipeline_ext.clone());

        let sky_pass = SkyPass {
            device: device.clone(),
            render_target: render_targets.add(SkyPass::render_target_def())?,
            is_init: RefCell::new(false),
        };

        let tonemap_pass = TonemapPass {
            device: device.clone(),
            render_target: render_targets.add(TonemapPass::render_target_def())?,
        };

        render_targets.add(
            RenderTargetBuilder::new("rt_out")
                .with_storage()
                .with_transfer()
                .with_format(vk::Format::R16G16B16A16_SFLOAT),
        )?;

        // need full fat buffer for increased precision
        render_targets.add(
            RenderTargetBuilder::new("acc_out")
                .with_storage()
                .with_transfer()
                .with_format(vk::Format::R32G32B32A32_SFLOAT),
        )?;

        pipeline_builder.build_compute(
            "sky",
            "sky|main",
            &SkyPass::DESC_LAYOUTS
                .iter()
                .map(|l| descriptors.borrow().layouts.get(l).unwrap().inner)
                .collect::<Vec<_>>(),
            (size_of::<i32>() * 1) as u32,
        )?;

        pipeline_builder.build_compute(
            "accumulator",
            "accumulator|main",
            &SkyPass::DESC_LAYOUTS
                .iter()
                .map(|l| descriptors.borrow().layouts.get(l).unwrap().inner)
                .collect::<Vec<_>>(),
            (size_of::<i32>() * 4) as u32,
        )?;

        pipeline_builder.build_compute(
            "tonemap",
            "tonemap|main",
            &TonemapPass::DESC_LAYOUTS
                .iter()
                .map(|l| descriptors.borrow().layouts.get(l).unwrap().inner)
                .collect::<Vec<_>>(),
            (size_of::<i32>() * 2) as u32,
        )?;

        pipeline_builder.build_rt(
            "pt",
            "pt_reference|raygen",
            "pt_reference|miss",
            "pt_reference|chit",
            &PathTracePass::DESC_LAYOUTS
                .iter()
                .map(|l| descriptors.borrow().layouts.get(l).unwrap().inner)
                .collect::<Vec<_>>(),
            (size_of::<i32>() * 9) as u32,
        )?;

        let shader_binding_table = ShaderBindingTable::new(
            device.clone(),
            context.allocator.clone(),
            &context.rt_pipeline_ext,
            pipeline_builder.get_rt("pt").unwrap(),
            1,
            1,
        )?;

        let command_buffers = context
            .graphics_command_pool
            .allocate_cmd_buffers(MAX_FRAMES_IN_FLIGHT as u32)?;
        let as_builder_cmd_buf = context.compute_command_pool.allocate_cmd_buffers(1)?.pop().unwrap();

        let mut img_available = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut render_finished = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut in_flight = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut uniform_buffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut uniform_buffers_globals = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut env_uniforms = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut mesh_bufs = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        let mut writes = Vec::new();
        let mut tlases = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            img_available.push(Semaphore::new(device.clone())?);
            in_flight.push(Fence::new(device.clone())?);

            img_available[i].name(format!("img_available[{}]", i))?;
            in_flight[i].name(format!("in_flight[{}]", i))?;
            command_buffers[i].name(format!("cmd_buffers[{}]", i))?;

            let tlas = TopLevelAs::prepare(
                device.clone(),
                context.allocator.clone(),
                context.rt_acc_struct_ext.clone(),
                &as_builder_cmd_buf,
                &BTreeMap::new(),
                TlasIndex { index: Vec::new() },
            )?;
            tlases.push(tlas);

            let uniform_buffer = Buffer::new(
                device.clone(),
                context.allocator.clone(),
                MemoryLocation::CpuToGpu,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                ViewProj::BUF_SIZE as u64,
            )?;

            uniform_buffers.push(uniform_buffer);

            let uniform_buffer_global = Buffer::new(
                device.clone(),
                context.allocator.clone(),
                MemoryLocation::CpuToGpu,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                size_of::<Globals>() as u64 + 4,
            )?;

            uniform_buffers_globals.push(uniform_buffer_global);

            let env_buf = Buffer::new(
                device.clone(),
                context.allocator.clone(),
                MemoryLocation::CpuToGpu,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                size_of::<GPUEnv>() as u64,
            )?;

            env_uniforms.push(env_buf);

            writes.extend(Self::init_global_descriptor_set(
                &descriptors.borrow().global_sets[i].inner,
                &uniform_buffers_globals[i].inner,
                &tlases[i],
                &uniform_buffers[i].inner,
                &env_uniforms[i].inner,
            ));
        }

        for i in 0..context.swap_chain.borrow().images.len() {
            render_finished.push(Semaphore::new(device.clone())?);
            render_finished[i].name(format!("render_finished[{}]", i))?;
        }

        let mut desc_borrow = descriptors.borrow_mut();

        writes.extend(desc_borrow.update_resources(&render_targets)?);

        DescriptorWriter::batch_write(&device, writes);

        drop(desc_borrow);

        info!("Storage images: {:?}", descriptors.borrow().storages.len());
        info!("Sampled images: {:?}", descriptors.borrow().samplers.len());

        Ok(Self {
            context: context.clone(),
            device: device.clone(),
            render_targets,
            pipeline_builder,
            shader_binding_table,
            tlases,
            blases: BTreeMap::new(),
            descriptor_pool,
            command_buffers,
            img_available,
            render_finished,
            descriptors,
            uniform_buffers,
            uniform_buffers_globals,
            env_uniforms,
            mesh_bufs,
            in_flight,
            meshes: BTreeMap::new(),
            quality: QualitySettings::new(),
            current_frame: 0,
            prev_jitter: (0.5, 0.5),
            frames_in_flight: MAX_FRAMES_IN_FLIGHT,
            sky_pass,
            tonemap_pass,
            render_scale: 1.0,
        })
    }

    fn create_tlas_update_descriptor_set<'a>(desc_set: &vk::DescriptorSet, tlas: &TopLevelAs) -> DescriptorWrite<'a> {
        let tlas_write = vk::WriteDescriptorSet {
            dst_set: *desc_set,
            dst_binding: 1,
            dst_array_element: 0,
            descriptor_type: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            descriptor_count: 1,
            ..Default::default()
        };

        DescriptorWrite {
            write: tlas_write,
            buffer_info: None,
            tlases: Some(vec![tlas.inner]),
            image_info: None,
        }
    }

    fn init_global_descriptor_set<'a>(
        desc_set: &vk::DescriptorSet,
        globals: &vk::Buffer,
        tlas: &TopLevelAs,
        view: &vk::Buffer,
        env: &vk::Buffer,
    ) -> Vec<DescriptorWrite<'a>> {
        vec![
            create_buffer_update(
                globals,
                size_of::<Globals>() as u64,
                desc_set,
                0,
                vk::DescriptorType::UNIFORM_BUFFER,
            ),
            Self::create_tlas_update_descriptor_set(desc_set, tlas),
            create_buffer_update(
                view,
                ViewProj::BUF_SIZE as u64,
                desc_set,
                2,
                vk::DescriptorType::UNIFORM_BUFFER,
            ),
            create_buffer_update(
                env,
                size_of::<GPUEnv>() as u64,
                desc_set,
                3,
                vk::DescriptorType::UNIFORM_BUFFER,
            ),
        ]
    }

    pub fn render_frame(
        &mut self,
        scene: &Scene,
        drawable_size: (u32, u32),
        context: &FrameContext,
        ui: Option<&imgui::DrawData>,
    ) -> Result<FrameStats, AppError> {
        self.in_flight[self.current_frame].wait()?;
        self.in_flight[self.current_frame].reset()?;

        let (image_index, _is_suboptimal) = self
            .context
            .swap_chain
            .borrow()
            .acquire_next_image(&self.img_available[self.current_frame])?;

        self.command_buffers[self.current_frame].reset()?;

        let start = Instant::now();

        self.prepare_meshes(scene)?;

        let tlas_start = Instant::now();

        let tlas_index = self.build_tlas_index(scene);

        self.tlases[self.current_frame] = TopLevelAs::prepare(
            self.device.clone(),
            self.context.allocator.clone(),
            self.context.rt_acc_struct_ext.clone(),
            &self.context.compute_command_pool.allocate_cmd_buffers(1)?[0],
            &self.blases,
            tlas_index,
        )?;

        let tlas_time = tlas_start.elapsed().as_secs_f32();

        let command_buffer = &self.command_buffers[self.current_frame];

        let width = (drawable_size.0 as f32 * self.render_scale).max(1.0) as u32;
        let height = (drawable_size.1 as f32 * self.render_scale).max(1.0) as u32;

        let aspect_ratio = width as f32 / height as f32;
        let fov_rad = math::deg_to_rad(scene.camera.fov);
        let fovy = math::fovx_to_fovy(fov_rad, aspect_ratio);

        let mut offset = math::halton(context.frame_index % 16 + 1);
        if context.clear_taa {
            offset = (0.5, 0.5);
        }
        let (offset_x, offset_y) = (
            (2.0 * offset.0 - 1.0) / width as f32,
            (2.0 * offset.1 - 1.0) / height as f32,
        );

        let mut proj = nalgebra_glm::perspective_rh_zo(aspect_ratio, fovy, 500.0, 0.01);

        // add jitter
        proj.m13 = offset_x;
        proj.m23 = offset_y;

        // flip because of vulkan
        proj.m22 *= -1.0;

        let view_proj = ViewProj {
            view: scene.camera.view(),
            projection: proj,
            view_prev: scene.camera.view(),
            projection_prev: proj,
            view_inverse: scene.camera.view().try_inverse().unwrap(),
            projection_inverse: proj.try_inverse().unwrap(),
        };

        let mesh_start = Instant::now();

        let mesh_time = mesh_start.elapsed().as_secs_f32();

        let globals = Globals {
            exposure: scene.env.exposure,
            debug_mode: 0,
            res_x: drawable_size.0 as f32,
            res_y: drawable_size.1 as f32,
            draw_res_x: width as f32,
            draw_res_y: height as f32,
            time: context.total_time,
            frame_index: context.frame_index,
            current_jitter: offset,
            prev_jitter: self.prev_jitter,
        };

        let env = env_to_buffer(&scene.env);

        self.uniform_buffers[self.current_frame].fill_host(view_proj.to_bytes().as_ref())?;
        self.uniform_buffers_globals[self.current_frame].fill_host(globals.to_bytes().as_ref())?;
        self.env_uniforms[self.current_frame].fill_host(env.as_ref())?;

        let mut writes = Vec::new();

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            writes.push(Self::create_tlas_update_descriptor_set(
                &self.descriptors.borrow().global_sets[i].inner,
                &self.tlases[i],
            ));
        }

        DescriptorWriter::batch_write(&self.device, writes);

        command_buffer.begin()?;

        let record_start = Instant::now();

        self.record_command_buffer(
            command_buffer,
            &self.context.swap_chain.borrow().images[image_index as usize],
            context,
            (width, height),
            fov_rad,
        )?;

        let attachments = [vk::RenderingAttachmentInfo {
            image_view: self.context.swap_chain_image_views.borrow()[image_index as usize].inner,
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            load_op: vk::AttachmentLoadOp::DONT_CARE,
            store_op: vk::AttachmentStoreOp::STORE,
            ..Default::default()
        }];

        let rendering_info = vk::RenderingInfo {
            render_area: vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: self.context.swap_chain.borrow().extent,
            },
            layer_count: 1,
            color_attachment_count: attachments.len() as u32,
            p_color_attachments: attachments.as_ptr(),
            p_depth_attachment: std::ptr::null(),
            ..Default::default()
        };

        command_buffer.begin_rendering(&rendering_info);

        if let Some(dd) = ui {
            self.context
                .imgui_renderer
                .borrow_mut()
                .cmd_draw(command_buffer.inner, dd)
                .map_err(|e| AppError::Other(format!("Failed to draw imgui: {e}")))?;
        }

        command_buffer.end_rendering();

        unsafe {
            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::ALL_GRAPHICS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier {
                    src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    dst_access_mask: vk::AccessFlags::TRANSFER_READ,
                    old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                    image: self.context.swap_chain.borrow().images[image_index as usize],
                    subresource_range: vk::ImageSubresourceRange {
                        base_mip_level: 0,
                        level_count: 1,
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                }],
            );
        }

        command_buffer.end()?;

        let record_time = record_start.elapsed().as_secs_f32();

        let wait_semaphores = [self.img_available[self.current_frame].inner];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.render_finished[image_index as usize].inner];

        let submit_info = vk::SubmitInfo {
            wait_semaphore_count: wait_semaphores.len() as u32,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: 1,
            p_command_buffers: &command_buffer.inner,
            signal_semaphore_count: 1,
            p_signal_semaphores: signal_semaphores.as_ptr(),
            ..Default::default()
        };

        let end = Instant::now();

        unsafe {
            self.device
                .inner
                .queue_submit(
                    self.device.graphics_queue,
                    &[submit_info],
                    self.in_flight[self.current_frame].inner,
                )
                .expect("failed to submit to queue")
        };

        let is_suboptimal = {
            let present_info = vk::PresentInfoKHR {
                wait_semaphore_count: 1,
                p_wait_semaphores: signal_semaphores.as_ptr(),
                swapchain_count: 1,
                p_swapchains: &self.context.swap_chain.borrow().swapchain,
                p_image_indices: &image_index,
                ..Default::default()
            };

            unsafe {
                self.context
                    .swap_chain
                    .borrow()
                    .loader
                    .queue_present(self.device.present_queue, &present_info)
                    .expect("cannot present")
            }
        };

        if is_suboptimal {
            self.resize(drawable_size)?;
        }

        self.current_frame = (self.current_frame + 1) % self.frames_in_flight;
        self.prev_jitter = offset;

        let stats = FrameStats {
            cpu_time: (end - start).as_secs_f32(),
            tlas_time,
            record_time,
            mesh_time,
            objects_rendered: 0,
            draw_calls: 0,
        };

        Ok(stats)
    }

    pub fn resize(&mut self, drawable_size: (u32, u32)) -> Result<(), AppError> {
        self.device.wait_idle()?;

        self.context.recreate_swapchain(drawable_size)?;

        let extent_3d = vk::Extent3D {
            width: drawable_size.0,
            height: drawable_size.1,
            depth: 1,
        };

        self.render_targets.set_extent(extent_3d);
        self.render_targets.resize()?;

        let mut desc_borrow = self.descriptors.borrow_mut();
        let writes = desc_borrow.update_resources(&self.render_targets)?;

        DescriptorWriter::batch_write(&self.device, writes);

        drop(desc_borrow);

        Ok(())
    }

    fn record_command_buffer(
        &self,
        command_buffer: &CommandBuffer,
        target: &vk::Image,
        context: &FrameContext,
        viewport_size: (u32, u32),
        fov: f32,
    ) -> Result<(), VulkanError> {
        self.tlases[self.current_frame].build(command_buffer)?;
        self.sky_pass.record_reference(command_buffer, self)?;

        unsafe {
            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier {
                    src_access_mask: vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
                    dst_access_mask: vk::AccessFlags::SHADER_READ,
                    ..Default::default()
                }],
                &[],
                &[],
            );
        }

        self.record_pt(command_buffer, context, viewport_size, fov)?;

        self.record_accumulate(command_buffer, context, viewport_size)?;

        self.tonemap_pass
            .record_reference(command_buffer, self, viewport_size)?;

        self.record_end_copy(command_buffer, target, viewport_size)?;

        Ok(())
    }

    pub fn record_pt(
        &self,
        command_buffer: &CommandBuffer,
        context: &FrameContext,
        viewport: (u32, u32),
        fov: f32,
    ) -> Result<(), VulkanError> {
        self.device.begin_label("Path Tracing", command_buffer);

        let pipeline = self.pipeline_builder.get_rt("pt").unwrap();

        command_buffer.bind_rt_pipeline(pipeline);

        unsafe {
            let barriers = [
                self.render_targets.get("rt_out").unwrap().borrow().image.inner,
                self.render_targets.get("acc_out").unwrap().borrow().image.inner,
                self.render_targets.get("tonemap_out").unwrap().borrow().image.inner,
            ]
            .map(|image| vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::NONE,
                dst_access_mask: vk::AccessFlags::SHADER_WRITE,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::GENERAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            });

            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );

            self.device.inner.cmd_bind_descriptor_sets(
                command_buffer.inner,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                pipeline.layout,
                0,
                &[
                    self.descriptors.borrow().global_sets[self.current_frame].inner,
                    self.descriptors.borrow().compute_sets[self.current_frame].inner,
                ],
                &[],
            );

            let pc = PushConstBuilder::new()
                .add_u32(context.frame_index)
                .add_u32(self.quality.pt_bounces as u32)
                .add_u32(*self.descriptors.borrow().storages.get("rt_out").unwrap() as u32)
                .add_u32(*self.descriptors.borrow().samplers.get("sky").unwrap() as u32)
                .add_f32(self.quality.rt_direct_trace_distance)
                .add_f32(self.quality.rt_indirect_trace_distance)
                .add_f32(fov)
                .build();

            self.device.inner.cmd_push_constants(
                command_buffer.inner,
                pipeline.layout,
                vk::ShaderStageFlags::RAYGEN_KHR,
                0,
                &pc,
            );

            self.context.rt_pipeline_ext.loader.cmd_trace_rays(
                command_buffer.inner,
                &self.shader_binding_table.raygen_region,
                &self.shader_binding_table.miss_region,
                &self.shader_binding_table.hit_region,
                &self.shader_binding_table.call_region,
                viewport.0,
                viewport.1,
                1,
            );

            let barriers =
                [self.render_targets.get("rt_out").unwrap().borrow().image.inner].map(|image| vk::ImageMemoryBarrier {
                    src_access_mask: vk::AccessFlags::SHADER_WRITE,
                    dst_access_mask: vk::AccessFlags::SHADER_READ,
                    old_layout: vk::ImageLayout::GENERAL,
                    new_layout: vk::ImageLayout::GENERAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                });

            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );
        }

        self.device.end_label(command_buffer);

        Ok(())
    }

    fn record_accumulate(
        &self,
        command_buffer: &CommandBuffer,
        context: &FrameContext,
        viewport_size: (u32, u32),
    ) -> Result<(), VulkanError> {
        let acc_image = self.render_targets.get("acc_out").unwrap().borrow().image.inner;

        let pipeline = self.pipeline_builder.get_compute("accumulator").unwrap();

        command_buffer.bind_compute_pipeline(pipeline);
        command_buffer.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            [
                self.descriptors.borrow().global_sets[self.current_frame].inner,
                self.descriptors.borrow().compute_sets[self.current_frame].inner,
            ],
        );

        unsafe {
            let pc = PushConstBuilder::with_capacity(4 * size_of::<u32>())
                .add_u32(context.frame_index)
                .add_u32(if context.clear_taa { 1 } else { 0 })
                .add_u32(*self.descriptors.borrow().storages.get("rt_out").unwrap() as u32)
                .add_u32(*self.descriptors.borrow().storages.get("acc_out").unwrap() as u32)
                .build();

            self.device.inner.cmd_push_constants(
                command_buffer.inner,
                pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &pc,
            );
        }

        let x = viewport_size.0 / 16 + 1;
        let y = viewport_size.1 / 16 + 1;

        command_buffer.dispatch(x, y, 1);

        let barrier = vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::SHADER_WRITE,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            old_layout: vk::ImageLayout::GENERAL,
            new_layout: vk::ImageLayout::GENERAL,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: acc_image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..Default::default()
        };

        unsafe {
            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }

        Ok(())
    }

    fn record_end_copy(
        &self,
        command_buffer: &CommandBuffer,
        target: &vk::Image,
        viewport_size: (u32, u32),
    ) -> Result<(), VulkanError> {
        let src_image = self.render_targets.get("tonemap_out").unwrap().borrow().image.inner;

        unsafe {
            let offset_min = vk::Offset3D { x: 0, y: 0, z: 0 };
            let offset_src_max = vk::Offset3D {
                x: viewport_size.0 as i32,
                y: viewport_size.1 as i32,
                z: 1,
            };
            let offset_dst_max = vk::Offset3D {
                x: self.context.swap_chain.borrow().extent.width as i32,
                y: self.context.swap_chain.borrow().extent.height as i32,
                z: 1,
            };

            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier {
                    src_access_mask: vk::AccessFlags::NONE,
                    dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    image: *target,
                    subresource_range: vk::ImageSubresourceRange {
                        base_mip_level: 0,
                        level_count: 1,
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                }],
            );

            self.device.inner.cmd_blit_image(
                command_buffer.inner,
                src_image,
                vk::ImageLayout::GENERAL,
                *target,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageBlit {
                    src_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    src_offsets: [offset_min, offset_src_max],
                    dst_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    dst_offsets: [offset_min, offset_dst_max],
                }],
                vk::Filter::LINEAR,
            );

            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_GRAPHICS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier {
                    src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    image: *target,
                    subresource_range: vk::ImageSubresourceRange {
                        base_mip_level: 0,
                        level_count: 1,
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                }],
            );
        }

        Ok(())
    }

    fn prepare_meshes(&mut self, scene: &Scene) -> Result<(), AppError> {
        let mut changed = false;

        for instance in &scene.meshes {
            if let std::collections::btree_map::Entry::Vacant(e) = self.meshes.entry(instance.resource.id) {
                let mesh = VulkanMesh::new(
                    self.device.clone(),
                    self.context.allocator.clone(),
                    &self.context.graphics_command_pool,
                    &instance.resource,
                )?;
                changed = true;
                e.insert(mesh);
            }
        }

        if changed {
            let mut geos = BTreeMap::new();
            let mut ranges = BTreeMap::new();

            let mut processed = BTreeSet::new();

            for mesh in &scene.meshes {
                if processed.contains(&mesh.resource.id) {
                    continue;
                }

                let res = self.meshes.get(&mesh.resource.id).unwrap();

                let addr = res.buf.inner.get_device_addr();
                let max_prim_count = res.index_count / 3;

                let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR {
                    vertex_format: vk::Format::R32G32B32_SFLOAT,
                    vertex_data: vk::DeviceOrHostAddressConstKHR { device_address: addr },
                    vertex_stride: std::mem::size_of::<Vertex>() as vk::DeviceSize,
                    max_vertex: mesh.resource.vertices.len() as u32 - 1,
                    index_type: vk::IndexType::UINT32,
                    index_data: vk::DeviceOrHostAddressConstKHR {
                        device_address: addr + (res.indices_offset),
                    },
                    ..Default::default()
                };

                let geo = vk::AccelerationStructureGeometryKHR {
                    geometry_type: vk::GeometryTypeKHR::TRIANGLES,
                    geometry: vk::AccelerationStructureGeometryDataKHR { triangles },
                    flags: vk::GeometryFlagsKHR::OPAQUE,
                    ..Default::default()
                };
                geos.insert(mesh.resource.id, geo);

                let range = vk::AccelerationStructureBuildRangeInfoKHR {
                    primitive_count: max_prim_count as u32,
                    primitive_offset: 0,
                    first_vertex: 0,
                    transform_offset: 0,
                };
                ranges.insert(mesh.resource.id, range);
                processed.insert(mesh.resource.id);
            }

            let batch = AccelerationStructure::batch_bottom_build(
                self.device.clone(),
                self.context.allocator.clone(),
                self.context.rt_acc_struct_ext.clone(),
                &self.context.compute_command_pool,
                ranges,
                geos,
                vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                    | vk::BuildAccelerationStructureFlagsKHR::ALLOW_DATA_ACCESS,
            )?;

            self.blases = batch;
        }

        Ok(())
    }

    fn build_tlas_index(&self, scene: &Scene) -> TlasIndex {
        let mut index = Vec::new();

        for mesh in &scene.meshes {
            let transform = mesh.transform;
            let id = mesh.resource.id;

            index.push((id, transform));
        }

        TlasIndex { index }
    }
}
