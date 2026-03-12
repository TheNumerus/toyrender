use crate::err::AppError;
use crate::math;
use crate::scene::{Scene, SkyVariant};
use crate::vulkan::{
    Buffer, CommandBuffer, CommandPool, DebugMarker, DescriptorPool, Device, Fence, IntoVulkanError, PresentInfo,
    Sampler, Semaphore, ShaderBindingTable, SubmitInfo, TopLevelAs, VulkanError,
};
use ash::vk;
use gpu_allocator::MemoryLocation;
use log::info;
use nalgebra_glm::Mat4;
use std::cell::{Ref, RefCell};
use std::collections::BTreeMap;
use std::rc::Rc;
use std::time::Instant;

mod buffers;
use buffers::{GPUEnv, Globals, ViewProj};

mod context;
pub use context::VulkanContext;

mod descriptors;
use descriptors::{DescriptorWrite, DescriptorWriter, RendererDescriptors};

mod mesh_collector;
use mesh_collector::{DrawData, MeshCollector};

mod resource_subsystem;
pub use resource_subsystem::ResourceSubsystem;

mod passes;
use passes::{DenoisePass, DepthDebugPass, GBufferPass, PathTracePass, ShadingPass, SkyPass, TaaPass, TonemapPass};

mod pipeline_builder;
use pipeline_builder::PipelineBuilder;

mod quality;
use quality::QualitySettings;

mod render_target;
use render_target::{RenderTargetBuilder, RenderTargets};

mod debug;
use debug::DebugMode;

mod push_const;
pub use push_const::PushConstBuilder;

mod reference_renderer;
use crate::renderer::descriptors::DescriptorLayouts;
use crate::renderer::passes::{DenoiseInputs, PathTraceInputs, ShadingInputs, TaaInputs};
use crate::renderer::render_target::RenderTarget;
pub use reference_renderer::VulkanMcPathTracer;

pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

struct VulkanRendererPasses {
    gbuffer: GBufferPass,
    sky: SkyPass,
    pt: PathTracePass,
    denoise: DenoisePass,
    shading: ShadingPass,
    taa: TaaPass,
    tonemap: TonemapPass,
    depth_debug: DepthDebugPass,
}

pub struct VulkanRenderer {
    pub context: Rc<VulkanContext>,
    pub device: Rc<Device>,
    pub render_targets: RenderTargets,
    pub last_depth: Rc<RefCell<RenderTarget>>,
    pub shader_binding_table: ShaderBindingTable,
    pub tlases: Vec<TopLevelAs>,
    pub _descriptor_pool: DescriptorPool,
    pub raster_command_buffers: Vec<CommandBuffer>,
    pub command_buffers: Vec<CommandBuffer>,
    pub compute_command_buffers: Vec<CommandBuffer>,
    tlas_prepare_cmd_buf: CommandBuffer,
    pub descriptor_layouts: DescriptorLayouts,
    pub descriptors: Vec<Rc<RefCell<RendererDescriptors>>>,
    pub uniform_buffers: Vec<Buffer>,
    pub uniform_buffers_globals: Vec<Buffer>,
    pub env_uniforms: Vec<Buffer>,
    pub mesh_bufs: Vec<Buffer>,
    pub current_frame: usize,
    pub quality: QualitySettings,
    pub debug_mode: DebugMode,
    pub render_scale: f32,
    img_available: Vec<Semaphore>,
    render_finished: Vec<Semaphore>,
    gbuf_done: Semaphore,
    sky_done: Semaphore,
    in_flight: Vec<Fence>,
    last_view_proj: ViewProj,
    prev_jitter: (f32, f32),
    frames_in_flight: usize,
    passes: VulkanRendererPasses,
}

impl VulkanRenderer {
    pub fn init(context: Rc<VulkanContext>) -> Result<Self, AppError> {
        let device = context.device.clone();

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

        let descriptor_layouts = DescriptorLayouts::create(device.clone())?;

        let extent_3d = vk::Extent3D {
            width: context.swap_chain.borrow().extent.width,
            height: context.swap_chain.borrow().extent.height,
            depth: 1,
        };

        let mut render_targets = RenderTargets::new(
            device.clone(),
            context.allocator.clone(),
            extent_3d,
            default_sampler,
            repeat_sampler,
            repeat_x_only_sampler,
        );

        let passes = Self::create_passes(
            device.clone(),
            context.clone(),
            &mut render_targets,
            &mut context.pipeline_builder.borrow_mut(),
            &descriptor_layouts,
        )?;

        let last_depth = render_targets.add(RenderTargetBuilder::new_depth("last_depth").with_transfer())?;

        let shader_binding_table = ShaderBindingTable::new(
            device.clone(),
            context.allocator.clone(),
            &context.rt_pipeline_ext,
            &passes.pt.pipeline_handle,
            1,
            1,
        )?;

        let raster_command_buffers = context
            .graphics_command_pool
            .allocate_cmd_buffers(MAX_FRAMES_IN_FLIGHT as u32)?;
        let command_buffers = context
            .graphics_command_pool
            .allocate_cmd_buffers(MAX_FRAMES_IN_FLIGHT as u32)?;
        let compute_command_buffers = context
            .compute_command_pool
            .allocate_cmd_buffers(MAX_FRAMES_IN_FLIGHT as u32)?;
        let tlas_prepare_cmd_buf = context.compute_command_pool.allocate_cmd_buffers(1)?.pop().unwrap();

        let mut img_available = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut render_finished = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut in_flight = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut uniform_buffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut uniform_buffers_globals = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut env_uniforms = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut mesh_bufs = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut descriptors = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        Self::init_history_images(
            &context.device,
            &context.graphics_command_pool,
            &[
                passes.taa.render_target_history.borrow().image.inner,
                passes.denoise.direct_render_target_acc.borrow().image.inner,
                passes.denoise.direct_render_target_history.borrow().image.inner,
                passes.denoise.indirect_render_target_acc.borrow().image.inner,
                passes.denoise.indirect_render_target_history.borrow().image.inner,
                passes.denoise.moments_direct_render_target.borrow().image.inner,
                passes.denoise.moments_indirect_render_target.borrow().image.inner,
            ],
            &[render_targets.get_ref("last_depth").unwrap().image.inner],
        )?;

        let mut writes = Vec::new();
        let mut tlases = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            img_available.push(Semaphore::new(device.clone())?);
            in_flight.push(Fence::new(device.clone())?);

            img_available[i].name(format!("img_available[{}]", i))?;
            in_flight[i].name(format!("in_flight[{}]", i))?;
            command_buffers[i].name(format!("cmd_buffers[{}]", i))?;
            raster_command_buffers[i].name(format!("raster_command_buffers[{}]", i))?;
            compute_command_buffers[i].name(format!("compute_command_buffers[{}]", i))?;

            descriptors.push(Rc::new(RefCell::new(RendererDescriptors::build(
                &descriptor_pool,
                &descriptor_layouts,
            )?)));

            let tlas = TopLevelAs::prepare(
                device.clone(),
                context.allocator.clone(),
                context.rt_acc_struct_ext.clone(),
                &tlas_prepare_cmd_buf,
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

            let mesh_transform_buffer = Buffer::new(
                device.clone(),
                context.allocator.clone(),
                MemoryLocation::CpuToGpu,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                size_of::<Mat4>() as u64 * 8192,
            )?;

            mesh_bufs.push(mesh_transform_buffer);

            writes.extend(Self::init_global_descriptor_set(
                &descriptors[i].borrow().global_set.inner,
                &uniform_buffers_globals[i].inner,
                &tlases[i],
                &uniform_buffers[i].inner,
                &env_uniforms[i].inner,
                &mesh_bufs[i].inner,
            ));
        }

        for i in 0..context.swap_chain.borrow().images.len() {
            render_finished.push(Semaphore::new(device.clone())?);
            render_finished[i].name(format!("render_finished[{}]", i))?;
        }

        DescriptorWriter::batch_write(&device, writes);

        let gbuf_done = Semaphore::new(device.clone())?;
        gbuf_done.name("gbuf_done")?;

        let sky_done = Semaphore::new(device.clone())?;
        sky_done.name("sky_done")?;

        Ok(Self {
            context,
            device: device.clone(),
            render_targets,
            last_depth,
            shader_binding_table,
            tlases,
            _descriptor_pool: descriptor_pool,
            raster_command_buffers,
            compute_command_buffers,
            command_buffers,
            img_available,
            render_finished,
            descriptors,
            descriptor_layouts,
            uniform_buffers,
            uniform_buffers_globals,
            env_uniforms,
            mesh_bufs,
            in_flight,
            last_view_proj: ViewProj::default(),
            debug_mode: DebugMode::None,
            quality: QualitySettings::new(),
            current_frame: 0,
            prev_jitter: (0.5, 0.5),
            frames_in_flight: MAX_FRAMES_IN_FLIGHT,
            tlas_prepare_cmd_buf,
            render_scale: 1.0,
            gbuf_done,
            sky_done,
            passes,
        })
    }

    fn create_passes(
        device: Rc<Device>,
        context: Rc<VulkanContext>,
        render_targets: &mut RenderTargets,
        pipeline_builder: &mut PipelineBuilder,
        descriptor_layouts: &DescriptorLayouts,
    ) -> Result<VulkanRendererPasses, AppError> {
        let sky = SkyPass::create(device.clone(), render_targets, pipeline_builder, descriptor_layouts)?;
        let gbuffer = GBufferPass::create(device.clone(), render_targets, pipeline_builder, descriptor_layouts)?;
        let tonemap = TonemapPass::create(device.clone(), render_targets, pipeline_builder, descriptor_layouts)?;
        let shading = ShadingPass::create(device.clone(), render_targets, pipeline_builder, descriptor_layouts)?;
        let denoise = DenoisePass::create(device.clone(), render_targets, pipeline_builder, descriptor_layouts)?;
        let taa = TaaPass::create(device.clone(), render_targets, pipeline_builder, descriptor_layouts)?;
        let pt = PathTracePass::create(context, render_targets, pipeline_builder, descriptor_layouts)?;
        let depth_debug = DepthDebugPass::create(device.clone(), pipeline_builder, descriptor_layouts)?;

        Ok(VulkanRendererPasses {
            pt,
            tonemap,
            sky,
            denoise,
            depth_debug,
            gbuffer,
            shading,
            taa,
        })
    }

    /// Only images which contain info from previous frames need to be transitioned here.
    /// Rest of images can be discarded between frames.
    fn init_history_images(
        device: &Device,
        command_pool: &CommandPool,
        color_images: &[vk::Image],
        depth_images: &[vk::Image],
    ) -> Result<(), VulkanError> {
        unsafe {
            let cmd_buf = command_pool.allocate_cmd_buffers(1)?.pop().unwrap();

            cmd_buf.begin_one_time()?;

            let mut barriers = Vec::with_capacity(color_images.len() + depth_images.len());

            let barrier_base = vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::MEMORY_WRITE,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::GENERAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                ..Default::default()
            };

            let color_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };
            let depth_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                ..color_range
            };

            for image in color_images {
                barriers.push(vk::ImageMemoryBarrier {
                    image: *image,
                    subresource_range: color_range,
                    ..barrier_base
                });
            }

            for image in depth_images {
                barriers.push(vk::ImageMemoryBarrier {
                    image: *image,
                    subresource_range: depth_range,
                    ..barrier_base
                });
            }

            device.inner.cmd_pipeline_barrier(
                cmd_buf.inner,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );

            cmd_buf.end()?;

            let submit_info = vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: &cmd_buf.inner,
                ..Default::default()
            };

            device
                .inner
                .queue_submit(device.graphics_queue, &[submit_info], vk::Fence::null())
                .map_to_err("Cannot submit queue")?;
            device
                .inner
                .queue_wait_idle(device.graphics_queue)
                .map_to_err("Cannot wait idle")
        }
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
        mesh: &vk::Buffer,
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
            create_buffer_update(
                mesh,
                size_of::<Mat4>() as u64 * 8192,
                desc_set,
                4,
                vk::DescriptorType::STORAGE_BUFFER,
            ),
        ]
    }

    pub fn render_frame(
        &mut self,
        scene: &Scene,
        resource_subsystem: &mut ResourceSubsystem,
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

        if resource_subsystem.prepare_resources(scene, &self.tlas_prepare_cmd_buf)? {
            info!("Resource prepare time: {:.3}s", start.elapsed().as_secs_f32());
        }

        let prepare_end = start.elapsed().as_secs_f32();

        let tlas_start = Instant::now();

        let tlas_index = resource_subsystem.build_tlas_index(scene);

        self.tlases[self.current_frame] = TopLevelAs::prepare(
            self.device.clone(),
            self.context.allocator.clone(),
            self.context.rt_acc_struct_ext.clone(),
            &self.tlas_prepare_cmd_buf,
            &resource_subsystem.blases,
            tlas_index,
        )?;

        let tlas_time = tlas_start.elapsed().as_secs_f32();

        let command_buffer = &self.command_buffers[self.current_frame];
        let raster_command_buffer = &self.raster_command_buffers[self.current_frame];
        let compute_command_buffer = &self.compute_command_buffers[self.current_frame];

        let width = (drawable_size.0 as f32 * self.render_scale).max(1.0) as u32;
        let height = (drawable_size.1 as f32 * self.render_scale).max(1.0) as u32;

        let aspect_ratio = width as f32 / height as f32;
        let fov_rad = math::deg_to_rad(scene.camera.fov);
        let fovy = math::fovx_to_fovy(fov_rad, aspect_ratio);

        let mut offset = math::halton(context.frame_index % 16 + 1);
        if context.clear_taa {
            offset = (0.5, 0.5);
        }
        let (offset_x, offset_y) = ((offset.0 - 0.5) / width as f32, (offset.1 - 0.5) / height as f32);

        let mut proj = nalgebra_glm::perspective_rh_zo(aspect_ratio, fovy, 500.0, 0.01);

        // add jitter
        proj.m13 = offset_x;
        proj.m23 = offset_y;

        // flip because of vulkan
        proj.m22 *= -1.0;

        let view_proj = ViewProj {
            view: scene.camera.view(),
            projection: proj,
            view_prev: self.last_view_proj.view,
            projection_prev: self.last_view_proj.projection,
            view_inverse: scene.camera.view().try_inverse().unwrap(),
            projection_inverse: proj.try_inverse().unwrap(),
            near: 500.0,
            far: 0.01,
        };

        let mesh_start = Instant::now();

        let collected_meshes =
            MeshCollector::collect_transforms(scene, context.culling, &view_proj.view, &view_proj.projection_inverse);

        let mesh_time = mesh_start.elapsed().as_secs_f32() + prepare_end;

        let globals = Globals {
            debug_mode: self.debug_mode as i32,
            res_x: drawable_size.0 as f32,
            res_y: drawable_size.1 as f32,
            draw_res_x: width as f32,
            draw_res_y: height as f32,
            time: context.total_time,
            frame_index: context.frame_index,
            _padding_0: 0,
            current_jitter: offset,
            prev_jitter: self.prev_jitter,
        };

        self.uniform_buffers[self.current_frame].fill_host(view_proj.to_bytes().as_ref())?;
        self.uniform_buffers_globals[self.current_frame].fill_host(globals.to_bytes().as_ref())?;
        self.env_uniforms[self.current_frame].fill_host(scene.env.to_bytes().as_ref())?;
        self.mesh_bufs[self.current_frame].fill_host(collected_meshes.data.as_ref())?;

        let mut writes = Vec::new();

        let mut borrow = self.descriptors[self.current_frame].borrow_mut();

        writes.extend(borrow.update_resources(&self.render_targets, Some(resource_subsystem))?);

        DescriptorWriter::batch_write(&self.device, writes);
        drop(borrow);
        let mut writes = Vec::new();

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            writes.push(Self::create_tlas_update_descriptor_set(
                &self.descriptors[i].borrow().global_set.inner,
                &self.tlases[i],
            ));
        }

        DescriptorWriter::batch_write(&self.device, writes);

        raster_command_buffer.begin()?;

        let record_start = Instant::now();

        self.record_command_buffer(
            scene,
            command_buffer,
            raster_command_buffer,
            compute_command_buffer,
            &self.context.swap_chain.borrow().images[image_index as usize],
            context,
            &collected_meshes.draws,
            resource_subsystem,
            (width, height),
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
            self.context.device.inner.cmd_pipeline_barrier(
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
                    subresource_range: crate::vulkan::Image::single_color_layer_range(),
                    ..Default::default()
                }],
            );
        }

        command_buffer.end()?;

        let record_time = record_start.elapsed().as_secs_f32();

        let end = Instant::now();

        let signal_semaphores = [self.render_finished[image_index as usize].inner];
        self.context.device.queue_submit(SubmitInfo {
            queue: &self.context.device.graphics_queue,
            fence: &self.in_flight[self.current_frame],
            wait_semaphores: &[
                self.img_available[self.current_frame].inner,
                self.gbuf_done.inner,
                self.sky_done.inner,
            ],
            signal_semaphores: &signal_semaphores,
            command_buffers: &[command_buffer.inner],
            wait_stages: vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
        })?;

        let is_suboptimal = self.context.swap_chain.borrow().present(PresentInfo {
            wait_semaphores: &signal_semaphores,
            image_index,
        })?;

        if is_suboptimal {
            self.resize(drawable_size)?;
        }

        self.current_frame = (self.current_frame + 1) % self.frames_in_flight;
        self.last_view_proj = view_proj;
        self.prev_jitter = offset;

        let stats = FrameStats {
            cpu_time: (end - start).as_secs_f32(),
            tlas_time,
            record_time,
            mesh_time,
            objects_rendered: collected_meshes.draws.iter().map(|a| a.count).sum(),
            draw_calls: collected_meshes.draws.len() as u32,
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

        Self::init_history_images(
            &self.device,
            &self.context.graphics_command_pool,
            &[
                self.passes.taa.render_target_history.borrow().image.inner,
                self.passes.denoise.direct_render_target_acc.borrow().image.inner,
                self.passes.denoise.direct_render_target_history.borrow().image.inner,
                self.passes.denoise.indirect_render_target_acc.borrow().image.inner,
                self.passes.denoise.indirect_render_target_history.borrow().image.inner,
                self.passes.denoise.moments_direct_render_target.borrow().image.inner,
                self.passes.denoise.moments_indirect_render_target.borrow().image.inner,
            ],
            &[self.render_targets.get_ref("last_depth").unwrap().image.inner],
        )?;

        Ok(())
    }

    fn record_command_buffer(
        &self,
        scene: &Scene,
        command_buffer: &CommandBuffer,
        raster_command_buffer: &CommandBuffer,
        compute_command_buffer: &CommandBuffer,
        target: &vk::Image,
        context: &FrameContext,
        draw_data: &Vec<DrawData>,
        resource_subsystem: &ResourceSubsystem,
        viewport_size: (u32, u32),
    ) -> Result<(), AppError> {
        let descriptors = self.descriptors[self.current_frame].borrow();

        self.passes.gbuffer.record(
            raster_command_buffer,
            Ref::clone(&descriptors),
            resource_subsystem,
            draw_data,
            viewport_size,
        )?;

        raster_command_buffer.end()?;

        self.context.device.queue_submit(SubmitInfo {
            queue: &self.context.device.graphics_queue,
            fence: &Fence::null(self.context.device.clone()),
            wait_semaphores: &[],
            signal_semaphores: &[self.gbuf_done.inner],
            command_buffers: &[raster_command_buffer.inner],
            wait_stages: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        })?;

        compute_command_buffer.begin()?;

        self.tlases[self.current_frame].build(compute_command_buffer)?;

        let sky_sampler = match &scene.env.sky.variant {
            SkyVariant::Shader => {
                self.passes.sky.record(compute_command_buffer, &descriptors)?;

                self.passes.sky.render_target.borrow().sampler_index.unwrap()
            }
            SkyVariant::SingleColor(_) => 0,
            SkyVariant::Textured(ir) => descriptors.samplers[&ir.id],
        };

        compute_command_buffer.end()?;

        self.context.device.queue_submit(SubmitInfo {
            queue: &self.context.device.compute_queue,
            fence: &Fence::null(self.context.device.clone()),
            wait_semaphores: &[],
            signal_semaphores: &[self.sky_done.inner],
            command_buffers: &[compute_command_buffer.inner],
            wait_stages: vk::PipelineStageFlags::COMPUTE_SHADER
                | vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
        })?;

        command_buffer.begin()?;
        self.passes.pt.record(
            command_buffer,
            &descriptors,
            PathTraceInputs {
                color: &self.passes.gbuffer.render_target_color.borrow(),
                depth: &self.passes.gbuffer.render_target_depth.borrow(),
                normal: &self.passes.gbuffer.render_target_normal.borrow(),
                sky_sampler,
                sbt: &self.shader_binding_table,
                bounces: self.quality.pt_bounces,
                direct_trace_distance: self.quality.rt_direct_trace_distance,
                indirect_trace_distance: self.quality.rt_indirect_trace_distance,
                indirect_intensity_clamp: self.quality.indirect_light_clamp,
            },
            viewport_size,
        )?;
        self.passes.denoise.record(
            command_buffer,
            &descriptors,
            DenoiseInputs {
                depth: &self.passes.gbuffer.render_target_depth.borrow(),
                last_depth: &self.last_depth.borrow(),
                normal: &self.passes.gbuffer.render_target_normal.borrow(),
                rt_direct: &self.passes.pt.direct_render_target.borrow(),
                rt_indirect: &self.passes.pt.indirect_render_target.borrow(),
                clear: context.clear_taa,
                use_spatial_denoise: self.quality.use_spatial_denoise,
            },
            viewport_size,
        )?;
        self.passes.shading.record(
            command_buffer,
            &descriptors,
            ShadingInputs {
                color: &self.passes.gbuffer.render_target_color.borrow(),
                depth: &self.passes.gbuffer.render_target_depth.borrow(),
                normal: &self.passes.gbuffer.render_target_normal.borrow(),
                direct: &self.passes.denoise.direct_render_target_acc.borrow(),
                indirect: &self.passes.denoise.indirect_render_target_acc.borrow(),
                sky_sampler,
            },
            viewport_size,
        )?;
        self.passes.taa.record(
            command_buffer,
            &descriptors,
            TaaInputs {
                depth: &self.passes.gbuffer.render_target_depth.borrow(),
                last_depth: &self.last_depth.borrow(),
                src: &self.passes.shading.render_target.borrow(),
                clear: context.clear_taa,
            },
            viewport_size,
        )?;

        unsafe {
            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier {
                    src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ,
                    dst_access_mask: vk::AccessFlags::SHADER_WRITE,
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::GENERAL,
                    image: self.passes.tonemap.render_target.borrow().image.inner,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                }],
            );
        }

        self.passes.tonemap.record(
            command_buffer,
            &descriptors,
            self.passes.taa.render_target.clone(),
            viewport_size,
            false,
        )?;

        self.record_end_copy(command_buffer, target, viewport_size)?;

        Ok(())
    }

    fn record_end_copy(
        &self,
        command_buffer: &CommandBuffer,
        target: &vk::Image,
        viewport_size: (u32, u32),
    ) -> Result<(), VulkanError> {
        let descriptors = self.descriptors[self.current_frame].borrow();

        let src_image = match self.debug_mode {
            DebugMode::None => self.passes.tonemap.render_target.borrow().image.inner,
            DebugMode::Direct => self.passes.pt.direct_render_target.borrow().image.inner,
            DebugMode::Indirect => self.passes.pt.indirect_render_target.borrow().image.inner,
            DebugMode::Time => self.render_targets.get_ref("rt_direct").unwrap().image.inner,
            DebugMode::BaseColor => self.render_targets.get_ref("gbuffer_color").unwrap().image.inner,
            DebugMode::Normal => self.render_targets.get_ref("gbuffer_normal").unwrap().image.inner,
            DebugMode::DisOcclusion => self.render_targets.get_ref("taa_target").unwrap().image.inner,
            DebugMode::VarianceDirect => self.render_targets.get_ref("denoise_direct_acc").unwrap().image.inner,
            DebugMode::VarianceIndirect => self.render_targets.get_ref("denoise_indirect_acc").unwrap().image.inner,
            DebugMode::DenoiseDirect => self.render_targets.get_ref("denoise_direct_out").unwrap().image.inner,
            DebugMode::DenoiseIndirect => self.render_targets.get_ref("denoise_indirect_out").unwrap().image.inner,
            DebugMode::Raw => self.render_targets.get_ref("tonemap").unwrap().image.inner,
            DebugMode::Depth => {
                // runs convert, result is in tonemap
                self.passes.depth_debug.record(
                    command_buffer,
                    &descriptors,
                    self.passes.shading.render_target.clone(),
                    self.passes.gbuffer.render_target_depth.clone(),
                    viewport_size,
                )?;
                self.render_targets.get_ref("tonemap").unwrap().image.inner
            }
        };

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
                    dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ,
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

            Ok(())
        }
    }
}

pub struct TlasIndex {
    pub index: Vec<(u64, Mat4)>,
}

pub struct FrameContext {
    pub delta_time: f32,
    pub total_time: f32,
    pub clear_taa: bool,
    pub culling: bool,
    pub frame_index: u32,
}

fn create_buffer_update<'a>(
    buffer: &vk::Buffer,
    range: u64,
    dst_set: &vk::DescriptorSet,
    binding: u32,
    desc_type: vk::DescriptorType,
) -> DescriptorWrite<'a> {
    let buffer_info = vk::DescriptorBufferInfo {
        buffer: *buffer,
        offset: 0,
        range,
    };

    let write = vk::WriteDescriptorSet {
        dst_set: *dst_set,
        dst_binding: binding,
        dst_array_element: 0,
        descriptor_type: desc_type,
        descriptor_count: 1,
        ..Default::default()
    };

    DescriptorWrite {
        write,
        buffer_info: Some(buffer_info),
        tlases: None,
        image_info: None,
    }
}

#[derive(Default, Copy, Clone)]
pub struct FrameStats {
    pub cpu_time: f32,
    pub tlas_time: f32,
    pub record_time: f32,
    pub mesh_time: f32,
    pub objects_rendered: u32,
    pub draw_calls: u32,
}
