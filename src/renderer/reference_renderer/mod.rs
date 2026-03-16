mod acc;
use acc::{AccumulateInputs, AccumulatePass};

mod pt;
use pt::{ReferencePathTraceInputs, ReferencePathtracePass};

use crate::app::frame_stats::FrameReport;
use crate::err::AppError;
use crate::math;
use crate::renderer::buffers::{Globals, ViewProj};
use crate::renderer::debug::DebugMode;
use crate::renderer::descriptors::{DescriptorLayouts, DescriptorWrite, DescriptorWriter, RendererDescriptors};
use crate::renderer::passes::{SkyPass, TonemapPass};
use crate::renderer::quality::QualitySettings;
use crate::renderer::render_target::RenderTargets;
use crate::renderer::{FrameContext, GPUEnv, ResourceSubsystem, TlasIndex, VulkanContext, create_buffer_update, stats};
use crate::scene::{Scene, SkyVariant};
use crate::vulkan::{
    Buffer, CommandBuffer, DebugMarker, DescriptorPool, Fence, PresentInfo, Sampler, Semaphore, ShaderBindingTable,
    SubmitInfo, TopLevelAs, VulkanError,
};
use ash::vk;
use gpu_allocator::MemoryLocation;
use log::info;
use nalgebra_glm::Mat4x4;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::time::Instant;

const MAX_FRAMES_IN_FLIGHT: usize = 2;

struct VulkanMcPathTracerPasses {
    sky: SkyPass,
    tonemap: TonemapPass,
    pt: ReferencePathtracePass,
    accumulate: AccumulatePass,
}

pub struct VulkanMcPathTracer {
    pub context: Rc<VulkanContext>,
    pub render_targets: RenderTargets,
    pub shader_binding_table: ShaderBindingTable,
    pub tlases: Vec<TopLevelAs>,
    pub _descriptor_pool: DescriptorPool,
    pub command_buffers: Vec<CommandBuffer>,
    pub descriptor_layouts: DescriptorLayouts,
    pub descriptors: Vec<Rc<RefCell<RendererDescriptors>>>,
    pub uniform_buffers: Vec<Buffer>,
    pub uniform_buffers_globals: Vec<Buffer>,
    pub mesh_data: Vec<Buffer>,
    pub env_uniforms: Vec<Buffer>,
    pub current_frame: usize,
    pub quality: QualitySettings,
    pub debug_mode: DebugMode,
    pub render_scale: f32,
    img_available: Vec<Semaphore>,
    render_finished: Vec<Semaphore>,
    in_flight: Vec<Fence>,
    frames_in_flight: usize,
    tlas_prepare_cmd_buf: CommandBuffer,
    passes: VulkanMcPathTracerPasses,
}

impl VulkanMcPathTracer {
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

        let sky_pass = SkyPass::create(
            device.clone(),
            &mut render_targets,
            &mut context.pipeline_builder.borrow_mut(),
            &descriptor_layouts,
        )?;

        let tonemap_pass = TonemapPass::create(
            device.clone(),
            &mut render_targets,
            &mut context.pipeline_builder.borrow_mut(),
            &descriptor_layouts,
        )?;

        let accumulate_pass = AccumulatePass::create(
            device.clone(),
            &mut render_targets,
            &mut context.pipeline_builder.borrow_mut(),
            &descriptor_layouts,
        )?;

        let pt_pass = ReferencePathtracePass::create(
            context.clone(),
            &mut render_targets,
            &mut context.pipeline_builder.borrow_mut(),
            &descriptor_layouts,
        )?;

        let passes = VulkanMcPathTracerPasses {
            sky: sky_pass,
            tonemap: tonemap_pass,
            accumulate: accumulate_pass,
            pt: pt_pass,
        };

        let shader_binding_table = ShaderBindingTable::new(
            device.clone(),
            context.allocator.clone(),
            &context.rt_pipeline_ext,
            &passes.pt.pipeline_handle,
            1,
            1,
        )?;

        let command_buffers = context
            .graphics_command_pool
            .allocate_cmd_buffers(MAX_FRAMES_IN_FLIGHT as u32)?;
        let tlas_prepare_cmd_buf = context.compute_command_pool.allocate_cmd_buffers(1)?.pop().unwrap();

        let mut img_available = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut render_finished = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut in_flight = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut uniform_buffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut uniform_buffers_globals = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut env_uniforms = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut descriptors = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut mesh_data = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        let mut writes = Vec::new();
        let mut tlases = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            img_available.push(Semaphore::new(device.clone())?);
            in_flight.push(Fence::new(device.clone())?);

            img_available[i].name(format!("img_available[{}]", i))?;
            in_flight[i].name(format!("in_flight[{}]", i))?;
            command_buffers[i].name(format!("cmd_buffers[{}]", i))?;

            {
                descriptors.push(Rc::new(RefCell::new(RendererDescriptors::build(
                    &descriptor_pool,
                    &descriptor_layouts,
                )?)));
            }

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

            let mesh_buf = Buffer::new(
                device.clone(),
                context.allocator.clone(),
                MemoryLocation::CpuToGpu,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                // will get proper size on first frame
                64,
            )?;

            mesh_data.push(mesh_buf);

            writes.extend(Self::init_global_descriptor_set(
                &descriptors[i].borrow().global_set.inner,
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

        DescriptorWriter::batch_write(&device, writes);

        Ok(Self {
            context: context.clone(),
            render_targets,
            shader_binding_table,
            tlases,
            _descriptor_pool: descriptor_pool,
            command_buffers,
            img_available,
            render_finished,
            descriptors,
            descriptor_layouts,
            uniform_buffers,
            uniform_buffers_globals,
            env_uniforms,
            mesh_data,
            in_flight,
            quality: QualitySettings::new(),
            debug_mode: DebugMode::None,
            current_frame: 0,
            frames_in_flight: MAX_FRAMES_IN_FLIGHT,
            render_scale: 1.0,
            passes,
            tlas_prepare_cmd_buf,
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
        resource_subsystem: &mut ResourceSubsystem,
        drawable_size: (u32, u32),
        context: &FrameContext,
        ui: Option<&imgui::DrawData>,
    ) -> Result<FrameReport, AppError> {
        self.in_flight[self.current_frame].wait()?;
        self.in_flight[self.current_frame].reset()?;

        let (image_index, _is_suboptimal) = self
            .context
            .swap_chain
            .borrow()
            .acquire_next_image(&self.img_available[self.current_frame])?;

        self.command_buffers[self.current_frame].reset()?;

        let mut report = FrameReport::new();

        let start = Instant::now();

        if resource_subsystem.prepare_resources(scene, &self.tlas_prepare_cmd_buf)? {
            info!("Resource prepare time: {:.3}s", start.elapsed().as_secs_f32());
        }

        report.log::<stats::ResourcePrepareStat>(start.elapsed().as_secs_f32() * 1000.0);

        let tlas_start = Instant::now();

        let tlas_index = resource_subsystem.build_tlas_index(scene);

        self.tlas_prepare_cmd_buf.reset()?;

        self.tlases[self.current_frame] = TopLevelAs::prepare(
            self.context.device.clone(),
            self.context.allocator.clone(),
            self.context.rt_acc_struct_ext.clone(),
            &self.tlas_prepare_cmd_buf,
            &resource_subsystem.blases,
            tlas_index,
        )?;

        report.log::<stats::TlasTimeStat>(tlas_start.elapsed().as_secs_f32() * 1000.0);

        let width = (drawable_size.0 as f32 * self.render_scale).max(1.0) as u32;
        let height = (drawable_size.1 as f32 * self.render_scale).max(1.0) as u32;

        let aspect_ratio = width as f32 / height as f32;
        let fov_rad = math::deg_to_rad(scene.camera.fov);
        let fovy = math::fovx_to_fovy(fov_rad, aspect_ratio);

        let mut offset = math::halton(context.frame_index + 1);
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
            near: 500.0,
            far: 0.01,
        };

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
            // reference renderer does not use this, so it can be safely ignored
            prev_jitter: offset,
        };

        let desc_start = Instant::now();

        self.setup_mesh_data(scene, resource_subsystem)?;

        self.uniform_buffers[self.current_frame].fill_host(view_proj.to_bytes().as_ref())?;
        self.uniform_buffers_globals[self.current_frame].fill_host(globals.to_bytes().as_ref())?;
        self.env_uniforms[self.current_frame].fill_host(scene.env.to_bytes().as_ref())?;

        let mut writes = Vec::new();

        let mut borrow = self.descriptors[self.current_frame].borrow_mut();

        writes.extend(borrow.update_resources(&self.render_targets, Some(resource_subsystem))?);

        DescriptorWriter::batch_write(&self.context.device, writes);

        drop(borrow);

        let mut writes = Vec::new();

        writes.push(create_buffer_update(
            &self.mesh_data[self.current_frame].inner,
            self.mesh_data[self.current_frame].size,
            &self.descriptors[self.current_frame].borrow().global_set.inner,
            5,
            vk::DescriptorType::STORAGE_BUFFER,
        ));

        for i in 0..crate::renderer::MAX_FRAMES_IN_FLIGHT {
            writes.push(Self::create_tlas_update_descriptor_set(
                &self.descriptors[i].borrow().global_set.inner,
                &self.tlases[i],
            ));
        }

        DescriptorWriter::batch_write(&self.context.device, writes);

        report.log::<stats::DescTimeStat>(desc_start.elapsed().as_secs_f32() * 1000.0);

        let command_buffer = &self.command_buffers[self.current_frame];

        command_buffer.begin()?;

        let record_start = Instant::now();

        self.record_command_buffer(
            scene,
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

        report.log::<stats::RecordTimeStat>(record_start.elapsed().as_secs_f32() * 1000.0);
        report.log::<stats::CpuTimeStat>(start.elapsed().as_secs_f32() * 1000.0);

        let signal_semaphores = [self.render_finished[image_index as usize].inner];

        self.context.device.queue_submit(SubmitInfo {
            queue: &self.context.device.graphics_queue,
            fence: &self.in_flight[self.current_frame],
            wait_semaphores: &[self.img_available[self.current_frame].inner],
            signal_semaphores: &signal_semaphores,
            command_buffers: &[command_buffer.inner],
            wait_stages: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        })?;

        let is_suboptimal = self.context.swap_chain.borrow().present(PresentInfo {
            wait_semaphores: &signal_semaphores,
            image_index,
        })?;

        if is_suboptimal {
            self.resize(drawable_size)?;
        }

        self.current_frame = (self.current_frame + 1) % self.frames_in_flight;

        let mem_report = self.context.allocator.lock().unwrap().generate_report();

        report.log::<stats::RenderTargetStat>(self.render_targets.targets.len() as u32);
        report.log::<stats::VramUsageStat>(mem_report.total_allocated_bytes as f32 / 1024.0 / 1024.0);
        report.log::<stats::VramAllocatedStat>(mem_report.total_reserved_bytes as f32 / 1024.0 / 1024.0);

        Ok(report)
    }

    pub fn resize(&mut self, drawable_size: (u32, u32)) -> Result<(), AppError> {
        self.context.device.wait_idle()?;
        self.context.recreate_swapchain(drawable_size)?;

        let extent_3d = vk::Extent3D {
            width: drawable_size.0,
            height: drawable_size.1,
            depth: 1,
        };

        self.render_targets.set_extent(extent_3d);
        self.render_targets.resize()?;

        Ok(())
    }

    fn record_command_buffer(
        &self,
        scene: &Scene,
        command_buffer: &CommandBuffer,
        target: &vk::Image,
        context: &FrameContext,
        viewport_size: (u32, u32),
        fov: f32,
    ) -> Result<(), AppError> {
        let descriptors = self.descriptors[self.current_frame].borrow();

        self.tlases[self.current_frame].build(command_buffer)?;

        self.init_frame_images(command_buffer);

        let sky_sampler = match &scene.env.sky.variant {
            SkyVariant::Shader => {
                self.passes.sky.record(command_buffer, &descriptors)?;

                self.passes.sky.render_target.borrow().sampler_index.unwrap()
            }
            SkyVariant::SingleColor(_) => 0,
            SkyVariant::Textured(ir, _) => descriptors.samplers[&ir.id],
        };

        unsafe {
            let barriers = [self.passes.sky.render_target.borrow().image.inner].map(|image| vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                old_layout: vk::ImageLayout::GENERAL,
                new_layout: vk::ImageLayout::GENERAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image,
                subresource_range: crate::vulkan::Image::single_color_layer_range(),
                ..Default::default()
            });

            self.context.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier {
                    src_access_mask: vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
                    dst_access_mask: vk::AccessFlags::SHADER_READ | vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR,
                    ..Default::default()
                }],
                &[],
                &barriers,
            );
        }

        self.passes.pt.record_pt(
            command_buffer,
            &descriptors,
            ReferencePathTraceInputs {
                sky_sampler,
                sbt: &self.shader_binding_table,
                bounces: self.quality.pt_bounces,
                direct_trace_distance: self.quality.rt_direct_trace_distance,
                indirect_trace_distance: self.quality.rt_indirect_trace_distance,
                fov,
                indirect_intensity_clamp: self.quality.indirect_light_clamp,
            },
            context,
            viewport_size,
        )?;
        self.passes.accumulate.record(
            command_buffer,
            &descriptors,
            AccumulateInputs {
                input: &self.passes.pt.render_target.borrow(),
                frame_index: context.frame_index,
                clear: context.clear_taa,
            },
            viewport_size,
        )?;
        self.passes.tonemap.record(
            command_buffer,
            &descriptors,
            self.passes.accumulate.render_target.clone(),
            viewport_size,
            true,
        )?;
        self.record_end_copy(command_buffer, target, viewport_size)?;

        Ok(())
    }

    pub fn init_frame_images(&self, command_buffer: &CommandBuffer) {
        let barriers = [
            self.render_targets.get("rt_out").unwrap().borrow().image.inner,
            self.render_targets.get("acc_out").unwrap().borrow().image.inner,
            self.render_targets.get("tonemap_out").unwrap().borrow().image.inner,
            self.render_targets.get("sky").unwrap().borrow().image.inner,
        ]
        .map(|image| vk::ImageMemoryBarrier {
            src_access_mask: vk::AccessFlags::NONE,
            dst_access_mask: vk::AccessFlags::SHADER_WRITE,
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::GENERAL,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image,
            subresource_range: crate::vulkan::Image::single_color_layer_range(),
            ..Default::default()
        });

        unsafe {
            self.context.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );
        }
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

            self.context.device.inner.cmd_pipeline_barrier(
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
                    subresource_range: crate::vulkan::Image::single_color_layer_range(),
                    ..Default::default()
                }],
            );

            let subresource = vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            };

            self.context.device.inner.cmd_blit_image(
                command_buffer.inner,
                src_image,
                vk::ImageLayout::GENERAL,
                *target,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageBlit {
                    src_subresource: subresource,
                    src_offsets: [offset_min, offset_src_max],
                    dst_subresource: subresource,
                    dst_offsets: [offset_min, offset_dst_max],
                }],
                vk::Filter::LINEAR,
            );

            self.context.device.inner.cmd_pipeline_barrier(
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
                    subresource_range: crate::vulkan::Image::single_color_layer_range(),
                    ..Default::default()
                }],
            );
        }

        Ok(())
    }

    fn setup_mesh_data(&mut self, scene: &Scene, resource_subsystem: &ResourceSubsystem) -> Result<(), AppError> {
        let mut handles = Vec::with_capacity(scene.meshes.len());

        for mesh in &scene.meshes {
            let vulkan_mesh = &resource_subsystem.meshes[&mesh.resource.id];
            let vertex_pointer = vulkan_mesh.buf.addr;
            let index_pointer = vertex_pointer + vulkan_mesh.indices_offset;

            if !mesh.visible {
                continue;
            }

            handles.push(MeshInstanceDataGPU {
                transform_inverse: mesh.inverse,
                vertex_pointer,
                index_pointer,
            })
        }

        let target_size = (handles.len() * size_of::<MeshInstanceDataGPU>()) as u64;

        if self.mesh_data[self.current_frame].size < target_size {
            self.mesh_data[self.current_frame] = Buffer::new(
                self.context.device.clone(),
                self.context.allocator.clone(),
                MemoryLocation::CpuToGpu,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                target_size,
            )?;
        }

        let slice = unsafe { core::slice::from_raw_parts(handles.as_ptr() as *const u8, target_size as usize) };

        self.mesh_data[self.current_frame].fill_host(slice)?;

        Ok(())
    }
}

#[repr(C)]
pub struct MeshInstanceDataGPU {
    pub transform_inverse: Mat4x4,
    pub vertex_pointer: u64,
    pub index_pointer: u64,
}
