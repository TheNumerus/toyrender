use crate::err::AppError;
use crate::math;
use crate::mesh::MeshResource;
use crate::scene::{Environment, Scene};
use crate::vulkan::{
    AccelerationStructure, Buffer, CommandBuffer, CommandPool, DescriptorPool, Device, DeviceQueryResult, Fence,
    Framebuffer, ImageView, Instance, IntoVulkanError, RayTracingAs, RayTracingPipeline, RenderPass, Semaphore,
    ShaderBindingTable, Surface, SwapChain, Vertex, VulkanError, VulkanMesh,
};
use ash::vk::Handle;
use ash::{vk, Entry};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use gpu_allocator::MemoryLocation;
use log::info;
use nalgebra_glm::Mat4;
use sdl2::video::Window;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::fs::File;
use std::path::Path;
use std::rc::Rc;
use std::time::Instant;
use zip::ZipArchive;

mod descriptors;
use descriptors::{DescriptorWriter, RendererDescriptors};

mod passes;
use passes::TonemapPass;

mod pipeline_builder;
use pipeline_builder::PipelineBuilder;

mod quality;
use quality::QualitySettings;

mod render_target;
use render_target::{
    DenoiseRenderTargets, MultipleRenderTargetBuilder, RenderTarget, RenderTargetBuilder, RenderTargetSize,
    RenderTargets,
};

mod debug;
use debug::DebugMode;

mod shader_loader;

use shader_loader::ShaderLoader;
pub use shader_loader::ShaderLoaderError;

const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct VulkanRenderer {
    pub instance: Rc<Instance>,
    pub device: Rc<Device>,
    pub allocator: Rc<RefCell<Allocator>>,
    pub swap_chain: SwapChain,
    pub surface: Surface,
    pub rt_pipeline_ext: RayTracingPipeline,
    pub rt_acc_struct_ext: Rc<RayTracingAs>,
    pub swap_chain_image_views: Vec<ImageView>,
    pub swap_chain_fbs: Vec<Framebuffer>,
    pub render_targets: RenderTargets,
    pub postprocess_render_target: RenderTarget,
    pub postprocess_render_target_2: RenderTarget,
    pub rt_direct_render_target: RenderTarget,
    pub rt_indirect_render_target: RenderTarget,
    pub denoise_render_targets: DenoiseRenderTargets,
    pub pipeline_builder: PipelineBuilder,
    pub render_passes: RenderPasses,
    pub shader_binding_table: ShaderBindingTable,
    pub tlas: AccelerationStructure,
    pub blases: HashMap<u64, AccelerationStructure>,
    pub as_builder_cmd_buf: CommandBuffer,
    pub graphics_command_pool: CommandPool,
    pub compute_command_pool: CommandPool,
    pub descriptor_pool: DescriptorPool,
    pub command_buffers: Vec<CommandBuffer>,
    pub descriptors: RendererDescriptors,
    pub uniform_buffers: Vec<Buffer>,
    pub uniform_buffers_globals: Vec<Buffer>,
    pub env_uniforms: Vec<Buffer>,
    pub current_frame: usize,
    pub quality: QualitySettings,
    pub debug_mode: DebugMode,
    meshes: HashMap<u64, VulkanMesh>,
    fs_quad: VulkanMesh,
    img_available: Vec<Semaphore>,
    render_finished: Vec<Semaphore>,
    in_flight: Vec<Fence>,
    last_view_proj: ViewProj,
    prev_jitter: (f32, f32),
    frames_in_flight: usize,
    tonemap_pass: TonemapPass,
}

impl VulkanRenderer {
    pub fn init(window: &Window) -> Result<Self, AppError> {
        let entry = unsafe { Entry::load().expect("cannot load vulkan entry") };

        let instance = Rc::new(Instance::new(&entry, &window.vulkan_instance_extensions().unwrap())?);

        let surface = window
            .vulkan_create_surface(instance.inner.handle().as_raw() as usize)
            .map_err(AppError::Other)?;
        let surface = Surface::new(&instance, &entry, surface)?;

        let device = Self::init_device(instance.clone(), &surface)?;

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.inner.clone(),
            device: device.inner.clone(),
            physical_device: device.physical_device,
            debug_settings: Default::default(),
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })?;

        let allocator = Rc::new(RefCell::new(allocator));

        let rt_pipeline_ext = RayTracingPipeline::new(&instance, &device)?;
        let rt_acc_struct_ext = Rc::new(RayTracingAs::new(&instance, &device)?);

        let swap_chain = SwapChain::new(device.clone(), &instance, window.drawable_size(), &surface)?;
        let swap_chain_image_views = swap_chain.create_image_views()?;

        let shader_loader = ShaderLoader::from_zip(open_shader_zip("shaders.zip")?)?;

        let descriptor_pool = DescriptorPool::new(
            device.clone(),
            &[
                vk::DescriptorPoolSize {
                    descriptor_count: 30 * MAX_FRAMES_IN_FLIGHT as u32,
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                },
                vk::DescriptorPoolSize {
                    descriptor_count: 30 * MAX_FRAMES_IN_FLIGHT as u32,
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                },
                vk::DescriptorPoolSize {
                    descriptor_count: MAX_FRAMES_IN_FLIGHT as u32,
                    ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                },
            ],
            200 * MAX_FRAMES_IN_FLIGHT as u32,
        )?;

        let descriptors = RendererDescriptors::build(device.clone(), &descriptor_pool, MAX_FRAMES_IN_FLIGHT as u32)?;

        let render_passes = RenderPasses::new(device.clone())?;

        let pipeline_builder = PipelineBuilder::build(
            shader_loader,
            device.clone(),
            &rt_pipeline_ext,
            &descriptors,
            &render_passes,
        )?;

        let extent_3d = vk::Extent3D {
            width: swap_chain.extent.width,
            height: swap_chain.extent.height,
            depth: 1,
        };

        let swap_chain_fbs = swap_chain_image_views
            .iter()
            .map(|iw| {
                Framebuffer::new(
                    device.clone(),
                    render_passes.passes.get("tonemap").unwrap(),
                    swap_chain.extent,
                    &[iw],
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut render_targets = RenderTargets::new(device.clone(), extent_3d);

        let gbuffer = render_targets.add_mrt(
            MultipleRenderTargetBuilder::new("gbuffer", render_passes.passes.get("gb").unwrap().clone())
                .add_target(
                    vk::Format::A2B10G10R10_UNORM_PACK32,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT
                        | vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::TRANSFER_SRC,
                    vk::ImageAspectFlags::COLOR,
                )
                .add_target(
                    vk::Format::A2B10G10R10_UNORM_PACK32,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT
                        | vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::TRANSFER_SRC,
                    vk::ImageAspectFlags::COLOR,
                )
                .add_target(
                    vk::Format::D32_SFLOAT,
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                        | vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::TRANSFER_SRC,
                    vk::ImageAspectFlags::DEPTH,
                ),
        )?;

        let mut postprocess_render_target = RenderTarget::new(
            device.clone(),
            extent_3d,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageAspectFlags::COLOR,
        )?;
        postprocess_render_target.set_names("tonemap")?;

        let mut postprocess_render_target_2 = RenderTarget::new(
            device.clone(),
            extent_3d,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageAspectFlags::COLOR,
        )?;
        postprocess_render_target_2.set_names("tonemap_2")?;

        let mut rt_direct_render_target = RenderTarget::new(
            device.clone(),
            extent_3d,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageAspectFlags::COLOR,
        )?;
        rt_direct_render_target.set_names("rt_direct")?;

        let mut rt_indirect_render_target = RenderTarget::new(
            device.clone(),
            extent_3d,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageAspectFlags::COLOR,
        )?;
        rt_indirect_render_target.set_names("rt_indirect")?;

        let taa_builder = RenderTargetBuilder::new("taa_target")
            .with_transfer()
            .with_storage()
            .with_size(RenderTargetSize::Scaled(1.0))
            .with_color_attachment();

        render_targets.add(taa_builder.duplicate("taa_history_target"))?;
        render_targets.add(taa_builder)?;

        let denoise_render_targets = DenoiseRenderTargets::new(device.clone(), extent_3d)?;

        render_targets.add(RenderTargetBuilder::new_depth("last_depth").with_transfer())?;

        let shader_binding_table = ShaderBindingTable::new(
            device.clone(),
            allocator.clone(),
            &rt_pipeline_ext,
            pipeline_builder.get_rt("pt").unwrap(),
            1,
            1,
        )?;

        let graphics_command_pool = CommandPool::new_graphics(device.clone())?;
        let compute_command_pool = CommandPool::new_compute(device.clone())?;
        let command_buffers = graphics_command_pool.allocate_cmd_buffers(MAX_FRAMES_IN_FLIGHT as u32)?;
        let as_builder_cmd_buf = compute_command_pool.allocate_cmd_buffers(1)?.pop().unwrap();

        let tlas = AccelerationStructure::top_build(
            device.clone(),
            allocator.clone(),
            rt_acc_struct_ext.clone(),
            &as_builder_cmd_buf,
            &HashMap::new(),
            TlasIndex { index: Vec::new() },
            vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                | vk::BuildAccelerationStructureFlagsKHR::ALLOW_DATA_ACCESS,
        )?;

        let mut img_available = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut render_finished = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut in_flight = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut uniform_buffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut uniform_buffers_globals = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut env_uniforms = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        let (fs_quad_v, fs_quad_i) = crate::mesh::fs_triangle();
        let fs_quad = VulkanMesh::new(
            device.clone(),
            allocator.clone(),
            &graphics_command_pool,
            &MeshResource::new(fs_quad_v.to_vec(), crate::mesh::Indices::U32(fs_quad_i.to_vec())),
        )?;

        unsafe {
            let cmd_buf = graphics_command_pool.allocate_cmd_buffers(1)?.pop().unwrap();

            cmd_buf.begin_one_time()?;

            let mut barriers = [
                rt_direct_render_target.image.inner,
                rt_indirect_render_target.image.inner,
                render_targets.get("taa_target").unwrap().borrow().image.inner,
                render_targets.get("taa_history_target").unwrap().borrow().image.inner,
                denoise_render_targets.direct_acc.image.inner,
                denoise_render_targets.indirect_acc.image.inner,
                denoise_render_targets.direct_out.image.inner,
                denoise_render_targets.indirect_out.image.inner,
                denoise_render_targets.direct_history.image.inner,
                denoise_render_targets.indirect_history.image.inner,
            ]
            .into_iter()
            .map(|image| vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::MEMORY_WRITE,
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
            })
            .collect::<Vec<_>>();

            barriers.push(vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::MEMORY_WRITE,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::GENERAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: render_targets.get("last_depth").unwrap().borrow().image.inner,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            });

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
                .map_to_err("Cannot wait idle")?;
        }

        let mut writes = Vec::new();

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            img_available.push(Semaphore::new(device.clone())?);
            render_finished.push(Semaphore::new(device.clone())?);
            in_flight.push(Fence::new(device.clone())?);

            let uniform_buffer = Buffer::new(
                device.clone(),
                allocator.clone(),
                MemoryLocation::CpuToGpu,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                std::mem::size_of::<ViewProj>() as u64 * 2,
            )?;

            uniform_buffers.push(uniform_buffer);

            let uniform_buffer_global = Buffer::new(
                device.clone(),
                allocator.clone(),
                MemoryLocation::CpuToGpu,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                std::mem::size_of::<Globals>() as u64,
            )?;

            uniform_buffers_globals.push(uniform_buffer_global);

            let env_buf = Buffer::new(
                device.clone(),
                allocator.clone(),
                MemoryLocation::CpuToGpu,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                std::mem::size_of::<f32>() as u64 * 8,
            )?;

            env_uniforms.push(env_buf);

            writes.push(descriptors.global_sets[i].update_globals(&uniform_buffers_globals[i]));
            writes.push(descriptors.global_sets[i].update_tlas(&tlas));
            writes.push(descriptors.global_sets[i].update_view(&uniform_buffers[i]));
            writes.push(descriptors.global_sets[i].update_env(&env_uniforms[i]));
        }

        writes.push(
            descriptors
                .rt_set
                .update_targets(&rt_direct_render_target, &rt_indirect_render_target),
        );
        writes.push(descriptors.rt_set.update_rt_src(&gbuffer));

        writes.push(
            descriptors
                .taa_set
                .update_target(&render_targets.get("taa_target").unwrap().borrow()),
        );
        writes.push(descriptors.taa_set.update_taa_src(
            &postprocess_render_target,
            &render_targets.get("taa_history_target").unwrap().borrow(),
            &gbuffer,
            &render_targets.get("last_depth").unwrap().borrow(),
        ));

        writes.push(
            descriptors
                .denoise_indirect_temporal_set
                .update_target(&denoise_render_targets.indirect_acc),
        );
        writes.push(descriptors.denoise_indirect_temporal_set.update_taa_src(
            &rt_indirect_render_target,
            &denoise_render_targets.indirect_history,
            &gbuffer,
            &render_targets.get("last_depth").unwrap().borrow(),
        ));
        writes.push(
            descriptors
                .denoise_direct_temporal_set
                .update_target(&denoise_render_targets.direct_acc),
        );
        writes.push(descriptors.denoise_direct_temporal_set.update_taa_src(
            &rt_direct_render_target,
            &denoise_render_targets.direct_history,
            &gbuffer,
            &render_targets.get("last_depth").unwrap().borrow(),
        ));

        writes.push(descriptors.atrous_direct_set.update_rt_src(&gbuffer));
        writes.push(
            descriptors
                .atrous_direct_set
                .update_atrous(&denoise_render_targets.direct_out, &denoise_render_targets.direct_acc),
        );

        writes.push(descriptors.atrous_indirect_set.update_rt_src(&gbuffer));
        writes.push(descriptors.atrous_indirect_set.update_atrous(
            &denoise_render_targets.indirect_out,
            &denoise_render_targets.indirect_acc,
        ));

        writes.push(descriptors.light_set.update_light(
            &gbuffer,
            &denoise_render_targets.direct_out,
            &denoise_render_targets.indirect_out,
        ));
        writes.push(
            descriptors
                .post_process_set
                .update_pp(&render_targets.get("taa_target").unwrap().borrow()),
        );

        DescriptorWriter::batch_write(&device, writes);

        Ok(Self {
            instance,
            device: device.clone(),
            allocator,
            surface,
            rt_pipeline_ext,
            rt_acc_struct_ext,
            swap_chain,
            swap_chain_image_views,
            swap_chain_fbs,
            render_targets,
            postprocess_render_target,
            postprocess_render_target_2,
            rt_direct_render_target,
            rt_indirect_render_target,
            denoise_render_targets,
            pipeline_builder,
            render_passes,
            shader_binding_table,
            tlas,
            blases: HashMap::new(),
            as_builder_cmd_buf,
            fs_quad,
            graphics_command_pool,
            compute_command_pool,
            descriptor_pool,
            command_buffers,
            img_available,
            render_finished,
            descriptors,
            uniform_buffers,
            uniform_buffers_globals,
            env_uniforms,
            in_flight,
            last_view_proj: ViewProj::default(),
            debug_mode: DebugMode::None,
            quality: QualitySettings::new(),
            meshes: HashMap::new(),
            current_frame: 0,
            prev_jitter: (0.5, 0.5),
            frames_in_flight: MAX_FRAMES_IN_FLIGHT,
            tonemap_pass: TonemapPass { device: device.clone() },
        })
    }

    fn init_device(instance: Rc<Instance>, surface: &Surface) -> Result<Rc<Device>, AppError> {
        let devices = Device::query_applicable(&instance, surface)?;
        if devices.is_empty() {
            return Err(AppError::Other("No GPUs with Vulkan support found".into()));
        }

        let no_applicable = devices.iter().all(|d| !d.is_applicable());

        if no_applicable {
            let mut message = String::from("No applicable device found: \n");
            for device in devices {
                if let DeviceQueryResult::NotApplicable(device) = device {
                    let extensions = device.missing_extensions.iter().fold(String::new(), |mut out, ext| {
                        let _ = write!(out, "\t\t - {}", ext);
                        out
                    });

                    let device_msg = format!("\t{}\n\t\tMissing extensions:\n{}", device.name, &extensions);
                    message.push_str(&device_msg);
                }
            }

            Err(AppError::Other(message))
        } else {
            for device in devices {
                if let DeviceQueryResult::Applicable(device) = device {
                    return Ok(Rc::new(Device::new(instance, device, surface)?));
                }
            }
            unreachable!()
        }
    }

    pub fn render_frame(
        &mut self,
        scene: &Scene,
        drawable_size: (u32, u32),
        context: &FrameContext,
    ) -> Result<f32, AppError> {
        self.prepare_meshes(scene)?;

        self.in_flight[self.current_frame].wait()?;
        let (image_index, _is_suboptimal) = self
            .swap_chain
            .acquire_next_image(&self.img_available[self.current_frame])?;
        self.in_flight[self.current_frame].reset()?;
        self.command_buffers[self.current_frame].reset()?;

        let command_buffer = &self.command_buffers[self.current_frame];
        let framebuffer = &self.swap_chain_fbs[image_index as usize];

        let aspect_ratio = drawable_size.0 as f32 / drawable_size.1 as f32;
        let fov_rad = math::deg_to_rad(scene.camera.fov);
        let fovy = math::fovx_to_fovy(fov_rad, aspect_ratio);

        let mut offset = math::halton(context.frame_index % 16 + 1);
        if context.clear_taa {
            offset = (0.5, 0.5);
        }
        let (offset_x, offset_y) = (
            (1.0 * offset.0 - 0.5) / drawable_size.0 as f32,
            (1.0 * offset.1 - 0.5) / drawable_size.1 as f32,
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
        };

        let globals = Globals {
            exposure: scene.env.exposure,
            debug_mode: self.debug_mode as i32,
            res_x: drawable_size.0 as f32,
            res_y: drawable_size.1 as f32,
            time: context.total_time,
            frame_index: context.frame_index,
            half_res: if self.quality.half_res { 1 } else { 0 },
            current_jitter: offset,
            prev_jitter: self.prev_jitter,
        };

        let env = env_to_buffer(&scene.env);
        let view_proj_buf = view_proj_to_buffer(&view_proj, &self.last_view_proj);

        self.uniform_buffers[self.current_frame].fill_host(view_proj_buf.as_ref())?;
        self.uniform_buffers_globals[self.current_frame].fill_host(globals.to_boxed_slice().as_ref())?;
        self.env_uniforms[self.current_frame].fill_host(env.as_ref())?;

        let tlas_index = self.build_tlas_index(scene);

        self.tlas = AccelerationStructure::top_build(
            self.device.clone(),
            self.allocator.clone(),
            self.rt_acc_struct_ext.clone(),
            &self.as_builder_cmd_buf,
            &self.blases,
            tlas_index,
            vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                | vk::BuildAccelerationStructureFlagsKHR::ALLOW_DATA_ACCESS,
        )?;

        let mut writes = Vec::new();

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            writes.push(self.descriptors.global_sets[i].update_tlas(&self.tlas));
        }

        DescriptorWriter::batch_write(&self.device, writes);

        let start = Instant::now();

        self.record_command_buffer(
            command_buffer,
            framebuffer,
            &self.swap_chain.images[image_index as usize],
            scene,
            context,
        )?;

        let end = Instant::now();

        let wait_semaphores = [self.img_available[self.current_frame].inner];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.render_finished[self.current_frame].inner];

        let submit_info = vk::SubmitInfo {
            wait_semaphore_count: 1,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: 1,
            p_command_buffers: &self.command_buffers[self.current_frame].inner,
            signal_semaphore_count: 1,
            p_signal_semaphores: signal_semaphores.as_ptr(),
            ..Default::default()
        };

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

        let present_info = vk::PresentInfoKHR {
            wait_semaphore_count: 1,
            p_wait_semaphores: signal_semaphores.as_ptr(),
            swapchain_count: 1,
            p_swapchains: &self.swap_chain.swapchain,
            p_image_indices: &image_index,
            ..Default::default()
        };

        let is_suboptimal = unsafe {
            self.swap_chain
                .loader
                .queue_present(self.device.present_queue, &present_info)
                .expect("cannot present")
        };

        if is_suboptimal {
            self.resize(drawable_size)?;
        }

        self.current_frame = (self.current_frame + 1) % self.frames_in_flight;
        self.last_view_proj = view_proj;
        self.prev_jitter = offset;

        Ok((end - start).as_secs_f32())
    }

    pub fn resize(&mut self, drawable_size: (u32, u32)) -> Result<(), VulkanError> {
        self.device.wait_idle()?;

        // need to drop before creating new ones
        self.swap_chain_fbs.clear();
        self.swap_chain_image_views.clear();

        self.swap_chain
            .recreate(self.device.clone(), drawable_size, &self.surface)?;

        self.swap_chain_image_views = self.swap_chain.create_image_views()?;

        let extent_3d = vk::Extent3D {
            width: self.swap_chain.extent.width,
            height: self.swap_chain.extent.height,
            depth: 1,
        };
        let extent_3d_half = vk::Extent3D {
            width: self.swap_chain.extent.width / 2,
            height: self.swap_chain.extent.height / 2,
            depth: 1,
        };

        self.render_targets.set_extent(extent_3d);
        self.render_targets.resize()?;

        let rtao_extent = if self.quality.half_res {
            extent_3d_half
        } else {
            extent_3d
        };

        self.postprocess_render_target.resize(extent_3d)?;
        self.postprocess_render_target_2.resize(extent_3d)?;
        self.rt_direct_render_target.resize(rtao_extent)?;
        self.rt_indirect_render_target.resize(rtao_extent)?;
        self.denoise_render_targets.direct_out.resize(rtao_extent)?;
        self.denoise_render_targets.indirect_out.resize(rtao_extent)?;
        self.denoise_render_targets.direct_acc.resize(rtao_extent)?;
        self.denoise_render_targets.indirect_acc.resize(rtao_extent)?;
        self.denoise_render_targets.direct_history.resize(rtao_extent)?;
        self.denoise_render_targets.indirect_history.resize(rtao_extent)?;

        unsafe {
            let cmd_buf = self.graphics_command_pool.allocate_cmd_buffers(1)?.pop().unwrap();

            cmd_buf.begin_one_time()?;

            let mut barriers = vec![
                self.rt_direct_render_target.image.inner,
                self.rt_indirect_render_target.image.inner,
                self.render_targets.get("taa_target").unwrap().borrow().image.inner,
                self.render_targets
                    .get("taa_history_target")
                    .unwrap()
                    .borrow()
                    .image
                    .inner,
                self.denoise_render_targets.direct_out.image.inner,
                self.denoise_render_targets.indirect_out.image.inner,
                self.denoise_render_targets.direct_acc.image.inner,
                self.denoise_render_targets.indirect_acc.image.inner,
                self.denoise_render_targets.direct_history.image.inner,
                self.denoise_render_targets.indirect_history.image.inner,
            ]
            .into_iter()
            .map(|image| vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::MEMORY_WRITE,
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
            })
            .collect::<Vec<_>>();

            barriers.push(vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::MEMORY_WRITE,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::GENERAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: self.render_targets.get("last_depth").unwrap().borrow().image.inner,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            });

            self.device.inner.cmd_pipeline_barrier(
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

            self.device
                .inner
                .queue_submit(self.device.graphics_queue, &[submit_info], vk::Fence::null())
                .map_to_err("Cannot submit queue")?;
            self.device
                .inner
                .queue_wait_idle(self.device.graphics_queue)
                .map_to_err("Cannot wait idle")?;
        }

        let writes = vec![
            self.descriptors
                .rt_set
                .update_targets(&self.rt_direct_render_target, &self.rt_indirect_render_target),
            self.descriptors
                .rt_set
                .update_rt_src(&self.render_targets.get_mrt("gbuffer").unwrap()),
            self.descriptors
                .taa_set
                .update_target(&self.render_targets.get("taa_target").unwrap().borrow()),
            self.descriptors.taa_set.update_taa_src(
                &self.postprocess_render_target,
                &self.render_targets.get("taa_history_target").unwrap().borrow(),
                &self.render_targets.get_mrt("gbuffer").unwrap(),
                &self.render_targets.get("last_depth").unwrap().borrow(),
            ),
            self.descriptors
                .denoise_indirect_temporal_set
                .update_target(&self.denoise_render_targets.indirect_acc),
            self.descriptors.denoise_indirect_temporal_set.update_taa_src(
                &self.rt_indirect_render_target,
                &self.denoise_render_targets.indirect_history,
                &self.render_targets.get_mrt("gbuffer").unwrap(),
                &self.render_targets.get("last_depth").unwrap().borrow(),
            ),
            self.descriptors
                .denoise_direct_temporal_set
                .update_target(&self.denoise_render_targets.direct_acc),
            self.descriptors.denoise_direct_temporal_set.update_taa_src(
                &self.rt_direct_render_target,
                &self.denoise_render_targets.direct_history,
                &self.render_targets.get_mrt("gbuffer").unwrap(),
                &self.render_targets.get("last_depth").unwrap().borrow(),
            ),
            self.descriptors
                .atrous_direct_set
                .update_rt_src(&self.render_targets.get_mrt("gbuffer").unwrap()),
            self.descriptors.atrous_direct_set.update_atrous(
                &self.denoise_render_targets.direct_out,
                &self.denoise_render_targets.direct_acc,
            ),
            self.descriptors
                .atrous_indirect_set
                .update_rt_src(&self.render_targets.get_mrt("gbuffer").unwrap()),
            self.descriptors.atrous_indirect_set.update_atrous(
                &self.denoise_render_targets.indirect_out,
                &self.denoise_render_targets.indirect_acc,
            ),
            self.descriptors.light_set.update_light(
                &self.render_targets.get_mrt("gbuffer").unwrap(),
                &self.denoise_render_targets.direct_out,
                &self.denoise_render_targets.indirect_out,
            ),
            self.descriptors
                .post_process_set
                .update_pp(&self.render_targets.get("taa_target").unwrap().borrow()),
        ];

        DescriptorWriter::batch_write(&self.device, writes);

        self.swap_chain_fbs = self
            .swap_chain_image_views
            .iter()
            .map(|iw| {
                Framebuffer::new(
                    self.device.clone(),
                    self.render_passes.passes.get("tonemap").unwrap(),
                    self.swap_chain.extent,
                    &[iw],
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(())
    }

    fn record_command_buffer(
        &self,
        command_buffer: &CommandBuffer,
        framebuffer: &Framebuffer,
        target: &vk::Image,
        scene: &Scene,
        context: &FrameContext,
    ) -> Result<(), VulkanError> {
        command_buffer.begin()?;

        self.record_gbuffer_pass(command_buffer, scene);
        self.record_rtao(command_buffer);
        self.record_denoise_temporal(command_buffer, context.clear_taa);
        self.record_denoise_pass(command_buffer);
        self.record_light_pass(command_buffer);
        self.record_taa_pass(command_buffer, context.clear_taa);
        //self.record_postprocess_pass(command_buffer, framebuffer);

        self.tonemap_pass.record(command_buffer, self)?;

        self.record_end_copy(command_buffer, target)?;

        command_buffer.end()
    }

    fn record_gbuffer_pass(&self, command_buffer: &CommandBuffer, scene: &Scene) {
        self.device.begin_label("GBuffer", command_buffer);

        let clears = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                },
            },
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.5, 0.5, 0.5, 1.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 },
            },
        ];

        let render_pass_info = vk::RenderPassBeginInfo {
            render_pass: self.render_passes.passes.get("gb").unwrap().inner,
            framebuffer: self
                .render_targets
                .get_mrt("gbuffer")
                .unwrap()
                .borrow()
                .framebuffer
                .inner,
            render_area: vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: self.swap_chain.extent,
            },
            clear_value_count: clears.len() as u32,
            p_clear_values: clears.as_ptr(),
            ..Default::default()
        };

        let viewport = vk::Viewport {
            width: self.swap_chain.extent.width as f32,
            height: self.swap_chain.extent.height as f32,
            max_depth: 1.0,
            ..Default::default()
        };

        command_buffer.begin_render_pass(&render_pass_info, vk::SubpassContents::INLINE);

        let pipeline = self.pipeline_builder.get_graphics("deferred").unwrap();

        command_buffer.bind_graphics_pipeline(pipeline);
        command_buffer.set_viewport(viewport);
        command_buffer.set_scissor(render_pass_info.render_area);

        unsafe {
            self.device.inner.cmd_bind_descriptor_sets(
                command_buffer.inner,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.layout,
                0,
                &[self.descriptors.global_sets[self.current_frame].inner.inner],
                &[],
            );
        }

        for instance in &scene.meshes {
            let mesh = &instance.resource;
            let mesh_data = &self.meshes[&mesh.id];

            command_buffer.bind_vertex_buffers(&[&mesh_data.buf], &[0]);

            unsafe {
                let constants = std::slice::from_raw_parts(instance.transform.as_ptr() as *const u8, 64);

                self.device.inner.cmd_push_constants(
                    command_buffer.inner,
                    pipeline.layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    constants,
                );

                self.device.inner.cmd_bind_index_buffer(
                    command_buffer.inner,
                    mesh_data.buf.inner.inner,
                    mesh_data.indices_offset,
                    vk::IndexType::UINT32,
                );

                self.device
                    .inner
                    .cmd_draw_indexed(command_buffer.inner, mesh_data.index_count as u32, 1, 0, 0, 0);
            }
        }

        command_buffer.end_render_pass();

        self.device.end_label(command_buffer);
    }

    fn record_rtao(&self, command_buffer: &CommandBuffer) {
        self.device.begin_label("Path Tracing", command_buffer);

        let pipeline = self.pipeline_builder.get_rt("pt").unwrap();

        command_buffer.bind_rt_pipeline(pipeline);

        unsafe {
            self.device.inner.cmd_bind_descriptor_sets(
                command_buffer.inner,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                pipeline.layout,
                0,
                &[
                    self.descriptors.global_sets[self.current_frame].inner.inner,
                    self.descriptors.rt_set.inner.inner,
                ],
                &[],
            );

            let samples = self.quality.rtao_samples.to_le_bytes();

            self.device.inner.cmd_push_constants(
                command_buffer.inner,
                pipeline.layout,
                vk::ShaderStageFlags::RAYGEN_KHR,
                0,
                &samples,
            );

            let (width, height) = if self.quality.half_res {
                (self.swap_chain.extent.width / 2, self.swap_chain.extent.height / 2)
            } else {
                (self.swap_chain.extent.width, self.swap_chain.extent.height)
            };

            self.rt_pipeline_ext.loader.cmd_trace_rays(
                command_buffer.inner,
                &self.shader_binding_table.raygen_region,
                &self.shader_binding_table.miss_region,
                &self.shader_binding_table.hit_region,
                &self.shader_binding_table.call_region,
                width,
                height,
                1,
            );

            let barrier = vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::MEMORY_WRITE,
                dst_access_mask: vk::AccessFlags::MEMORY_READ,
                old_layout: vk::ImageLayout::GENERAL,
                new_layout: vk::ImageLayout::GENERAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: self.rt_direct_render_target.image.inner,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            };

            let barrier_2 = vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::MEMORY_WRITE,
                dst_access_mask: vk::AccessFlags::MEMORY_READ,
                old_layout: vk::ImageLayout::GENERAL,
                new_layout: vk::ImageLayout::GENERAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: self.rt_indirect_render_target.image.inner,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            };

            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier, barrier_2],
            );
        }

        self.device.end_label(command_buffer);
    }

    fn record_denoise_temporal(&self, command_buffer: &CommandBuffer, clear: bool) {
        self.device.begin_label("RT Denoise Temporal", command_buffer);

        let pipeline = self.pipeline_builder.get_compute("denoise_temporal").unwrap();

        command_buffer.bind_compute_pipeline(pipeline);

        unsafe {
            self.device.inner.cmd_bind_descriptor_sets(
                command_buffer.inner,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                0,
                &[
                    self.descriptors.global_sets[self.current_frame].inner.inner,
                    self.descriptors.denoise_indirect_temporal_set.inner.inner,
                ],
                &[],
            );

            let clear = if clear {
                1_i32.to_le_bytes()
            } else {
                0_i32.to_le_bytes()
            };

            self.device.inner.cmd_push_constants(
                command_buffer.inner,
                pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &clear,
            );

            let x = (self.swap_chain.extent.width / 16) + 1;
            let y = (self.swap_chain.extent.height / 16) + 1;

            self.device.inner.cmd_dispatch(command_buffer.inner, x, y, 1);

            self.device.inner.cmd_bind_descriptor_sets(
                command_buffer.inner,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                0,
                &[
                    self.descriptors.global_sets[self.current_frame].inner.inner,
                    self.descriptors.denoise_direct_temporal_set.inner.inner,
                ],
                &[],
            );

            self.device.inner.cmd_push_constants(
                command_buffer.inner,
                pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &clear,
            );

            self.device.inner.cmd_dispatch(command_buffer.inner, x, y, 1);

            let barriers = [
                self.denoise_render_targets.indirect_acc.image.inner,
                self.denoise_render_targets.direct_acc.image.inner,
            ]
            .map(|image| vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::MEMORY_WRITE,
                dst_access_mask: vk::AccessFlags::MEMORY_READ,
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
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );
        }

        self.device.end_label(command_buffer);
    }

    fn record_denoise_pass(&self, command_buffer: &CommandBuffer) {
        self.device.begin_label("RT Denoise", command_buffer);

        let pipeline = self.pipeline_builder.get_compute("atrous").unwrap();

        command_buffer.bind_compute_pipeline(pipeline);

        let (width, height) = if self.quality.half_res {
            (self.swap_chain.extent.width / 2, self.swap_chain.extent.height / 2)
        } else {
            (self.swap_chain.extent.width, self.swap_chain.extent.height)
        };

        let x = (width / 16) + 1;
        let y = (height / 16) + 1;

        unsafe {
            for level in 0..4_i32 {
                let level = level.to_le_bytes();

                self.device.inner.cmd_bind_descriptor_sets(
                    command_buffer.inner,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.layout,
                    0,
                    &[
                        self.descriptors.global_sets[self.current_frame].inner.inner,
                        self.descriptors.atrous_indirect_set.inner.inner,
                    ],
                    &[],
                );

                self.device.inner.cmd_push_constants(
                    command_buffer.inner,
                    pipeline.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &level,
                );

                self.device.inner.cmd_dispatch(command_buffer.inner, x, y, 1);

                self.device.inner.cmd_bind_descriptor_sets(
                    command_buffer.inner,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.layout,
                    0,
                    &[
                        self.descriptors.global_sets[self.current_frame].inner.inner,
                        self.descriptors.atrous_direct_set.inner.inner,
                    ],
                    &[],
                );

                self.device.inner.cmd_push_constants(
                    command_buffer.inner,
                    pipeline.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &level,
                );

                self.device.inner.cmd_dispatch(command_buffer.inner, x, y, 1);

                let barriers = [
                    self.denoise_render_targets.indirect_out.image.inner,
                    self.denoise_render_targets.direct_out.image.inner,
                ]
                .map(|image| vk::ImageMemoryBarrier {
                    src_access_mask: vk::AccessFlags::MEMORY_WRITE,
                    dst_access_mask: vk::AccessFlags::MEMORY_READ,
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
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &barriers,
                );
            }
        }

        self.device.end_label(command_buffer);
    }

    fn record_light_pass(&self, command_buffer: &CommandBuffer) {
        self.device.begin_label("Lighting", command_buffer);

        let attachments = [vk::RenderingAttachmentInfo {
            image_view: self.postprocess_render_target.view.inner,
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            load_op: vk::AttachmentLoadOp::DONT_CARE,
            store_op: vk::AttachmentStoreOp::STORE,
            ..Default::default()
        }];

        let rendering_info = vk::RenderingInfo {
            render_area: vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: self.swap_chain.extent,
            },
            layer_count: 1,
            color_attachment_count: attachments.len() as u32,
            p_color_attachments: attachments.as_ptr(),
            ..Default::default()
        };

        unsafe {
            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier {
                    src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ,
                    dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    image: self.postprocess_render_target.image.inner,
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

        let pipeline = self.pipeline_builder.get_graphics("light").unwrap();

        command_buffer.begin_rendering(&rendering_info);

        command_buffer.bind_graphics_pipeline(pipeline);

        command_buffer.bind_vertex_buffers(&[&self.fs_quad.buf], &[0]);

        unsafe {
            self.device.inner.cmd_bind_index_buffer(
                command_buffer.inner,
                self.fs_quad.buf.inner.inner,
                self.fs_quad.indices_offset,
                vk::IndexType::UINT32,
            );

            self.device.inner.cmd_bind_descriptor_sets(
                command_buffer.inner,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.layout,
                1,
                &[self.descriptors.light_set.inner.inner],
                &[],
            );

            self.device
                .inner
                .cmd_draw_indexed(command_buffer.inner, self.fs_quad.index_count as u32, 1, 0, 0, 0);
        }

        command_buffer.end_rendering();

        unsafe {
            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier {
                    src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    dst_access_mask: vk::AccessFlags::SHADER_READ,
                    old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    image: self.postprocess_render_target.image.inner,
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

        self.device.end_label(command_buffer);
    }

    fn record_taa_pass(&self, command_buffer: &CommandBuffer, clear: bool) {
        self.device.begin_label("TAA Resolve", command_buffer);

        let pipeline = self.pipeline_builder.get_compute("taa").unwrap();

        command_buffer.bind_compute_pipeline(pipeline);

        unsafe {
            self.device.inner.cmd_bind_descriptor_sets(
                command_buffer.inner,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                0,
                &[
                    self.descriptors.global_sets[self.current_frame].inner.inner,
                    self.descriptors.taa_set.inner.inner,
                ],
                &[],
            );

            let clear = if clear {
                1_i32.to_le_bytes()
            } else {
                0_i32.to_le_bytes()
            };

            self.device.inner.cmd_push_constants(
                command_buffer.inner,
                pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &clear,
            );

            let x = (self.swap_chain.extent.width / 16) + 1;
            let y = (self.swap_chain.extent.height / 16) + 1;

            self.device.inner.cmd_dispatch(command_buffer.inner, x, y, 1);

            let extent_3d = vk::Extent3D {
                width: self.swap_chain.extent.width,
                height: self.swap_chain.extent.height,
                depth: 1,
            };

            let barriers = [self.render_targets.get("last_depth").unwrap().borrow().image.inner].map(|image| {
                vk::ImageMemoryBarrier {
                    src_access_mask: vk::AccessFlags::MEMORY_WRITE,
                    dst_access_mask: vk::AccessFlags::MEMORY_READ,
                    old_layout: vk::ImageLayout::GENERAL,
                    new_layout: vk::ImageLayout::GENERAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::DEPTH,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                }
            });

            self.device.inner.cmd_pipeline_barrier(
                command_buffer.inner,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );

            self.device.inner.cmd_copy_image(
                command_buffer.inner,
                self.render_targets.get_mrt("gbuffer").unwrap().borrow().images[2].inner,
                vk::ImageLayout::GENERAL,
                self.render_targets.get("last_depth").unwrap().borrow().image.inner,
                vk::ImageLayout::GENERAL,
                &[vk::ImageCopy {
                    extent: extent_3d,
                    dst_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::DEPTH,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    src_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::DEPTH,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                }],
            );

            self.device.inner.cmd_copy_image(
                command_buffer.inner,
                self.denoise_render_targets.direct_out.image.inner,
                vk::ImageLayout::GENERAL,
                self.denoise_render_targets.direct_history.image.inner,
                vk::ImageLayout::GENERAL,
                &[vk::ImageCopy {
                    extent: extent_3d,
                    dst_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    src_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                }],
            );
            self.device.inner.cmd_copy_image(
                command_buffer.inner,
                self.denoise_render_targets.indirect_out.image.inner,
                vk::ImageLayout::GENERAL,
                self.denoise_render_targets.indirect_history.image.inner,
                vk::ImageLayout::GENERAL,
                &[vk::ImageCopy {
                    extent: extent_3d,
                    dst_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    src_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                }],
            );

            self.device.inner.cmd_copy_image(
                command_buffer.inner,
                self.render_targets.get("taa_target").unwrap().borrow().image.inner,
                vk::ImageLayout::GENERAL,
                self.render_targets
                    .get("taa_history_target")
                    .unwrap()
                    .borrow()
                    .image
                    .inner,
                vk::ImageLayout::GENERAL,
                &[vk::ImageCopy {
                    extent: extent_3d,
                    dst_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    src_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                }],
            );
        }

        self.device.end_label(command_buffer);
    }

    fn record_end_copy(&self, command_buffer: &CommandBuffer, target: &vk::Image) -> Result<(), VulkanError> {
        let src_image = match self.debug_mode {
            DebugMode::None => self.postprocess_render_target.image.inner,
            DebugMode::Direct => self.rt_direct_render_target.image.inner,
            DebugMode::Indirect => self.rt_indirect_render_target.image.inner,
            DebugMode::Time => self.postprocess_render_target.image.inner,
            DebugMode::BaseColor => self.render_targets.get_mrt("gbuffer").unwrap().borrow().images[0].inner,
            DebugMode::Normal => self.render_targets.get_mrt("gbuffer").unwrap().borrow().images[1].inner,
            DebugMode::Depth => self.render_targets.get_mrt("gbuffer").unwrap().borrow().images[0].inner,
            DebugMode::DisOcclusion => self.postprocess_render_target.image.inner,
            DebugMode::VarianceDirect => self.denoise_render_targets.direct_out.image.inner,
            DebugMode::VarianceIndirect => self.denoise_render_targets.indirect_out.image.inner,
        };

        unsafe {
            let offset_min = vk::Offset3D { x: 0, y: 0, z: 1 };
            let offset_max = vk::Offset3D {
                x: self.swap_chain.extent.width as i32,
                y: self.swap_chain.extent.height as i32,
                z: 1,
            };

            self.device.inner.cmd_blit_image(
                command_buffer.inner,
                src_image,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                *target,
                vk::ImageLayout::PRESENT_SRC_KHR,
                &[vk::ImageBlit {
                    src_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    src_offsets: [offset_min, offset_max],
                    dst_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    dst_offsets: [offset_min, offset_max],
                }],
                vk::Filter::LINEAR,
            );
            Ok(())
        }
    }

    fn prepare_meshes(&mut self, scene: &Scene) -> Result<(), AppError> {
        let mut changed = false;

        for instance in &scene.meshes {
            if let std::collections::hash_map::Entry::Vacant(e) = self.meshes.entry(instance.resource.id) {
                let mesh = VulkanMesh::new(
                    self.device.clone(),
                    self.allocator.clone(),
                    &self.graphics_command_pool,
                    &instance.resource,
                )?;
                changed = true;
                e.insert(mesh);
            }
        }

        if changed {
            let mut geos = HashMap::new();
            let mut ranges = HashMap::new();

            let mut processed = HashSet::new();

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
                self.allocator.clone(),
                self.rt_acc_struct_ext.clone(),
                &self.compute_command_pool,
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

pub struct TlasIndex {
    pub index: Vec<(u64, Mat4)>,
}

pub struct FrameContext {
    pub delta_time: f32,
    pub total_time: f32,
    pub clear_taa: bool,
    pub frame_index: u32,
}

#[derive(Default)]
pub struct ViewProj {
    pub view: nalgebra_glm::Mat4,
    pub projection: nalgebra_glm::Mat4,
}

pub struct Globals {
    pub exposure: f32,
    pub debug_mode: i32,
    pub res_x: f32,
    pub res_y: f32,
    pub time: f32,
    pub frame_index: u32,
    pub half_res: i32,
    pub current_jitter: (f32, f32),
    pub prev_jitter: (f32, f32),
}

impl Globals {
    pub fn to_boxed_slice(&self) -> Box<[u8]> {
        let mut vec = Vec::with_capacity(std::mem::size_of::<Self>());

        vec.extend(self.exposure.to_le_bytes());
        vec.extend(self.debug_mode.to_le_bytes());
        vec.extend(self.res_x.to_le_bytes());
        vec.extend(self.res_y.to_le_bytes());
        vec.extend(self.time.to_le_bytes());
        vec.extend(self.frame_index.to_le_bytes());
        vec.extend(self.half_res.to_le_bytes());
        vec.extend(self.current_jitter.0.to_le_bytes());
        vec.extend(self.current_jitter.1.to_le_bytes());
        vec.extend(self.prev_jitter.0.to_le_bytes());
        vec.extend(self.prev_jitter.1.to_le_bytes());

        vec.into_boxed_slice()
    }
}

pub struct RenderPasses {
    pub passes: HashMap<String, Rc<RenderPass>>,
}

impl RenderPasses {
    pub fn new(device: Rc<Device>) -> Result<Self, VulkanError> {
        let pass_gb = RenderPass::new_gbuffer(device.clone())?;
        let pass_tonemap = RenderPass::new_post_process(device.clone())?;

        Ok(Self {
            passes: HashMap::from([
                ("gb".to_owned(), Rc::new(pass_gb)),
                ("tonemap".to_owned(), Rc::new(pass_tonemap)),
            ]),
        })
    }
}

fn open_shader_zip(path: impl AsRef<Path>) -> Result<ZipArchive<File>, AppError> {
    let mut base =
        std::env::current_exe().map_err(|_| AppError::Other("Cannot get current path to executable".into()))?;

    base.pop();
    base.push(path);

    info!("Loading shaders from {:?}", base);

    let file = File::open(&base).map_err(|e| AppError::Other(format!("Cannot open shaders library: {e}")))?;
    let arch = ZipArchive::new(file).map_err(|e| AppError::Other(format!("Cannot read shaders library: {e}")))?;

    Ok(arch)
}

fn env_to_buffer(env: &Environment) -> Box<[u8]> {
    let mut buf = Vec::new();

    for x in [
        env.sun_direction.x,
        env.sun_direction.y,
        env.sun_direction.z,
        0.0,
        env.sun_color.x,
        env.sun_color.y,
        env.sun_color.z,
        0.0,
    ] {
        buf.extend(x.to_le_bytes());
    }

    buf.into_boxed_slice()
}

fn view_proj_to_buffer(current: &ViewProj, old: &ViewProj) -> Box<[u8]> {
    let mut buf = Vec::new();

    for x in [current.view, old.view, current.projection, old.projection] {
        buf.extend(unsafe {
            std::slice::from_raw_parts(x.as_ptr() as *const u8, std::mem::size_of::<f32>() * x.len())
        });
    }

    buf.into_boxed_slice()
}
