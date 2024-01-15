use crate::err::AppError;
use crate::mesh::MeshResource;
use crate::scene::Scene;
use crate::vulkan::{
    AccelerationStructure, Buffer, CommandBuffer, CommandPool, DescriptorPool, Device, DeviceQueryResult, Fence,
    Framebuffer, ImageView, Instance, IntoVulkanError, RayTracingAs, RayTracingPipeline, RenderPass, Semaphore,
    ShaderBindingTable, Surface, SwapChain, Vertex, VulkanError, VulkanMesh,
};
use ash::vk::{Extent3D, Handle};
use ash::{vk, Entry};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use log::info;
use nalgebra_glm::Mat4;
use sdl2::video::Window;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::Path;
use std::rc::Rc;
use std::time::Instant;
use zip::ZipArchive;

mod descriptors;
use descriptors::{DescriptorWriter, RendererDescriptors};

mod pipeline_builder;

mod quality;
use quality::QualitySettings;

mod render_target;
use render_target::{GBuffer, RenderTarget};

mod shader_loader;
use crate::math;
use crate::renderer::pipeline_builder::PipelineBuilder;
use shader_loader::ShaderLoader;
pub use shader_loader::ShaderLoaderError;

const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct VulkanRenderer {
    pub instance: Rc<Instance>,
    pub device: Rc<Device>,
    pub allocator: Allocator,
    pub swap_chain: SwapChain,
    pub surface: Surface,
    pub rt_pipeline_ext: RayTracingPipeline,
    pub rt_acc_struct_ext: Rc<RayTracingAs>,
    pub swap_chain_image_views: Vec<ImageView>,
    pub swap_chain_fbs: Vec<Framebuffer>,
    pub gbuffer: GBuffer,
    pub postprocess_render_target: RenderTarget,
    pub rtao_render_target: RenderTarget,
    pub taa_render_target: RenderTarget,
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
    pub current_frame: usize,
    pub quality: QualitySettings,
    pub debug_mode: i32,
    meshes: HashMap<u64, VulkanMesh>,
    fs_quad: VulkanMesh,
    img_available: Vec<Semaphore>,
    render_finished: Vec<Semaphore>,
    in_flight: Vec<Fence>,
    frames_in_flight: usize,
}

impl VulkanRenderer {
    pub fn init(window: &Window) -> Result<Self, AppError> {
        let entry = unsafe { Entry::load().expect("cannot load vulkan entry") };

        let instance = Rc::new(Instance::new(&entry, &window.vulkan_instance_extensions().unwrap())?);

        let surface = window
            .vulkan_create_surface(instance.inner.handle().as_raw() as usize)
            .map_err(AppError::Other)?;
        let surface = Surface::new(&instance, &entry, surface)?;

        let devices = match Device::query_applicable(&instance, &surface)? {
            DeviceQueryResult::ApplicableDevices(d) => Ok(d),
            DeviceQueryResult::NoDevice => Err(AppError::Other("no GPUs with Vulkan support found".into())),
            DeviceQueryResult::NoApplicableDevice => Err(AppError::Other("No suitable physical device found".into())),
        }?;

        let device = Rc::new(Device::new(instance.clone(), devices[0], &surface)?);

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.inner.clone(),
            device: device.inner.clone(),
            physical_device: device.physical_device,
            debug_settings: Default::default(),
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })?;

        let rt_pipeline_ext = RayTracingPipeline::new(&instance, &device)?;
        let rt_acc_struct_ext = Rc::new(RayTracingAs::new(&instance, &device)?);

        let swap_chain = SwapChain::new(device.clone(), &instance, window.drawable_size(), &surface)?;
        let swap_chain_image_views = swap_chain.create_image_views()?;

        let shader_loader = ShaderLoader::from_zip(open_shader_zip("shaders.zip")?)?;

        let descriptor_pool = DescriptorPool::new(
            device.clone(),
            &[
                vk::DescriptorPoolSize {
                    descriptor_count: 8 * MAX_FRAMES_IN_FLIGHT as u32,
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                },
                vk::DescriptorPoolSize {
                    descriptor_count: 16 * MAX_FRAMES_IN_FLIGHT as u32,
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

        let extent_3d = Extent3D {
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

        let gbuffer = GBuffer::new(device.clone(), extent_3d, render_passes.passes.get("gb").unwrap())?;

        let postprocess_render_target = RenderTarget::new(
            device.clone(),
            extent_3d,
            Some(render_passes.passes.get("light").unwrap()),
            vk::Format::A2B10G10R10_UNORM_PACK32,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
        )?;
        let rtao_render_target = RenderTarget::new(
            device.clone(),
            extent_3d,
            None,
            vk::Format::A2B10G10R10_UNORM_PACK32,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE,
            vk::ImageAspectFlags::COLOR,
        )?;
        let taa_render_target = RenderTarget::new(
            device.clone(),
            extent_3d,
            None,
            vk::Format::A2B10G10R10_UNORM_PACK32,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE,
            vk::ImageAspectFlags::COLOR,
        )?;

        let shader_binding_table = ShaderBindingTable::new(
            device.clone(),
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

        let (fs_quad_v, fs_quad_i) = crate::mesh::fs_triangle();
        let fs_quad = VulkanMesh::new(
            device.clone(),
            &graphics_command_pool,
            &MeshResource::new(fs_quad_v.to_vec(), crate::mesh::Indices::U32(fs_quad_i.to_vec())),
        )?;

        unsafe {
            let cmd_buf = graphics_command_pool.allocate_cmd_buffers(1)?.pop().unwrap();

            cmd_buf.begin_one_time()?;

            let barrier = vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::MEMORY_WRITE,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::GENERAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: rtao_render_target.image.inner,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            };

            device.inner.cmd_pipeline_barrier(
                cmd_buf.inner,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
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
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                std::mem::size_of::<ViewProj>() as u64,
                true,
                false,
            )?;

            uniform_buffers.push(uniform_buffer);

            let uniform_buffer_global = Buffer::new(
                device.clone(),
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                std::mem::size_of::<Globals>() as u64,
                true,
                false,
            )?;

            uniform_buffers_globals.push(uniform_buffer_global);

            writes.push(descriptors.global_sets[i].update_globals(&uniform_buffers_globals[i]));
            writes.push(descriptors.global_sets[i].update_tlas(&tlas));
            writes.push(descriptors.global_sets[i].update_view(&uniform_buffers[i]));
        }

        writes.push(descriptors.rtao_set.update_target(&rtao_render_target));
        writes.push(descriptors.rtao_set.update_rt_src(&gbuffer));

        writes.push(descriptors.taa_set.update_target(&taa_render_target));
        writes.push(descriptors.taa_set.update_taa_src(&postprocess_render_target));

        writes.push(descriptors.light_set.update_light(&gbuffer, &rtao_render_target));
        writes.push(descriptors.post_process_set.update_pp(&taa_render_target));

        DescriptorWriter::batch_write(&device, writes);

        Ok(Self {
            instance,
            device,
            allocator,
            surface,
            rt_pipeline_ext,
            rt_acc_struct_ext,
            swap_chain,
            swap_chain_image_views,
            swap_chain_fbs,
            gbuffer,
            postprocess_render_target,
            rtao_render_target,
            taa_render_target,
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
            in_flight,
            debug_mode: 0,
            quality: QualitySettings::new(),
            meshes: HashMap::new(),
            current_frame: 0,
            frames_in_flight: MAX_FRAMES_IN_FLIGHT,
        })
    }

    pub fn render_frame(
        &mut self,
        scene: &Scene,
        drawable_size: (u32, u32),
        context: &FrameContext,
    ) -> Result<f32, VulkanError> {
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

        let offset = math::halton(context.frame_index);
        let (offset_x, offset_y) = (
            (offset.0 - 0.5) / drawable_size.0 as f32,
            (offset.1 - 0.5) / drawable_size.1 as f32,
        );

        let mut proj = nalgebra_glm::perspective_rh_zo(aspect_ratio, fovy, 0.01, 5000.0);

        // add jitter
        proj.m13 = offset_x * 1.5;
        proj.m23 = offset_y * 1.5;

        // flip because of vulkan
        proj.m22 *= -1.0;

        let ubo = ViewProj {
            view: scene.camera.view(),
            projection: proj,
        };

        let globals = Globals {
            max_bright: 4.0,
            debug_mode: self.debug_mode,
            res_x: drawable_size.0 as f32,
            res_y: drawable_size.1 as f32,
            time: context.total_time,
            frame_index: context.frame_index,
            half_res: if self.quality.half_res { 1 } else { 0 },
        };

        unsafe {
            self.uniform_buffers[self.current_frame].fill_host(std::slice::from_raw_parts(
                std::ptr::addr_of!(ubo) as *const u8,
                std::mem::size_of::<ViewProj>(),
            ))?;

            self.uniform_buffers_globals[self.current_frame].fill_host(globals.to_boxed_slice().as_ref())?;
        }

        let tlas_index = self.build_tlas_index(scene);

        self.tlas = AccelerationStructure::top_build(
            self.device.clone(),
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

        self.record_command_buffer(command_buffer, framebuffer, scene, context)?;

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

        let rtao_extent = if self.quality.half_res {
            extent_3d_half
        } else {
            extent_3d
        };

        self.gbuffer
            .resize(extent_3d, self.render_passes.passes.get("gb").unwrap())?;
        self.postprocess_render_target
            .resize(extent_3d, Some(self.render_passes.passes.get("light").unwrap()))?;
        self.rtao_render_target.resize(rtao_extent, None)?;
        self.taa_render_target.resize(extent_3d, None)?;

        unsafe {
            let cmd_buf = self.graphics_command_pool.allocate_cmd_buffers(1)?.pop().unwrap();

            cmd_buf.begin_one_time()?;

            let barrier = vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::MEMORY_WRITE,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::GENERAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: self.rtao_render_target.image.inner,
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
                cmd_buf.inner,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
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
            self.descriptors.rtao_set.update_target(&self.rtao_render_target),
            self.descriptors.rtao_set.update_rt_src(&self.gbuffer),
            self.descriptors.taa_set.update_target(&self.taa_render_target),
            self.descriptors.taa_set.update_taa_src(&self.postprocess_render_target),
            self.descriptors
                .light_set
                .update_light(&self.gbuffer, &self.rtao_render_target),
            self.descriptors.post_process_set.update_pp(&self.taa_render_target),
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
        scene: &Scene,
        context: &FrameContext,
    ) -> Result<(), VulkanError> {
        command_buffer.begin()?;

        self.record_gbuffer_pass(command_buffer, scene);
        self.record_rtao(command_buffer);
        self.record_light_pass(command_buffer);
        self.record_taa_pass(command_buffer, context.clear_taa);
        self.record_postprocess_pass(command_buffer, framebuffer);

        command_buffer.end()
    }

    fn record_gbuffer_pass(&self, command_buffer: &CommandBuffer, scene: &Scene) {
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
                depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
            },
        ];

        let render_pass_info = vk::RenderPassBeginInfo {
            render_pass: self.render_passes.passes.get("gb").unwrap().inner,
            framebuffer: self.gbuffer.framebuffer.inner,
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
    }

    fn record_rtao(&self, command_buffer: &CommandBuffer) {
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
                    self.descriptors.rtao_set.inner.inner,
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
        }
    }

    fn record_light_pass(&self, command_buffer: &CommandBuffer) {
        let render_pass_info = vk::RenderPassBeginInfo {
            render_pass: self.render_passes.passes.get("light").unwrap().inner,
            framebuffer: self.postprocess_render_target.framebuffer.as_ref().unwrap().inner,
            render_area: vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: self.swap_chain.extent,
            },
            clear_value_count: 0,
            ..Default::default()
        };

        let pipeline = self.pipeline_builder.get_graphics("light").unwrap();

        command_buffer.begin_render_pass(&render_pass_info, vk::SubpassContents::INLINE);

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

        command_buffer.end_render_pass();
    }

    fn record_taa_pass(&self, command_buffer: &CommandBuffer, clear: bool) {
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
        }
    }

    fn record_postprocess_pass(&self, command_buffer: &CommandBuffer, framebuffer: &Framebuffer) {
        let render_pass_info = vk::RenderPassBeginInfo {
            render_pass: self.render_passes.passes.get("tonemap").unwrap().inner,
            framebuffer: framebuffer.inner,
            render_area: vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: self.swap_chain.extent,
            },
            clear_value_count: 0,
            ..Default::default()
        };

        let pipeline = self.pipeline_builder.get_graphics("tonemap").unwrap();

        command_buffer.begin_render_pass(&render_pass_info, vk::SubpassContents::INLINE);

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
                &[self.descriptors.post_process_set.inner.inner],
                &[],
            );

            self.device
                .inner
                .cmd_draw_indexed(command_buffer.inner, self.fs_quad.index_count as u32, 1, 0, 0, 0);
        }

        command_buffer.end_render_pass();
    }

    fn prepare_meshes(&mut self, scene: &Scene) -> Result<(), VulkanError> {
        let mut changed = false;

        for instance in &scene.meshes {
            if let std::collections::hash_map::Entry::Vacant(e) = self.meshes.entry(instance.resource.id) {
                let mesh = VulkanMesh::new(self.device.clone(), &self.graphics_command_pool, &instance.resource)?;
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

                let addr = res.buf.inner.get_device_addr().unwrap();
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

pub struct ViewProj {
    pub view: nalgebra_glm::Mat4,
    pub projection: nalgebra_glm::Mat4,
}

pub struct Globals {
    pub max_bright: f32,
    pub debug_mode: i32,
    pub res_x: f32,
    pub res_y: f32,
    pub time: f32,
    pub frame_index: u32,
    pub half_res: i32,
}

impl Globals {
    pub fn to_boxed_slice(&self) -> Box<[u8]> {
        let mut vec = Vec::with_capacity(std::mem::size_of::<Self>());

        vec.extend(self.max_bright.to_ne_bytes());
        vec.extend(self.debug_mode.to_ne_bytes());
        vec.extend(self.res_x.to_ne_bytes());
        vec.extend(self.res_y.to_ne_bytes());
        vec.extend(self.time.to_ne_bytes());
        vec.extend(self.frame_index.to_ne_bytes());
        vec.extend(self.half_res.to_ne_bytes());

        vec.into_boxed_slice()
    }
}

pub struct RenderPasses {
    pub passes: HashMap<String, RenderPass>,
}

impl RenderPasses {
    pub fn new(device: Rc<Device>) -> Result<Self, VulkanError> {
        let pass_gb = RenderPass::new_gbuffer(device.clone())?;
        let pass_light = RenderPass::new_light(device.clone())?;
        let pass_tonemap = RenderPass::new_post_process(device.clone())?;

        Ok(Self {
            passes: HashMap::from([
                ("gb".to_owned(), pass_gb),
                ("light".to_owned(), pass_light),
                ("tonemap".to_owned(), pass_tonemap),
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
