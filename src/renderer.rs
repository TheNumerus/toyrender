use crate::err::AppError;
use crate::mesh::MeshResource;
use crate::renderer::descriptors::{DescLayout, RendererDescriptors};
use crate::scene::Scene;
use crate::vulkan::{
    Buffer, CommandBuffer, CommandPool, DescriptorPool, Device, DeviceQueryResult, Fence, Framebuffer, Image,
    ImageView, Instance, Pipeline, RenderPass, Sampler, Semaphore, ShaderModule, ShaderStage, Surface, SwapChain,
    VulkanError, VulkanMesh,
};
use ash::vk::{Extent2D, Extent3D, Handle};
use ash::{vk, Entry};
use sdl2::video::Window;
use std::collections::HashMap;
use std::rc::Rc;

mod descriptors;

const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct VulkanRenderer {
    pub instance: Rc<Instance>,
    pub device: Rc<Device>,
    pub swap_chain: SwapChain,
    pub surface: Surface,
    pub swap_chain_image_views: Vec<ImageView>,
    pub swap_chain_fbs: Vec<Framebuffer>,
    pub gbuffer: GBuffer,
    pub tonemap: Tonemap,
    pub pipeline_gb: Pipeline,
    pub pipeline_light: Pipeline,
    pub pipeline_tonemap: Pipeline,
    pub render_pass_gb: RenderPass,
    pub render_pass_light: RenderPass,
    pub render_pass_tonemap: RenderPass,
    pub command_pool: CommandPool,
    pub descriptor_pool: DescriptorPool,
    pub command_buffers: Vec<CommandBuffer>,
    pub descriptors: RendererDescriptors,
    pub uniform_buffers: Vec<Buffer>,
    pub uniform_buffers_globals: Vec<Buffer>,
    pub current_frame: usize,
    meshes: HashMap<u64, VulkanMesh>,
    fs_quad: VulkanMesh,
    img_available: Vec<Semaphore>,
    render_finished: Vec<Semaphore>,
    in_flight: Vec<Fence>,
    max_frames_in_flight: usize,
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

        let swap_chain = SwapChain::new(device.clone(), &instance, window.drawable_size(), &surface)?;
        let swap_chain_image_views = swap_chain.create_image_views()?;

        let deferred_vert_module = ShaderModule::new(
            include_bytes!("../build/deferred_vert.spv"),
            device.clone(),
            ShaderStage::Vertex,
            None,
        )?;
        let deferred_frag_module = ShaderModule::new(
            include_bytes!("../build/deferred_frag.spv"),
            device.clone(),
            ShaderStage::Fragment,
            None,
        )?;

        let deferred_stages = [deferred_vert_module.stage_info(), deferred_frag_module.stage_info()];

        let light_vert_module = ShaderModule::new(
            include_bytes!("../build/light_vert.spv"),
            device.clone(),
            ShaderStage::Vertex,
            None,
        )?;
        let light_frag_module = ShaderModule::new(
            include_bytes!("../build/light_frag.spv"),
            device.clone(),
            ShaderStage::Fragment,
            None,
        )?;
        let tonemap_frag_module = ShaderModule::new(
            include_bytes!("../build/tonemap_frag.spv"),
            device.clone(),
            ShaderStage::Fragment,
            None,
        )?;

        let light_stages = [light_vert_module.stage_info(), light_frag_module.stage_info()];

        let tonemap_stages = [light_vert_module.stage_info(), tonemap_frag_module.stage_info()];

        let mut descriptors = RendererDescriptors::build(device.clone())?;

        let render_pass_gb = RenderPass::new_gbuffer(device.clone())?;
        let render_pass_light = RenderPass::new_light(device.clone())?;
        let render_pass_tonemap = RenderPass::new_post_process(device.clone())?;

        let pipeline_gb = Pipeline::new(
            device.clone(),
            swap_chain.extent,
            &render_pass_gb,
            &deferred_stages,
            &[
                descriptors.get_layout(DescLayout::Global).inner,
                descriptors.get_layout(DescLayout::View).inner,
            ],
            2,
        )?;

        let pipeline_light = Pipeline::new(
            device.clone(),
            swap_chain.extent,
            &render_pass_light,
            &light_stages,
            &[
                descriptors.get_layout(DescLayout::Global).inner,
                descriptors.get_layout(DescLayout::View).inner,
                descriptors.get_layout(DescLayout::GBuffer).inner,
            ],
            1,
        )?;

        let pipeline_tonemap = Pipeline::new(
            device.clone(),
            swap_chain.extent,
            &render_pass_tonemap,
            &tonemap_stages,
            &[
                descriptors.get_layout(DescLayout::Global).inner,
                descriptors.get_layout(DescLayout::View).inner,
                descriptors.get_layout(DescLayout::PostProcess).inner,
            ],
            1,
        )?;

        let extent_3d = Extent3D {
            width: swap_chain.extent.width,
            height: swap_chain.extent.height,
            depth: 1,
        };

        let swap_chain_fbs = swap_chain_image_views
            .iter()
            .map(|iw| Framebuffer::new(device.clone(), &render_pass_tonemap, swap_chain.extent, &[iw]))
            .collect::<Result<Vec<_>, _>>()?;

        let gbuffer = GBuffer::new(device.clone(), extent_3d, &render_pass_gb)?;
        let tonemap = Tonemap::new(device.clone(), extent_3d, &render_pass_light)?;

        let descriptor_pool = DescriptorPool::new(
            device.clone(),
            &[
                vk::DescriptorPoolSize {
                    descriptor_count: 4 * MAX_FRAMES_IN_FLIGHT as u32,
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
            4 * MAX_FRAMES_IN_FLIGHT as u32,
        )?;

        descriptors.allocate_sets(&descriptor_pool, MAX_FRAMES_IN_FLIGHT as u32)?;

        let command_pool = CommandPool::new(device.clone())?;
        let command_buffers = command_pool.allocate_cmd_buffers(MAX_FRAMES_IN_FLIGHT as u32)?;

        let mut img_available = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut render_finished = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut in_flight = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut uniform_buffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut uniform_buffers_globals = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        let (fs_quad_v, fs_quad_i) = crate::mesh::fs_triangle();
        let fs_quad = VulkanMesh::new(
            device.clone(),
            &command_pool,
            &MeshResource::new(fs_quad_v.to_vec(), crate::mesh::Indices::U32(fs_quad_i.to_vec())),
        )?;

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
            )?;

            let buffer_info = vk::DescriptorBufferInfo {
                buffer: uniform_buffer.inner,
                offset: 0,
                range: std::mem::size_of::<ViewProj>() as u64,
            };

            uniform_buffers.push(uniform_buffer);

            let desc_write = vk::WriteDescriptorSet {
                dst_set: descriptors.get_sets(DescLayout::View)[i].inner,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                p_buffer_info: &buffer_info,
                ..Default::default()
            };

            let uniform_buffer_global = Buffer::new(
                device.clone(),
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                std::mem::size_of::<Globals>() as u64,
                true,
            )?;

            let buffer_info = vk::DescriptorBufferInfo {
                buffer: uniform_buffer_global.inner,
                offset: 0,
                range: std::mem::size_of::<f32>() as u64,
            };

            uniform_buffers_globals.push(uniform_buffer_global);

            let desc_write_global = vk::WriteDescriptorSet {
                dst_set: descriptors.get_sets(DescLayout::Global)[i].inner,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                p_buffer_info: &buffer_info,
                ..Default::default()
            };

            let gb_image_infos = [
                vk::DescriptorImageInfo {
                    sampler: gbuffer.sampler.inner,
                    image_view: gbuffer.views[0].inner,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                },
                vk::DescriptorImageInfo {
                    sampler: gbuffer.sampler.inner,
                    image_view: gbuffer.views[1].inner,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                },
                vk::DescriptorImageInfo {
                    sampler: gbuffer.sampler.inner,
                    image_view: gbuffer.views[2].inner,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                },
            ];

            let tonemap_image_infos = [vk::DescriptorImageInfo {
                sampler: tonemap.sampler.inner,
                image_view: tonemap.view.inner,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            }];

            let desc_write_gb = vk::WriteDescriptorSet {
                dst_set: descriptors.get_sets(DescLayout::GBuffer)[i].inner,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: gb_image_infos.len() as u32,
                p_image_info: gb_image_infos.as_ptr(),
                ..Default::default()
            };

            let desc_write_tonemap = vk::WriteDescriptorSet {
                dst_set: descriptors.get_sets(DescLayout::PostProcess)[i].inner,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: tonemap_image_infos.len() as u32,
                p_image_info: tonemap_image_infos.as_ptr(),
                ..Default::default()
            };

            unsafe {
                device
                    .inner
                    .update_descriptor_sets(&[desc_write_global, desc_write, desc_write_gb, desc_write_tonemap], &[]);
            }
        }

        Ok(Self {
            instance,
            device,
            surface,
            swap_chain,
            swap_chain_image_views,
            swap_chain_fbs,
            gbuffer,
            tonemap,
            pipeline_gb,
            pipeline_light,
            pipeline_tonemap,
            render_pass_gb,
            render_pass_light,
            render_pass_tonemap,
            fs_quad,
            command_pool,
            descriptor_pool,
            command_buffers,
            img_available,
            render_finished,
            descriptors,
            uniform_buffers,
            uniform_buffers_globals,
            in_flight,
            meshes: HashMap::new(),
            current_frame: 0,
            max_frames_in_flight: MAX_FRAMES_IN_FLIGHT,
        })
    }

    pub fn render_frame(
        &mut self,
        scene: &Scene,
        drawable_size: (u32, u32),
        context: &FrameContext,
    ) -> Result<(), VulkanError> {
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
        let fov_rad = scene.camera.fov / (180.0 / std::f32::consts::PI);
        let fovy = 2.0 * ((fov_rad / 2.0).tan() * (1.0 / aspect_ratio)).atan();

        let mut proj = nalgebra_glm::perspective_rh_zo(aspect_ratio, fovy, 0.01, 5000.0);

        // flip because of vulkan
        proj.m22 *= -1.0;

        let ubo = ViewProj {
            view: scene.camera.view(),
            projection: proj,
        };

        let globals = Globals {
            max_bright: 10.0,
            res_x: drawable_size.0 as f32,
            res_y: drawable_size.1 as f32,
        };

        unsafe {
            self.uniform_buffers[self.current_frame].fill_host(std::slice::from_raw_parts(
                std::ptr::addr_of!(ubo) as *const u8,
                std::mem::size_of::<ViewProj>(),
            ))?;

            self.uniform_buffers_globals[self.current_frame].fill_host(globals.to_boxed_slice().as_ref())?;
        }

        self.record_command_buffer(command_buffer, framebuffer, scene, &context.total_time)?;

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

        self.device.wait_idle()?;

        if is_suboptimal {
            self.resize(drawable_size)?;
        }

        self.current_frame = (self.current_frame + 1) % self.max_frames_in_flight;

        Ok(())
    }

    pub fn resize(&mut self, drawable_size: (u32, u32)) -> Result<(), VulkanError> {
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

        self.gbuffer.resize(extent_3d, &self.render_pass_gb)?;
        self.tonemap.resize(extent_3d, &self.render_pass_light)?;

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let gb_image_infos = [
                vk::DescriptorImageInfo {
                    sampler: self.gbuffer.sampler.inner,
                    image_view: self.gbuffer.views[0].inner,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                },
                vk::DescriptorImageInfo {
                    sampler: self.gbuffer.sampler.inner,
                    image_view: self.gbuffer.views[1].inner,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                },
                vk::DescriptorImageInfo {
                    sampler: self.gbuffer.sampler.inner,
                    image_view: self.gbuffer.views[2].inner,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                },
            ];

            let tonemap_image_infos = [vk::DescriptorImageInfo {
                sampler: self.tonemap.sampler.inner,
                image_view: self.tonemap.view.inner,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            }];

            let desc_write_gb = vk::WriteDescriptorSet {
                dst_set: self.descriptors.get_sets(DescLayout::GBuffer)[i].inner,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: gb_image_infos.len() as u32,
                p_image_info: gb_image_infos.as_ptr(),
                ..Default::default()
            };

            let desc_write_tonemap = vk::WriteDescriptorSet {
                dst_set: self.descriptors.get_sets(DescLayout::PostProcess)[i].inner,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: tonemap_image_infos.len() as u32,
                p_image_info: tonemap_image_infos.as_ptr(),
                ..Default::default()
            };

            unsafe {
                self.device
                    .inner
                    .update_descriptor_sets(&[desc_write_gb, desc_write_tonemap], &[]);
            }
        }

        self.swap_chain_fbs = self
            .swap_chain_image_views
            .iter()
            .map(|iw| {
                Framebuffer::new(
                    self.device.clone(),
                    &self.render_pass_tonemap,
                    self.swap_chain.extent,
                    &[iw],
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        self.pipeline_gb.viewport.width = self.swap_chain.extent.width as f32;
        self.pipeline_gb.viewport.height = self.swap_chain.extent.height as f32;
        self.pipeline_gb.scissor.extent = self.swap_chain.extent;

        self.pipeline_light.viewport.width = self.swap_chain.extent.width as f32;
        self.pipeline_light.viewport.height = self.swap_chain.extent.height as f32;
        self.pipeline_light.scissor.extent = self.swap_chain.extent;

        self.pipeline_tonemap.viewport.width = self.swap_chain.extent.width as f32;
        self.pipeline_tonemap.viewport.height = self.swap_chain.extent.height as f32;
        self.pipeline_tonemap.scissor.extent = self.swap_chain.extent;

        Ok(())
    }

    fn record_command_buffer(
        &self,
        command_buffer: &CommandBuffer,
        framebuffer: &Framebuffer,
        scene: &Scene,
        push_constants: &f32,
    ) -> Result<(), VulkanError> {
        command_buffer.begin()?;

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
            render_pass: self.render_pass_gb.inner,
            framebuffer: self.gbuffer.framebuffer.inner,
            render_area: vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: self.swap_chain.extent,
            },
            clear_value_count: clears.len() as u32,
            p_clear_values: clears.as_ptr(),
            ..Default::default()
        };

        command_buffer.begin_render_pass(&render_pass_info, vk::SubpassContents::INLINE);

        command_buffer.bind_pipeline(&self.pipeline_gb, vk::PipelineBindPoint::GRAPHICS);
        command_buffer.set_viewport(self.pipeline_gb.viewport);
        command_buffer.set_scissor(self.pipeline_gb.scissor);

        unsafe {
            self.device.inner.cmd_bind_descriptor_sets(
                command_buffer.inner,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_gb.layout,
                0,
                &[
                    self.descriptors.get_sets(DescLayout::Global)[self.current_frame].inner,
                    self.descriptors.get_sets(DescLayout::View)[self.current_frame].inner,
                ],
                &[],
            );
        }

        for instance in &scene.meshes {
            let mesh = &instance.resource;
            let mesh_data = &self.meshes[&mesh.id];

            command_buffer.bind_vertex_buffers(&[&mesh_data.buf], &[0]);

            unsafe {
                let mut constants = [0_u8; 68];

                constants[0..64]
                    .copy_from_slice(std::slice::from_raw_parts(instance.transform.as_ptr() as *const u8, 64));
                constants[64..].copy_from_slice(&(push_constants).to_le_bytes());

                self.device.inner.cmd_push_constants(
                    command_buffer.inner,
                    self.pipeline_gb.layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    &constants,
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

        let render_pass_info = vk::RenderPassBeginInfo {
            render_pass: self.render_pass_light.inner,
            framebuffer: self.tonemap.framebuffer.inner,
            render_area: vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: self.swap_chain.extent,
            },
            clear_value_count: 0,
            ..Default::default()
        };

        command_buffer.begin_render_pass(&render_pass_info, vk::SubpassContents::INLINE);

        command_buffer.bind_pipeline(&self.pipeline_light, vk::PipelineBindPoint::GRAPHICS);

        command_buffer.bind_vertex_buffers(&[&self.fs_quad.buf], &[0]);

        unsafe {
            let mut constants = [0_u8; 68];
            constants[64..].copy_from_slice(&(push_constants).to_le_bytes());

            self.device.inner.cmd_push_constants(
                command_buffer.inner,
                self.pipeline_light.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                &constants,
            );

            self.device.inner.cmd_bind_index_buffer(
                command_buffer.inner,
                self.fs_quad.buf.inner.inner,
                self.fs_quad.indices_offset,
                vk::IndexType::UINT32,
            );

            self.device.inner.cmd_bind_descriptor_sets(
                command_buffer.inner,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_light.layout,
                2,
                &[self.descriptors.get_sets(DescLayout::GBuffer)[self.current_frame].inner],
                &[],
            );

            self.device
                .inner
                .cmd_draw_indexed(command_buffer.inner, self.fs_quad.index_count as u32, 1, 0, 0, 0);
        }

        command_buffer.end_render_pass();

        let render_pass_info = vk::RenderPassBeginInfo {
            render_pass: self.render_pass_tonemap.inner,
            framebuffer: framebuffer.inner,
            render_area: vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: self.swap_chain.extent,
            },
            clear_value_count: 0,
            ..Default::default()
        };

        command_buffer.begin_render_pass(&render_pass_info, vk::SubpassContents::INLINE);

        command_buffer.bind_pipeline(&self.pipeline_tonemap, vk::PipelineBindPoint::GRAPHICS);

        command_buffer.bind_vertex_buffers(&[&self.fs_quad.buf], &[0]);

        unsafe {
            let mut constants = [0_u8; 68];
            constants[64..].copy_from_slice(&(push_constants).to_le_bytes());

            self.device.inner.cmd_push_constants(
                command_buffer.inner,
                self.pipeline_tonemap.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                &constants,
            );

            self.device.inner.cmd_bind_index_buffer(
                command_buffer.inner,
                self.fs_quad.buf.inner.inner,
                self.fs_quad.indices_offset,
                vk::IndexType::UINT32,
            );

            self.device.inner.cmd_bind_descriptor_sets(
                command_buffer.inner,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_tonemap.layout,
                2,
                &[self.descriptors.get_sets(DescLayout::PostProcess)[self.current_frame].inner],
                &[],
            );

            self.device
                .inner
                .cmd_draw_indexed(command_buffer.inner, self.fs_quad.index_count as u32, 1, 0, 0, 0);
        }

        command_buffer.end_render_pass();

        command_buffer.end()
    }

    fn prepare_meshes(&mut self, scene: &Scene) -> Result<(), VulkanError> {
        for instance in &scene.meshes {
            if let std::collections::hash_map::Entry::Vacant(e) = self.meshes.entry(instance.resource.id) {
                let mesh = VulkanMesh::new(self.device.clone(), &self.command_pool, &instance.resource)?;
                e.insert(mesh);
            }
        }
        Ok(())
    }
}

pub struct FrameContext {
    pub delta_time: f32,
    pub total_time: f32,
}

pub struct ViewProj {
    pub view: nalgebra_glm::Mat4,
    pub projection: nalgebra_glm::Mat4,
}

pub struct Globals {
    pub max_bright: f32,
    pub res_x: f32,
    pub res_y: f32,
}

impl Globals {
    pub fn to_boxed_slice(&self) -> Box<[u8]> {
        let mut vec = Vec::with_capacity(std::mem::size_of::<f32>() * 3);

        vec.extend(self.max_bright.to_ne_bytes());
        vec.extend(self.res_x.to_ne_bytes());
        vec.extend(self.res_y.to_ne_bytes());

        vec.into_boxed_slice()
    }
}

pub struct GBuffer {
    pub framebuffer: Framebuffer,
    pub images: Vec<Image>,
    pub views: Vec<ImageView>,
    pub sampler: Sampler,
    device: Rc<Device>,
}

impl GBuffer {
    pub fn attrs() -> [(vk::Format, vk::ImageUsageFlags, vk::ImageAspectFlags); 3] {
        [
            (
                vk::Format::A2B10G10R10_UNORM_PACK32,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                vk::ImageAspectFlags::COLOR,
            ),
            (
                vk::Format::A2B10G10R10_UNORM_PACK32,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                vk::ImageAspectFlags::COLOR,
            ),
            (
                vk::Format::D32_SFLOAT,
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                vk::ImageAspectFlags::DEPTH,
            ),
        ]
    }

    pub fn new(device: Rc<Device>, extent: Extent3D, render_pass: &RenderPass) -> Result<Self, VulkanError> {
        let images = Self::attrs()
            .into_iter()
            .map(|(format, usage, _)| Image::new(device.clone(), format, extent, usage))
            .collect::<Result<Vec<_>, _>>()?;

        let views = images
            .iter()
            .zip(Self::attrs().into_iter())
            .map(|(image, (format, _, aspect))| ImageView::new(device.clone(), image.inner, format, aspect))
            .collect::<Result<Vec<_>, _>>()?;

        let framebuffer = Framebuffer::new(
            device.clone(),
            render_pass,
            Extent2D {
                width: extent.width,
                height: extent.height,
            },
            &[&views[0], &views[1], &views[2]],
        )?;

        let sampler = Sampler::new(device.clone())?;

        Ok(Self {
            images,
            views,
            framebuffer,
            sampler,
            device,
        })
    }

    pub fn resize(&mut self, extent: Extent3D, render_pass: &RenderPass) -> Result<(), VulkanError> {
        let attrs = Self::attrs();

        for (idx, image) in &mut self.images.iter_mut().enumerate() {
            *image = Image::new(self.device.clone(), attrs[idx].0, extent, attrs[idx].1)?;
        }

        for (idx, image) in &mut self.views.iter_mut().enumerate() {
            *image = ImageView::new(self.device.clone(), self.images[idx].inner, attrs[idx].0, attrs[idx].2)?;
        }

        self.framebuffer = Framebuffer::new(
            self.device.clone(),
            render_pass,
            Extent2D {
                width: extent.width,
                height: extent.height,
            },
            &[&self.views[0], &self.views[1], &self.views[2]],
        )?;

        Ok(())
    }
}

pub struct Tonemap {
    pub framebuffer: Framebuffer,
    pub image: Image,
    pub view: ImageView,
    pub sampler: Sampler,
    device: Rc<Device>,
}

impl Tonemap {
    pub fn attrs() -> (vk::Format, vk::ImageUsageFlags, vk::ImageAspectFlags) {
        (
            vk::Format::A2B10G10R10_UNORM_PACK32,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
        )
    }

    pub fn new(device: Rc<Device>, extent: Extent3D, render_pass: &RenderPass) -> Result<Self, VulkanError> {
        let (format, usage, aspect) = Self::attrs();

        let image = Image::new(device.clone(), format, extent, usage)?;

        let view = ImageView::new(device.clone(), image.inner, format, aspect)?;

        let framebuffer = Framebuffer::new(
            device.clone(),
            render_pass,
            Extent2D {
                width: extent.width,
                height: extent.height,
            },
            &[&view],
        )?;

        let sampler = Sampler::new(device.clone())?;

        Ok(Self {
            image,
            view,
            framebuffer,
            sampler,
            device,
        })
    }

    pub fn resize(&mut self, extent: Extent3D, render_pass: &RenderPass) -> Result<(), VulkanError> {
        let attrs = Self::attrs();

        self.image = Image::new(self.device.clone(), attrs.0, extent, attrs.1)?;

        self.view = ImageView::new(self.device.clone(), self.image.inner, attrs.0, attrs.2)?;

        self.framebuffer = Framebuffer::new(
            self.device.clone(),
            render_pass,
            Extent2D {
                width: extent.width,
                height: extent.height,
            },
            &[&self.view],
        )?;

        Ok(())
    }
}
