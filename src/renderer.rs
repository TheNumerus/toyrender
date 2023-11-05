use crate::app::App;
use crate::err::AppError;
use crate::scene::Scene;
use crate::vulkan::{
    Buffer, CommandBuffer, CommandPool, DescriptorPool, DescriptorSet, Device, DeviceQueryResult, Fence, Image,
    ImageView, Instance, Pipeline, RenderPass, Semaphore, ShaderModule, ShaderStage, Surface, SwapChain,
    SwapChainFramebuffer, UniformDescriptorSetLayout, VulkanError,
};
use ash::{vk, Entry};
use std::rc::Rc;

const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct VulkanRenderer {
    pub instance: Rc<Instance>,
    pub device: Rc<Device>,
    pub swap_chain: SwapChain,
    pub surface: Surface,
    pub swap_chain_image_views: Vec<ImageView>,
    pub swap_chain_fbs: Vec<SwapChainFramebuffer>,
    pub depth_image: Image,
    pub depth_image_view: ImageView,
    pub pipeline: Pipeline,
    pub render_pass: RenderPass,
    pub command_pool: CommandPool,
    pub descriptor_pool: DescriptorPool,
    pub command_buffers: Vec<CommandBuffer>,
    pub descriptor_layouts: Vec<UniformDescriptorSetLayout>,
    pub descriptor_sets: Vec<DescriptorSet>,
    pub uniform_buffers: Vec<Buffer>,
    pub current_frame: usize,
    img_available: Vec<Semaphore>,
    render_finished: Vec<Semaphore>,
    in_flight: Vec<Fence>,
    max_frames_in_flight: usize,
}

impl VulkanRenderer {
    pub fn init(app: &App) -> Result<Self, AppError> {
        let entry = unsafe { Entry::load().expect("cannot load vulkan entry") };

        let instance = Rc::new(Instance::new(
            &entry,
            &app.window.vulkan_instance_extensions().unwrap(),
        )?);
        let surface = Surface::new(&instance, &entry, app.create_vulkan_surface(&instance)?)?;

        let devices = match Device::query_applicable(&instance, &surface)? {
            DeviceQueryResult::ApplicableDevices(d) => Ok(d),
            DeviceQueryResult::NoDevice => Err(AppError::Other("no GPUs with Vulkan support found".into())),
            DeviceQueryResult::NoApplicableDevice => Err(AppError::Other("No suitable physical device found".into())),
        }?;

        let device = Rc::new(Device::new(instance.clone(), devices[0], &surface)?);

        let swap_chain = SwapChain::new(device.clone(), &instance, app, &surface)?;
        let swap_chain_image_views = swap_chain.create_image_views()?;

        let vert_module = ShaderModule::new(
            include_bytes!("../build/triangle_vert.spv"),
            device.clone(),
            ShaderStage::Vertex,
            None,
        )?;
        let frag_module = ShaderModule::new(
            include_bytes!("../build/triangle_frag.spv"),
            device.clone(),
            ShaderStage::Fragment,
            None,
        )?;

        let stages = [vert_module.stage_info(), frag_module.stage_info()];

        let descriptor_layouts = vec![UniformDescriptorSetLayout::new(
            device.clone(),
            0,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        )?];

        let render_pass = RenderPass::new(device.clone(), swap_chain.format.format)?;

        let pipeline = Pipeline::new(
            device.clone(),
            swap_chain.extent,
            &render_pass,
            &stages,
            &descriptor_layouts,
        )?;

        let extent_3d = vk::Extent3D {
            width: swap_chain.extent.width,
            height: swap_chain.extent.height,
            depth: 1,
        };
        let depth_image = Image::new(
            device.clone(),
            vk::Format::D32_SFLOAT,
            extent_3d,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        )?;

        let depth_image_view = ImageView::new(
            device.clone(),
            depth_image.inner,
            vk::Format::D32_SFLOAT,
            vk::ImageAspectFlags::DEPTH,
        )?;

        let swap_chain_fbs = swap_chain_image_views
            .iter()
            .map(|iw| SwapChainFramebuffer::new(device.clone(), &render_pass, &swap_chain, &[iw, &depth_image_view]))
            .collect::<Result<Vec<_>, _>>()?;

        let descriptor_pool = DescriptorPool::new(device.clone(), MAX_FRAMES_IN_FLIGHT as u32)?;
        let descriptor_sets =
            descriptor_pool.allocate_sets(&[descriptor_layouts[0].inner, descriptor_layouts[0].inner])?;

        let command_pool = CommandPool::new(device.clone())?;
        let command_buffers = command_pool.allocate_cmd_buffers(MAX_FRAMES_IN_FLIGHT as u32)?;

        let mut img_available = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut render_finished = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut in_flight = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut uniform_buffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

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
                dst_set: descriptor_sets[i].inner,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                p_buffer_info: &buffer_info,
                ..Default::default()
            };

            unsafe {
                device.inner.update_descriptor_sets(&[desc_write], &[]);
            }
        }

        Ok(Self {
            instance,
            device,
            surface,
            swap_chain,
            swap_chain_image_views,
            swap_chain_fbs,
            depth_image,
            depth_image_view,
            pipeline,
            render_pass,
            command_pool,
            descriptor_pool,
            descriptor_sets,
            command_buffers,
            img_available,
            render_finished,
            descriptor_layouts,
            uniform_buffers,
            in_flight,
            current_frame: 0,
            max_frames_in_flight: MAX_FRAMES_IN_FLIGHT,
        })
    }

    pub fn render_frame(
        &mut self,
        app: &App,
        scene: &crate::scene::Scene,
        context: &FrameContext,
    ) -> Result<(), VulkanError> {
        self.in_flight[self.current_frame].wait()?;
        let (image_index, _is_suboptimal) = self
            .swap_chain
            .acquire_next_image(&self.img_available[self.current_frame])?;
        self.in_flight[self.current_frame].reset()?;
        self.command_buffers[self.current_frame].reset()?;

        let command_buffer = &self.command_buffers[self.current_frame];
        let framebuffer = &self.swap_chain_fbs[image_index as usize];
        let descriptor_set = &self.descriptor_sets[self.current_frame];

        let aspect_ratio = app.window.drawable_size();
        let aspect_ratio = aspect_ratio.0 as f32 / aspect_ratio.1 as f32;

        let mut proj = nalgebra_glm::perspective_rh_zo(
            aspect_ratio,
            (scene.camera.fov / aspect_ratio) / 180.0 * std::f32::consts::PI,
            0.01,
            5000.0,
        );

        proj.m22 *= -1.0;

        let ubo = ViewProj {
            view: scene.camera.view(),
            projection: proj,
        };

        unsafe {
            self.uniform_buffers[self.current_frame].fill_host(std::slice::from_raw_parts(
                std::ptr::addr_of!(ubo) as *const u8,
                std::mem::size_of::<ViewProj>(),
            ))?;
        }

        self.record_command_buffer(command_buffer, framebuffer, descriptor_set, scene, &context.total_time)?;

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
            self.resize(app)?;
        }

        self.current_frame = (self.current_frame + 1) % self.max_frames_in_flight;

        Ok(())
    }

    pub fn resize(&mut self, app: &App) -> Result<(), VulkanError> {
        // need to drop before creating new ones
        self.swap_chain_fbs.clear();
        self.swap_chain_image_views.clear();

        self.swap_chain.recreate(self.device.clone(), app, &self.surface)?;

        self.swap_chain_image_views = self.swap_chain.create_image_views()?;
        self.swap_chain_fbs = self
            .swap_chain_image_views
            .iter()
            .map(|iw| SwapChainFramebuffer::new(self.device.clone(), &self.render_pass, &self.swap_chain, &[iw]))
            .collect::<Result<Vec<_>, _>>()?;

        self.pipeline.viewport.width = self.swap_chain.extent.width as f32;
        self.pipeline.viewport.height = self.swap_chain.extent.height as f32;
        self.pipeline.scissor.extent = self.swap_chain.extent;

        Ok(())
    }

    fn record_command_buffer(
        &self,
        command_buffer: &CommandBuffer,
        framebuffer: &SwapChainFramebuffer,
        descriptor_set: &DescriptorSet,
        scene: &Scene,
        push_constants: &f32,
    ) -> Result<(), VulkanError> {
        command_buffer.begin()?;

        let clears = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
            },
        ];

        let render_pass_info = vk::RenderPassBeginInfo {
            render_pass: self.render_pass.inner,
            framebuffer: framebuffer.inner,
            render_area: vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: self.swap_chain.extent,
            },
            clear_value_count: clears.len() as u32,
            p_clear_values: clears.as_ptr(),
            ..Default::default()
        };

        command_buffer.begin_render_pass(&render_pass_info, vk::SubpassContents::INLINE);

        command_buffer.bind_pipeline(&self.pipeline, vk::PipelineBindPoint::GRAPHICS);
        command_buffer.set_viewport(self.pipeline.viewport);
        command_buffer.set_scissor(self.pipeline.scissor);

        for instance in &scene.meshes {
            let mesh = &instance.instance;

            command_buffer.bind_vertex_buffers(&[&mesh.buf], &[0]);

            unsafe {
                let mut constants = [0_u8; 68];

                constants[0..64]
                    .copy_from_slice(std::slice::from_raw_parts(instance.transform.as_ptr() as *const u8, 64));
                constants[64..].copy_from_slice(&(push_constants).to_le_bytes());

                self.device.inner.cmd_push_constants(
                    command_buffer.inner,
                    self.pipeline.layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    &constants,
                );

                self.device.inner.cmd_bind_index_buffer(
                    command_buffer.inner,
                    mesh.buf.inner.inner,
                    mesh.indices_offset,
                    vk::IndexType::UINT32,
                );

                self.device.inner.cmd_bind_descriptor_sets(
                    command_buffer.inner,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline.layout,
                    0,
                    &[descriptor_set.inner],
                    &[],
                );

                self.device
                    .inner
                    .cmd_draw_indexed(command_buffer.inner, mesh.index_count as u32, 1, 0, 0, 0);
            }
        }

        command_buffer.end_render_pass();
        command_buffer.end()
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
