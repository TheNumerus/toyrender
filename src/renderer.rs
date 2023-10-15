use crate::app::App;
use crate::err::AppError;
use crate::vulkan::{
    Buffer, CommandBuffer, CommandPool, DescriptorPool, DescriptorSet, Device, DeviceQueryResult, Fence, Instance,
    Pipeline, RenderPass, Semaphore, ShaderModule, ShaderStage, Surface, SwapChain, SwapChainFramebuffer,
    SwapChainImageView, UniformDescriptorSetLayout, VulkanError,
};
use ash::{vk, Entry};
use std::rc::Rc;

const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct VulkanRenderer {
    pub instance: Rc<Instance>,
    pub device: Rc<Device>,
    pub swap_chain: SwapChain,
    pub surface: Surface,
    pub swap_chain_image_views: Vec<SwapChainImageView>,
    pub swap_chain_fbs: Vec<SwapChainFramebuffer>,
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
            vk::ShaderStageFlags::VERTEX,
        )?];

        let render_pass = RenderPass::new(device.clone(), swap_chain.format.format)?;

        let pipeline = Pipeline::new(
            device.clone(),
            swap_chain.extent,
            &render_pass,
            &stages,
            &descriptor_layouts,
        )?;

        let swap_chain_fbs = swap_chain_image_views
            .iter()
            .map(|iw| SwapChainFramebuffer::new(device.clone(), &render_pass, &swap_chain, iw))
            .collect::<Result<Vec<_>, _>>()?;

        let descriptor_pool = DescriptorPool::new(device.clone(), MAX_FRAMES_IN_FLIGHT as u32)?;
        let descriptor_sets =
            descriptor_pool.allocate_sets(&vec![descriptor_layouts[0].inner, descriptor_layouts[0].inner])?;

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
                std::mem::size_of::<ModelViewProj>() as u64,
                true,
            )?;

            let buffer_info = vk::DescriptorBufferInfo {
                buffer: uniform_buffer.inner,
                offset: 0,
                range: std::mem::size_of::<ModelViewProj>() as u64,
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
        record_cmd_buffer: impl Fn(&Self, &CommandBuffer, &SwapChainFramebuffer, &DescriptorSet) -> Result<(), VulkanError>,
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

        let ubo = ModelViewProj {
            view: nalgebra_glm::Mat4::identity(),
            projection: nalgebra_glm::Mat4::identity(),
            model: nalgebra_glm::Mat4::new_scaling(1.8),
        };

        unsafe {
            self.uniform_buffers[self.current_frame].fill_host(std::slice::from_raw_parts(
                std::ptr::addr_of!(ubo) as *const u8,
                std::mem::size_of::<ModelViewProj>(),
            ))?;
        }

        record_cmd_buffer(self, command_buffer, framebuffer, descriptor_set)?;

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
            .map(|iw| SwapChainFramebuffer::new(self.device.clone(), &self.render_pass, &self.swap_chain, iw))
            .collect::<Result<Vec<_>, _>>()?;

        self.pipeline.viewport.width = self.swap_chain.extent.width as f32;
        self.pipeline.viewport.height = self.swap_chain.extent.height as f32;
        self.pipeline.scissor.extent = self.swap_chain.extent;

        Ok(())
    }
}

pub struct ModelViewProj {
    pub model: nalgebra_glm::Mat4,
    pub view: nalgebra_glm::Mat4,
    pub projection: nalgebra_glm::Mat4,
}
