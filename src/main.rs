use ash::{vk, Entry};
use sdl2::event::{Event, WindowEvent};
use std::ffi::CString;

mod app;
mod err;
mod vulkan;

use err::AppError;

const MAX_FRAMES_IN_FLIGHT: usize = 2;

fn main() -> Result<(), AppError> {
    let mut app = app::App::create();

    let entry = unsafe { Entry::load().expect("cannot load vulkan entry") };

    let instance = vulkan::Instance::new(&entry, &app.window.vulkan_instance_extensions().unwrap())?;
    let surface = vulkan::Surface::new(&instance, &entry, app.create_vulkan_surface(&instance)?)?;

    let devices = match vulkan::Device::query_applicable(&instance, &surface) {
        vulkan::DeviceQueryResult::NoDevice => Err(AppError::Other("no GPUs with Vulkan support found".into())),
        vulkan::DeviceQueryResult::NoApplicableDevice => {
            Err(AppError::Other("No suitable physical device found".into()))
        }
        vulkan::DeviceQueryResult::ApplicableDevices(d) => Ok(d),
        vulkan::DeviceQueryResult::VulkanError(e) => Err(e.into()),
    }?;

    let device = std::rc::Rc::new(vulkan::Device::new(&instance, devices[0], &surface)?);

    let mut swapchain = vulkan::SwapChain::new(device.clone(), &instance, &app, &surface)?;
    let mut swapchain_image_views = swapchain.create_image_views()?;

    let vert_module = vulkan::ShaderModule::new(include_bytes!("../build/vert.spv"), device.clone())?;
    let frag_module = vulkan::ShaderModule::new(include_bytes!("../build/frag.spv"), device.clone())?;

    let shader_entry = CString::new("main").unwrap();

    let vert_stage_info = vk::PipelineShaderStageCreateInfo {
        stage: vk::ShaderStageFlags::VERTEX,
        module: vert_module.inner,
        p_name: shader_entry.as_ptr(),
        ..Default::default()
    };

    let frag_stage_info = vk::PipelineShaderStageCreateInfo {
        stage: vk::ShaderStageFlags::FRAGMENT,
        module: frag_module.inner,
        p_name: shader_entry.as_ptr(),
        ..Default::default()
    };

    let stages = [vert_stage_info, frag_stage_info];

    let render_pass = vulkan::RenderPass::new(device.clone(), swapchain.format.format)?;

    let mut pipeline = vulkan::Pipeline::new(device.clone(), &swapchain, &render_pass, &stages)?;

    let mut swapchain_framebuffers = swapchain_image_views
        .iter()
        .map(|iw| vulkan::SwapChainFramebuffer::new(device.clone(), &render_pass, &swapchain, iw))
        .collect::<Result<Vec<_>, _>>()?;

    let command_pool = vulkan::CommandPool::new(device.clone())?;

    let alloc_info = vk::CommandBufferAllocateInfo {
        command_pool: command_pool.inner,
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: MAX_FRAMES_IN_FLIGHT as u32,
        ..Default::default()
    };

    let command_buffers = unsafe {
        device
            .inner
            .allocate_command_buffers(&alloc_info)
            .expect("cannot allocate command buffer")
    };

    let mut img_available = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
    let mut render_finished = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
    let mut in_flight = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        img_available.push(vulkan::Semaphore::new(device.clone())?);
        render_finished.push(vulkan::Semaphore::new(device.clone())?);
        in_flight.push(vulkan::Fence::new(device.clone())?);
    }

    let mut current_frame = 0;

    'running: loop {
        let mut resized = false;

        for event in app.event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => {
                    break 'running;
                }
                Event::Window { win_event, .. } => match win_event {
                    WindowEvent::Resized(_, _) => {
                        resized = true;
                    }
                    _ => {}
                },
                _ => {}
            }
        }

        in_flight[current_frame].wait()?;

        let (image_index, _is_suboptimal);

        unsafe {
            (image_index, _is_suboptimal) = swapchain
                .loader
                .acquire_next_image(
                    swapchain.swapchain,
                    u64::MAX,
                    img_available[current_frame].inner,
                    vk::Fence::null(),
                )
                .expect("cannot acquire image");

            in_flight[current_frame].reset()?;

            device
                .inner
                .reset_command_buffer(command_buffers[current_frame], vk::CommandBufferResetFlags::empty())
                .expect("cannot reset command buffer");
        }

        record_command_buffer(
            command_buffers[current_frame],
            image_index,
            &device,
            &render_pass,
            &swapchain_framebuffers,
            &swapchain,
            &pipeline,
        );

        let wait_semaphores = [img_available[current_frame].inner];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [render_finished[current_frame].inner];

        let submit_info = vk::SubmitInfo {
            wait_semaphore_count: 1,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: 1,
            p_command_buffers: &command_buffers[current_frame],
            signal_semaphore_count: 1,
            p_signal_semaphores: signal_semaphores.as_ptr(),
            ..Default::default()
        };

        unsafe {
            device
                .inner
                .queue_submit(device.graphics_queue, &[submit_info], in_flight[current_frame].inner)
                .expect("failed to submit to queue")
        };

        let present_info = vk::PresentInfoKHR {
            wait_semaphore_count: 1,
            p_wait_semaphores: signal_semaphores.as_ptr(),
            swapchain_count: 1,
            p_swapchains: &swapchain.swapchain,
            p_image_indices: &image_index,
            ..Default::default()
        };

        let is_suboptimal = unsafe {
            swapchain
                .loader
                .queue_present(device.present_queue, &present_info)
                .expect("cannot present")
        };

        device.wait_idle()?;

        if is_suboptimal || resized {
            drop(swapchain_framebuffers);
            drop(swapchain_image_views);
            drop(swapchain);

            swapchain = vulkan::SwapChain::new(device.clone(), &instance, &app, &surface)?;
            swapchain_image_views = swapchain.create_image_views()?;
            swapchain_framebuffers = swapchain_image_views
                .iter()
                .map(|iw| vulkan::SwapChainFramebuffer::new(device.clone(), &render_pass, &swapchain, iw))
                .collect::<Result<Vec<_>, _>>()?;

            pipeline.viewport.width = swapchain.extent.width as f32;
            pipeline.viewport.height = swapchain.extent.height as f32;
        }

        current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    Ok(())
}

fn record_command_buffer(
    command_buffer: vk::CommandBuffer,
    image_index: u32,
    device: &std::rc::Rc<vulkan::Device>,
    render_pass: &vulkan::RenderPass,
    swapchain_fbs: &[vulkan::SwapChainFramebuffer],
    swapchain: &vulkan::SwapChain,
    pipeline: &vulkan::Pipeline,
) {
    let begin_info = vk::CommandBufferBeginInfo::default();

    unsafe {
        device
            .inner
            .begin_command_buffer(command_buffer, &begin_info)
            .expect("cannot begin recording")
    };

    let clear_color = vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 1.0],
        },
    };

    let render_pass_info = vk::RenderPassBeginInfo {
        render_pass: render_pass.inner,
        framebuffer: swapchain_fbs[image_index as usize].inner,
        render_area: vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: swapchain.extent,
        },
        clear_value_count: 1,
        p_clear_values: &clear_color,
        ..Default::default()
    };

    unsafe {
        device
            .inner
            .cmd_begin_render_pass(command_buffer, &render_pass_info, vk::SubpassContents::INLINE);
        device
            .inner
            .cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline.inner);
        device.inner.cmd_set_viewport(command_buffer, 0, &[pipeline.viewport]);
        device.inner.cmd_set_scissor(command_buffer, 0, &[pipeline.scissor]);
        device.inner.cmd_draw(command_buffer, 3, 1, 0, 0);
        device.inner.cmd_end_render_pass(command_buffer);
        device
            .inner
            .end_command_buffer(command_buffer)
            .expect("cannot end command buffer");
    }
}
