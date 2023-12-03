use crate::vulkan::{Buffer, Device, Instance, IntoVulkanError, RtPipeline, VulkanError};
use ash::extensions::khr::RayTracingPipeline as RayTracingPipelineLoader;
use ash::vk;
use std::rc::Rc;

pub struct RayTracingPipeline {
    pub loader: RayTracingPipelineLoader,
}

impl RayTracingPipeline {
    pub fn new(instance: &Instance, device: &Device) -> Result<Self, VulkanError> {
        let loader = RayTracingPipelineLoader::new(&instance.inner, &device.inner);

        Ok(Self { loader })
    }
}

pub struct ShaderBindingTable {
    pub buffer: Buffer,
    pub raygen_region: vk::StridedDeviceAddressRegionKHR,
    pub miss_region: vk::StridedDeviceAddressRegionKHR,
    pub hit_region: vk::StridedDeviceAddressRegionKHR,
    pub call_region: vk::StridedDeviceAddressRegionKHR,
}

impl ShaderBindingTable {
    pub fn new(
        device: Rc<Device>,
        rt_pipeline: &RayTracingPipeline,
        pipeline: &RtPipeline,
        miss_count: usize,
        hit_count: usize,
    ) -> Result<Self, VulkanError> {
        let align_up = |size: u32, alignment: u32| (size + (alignment - 1)) & !(alignment - 1);

        let handle_count = 1 + miss_count + hit_count;
        let handle_size_aligned = align_up(
            device.rt_properties.shader_group_handle_size,
            device.rt_properties.shader_group_handle_alignment,
        );

        let mut raygen_region = vk::StridedDeviceAddressRegionKHR {
            stride: align_up(handle_size_aligned, device.rt_properties.shader_group_base_alignment) as u64,
            size: align_up(handle_size_aligned, device.rt_properties.shader_group_base_alignment) as u64,
            ..Default::default()
        };

        let mut miss_region = vk::StridedDeviceAddressRegionKHR {
            stride: handle_size_aligned as u64,
            size: align_up(
                handle_size_aligned * miss_count as u32,
                device.rt_properties.shader_group_base_alignment,
            ) as u64,
            ..Default::default()
        };

        let mut hit_region = vk::StridedDeviceAddressRegionKHR {
            stride: handle_size_aligned as u64,
            size: align_up(
                handle_size_aligned * hit_count as u32,
                device.rt_properties.shader_group_base_alignment,
            ) as u64,
            ..Default::default()
        };

        let call_region = vk::StridedDeviceAddressRegionKHR { ..Default::default() };

        let size = handle_count * device.rt_properties.shader_group_handle_size as usize;

        let handles = unsafe {
            rt_pipeline
                .loader
                .get_ray_tracing_shader_group_handles(pipeline.inner, 0, handle_count as u32, size)
                .map_to_err("Cannot get RT shader group handles")?
        };

        let buf_size = raygen_region.size + miss_region.size + call_region.size + hit_region.size;

        let buffer = Buffer::new(
            device.clone(),
            vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            buf_size,
            false,
            true,
        )?;

        let addr_info = vk::BufferDeviceAddressInfo {
            buffer: buffer.inner,
            ..Default::default()
        };

        let addr = unsafe { device.inner.get_buffer_device_address(&addr_info) };
        raygen_region.device_address = addr;
        miss_region.device_address = addr + raygen_region.size;
        hit_region.device_address = addr + raygen_region.size + miss_region.size;

        let get_handle = |i: usize| unsafe {
            handles
                .as_ptr()
                .add(i * device.rt_properties.shader_group_handle_size as usize)
        };

        let mut data = vec![0; buf_size as usize];
        let mut ptr = data.as_mut_ptr();
        let mut handle_idx = 0_usize;

        unsafe {
            std::ptr::copy_nonoverlapping(
                get_handle(handle_idx),
                ptr,
                device.rt_properties.shader_group_handle_size as usize,
            );
            handle_idx += 1;

            ptr = data.as_mut_ptr().add(raygen_region.size as usize);
            for _ in 0..miss_count {
                std::ptr::copy_nonoverlapping(
                    get_handle(handle_idx),
                    ptr,
                    device.rt_properties.shader_group_handle_size as usize,
                );
                handle_idx += 1;
                ptr = ptr.add(miss_region.stride as usize);
            }

            ptr = data
                .as_mut_ptr()
                .add(raygen_region.size as usize + miss_region.size as usize);
            for _ in 0..hit_count {
                std::ptr::copy_nonoverlapping(
                    get_handle(handle_idx),
                    ptr,
                    device.rt_properties.shader_group_handle_size as usize,
                );
                handle_idx += 1;
                ptr = ptr.add(hit_region.stride as usize);
            }

            buffer.fill_host(&data)?;
        }

        Ok(Self {
            buffer,
            raygen_region,
            miss_region,
            hit_region,
            call_region,
        })
    }
}
