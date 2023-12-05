use crate::vulkan::{Buffer, CommandPool, Device, Fence, Instance, IntoVulkanError, RtPipeline, VulkanError};
use ash::extensions::khr::AccelerationStructure as AccelerationStructureLoader;
use ash::extensions::khr::RayTracingPipeline as RayTracingPipelineLoader;
use ash::vk;
use ash::vk::{AccelerationStructureKHR, Packed24_8};
use nalgebra_glm::Mat4;
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

pub struct RayTracingAs {
    pub loader: AccelerationStructureLoader,
}

impl RayTracingAs {
    pub fn new(instance: &Instance, device: &Device) -> Result<Self, VulkanError> {
        let loader = AccelerationStructureLoader::new(&instance.inner, &device.inner);

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

        let addr = buffer.get_device_addr().unwrap();
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

pub enum AsLevel {
    Top,
    Bottom,
}

impl From<AsLevel> for vk::AccelerationStructureTypeKHR {
    fn from(value: AsLevel) -> Self {
        match value {
            AsLevel::Top => Self::TOP_LEVEL,
            AsLevel::Bottom => Self::BOTTOM_LEVEL,
        }
    }
}

pub struct AccelerationStructure {
    pub inner: AccelerationStructureKHR,
    buf: Buffer,
    ray_tracing_as: Rc<RayTracingAs>,
}

impl AccelerationStructure {
    fn create(
        device: Rc<Device>,
        ray_tracing_as: Rc<RayTracingAs>,
        size: u64,
        level: AsLevel,
    ) -> Result<Self, VulkanError> {
        let buf = Buffer::new(
            device.clone(),
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            size,
            false,
            true,
        )?;

        let create_info = vk::AccelerationStructureCreateInfoKHR {
            create_flags: Default::default(),
            buffer: buf.inner,
            size,
            ty: level.into(),
            ..Default::default()
        };

        let inner = unsafe {
            ray_tracing_as
                .loader
                .create_acceleration_structure(&create_info, None)
                .map_to_err("Cannot create top level acceleration structure.")?
        };

        Ok(Self {
            inner,
            buf,
            ray_tracing_as,
        })
    }

    pub fn top_build(
        device: Rc<Device>,
        rt_acc_struct_ext: Rc<RayTracingAs>,
        cmd_pool: &CommandPool,
        blases: &Vec<Self>,
        transforms: &Vec<Mat4>,
        flags: vk::BuildAccelerationStructureFlagsKHR,
    ) -> Result<Self, VulkanError> {
        let mut instances = Vec::new();

        if blases.is_empty() {
            return Self::create(device.clone(), rt_acc_struct_ext.clone(), 256, AsLevel::Top);
        }

        for (idx, blas) in blases.iter().enumerate() {
            let transform = transforms[idx];

            let instance = vk::AccelerationStructureInstanceKHR {
                transform: vk::TransformMatrixKHR {
                    matrix: [
                        transform.m11,
                        transform.m12,
                        transform.m13,
                        transform.m14,
                        transform.m21,
                        transform.m22,
                        transform.m23,
                        transform.m24,
                        transform.m31,
                        transform.m32,
                        transform.m33,
                        transform.m34,
                    ],
                },
                instance_custom_index_and_mask: Packed24_8::new(0, 0xFF),
                instance_shader_binding_table_record_offset_and_flags: Packed24_8::new(0, 1),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: blas.buf.get_device_addr().unwrap(),
                },
            };

            instances.push(instance);
        }

        let cmd_buf = cmd_pool.allocate_cmd_buffers(1)?.pop().unwrap();

        let mem_size = instances.len() as u64 * std::mem::size_of::<vk::AccelerationStructureInstanceKHR>() as u64;

        let stag_buf = Buffer::new(
            device.clone(),
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            mem_size,
            false,
            false,
        )?;

        let buf = Buffer::new(
            device.clone(),
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            mem_size,
            false,
            true,
        )?;

        unsafe {
            cmd_buf.begin_one_time()?;

            stag_buf.fill_host(std::slice::from_raw_parts(
                instances.as_ptr() as *const u8,
                mem_size as usize,
            ))?;

            let region = vk::BufferCopy {
                size: mem_size,
                src_offset: 0,
                dst_offset: 0,
            };

            device
                .inner
                .cmd_copy_buffer(cmd_buf.inner, stag_buf.inner, buf.inner, &[region]);
        }

        let barrier = vk::MemoryBarrier {
            src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
            dst_access_mask: vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
            ..Default::default()
        };

        unsafe {
            device.inner.cmd_pipeline_barrier(
                cmd_buf.inner,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }

        let addr = buf.get_device_addr().unwrap();

        let instances_data = vk::AccelerationStructureGeometryInstancesDataKHR {
            data: vk::DeviceOrHostAddressConstKHR { device_address: addr },
            ..Default::default()
        };

        let acc_struct_geo = vk::AccelerationStructureGeometryKHR {
            geometry_type: vk::GeometryTypeKHR::INSTANCES,
            geometry: vk::AccelerationStructureGeometryDataKHR {
                instances: instances_data,
            },
            ..Default::default()
        };

        let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR {
            ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            flags,
            mode: vk::BuildAccelerationStructureModeKHR::BUILD,
            geometry_count: 1,
            p_geometries: &acc_struct_geo,
            src_acceleration_structure: vk::AccelerationStructureKHR::null(),
            ..Default::default()
        };

        let size_info = unsafe {
            rt_acc_struct_ext.loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[instances.len() as u32],
            )
        };

        let scr_buf = Buffer::new(
            device.clone(),
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            size_info.build_scratch_size,
            false,
            true,
        )?;

        let tlas = Self::create(
            device.clone(),
            rt_acc_struct_ext.clone(),
            size_info.acceleration_structure_size,
            AsLevel::Top,
        )?;

        build_info.dst_acceleration_structure = tlas.inner;
        build_info.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: scr_buf.get_device_addr().unwrap(),
        };

        let offset_info = vk::AccelerationStructureBuildRangeInfoKHR {
            primitive_count: instances.len() as u32,
            primitive_offset: 0,
            first_vertex: 0,
            transform_offset: 0,
        };

        let fence = Fence::new(device.clone())?;

        unsafe {
            fence.reset()?;

            rt_acc_struct_ext
                .loader
                .cmd_build_acceleration_structures(cmd_buf.inner, &[build_info], &[&[offset_info]]);

            cmd_buf.end()?;

            let submit_info = vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: &cmd_buf.inner,
                ..Default::default()
            };

            device
                .inner
                .queue_submit(device.compute_queue, &[submit_info], fence.inner)
                .map_to_err("Cannot submit queue")?;

            fence.wait()?;
        }

        Ok(tlas)
    }

    pub fn batch_bottom_build(
        device: Rc<Device>,
        rt_acc_struct_ext: Rc<RayTracingAs>,
        cmd_pool: &CommandPool,
        ranges: Vec<vk::AccelerationStructureBuildRangeInfoKHR>,
        geos: Vec<vk::AccelerationStructureGeometryKHR>,
        flags: vk::BuildAccelerationStructureFlagsKHR,
    ) -> Result<Vec<Self>, VulkanError> {
        let mut blases = Vec::new();

        let cmd_buf = cmd_pool.allocate_cmd_buffers(1)?.pop().unwrap();

        for (range, geo) in ranges.into_iter().zip(geos.into_iter()) {
            let mut build_info = vk::AccelerationStructureBuildGeometryInfoKHR {
                ty: vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                flags,
                mode: vk::BuildAccelerationStructureModeKHR::BUILD,
                src_acceleration_structure: vk::AccelerationStructureKHR::null(),
                geometry_count: 1,
                p_geometries: [geo].as_ptr(),
                ..Default::default()
            };

            let size_info = unsafe {
                rt_acc_struct_ext.loader.get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &build_info,
                    &[range.primitive_count],
                )
            };

            let buf = Buffer::new(
                device.clone(),
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                size_info.build_scratch_size,
                false,
                true,
            )?;

            let blas = Self::create(
                device.clone(),
                rt_acc_struct_ext.clone(),
                size_info.acceleration_structure_size,
                AsLevel::Bottom,
            )?;

            build_info.dst_acceleration_structure = blas.inner;
            build_info.scratch_data = vk::DeviceOrHostAddressKHR {
                device_address: buf.get_device_addr().unwrap(),
            };

            cmd_buf.begin_one_time()?;

            let fence = Fence::new(device.clone())?;

            unsafe {
                fence.reset()?;

                rt_acc_struct_ext
                    .loader
                    .cmd_build_acceleration_structures(cmd_buf.inner, &[build_info], &[&[range]]);

                cmd_buf.end()?;

                let submit_info = vk::SubmitInfo {
                    command_buffer_count: 1,
                    p_command_buffers: &cmd_buf.inner,
                    ..Default::default()
                };

                device
                    .inner
                    .queue_submit(device.compute_queue, &[submit_info], fence.inner)
                    .map_to_err("Cannot submit queue")?;

                fence.wait()?;
            }

            blases.push(blas);
        }

        Ok(blases)
    }
}

impl Drop for AccelerationStructure {
    fn drop(&mut self) {
        unsafe {
            self.ray_tracing_as
                .loader
                .destroy_acceleration_structure(self.inner, None);
        }
    }
}
