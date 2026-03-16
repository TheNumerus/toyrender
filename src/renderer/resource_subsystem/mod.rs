use crate::err::AppError;
use crate::renderer::{TlasIndex, VulkanContext};
use crate::scene::{Scene, SkyVariant};
use crate::vulkan::{
    BottomLevelAs, Buffer, CommandBuffer, DebugMarker, Image, ImageView, IntoVulkanError, Sampler, Vertex, VulkanMesh,
};
use ash::vk;
use gpu_allocator::MemoryLocation;
use image::EncodableLayout;
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

pub struct ResourceSubsystem {
    pub blases: BTreeMap<u64, BottomLevelAs>,
    pub meshes: BTreeMap<u64, VulkanMesh>,
    pub textures: BTreeMap<uuid::Uuid, (Image, ImageView, Rc<Sampler>)>,
    context: Rc<VulkanContext>,
}

impl ResourceSubsystem {
    pub fn new(context: Rc<VulkanContext>) -> Self {
        Self {
            blases: BTreeMap::new(),
            meshes: BTreeMap::new(),
            textures: BTreeMap::new(),
            context,
        }
    }

    pub fn prepare_resources(&mut self, scene: &Scene, command_buffer: &CommandBuffer) -> Result<bool, AppError> {
        let mut changed = false;

        for instance in &scene.meshes {
            if let Entry::Vacant(e) = self.meshes.entry(instance.resource.id) {
                // only run command if needed
                if !changed {
                    command_buffer.reset()?;
                    command_buffer.begin_one_time()?;
                    changed = true;
                }

                let mesh = VulkanMesh::new_nonblocking(
                    self.context.device.clone(),
                    self.context.allocator.clone(),
                    command_buffer,
                    &instance.resource,
                )?;
                e.insert(mesh);
            }
        }

        let mut src_buf;

        if let SkyVariant::Textured(ir, _) = &scene.env.sky.variant
            && let Entry::Vacant(e) = self.textures.entry(ir.id)
        {
            // only run command if needed
            if !changed {
                command_buffer.reset()?;
                command_buffer.begin_one_time()?;
                changed = true;
            }

            let extent = vk::Extent3D {
                width: ir.data.width(),
                height: ir.data.height(),
                depth: 1,
            };

            let image = Image::new(
                self.context.device.clone(),
                self.context.allocator.clone(),
                vk::Format::R32G32B32A32_SFLOAT,
                extent,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            )?;
            image.name(format!("Sky image {}", ir.id))?;

            let view = ImageView::new(
                self.context.device.clone(),
                image.inner,
                vk::Format::R32G32B32A32_SFLOAT,
                vk::ImageAspectFlags::COLOR,
            )?;
            view.name(format!("Sky imageview {}", ir.id))?;

            unsafe {
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
                        image: image.inner,
                        subresource_range: crate::vulkan::Image::single_color_layer_range(),
                        ..Default::default()
                    }],
                );
            }

            src_buf = Buffer::new(
                self.context.device.clone(),
                self.context.allocator.clone(),
                MemoryLocation::CpuToGpu,
                vk::BufferUsageFlags::TRANSFER_SRC,
                ir.data.width() as u64 * ir.data.height() as u64 * 4 * size_of::<f32>() as u64,
            )?;

            src_buf.fill_host(ir.data.clone().into_rgba32f().as_bytes())?;

            command_buffer.copy_buffer_to_image(
                &src_buf,
                &image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_row_length: ir.data.width(),
                    buffer_image_height: ir.data.height(),
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_offset: Default::default(),
                    image_extent: extent,
                },
            );

            unsafe {
                self.context.device.inner.cmd_pipeline_barrier(
                    command_buffer.inner,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier {
                        src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                        dst_access_mask: vk::AccessFlags::SHADER_READ,
                        old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        image: image.inner,
                        subresource_range: crate::vulkan::Image::single_color_layer_range(),
                        ..Default::default()
                    }],
                );
            }

            e.insert((
                image,
                view,
                Rc::new(Sampler::new_repeat_x_only(self.context.device.clone())?),
            ));
        }

        if changed {
            command_buffer.end()?;

            let submit_info = vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: &command_buffer.inner,
                ..Default::default()
            };

            unsafe {
                self.context
                    .device
                    .inner
                    .queue_submit(self.context.device.compute_queue, &[submit_info], vk::Fence::null())
                    .map_to_err("Cannot submit queue")?;
                self.context
                    .device
                    .inner
                    .queue_wait_idle(self.context.device.compute_queue)
                    .map_to_err("Cannot wait idle")?;
            }

            for mesh in self.meshes.values_mut() {
                mesh.buf.finalize();
            }

            let mut geos = BTreeMap::new();
            let mut ranges = BTreeMap::new();

            let mut processed = BTreeSet::new();

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

            let batch = BottomLevelAs::batch_bottom_build(
                self.context.device.clone(),
                self.context.allocator.clone(),
                self.context.rt_acc_struct_ext.clone(),
                &self.context.compute_command_pool,
                ranges,
                geos,
                vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE
                    | vk::BuildAccelerationStructureFlagsKHR::ALLOW_DATA_ACCESS,
            )?;

            self.blases = batch;
        }

        Ok(changed)
    }

    pub(crate) fn build_tlas_index(&self, scene: &Scene) -> TlasIndex {
        let mut index = Vec::with_capacity(scene.meshes.len());

        for mesh in &scene.meshes {
            let transform = mesh.transform;
            let id = mesh.resource.id;

            if !mesh.visible {
                continue;
            }

            index.push((id, transform));
        }

        TlasIndex { index }
    }
}
