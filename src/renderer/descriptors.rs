use crate::renderer::{Globals, ViewProj};
use crate::scene::Environment;
use crate::vulkan::{
    AccelerationStructure, Buffer, DescriptorPool, DescriptorSet, DescriptorSetLayout, Device, VulkanError,
};
use ash::vk;
use ash::vk::{ImageView, Sampler};
use std::collections::HashMap;
use std::ffi::c_void;
use std::rc::Rc;

pub struct RendererDescriptors {
    pub global_sets: Vec<GlobalSet>,
    pub image_sets: Vec<ImageSet>,
    pub layouts: HashMap<DescLayout, DescriptorSetLayout>,
}

impl RendererDescriptors {
    pub fn build(device: Rc<Device>, pool: &DescriptorPool, frames_in_flight: u32) -> Result<Self, VulkanError> {
        let mut layouts = HashMap::new();

        for layout in [DescLayout::Global, DescLayout::Images] {
            let l = DescriptorSetLayout::new(device.clone(), &layout.get_bindings())?;

            layouts.insert(layout, l);
        }

        let global_sets = pool
            .allocate_sets(frames_in_flight, layouts.get(&DescLayout::Global).unwrap().inner)?
            .into_iter()
            .map(|inner| GlobalSet { inner })
            .collect();

        let image_sets = pool
            .allocate_sets(frames_in_flight, layouts.get(&DescLayout::Images).unwrap().inner)?
            .into_iter()
            .map(|inner| ImageSet { inner })
            .collect::<Vec<_>>();

        Ok(Self {
            layouts,
            global_sets,
            image_sets,
        })
    }

    pub fn get_layout(&self, layout: DescLayout) -> &DescriptorSetLayout {
        self.layouts.get(&layout).expect("Incorrectly built layouts")
    }
}

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub enum DescLayout {
    Global,
    Images,
}

impl DescLayout {
    pub fn get_bindings(&self) -> Vec<vk::DescriptorSetLayoutBinding> {
        let uniform = |binding: u32| vk::DescriptorSetLayoutBinding {
            binding,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::VERTEX
                | vk::ShaderStageFlags::FRAGMENT
                | vk::ShaderStageFlags::RAYGEN_KHR
                | vk::ShaderStageFlags::COMPUTE,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            ..Default::default()
        };

        match self {
            DescLayout::Global => {
                vec![
                    uniform(0),
                    vk::DescriptorSetLayoutBinding {
                        binding: 1,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::VERTEX
                            | vk::ShaderStageFlags::FRAGMENT
                            | vk::ShaderStageFlags::RAYGEN_KHR
                            | vk::ShaderStageFlags::COMPUTE,
                        descriptor_type: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                        ..Default::default()
                    },
                    uniform(2),
                    uniform(3),
                ]
            }
            DescLayout::Images => {
                vec![
                    vk::DescriptorSetLayoutBinding {
                        binding: 0,
                        descriptor_count: 16,
                        stage_flags: vk::ShaderStageFlags::COMPUTE
                            | vk::ShaderStageFlags::FRAGMENT
                            | vk::ShaderStageFlags::RAYGEN_KHR,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        ..Default::default()
                    },
                    vk::DescriptorSetLayoutBinding {
                        binding: 1,
                        descriptor_count: 16,
                        stage_flags: vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::RAYGEN_KHR,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        ..Default::default()
                    },
                ]
            }
        }
    }
}

pub struct DescriptorWrite {
    write: vk::WriteDescriptorSet,
    buffer_info: Option<vk::DescriptorBufferInfo>,
    image_info: Option<Vec<vk::DescriptorImageInfo>>,
    tlases: Option<Vec<vk::AccelerationStructureKHR>>,
}

pub struct DescriptorWriter {}

impl DescriptorWriter {
    pub fn batch_write(device: &Device, writes: Vec<DescriptorWrite>) {
        let mut nexts = Vec::with_capacity(writes.len());

        let mapped_writes = writes
            .iter()
            .map(|w| {
                let p_buffer_info = match w.buffer_info {
                    Some(ref a) => std::ptr::addr_of!(*a),
                    None => std::ptr::null(),
                };

                let p_next = match w.tlases {
                    Some(ref a) => {
                        let p_next = vk::WriteDescriptorSetAccelerationStructureKHR {
                            acceleration_structure_count: a.len() as u32,
                            p_acceleration_structures: (*a).as_ptr(),
                            ..Default::default()
                        };

                        nexts.push(p_next);

                        unsafe { nexts.as_ptr().add(nexts.len() - 1) }
                    }
                    None => std::ptr::null(),
                };

                let descriptor_count = match w.image_info {
                    Some(ref a) => a.len() as u32,
                    None => w.write.descriptor_count,
                };

                let p_image_info = match w.image_info {
                    Some(ref a) => (*a).as_ptr(),
                    None => std::ptr::null(),
                };

                vk::WriteDescriptorSet {
                    p_buffer_info,
                    p_next: p_next as *const c_void,
                    descriptor_count,
                    p_image_info,
                    ..w.write
                }
            })
            .collect::<Vec<_>>();

        unsafe { device.inner.update_descriptor_sets(&mapped_writes, &[]) }

        drop(nexts);
    }
}

pub struct GlobalSet {
    pub inner: DescriptorSet,
}

impl GlobalSet {
    pub fn update_globals(&self, buffer: &Buffer) -> DescriptorWrite {
        create_buffer_update(buffer, std::mem::size_of::<Globals>() as u64, &self.inner, 0)
    }

    pub fn update_tlas(&self, tlas: &AccelerationStructure) -> DescriptorWrite {
        let write = vk::WriteDescriptorSet {
            dst_set: self.inner.inner,
            dst_binding: 1,
            dst_array_element: 0,
            descriptor_type: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
            descriptor_count: 1,
            ..Default::default()
        };

        DescriptorWrite {
            write,
            buffer_info: None,
            tlases: Some(vec![tlas.inner]),
            image_info: None,
        }
    }

    pub fn update_view(&self, buffer: &Buffer) -> DescriptorWrite {
        create_buffer_update(buffer, std::mem::size_of::<ViewProj>() as u64 * 2, &self.inner, 2)
    }

    pub fn update_env(&self, buffer: &Buffer) -> DescriptorWrite {
        create_buffer_update(buffer, std::mem::size_of::<Environment>() as u64, &self.inner, 3)
    }
}

pub struct ImageSet {
    pub inner: DescriptorSet,
}

impl ImageSet {
    pub fn update(&self, textures: Vec<(Sampler, ImageView)>, storage: Vec<ImageView>) -> Vec<DescriptorWrite> {
        let storage_infos = storage
            .iter()
            .map(|view| vk::DescriptorImageInfo {
                sampler: vk::Sampler::null(),
                image_view: *view,
                image_layout: vk::ImageLayout::GENERAL,
            })
            .collect::<Vec<_>>();

        let texture_infos = textures
            .into_iter()
            .map(|(sampler, view)| vk::DescriptorImageInfo {
                sampler,
                image_view: view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            })
            .collect::<Vec<_>>();

        if texture_infos.len() > DescLayout::Images.get_bindings()[0].descriptor_count as usize {
            panic!("INVALID DESCRIPTOR COUNT")
        }

        if storage_infos.len() > DescLayout::Images.get_bindings()[1].descriptor_count as usize {
            panic!("INVALID DESCRIPTOR COUNT")
        }

        let texture_write = vk::WriteDescriptorSet {
            dst_set: self.inner.inner,
            dst_binding: 0,
            dst_array_element: 0,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            ..Default::default()
        };

        let storage_write = vk::WriteDescriptorSet {
            dst_set: self.inner.inner,
            dst_binding: 1,
            dst_array_element: 0,
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            ..Default::default()
        };

        vec![
            DescriptorWrite {
                write: texture_write,
                buffer_info: None,
                image_info: Some(texture_infos),
                tlases: None,
            },
            DescriptorWrite {
                write: storage_write,
                buffer_info: None,
                image_info: Some(storage_infos),
                tlases: None,
            },
        ]
    }
}

fn create_buffer_update(buffer: &Buffer, range: u64, dst_set: &DescriptorSet, binding: u32) -> DescriptorWrite {
    let buffer_info = vk::DescriptorBufferInfo {
        buffer: buffer.inner,
        offset: 0,
        range,
    };

    let write = vk::WriteDescriptorSet {
        dst_set: dst_set.inner,
        dst_binding: binding,
        dst_array_element: 0,
        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count: 1,
        ..Default::default()
    };

    DescriptorWrite {
        write,
        buffer_info: Some(buffer_info),
        tlases: None,
        image_info: None,
    }
}
