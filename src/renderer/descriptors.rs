use crate::err::AppError;
use crate::renderer::render_target::RenderTargets;
use crate::vulkan::{DescriptorPool, DescriptorSet, DescriptorSetLayout, Device, VulkanError};
use ash::vk;
use std::collections::HashMap;
use std::ffi::c_void;
use std::rc::Rc;

pub struct RendererDescriptors {
    pub global_sets: Vec<DescriptorSet>,
    pub image_sets: Vec<DescriptorSet>,
    pub compute_sets: Vec<DescriptorSet>,
    pub layouts: HashMap<DescLayout, DescriptorSetLayout>,
    pub samplers: HashMap<String, usize>,
    pub storages: HashMap<String, usize>,
}

impl RendererDescriptors {
    pub fn build(device: Rc<Device>, pool: &DescriptorPool, frames_in_flight: u32) -> Result<Self, VulkanError> {
        let mut layouts = HashMap::new();

        for layout in [DescLayout::Global, DescLayout::Compute, DescLayout::Image] {
            let l = DescriptorSetLayout::new(device.clone(), &layout.get_bindings(), format!("{layout:?}"))?;

            layouts.insert(layout, l);
        }

        let global_sets = pool
            .allocate_sets(frames_in_flight, layouts.get(&DescLayout::Global).unwrap().inner)?
            .into_iter()
            .collect();

        let compute_sets = pool
            .allocate_sets(frames_in_flight, layouts.get(&DescLayout::Compute).unwrap().inner)?
            .into_iter()
            .collect::<Vec<_>>();

        let image_sets = pool
            .allocate_sets(frames_in_flight, layouts.get(&DescLayout::Image).unwrap().inner)?
            .into_iter()
            .collect::<Vec<_>>();

        Ok(Self {
            layouts,
            global_sets,
            image_sets,
            compute_sets,
            samplers: HashMap::new(),
            storages: HashMap::new(),
        })
    }

    pub fn get_layout(&self, layout: DescLayout) -> &DescriptorSetLayout {
        self.layouts.get(&layout).expect("Incorrectly built layouts")
    }

    pub fn update_resources(&mut self, targets: &RenderTargets) -> Result<Vec<DescriptorWrite>, AppError> {
        self.samplers = HashMap::new();
        self.storages = HashMap::new();

        let mut writes = Vec::new();

        for (name, item) in &targets.targets {
            if item.value.borrow().usage.contains(vk::ImageUsageFlags::STORAGE) {
                for set in &self.compute_sets {
                    let image_info = vec![item.value.borrow().descriptor_image_info(vk::ImageLayout::GENERAL)];

                    let write = vk::WriteDescriptorSet {
                        dst_set: set.inner,
                        dst_binding: 1,
                        dst_array_element: self.storages.len() as u32,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        ..Default::default()
                    };

                    writes.push(DescriptorWrite {
                        write,
                        buffer_info: None,
                        image_info: Some(image_info),
                        tlases: None,
                    })
                }
                self.storages.insert(name.clone(), self.storages.len());
            }

            if item.value.borrow().usage.contains(vk::ImageUsageFlags::SAMPLED) {
                for set in &self.compute_sets {
                    let image_info = vec![
                        item.value
                            .borrow()
                            .descriptor_image_info(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                    ];

                    let write = vk::WriteDescriptorSet {
                        dst_set: set.inner,
                        dst_binding: 0,
                        dst_array_element: self.samplers.len() as u32,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        ..Default::default()
                    };

                    writes.push(DescriptorWrite {
                        write,
                        buffer_info: None,
                        image_info: Some(image_info),
                        tlases: None,
                    })
                }
                for set in &self.image_sets {
                    let image_info = vec![
                        item.value
                            .borrow()
                            .descriptor_image_info(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                    ];

                    let write = vk::WriteDescriptorSet {
                        dst_set: set.inner,
                        dst_binding: 0,
                        dst_array_element: self.samplers.len() as u32,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        ..Default::default()
                    };

                    writes.push(DescriptorWrite {
                        write,
                        buffer_info: None,
                        image_info: Some(image_info),
                        tlases: None,
                    })
                }
                self.samplers.insert(name.clone(), self.samplers.len());
            }
        }

        if self.samplers.len() > DescLayout::Compute.get_bindings()[0].descriptor_count as usize {
            return Err(AppError::Other(format!(
                "not enough descriptor capacity for sampled images, got {}, expected {}",
                self.samplers.len(),
                DescLayout::Compute.get_bindings()[0].descriptor_count
            )));
        }

        if self.samplers.len() > DescLayout::Image.get_bindings()[0].descriptor_count as usize {
            return Err(AppError::Other(format!(
                "not enough descriptor capacity for sampled images, got {}, expected {}",
                self.samplers.len(),
                DescLayout::Compute.get_bindings()[0].descriptor_count
            )));
        }

        if self.storages.len() > DescLayout::Compute.get_bindings()[1].descriptor_count as usize {
            return Err(AppError::Other(format!(
                "not enough descriptor capacity for storage images, got {}, expected {}",
                self.storages.len(),
                DescLayout::Compute.get_bindings()[1].descriptor_count
            )));
        }

        Ok(writes)
    }
}

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub enum DescLayout {
    Global,
    Image,
    Compute,
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
                    vk::DescriptorSetLayoutBinding {
                        binding: 4,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                        ..Default::default()
                    },
                ]
            }
            DescLayout::Image => {
                vec![vk::DescriptorSetLayoutBinding {
                    binding: 0,
                    descriptor_count: 24,
                    stage_flags: vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::RAYGEN_KHR,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    ..Default::default()
                }]
            }
            DescLayout::Compute => {
                vec![
                    vk::DescriptorSetLayoutBinding {
                        binding: 0,
                        descriptor_count: 24,
                        stage_flags: vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::RAYGEN_KHR,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        ..Default::default()
                    },
                    vk::DescriptorSetLayoutBinding {
                        binding: 1,
                        descriptor_count: 24,
                        stage_flags: vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::RAYGEN_KHR,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        ..Default::default()
                    },
                ]
            }
        }
    }
}

pub struct DescriptorWrite<'a> {
    pub write: vk::WriteDescriptorSet<'a>,
    pub buffer_info: Option<vk::DescriptorBufferInfo>,
    pub image_info: Option<Vec<vk::DescriptorImageInfo>>,
    pub tlases: Option<Vec<vk::AccelerationStructureKHR>>,
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

                        // pointer to the newly pushed element
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
