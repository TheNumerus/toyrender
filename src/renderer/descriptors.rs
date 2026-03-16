use crate::err::AppError;
use crate::renderer::ResourceSubsystem;
use crate::renderer::render_target::RenderTargets;
use crate::vulkan::{DescriptorPool, DescriptorSet, DescriptorSetLayout, Device, VulkanError};
use ash::vk;
use rspirv_reflect::DescriptorInfo;
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::rc::Rc;

pub struct DescriptorLayouts {
    pub inner: HashMap<DescLayout, DescriptorSetLayout>,
}

impl DescriptorLayouts {
    pub fn create(device: Rc<Device>) -> Result<DescriptorLayouts, VulkanError> {
        let mut inner = HashMap::new();

        for layout in [DescLayout::Global, DescLayout::Compute, DescLayout::Image] {
            let l = DescriptorSetLayout::new(device.clone(), &layout.get_bindings(), format!("{layout:?}"))?;

            inner.insert(layout, l);
        }

        Ok(Self { inner })
    }

    pub fn guess_layout_from_reflection(
        &self,
        info: &BTreeMap<u32, DescriptorInfo>,
    ) -> Result<vk::DescriptorSetLayout, AppError> {
        for layout in [DescLayout::Global, DescLayout::Compute, DescLayout::Image] {
            let mut matches = true;
            let binds = layout.get_bindings();

            for (index, bind) in binds.iter().enumerate() {
                if let Some(di) = info.get(&(index as u32))
                    && di.ty.0 != bind.descriptor_type.as_raw() as u32
                {
                    matches = false;
                    break;
                }
            }

            if matches {
                return Ok(self.inner[&layout].inner);
            }
        }

        Err(AppError::Import(String::from("Unknown layout defined")))
    }

    pub fn get_layout(&self, layout: DescLayout) -> &DescriptorSetLayout {
        self.inner.get(&layout).expect("Incorrectly built layouts")
    }
}

pub struct RendererDescriptors {
    pub global_set: DescriptorSet,
    pub image_set: DescriptorSet,
    pub compute_set: DescriptorSet,
    pub samplers: HashMap<uuid::Uuid, u32>,
    pub storages: HashMap<uuid::Uuid, u32>,
}

impl RendererDescriptors {
    pub fn build(pool: &DescriptorPool, layouts: &DescriptorLayouts) -> Result<Self, VulkanError> {
        let global_set = pool
            .allocate_sets(1, layouts.get_layout(DescLayout::Global).inner)?
            .pop()
            .unwrap();

        let compute_set = pool
            .allocate_sets(1, layouts.get_layout(DescLayout::Compute).inner)?
            .pop()
            .unwrap();

        let image_set = pool
            .allocate_sets(1, layouts.get_layout(DescLayout::Image).inner)?
            .pop()
            .unwrap();

        Ok(Self {
            global_set,
            image_set,
            compute_set,
            samplers: HashMap::new(),
            storages: HashMap::new(),
        })
    }

    pub fn update_resources(
        &mut self,
        targets: &RenderTargets,
        resource_subsystem: Option<&ResourceSubsystem>,
    ) -> Result<Vec<DescriptorWrite>, AppError> {
        self.samplers = HashMap::new();
        self.storages = HashMap::new();

        let mut writes = Vec::new();

        for (_name, item) in &targets.targets {
            if item.value.borrow().usage.contains(vk::ImageUsageFlags::STORAGE) {
                let image_info = vec![item.value.borrow().descriptor_image_info(vk::ImageLayout::GENERAL)];
                let index = self.storages.len() as u32;

                let write = vk::WriteDescriptorSet {
                    dst_set: self.compute_set.inner,
                    dst_binding: 1,
                    dst_array_element: index,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    ..Default::default()
                };

                writes.push(DescriptorWrite {
                    write,
                    buffer_info: None,
                    image_info: Some(image_info),
                    tlases: None,
                });

                self.storages.insert(item.value.borrow().id, index);
                item.value.borrow_mut().storage_index = Some(index);
            }

            if item.value.borrow().usage.contains(vk::ImageUsageFlags::SAMPLED) {
                let image_info = vec![
                    item.value
                        .borrow()
                        .descriptor_image_info(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                ];

                let index = self.samplers.len() as u32;

                let write = vk::WriteDescriptorSet {
                    dst_set: self.compute_set.inner,
                    dst_binding: 0,
                    dst_array_element: index,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    ..Default::default()
                };
                writes.push(DescriptorWrite {
                    write,
                    buffer_info: None,
                    image_info: Some(image_info.clone()),
                    tlases: None,
                });

                let write = vk::WriteDescriptorSet {
                    dst_set: self.image_set.inner,
                    dst_binding: 0,
                    dst_array_element: index,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    ..Default::default()
                };
                writes.push(DescriptorWrite {
                    write,
                    buffer_info: None,
                    image_info: Some(image_info),
                    tlases: None,
                });

                self.samplers.insert(item.value.borrow().id, index);
                item.value.borrow_mut().sampler_index = Some(index);
            }
        }

        if let Some(rs) = resource_subsystem {
            for (id, (_, iw, s)) in &rs.textures {
                let image_info = vec![vk::DescriptorImageInfo {
                    sampler: s.inner,
                    image_view: iw.inner,
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                }];

                let index = self.samplers.len() as u32;

                let write = vk::WriteDescriptorSet {
                    dst_set: self.compute_set.inner,
                    dst_binding: 0,
                    dst_array_element: index,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    ..Default::default()
                };
                writes.push(DescriptorWrite {
                    write,
                    buffer_info: None,
                    image_info: Some(image_info.clone()),
                    tlases: None,
                });

                let write = vk::WriteDescriptorSet {
                    dst_set: self.image_set.inner,
                    dst_binding: 0,
                    dst_array_element: index,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    ..Default::default()
                };
                writes.push(DescriptorWrite {
                    write,
                    buffer_info: None,
                    image_info: Some(image_info.clone()),
                    tlases: None,
                });

                self.samplers.insert(*id, index);
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
                | vk::ShaderStageFlags::COMPUTE
                | vk::ShaderStageFlags::MISS_KHR,
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
                    vk::DescriptorSetLayoutBinding {
                        binding: 5,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
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
                        stage_flags: vk::ShaderStageFlags::COMPUTE
                            | vk::ShaderStageFlags::RAYGEN_KHR
                            | vk::ShaderStageFlags::MISS_KHR,
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
