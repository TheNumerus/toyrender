use crate::vulkan::{DescriptorPool, DescriptorSet, DescriptorSetLayout, Device, VulkanError};
use ash::vk;
use std::collections::HashMap;
use std::rc::Rc;

pub struct RendererDescriptors {
    pub layouts: HashMap<DescLayout, DescriptorSetLayout>,
    pub sets: HashMap<DescLayout, Vec<DescriptorSet>>,
}

impl RendererDescriptors {
    pub fn build(device: Rc<Device>) -> Result<Self, VulkanError> {
        let mut layouts = HashMap::new();

        for layout in DescLayout::all() {
            let l = DescriptorSetLayout::new(device.clone(), &layout.get_bindings())?;

            layouts.insert(layout, l);
        }

        Ok(Self {
            layouts,
            sets: HashMap::new(),
        })
    }

    pub fn get_layout(&self, layout: DescLayout) -> &DescriptorSetLayout {
        self.layouts.get(&layout).expect("Incorrectly built layouts")
    }

    pub fn get_sets(&self, layout: DescLayout) -> &Vec<DescriptorSet> {
        self.sets.get(&layout).expect("Incorrectly built sets")
    }

    pub fn allocate_sets(&mut self, pool: &DescriptorPool, frames_in_flight: u32) -> Result<(), VulkanError> {
        for layout in DescLayout::all() {
            let sets = pool.allocate_sets(frames_in_flight, self.get_layout(layout).inner)?;
            self.sets.insert(layout, sets);
        }

        Ok(())
    }
}

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub enum DescLayout {
    Global,
    View,
    GBuffer,
    GBufferPlus,
    PostProcess,
}

impl DescLayout {
    pub const fn all() -> [Self; 5] {
        [
            Self::Global,
            Self::View,
            Self::GBuffer,
            Self::GBufferPlus,
            Self::PostProcess,
        ]
    }

    pub fn get_bindings(&self) -> Vec<vk::DescriptorSetLayoutBinding> {
        match self {
            DescLayout::Global => {
                vec![
                    vk::DescriptorSetLayoutBinding {
                        binding: 0,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::VERTEX
                            | vk::ShaderStageFlags::FRAGMENT
                            | vk::ShaderStageFlags::RAYGEN_KHR,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        ..Default::default()
                    },
                    vk::DescriptorSetLayoutBinding {
                        binding: 1,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::VERTEX
                            | vk::ShaderStageFlags::FRAGMENT
                            | vk::ShaderStageFlags::RAYGEN_KHR,
                        descriptor_type: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                        ..Default::default()
                    },
                    vk::DescriptorSetLayoutBinding {
                        binding: 2,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::RAYGEN_KHR,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        ..Default::default()
                    },
                ]
            }
            DescLayout::View => {
                vec![vk::DescriptorSetLayoutBinding {
                    binding: 0,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::VERTEX
                        | vk::ShaderStageFlags::FRAGMENT
                        | vk::ShaderStageFlags::RAYGEN_KHR,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    ..Default::default()
                }]
            }
            DescLayout::GBuffer => {
                vec![vk::DescriptorSetLayoutBinding {
                    binding: 0,
                    descriptor_count: 3,
                    stage_flags: vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::RAYGEN_KHR,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    ..Default::default()
                }]
            }
            DescLayout::GBufferPlus => {
                vec![vk::DescriptorSetLayoutBinding {
                    binding: 0,
                    descriptor_count: 4,
                    stage_flags: vk::ShaderStageFlags::FRAGMENT,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    ..Default::default()
                }]
            }
            DescLayout::PostProcess => {
                vec![vk::DescriptorSetLayoutBinding {
                    binding: 0,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::FRAGMENT,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    ..Default::default()
                }]
            }
        }
    }
}
