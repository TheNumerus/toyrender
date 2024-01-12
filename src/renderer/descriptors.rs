use crate::renderer::render_target::{GBuffer, RenderTarget};
use crate::renderer::{Globals, ViewProj};
use crate::vulkan::{
    AccelerationStructure, Buffer, DescriptorPool, DescriptorSet, DescriptorSetLayout, Device, VulkanError,
};
use ash::vk;
use std::collections::HashMap;
use std::ffi::c_void;
use std::rc::Rc;

pub struct RendererDescriptors {
    pub global_sets: Vec<GlobalSet>,
    pub taa_set: ComputeSet,
    pub rtao_set: ComputeSet,
    pub light_set: ImageSet,
    pub post_process_set: ImageSet,
    pub layouts: HashMap<DescLayout, DescriptorSetLayout>,
}

impl RendererDescriptors {
    pub fn build(device: Rc<Device>, pool: &DescriptorPool, frames_in_flight: u32) -> Result<Self, VulkanError> {
        let mut layouts = HashMap::new();

        for layout in [DescLayout::Global, DescLayout::Compute, DescLayout::Image] {
            let l = DescriptorSetLayout::new(device.clone(), &layout.get_bindings())?;

            layouts.insert(layout, l);
        }

        let global_sets = pool
            .allocate_sets(frames_in_flight, layouts.get(&DescLayout::Global).unwrap().inner)?
            .into_iter()
            .map(|inner| GlobalSet { inner })
            .collect();

        let mut compute_sets = pool.allocate_sets(2, layouts.get(&DescLayout::Compute).unwrap().inner)?;
        let taa_set = ComputeSet {
            inner: compute_sets.pop().unwrap(),
        };
        let rtao_set = ComputeSet {
            inner: compute_sets.pop().unwrap(),
        };

        let mut image_sets = pool.allocate_sets(2, layouts.get(&DescLayout::Image).unwrap().inner)?;
        let light_set = ImageSet {
            inner: image_sets.pop().unwrap(),
        };
        let post_process_set = ImageSet {
            inner: image_sets.pop().unwrap(),
        };

        Ok(Self {
            layouts,
            global_sets,
            taa_set,
            rtao_set,
            light_set,
            post_process_set,
        })
    }

    pub fn get_layout(&self, layout: DescLayout) -> &DescriptorSetLayout {
        self.layouts.get(&layout).expect("Incorrectly built layouts")
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
        match self {
            DescLayout::Global => {
                vec![
                    vk::DescriptorSetLayoutBinding {
                        binding: 0,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::VERTEX
                            | vk::ShaderStageFlags::FRAGMENT
                            | vk::ShaderStageFlags::RAYGEN_KHR
                            | vk::ShaderStageFlags::COMPUTE,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        ..Default::default()
                    },
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
                    vk::DescriptorSetLayoutBinding {
                        binding: 2,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::VERTEX
                            | vk::ShaderStageFlags::FRAGMENT
                            | vk::ShaderStageFlags::RAYGEN_KHR
                            | vk::ShaderStageFlags::COMPUTE,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        ..Default::default()
                    },
                ]
            }
            DescLayout::Image => {
                vec![vk::DescriptorSetLayoutBinding {
                    binding: 0,
                    descriptor_count: 8,
                    stage_flags: vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::RAYGEN_KHR,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    ..Default::default()
                }]
            }
            DescLayout::Compute => {
                vec![
                    vk::DescriptorSetLayoutBinding {
                        binding: 0,
                        descriptor_count: 4,
                        stage_flags: vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::RAYGEN_KHR,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        ..Default::default()
                    },
                    vk::DescriptorSetLayoutBinding {
                        binding: 1,
                        descriptor_count: 1,
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
        create_buffer_update(buffer, std::mem::size_of::<ViewProj>() as u64, &self.inner, 2)
    }
}

pub struct ComputeSet {
    pub inner: DescriptorSet,
}

impl ComputeSet {
    pub fn update_rt_src(&self, gbuffer: &GBuffer) -> DescriptorWrite {
        let image_info = vec![
            vk::DescriptorImageInfo {
                sampler: gbuffer.sampler.inner,
                image_view: gbuffer.views[0].inner,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            vk::DescriptorImageInfo {
                sampler: gbuffer.sampler.inner,
                image_view: gbuffer.views[1].inner,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            vk::DescriptorImageInfo {
                sampler: gbuffer.sampler.inner,
                image_view: gbuffer.views[2].inner,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
        ];

        let write = vk::WriteDescriptorSet {
            dst_set: self.inner.inner,
            dst_binding: 0,
            dst_array_element: 0,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            ..Default::default()
        };

        DescriptorWrite {
            write,
            buffer_info: None,
            image_info: Some(image_info),
            tlases: None,
        }
    }

    pub fn update_target(&self, target: &RenderTarget) -> DescriptorWrite {
        let image_info = vec![target.descriptor_image_info(vk::ImageLayout::GENERAL)];

        let write = vk::WriteDescriptorSet {
            dst_set: self.inner.inner,
            dst_binding: 1,
            dst_array_element: 0,
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            ..Default::default()
        };

        DescriptorWrite {
            write,
            buffer_info: None,
            image_info: Some(image_info),
            tlases: None,
        }
    }

    pub fn update_taa_src(&self, src: &RenderTarget) -> DescriptorWrite {
        let image_info = vec![src.descriptor_image_info(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];

        let write = vk::WriteDescriptorSet {
            dst_set: self.inner.inner,
            dst_binding: 0,
            dst_array_element: 0,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            ..Default::default()
        };

        DescriptorWrite {
            write,
            buffer_info: None,
            image_info: Some(image_info),
            tlases: None,
        }
    }
}

pub struct ImageSet {
    pub inner: DescriptorSet,
}

impl ImageSet {
    pub fn update_pp(&self, src: &RenderTarget) -> DescriptorWrite {
        let image_info = vec![src.descriptor_image_info(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];

        let write = vk::WriteDescriptorSet {
            dst_set: self.inner.inner,
            dst_binding: 0,
            dst_array_element: 0,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            ..Default::default()
        };

        DescriptorWrite {
            write,
            buffer_info: None,
            image_info: Some(image_info),
            tlases: None,
        }
    }

    pub fn update_light(&self, gbuffer: &GBuffer, rt: &RenderTarget) -> DescriptorWrite {
        let image_info = vec![
            vk::DescriptorImageInfo {
                sampler: gbuffer.sampler.inner,
                image_view: gbuffer.views[0].inner,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            vk::DescriptorImageInfo {
                sampler: gbuffer.sampler.inner,
                image_view: gbuffer.views[1].inner,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            vk::DescriptorImageInfo {
                sampler: gbuffer.sampler.inner,
                image_view: gbuffer.views[2].inner,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            rt.descriptor_image_info(vk::ImageLayout::GENERAL),
        ];

        let write = vk::WriteDescriptorSet {
            dst_set: self.inner.inner,
            dst_binding: 0,
            dst_array_element: 0,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            ..Default::default()
        };

        DescriptorWrite {
            write,
            buffer_info: None,
            image_info: Some(image_info),
            tlases: None,
        }
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
