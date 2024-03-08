use crate::renderer::render_target::{MultipleRenderTarget, RenderTarget};
use crate::renderer::{Globals, ViewProj};
use crate::scene::Environment;
use crate::vulkan::{
    AccelerationStructure, Buffer, DescriptorPool, DescriptorSet, DescriptorSetLayout, Device, VulkanError,
};
use ash::vk;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::rc::Rc;

pub struct RendererDescriptors {
    pub global_sets: Vec<GlobalSet>,
    pub taa_set: ComputeSet,
    pub denoise_direct_temporal_set: ComputeSet,
    pub denoise_indirect_temporal_set: ComputeSet,
    pub atrous_direct_set: ComputeSet,
    pub atrous_indirect_set: ComputeSet,
    pub rt_set: ComputeSet,
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

        let mut compute_sets = pool
            .allocate_sets(6, layouts.get(&DescLayout::Compute).unwrap().inner)?
            .into_iter()
            .map(|inner| ComputeSet { inner })
            .collect::<Vec<_>>();

        let taa_set = compute_sets.pop().unwrap();
        let denoise_direct_temporal_set = compute_sets.pop().unwrap();
        let denoise_indirect_temporal_set = compute_sets.pop().unwrap();
        let atrous_direct_set = compute_sets.pop().unwrap();
        let atrous_indirect_set = compute_sets.pop().unwrap();
        let rt_set = compute_sets.pop().unwrap();

        let mut image_sets = pool
            .allocate_sets(2, layouts.get(&DescLayout::Image).unwrap().inner)?
            .into_iter()
            .map(|inner| ImageSet { inner })
            .collect::<Vec<_>>();

        let light_set = image_sets.pop().unwrap();
        let post_process_set = image_sets.pop().unwrap();

        Ok(Self {
            layouts,
            global_sets,
            taa_set,
            denoise_direct_temporal_set,
            denoise_indirect_temporal_set,
            atrous_direct_set,
            atrous_indirect_set,
            rt_set,
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
                        descriptor_count: 6,
                        stage_flags: vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::RAYGEN_KHR,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        ..Default::default()
                    },
                    vk::DescriptorSetLayoutBinding {
                        binding: 1,
                        descriptor_count: 5,
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

pub struct ComputeSet {
    pub inner: DescriptorSet,
}

impl ComputeSet {
    fn get_layout() -> DescLayout {
        DescLayout::Compute
    }

    pub fn update_rt_src(&self, gbuffer: &Rc<RefCell<MultipleRenderTarget>>) -> DescriptorWrite {
        let image_info = vec![
            gbuffer
                .borrow()
                .descriptor_image_info(0, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            gbuffer
                .borrow()
                .descriptor_image_info(1, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            gbuffer
                .borrow()
                .descriptor_image_info(2, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        ];

        if image_info.len() > Self::get_layout().get_bindings()[0].descriptor_count as usize {
            panic!("INVALID DESCRIPTOR COUNT")
        }

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

        if image_info.len() > Self::get_layout().get_bindings()[1].descriptor_count as usize {
            panic!("INVALID DESCRIPTOR COUNT")
        }

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

    pub fn update_targets(&self, target_1: &RenderTarget, target_2: &RenderTarget) -> DescriptorWrite {
        let image_info = vec![
            target_1.descriptor_image_info(vk::ImageLayout::GENERAL),
            target_2.descriptor_image_info(vk::ImageLayout::GENERAL),
        ];

        if image_info.len() > Self::get_layout().get_bindings()[1].descriptor_count as usize {
            panic!("INVALID DESCRIPTOR COUNT")
        }

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

    pub fn update_taa_src(
        &self,
        src: &RenderTarget,
        hist: &RenderTarget,
        gbuffer: &Rc<RefCell<MultipleRenderTarget>>,
        last_depth: &RenderTarget,
    ) -> DescriptorWrite {
        let image_info = vec![
            src.descriptor_image_info(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            hist.descriptor_image_info(vk::ImageLayout::GENERAL),
            gbuffer
                .borrow()
                .descriptor_image_info(0, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            gbuffer
                .borrow()
                .descriptor_image_info(1, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            gbuffer
                .borrow()
                .descriptor_image_info(2, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            last_depth.descriptor_image_info(vk::ImageLayout::GENERAL),
        ];

        if image_info.len() > Self::get_layout().get_bindings()[0].descriptor_count as usize {
            panic!("INVALID DESCRIPTOR COUNT")
        }

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

    pub fn update_atrous(&self, src: &RenderTarget, dst: &RenderTarget) -> DescriptorWrite {
        let image_info = vec![
            src.descriptor_image_info(vk::ImageLayout::GENERAL),
            dst.descriptor_image_info(vk::ImageLayout::GENERAL),
        ];

        if image_info.len() > Self::get_layout().get_bindings()[1].descriptor_count as usize {
            panic!("INVALID DESCRIPTOR COUNT")
        }

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
}

pub struct NewImageSet {
    pub inner: DescriptorSet,
    targets: Vec<Rc<RefCell<RenderTarget>>>,
}

impl NewImageSet {
    pub fn build(inner: DescriptorSet, targets: Vec<Rc<RefCell<RenderTarget>>>) -> Result<Self, ()> {
        Err(())
    }
}

pub struct ImageSet {
    pub inner: DescriptorSet,
}

impl ImageSet {
    fn get_layout() -> DescLayout {
        DescLayout::Image
    }

    pub fn update_pp(&self, src: &RenderTarget) -> DescriptorWrite {
        let image_info = vec![src.descriptor_image_info(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];

        if image_info.len() > Self::get_layout().get_bindings()[0].descriptor_count as usize {
            panic!("INVALID DESCRIPTOR COUNT")
        }

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

    pub fn update_light(
        &self,
        gbuffer: &Rc<RefCell<MultipleRenderTarget>>,
        rt_direct: &RenderTarget,
        rt_indirect: &RenderTarget,
    ) -> DescriptorWrite {
        let image_info = vec![
            gbuffer
                .borrow()
                .descriptor_image_info(0, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            gbuffer
                .borrow()
                .descriptor_image_info(1, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            gbuffer
                .borrow()
                .descriptor_image_info(2, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            rt_direct.descriptor_image_info(vk::ImageLayout::GENERAL),
            rt_indirect.descriptor_image_info(vk::ImageLayout::GENERAL),
        ];

        if image_info.len() > Self::get_layout().get_bindings()[0].descriptor_count as usize {
            panic!("INVALID DESCRIPTOR COUNT")
        }

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
