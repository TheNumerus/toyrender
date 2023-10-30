use crate::err::AppError;
use crate::vulkan::Vertex;
use gltf::buffer::Data;
use gltf::{Accessor, Semantic};
use nalgebra_glm::{Vec2, Vec3, Vec4};

pub fn extract_mesh(slice: &[u8]) -> Result<(Vec<Vertex>, Vec<u32>), AppError> {
    let (document, buffers, _images) = gltf::import_slice(slice).unwrap();

    let mesh = match document.meshes().next() {
        Some(m) => m,
        None => return Err(AppError::Other("No meshes in file".into())),
    };

    let (mut vertices, mut indices) = (Vec::new(), Vec::new());

    for ref primitive in mesh.primitives() {
        let pos = primitive.get(&Semantic::Positions);
        let normal = primitive.get(&Semantic::Normals);
        let uv = primitive.get(&Semantic::TexCoords(0));
        let indices_ac = primitive.indices();
        let color = primitive.get(&Semantic::Colors(0));

        let pos = accessor_to_slice(pos.unwrap(), &buffers);
        let uv = accessor_to_slice(uv.unwrap(), &buffers);
        let indices_slice = accessor_to_slice::<u16>(indices_ac.unwrap(), &buffers);

        if pos.len() / 3 != uv.len() / 2 {
            return Err(AppError::Other("Mismatched mesh data length".into()));
        }

        vertices = pos
            .chunks_exact(3)
            .zip(uv.chunks_exact(2))
            .map(|(pos, uv)| Vertex {
                pos: Vec3::new(pos[0], pos[1], pos[2]),
                uv: Vec2::new(uv[0], uv[1]),
                color: Vec4::new(0.0, 0.0, 0.0, 0.0),
            })
            .collect();
        indices = indices_slice.iter().map(|&a| a as u32).collect();
    }

    Ok((vertices, indices))
}

fn accessor_to_slice<'a, T>(accessor: Accessor<'a>, data: &[Data]) -> &'a [T] {
    let count = accessor.count();
    let size = accessor.size();
    let accessor_offset = accessor.offset();

    let buffer_view = accessor.view().unwrap();
    let view_offset = buffer_view.offset();
    let pos = data.get(buffer_view.buffer().index()).unwrap().as_ptr() as *const T;
    let slice = unsafe {
        std::slice::from_raw_parts(
            pos.add((view_offset + accessor_offset) / std::mem::size_of::<T>()),
            count * size / std::mem::size_of::<T>(),
        )
    };

    slice
}
