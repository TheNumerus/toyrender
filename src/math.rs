pub const fn halton(index: u32) -> (f32, f32) {
    match index % 8 {
        0 => (0.5, 0.333),
        1 => (0.25, 0.666),
        2 => (0.75, 0.111),
        3 => (0.125, 0.444),
        4 => (0.625, 0.777),
        5 => (0.375, 0.222),
        6 => (0.875, 0.555),
        7 => (0.0625, 0.888),
        _ => (0.5, 0.5),
    }
}

pub fn deg_to_rad(deg: f32) -> f32 {
    deg / (180.0 / std::f32::consts::PI)
}

pub fn rad_to_deg(rad: f32) -> f32 {
    rad * (180.0 / std::f32::consts::PI)
}

pub fn fovx_to_fovy(fovx: f32, ratio: f32) -> f32 {
    2.0 * ((fovx / 2.0).tan() * (1.0 / ratio)).atan()
}

pub fn fovy_to_fovx(fovy: f32, ratio: f32) -> f32 {
    2.0 * ((fovy / 2.0).tan() * ratio).atan()
}
