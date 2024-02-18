pub fn halton(index: u32) -> (f32, f32) {
    (halton_sequence(index, 2), halton_sequence(index, 3))
}

pub fn halton_sequence(mut index: u32, base: u32) -> f32 {
    let mut f = 1.0;
    let mut r = 0.0;

    while index > 0 {
        f /= base as f32;
        r += f * (index % base) as f32;
        index = (index as f32 / base as f32).floor() as u32;
    }

    r
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
