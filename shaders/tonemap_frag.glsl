#version 450
#pragma shader_stage(fragment)

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform Global {
    float exposure;
    int debug;
    float res_x;
    float res_y;
    float time;
    int half_res;
} globals;

layout(set = 1, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(set = 2, binding = 0) uniform sampler2D colorBuf;

layout( push_constant ) uniform constants {
    mat4 model;
    float time;
} push_consts;

float luminance(vec3 v) {
    return dot(v, vec3(0.2126f, 0.7152f, 0.0722f));
}

vec3 change_luminance(vec3 c_in, float l_out) {
    float l_in = luminance(c_in);
    return c_in * (l_out / l_in);
}

vec3 reinhard_extended_luminance(vec3 v, float max_white_l) {
    float l_old = luminance(v);
    float numerator = l_old * (1.0f + (l_old / (max_white_l * max_white_l)));
    float l_new = numerator / (1.0f + l_old);
    return change_luminance(v, l_new);
}

vec3 abberation(int i) {
    float scale = (float(i) - 3.0) * 0.5;

    return vec3(
        1.0 - min(abs(scale + 1.0), 1.0),
        1.0 - min(abs(scale + 0.0), 1.0),
        1.0 - min(abs(scale - 1.0), 1.0)
    );
}

void main() {
    if (globals.debug != 0) {
        outColor = vec4(
            texture(colorBuf, uv).rgb,
            1.0
        );
        return;
    }

    float ratio = globals.res_x / globals.res_y;

    vec3 sum = vec3(0.0);

    for (int i = 0; i < 7; i++) {
        float scale = (float(i) - 3.0) * 0.001 + 1.0;

        vec2 uv_scaled = ((uv - 0.5) * scale) + 0.5;

        vec3 color = texture(colorBuf, uv_scaled).rgb * globals.exposure * abberation(i);

        sum += color * 0.5;
    }

    vec3 tonemapped = reinhard_extended_luminance(sum, 2.5);
    vec3 color_correction = vec3(
        smoothstep(0.0, 1.0, tonemapped.r),
        smoothstep(0.0, 1.0, tonemapped.g),
        pow(tonemapped.b, 0.9)
    );

    outColor = vec4(
        mix(color_correction, tonemapped, vec3(0.5, 0.6, 0.4)),
        1.0
    );
}
