#version 450
#pragma shader_stage(compute)
#extension GL_EXT_nonuniform_qualifier : enable

#include "common/debug_modes.glsl"
#include "common/defs.glsl"

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) GLOBAL;
layout(set = 0, binding = 2) VIEW_PROJ;

layout(set = 1, binding = 0) uniform sampler2D images[];
layout(set = 1, binding = 1, rgb10_a2) uniform image2D storages[];

layout(push_constant) uniform constants {
    int clear;
    int out_idx;
    int curr_depth_idx;
    int prev_depth_idx;
    int curr_idx;
    int prev_idx;
    int normal_idx;
} push_consts;

float get_linear_depth(sampler2D depth, vec2 uv) {
    float curr_depth = texture(depth, uv).x;

    vec4 per_pos = inverse(view_proj.proj[0]) * vec4(uv * 2.0 - 1.0, curr_depth, 1.0);
    per_pos /= per_pos.w;
    vec3 curr_pos = (inverse(view_proj.view[0]) * per_pos).xyz;
    vec4 pos = view_proj.proj[0] * view_proj.view[0] * vec4(curr_pos, 1.0);

    return pos.z * pos.w;
}

bool reproject(vec2 uv_in, out vec2 uv_out) {
    ivec2 size = imageSize(storages[push_consts.out_idx]);

    float curr_depth = texture(images[push_consts.curr_depth_idx], uv_in).x;

    vec4 per_pos = inverse(view_proj.proj[0]) * vec4(uv_in * 2.0 - 1.0, curr_depth, 1.0);
    per_pos /= per_pos.w;
    vec3 curr_pos = (inverse(view_proj.view[0]) * per_pos).xyz;
    vec4 pos = view_proj.proj[0] * view_proj.view[0] * vec4(curr_pos, 1.0);
    float lin_depth = pos.z * pos.w;

    vec4 prev_uv = view_proj.proj[1] * view_proj.view[1] * vec4(curr_pos, 1.0);
    prev_uv /= prev_uv.w;
    prev_uv = prev_uv * 0.5 + 0.5;

    vec2 snapped_uv = prev_uv.xy;
    snapped_uv.x = (round((snapped_uv.x * size.x) - 0.5) + 0.5) / size.x;
    snapped_uv.y = (round((snapped_uv.y * size.y) - 0.5) + 0.5) / size.y;

    float prev_lin_depth = get_linear_depth(images[push_consts.prev_depth_idx], snapped_uv.xy);
    uv_out = prev_uv.xy;

    if (prev_uv.x < 0.0 || prev_uv.x > 1.0 || prev_uv.y < 0.0 || prev_uv.y > 1.0 || abs(lin_depth - prev_lin_depth) > 0.01) {
        return false;
    }

    return true;
}

float get_depth_gradient(uint x, uint y, sampler2D depth) {
    ivec2 size = imageSize(storages[push_consts.out_idx]);

    vec2 uv = vec2((x + 0.5) / size.x, (y + 0.5) / size.y);
    float x_offset = 1.0 / size.x;
    float y_offset = 1.0 / size.y;

    float m = get_linear_depth(depth, uv);
    float t = get_linear_depth(depth, uv + vec2(0.0, -y_offset));
    float b = get_linear_depth(depth, uv + vec2(0.0, y_offset));
    float l = get_linear_depth(depth, uv + vec2(-x_offset, 0.0));
    float r = get_linear_depth(depth, uv + vec2(x_offset, 0.0));

    return max(max(abs(t-m), abs(m-b)), max(abs(l-m), abs(m-r)));
}

float get_depth_weight(float base, float sample_depth, float gradient, float dist) {
    float a = abs(base - sample_depth);
    float b = 1.0 * abs(gradient * dist) + 0.0001;

    return exp(-(a / b));
}

float get_luma_weight(float base, float sample_luma, float variance) {
    float a = abs(base - sample_luma);
    float b = 4.0 * (sqrt(variance) + 0.0001);
    return exp(-(a / b));
}

float luminance(vec3 color) {
    return dot(color, vec3(0.2126f, 0.7152f, 0.0722f));
}

float compute_spatial_variance(ivec2 uv_int) {
    vec2 uv = (vec2(uv_int) + 0.5) / vec2(globals.res_x, globals.res_y);
    vec3 base_normal = normalize(texture(images[push_consts.normal_idx], uv).xyz * 2.0 - 1.0);
    float base_depth = get_linear_depth(images[push_consts.curr_depth_idx], uv);
    float luma_base = luminance(texture(images[push_consts.curr_idx], uv).xyz);
    float gradient = get_depth_gradient(uv_int.x, uv_int.y, images[push_consts.curr_depth_idx]);

    ivec2 size = imageSize(storages[push_consts.out_idx]);

    float sum = 0.0;
    float sum_sq = 0.0;
    float sum_weight = 0.0;
    for (int x = -3; x <= 3; x++) {
        for (int y = -3; y <= 3; y++) {
            ivec2 uv = uv_int + ivec2(x, y);
            if (uv.x >= 0 || uv.y >= 0 || uv.x < size.x || uv.y < size.y) {
                vec2 texture_uv = (vec2(uv) + 0.5) / vec2(globals.res_x, globals.res_y);

                vec4 current = texture(images[push_consts.curr_idx], texture_uv);
                float luma_sample = luminance(current.xyz);
                vec3 normal_sample = normalize(texture(images[push_consts.normal_idx], texture_uv).xyz * 2.0 - 1.0);
                float depth_sample = get_linear_depth(images[push_consts.curr_depth_idx], texture_uv);
                float sample_dist = distance(texture_uv, uv);

                float normal_weight = pow(max(dot(base_normal, normal_sample), 0.0), 128.0);
                float depth_weight = get_depth_weight(base_depth, depth_sample, gradient, sample_dist);
                float luma_weight = get_luma_weight(luminance(current.xyz), luma_base, 1.0);

                float weight = max(normal_weight * depth_weight * luma_weight, 0.0);

                sum += luma_sample * weight;
                sum_sq += (luma_sample * luma_sample) * weight;
                sum_weight += weight;
            }
        }
    }

    sum /= sum_weight;
    sum_sq /= sum_weight;

    return sum_sq - (sum * sum);
}

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;

    ivec2 size = imageSize(storages[push_consts.out_idx]);

    float blend_factor = 0.2;
    if (push_consts.clear == 1) {
        blend_factor = 1.0;
    }

    if (x <= size.x && y <= size.y) {
        vec2 uv = (vec2(x, y) + 0.5) / vec2(globals.res_x, globals.res_y);

        vec4 current = texture(images[push_consts.curr_idx], uv);

        float variance = compute_spatial_variance(ivec2(x, y));

        if (globals.debug == DEBUG_DIRECT_VARIANCE || globals.debug == DEBUG_INDIRECT_VARIANCE) {
            imageStore(storages[push_consts.out_idx], ivec2(x, y), vec4(vec3(variance * 5.0), 1.0));
            return;
        }

        vec2 uv_reprojected;
        if (!reproject(uv, uv_reprojected)) {
            blend_factor = 1.0;
        }

        vec4 prev = max(texture(images[push_consts.prev_idx], uv_reprojected), vec4(0.000001));
        imageStore(storages[push_consts.out_idx], ivec2(x, y), mix(prev, vec4(current.xyz, variance), blend_factor));
    }
}