#version 450
#pragma shader_stage(compute)

#include "common/debug_modes.glsl"
#include "common/defs.glsl"

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) GLOBAL;
layout(set = 0, binding = 2) VIEW_PROJ;

layout(set = 1, binding = 0) uniform sampler2D[] gb;
layout(set = 1, binding = 1, rgba16) uniform image2D atrous[];

layout( push_constant ) uniform constants {
    int level;
} push_consts;

int stride(int level) {
    return int(pow(2, level));
}

float luminance(vec3 color) {
    return dot(color, vec3(0.2126f, 0.7152f, 0.0722f));
}

float get_weight(int offset) {
    if (offset == 0) {
        return 0.375;
    } else if (abs(offset) == 1) {
        return 0.25;
    } else {
        return 0.0625;
    }
}

float get_gauss_3x3_weight(int offset) {
    if (offset == 0) {
        return 0.25;
    } else {
        return 0.125;
    }
}

float get_linear_depth(sampler2D depth, vec2 uv) {
    float curr_depth = texture(depth, uv).x;

    vec4 per_pos = inverse(view_proj.proj[0]) * vec4(uv * 2.0 - 1.0, curr_depth, 1.0);
    per_pos /= per_pos.w;
    vec3 curr_pos = (inverse(view_proj.view[0]) * per_pos).xyz;
    vec4 pos = view_proj.proj[0] * view_proj.view[0] * vec4(curr_pos, 1.0);

    return pos.z * pos.w;
}

float get_depth_gradient(uint x, uint y, sampler2D depth) {
    ivec2 size = imageSize(atrous[0]);

    vec2 uv = vec2((x + 0.5) / size.x, (y + 0.5) / size.y);
    float x_offset = 1.0 / size.x;
    float y_offset = 1.0 / size.y;

    float m = get_linear_depth(depth, uv);
    float t = get_linear_depth(depth, uv + vec2(0.0, -y_offset));
    float b = get_linear_depth(depth, uv + vec2(0.0, y_offset));
    float l = get_linear_depth(depth, uv + vec2(-x_offset, 0.0));
    float r = get_linear_depth(depth, uv + vec2(x_offset, 0.0));

    return max(max(abs(t-m), abs(m-b)),  max(abs(l-m), abs(m-r)));
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

float get_blurred_variance(ivec2 uv) {
    ivec2 size = imageSize(atrous[0]);
    float sum = 0.0;
    float weight = 0.0;

    for (int x = -1; x <= -1; x++) {
        for (int y = -1; y <= -1; y++) {
            float val = imageLoad(atrous[0], ivec2(clamp(uv.x + x, 0, size.x - 1), clamp(uv.y + y, 0, size.y - 1))).x;
            float weight_sample = get_gauss_3x3_weight(x) * get_gauss_3x3_weight(y);
            sum += val * weight_sample;
            weight += weight_sample;
        }
    }

    return sum / weight;
}

float get_blurred_variance_1(ivec2 uv) {
    ivec2 size = imageSize(atrous[1]);
    float sum = 0.0;
    float weight = 0.0;

    for (int x = -1; x <= -1; x++) {
        for (int y = -1; y <= -1; y++) {
            float val = imageLoad(atrous[1], ivec2(clamp(uv.x + x, 0, size.x - 1), clamp(uv.y + y, 0, size.y - 1))).x;
            float weight_sample = get_gauss_3x3_weight(x) * get_gauss_3x3_weight(y);
            sum += val * weight_sample;
            weight += weight_sample;
        }
    }

    return sum / weight;
}

void main() {
    uint x_base = gl_GlobalInvocationID.x;
    uint y_base = gl_GlobalInvocationID.y;

    ivec2 size = imageSize(atrous[0]);

    if (globals.debug == DEBUG_INDIRECT || globals.debug == DEBUG_DIRECT) {
        vec4 raw = imageLoad(atrous[1], ivec2(x_base, y_base));
        imageStore(atrous[0], ivec2(x_base, y_base), raw);
        return;
    } else if (globals.debug == DEBUG_DIRECT_VARIANCE || globals.debug == DEBUG_INDIRECT_VARIANCE) {
        vec4 raw = imageLoad(atrous[1], ivec2(x_base, y_base));
        imageStore(atrous[0], ivec2(x_base, y_base), raw);
        return;
    }

    if (x_base <= size.x && y_base <= size.y) {
        vec4 result = imageLoad(atrous[0], ivec2(x_base, y_base));
        vec4 raw = imageLoad(atrous[1], ivec2(x_base, y_base));

        vec4 sum = vec4(0.0);
        float weight_sum = 0.0;

        float new_variance_sum = 0.0;
        float new_variance_weight_sum = 0.0;

        vec2 uv = vec2((x_base + 0.5) / size.x, (y_base + 0.5) / size.y);

        float luma_base = luminance(raw.xyz);
        vec3 normal_base = normalize(texture(gb[1], uv).xyz * 2.0 - 1.0);
        float depth_base = get_linear_depth(gb[2], uv);
        float gradient = get_depth_gradient(x_base, y_base, gb[2]);

        for (int y = -2; y <= 2; y++) {
            for (int x = -2; x <= 2; x++) {
                int x_final = int(x_base) + x * stride(push_consts.level);
                int y_final = int(y_base) + y * stride(push_consts.level);

                if (x_final >= size.x || x_final < 0 || y_final >= size.y || y_final < 0) {
                    continue;
                }

                vec2 sample_uv = vec2((x_final + 0.5) / size.x, (y_final + 0.5) / size.y);
                vec3 sample_normal = normalize(texture(gb[1], sample_uv).xyz * 2.0 - 1.0);
                float sample_depth = get_linear_depth(gb[2], sample_uv);
                float sample_dist = distance(sample_uv, uv);

                vec4 img_in;
                float variance;
                if (push_consts.level == 0) {
                    img_in = imageLoad(atrous[1], ivec2(x_final, y_final));
                    variance = get_blurred_variance_1(ivec2(x_final, y_final));
                } else {
                    img_in = imageLoad(atrous[0], ivec2(x_final, y_final));
                    variance = get_blurred_variance(ivec2(x_final, y_final));
                }

                float normal_weight = pow(max(dot(normal_base, sample_normal), 0.0), 128.0);
                float depth_weight = get_depth_weight(depth_base, sample_depth, gradient, sample_dist);
                float luma_weight = get_luma_weight(luminance(img_in.xyz), luma_base, variance);

                float weight = max(get_weight(x) * get_weight(y) * normal_weight * depth_weight * luma_weight, 0.00001);
                float variance_weight = max(pow(get_weight(x) * get_weight(y), 2.0) * pow(normal_weight * depth_weight * luma_weight, 2.0), 0.00001);

                sum += img_in * weight;
                weight_sum += weight;
                new_variance_sum += variance * variance_weight;
                new_variance_weight_sum += weight * weight;
            }
        }

        vec4 total = vec4((sum / weight_sum).xyz, new_variance_sum / new_variance_weight_sum);

        if (push_consts.level == 0) {
            imageStore(atrous[1], ivec2(x_base, y_base), total);
        }

        imageStore(atrous[0], ivec2(x_base, y_base), total);
    }
}
