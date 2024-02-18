#version 450
#pragma shader_stage(compute)
#extension GL_EXT_debug_printf : enable

#include "common/debug_modes.glsl"
#include "common/defs.glsl"

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) GLOBAL;
layout(set = 0, binding = 2) VIEW_PROJ;

layout(set = 1, binding = 0) uniform sampler2D resolve[];
layout(set = 1, binding = 1, rgb10_a2) uniform image2D taa_out;

layout( push_constant ) uniform constants {
    int clear;
} push_consts;

bool reproject(vec2 uv_in, out vec2 uv_out) {
    ivec2 size = imageSize(taa_out);
    float curr_depth = texture(resolve[4], uv_in).x;

    vec4 per_pos = inverse(view_proj.proj[0]) * vec4(uv_in * 2.0 - 1.0, curr_depth, 1.0);
    per_pos /= per_pos.w;
    vec3 curr_pos = (inverse(view_proj.view[0]) * per_pos).xyz;

    vec4 prev_uv = view_proj.proj[1] * view_proj.view[1] * vec4(curr_pos, 1.0);
    prev_uv /= prev_uv.w;
    prev_uv = prev_uv * 0.5 + 0.5;

    float prev_depth = texture(resolve[5], prev_uv.xy).x;
    uv_out = prev_uv.xy;

    if (prev_uv.x < 0.0 || prev_uv.x > 1.0 || prev_uv.y < 0.0 || prev_uv.y > 1.0/* || abs(curr_depth - prev_depth) > 0.0002*/) {
        return false;
    }

    return true;
}

vec2 get_motion_vector(vec2 uv_in) {
    ivec2 size = imageSize(taa_out);
    float curr_depth = texture(resolve[4], uv_in).x;

    vec4 per_pos = inverse(view_proj.proj[0]) * vec4(uv_in * 2.0 - 1.0, curr_depth, 1.0);
    per_pos /= per_pos.w;
    vec3 curr_pos = (inverse(view_proj.view[0]) * per_pos).xyz;

    vec4 prev_uv = view_proj.proj[1] * view_proj.view[1] * vec4(curr_pos, 1.0);
    prev_uv /= prev_uv.w;
    prev_uv = prev_uv * 0.5 + 0.5;

    return (curr_pos.xy - globals.curr_jitter) - (prev_uv.xy - globals.prev_jitter);
}

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;

    ivec2 size = imageSize(taa_out);

    float blend_factor = 0.1;
    if (push_consts.clear == 1) {
        blend_factor = 1.0;
    }

    if (x <= size.x && y <= size.y) {
        vec2 uv = (vec2(x,y) + 0.5) / vec2(globals.res_x, globals.res_y);

        vec4 current = texture(resolve[0], uv);

        for(int offset_x = -1; offset_x <= 1; offset_x ++) {
            for(int offset_y = -1; offset_y <= 1; offset_y ++) {
                vec2 uv_sample = (vec2(x + offset_x, y + offset_y) + 0.5) / vec2(globals.res_x, globals.res_y);
                uv_sample.x = clamp(uv_sample.x, 0.0, 1.0);
                uv_sample.y = clamp(uv_sample.y, 0.0, 1.0);
            }
        }

        vec2 uv_reprojected;
        if (!reproject(uv, uv_reprojected)) {
            blend_factor = 1.0;
        }

        if (globals.debug == DEBUG_DISOCCLUSION) {
            vec4 history_color = mix(vec4(0.0, 1.0, 0.0, 1.0), vec4(1.0, 0.0, 0.0, 1.0), blend_factor);
            vec4 color = mix(current, history_color, 0.3);
            imageStore(taa_out, ivec2(x,y), color);
            return;
        }

        vec4 prev = texture(resolve[1], uv_reprojected);

        vec4 min_color = vec4(1.0);
        vec4 max_color = vec4(0.0);

        for(int offset_x = -1; offset_x <= 1; offset_x ++) {
            for(int offset_y = -1; offset_y <= 1; offset_y ++) {
                vec2 uv_sample = (vec2(x + offset_x, y + offset_y) + 0.5) / vec2(globals.res_x, globals.res_y);

                uv_sample.x = clamp(uv_sample.x, 0.0, 1.0);
                uv_sample.y = clamp(uv_sample.y, 0.0, 1.0);

                vec4 sample_color = texture(resolve[0], uv_sample);
                min_color = min(min_color, sample_color);
                max_color = max(max_color, sample_color);
            }
        }

        vec4 prev_clamped = clamp(prev, min_color, max_color);
        imageStore(taa_out, ivec2(x,y), mix(prev_clamped, current, blend_factor));
    }
}
