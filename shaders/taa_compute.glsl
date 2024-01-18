#version 450
#pragma shader_stage(compute)

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) uniform Global {
    float exposure;
    int debug;
    float res_x;
    float res_y;
    float time;
    int frame_index;
    int half_res;
} globals;

layout(set = 1, binding = 0) uniform sampler2D resolve[];
layout(set = 1, binding = 1, rgb10_a2) uniform image2D taa_out;

layout( push_constant ) uniform constants {
    int clear;
} push_consts;

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;

    ivec2 size = imageSize(taa_out);

    if (x <= size.x && y <= size.y) {
        vec2 uv = (vec2(x,y) + 0.5) / vec2(globals.res_x, globals.res_y);

        vec4 current = texture(resolve[0], uv);
        vec4 prev = imageLoad(taa_out, ivec2(x,y));

        vec4 min_color = vec4(1.0);
        vec4 max_color = vec4(0.0);

        for(int offset_x = -1; offset_x <= 1; offset_x ++) {
            for(int offset_y = -1; offset_y <= 1; offset_y ++) {
                vec2 uv_sample = (vec2(x + offset_x, y + offset_y) + 0.5) / vec2(globals.res_x, globals.res_y);

                vec4 sample_color = texture(resolve[0], uv_sample);
                min_color = min(min_color, sample_color);
                max_color = max(max_color, sample_color);
            }
        }

        vec4 prev_clamped = clamp(prev, min_color, max_color);

        if (push_consts.clear == 1) {
            imageStore(taa_out, ivec2(x,y), current);
        } else {
            imageStore(taa_out, ivec2(x,y), mix(prev_clamped, current, 0.1));
        }
    }
}
