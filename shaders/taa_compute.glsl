#version 450
#pragma shader_stage(compute)

layout(set = 0, binding = 0) uniform Global {
    float exposure;
    int debug;
    float res_x;
    float res_y;
    float time;
    int frame_index;
    int half_res;
} globals;

layout(set = 2, binding = 1, rgb10_a2) uniform image2D taa_out;
layout(set = 2, binding = 0) uniform sampler2D resolve[];

void main() {
    vec2 uv = (vec2(gl_GlobalInvocationID.xy) + 0.5) / vec2(globals.res_x/2, globals.res_y/2);

    vec4 current = texture(resolve[0], uv);
    vec4 prev = texture(resolve[1], uv);

    imageStore(taa_out, ivec2(gl_GlobalInvocationID.xy), mix(current, prev, 0.1));
}
