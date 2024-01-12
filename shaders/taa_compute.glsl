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

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;

    ivec2 size = imageSize(taa_out);

    if (x <= size.x && y <= size.y) {
        vec2 uv = (vec2(x,y) + 0.5) / vec2(globals.res_x, globals.res_y);

        vec4 current = texture(resolve[0], uv);
        vec4 prev = imageLoad(taa_out, ivec2(x,y));

        imageStore(taa_out, ivec2(x,y), mix(prev, current, 0.2));
    }
}
