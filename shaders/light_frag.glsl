#version 450
#pragma shader_stage(compute)
#extension GL_EXT_nonuniform_qualifier : enable

#include "common/debug_modes.glsl"
#include "common/defs.glsl"

layout (local_size_x = 16, local_size_y = 16) in;

#define PI 3.14159265359
#define PHI 1.618033988

layout(set = 0, binding = 0) GLOBAL;
layout(set = 0, binding = 2) VIEW_PROJ;
layout(set = 0, binding = 3) ENV;

layout(set = 1, binding = 0) uniform sampler2D images[];
layout(set = 1, binding = 1, rgba16) uniform image2D storages[];

layout(push_constant) uniform constants {
    int out_idx;
    int color_idx;
    int normal_idx;
    int depth_idx;
    int direct_idx;
    int indirect_idx;
    int sky_idx;
} push_consts;

vec3 light(vec3 normal, vec3 light_dir, vec3 shadow)  {
    return max(dot(normal, light_dir) * shadow, 0.0);
}

float spec(vec3 loc, vec3 normal, vec3 pos, vec3 lightPos, float shadow, float power) {
    vec3 view_dir = normalize(pos - loc);
    vec3 ref_dir = view_dir - 2.0 * normal * dot(normal, view_dir);

    return pow(max(dot(ref_dir, normalize(lightPos - pos)), 0.0), power) * shadow;
}

float schlick(float f0, vec3 l, vec3 n) {
    return f0 + (1.0 - f0) * pow(1.0 - dot(l, n), 5.0);
}

vec3 sky_color(vec3 view_dir) {
    ivec2 size = imageSize(storages[push_consts.sky_idx]);
    vec2 dir = normalize(view_dir.xy);

    float u = 0.0;
    if (dir.y >= 0.0) {
        u = acos(dir.x);
    } else {
        u = -acos(dir.x);
    }

    u = (u + PI) / (2.0 * PI);

    float v = ((asin(view_dir.z) / (PI / 2.0)) * 0.5) + 0.5;

    ivec2 uv = ivec2(max(min(int(u * size.x), size.x - 1), 0), max(min(int(v * size.y), size.y - 1), 0));

    vec3 color = imageLoad(storages[push_consts.sky_idx], uv).xyz;

    return color;
}

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    ivec2 size = imageSize(storages[push_consts.out_idx]);
    vec2 uv = (vec2(x, y) + 0.5) / vec2(globals.res_x, globals.res_y);

    vec3 light_dir = env.sun_direction;

    float ratio = globals.res_x / globals.res_y;

    vec4 color = texture(images[push_consts.color_idx], uv);
    vec3 normal = texture(images[push_consts.normal_idx], uv).xyz * 2.0 - 1.0;
    float depth = 1.0 - texture(images[push_consts.depth_idx], uv).x;

    vec4 per_pos = inverse(view_proj.proj[0]) * vec4(uv * 2.0 - 1.0, depth, 1.0);
    per_pos /= per_pos.w;
    vec3 vertPos = (inverse(view_proj.view[0]) *  per_pos).xyz;
    vec3 loc = inverse(view_proj.view[0])[3].xyz;
    vec3 view_dir = normalize(vertPos - loc);

    vec3 direct = texture(images[push_consts.direct_idx], uv).xyz;
    vec3 indirect = texture(images[push_consts.indirect_idx], uv).xyz;

    vec3 diffuse_dir = light(normalize(normal), light_dir, direct) + indirect;
    vec3 specular = vec3(0.0);
    vec3 lighted = ((diffuse_dir + specular));

    vec3 sky_col = sky_color(view_dir) * pow(2.0, globals.exposure);

    if (x <= size.x && y <= size.y) {
        imageStore(storages[push_consts.out_idx], ivec2(x, y), vec4(mix(sky_col, lighted, color.a), 1.0));
    }
}
