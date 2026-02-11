#version 460 core
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_shader_realtime_clock : enable
#extension GL_EXT_nonuniform_qualifier : enable
#pragma shader_stage(raygen)

#include "common/sampling.glsl"
#include "common/debug_modes.glsl"
#include "common/defs.glsl"

#define PI 3.141569

layout(location = 0) rayPayloadEXT struct {
    bool isMiss;
    float dist;
    vec3 normal;
    vec3 color;
} payload;

layout(set = 0, binding = 0) GLOBAL;
layout(set = 0, binding = 1) uniform accelerationStructureEXT tlas;
layout(set = 0, binding = 2) VIEW_PROJ;
layout(set = 0, binding = 3) ENV;

layout(set = 1, binding = 0) uniform sampler2D images[];
layout(set = 1, binding = 1, rgba16f) uniform image2D storages[];

layout( push_constant ) uniform constants {
    int samples;
    int depth_idx;
    int normal_idx;
    int direct_idx;
    int indirect_idx;
    int sky_idx;
    float trace_distance;
    int sky_only;
} push_consts;

uint pcg_hash(uint i) {
    uint state = i * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

void compute_default_basis(const vec3 normal, out vec3 x, out vec3 y) {
    vec3 z  = normal;
    const float yz = -z.y * z.z;
    y = normalize(((abs(z.z) > 0.99999f) ? vec3(-z.x * z.y, 1.0f - z.y * z.y, yz) : vec3(-z.x * z.z, yz, 1.0f - z.z * z.z)));

    x = cross(y, z);
}

void trace (vec3 pos, vec3 dir) {
    traceRayEXT(
        tlas,
        gl_RayFlagsOpaqueEXT,
        0xFF,
        0,
        0,
        0,
        pos,
        0.0001,
        dir,
        push_consts.trace_distance,
        0
    );
}

void traceShadow (vec3 pos, vec3 dir) {
    traceRayEXT(
        tlas,
        gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT,
        0xFF,
        0,
        0,
        0,
        pos,
        0.0001,
        dir,
        100.0,
        0
    );
}

uint time_diff(uint startTime, uint endTime) {
    return endTime >= startTime ? (endTime-startTime) : (~0u-(startTime-endTime));
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
    if (isnan(u) || isinf(u)) {
        u = 0.0;
    }

    float v = ((asin(view_dir.z) / (PI / 2.0)) * 0.5) + 0.5;

    if (isnan(v) || isinf(v)) {
        v = 0.0;
    }

    u = clamp(u, 0.0, 1.0);
    v = clamp(v, 0.0, 1.0);
    v = (sqrt(abs((v - 0.5) * 2.0)) * sign(v - 0.5)) * 0.5 + 0.5;

    return texture(images[push_consts.sky_idx], vec2(u, v)).xyz;
}

void main () {
    payload.isMiss = false;
    payload.color = vec3(0.0);

    uint start = clockRealtime2x32EXT().x;

    vec2 uv = (vec2(gl_LaunchIDEXT.xy) + 0.5) / vec2(globals.res_x, globals.res_y);
    if (globals.half_res == 1) {
        uv = (vec2(gl_LaunchIDEXT.xy) + 0.5) / vec2(globals.res_x/2, globals.res_y/2);
    }

    float depth = texture(images[push_consts.depth_idx], uv).x;

    if (depth == 0.0) {
        imageStore(
            storages[push_consts.direct_idx],
            ivec2(gl_LaunchIDEXT.xy),
            vec4(0.0)
        );
        imageStore(
            storages[push_consts.indirect_idx],
            ivec2(gl_LaunchIDEXT.xy),
            vec4(0.0)
        );
        return;
    }

    vec3 normal = normalize((texture(images[push_consts.normal_idx], uv).xyz - 0.5) * 2.0);

    vec4 per_pos = inverse(view_proj.proj[0]) * vec4(uv * 2.0 - 1.0, depth, 1.0);
    per_pos /= per_pos.w;
    vec3 vertPos = (inverse(view_proj.view[0]) * per_pos).xyz;

    vec3 loc = inverse(view_proj.view[0])[3].xyz;
    vec3 view_dir = normalize(vertPos - loc);
    float dis = distance(loc, vertPos);
    vec3 bias = dis * normal * 0.001;

    vec3 indirect = vec3(0.0);

    int samples = push_consts.samples;

    vec3 tangent, bitangent;

    vec2 rand = vec2(
        float((pcg_hash((2 + int(globals.frame_index)) * (gl_LaunchIDEXT.x + gl_LaunchIDEXT.y * uint(globals.res_x)))) % 1024) / 1024.0,
        float((pcg_hash((2 + 1 + int(globals.frame_index)) * (gl_LaunchIDEXT.x + gl_LaunchIDEXT.y * uint(globals.res_x)))) % 1024) / 1024.0
    );

    vec3 sphere = rand_sphere(rand) * 0.1;

    compute_default_basis(env.sun_dir_sun_angle.xyz, tangent, bitangent);

    vec3 cone = rand_cone(rand, cos(env.sun_dir_sun_angle.w));
    vec3 ray_shadow_dir = cone.x * tangent + cone.y * bitangent + cone.z * env.sun_dir_sun_angle.xyz;

    float intensity_indirect = 1.0;
    float total_intensity_indirect = 1.0;
    vec3 pos = vertPos + bias;
    vec3 hitNormal = normal;
    vec3 dir = view_dir;

    float bounce_energy = 0.95;

    for (int i = 0; i < samples; i++) {
        compute_default_basis(hitNormal, tangent, bitangent);

        vec3 hemi = rand_hemi(
            vec2(
                float((pcg_hash((int(globals.frame_index)) * (gl_LaunchIDEXT.x + gl_LaunchIDEXT.y * uint(globals.res_x) + i))) % 1024) / 1024.0,
                float((pcg_hash((1 + int(globals.frame_index)) * (gl_LaunchIDEXT.x + gl_LaunchIDEXT.y * uint(globals.res_x) + i))) % 1024) / 1024.0
            )
        );

        float d = 0.8 + (1.0 - 0.8) * pow(dot(-hitNormal, dir), 5.0);

        if ((float((pcg_hash((2 + int(globals.frame_index)) * (gl_LaunchIDEXT.x + gl_LaunchIDEXT.y * uint(globals.res_x) + i))) % 1024) / 1024.0) > d) {
            float ddot = dot(dir, hitNormal);
            dir = dir - 2.0 * hitNormal * (ddot);
            compute_default_basis(dir, tangent, bitangent);
            vec3 cone = rand_cone(rand, cos(ddot * 0.3));
            dir = cone.x * tangent + cone.y * bitangent + cone.z * dir;
        } else {
            dir = hemi.x * tangent + hemi.y * bitangent + hemi.z * hitNormal;
        }

        trace(pos, dir);

        // sky
        if (payload.isMiss) {
            indirect += intensity_indirect * bounce_energy * sky_color(dir) * env.sun_color_sky_int.w;
            intensity_indirect *= bounce_energy;
            total_intensity_indirect += intensity_indirect;
            break;
        } else {
            // try shadow on new position
            pos += dir * payload.dist;
            hitNormal = payload.normal;

            // flip on backface hit
            if (dot(dir, hitNormal) > 0.0) {
                hitNormal = -hitNormal;
            }

            // ignore shadow rays from unreachable normals
            if (dot(ray_shadow_dir, hitNormal) <= 0.0 || push_consts.sky_only == 1) {
                intensity_indirect *= 0.9;
                total_intensity_indirect += intensity_indirect;
                continue;
            }

            traceShadow(pos, ray_shadow_dir);

            // sun boounce
            if (payload.isMiss) {
                indirect += intensity_indirect * bounce_energy * env.sun_color_sky_int.xyz;
            } else {
                indirect += intensity_indirect * bounce_energy * payload.color;
            }
        }

        intensity_indirect *= bounce_energy;
        total_intensity_indirect += intensity_indirect;
    }

    indirect /= total_intensity_indirect;

    payload.isMiss = false;

    if (dot(ray_shadow_dir, normal) > 0.0 && push_consts.sky_only != 1) {
        payload.isMiss = true;
        traceShadow(vertPos + bias, ray_shadow_dir);
    }

    float shadow = float(payload.isMiss);
    vec3 direct = shadow * env.sun_color_sky_int.xyz;

    uint end = clockRealtime2x32EXT().x;

    if (globals.debug == DEBUG_TIME) {
        float time = float(time_diff(start, end)) / 1024 / 128;
        imageStore(
            storages[push_consts.direct_idx],
            ivec2(gl_LaunchIDEXT.xy),
            vec4(
                time * time,
                -(pow(2.0 * time - 1.0, 2.0)) + 1.0,
                pow(1.0 - time, 2.0),
                0.0
            )
        );
    } else {
        imageStore(
            storages[push_consts.direct_idx],
            ivec2(gl_LaunchIDEXT.xy),
                vec4(
                direct,
                0.0
            )
        );
        imageStore(
            storages[push_consts.indirect_idx],
            ivec2(gl_LaunchIDEXT.xy),
            vec4(
                indirect,
                0.0
            )
        );
    }
}