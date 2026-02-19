import Common;
import Sampling;

#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_shader_realtime_clock : enable
#extension GL_EXT_nonuniform_qualifier : enable


#define PI 3.141569

struct Payload {
    bool isMiss;
    float dist;
    float3 normal;
    float3 color;
};

[[vk::binding(0, 0)]]
ConstantBuffer<Global> global;

[[vk::binding(1, 0)]]
RaytracingAccelerationStructure tlas;

[[vk::binding(2, 0)]]
ConstantBuffer<ViewProj> vp;

[[vk::binding(3, 0)]]
ConstantBuffer<Environment> env;

[[vk::binding(0, 1)]]
Sampler2D images[];

[[vk::binding(1, 1)]]
RWTexture2D<float4> storages[];

struct PushConsts {
    int samples;
    int depth_idx;
    int normal_idx;
    int direct_idx;
    int indirect_idx;
    int sky_idx;
    float direct_trace_distance;
    float indirect_trace_distance;
};

[vk::push_constant]
uniform PushConsts pc;

uint pcg_hash(uint i) {
    uint state = i * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

void compute_default_basis(const float3 normal, out float3 x, out float3 y) {
    float3 z  = normal;
    const float yz = -z.y * z.z;
    y = normalize(((abs(z.z) > 0.99999f) ? float3(-z.x * z.y, 1.0f - z.y * z.y, yz) : float3(-z.x * z.z, yz, 1.0f - z.z * z.z)));

    x = cross(y, z);
}

void trace(float3 pos, float3 dir, inout Payload payload) {
    RayDesc ray;
    ray.Direction = dir;
    ray.Origin = pos;
    ray.TMax = pc.indirect_trace_distance;
    ray.TMin = 0.0001;

    TraceRay(
        tlas,
        RAY_FLAG_FORCE_OPAQUE,
        0xFF,
        0,
        0,
        0,
        ray,
        payload
    );
}

void traceShadow(float3 pos, float3 dir, inout Payload payload) {
    RayDesc ray;
    ray.Direction = dir;
    ray.Origin = pos;
    ray.TMax = pc.direct_trace_distance;
    ray.TMin = 0.0001;

    TraceRay(
        tlas,
        RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
        0xFF,
        0,
        0,
        0,
        ray,
        payload
    );
}

uint time_diff(uint startTime, uint endTime) {
    return endTime >= startTime ? (endTime-startTime) : (~0u-(startTime-endTime));
}

float3 sky_color(float3 view_dir) {
    float2 dir = normalize(view_dir.xy);

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

    return images[pc.sky_idx].Sample(float2(u, v)).xyz;
}

[shader("raygeneration")]
void raygen() {
    uint3 launch_idx = DispatchRaysIndex();

    int2 size = int2(int(global.res_x), int(global.res_y));

    if (launch_idx.x >= size.x || launch_idx.y >= size.y) {
        return;
    }

    Payload payload;

    payload.isMiss = false;
    payload.color = float3(0.0);

    uint start = getRealtimeClockLow().x;

    float2 sampleUv = (float2(launch_idx.xy) + 0.5) / float2(global.draw_res_x, global.draw_res_y);
    float2 uv = (float2(launch_idx.xy) + 0.5) / float2(global.res_x, global.res_y);

    float depth = images[pc.depth_idx].Sample(uv).x;

    if (depth == 0.0) {
        storages[pc.direct_idx].Store(
            int2(launch_idx.xy),
            float4(0.0)
        );
        storages[pc.indirect_idx].Store(
            int2(launch_idx.xy),
            float4(0.0)
        );
        return;
    }

    float3 normal = normalize((images[pc.normal_idx].Sample(uv).xyz - 0.5) * 2.0);

    float4 per_pos = mul(vp.proj_inverse, float4(sampleUv * 2.0 - 1.0, depth, 1.0));

    per_pos /= per_pos.w;
    float3 vertPos = mul(vp.view_inverse, per_pos).xyz;

    float3 loc = transpose(vp.view_inverse)[3].xyz;
    float3 view_dir = normalize(vertPos - loc);
    float dis = distance(loc, vertPos);
    float3 bias = dis * normal * 0.001;

    float3 indirect = float3(0.0);

    int samples = pc.samples;

    float3 tangent, bitangent;

    float2 rand = float2(
        float((pcg_hash((2 + int(global.frame_index)) * (launch_idx.x + launch_idx.y * uint(global.res_x)))) % 1024) / 1024.0,
        float((pcg_hash((2 + 1 + int(global.frame_index)) * (launch_idx.x + launch_idx.y * uint(global.res_x)))) % 1024) / 1024.0
    );

    float3 sphere = rand_sphere(rand) * 0.1;

    compute_default_basis(env.sun_dir_sun_angle.xyz, tangent, bitangent);

    float3 cone = rand_cone(rand, cos(env.sun_dir_sun_angle.w));
    float3 ray_shadow_dir = cone.x * tangent + cone.y * bitangent + cone.z * env.sun_dir_sun_angle.xyz;

    float intensity_indirect = 1.0;
    float total_intensity_indirect = 1.0;
    float3 pos = vertPos + bias;
    float3 hitNormal = normal;
    float3 dir = view_dir;

    float bounce_energy = 0.95;

    for (int i = 0; i < samples; i++) {
        compute_default_basis(hitNormal, tangent, bitangent);

        float3 hemi = rand_hemi(
            float2(
                float((pcg_hash((int(global.frame_index)) * (launch_idx.x + launch_idx.y * uint(global.res_x) + i))) % 1024) / 1024.0,
                float((pcg_hash((1 + int(global.frame_index)) * (launch_idx.x + launch_idx.y * uint(global.res_x) + i))) % 1024) / 1024.0
            )
        );

        float d = 0.8 + (1.0 - 0.8) * pow(abs(dot(-hitNormal, dir)), 5.0);

        if ((float((pcg_hash((2 + int(global.frame_index)) * (launch_idx.x + launch_idx.y * uint(global.res_x) + i))) % 1024) / 1024.0) > d) {
            float ddot = dot(dir, hitNormal);
            dir = dir - 2.0 * hitNormal * (ddot);
            compute_default_basis(dir, tangent, bitangent);

            float2 new_rand = float2(
                float((pcg_hash((4 + samples * 2 + int(global.frame_index)) * (launch_idx.x + launch_idx.y * uint(global.res_x) + i))) % 1024) / 1024.0,
                float((pcg_hash((3 + samples * 2 + int(global.frame_index)) * (launch_idx.x + launch_idx.y * uint(global.res_x) + i))) % 1024) / 1024.0
            );

            float3 cone = rand_cone(new_rand, cos(ddot * 0.3));
            dir = cone.x * tangent + cone.y * bitangent + cone.z * dir;
        } else {
            dir = hemi.x * tangent + hemi.y * bitangent + hemi.z * hitNormal;
        }

        trace(pos, dir, payload);

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
            if (dot(ray_shadow_dir, hitNormal) <= 0.0 || env.sun_int == 0.0) {
                intensity_indirect *= 0.9;
                total_intensity_indirect += intensity_indirect;
                continue;
            }

            traceShadow(pos, ray_shadow_dir, payload);

            // sun boounce
            if (payload.isMiss) {
                indirect += intensity_indirect * bounce_energy * env.sun_color_sky_int.xyz * env.sun_int;
            } else {
                indirect += intensity_indirect * bounce_energy * payload.color;
            }
        }

        intensity_indirect *= bounce_energy;
        total_intensity_indirect += intensity_indirect;
    }

    indirect /= total_intensity_indirect;

    payload.isMiss = false;

    if (dot(ray_shadow_dir, normal) > 0.0 && env.sun_int > 0.0) {
        payload.isMiss = true;
        traceShadow(vertPos + bias, ray_shadow_dir, payload);
    }

    float shadow = float(payload.isMiss);
    float3 direct = shadow * env.sun_color_sky_int.xyz * env.sun_int;

    uint end = getRealtimeClockLow().x;

    if (global.debug == DebugMode::Time) {
        float time = float(time_diff(start, end)) / 1024 / 128;
        storages[pc.direct_idx].Store(
            int2(launch_idx.xy),
            float4(
                time * time,
                -(pow(2.0 * time - 1.0, 2.0)) + 1.0,
                pow(1.0 - time, 2.0),
                0.0
            )
        );
    } else {
        storages[pc.direct_idx].Store(
            int2(launch_idx.xy),
            float4(
                direct,
                0.0
            )
        );
        storages[pc.indirect_idx].Store(
            int2(launch_idx.xy),
            float4(
                indirect,
                0.0
            )
        );
    }
}


[shader("miss")]
void miss(inout Payload payload) {
    payload.isMiss = true;
    payload.color = float3(0.0);
}


[shader("closesthit")]
void chit(inout Payload payload, in BuiltInTriangleIntersectionAttributes attr) {
    payload.isMiss = false;
    payload.dist = RayTCurrent();

    float3x4 otw = ObjectToWorld();

    float3 vertPos0 = mul(otw, float4(HitTriangleVertexPosition(0), 0.0));
    float3 vertPos1 = mul(otw, float4(HitTriangleVertexPosition(1), 0.0));
    float3 vertPos2 = mul(otw, float4(HitTriangleVertexPosition(2), 0.0));

    float3 normal = cross(vertPos1 - vertPos0, vertPos2 - vertPos0);

    payload.normal = normalize(normal);
}
