import Common;
import Sampling;
import Sky;

#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_shader_realtime_clock : enable
#extension GL_EXT_nonuniform_qualifier : enable

#define PI 3.141569

struct Payload {
    bool isMiss;
    float dist;
    float3 normal;
    float3 color;
    float3 hit;
};

[[vk::binding(0, 0)]]
ConstantBuffer<Global> global;

[[vk::binding(1, 0)]]
RaytracingAccelerationStructure tlas;

[[vk::binding(2, 0)]]
ConstantBuffer<ViewProj> vp;

[[vk::binding(3, 0)]]
ConstantBuffer<Environment> env;

[[vk::binding(5, 0)]]
StructuredBuffer<MeshInstanceInfo> meshData;

[[vk::binding(0, 1)]]
Sampler2D images[];

[[vk::binding(1, 1)]]
RWTexture2D<float4> storages[];

struct PushConsts {
    int sample;
    int bounces;
    int out_idx;
    int sky_idx;
    float direct_trace_distance;
    float indirect_trace_distance;
    float fov;
    float indirect_intensity_clamp;
};

[vk::push_constant]
uniform PushConsts pc;

static uint hash_state;

float pcg_hash() {
    uint state = hash_state;
    hash_state = hash_state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    uint res = (word >> 22u) ^ word;
    return float(res % 4096) / 4096.0;
}

void compute_default_basis(const float3 normal, out float3 x, out float3 y) {
    float3 z  = normal;
    const float yz = -z.y * z.z;
    y = normalize(((abs(z.z) > 0.99999f) ? float3(-z.x * z.y, 1.0f - z.y * z.y, yz) : float3(-z.x * z.z, yz, 1.0f - z.z * z.z)));

    x = cross(y, z);
}

void tracePrimary(float3 pos, float3 dir, inout Payload payload) {
    RayDesc ray;
    ray.Direction = dir;
    ray.Origin = pos;
    ray.TMax = 1000.0;
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

float3 generatePinholeRay(int2 pixel, int width, int height) {
    float tanHalfAngle = tan(pc.fov / 2.f);
    return normalize(float3(float2(2 * (pixel.x + global.curr_jitter.x) - width, -2 * (pixel.y + global.curr_jitter.y) + height) * tanHalfAngle / width, -1));
}

constexpr static float origin = 1.0f / 32.0f;
constexpr static float float_scale = 1.0f / 65536.0f;
constexpr static float int_scale = 256.0f;

float3 offset_ray(const float3 p, const float3 n) {
    int3 of_i = int3(int(int_scale * n.x), int(int_scale * n.y), int(int_scale * n.z));

    float3 p_i = float3(
        asfloat(asint(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
        asfloat(asint(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
        asfloat(asint(p.z) + ((p.z < 0) ? -of_i.z : of_i.z))
    );

    return float3(
        abs(p.x) < origin ? p.x + float_scale * n.x : p_i.x,
        abs(p.y) < origin ? p.y + float_scale * n.y : p_i.y,
        abs(p.z) < origin ? p.z + float_scale * n.z : p_i.z
    );
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

    float3 indirect = float3(0.0);
    float3 direct = float3(0.0);

    float3 tangent, bitangent;

    // setup hasher with min index 1 so that even first sample is noisy
    hash_state = (launch_idx.x + launch_idx.y * uint(global.res_x)) * (global.frame_index + 1);

    float2 rand = float2(pcg_hash(), pcg_hash());

    compute_default_basis(env.sun_dir_sun_angle.xyz, tangent, bitangent);

    float3 cone = rand_cone(rand, cos(env.sun_dir_sun_angle.w));
    float3 ray_shadow_dir = cone.x * tangent + cone.y * bitangent + cone.z * env.sun_dir_sun_angle.xyz;

    // scale of energy for subsequent bounces
    float3 intensity_indirect = 1.0;

    float3 dir = generatePinholeRay(int2(launch_idx.x, launch_idx.y), int(global.draw_res_x), int(global.draw_res_y));

    dir = mul(float3x3(vp.view_inverse), dir).xyz;

    float3 pos = transpose(vp.view_inverse)[3].xyz;
    float3 hitNormal;

    float3 primaryPos; 
    float3 primaryNormal;

    tracePrimary(pos, dir, payload);

    float3 primaryColor = payload.color;

    pos = payload.hit;

    if (payload.isMiss) {
        storages[pc.out_idx].Store(
            int2(launch_idx.xy),
            float4(
                payload.color * env.sky_color_intensity.w,
                0.0
            )
        );
        return;
    } else {
        primaryPos = pos;

        hitNormal = payload.normal;
        primaryNormal = hitNormal;
    }

    if (global.debug == DebugMode::Normal) {
        storages[pc.out_idx].Store(
            int2(launch_idx.xy),
            float4(
                (primaryNormal * 0.5) + 0.5,
                0.0
            )
        );
        return;
    }

    // TODO model proper energy passing from material
    float bounce_energy = 0.95;

    float roughness = 0.3;

    // current surface_color
    float3 current_color = payload.color;

    for (int i = 0; i < pc.bounces; i++) {
        compute_default_basis(hitNormal, tangent, bitangent);

        float d = roughness + (1.0 - roughness) * pow(abs(dot(-hitNormal, dir)), 5.0);

        if (pcg_hash() > d) {
            // specular ray
            float ddot = dot(dir, hitNormal);
            dir = dir - 2.0 * hitNormal * (ddot);

            float2 new_rand = float2(pcg_hash(), pcg_hash());

            float3 ggx_half_vector = ggx(new_rand, roughness);

            float3 new_norm = ggx_half_vector.x * tangent + ggx_half_vector.y * bitangent + ggx_half_vector.z * hitNormal;

            // TODO fix this monstrosity
            dir = reflect(dir, new_norm);
            dir = reflect(dir, hitNormal);

            if (dot(dir, hitNormal) <= 0.0) {
                break;
            }
            // if tracing specular, reset color - not true for metals
            current_color = float3(1.0);
        } else {
            float3 hemi = rand_hemi(float2(pcg_hash(), pcg_hash()));
            dir = hemi.x * tangent + hemi.y * bitangent + hemi.z * hitNormal;
        }

        trace(offset_ray(pos, hitNormal), dir, payload);

        // sky
        if (payload.isMiss) {
            indirect += intensity_indirect * bounce_energy * payload.color * env.sky_color_intensity.w * current_color;
            intensity_indirect *= bounce_energy * current_color;
            break;
        } else {
            // try shadow on new position
            pos = payload.hit;
            hitNormal = payload.normal;

            // flip on backface hit
            if (dot(dir, hitNormal) > 0.0) {
                hitNormal = -hitNormal;
            }

            // ignore shadow rays from unreachable normals
            if (dot(ray_shadow_dir, hitNormal) <= 0.0 || env.sun_color_intensity.w == 0.0) {
                intensity_indirect *= bounce_energy * current_color;
                current_color = payload.color;
                continue;
            }

            float3 hit_color = payload.color;

            traceShadow(offset_ray(pos, hitNormal), ray_shadow_dir, payload);

            // sun bounce
            if (payload.isMiss) {
                direct += intensity_indirect * bounce_energy * env.sun_color_intensity.xyz * env.sun_color_intensity.w * current_color * hit_color;
            }
            current_color = hit_color;
            intensity_indirect *= bounce_energy * current_color * payload.color;
        }
    }

    payload.isMiss = false;

    if (dot(ray_shadow_dir, primaryNormal) > 0.0 && env.sun_color_intensity.w > 0.0) {
        payload.isMiss = true;
        traceShadow(primaryPos, ray_shadow_dir, payload);
    }

    float shadow = float(payload.isMiss) * dot(ray_shadow_dir, primaryNormal);
    direct += shadow * env.sun_color_intensity.xyz * env.sun_color_intensity.w * primaryColor;

    uint end = getRealtimeClockLow().x;

    if (global.debug == DebugMode::Direct) {
        storages[pc.out_idx].Store(
            int2(launch_idx.xy),
            float4(
                (direct),
                0.0
            )
        );
    } else if (global.debug == DebugMode::Indirect) {
        storages[pc.out_idx].Store(
            int2(launch_idx.xy),
            float4(
                (indirect),
                0.0
            )
        );
    } else {
        storages[pc.out_idx].Store(
            int2(launch_idx.xy),
            float4(
                (direct + indirect),
                0.0
            )
        );
    }
}


[shader("miss")]
void miss(inout Payload payload) {
    if (env.mode == SkyMode::Shader) {
        payload.color = Toyrender::Sky::sample_distorted(WorldRayDirection(), images[pc.sky_idx]);
    } else if (env.mode == SkyMode::SingleColor) {
        payload.color = env.sky_color_intensity.xyz;
    } else {
        payload.color = Toyrender::Sky::sample(WorldRayDirection(), images[pc.sky_idx], env.rotation);
    }

    if (pc.indirect_intensity_clamp != 0.0) {
        payload.color = min(float3(pc.indirect_intensity_clamp), payload.color);
    }

    payload.isMiss = true;
}


[shader("closesthit")]
void chit(inout Payload payload, in BuiltInTriangleIntersectionAttributes attr) {
    payload.isMiss = false;
    payload.dist = RayTCurrent();

    float3x4 otw = ObjectToWorld();

    float3 vertPos0 = mul(otw, float4(HitTriangleVertexPosition(0), 1.0));
    float3 vertPos1 = mul(otw, float4(HitTriangleVertexPosition(1), 1.0));
    float3 vertPos2 = mul(otw, float4(HitTriangleVertexPosition(2), 1.0));

    float3 edgeA = vertPos1 - vertPos0;
    float3 edgeB = vertPos2 - vertPos0;

    //payload.normal = normalize(normal);
    payload.hit = attr.barycentrics.x * edgeA + attr.barycentrics.y * edgeB + vertPos0;

    let vertex0 = meshData[InstanceIndex()].indexData[PrimitiveIndex() * 3];
    let vertex1 = meshData[InstanceIndex()].indexData[PrimitiveIndex() * 3 + 1];
    let vertex2 = meshData[InstanceIndex()].indexData[PrimitiveIndex() * 3 + 2];

    let color0 = *((float3 *)&meshData[InstanceIndex()].vertexData[vertex0][6]);
    let color1 = *((float3 *)&meshData[InstanceIndex()].vertexData[vertex1][6]);
    let color2 = *((float3 *)&meshData[InstanceIndex()].vertexData[vertex2][6]);

    let colorA = color1 - color0;
    let colorB = color2 - color0;

    payload.color = attr.barycentrics.x * colorA + attr.barycentrics.y * colorB + color0;

    let normal0 = *((float3 *)&meshData[InstanceIndex()].vertexData[vertex0][3]);
    let normal1 = *((float3 *)&meshData[InstanceIndex()].vertexData[vertex1][3]);
    let normal2 = *((float3 *)&meshData[InstanceIndex()].vertexData[vertex2][3]);

    float3 normalA = normal1 - normal0;
    float3 normalB = normal2 - normal0;

    let normalBlend = attr.barycentrics.x * normalA + attr.barycentrics.y * normalB + normal0;
    payload.normal = normalize(mul(float3x3(transpose(meshData[InstanceIndex()].model_inverse)), normalBlend));

    if (HitKind() == HIT_KIND_TRIANGLE_BACK_FACE) {
        payload.normal = -payload.normal;
    }

    //hash_state = InstanceID() * 4096;
    //payload.color = pow(float3(pcg_hash(), pcg_hash(), pcg_hash()), 2.2);

    //payload.color = float3(0.7);
}
