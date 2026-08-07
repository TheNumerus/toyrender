import Common;
import Sampling;
import Sky;

#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_shader_realtime_clock : enable
#extension GL_EXT_nonuniform_qualifier : enable

struct Payload {
    bool isMiss;
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
    int depth_idx;
    int normal_idx;
    int out_idx;
    float indirect_trace_distance;
};

[vk::push_constant]
uniform PushConsts pc;

[vk::constant_id(0)]
const SkyMode skyVariant;

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

void trace(float3 pos, float3 dir, inout Payload payload) {
    RayDesc ray;
    ray.Direction = dir;
    ray.Origin = pos;
    ray.TMax = pc.indirect_trace_distance;
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

    payload.isMiss = true;

    float2 sampleUv = (float2(launch_idx.xy) + 0.5) / float2(global.draw_res_x, global.draw_res_y);
    float2 uv = (float2(launch_idx.xy) + 0.5) / float2(global.res_x, global.res_y);

    float depth = storages[pc.depth_idx].Load(launch_idx.xy).x;

    if (depth == 0.0) {
        storages[pc.out_idx].Store(
            int2(launch_idx.xy),
            float4(0.0)
        );
        return;
    }

    uint start = getRealtimeClockLow().x;

    float3 normal = normalize((storages[pc.normal_idx].Load(launch_idx.xy).xyz - 0.5) * 2.0);

    float4 per_pos = mul(vp.proj_inverse, float4(sampleUv * 2.0 - 1.0, depth, 1.0));

    per_pos /= per_pos.w;
    float3 vertPos = mul(vp.view_inverse, per_pos).xyz;

    float3 loc = transpose(vp.view_inverse)[3].xyz;
    float3 view_dir = normalize(vertPos - loc);
    float dis = distance(loc, vertPos);

    float3 occlusion = float3(0.0);

    float3 tangent, bitangent;

    // setup hasher with min index 1 so that even first sample is noisy
    hash_state = (launch_idx.x + launch_idx.y * uint(global.res_x)) * (global.frame_index + 1);

    float2 rand = float2(pcg_hash(), pcg_hash());

    compute_default_basis(env.sun_dir_sun_angle.xyz, tangent, bitangent);

    float3 pos = vertPos;
    float3 hitNormal = normal;
    float3 dir = view_dir;

    float3 hemi = rand_hemi(float2(pcg_hash(), pcg_hash()));
    dir = hemi.x * tangent + hemi.y * bitangent + hemi.z * hitNormal;

    trace(offset_ray(pos, hitNormal), dir, payload);

    if (payload.isMiss) {
        occlusion = 1.0;
    }

    uint end = getRealtimeClockLow().x;

    if (global.debug == DebugMode::Time) {
        float time = float(time_diff(start, end)) / 1024 / 128;
        storages[pc.out_idx].Store(
            int2(launch_idx.xy),
            float4(
                time * time,
                -(pow(2.0 * time - 1.0, 2.0)) + 1.0,
                pow(1.0 - time, 2.0),
                0.0
            )
        );
    } else {
        storages[pc.out_idx].Store(
            int2(launch_idx.xy),
            float4(
                occlusion,
                1.0
            )
        );
    }
}

[shader("miss")]
void miss(inout Payload payload) {
    payload.isMiss = true;
}


[shader("closesthit")]
void chit(inout Payload payload, in BuiltInTriangleIntersectionAttributes attr) {
    payload.isMiss = false;
}
