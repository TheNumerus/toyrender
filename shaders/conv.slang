import Common;
import Sky;
import Octa;
import Sampling;

[[vk::binding(0, 0)]]
ConstantBuffer<Global> global;

[[vk::binding(2, 0)]]
ConstantBuffer<ViewProj> vp;

[[vk::binding(3, 0)]]
ConstantBuffer<Environment> env;

[[vk::binding(0, 1)]]
Sampler2D images[];

[[vk::binding(1, 1)]]
RWTexture2D<float4> storages[];

struct PushConsts {
    int src_idx;
    int out_idx;
    float clamp;
    int iteration;
};

[vk::push_constant]
uniform PushConsts pc;

static uint hash_state;

float uintToFloat(uint x) {
    return asfloat(0x3f800000 | (x >> 9)) - 1.f;
}

uint xorshift(inout uint rngState) {
    rngState ^= rngState << 13;
    rngState ^= rngState >> 17;
    rngState ^= rngState << 5;

    return rngState;
}

float rand(inout uint rngState) {
    return uintToFloat(xorshift(rngState));
}

void compute_default_basis(const float3 normal, out float3 x, out float3 y) {
    float3 z = normal;
    const float yz = -z.y * z.z;
    y = normalize(((abs(z.z) > 0.99999f) ? float3(-z.x * z.y, 1.0f - z.y * z.y, yz) : float3(-z.x * z.z, yz, 1.0f - z.z * z.z)));

    x = cross(y, z);
}

float luminance(float3 v) {
    return dot(v, float3(0.2126f, 0.7152f, 0.0722f));
}

groupshared float3 colors[1024];

[numthreads(1, 1, 1024)]
[shader("compute")]
void main(uint3 tid: SV_DispatchThreadID, uniform PushConsts pc) {
    int width;
    int height;
    storages[pc.out_idx].GetDimensions(width, height);
    // padding to ensure not seams are visible
    height -= 1;
    width -= 1;

    float x = (float(tid.x)) / width;
    float y = (float(tid.y)) / height;

    float3 dirBase = Toyrender::Octa::decode(float2(x, y) * 2.0 - 1.0);

    float3 tangent, bitangent;
    compute_default_basis(dirBase, tangent, bitangent);

    hash_state = (tid.z + pc.iteration * 1024);
    rand(hash_state);

    float3 acc = 0.0;
    
    float2 rand_vars_sphere = float2(rand(hash_state), rand(hash_state));
    float3 hemi = rand_hemi(rand_vars_sphere);
    let dir = hemi.x * tangent + hemi.y * bitangent + hemi.z * dirBase;

    float3 col;
    switch (env.mode) {
    case SkyMode::Shader:
        col = Toyrender::Sky::sample_distorted(dir, images[pc.src_idx]);
            break;
    case SkyMode::SingleColor:
        col = env.sky_color_intensity.xyz;
            break;
    case SkyMode::Textured:
        col = Toyrender::Sky::sample(dir, images[pc.src_idx], 0.0);
            break;
    }

    acc += col;

    colors[tid.z] = acc;

    GroupMemoryBarrierWithGroupSync();

    for (uint s = 1024 / 2; s > 0; s >>= 1) {
        if (tid.z < s) {
            colors[tid.z] += colors[tid.z + s];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (tid.z == 0) {
        float3 blend = (colors[0] / 1024.0);

        float3 orig = storages[pc.out_idx].Load(tid.xy).xyz;

        float alpha = (pc.iteration - 1.0) / float(pc.iteration);
        blend = lerp(blend, orig, alpha);

        storages[pc.out_idx].Store(
            tid.xy,
            float4(
                float3(blend),
                0.0
            )
        );

        return;
    }
}

groupshared float4 colorWeigths[1024];

[numthreads(1, 1, 1024)]
[shader("compute")]
void mainSunless(uint3 tid: SV_DispatchThreadID, uniform PushConsts pc) {
    int width;
    int height;
    storages[pc.out_idx].GetDimensions(width, height);
    // padding to ensure not seams are visible
    height -= 1;
    width -= 1;

    float x = (float(tid.x)) / width;
    float y = (float(tid.y)) / height;

    float3 dirBase = Toyrender::Octa::decode(float2(x, y) * 2.0 - 1.0);

    float3 tangent, bitangent;
    compute_default_basis(dirBase, tangent, bitangent);

    hash_state = (1024 * tid.z + pc.iteration * 1024 * 1024);

    float4 acc = 0.0;

    float2 rand_vars_sphere = float2(rand(hash_state), rand(hash_state));
    float3 hemi = rand_hemi(rand_vars_sphere);
    let dir = hemi.x * tangent + hemi.y * bitangent + hemi.z * dirBase;

    float weight = step(0.95, dot(env.sun_dir_sun_angle.xyz, dir));

    float3 col;
    switch (env.mode) {
    case SkyMode::Shader:
        col = Toyrender::Sky::sample_distorted(dir, images[pc.src_idx]);
        break;
    case SkyMode::SingleColor:
        col = env.sky_color_intensity.xyz;
        break;
    case SkyMode::Textured:
        col = Toyrender::Sky::sample(dir, images[pc.src_idx], 0.0);
        break;
    }
    acc += float4(col, weight);

    colorWeigths[tid.z] = acc;

    GroupMemoryBarrierWithGroupSync();

    for (uint s = 1024 / 2; s > 0; s >>= 1) {
        if (tid.z < s) {
            colorWeigths[tid.z] += colorWeigths[tid.z + s];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (tid.z == 0) {
        float3 blend = (colorWeigths[0].xyz / colorWeigths[0].w);

        float3 orig = storages[pc.out_idx].Load(tid.xy).xyz;

        float alpha = (pc.iteration - 1.0) / float(pc.iteration);
        blend = lerp(blend, orig, alpha);

        storages[pc.out_idx].Store(
            tid.xy,
            float4(
                float3(blend),
                0.0
            )
        );

        return;
    }
}