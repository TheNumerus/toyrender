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
    float x = (float(tid.x) + 0.5) / 32;
    float y = (float(tid.y) + 0.5) / 32;

    float3 dirBase = Toyrender::Octa::decode(float2(x, y) * 2.0 - 1.0);

    float3 tangent, bitangent;
    compute_default_basis(dirBase, tangent, bitangent);

    hash_state = (1024 * tid.z);

    float3 acc = 0.0;

    // might be useful later
    int iterations = 1;

    // use later when discarding sun from sky
    float3 dirSun = env.sun_dir_sun_angle.xyz;

    for (int iter = 0; iter < iterations; iter++) {
        float2 rand_vars_sphere = float2(pcg_hash(), pcg_hash());
        float3 hemi = rand_hemi(rand_vars_sphere);
        let dir = hemi.x * tangent + hemi.y * bitangent + hemi.z * dirBase;

        if (dot(dir, dirSun) > 0.95) {
            //continue;
        }

        float3 col = Toyrender::Sky::sample(dir, images[pc.src_idx], env.rotation);
        if (pc.clamp > 0.0) {
            float luma = max(luminance(col), 0.0001);
            col *= min(pc.clamp / luma, 1.0);
        }
        acc += col;
    }

    acc /= float(iterations);

    colors[tid.z] = acc;

    GroupMemoryBarrierWithGroupSync();

    for (uint s = 1024 / 2; s > 0; s >>= 1) {
        if (tid.z < s) {
            colors[tid.z] += colors[tid.z + s];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (tid.z == 0) {
        storages[pc.out_idx].Store(
            tid.xy,
            float4(
                colors[0] / 1024.0,
                0.0
            )
        );
        return;
    }

    // workaround to force usage of desc set 0
    if (env.mode == SkyMode::SingleColor) {
        return;
    }
}