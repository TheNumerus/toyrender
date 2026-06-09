import Common;
import Sky;

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
};

[vk::push_constant]
uniform PushConsts pc;

groupshared float row[1024];

[numthreads(1024, 1, 1)]
[shader("compute")]
void main(uint3 tid: SV_DispatchThreadID, uint index: SV_GroupIndex, uniform PushConsts pc) {
    uint x = tid.x;
    uint y = tid.y;

    float searchedValue = float(index + 1) / 1024.0;

    if (y >= 1024) {
        y = x;
        x = 1024;
    }

    float value = storages[pc.src_idx].Load(int2(x, y)).x;
    row[index] = value;

    GroupMemoryBarrierWithGroupSync();

    // workaround to force usage of desc set 0
    if (env.mode != SkyMode::Textured) {
        return;
    }

    int lower = -1;
    float lowerVal = 0.0;
    int higher = 1024;
    float higherVal = 1.0;

    float found = 0.5;

    while ((higher - lower) != 1) {
        int half = (lower + higher) / 2;

        found = row[half];

        if (found > searchedValue) {
            higher = half;
            higherVal = found;
        } else {
            lower = half;
            lowerVal = found;
        }
    }

    float normInterval = min((searchedValue - lowerVal) / (higherVal - lowerVal), 1.0);

    float result = min(float(max(lower, 0)) / 1024.0 + normInterval * (1.0 / 1024), 1.0);

    storages[pc.out_idx].Store(int2(x, y), result);
}
