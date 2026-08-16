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

[[vk::binding(6, 0)]]
RWStructuredBuffer<float> outBuf;

[[vk::binding(0, 1)]]
Sampler2D images[];

[[vk::binding(1, 1)]]
RWTexture2D<float4> storages[];

struct PushConsts {
    int src_idx;
};

[vk::push_constant]
uniform PushConsts pc;

groupshared float3 values[64];

[numthreads(64, 1, 1)]
[shader("compute")]
void main(uint3 tid: SV_DispatchThreadID, uniform PushConsts pc) {
    int width;
    int height;
    storages[pc.src_idx].GetDimensions(width, height);

    float3 acc = 0.0;

    uint y = tid.x;

    for (int x = 0; x < width - 1; x++) {
        float3 pixel = storages[pc.src_idx].Load(uint2(x, y)).xyz;
        acc += pixel;
    }

    values[tid.x] = acc;

    GroupMemoryBarrierWithGroupSync();

    for (uint s = 64 / 2; s > 0; s >>= 1) {
        if (tid.x < s) {
            values[tid.x] += values[tid.x + s];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (tid.x == 0) {
        float pixels = float((width - 1) * (height - 1));
        outBuf[0] = values[tid.x].x / pixels;
        outBuf[1] = values[tid.x].y / pixels;
        outBuf[2] = values[tid.x].z / pixels;
    }
}