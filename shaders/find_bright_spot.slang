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

float luminance(float3 v) {
    return dot(v, float3(0.2126f, 0.7152f, 0.0722f));
}

groupshared float maxes[1024];
groupshared uint2 coords[1024];

[numthreads(1024, 1, 1)]
[shader("compute")]
void main(uint3 tid: SV_DispatchThreadID, uniform PushConsts pc) {
    int width;
    int height;
    storages[pc.src_idx].GetDimensions(width, height);

    float max = 0.0;
    uint2 maxCoords = uint2(0, 0);

    uint pixel = tid.x;
    while (pixel < (width * height)) {
        int x = pixel % width;
        int y = pixel / width;
        uint2 coords = uint2(x, y);

        float lum = luminance(storages[pc.src_idx].Load(coords).xyz);

        if (lum > max) {
            max = lum;
            maxCoords = coords;
        }

        pixel += 1024;
    }

    maxes[tid.x] = max;
    coords[tid.x] = maxCoords;

    GroupMemoryBarrierWithGroupSync();

    for (uint s = 1024 / 2; s > 0; s >>= 1) {
        if (tid.x < s) {
            if (maxes[tid.x] < maxes[tid.x + s]) {
                maxes[tid.x] = maxes[tid.x + s];
                coords[tid.x] = coords[tid.x + s];
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (tid.x == 0) {
        float2 uv = float2(coords[0].x / float(width), coords[0].y / float(height));

        outBuf[4] = uv.x;
        outBuf[5] = uv.y;
    }
}