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

[[vk::binding(2, 1)]]
RWStructuredBuffer<float> scratch;

struct PushConsts {
    int src_idx;
    int count;
};

[vk::push_constant]
uniform PushConsts pc;

float luminance(float3 v) {
    return dot(v, float3(0.2126f, 0.7152f, 0.0722f));
}

groupshared float shared[1024 * 3];

[numthreads(1024, 1, 1)]
[shader("compute")]
void mainInit(uint3 tid: SV_DispatchThreadID, uniform PushConsts pc) {
    int width;
    int height;
    storages[pc.src_idx].GetDimensions(width, height);

    float max = 0.0;

    uint pixel = tid.x + tid.y * 1024;
    int x = pixel % width;
    int y = pixel / width;
    uint2 pixCoords = uint2(x, y);

    float lum = luminance(storages[pc.src_idx].Load(pixCoords).xyz);

    if (pixel >= pc.count) {
        lum = 0.0;
    }

    shared[tid.x] = lum;
    shared[tid.x + 1024] = pixCoords.x / float(width);
    shared[tid.x + 1024 * 2] = pixCoords.y / float(height);

    GroupMemoryBarrierWithGroupSync();

    for (uint s = 1024 / 2; s > 0; s >>= 1) {
        if (tid.x < s) {
            if (shared[tid.x] < shared[tid.x + s]) {
                shared[tid.x] = shared[tid.x + s];
                shared[tid.x + 1024] = shared[tid.x + s + 1024];
                shared[tid.x + 1024 * 2] = shared[tid.x + s + 1024 * 2];
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (tid.x == 0) {
        scratch[tid.y * 3] = shared[0];
        scratch[tid.y * 3 + 1] = shared[1024];
        scratch[tid.y * 3 + 2] = shared[2048];
    }

    if (tid.x == 0 && pc.count < 1024) {
        outBuf[4] = shared[1024];
        outBuf[5] = shared[2048];
    }
}

[numthreads(1024, 1, 1)]
[shader("compute")]
void mainSuccessive(uint3 tid: SV_DispatchThreadID, uniform PushConsts pc) {
    int width;
    int height;
    storages[pc.src_idx].GetDimensions(width, height);

    float max = 0.0;

    uint pixel = tid.x + tid.y * 1024;

    float lum = scratch[pixel * 3];

    if (pixel >= pc.count) {
        lum = 0.0;
    }

    float2 pixCoords;
    pixCoords.x = scratch[pixel * 3 + 1];
    pixCoords.y = scratch[pixel * 3 + 2];

    shared[tid.x] = lum;
    shared[tid.x + 1024] = pixCoords.x;
    shared[tid.x + 1024 * 2] = pixCoords.y;

    GroupMemoryBarrierWithGroupSync();

    for (uint s = 1024 / 2; s > 0; s >>= 1) {
        if (tid.x < s) {
            if (shared[tid.x] < shared[tid.x + s]) {
                shared[tid.x] = shared[tid.x + s];
                shared[tid.x + 1024] = shared[tid.x + s + 1024];
                shared[tid.x + 1024 * 2] = shared[tid.x + s + 1024 * 2];
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (tid.x == 0) {
        scratch[tid.y * 3] = shared[0];
        scratch[tid.y * 3 + 1] = shared[1024];
        scratch[tid.y * 3 + 2] = shared[2048];
    }

    if (tid.x == 0 && pc.count < 1024) {
        outBuf[4] = shared[1024];
        outBuf[5] = shared[2048];
    }
}