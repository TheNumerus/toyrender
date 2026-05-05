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

[numthreads(16, 16, 1)]
[shader("compute")]
void horizontal(uint3 tid: SV_DispatchThreadID, uniform PushConsts pc) {
    uint x = tid.x;
    uint y = tid.y;

    // workaround to force usage of desc set 0
    if (!env.mode == SkyMode::Textured) {
        return;
    }

    float sum = 0.0;

    for (int i = 0; i < x; i++) {
        sum += storages[pc.src_idx].Load(int2(i, y)).x;
    }

    storages[pc.out_idx].Store(int2(x, y), sum);
}


[numthreads(1024, 1, 1)]
[shader("compute")]
void vertical(uint3 tid: SV_DispatchThreadID, uniform PushConsts pc) {
    uint x = tid.x;
    uint y = tid.y;

    // workaround to force usage of desc set 0
    if (!env.mode == SkyMode::Textured) {
        return;
    }

    float sum = 0.0;

    for (int i = 0; i < x; i++) {
        sum += storages[pc.src_idx].Load(int2(1023, i)).x;
    }

    storages[pc.src_idx].Store(int2(1024, x), sum);
}


[numthreads(16, 16, 1)]
[shader("compute")]
void pdfNormalize(uint3 tid: SV_DispatchThreadID, uniform PushConsts pc) {
    uint x = tid.x;
    uint y = tid.y;

    // workaround to force usage of desc set 0
    if (!env.mode == SkyMode::Textured) {
        return;
    }

    float value = storages[pc.out_idx].Load(int2(x, y)).x;
    float total = storages[pc.src_idx].Load(int2(1024, 1023)).x;

    storages[pc.out_idx].Store(int2(x, y), value / total);
}

[numthreads(16, 16, 1)]
[shader("compute")]
void cdfNormalize(uint3 tid: SV_DispatchThreadID, uint3 gid: SV_GroupID, uint index: SV_GroupIndex, uniform PushConsts pc) {
    uint x = tid.x;
    uint y = tid.y;

    float totalRow = storages[pc.src_idx].Load(int2(1023, y)).x;
    float total = storages[pc.src_idx].Load(int2(1024, 1023)).x;

    // workaround to force usage of desc set 0
    if (!env.mode == SkyMode::Textured) {
        return;
    }

    float value = storages[pc.src_idx].Load(int2(x, y)).x / totalRow;
    storages[pc.out_idx].Store(int2(x, y), value);

    if (gid.x < 4 && gid.y == 0) {
        float value = storages[pc.src_idx].Load(int2(1024, index + 256 * gid.x)).x / total;
        storages[pc.out_idx].Store(int2(1024, index + 256 * gid.x), value);
    }
}
