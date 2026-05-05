import Common;

[[vk::binding(0, 0)]]
ConstantBuffer<Global> global;

[[vk::binding(2, 0)]]
ConstantBuffer<ViewProj> vp;

[[vk::binding(0, 1)]]
Sampler2D images[];

[[vk::binding(1, 1)]]
RWTexture2D<float4> storages[];

struct PushConsts {
    int sample;
    int clear;
    int src_idx;
    int acc_idx;
};

[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 tid: SV_DispatchThreadID, uniform PushConsts pc) {
    uint x = tid.x;
    uint y = tid.y;

    if (x >= global.draw_res_x || y >= global.draw_res_y) {
        return;
    }

    float3 inputNew = storages[pc.src_idx].Load(tid.xy).xyz;
    float3 inputLast = storages[pc.acc_idx].Load(tid.xy).xyz;

    if (pc.clear == 1) {
        storages[pc.acc_idx].Store(tid.xy, float4(inputNew, 1.0));
    } else {
        var pred = isfinite(inputNew);
        if (!pred.x || !pred.y || !pred.z) {
            storages[pc.acc_idx].Store(tid.xy, float4(inputLast, 1.0));
            return;
        }

        float sampleWeight = float(pc.sample) / float(pc.sample + 1);

        storages[pc.acc_idx].Store(tid.xy, float4(lerp(inputNew, inputLast, sampleWeight), 1.0));
    }
}
