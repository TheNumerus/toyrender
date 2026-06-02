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
    int src_idx;
    int out_idx;
};

[vk::push_constant]
uniform PushConsts pc;

float to_linear_depth(float depth) {
    return vp.near * vp.far / (vp.far + depth * (vp.near - vp.far));
}

float get_depth_gradient(uint3 localid) {
    float t = to_linear_depth(depths[localid.y][localid.x + 1]);

    float l = to_linear_depth(depths[localid.y + 1][localid.x]);
    float m = to_linear_depth(depths[localid.y + 1][localid.x + 1]);
    float r = to_linear_depth(depths[localid.y + 1][localid.x + 2]);

    float b = to_linear_depth(depths[localid.y + 2][localid.x + 1]);

    return length(float2(((l - m) + (m - r)) / 2.0, ((t - m) + (m - b)) / 2.0));
}

groupshared float depths[18][18];

void populateSharedDepths(uint3 localid, uint3 tid) {
    depths[localid.y + 1][localid.x + 1] = storages[pc.src_idx].Load(tid.xy).x;

    // this is 4x16 continuous threads, which should be exacly 1 or 2 wavefronts
    if (localid.y < 4) {
        uint2 start = tid.xy - localid.xy;

        // 0-15
        uint posInSequence = localid.x;

        uint vertical = localid.y & 1;
        uint horizontal = 1 - vertical;

        uint line = (localid.y >> 1) & 1;

        let offset = int2(posInSequence * horizontal + vertical * (17 * line - 1), posInSequence * vertical + horizontal * (17 * line - 1));

        depths[offset.y + 1][offset.x + 1] = storages[pc.src_idx].Load(start + offset).x;
    }

    GroupMemoryBarrierWithGroupSync();
}

[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 tid: SV_DispatchThreadID, uint3 localid: SV_GroupThreadID) {
    uint x_base = tid.x;
    uint y_base = tid.y;

    int2 size = int2(global.res_x, global.res_y);
    int2 drawSize = int2(global.draw_res_x, global.draw_res_y);

    populateSharedDepths(localid, tid);

    float gradient = get_depth_gradient(localid);

    storages[pc.out_idx].Store(int2(x_base, y_base), gradient);
}
