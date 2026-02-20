import Common;

[[vk::binding(0, 0)]]
ConstantBuffer<Global> global;

[[vk::binding(0, 1)]]
Sampler2D images[];

[[vk::binding(1, 1)]]
RWTexture2D<float4> storages[];

struct PushConsts {
    int out_idx;
    int depth_idx;
    int sky_idx;
};


[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 tid: SV_DispatchThreadID, uniform PushConsts pc) {
    uint x = tid.x;
    uint y = tid.y;

    int width;
    int height;
    storages[pc.out_idx].GetDimensions(width, height);

    float2 uv = (float2(x, y) + 0.5) / float2(global.res_x, global.res_y);

    if (x <= width && y <= height) {
        float input = images[pc.depth_idx].Sample(uv).x;
        input = fract(input * 32.0);

        storages[pc.out_idx].Store(int2(x, y), float4(float3(input), 1.0));
    }
}
