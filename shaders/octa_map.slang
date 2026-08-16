import Common;
import Sky;
import Octa;

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

float luminance(float3 v) {
    return dot(v, float3(0.2126f, 0.7152f, 0.0722f));
}

[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 tid: SV_DispatchThreadID, uniform PushConsts pc) {
    uint x = tid.x;
    uint y = tid.y;

    int width;
    int height;
    storages[pc.out_idx].GetDimensions(width, height);

    float2 pixel = float2(x, y) / float2(width, height);
    float3 dir = Toyrender::Octa::rand(pixel);

    float3 sky = Toyrender::Sky::sample(dir, images[pc.src_idx], env.rotation);

    storages[pc.out_idx].Store(int2(x, y), luminance(sky));
}