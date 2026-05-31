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

float signNotZero(in float k) {
    return (k >= 0.f) ? 1.f : -1.f;
}

float2 signNotZero(in float2 v) {
    return float2(signNotZero(v.x), signNotZero(v.y));
}

float2 octEncode(float3 v) {
    float l1norm = abs(v.x) + abs(v.y) + abs(v.z);
    float2 result = v.xy * (1.f / l1norm);
    if (v.z < 0.f) {
        result = (1.f - abs(result.yx)) * signNotZero(result.xy);
    }
    return result;
}

float3 octDecode(float2 o) {
    float3 v = float3(o.x, o.y, 1.f - abs(o.x) - abs(o.y));
    if (v.z < 0.f) {
        v.xy = (1.f - abs(v.yx)) * signNotZero(v.xy);
    }
    return normalize(v);
}

float luminance(float3 v) {
    return dot(v, float3(0.2126f, 0.7152f, 0.0722f));
}

[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 tid: SV_DispatchThreadID, uniform PushConsts pc) {
    uint x = tid.x;
    uint y = tid.y;

    // workaround to force usage of desc set 0
    if (!env.mode == SkyMode::Textured) {
        return;
    }

    int width;
    int height;
    storages[pc.out_idx].GetDimensions(width, height);

    float2 pixel = float2(x, y) / float2(width, height);
    pixel = (pixel - float2(0.5f)) * 2.f;

    float3 dir = octDecode(pixel);

    float3 sky = Toyrender::Sky::sample(dir, images[pc.src_idx], env.rotation);

    storages[pc.out_idx].Store(int2(x, y), luminance(sky));
}