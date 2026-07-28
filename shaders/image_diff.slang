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
    int src1_idx;
    int src2_idx;
};

[vk::push_constant]
uniform PushConsts pc;

float luminance(float3 v) {
    return dot(v, float3(0.2126f, 0.7152f, 0.0722f));
}

[numthreads(1, 1, 1)]
[shader("compute")]
void main(uint3 tid: SV_DispatchThreadID, uniform PushConsts pc) {
    int width1;
    int height1;
    storages[pc.src1_idx].GetDimensions(width1, height1);

    int width2;
    int height2;
    storages[pc.src2_idx].GetDimensions(width2, height2);

    int width = min(width1, width2);
    int height = min(height1, height2);

    float acc = 0.0;
    float minVal = float.maxValue;
    float maxVal = float.minValue;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float lum = luminance(storages[pc.src1_idx].Load(uint2(x, y)).xyz);
            minVal = min(minVal, lum);
            maxVal = max(maxVal, lum);
            acc += lum;
        }
    }

    float variance = 0.0;

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float pixel = luminance(storages[pc.src1_idx].Load(uint2(x, y)).xyz);

            pixel -= luminance(storages[pc.src1_idx].Load(uint2(x - 1, y)).xyz) * 0.25;
            pixel -= luminance(storages[pc.src1_idx].Load(uint2(x + 1, y)).xyz) * 0.25;
            pixel -= luminance(storages[pc.src1_idx].Load(uint2(x, y - 1)).xyz) * 0.25;
            pixel -= luminance(storages[pc.src1_idx].Load(uint2(x, y + 1)).xyz) * 0.25;

            variance += abs(pixel);
        }
    }

    outBuf[0] = acc / float(width * height);
    outBuf[1] = minVal;
    outBuf[2] = maxVal;
    outBuf[3] = variance / float(width * height);
}