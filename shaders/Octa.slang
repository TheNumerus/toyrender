module Octa;

float signNotZero(in float k) {
    return (k >= 0.f) ? 1.f : -1.f;
}

float2 signNotZero(in float2 v) {
    return float2(signNotZero(v.x), signNotZero(v.y));
}

namespace Toyrender::Octa {
public float2 encode(float3 v) {
    float l1norm = abs(v.x) + abs(v.y) + abs(v.z);
    float2 result = v.xy * (1.f / l1norm);
    if (v.z < 0.f) {
        result = (1.f - abs(result.yx)) * signNotZero(result.xy);
    }
    return result;
}

public float3 decode(float2 o) {
    float3 v = float3(o.x, o.y, 1.f - abs(o.x) - abs(o.y));
    if (v.z < 0.f) {
        v.xy = (1.f - abs(v.yx)) * signNotZero(v.xy);
    }
    return normalize(v);
}
}