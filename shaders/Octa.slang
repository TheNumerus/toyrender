import Common;

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

public float3 rand(float2 u) {
    float2 u = 2.0 * u - 1.0;
    float d = 1.0 - (abs(u.x) + abs(u.y));
    float r = 1.0 - abs(d);

    float phi = (r == 0.0) ? 0.0 : (PI / 4.0) * ((abs(u.y) - abs(u.x)) / r + 1.0);
    float f = r * sqrt(2.0 - r * r);
    float x = f * sign(u.x) * cos(phi);
    float y = f * sign(u.y) * sin(phi);
    float z = sign(d) * (1.0 - r * r);
    return normalize(float3(x, y, z));
}
}