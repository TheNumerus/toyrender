module Sky;

#define PI 3.14159265359

namespace Toyrender::Sky {
public float3 sample_distorted(float3 view_dir, Sampler2D image) {
    float2 dir = normalize(view_dir.xy);

    float u = 0.0;
    if (dir.y >= 0.0) {
        u = acos(dir.x);
    } else {
        u = -acos(dir.x);
    }

    u = (u + PI) / (2.0 * PI);
    if (isnan(u) || isinf(u)) {
        u = 0.0;
    }

    float v = ((asin(-view_dir.z) / (PI / 2.0)) * 0.5) + 0.5;

    if (isnan(v) || isinf(v)) {
        v = 0.0;
    }

    u = clamp(u, 0.0, 1.0);
    v = clamp(v, 0.0, 1.0);
    v = (sqrt(abs((v - 0.5) * 2.0)) * sign(v - 0.5)) * 0.5 + 0.5;

    return image.Sample(float2(u, v)).xyz;
}

public float3 sample(float3 view_dir, Sampler2D image, float rotation) {
    float2 dir = normalize(view_dir.xy);

    float u = 0.0;
    if (dir.y >= 0.0) {
        u = acos(dir.x);
    } else {
        u = -acos(dir.x);
    }

    u = (u + PI) / (2.0 * PI);
    if (isnan(u) || isinf(u)) {
        u = 0.0;
    }

    float v = ((asin(-view_dir.z) / (PI / 2.0)) * 0.5) + 0.5;

    if (isnan(v) || isinf(v)) {
        v = 0.0;
    }

    u = clamp(u, 0.0, 1.0);
    v = clamp(v, 0.0, 1.0);

    u = fract(u + rotation);

    return image.Sample(float2(u, v)).xyz;
}
}

