module Sampling;

#define PI 3.141569

public float3 reflect(float3 incoming, float3 normal) {
    return 2.0 * dot(normal, incoming) * normal - incoming;
}

public float3 rand_hemi(float2 rand) {
    float sq_x = sqrt(rand.x);

    float x = sq_x * cos(2.0 * PI * rand.y);
    float y = sq_x * sin(2.0 * PI * rand.y);
    float z = sqrt(1.0 - rand.x);

    return float3(x, y, z);
}

public float3 rand_sphere(float2 rand) {
    float a = 1 - 2 * rand.x;
    float b = sqrt(1 - a * a);
    float phi = 2 * PI * rand.y;
    float x = b * cos(phi);
    float y = b * sin(phi);
    float z = a;
    return normalize(float3(x, y, z));
}

public float3 rand_cone(float2 rand, float angle) {
    float cos_angle = (1.0 - rand.x) + rand.x * angle;
    float sin_angle = sqrt(1.0 - cos_angle * cos_angle);
    float phi = rand.y * 2.0 * PI;
    float x = cos(phi) * sin_angle;
    float y = sin(phi) * sin_angle;
    float z = cos_angle;

    return float3(x, y, z);
}

public float3 phong(float2 rand, float exponent) {
    float cos_theta = pow(1 - rand.x, 1 / (2.0 + exponent));
    float sin_theta = sqrt(1 - cos_theta * cos_theta);
    float phi = 2.0 * PI * rand.y;
    float x = cos(phi) * sin_theta;
    float y = sin(phi) * sin_theta;
    float z = cos_theta;

    return float3(x, y, z);
}

public float3 ggx(float2 rand, float roughness) {
    float cos_theta = sqrt((1.0 - rand.x) / ((roughness * roughness - 1.0) * rand.x + 1.0));
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    float phi = 2.0 * PI * rand.y;
    float x = cos(phi) * sin_theta;
    float y = sin(phi) * sin_theta;
    float z = cos_theta;

    var normal = normalize(float3(x, y, z));

    return normal;
}