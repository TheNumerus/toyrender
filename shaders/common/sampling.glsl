#define PI 3.141569

vec3 rand_hemi(vec2 rand) {
    float sq_x = sqrt(rand.x);

    float x = sq_x * cos(2.0 * PI * rand.y);
    float y = sq_x * sin(2.0 * PI * rand.y);
    float z = sqrt(1.0 - rand.x);

    return vec3(x, y, z);
}

vec3 rand_sphere(vec2 rand) {
    float a = 1 - 2 * rand.x;
    float b = sqrt(1 - a * a);
    float phi = 2 * PI * rand.y;
    float x = b * cos(phi);
    float y = b * sin(phi);
    float z = a;
    return normalize(vec3(x, y, z));
}

vec3 rand_cone(vec2 rand, float angle) {
    float cos_angle = (1.0 - rand.x) + rand.x * angle;
    float sin_angle = sqrt(1.0 - cos_angle * cos_angle);
    float phi = rand.y * 2.0 * PI;
    float x = cos(phi) * sin_angle;
    float y = sin(phi) * sin_angle;
    float z = cos_angle;

    return vec3(x, y, z);
}