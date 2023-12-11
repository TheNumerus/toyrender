#version 460 core
#extension GL_EXT_ray_tracing : enable
#pragma shader_stage(raygen)

#define PI 3.141569

layout(location = 0) rayPayloadEXT struct {
    bool isMiss;
    float dist;
} payload;

layout(set = 0, binding = 0) uniform Global {
    float exposure;
    float res_x;
    float res_y;
    float time;
} globals;

layout(set = 0, binding = 1) uniform accelerationStructureEXT tlas;

layout(set = 0, binding = 2, rgb10_a2) uniform image2D rtao_out;

layout(set = 1, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(set = 2, binding = 0) uniform sampler2D[] gb;

layout( push_constant ) uniform constants {
    int samples;
} push_consts;

uint pcg_hash(uint i) {
    uint state = i * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

void compute_default_basis(const vec3 normal, out vec3 x, out vec3 y)
{
    vec3        z  = normal;
    const float yz = -z.y * z.z;
    y = normalize(((abs(z.z) > 0.99999f) ? vec3(-z.x * z.y, 1.0f - z.y * z.y, yz) : vec3(-z.x * z.z, yz, 1.0f - z.z * z.z)));

    x = cross(y, z);
}

vec3 rand_hemi(vec2 rand) {
    float sq_x = sqrt(rand.x);

    float x = sq_x * cos(2.0 * PI * rand.y);
    float y = sq_x * sin(2.0 * PI * rand.y);
    float z = sqrt(1.0 - rand.x);

    return vec3(x, y, z);
}

void main () {
    payload.isMiss = false;

    vec2 uv = (vec2(gl_LaunchIDEXT.xy) + 0.5) / vec2(globals.res_x, globals.res_y);

    float depth = texture(gb[2], uv).x;

    if (depth == 1.0) {
        return;
    }

    vec3 normal = normalize((texture(gb[1], uv).xyz - 0.5) * 2.0);

    vec4 per_pos = inverse(ubo.proj) * vec4(uv * 2.0 - 1.0, depth, 1.0);
    per_pos /= per_pos.w;
    vec3 vertPos = (inverse(ubo.view) * per_pos).xyz;

    vec3 loc = inverse(ubo.view)[3].xyz;
    vec3 view_dir = normalize(vertPos - loc);

    float sum = 0.0;
    float dist = 10.0;

    int samples = push_consts.samples;

    vec3 tangent, bitangent;
    compute_default_basis(normal, tangent, bitangent);

    for (int i = 0; i < samples; i++) {
        vec3 hemi = rand_hemi(
            vec2(
                float((pcg_hash((2 * i + (int(globals.time * 137.0))) * (gl_LaunchIDEXT.x + gl_LaunchIDEXT.y * uint(globals.res_x)))) % 1024) / 1024.0,
                float((pcg_hash((2 * i + 1 + (int(globals.time * 137.0))) * (gl_LaunchIDEXT.x + gl_LaunchIDEXT.y * uint(globals.res_x)))) % 1024) / 1024.0
            )
        );

        vec3 dir = hemi.x * tangent + hemi.y * bitangent + hemi.z * normal;

        traceRayEXT(
            tlas,
            gl_RayFlagsOpaqueEXT,
            0xFF,
            0,
            0,
            0,
            vertPos,
            0.01,
            dir,
            1000.0,
            0
        );

        sum += float(payload.isMiss);
        dist = min(payload.dist, dist);
    }

    payload.isMiss = false;

    vec3 ref_dir = view_dir - 2.0 * normal * dot(normal, view_dir);

    traceRayEXT(
        tlas,
        gl_RayFlagsOpaqueEXT,
        0xFF,
        0,
        0,
        0,
        vertPos,
        0.01,
        ref_dir,
        1000.0,
        0
    );

    float refl = float(payload.isMiss);

    imageStore(
        rtao_out,
        ivec2(gl_LaunchIDEXT.xy),
        vec4(
            sum / float(samples),
            refl,
            dist,
            1.0
        )
    );
}