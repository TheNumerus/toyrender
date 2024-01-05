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
    int debug;
    float res_x;
    float res_y;
    float time;
    uint frame_index;
    int half_res;
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

vec3 rand_sphere(vec2 rand) {
    float a = 1 - 2 * rand.x;
    float b = sqrt(1 - a * a);
    float phi = 2 * PI * rand.y;
    float x = b * cos(phi);
    float y = b * sin(phi);
    float z = a;
    return normalize(vec3(x, y, z));
}

void main () {
    payload.isMiss = false;

    vec2 uv = (vec2(gl_LaunchIDEXT.xy) + 0.5) / vec2(globals.res_x, globals.res_y);
    if (globals.half_res == 1) {
        uv = (vec2(gl_LaunchIDEXT.xy) + 0.5) / vec2(globals.res_x/2, globals.res_y/2);
    }

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
    float dis = distance(loc, vertPos);
    vec3 bias = dis * normal * 0.001;

    float sum = 0.0;
    float dist = 10.0;

    int samples = push_consts.samples;

    vec3 tangent, bitangent;
    compute_default_basis(normal, tangent, bitangent);

    for (int i = 0; i < samples; i++) {
        vec3 hemi = rand_hemi(
            vec2(
                float((pcg_hash((2 * i + int(globals.frame_index)) * (gl_LaunchIDEXT.x + gl_LaunchIDEXT.y * uint(globals.res_x)))) % 1024) / 1024.0,
                float((pcg_hash((2 * i + 1 + int(globals.frame_index)) * (gl_LaunchIDEXT.x + gl_LaunchIDEXT.y * uint(globals.res_x)))) % 1024) / 1024.0
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
            vertPos + bias,
            0.0001,
            dir,
            5.0,
            0
        );

        sum += float(payload.isMiss);

        if (!payload.isMiss) {
            dist = min(payload.dist, dist);
        }
    }

    payload.isMiss = false;

    vec3 ref_dir = view_dir - 2.0 * normal * dot(normal, view_dir);

    vec3 sphere = rand_sphere(
        vec2(
        float((pcg_hash((2 + int(globals.frame_index)) * (gl_LaunchIDEXT.x + gl_LaunchIDEXT.y * uint(globals.res_x)))) % 1024) / 1024.0,
        float((pcg_hash((2 + 1 + int(globals.frame_index)) * (gl_LaunchIDEXT.x + gl_LaunchIDEXT.y * uint(globals.res_x)))) % 1024) / 1024.0
        )
    ) * 0.1;

    ref_dir = normalize(ref_dir + sphere * dot(view_dir, normal));

    traceRayEXT(
        tlas,
        gl_RayFlagsOpaqueEXT,
        0xFF,
        0,
        0,
        0,
        vertPos + bias,
        0.001,
        ref_dir,
        1000.0,
        0
    );

    float refl = float(payload.isMiss);

    payload.isMiss = false;

    vec3 ray_shadow_dir = normalize(vec3(0.2, -0.5, 1.0) + sphere * 0.1);

    if (dot(ray_shadow_dir, normal) > 0.0) {
        payload.isMiss = true;
        traceRayEXT(
            tlas,
            gl_RayFlagsOpaqueEXT,
            0xFF,
            0,
            0,
            0,
            vertPos + bias,
            0.001,
            ray_shadow_dir,
            1000.0,
            0
        );
    }

    float shadow = float(payload.isMiss);

    imageStore(
        rtao_out,
        ivec2(gl_LaunchIDEXT.xy),
        vec4(
            sum / float(samples),
            refl,
            dist,
            shadow
        )
    );
}