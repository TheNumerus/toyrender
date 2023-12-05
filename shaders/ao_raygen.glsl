#version 460 core
#extension GL_EXT_ray_tracing : enable
#pragma shader_stage(raygen)

layout(location = 0) rayPayloadEXT struct {
    bool isMiss;
} payload;

layout(set = 0, binding = 0) uniform Global {
    float exposure;
    float res_x;
    float res_y;
} globals;

layout(set = 0, binding = 1) uniform accelerationStructureEXT tlas;

layout(set = 0, binding = 2, rgb10_a2) uniform image2D rtao_out;

layout(set = 1, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(set = 2, binding = 0) uniform sampler2D[] gb;

uint pcg_hash(uint i) {
    uint state = i * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

void main () {
    payload.isMiss = false;

    vec2 uv = (vec2(gl_LaunchIDEXT.xy) + 0.5) / vec2(globals.res_x, globals.res_y);

    vec3 rand_sphere = normalize(vec3(
        (float(pcg_hash(gl_LaunchIDEXT.x * 1813 + gl_LaunchIDEXT.y * 524) % 255) / 128) - 0.5,
        (float(pcg_hash(gl_LaunchIDEXT.x * 907 + gl_LaunchIDEXT.y * 1541) % 255) / 128) - 0.5,
        (float(pcg_hash(gl_LaunchIDEXT.x * 15801 + gl_LaunchIDEXT.y * 7137) % 255) / 128) - 0.5
    ));

    float depth = texture(gb[2], uv).x;
    vec3 normal = normalize((texture(gb[1], uv).xyz - 0.5) * 2.0) + rand_sphere;

    vec4 per_pos = inverse(ubo.proj) * vec4(uv * 2.0 - 1.0, depth, 1.0);
    per_pos /= per_pos.w;
    vec3 vertPos = (inverse(ubo.view) *  per_pos).xyz;

    traceRayEXT(
        tlas,
        gl_RayFlagsOpaqueEXT,
        0xFF,
        0,
        0,
        0,
        vertPos,
        0.01,
        normal,
        1000.0,
        0
    );

    imageStore(
        rtao_out,
        ivec2(gl_LaunchIDEXT.xy),
        vec4(
            vec3(float(payload.isMiss)),
            1.0
        )
    );
}