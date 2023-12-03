#version 460 core
#extension GL_EXT_ray_tracing : enable
#pragma shader_stage(raygen)

layout(location = 0) rayPayloadEXT struct {
    bool isHit;
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

void main () {
    payload.isHit = true;

    vec2 uv = vec2(gl_LaunchIDEXT.xy) / vec2(globals.res_x, globals.res_y);

    float depth = texture(gb[2], uv).x;

    vec4 per_pos = inverse(ubo.proj) * vec4(uv * 2.0 - 1.0, depth, 1.0);
    per_pos /= per_pos.w;
    vec3 vertPos = (inverse(ubo.view) *  per_pos).xyz;

    //imageStore(rtao_out, ivec2(gl_LaunchIDEXT.xy), vec4(vec3(float(payload.isHit)), 1.0));
    imageStore(
        rtao_out,
        ivec2(gl_LaunchIDEXT.xy),
        vec4(
            vertPos,
            1.0
        )
    );
}