#version 450
#pragma shader_stage(fragment)

layout(location = 0) in vec3 vertPos;
layout(location = 1) in vec3 fragColor;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec3 fragNormal;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout( push_constant ) uniform constants {
    mat4 model;
    float time;
} push_consts;

void main() {
    vec3 lightPos = vec3(0.0, 0.0, 4.0);

    vec3 delta = normalize(lightPos - vertPos);
    float shade = max(dot(delta, normalize(fragNormal)), 0.0);

    vec3 loc = inverse(ubo.view)[3].xyz;
    float fresnel = pow(1.0 - max(dot(normalize(loc - vertPos), normalize(fragNormal)), 0.0), 4.0);

    outColor = vec4(
        (fragColor) * (vec3(shade) + 0.1) + vec3(fresnel) * 0.1,
        1.0
    );
}
