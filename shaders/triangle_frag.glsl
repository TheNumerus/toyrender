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
    outColor = vec4(
        vertPos,
        1.0
    );
}
