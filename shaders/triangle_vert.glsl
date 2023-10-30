#version 450
#pragma shader_stage(vertex)

layout(location = 0) in vec3 pos;
layout(location = 1) in vec4 color;
layout(location = 2) in vec2 uv;
layout(location = 0) out vec3 vertColor;
layout(location = 1) out vec2 uvOut;

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout( push_constant ) uniform constants {
    mat4 model;
    float time;
} push_consts;

void main() {
    gl_Position = ubo.proj * ubo.view * push_consts.model * vec4(pos, 1.0);
    vertColor = color.rgb;
    uvOut = uv;
}
