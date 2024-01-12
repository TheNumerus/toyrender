#version 450
#pragma shader_stage(vertex)

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 color;
layout(location = 3) in vec2 uv;

layout(location = 0) out vec2 uvOut;

layout(set = 0, binding = 2) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout( push_constant ) uniform constants {
    mat4 model;
} push_consts;

void main() {
    gl_Position = vec4(pos, 1.0);
    uvOut = uv;
}
