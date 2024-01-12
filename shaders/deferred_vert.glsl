#version 450
#pragma shader_stage(vertex)

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 color;
layout(location = 3) in vec2 uv;

layout(location = 0) out vec3 vertPos;
layout(location = 1) out vec3 vertColor;
layout(location = 2) out vec2 uvOut;
layout(location = 3) out vec3 vertNormal;

layout(set = 0, binding = 2) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout( push_constant ) uniform constants {
    mat4 model;
} push_consts;

void main() {
    gl_Position = ubo.proj * ubo.view * push_consts.model * vec4(pos, 1.0);
    vertColor = color.rgb;
    vertPos = (push_consts.model * vec4(pos, 1.0)).xyz;
    vertNormal = normalize(mat3(transpose(inverse(push_consts.model))) * normal);
    uvOut = uv;
}
