#version 450
#pragma shader_stage(fragment)

layout(location = 0) in vec3 vertPos;
layout(location = 1) in vec3 fragColor;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec3 fragNormal;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out float outDepth;

layout(set = 0, binding = 2) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout( push_constant ) uniform constants {
    mat4 model;
} push_consts;

void main() {
    outColor = vec4(
        fragColor,
        1.0
    );

    outNormal = vec4((fragNormal + 1.0) * 0.5, 1.0);
    outDepth = gl_FragCoord.z;
}
