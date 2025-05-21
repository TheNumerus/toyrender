#version 450
#pragma shader_stage(fragment)
#extension GL_EXT_nonuniform_qualifier : enable

#include "common/defs.glsl"

layout(location = 0) in vec3 vertPos;
layout(location = 1) in vec3 fragColor;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec3 fragNormal;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec4 outNormal;

layout(set = 0, binding = 2) VIEW_PROJ;

layout(push_constant) uniform constants {
    mat4 model;
} push_consts;

void main() {
    outColor = vec4(
    fragColor,
    1.0
    );

    outNormal = vec4((fragNormal + 1.0) * 0.5, 1.0);
}
