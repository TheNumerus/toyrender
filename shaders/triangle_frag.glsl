#version 450
#pragma shader_stage(fragment)

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout( push_constant ) uniform constants {
    mat4 model;
    float time;
} push_consts;

void main() {
    float time = push_consts.time;

    outColor = vec4(
        abs(uv.x + sin(time)),
        abs(uv.y + sin(time)),
        abs(fragColor.b + sin(time)),
        1.0
    );
}
