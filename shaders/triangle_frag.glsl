#version 450
#pragma shader_stage(fragment)

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

layout( push_constant ) uniform constants {
    float time;
} push_consts;

void main() {
    float time = push_consts.time;

    outColor = vec4(
        abs(fragColor.r + sin(time)),
        abs(fragColor.g + sin(time)),
        abs(fragColor.b + sin(time)),
        1.0
    );
}
