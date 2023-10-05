#version 450
#pragma shader_stage(vertex)

layout(location = 0) in vec3 pos;
layout(location = 1) in vec4 color;
layout(location = 2) in vec2 uv;
layout(location = 0) out vec3 fragColor;

layout( push_constant ) uniform constants {
    float time;
} push_consts;

void main() {
    gl_Position = vec4(pos * 1.8, 1.0);
    fragColor = color.rgb;
}
