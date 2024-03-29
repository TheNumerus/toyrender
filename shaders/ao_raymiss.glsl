#version 460 core
#extension GL_EXT_ray_tracing : enable
#pragma shader_stage(miss)

layout(location = 0) rayPayloadInEXT struct {
    bool isMiss;
    float dist;
    vec3 normal;
    vec3 color;
} payload;


void main () {
    payload.isMiss = true;
    payload.color = vec3(0.0);
}