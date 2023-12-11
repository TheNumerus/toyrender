#version 460 core
#extension GL_EXT_ray_tracing : enable
#pragma shader_stage(miss)

layout(location = 0) rayPayloadInEXT struct {
    bool isMiss;
    float dist;
} payload;


void main () {
    payload.isMiss = true;
}