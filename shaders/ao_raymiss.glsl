#version 460 core
#extension GL_EXT_ray_tracing : enable
#pragma shader_stage(miss)

layout(location = 0) rayPayloadEXT struct {
    bool isHit;
} payload;


void main () {
    payload.isHit = false;
}