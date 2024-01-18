#version 460 core
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_ray_tracing_position_fetch : enable
#pragma shader_stage(closest)

layout(location = 0) rayPayloadInEXT struct {
    bool isMiss;
    float dist;
    vec3 normal;
} payload;


void main () {
    payload.isMiss = false;
    payload.dist = gl_HitTEXT;

    vec3 vertPos0 = gl_ObjectToWorldEXT * vec4(gl_HitTriangleVertexPositionsEXT[0], 0.0);
    vec3 vertPos1 = gl_ObjectToWorldEXT * vec4(gl_HitTriangleVertexPositionsEXT[1], 0.0);
    vec3 vertPos2 = gl_ObjectToWorldEXT * vec4(gl_HitTriangleVertexPositionsEXT[2], 0.0);

    vec3 normal = cross(vertPos1 - vertPos0, vertPos2 - vertPos0);

    payload.normal = normalize(normal);
}