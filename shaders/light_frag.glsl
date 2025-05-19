#version 450
#pragma shader_stage(fragment)

#include "common/debug_modes.glsl"
#include "common/defs.glsl"

#define PI 3.141569
#define PHI 1.618033988

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) GLOBAL;
layout(set = 0, binding = 2) VIEW_PROJ;
layout(set = 0, binding = 3) ENV;

layout(set = 1, binding = 0) uniform sampler2D[] gb;

layout(push_constant) uniform constants {
    mat4 model;
} push_consts;

uint pcg_hash(uint i) {
    uint state = i * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

vec3 light(vec3 normal, vec3 light_dir, vec3 shadow)  {
    return max(dot(normal, light_dir) * shadow, 0.0);
}

float spec(vec3 loc, vec3 normal, vec3 pos, vec3 lightPos, float shadow, float power) {
    vec3 view_dir = normalize(pos - loc);
    vec3 ref_dir = view_dir - 2.0 * normal * dot(normal, view_dir);

    return pow(max(dot(ref_dir, normalize(lightPos - pos)), 0.0), power) * shadow;
}

float schlick(float f0, vec3 l, vec3 n) {
    return f0 + (1.0 - f0) * pow(1.0 - dot(l, n), 5.0);
}

vec3 sky_color(vec3 view_dir) {
    float vertical = dot(view_dir, vec3(0.0, 0.0, 1.0));

    vec3 sky = mix(vec3(0.5, 0.5, 0.6), vec3(0.2, 0.4, 0.9), pow(abs(vertical), 0.7));
    vec3 ground = mix(vec3(0.2, 0.2, 0.2), vec3(0.03, 0.02, 0.01), -vertical);

    float factor = smoothstep(0.02, -0.02, vertical);

    return mix(sky, ground, factor);
}

vec4 get_debug_color() {
    if (globals.debug == DEBUG_DIRECT) {
        return vec4(
        texture(gb[3], uv).xyz,
        1.0
        );
    } else if (globals.debug == DEBUG_INDIRECT) {
        return vec4(
        texture(gb[4], uv).xyz,
        1.0
        );
    } else if (globals.debug == DEBUG_TIME) {
        return vec4(
        texture(gb[3], uv).xyz,
        1.0
        );
    } else if (globals.debug == DEBUG_BASE_COLOR) {
        /*return vec4(
            texture(gb[0], uv).xyz,
            1.0
        );*/
    } else if (globals.debug == DEBUG_NORMAL) {
        /*return vec4(
            texture(gb[1], uv).xyz,
            1.0
        );*/
    } else if (globals.debug == DEBUG_DEPTH) {
        return vec4(
        vec3(fract(texture(gb[2], uv).x * 500.0)),
        1.0
        );
    } else if (globals.debug == DEBUG_DIRECT_VARIANCE) {
        return vec4(
        vec3(texture(gb[3], uv).x),
        1.0
        );
    } else if (globals.debug == DEBUG_INDIRECT_VARIANCE) {
        return vec4(
        vec3(texture(gb[4], uv).x),
        1.0
        );
    }

    return vec4(1.0);
}

void main() {
    if (!(globals.debug == DEBUG_NONE || globals.debug == DEBUG_DISOCCLUSION)) {
        outColor = get_debug_color();
        //return;
    }

    vec3 light_dir = normalize(vec3(0.2, -0.5, 1.0));

    float ratio = globals.res_x / globals.res_y;

    vec4 color = texture(gb[0], uv);
    vec3 normal = texture(gb[1], uv).xyz * 2.0 - 1.0;
    float depth = 1.0 - texture(gb[2], uv).x;

    vec4 per_pos = inverse(view_proj.proj[0]) * vec4(uv * 2.0 - 1.0, depth, 1.0);
    per_pos /= per_pos.w;
    vec3 vertPos = (inverse(view_proj.view[0]) *  per_pos).xyz;
    vec3 loc = inverse(view_proj.view[0])[3].xyz;
    vec3 view_dir = normalize(vertPos - loc);

    vec3 direct = texture(gb[3], uv).xyz;
    vec3 indirect = texture(gb[4], uv).xyz;

    vec3 diffuse_dir = light(normalize(normal), light_dir, direct) + indirect;
    vec3 specular = vec3(0.0);
    vec3 lighted = ((diffuse_dir + specular));

    vec3 sky_col = sky_color(view_dir) * pow(2.0, globals.exposure);

    outColor = vec4(
    mix(sky_col, lighted, color.a),
    1.0
    );
}
