#version 450
#pragma shader_stage(fragment)

#define PI 3.141569
#define PHI 1.618033988

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform Global {
    float exposure;
    int debug;
    float res_x;
    float res_y;
    float time;
    int frame_index;
    int half_res;
} globals;

layout(set = 0, binding = 2) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(set = 1, binding = 0) uniform sampler2D[] gb;

layout( push_constant ) uniform constants {
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

    return pow( max(dot(ref_dir, normalize(lightPos - pos)), 0.0), power) * shadow;
}

float schlick(float f0, vec3 l, vec3 n) {
    return f0 + (1.0 - f0) * pow(1.0 - dot(l, n), 5.0);
}

void main() {
    if (globals.debug == 1) {
        vec3 indir = texture(gb[3], uv).xyz;

        outColor = vec4(
            indir,
            1.0
        );
        return;
    } else if (globals.debug == 2) {
        outColor = vec4(
            vec3(texture(gb[3], uv).xyz),
            1.0
        );
        return;
    } else if (globals.debug == 3) {
        outColor = vec4(
            vec3(texture(gb[3], uv).y),
            1.0
        );
        return;
    } else if (globals.debug == 4) {
        float shadow = texture(gb[3], uv).w;

        outColor = vec4(
            vec3(shadow),
            1.0
        );
        return;
    } else if (globals.debug == 5) {
        vec3 normal = texture(gb[1], uv).xyz;

        outColor = vec4(
            normal,
            1.0
        );
        return;
    } else if (globals.debug == 6) {
        float depth = texture(gb[2], uv).x;

        outColor = vec4(
            vec3(depth),
            1.0
        );
        return;
    } else if (globals.debug == 7) {
        float color = texture(gb[3], uv).z;

        outColor = vec4(vec3(color), 1.0);
        return;
    }

    vec3 light_dir = normalize(vec3(0.2, -0.5, 1.0));

    float noise = ((float(pcg_hash(uint(globals.frame_index) + uint(gl_FragCoord.x + 1000) * uint(gl_FragCoord.y * 4)) % 255) / 128.0) - 0.5) * 0.001 * globals.exposure;

    float ratio = globals.res_x / globals.res_y;

    vec4 color = texture(gb[0], uv);
    vec3 normal = texture(gb[1], uv).xyz * 2.0 - 1.0;
    float depth = texture(gb[2], uv).x;

    vec4 per_pos = inverse(ubo.proj) * vec4(uv * 2.0 - 1.0, depth, 1.0);
    per_pos /= per_pos.w;
    vec3 vertPos = (inverse(ubo.view) *  per_pos).xyz;
    vec3 loc = inverse(ubo.view)[3].xyz;
    vec3 view_dir = normalize(vertPos - loc);

    vec3 indir = texture(gb[3], uv).xyz;
    float shadow = texture(gb[3], uv).a;

    vec3 color_override = vec3(0.9);

    vec3 sky = vec3(0.4, 0.6, 0.9);
    vec3 sun = vec3(0.9, 0.8, 0.7);

    vec3 diffuse_dir = light(normalize(normal), light_dir, shadow * sun) + indir;
    vec3 specular = vec3(0.0);
    vec3 lighted = ((diffuse_dir + specular)) + noise;

    vec3 sky_col = dot(view_dir, vec3(0.0, 0.0, 1.0)) * sky + noise;

    outColor = vec4(
        mix(sky_col, lighted, color.a) * (1.0 / globals.exposure),
        1.0
    );
}
