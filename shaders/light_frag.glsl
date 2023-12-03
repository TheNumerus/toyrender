#version 450
#pragma shader_stage(fragment)

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform Global {
    float exposure;
    float res_x;
    float res_y;
} globals;

layout(set = 1, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(set = 2, binding = 0) uniform sampler2D[] gb;

layout( push_constant ) uniform constants {
    mat4 model;
    float time;
} push_consts;

uint pcg_hash(uint i) {
    uint state = i * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

vec3 light(vec3 color, vec3 normal, vec3 lightDir)  {
    float d = max(dot(normal, lightDir) * 0.5 + 0.5, 0.01);

    return vec3(d);
}

void main() {
    vec3 sun_dir = normalize(vec3(0.1, 0.1, 1.0));

    float noise = ((float(pcg_hash(uint(gl_FragCoord.x + 1000) * uint(gl_FragCoord.y * 4)) % 255) / 128.0) - 1.0) * 0.001 * globals.exposure;

    float ratio = globals.res_x / globals.res_y;

    vec4 color = texture(gb[0], uv);
    vec3 normal = texture(gb[1], uv).xyz * 2.0 - 1.0;
    float depth = texture(gb[2], uv).x;
    vec4 ao = texture(gb[3], uv);

    vec4 per_pos = inverse(ubo.proj) * vec4(uv * 2.0 - 1.0, depth, 1.0);
    per_pos /= per_pos.w;
    vec3 vertPos = (inverse(ubo.view) *  per_pos).xyz;

    vec3 loc = inverse(ubo.view)[3].xyz;
    float fresnel = pow(1.0 - max(dot(normalize(loc - vertPos), normalize(normal)), 0.0), 6.0) * 5.0;

    vec3 lighted = (light(color.rgb, normalize(normal), sun_dir) + noise + fresnel) * ao.rgb;
    vec3 sky = vec3(0.5, 0.5, 0.5);

    outColor = vec4(
        mix(sky, lighted, color.a) * (1.0 / globals.exposure),
        1.0
    );
}
