#version 450
#pragma shader_stage(fragment)

layout(location = 0) in vec2 uv;

layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(binding = 1) uniform sampler2D[] gb;

layout( push_constant ) uniform constants {
    mat4 model;
    float time;
} push_consts;

float strength(int i, int j) {
    int b = abs(i - 1) + abs(j - 1);
    if (b == 0) {
        return 1.0;
    } else if (b == 1) {
        return 0.0;
    }
    return -0.25;
}

void main() {
    float ratio = abs(ubo.proj[0][0] / ubo.proj[1][1]);
    float radius = 0.002;
    float offset = 1.0;

    vec3 sum = vec3(0.0);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            vec2 moved_uv = uv + vec2((float(i) - offset) * radius * ratio, (float(j) - offset) * radius);

            vec3 color = texture(gb[0], moved_uv).xyz;
            vec3 normal = texture(gb[1], moved_uv).xyz * 2.0 - 1.0;
            float depth = texture(gb[2], moved_uv).x;

            vec4 per_pos = inverse(ubo.proj) * vec4(moved_uv * 2.0 - 1.0, depth, 1.0);
            per_pos /= per_pos.w;
            vec3 vertPos = (inverse(ubo.view) *  per_pos).xyz;

            vec3 loc = inverse(ubo.view)[3].xyz;
            float fresnel = pow(1.0 - max(dot(normalize(loc - vertPos), normalize(normal)), 0.0), 4.0);

            sum += (color + normal + fresnel + depth * 50000.0) * strength(i, j);
        }
    }

    float vignette = 1.0 - length(((uv - 0.5) * vec2(1.0, ratio)));

    outColor = vec4(
        sum * vignette,
        1.0
    );
}
