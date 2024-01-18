#version 450
#pragma shader_stage(compute)

layout (local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) uniform Global {
    float exposure;
    int debug;
    float res_x;
    float res_y;
    float time;
    int frame_index;
    int half_res;
} globals;

layout(set = 1, binding = 0) uniform sampler2D[] gb;
layout(set = 1, binding = 1, rgb10_a2) uniform image2D atrous[];

layout( push_constant ) uniform constants {
    int level;
} push_consts;

int stride(int level) {
    return int(pow(2, level));
}

float get_weight(int offset) {
    if (offset == 0) {
        return 0.375;
    } else if (abs(offset) == 1) {
        return 0.25;
    } else {
        return 0.0625;
    }
}

float get_depth_gradient(uint x, uint y, sampler2D depth) {
    ivec2 size = imageSize(atrous[0]);

    vec2 uv = vec2((x + 0.5) / size.x, (y + 0.5) / size.y);
    float x_offset = 1.0 / size.x;
    float y_offset = 1.0 / size.y;

    float m = texture(depth, uv).x;
    float t = texture(depth, uv + vec2(0.0, -y_offset)).x;
    float b = texture(depth, uv + vec2(0.0, y_offset)).x;
    float l = texture(depth, uv + vec2(-x_offset, 0.0)).x;
    float r = texture(depth, uv + vec2(x_offset, 0.0)).x;

    return ((t-m)-(m-b)) + ((l-m)-(m-r));
}

float get_depth_weight(float base, float sample_depth, float gradient, float dist) {
    float a = abs(base - sample_depth);
    float b = 1.0 * abs(gradient * dist) + 0.0001;

    return exp(-(a / b));
}

void main() {
    uint x_base = gl_GlobalInvocationID.x;
    uint y_base = gl_GlobalInvocationID.y;

    ivec2 size = imageSize(atrous[0]);

    if (globals.debug == 2) {
        vec4 raw = imageLoad(atrous[1], ivec2(x_base, y_base));
        imageStore(atrous[0], ivec2(x_base, y_base), raw);
        return;
    }

    if (x_base <= size.x && y_base <= size.y) {
        vec4 result = imageLoad(atrous[0], ivec2(x_base, y_base));
        vec4 raw = imageLoad(atrous[1], ivec2(x_base, y_base));

        vec4 sum = vec4(0.0);
        float weight_sum = 0.0;

        vec2 uv = vec2((x_base + 0.5) / size.x, (y_base + 0.5) / size.y);

        vec3 normal_base = normalize(texture(gb[1], uv).xyz * 2.0 - 1.0);
        float depth_base = texture(gb[2], uv).x;
        float gradient = get_depth_gradient(x_base, y_base, gb[2]);

        for (int y = -2; y <= 2; y++) {
            for (int x = -2; x <= 2; x++) {
                int x_final = int(x_base) + x * stride(push_consts.level);
                int y_final = int(y_base) + y * stride(push_consts.level);

                if (x_final >= size.x || x_final < 0 || y_final >= size.y || y_final < 0) {
                    continue;
                }

                vec2 sample_uv = vec2((x_final + 0.5) / size.x, (y_final + 0.5) / size.y);
                vec3 sample_normal = normalize(texture(gb[1], sample_uv).xyz * 2.0 - 1.0);
                float sample_depth = texture(gb[2], sample_uv).x;
                float sample_dist = distance(sample_uv, uv);

                vec4 img_in;
                if (push_consts.level == 0) {
                    img_in = imageLoad(atrous[1], ivec2(x_final, y_final));
                } else {
                    img_in = imageLoad(atrous[0], ivec2(x_final, y_final));
                }

                float normal_weight = pow(max(dot(normal_base, sample_normal), 0.0), 128.0);
                float depth_weight = get_depth_weight(depth_base, sample_depth, gradient, sample_dist);

                float weight = get_weight(x) * get_weight(y) * normal_weight * depth_weight;

                sum += img_in * weight;
                weight_sum += weight;
            }
        }

        vec4 total = vec4((sum / weight_sum).xyz, raw.a);

        imageStore(atrous[0], ivec2(x_base, y_base), total);
    }
}
