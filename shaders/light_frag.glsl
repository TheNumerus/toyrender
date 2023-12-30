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
    int half_res;
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

vec3 light(vec3 color, vec3 normal, vec3 pos, vec3 lightPos, float shadow)  {
    return color * max(dot(normal, normalize(lightPos - pos)) * shadow, 0.0);
}

float spec(vec3 loc, vec3 normal, vec3 pos, vec3 lightPos, float shadow, float power) {
    vec3 view_dir = normalize(pos - loc);
    vec3 ref_dir = view_dir - 2.0 * normal * dot(normal, view_dir);

    return pow( max(dot(ref_dir, normalize(lightPos - pos)), 0.0), power) * shadow;
}

float gauss(float x, float var) {
    return pow(2.7182818, -pow(x, 2.0) / (2.0 * var));
}

float blurred_ao() {
    float base_depth = texture(gb[2], uv).x;

    if (base_depth == 1.0) {
        return 1.0;
    }

    vec3 base_normal = texture(gb[1], uv).xyz * 2.0 - 1.0;

    vec4 per_pos = inverse(ubo.proj) * vec4(uv * 2.0 - 1.0, base_depth, 1.0);
    per_pos /= per_pos.w;
    vec3 base_pos = (inverse(ubo.view) *  per_pos).xyz;
    vec3 loc = inverse(ubo.view)[3].xyz;
    float vert_dist = distance(base_pos, loc);

    float ao_sample = texture(gb[3], uv).x;

    float sum_weigth = 0.0;
    float sum = ao_sample * sum_weigth;

    float x_offset = 1.0 / globals.res_x;
    float y_offset = 1.0 / globals.res_y;

    float max_dim = float(max(globals.res_x, globals.res_y));

    vec2 offset_scale = (vec2(globals.res_y, globals.res_x) / max_dim) * 0.001;

    int samples = 64;
    float sq_samples = sqrt(float(samples));

    for (int i = 0; i < samples; i++) {
        float sq_i = sqrt(i);
        float x = sq_i * sin((2 * PI * samples * float(i)) / PHI);
        float y = sq_i * cos((2 * PI * samples * float(i)) / PHI);

        vec2 offset = vec2(x, y) * offset_scale;
        vec2 uv_new = uv + offset;

        if (uv_new.x > 1.0 || uv_new.y > 1.0 || uv_new.x < 0.0 || uv_new.y < 0.0) {
            continue;
        }

        vec3 normal = texture(gb[1], uv + offset).xyz * 2.0 - 1.0;

        float depth_s = texture(gb[2], uv + offset).x;

        vec4 per_pos_s = inverse(ubo.proj) * vec4((uv + offset) * 2.0 - 1.0, depth_s, 1.0);
        per_pos_s /= per_pos_s.w;
        vec3 vertPos = (inverse(ubo.view) *  per_pos_s).xyz;

        float dist = distance(vertPos, base_pos);

        float z_min = texture(gb[3], uv + offset).z;

        float max_x = 0.3 * (1.0 / vert_dist);
        float dev = min(max_x, 2.8 / z_min);

        float weight = gauss(dist, dev);
        if (isnan(weight) || dot(base_normal, normal) < 0.9) {
            weight = 0.0;
        }

        float ao_sample = texture(gb[3], uv + offset).x;

        sum += ao_sample * weight;
        sum_weigth += weight;
    }

    return sum / sum_weigth;
}

float schlick(float f0, vec3 l, vec3 n) {
    return f0 + (1.0 - f0) * pow(1.0 - dot(l, n), 5.0);
}

void main() {
    if (globals.debug == 1) {
        float ao_sample = texture(gb[3], uv).x;

        outColor = vec4(
            vec3(ao_sample),
            1.0
        );
        return;
    } else if (globals.debug == 2) {
        outColor = vec4(
            vec3(blurred_ao()),
            1.0
        );
        return;
    } else if (globals.debug == 3) {
        float ref = texture(gb[3], uv).y;

        outColor = vec4(
            vec3(ref),
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
        vec4 color = texture(gb[0], uv);

        outColor = color;
        return;
    }

    vec3 lightPos = vec3(0.0, 0.0, 10.0);

    float noise = ((float(pcg_hash(uint(gl_FragCoord.x + 1000) * uint(gl_FragCoord.y * 4)) % 255) / 128.0) - 1.0) * 0.001 * globals.exposure;

    float ratio = globals.res_x / globals.res_y;

    vec4 color = texture(gb[0], uv);
    vec3 normal = texture(gb[1], uv).xyz * 2.0 - 1.0;
    float depth = texture(gb[2], uv).x;

    vec4 per_pos = inverse(ubo.proj) * vec4(uv * 2.0 - 1.0, depth, 1.0);
    per_pos /= per_pos.w;
    vec3 vertPos = (inverse(ubo.view) *  per_pos).xyz;

    float shadow = texture(gb[3], uv).a;

    vec3 loc = inverse(ubo.view)[3].xyz;

    float ref_sample = texture(gb[3], uv).y;
    float ao = blurred_ao();
    float fresnel = schlick(0.3, normalize(loc - vertPos), normalize(normal)) * ref_sample * ao;

    vec3 color_override = vec3(0.9);

    vec3 sky = vec3(0.5, 0.6, 0.9);

    vec3 diffuse_dir = light(color_override, normalize(normal), vertPos, lightPos, shadow) * 0.3 * (1.0 - (1.0 - ao) * 0.1);
    vec3 specular = fresnel * sky + spec(loc, normalize(normal), vertPos, lightPos, shadow, 200.0);
    vec3 ambient = sky * ao * 0.5 * color_override;
    vec3 lighted = ((diffuse_dir + specular + ambient)) + noise;

    outColor = vec4(
        mix(sky, lighted, color.a) * (1.0 / globals.exposure),
        1.0
    );
}
