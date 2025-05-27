#version 450
#pragma shader_stage(fragment)
#extension GL_EXT_nonuniform_qualifier : enable

#include "common/debug_modes.glsl"
#include "common/defs.glsl"

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) GLOBAL;
layout(set = 0, binding = 2) VIEW_PROJ;

layout(set = 1, binding = 0) uniform sampler2D images[];

layout(push_constant) uniform constants {
    int src;
} push_consts;

float luminance(vec3 v) {
    return dot(v, vec3(0.2126f, 0.7152f, 0.0722f));
}

vec3 change_luminance(vec3 c_in, float l_out) {
    float l_in = luminance(c_in);
    return c_in * (l_out / l_in);
}

vec3 reinhard_extended_luminance(vec3 v, float max_white_l) {
    float l_old = luminance(v);
    float numerator = l_old * (1.0f + (l_old / (max_white_l * max_white_l)));
    float l_new = numerator / (1.0f + l_old);
    return change_luminance(v, l_new);
}

vec3 abberation(int i) {
    float scale = (float(i) - 3.0) * 0.5;

    return vec3(
    1.0 - min(abs(scale + 1.0), 1.0),
    1.0 - min(abs(scale + 0.0), 1.0),
    1.0 - min(abs(scale - 1.0), 1.0)
    );
}

// Mean error^2: 3.6705141e-06
vec3 agxDefaultContrastApprox(vec3 x) {
    vec3 x2 = x * x;
    vec3 x4 = x2 * x2;

    return + 15.5     * x4 * x2
    - 40.14    * x4 * x
    + 31.96    * x4
    - 6.868    * x2 * x
    + 0.4298   * x2
    + 0.1191   * x
    - 0.00232;
}

vec3 agx(vec3 val) {
    const mat3 agx_mat = mat3(
    0.842479062253094, 0.0423282422610123, 0.0423756549057051,
    0.0784335999999992, 0.878468636469772, 0.0784336,
    0.0792237451477643, 0.0791661274605434, 0.879142973793104);

    const float min_ev = -12.47393f;
    const float max_ev = 4.026069f;

    // Input transform (inset)
    val = agx_mat * val;

    // Log2 space encoding
    val = clamp(log2(val), min_ev, max_ev);
    val = (val - min_ev) / (max_ev - min_ev);

    // Apply sigmoid function approximation
    val = agxDefaultContrastApprox(val);

    return val;
}

vec3 agxEotf(vec3 val) {
    const mat3 agx_mat_inv = mat3(
    1.19687900512017, -0.0528968517574562, -0.0529716355144438,
    -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
    -0.0990297440797205, -0.0989611768448433, 1.15107367264116);

    // Inverse input transform (outset)
    val = agx_mat_inv * val;

    // sRGB IEC 61966-2-1 2.2 Exponent Reference EOTF Display
    // NOTE: We're linearizing the output here. Comment/adjust when
    // *not* using a sRGB render target
    val = pow(val, vec3(2.2));

    return val;
}

vec3 agxLook(vec3 val) {
    // Default
    vec3 offset = vec3(0.0);
    vec3 slope = vec3(1.0);
    vec3 power = vec3(1.0);
    float sat = 1.0;

    val = pow(val * slope + offset, power);

    const vec3 lw = vec3(0.2126, 0.7152, 0.0722);
    float luma = dot(val, lw);

    return luma + sat * (val - luma);
}

void main() {
    float ratio = globals.res_x / globals.res_y;

    vec3 sum = vec3(0.0);

    vec3 inputColor = texture(images[push_consts.src], uv).rgb;

    inputColor = agx(inputColor);
    inputColor = agxEotf(inputColor);

    outColor = vec4(
    inputColor,
    1.0
    );
}
