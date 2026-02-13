import Common;

[[vk::binding(0, 0)]]
ConstantBuffer<Global> global;

[[vk::binding(0, 1)]]
Sampler2D images[];

[[vk::binding(2, 0)]]
ConstantBuffer<ViewProj> vp;

struct VSInput {
    float3 pos;
    float3 normal;
    float4 color;
    float2 uv;
};

struct VSOutput {
    float4 pos : SV_Position;
    float2 uv;
};

float luminance(float3 v) {
    return dot(v, float3(0.2126f, 0.7152f, 0.0722f));
}

float3 change_luminance(float3 c_in, float l_out) {
    float l_in = luminance(c_in);
    return c_in * (l_out / l_in);
}

float3 reinhard_extended_luminance(float3 v, float max_white_l) {
    float l_old = luminance(v);
    float numerator = l_old * (1.0f + (l_old / (max_white_l * max_white_l)));
    float l_new = numerator / (1.0f + l_old);
    return change_luminance(v, l_new);
}

float3 abberation(int i) {
    float scale = (float(i) - 3.0) * 0.5;

    return float3(
        1.0 - min(abs(scale + 1.0), 1.0),
        1.0 - min(abs(scale + 0.0), 1.0),
        1.0 - min(abs(scale - 1.0), 1.0));
}

// Mean error^2: 3.6705141e-06
float3 agxDefaultContrastApprox(float3 x) {
    float3 x2 = x * x;
    float3 x4 = x2 * x2;

    return +15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

float3 agx(float3 val) {
    const float3x3 agx_mat = transpose(float3x3(
        0.842479062253094, 0.0423282422610123, 0.0423756549057051,
        0.0784335999999992, 0.878468636469772, 0.0784336,
        0.0792237451477643, 0.0791661274605434, 0.879142973793104));

        

    const float min_ev = -12.47393f;
    const float max_ev = 4.026069f;

    // Input transform (inset)
    val = mul(agx_mat, val);

    // Log2 space encoding
    val = clamp(log2(val), min_ev, max_ev);
    val = (val - min_ev) / (max_ev - min_ev);

    // Apply sigmoid function approximation
    val = agxDefaultContrastApprox(val);

    return val;
}

float3 agxEotf(float3 val) {
    const float3x3 agx_mat_inv = transpose(float3x3(
        1.19687900512017, -0.0528968517574562, -0.0529716355144438,
        -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
        -0.0990297440797205, -0.0989611768448433, 1.15107367264116));

    // Inverse input transform (outset)
    val = mul(agx_mat_inv, val);

    // sRGB IEC 61966-2-1 2.2 Exponent Reference EOTF Display
    // NOTE: We're linearizing the output here. Comment/adjust when
    // *not* using a sRGB render target
    val = pow(val, float3(2.2));

    return val;
}

float3 agxLook(float3 val) {
    // Default
    float3 offset = float3(0.0);
    float3 slope = float3(1.0);
    float3 power = float3(1.0);
    float sat = 1.0;

    val = pow(val * slope + offset, power);

    const float3 lw = float3(0.2126, 0.7152, 0.0722);
    float luma = dot(val, lw);

    return luma + sat * (val - luma);
}

[shader("vertex")]
VSOutput vertMain(VSInput input, uniform int src) {
    VSOutput output;
    output.pos = float4(input.pos, 1.0);
    output.uv = input.uv;
    return output;
}

[shader("fragment")]
float4 fragMain(VSOutput input, uniform int src) : SV_Target0 {
    float ratio = global.res_x / global.res_y;

    float3 sum = float3(0.0);

    float3 inputColor = images[src].Sample(input.uv).rgb;

    inputColor = agx(inputColor);
    inputColor = agxEotf(inputColor);

    return float4(
        inputColor,
        1.0
    );
}
