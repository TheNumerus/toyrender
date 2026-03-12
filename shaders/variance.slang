import Common;

[[vk::binding(0, 0)]]
ConstantBuffer<Global> global;

[[vk::binding(2, 0)]]
ConstantBuffer<ViewProj> vp;

[[vk::binding(0, 1)]]
Sampler2D images[];

[[vk::binding(1, 1)]]
RWTexture2D<float4> storages[];

struct PushConsts {
    int src_idx;
    int out_idx;
    int curr_depth_idx;
    int normal_idx;
    int moments_idx;
};

[vk::push_constant]
uniform PushConsts pc;

float2 uv_to_abs(float2 uv) {
    return float2(uv.x * (global.draw_res_x / global.res_x), uv.y * (global.draw_res_y / global.res_y));
}

float to_linear_depth(float depth) {
    return vp.near * vp.far / (vp.far + depth * (vp.near - vp.far));
}

float get_linear_depth(uint2 pos) {
    return to_linear_depth(storages[pc.curr_depth_idx].Load(pos).x);
}

float get_depth_gradient(int2 uv, int2 size, float m) {
    float t = get_linear_depth(uv + int2(0, -1));

    float l = get_linear_depth(uv + int2(-1, 0));
    float m = to_linear_depth(m);
    float r = get_linear_depth(uv + int2(1, 0));

    float b = get_linear_depth(uv + int2(0, 1));

    return length(float2(((l - m) + (m - r)) / 2.0, ((t - m) + (m - b)) / 2.0));
}

float get_depth_weight(float base, float sample_depth, float gradient, float dist) {
    float a = abs(base - sample_depth);
    float b = 5.0 * length(gradient * dist) + 0.00001;

    return exp(-(a / b));
}

float get_luma_weight(float base, float sample_luma, float variance) {
    float a = abs(base - sample_luma);
    float b = 4.0 * (sqrt(variance) + 0.0001);
    return exp(-(a / b));
}

float luminance(float3 color) {
    return dot(color, float3(0.2126f, 0.7152f, 0.0722f));
}

float compute_spatial_variance(int2 uv_int, float luma_base, int2 size) {
    float2 uv = (float2(uv_int) + 0.5) / float2(global.draw_res_x, global.draw_res_y);
    float3 base_normal = normalize(storages[pc.normal_idx].Load(uv_int).xyz * 2.0 - 1.0);

    let base_depth = to_linear_depth(storages[pc.curr_depth_idx].Load(uv_int).x);

    if (base_depth == 0.0) {
        return 0.0;
    }

    float gradient = get_depth_gradient(uv_int, size, base_depth);

    float sum = 0.0;
    float sum_sq = 0.0;
    float sum_weight = 0.0;
    for (int x = -3; x <= 3; x++) {
        for (int y = -3; y <= 3; y++) {
            int2 uv = uv_int + int2(x, y);
            if (uv.x >= 0 || uv.y >= 0 || uv.x < size.x || uv.y < size.y) {
                float2 texture_uv = (float2(uv) + 0.5) / float2(global.draw_res_x, global.draw_res_y);

                float4 current = storages[pc.src_idx].Load(uv);
                float luma_sample = luminance(current.xyz);
                float3 normal_sample = normalize(storages[pc.normal_idx].Load(uv).xyz * 2.0 - 1.0);
                float depth_sample = to_linear_depth(storages[pc.curr_depth_idx].Load(uv).x);
                float sample_dist = length(float2(x, y));

                float normal_weight = pow(max(dot(base_normal, normal_sample), 0.0), 128.0);
                float depth_weight = get_depth_weight(base_depth, depth_sample, gradient, sample_dist);
                float luma_weight = get_luma_weight(luma_sample, luma_base, 1.0);

                float weight = max(normal_weight * depth_weight * luma_weight, 0.0);

                sum += luma_sample * weight;
                sum_sq += (luma_sample * luma_sample) * weight;
                sum_weight += weight;
            }
        }
    }

    sum /= sum_weight;
    sum_sq /= sum_weight;

    return sum_sq - (sum * sum);
}

[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 tid: SV_DispatchThreadID) {
    uint x = tid.x;
    uint y = tid.y;

    int2 size = int2(int(global.res_x), int(global.res_y));
    int2 drawSize = int2(int(global.draw_res_x), int(global.draw_res_y));

    if (x >= drawSize.x || y >= drawSize.y) {
        return;
    }

    float2 uv = (float2(x, y) + 0.5) / float2(global.draw_res_x, global.draw_res_y);

    float3 current = storages[pc.src_idx].Load(int2(x, y)).xyz;

    float3 moments = storages[pc.moments_idx].Load(int2(x, y)).xyz;

    float variance;
    if (moments.z < 4.0) {
        variance = compute_spatial_variance(int2(x, y), luminance(current), drawSize) * (4.0 / moments.z) + (moments.y) - pow(moments.x, 2.0);
    } else {
        variance = (moments.y) - pow(moments.x, 2.0);
    }

    if (global.debug == DebugMode::DirectVariance || global.debug == DebugMode::IndirectVariance) {
        storages[pc.out_idx].Store(int2(x, y), float4(float3(variance * 5.0), 1.0));
        return;
    }

    storages[pc.out_idx].Store(int2(x, y), float4(current, variance));

}