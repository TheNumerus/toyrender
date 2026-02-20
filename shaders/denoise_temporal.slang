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
    int clear;
    int out_idx;
    int curr_depth_idx;
    int prev_depth_idx;
    int curr_idx;
    int prev_idx;
    int normal_idx;
};

[vk::push_constant]
uniform PushConsts pc;

float2 uv_to_abs(float2 uv) {
    return float2(uv.x * (global.draw_res_x / global.res_x), uv.y * (global.draw_res_y / global.res_y));
}

float get_linear_depth(Sampler2D depth, float2 uv) {
    float curr_depth = depth.Sample(uv_to_abs(uv)).x;

    float4 per_pos = mul(vp.proj_inverse, float4(uv * 2.0 - 1.0, curr_depth, 1.0));
    per_pos /= per_pos.w;
    float3 curr_pos = mul(vp.view_inverse, per_pos).xyz;
    float4 pos = mul(vp.proj[0], mul(vp.view[0], float4(curr_pos, 1.0)));

    return pos.z * pos.w;
}

bool reproject(float2 uv_in, out float2 uv_out, int2 size) {
    float curr_depth = images[pc.curr_depth_idx].Sample(uv_to_abs(uv_in)).x;

    float4 per_pos = mul(vp.proj_inverse, float4(uv_in * 2.0 - 1.0, curr_depth, 1.0));
    per_pos /= per_pos.w;
    float3 curr_pos = mul(vp.view_inverse, per_pos).xyz;
    float4 pos = mul(vp.proj[0], mul(vp.view[0], float4(curr_pos, 1.0)));
    float lin_depth = pos.z * pos.w;

    float4 prev_uv = mul(vp.proj[1], mul(vp.view[1], float4(curr_pos, 1.0)));
    prev_uv /= prev_uv.w;
    prev_uv = prev_uv * 0.5 + 0.5;

    if (prev_uv.x < 0.0 || prev_uv.x > 1.0 || prev_uv.y < 0.0 || prev_uv.y > 1.0) {
        return false;
    }

    // try 2x2 reproject
    float lx = (floor((prev_uv.x * size.x) - 0.5) + 0.5) / size.x;
    float hx = (ceil((prev_uv.x * size.x) - 0.5) + 0.5) / size.x;

    float ly = (floor((prev_uv.y * size.y) - 0.5) + 0.5) / size.y;
    float hy = (ceil((prev_uv.y * size.y) - 0.5) + 0.5) / size.y;

    uv_out = prev_uv.xy;

    float prev_lin_depth = get_linear_depth(images[pc.prev_depth_idx], float2(lx, ly));
    if (abs(lin_depth - prev_lin_depth) <= 0.01) {
        return true;
    }

    prev_lin_depth = get_linear_depth(images[pc.prev_depth_idx], float2(hx, ly));
    if (abs(lin_depth - prev_lin_depth) <= 0.01) {
        return true;
    }

    prev_lin_depth = get_linear_depth(images[pc.prev_depth_idx], float2(lx, hy));
    if (abs(lin_depth - prev_lin_depth) <= 0.01) {
        return true;
    }

    prev_lin_depth = get_linear_depth(images[pc.prev_depth_idx], float2(hx, hy));
    if (abs(lin_depth - prev_lin_depth) <= 0.01) {
        return true;
    }

    return false;
}

float get_depth_gradient(uint x, uint y, Sampler2D depth, int2 size) {
    float2 uv = float2((x + 0.5) / size.x, (y + 0.5) / size.y);
    float x_offset = 1.0 / size.x;
    float y_offset = 1.0 / size.y;

    float m = get_linear_depth(depth, uv);
    float t = get_linear_depth(depth, uv + float2(0.0, -y_offset));
    float b = get_linear_depth(depth, uv + float2(0.0, y_offset));
    float l = get_linear_depth(depth, uv + float2(-x_offset, 0.0));
    float r = get_linear_depth(depth, uv + float2(x_offset, 0.0));

    return max(max(abs(t-m), abs(m-b)), max(abs(l-m), abs(m-r)));
}

float get_depth_weight(float base, float sample_depth, float gradient, float dist) {
    float a = abs(base - sample_depth);
    float b = 1.0 * abs(gradient * dist) + 0.0001;

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
    float3 base_normal = normalize(images[pc.normal_idx].Sample(uv_to_abs(uv)).xyz * 2.0 - 1.0);

    if (images[pc.curr_depth_idx].Sample(uv_to_abs(uv)).x == 0.0) {
        return 0.0;
    }

    float base_depth = get_linear_depth(images[pc.curr_depth_idx], uv);
    float gradient = get_depth_gradient(uv_int.x, uv_int.y, images[pc.curr_depth_idx], size);

    float sum = 0.0;
    float sum_sq = 0.0;
    float sum_weight = 0.0;
    for (int x = -3; x <= 3; x++) {
        for (int y = -3; y <= 3; y++) {
            int2 uv = uv_int + int2(x, y);
            if (uv.x >= 0 || uv.y >= 0 || uv.x < size.x || uv.y < size.y) {
                float2 texture_uv = (float2(uv) + 0.5) / float2(global.draw_res_x, global.draw_res_y);

                float4 current = storages[pc.curr_idx].Load(uv);
                float luma_sample = luminance(current.xyz);
                float3 normal_sample = normalize(images[pc.normal_idx].Sample(uv_to_abs(texture_uv)).xyz * 2.0 - 1.0);
                float depth_sample = get_linear_depth(images[pc.curr_depth_idx], texture_uv);
                float sample_dist = distance(texture_uv, uv);

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

    float blend_factor = 0.1;
    if (pc.clear == 1) {
        blend_factor = 1.0;
    }

    if (x <= drawSize.x && y <= drawSize.y) {
        float2 uv = (float2(x, y) + 0.5) / float2(global.draw_res_x, global.draw_res_y);

        float3 current = storages[pc.curr_idx].Load(int2(x, y)).xyz;

        float variance = compute_spatial_variance(int2(x, y), luminance(current), drawSize);

        if (global.debug == DebugMode::DirectVariance || global.debug == DebugMode::IndirectVariance) {
            storages[pc.out_idx].Store(int2(x, y), float4(float3(variance * 5.0), 1.0));
            return;
        }

        float2 uv_reprojected;
        if (!reproject(uv, uv_reprojected, drawSize)) {
            blend_factor = 1.0;
        }

        float4 prev = max(images[pc.prev_idx].Sample(uv_to_abs(uv_reprojected)), float4(0.000001));
        storages[pc.out_idx].Store(int2(x, y), lerp(prev, float4(current, variance), blend_factor));
    }
}