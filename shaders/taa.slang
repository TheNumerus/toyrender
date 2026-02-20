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
    int curr_idx;
    int prev_idx;
    int curr_depth_idx;
    int prev_depth_idx;
};

[vk::push_constant]
uniform PushConsts pc;

float2 uv_to_abs(float2 uv) {
    return float2(uv.x * (global.draw_res_x / global.res_x), uv.y * (global.draw_res_y / global.res_y));
}

float get_linear_depth(float4x4 viewproj, Sampler2D depth, float2 uv) {
    float curr_depth = depth.Sample(uv_to_abs(uv)).x;

    float4 per_pos = mul(vp.proj_inverse, float4(uv * 2.0 - 1.0, curr_depth, 1.0));
    per_pos /= per_pos.w;
    float3 curr_pos = mul(vp.view_inverse, per_pos).xyz;
    float4 pos = mul(viewproj, float4(curr_pos, 1.0));

    return pos.z * pos.w;
}

bool reproject(float2 uv_in, out float2 uv_out, int2 size) {
    float curr_depth = images[pc.curr_depth_idx].Sample(uv_to_abs(uv_in)).x;

    float4x4 viewproj = mul(vp.proj[0], vp.view[0]);

    float4 per_pos = mul(vp.proj_inverse, float4(uv_in * 2.0 - 1.0, curr_depth, 1.0));
    per_pos /= per_pos.w;
    float3 curr_pos = mul(vp.view_inverse, per_pos).xyz;
    float4 pos = mul(viewproj, float4(curr_pos, 1.0));
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

    float prev_lin_depth = get_linear_depth(viewproj, images[pc.prev_depth_idx], float2(lx, ly));
    if (abs(lin_depth - prev_lin_depth) <= 0.01) {
        //uv_out = float2(floor(prev_uv.x * size.x) / size.x, floor(prev_uv.y * size.y) / size.y);
        return true;
    }

    prev_lin_depth = get_linear_depth(viewproj, images[pc.prev_depth_idx], float2(hx, ly));
    if (abs(lin_depth - prev_lin_depth) <= 0.01) {
        //uv_out = float2(ceil(prev_uv.x * size.x) / size.x, floor(prev_uv.y * size.y) / size.y);
        return true;
    }

    prev_lin_depth = get_linear_depth(viewproj, images[pc.prev_depth_idx], float2(lx, hy));
    if (abs(lin_depth - prev_lin_depth) <= 0.01) {
        //uv_out = float2(floor(prev_uv.x * size.x) / size.x, ceil(prev_uv.y * size.y) / size.y);
        return true;
    }

    prev_lin_depth = get_linear_depth(viewproj, images[pc.prev_depth_idx], float2(hx, hy));
    if (abs(lin_depth - prev_lin_depth) <= 0.01) {
        //uv_out = float2(ceil(prev_uv.x * size.x) / size.x, ceil(prev_uv.y * size.y) / size.y);
        return true;
    }

    return false;
}

[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 tid: SV_DispatchThreadID) {
    uint x = tid.x;
    uint y = tid.y;

    int2 size = int2(int(global.res_x), int(global.res_y));
    int2 drawSize = int2(int(global.res_x), int(global.res_y));

    float blend_factor = 0.1;
    if (pc.clear == 1) {
        blend_factor = 1.0;
    }

    if (x <= global.res_x && y <= global.res_y) {
        float2 uv = (float2(x, y) + 0.5) / float2(global.res_x, global.res_y);
        float2 sampleUv = (float2(x, y) + 0.5) / float2(global.draw_res_x, global.draw_res_y);

        float4 current = images[pc.curr_idx].Sample(uv);

        float2 uv_reprojected;
        if (!reproject(sampleUv, uv_reprojected, drawSize)) {
            blend_factor = 1.0;
        }

        if (global.debug == DebugMode::Disocclusion) {
            float4 history_color = lerp(float4(0.0, 1.0, 0.0, 1.0), float4(1.0, 0.0, 0.0, 1.0), blend_factor);
            float4 color = lerp(current, history_color, 0.3);
            storages[pc.out_idx].Store(int2(x, y), color);
            return;
        }

        float4 prev = images[pc.prev_idx].Sample(uv_to_abs(uv_reprojected));

        float4 min_color = float4(1.0);
        float4 max_color = float4(0.0);

        for (int offset_x = -1; offset_x <= 1; offset_x ++) {
            for (int offset_y = -1; offset_y <= 1; offset_y ++) {
                float2 uv_sample = (float2(x + offset_x, y + offset_y) + 0.5) / float2(global.draw_res_x, global.draw_res_y);

                uv_sample.x = clamp(uv_sample.x, 0.0, 1.0);
                uv_sample.y = clamp(uv_sample.y, 0.0, 1.0);

                float4 sample_color = images[pc.curr_idx].Sample(uv_to_abs(uv_sample));
                min_color = min(min_color, sample_color);
                max_color = max(max_color, sample_color);
            }
        }

        float4 prev_clamped = clamp(prev, min_color, max_color);
        storages[pc.out_idx].Store(int2(x, y), lerp(prev_clamped, current, blend_factor));
    }
}
