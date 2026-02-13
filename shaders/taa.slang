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

float get_linear_depth(Sampler2D depth, float2 uv) {
    float curr_depth = depth.Sample(uv).x;

    float4 per_pos = mul(vp.proj_inverse, float4(uv * 2.0 - 1.0, curr_depth, 1.0));
    per_pos /= per_pos.w;
    float3 curr_pos = mul(vp.view_inverse, per_pos).xyz;
    float4 pos = mul(vp.proj[0], mul(vp.view[0], float4(curr_pos, 1.0)));

    return pos.z * pos.w;
}

bool reproject(float2 uv_in, out float2 uv_out, int2 size) {
    float curr_depth = images[pc.curr_depth_idx].Sample(uv_in).x;

    float4 per_pos = mul(vp.proj_inverse, float4(uv_in * 2.0 - 1.0, curr_depth, 1.0));
    per_pos /= per_pos.w;
    float3 curr_pos = mul(vp.view_inverse, per_pos).xyz;
    float4 pos = mul(vp.proj[0], mul(vp.view[0], float4(curr_pos, 1.0)));
    float lin_depth = pos.z * pos.w;

    float4 prev_uv = mul(vp.proj[1], mul(vp.view[1], float4(curr_pos, 1.0)));
    prev_uv /= prev_uv.w;
    prev_uv = prev_uv * 0.5 + 0.5;

    float2 snapped_uv = prev_uv.xy;
    snapped_uv.x = (round((snapped_uv.x * size.x) - 0.5) + 0.5) / size.x;
    snapped_uv.y = (round((snapped_uv.y * size.y) - 0.5) + 0.5) / size.y;

    float prev_lin_depth = get_linear_depth(images[pc.prev_depth_idx], snapped_uv.xy);
    uv_out = prev_uv.xy;

    if (prev_uv.x < 0.0 || prev_uv.x > 1.0 || prev_uv.y < 0.0 || prev_uv.y > 1.0 || abs(lin_depth - prev_lin_depth) > 0.01) {
        return false;
    }

    return true;
}

[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 tid: SV_DispatchThreadID) {
    uint x = tid.x;
    uint y = tid.y;

    int width;
    int height;
    storages[pc.out_idx].GetDimensions(width, height);
    int2 size = int2(width, height);

    float blend_factor = 0.1;
    if (pc.clear == 1) {
        blend_factor = 1.0;
    }

    if (x <= width && y <= height) {
        float2 uv = (float2(x, y) + 0.5) / float2(global.res_x, global.res_y);

        float4 current = images[pc.curr_idx].Sample(uv);

        for (int offset_x = -1; offset_x <= 1; offset_x ++) {
            for (int offset_y = -1; offset_y <= 1; offset_y ++) {
                float2 uv_sample = (float2(x + offset_x, y + offset_y) + 0.5) / float2(global.res_x, global.res_y);
                uv_sample.x = clamp(uv_sample.x, 0.0, 1.0);
                uv_sample.y = clamp(uv_sample.y, 0.0, 1.0);
            }
        }

        float2 uv_reprojected;
        if (!reproject(uv, uv_reprojected, size)) {
            blend_factor = 1.0;
        }

        if (global.debug == DebugMode::Disocclusion) {
            float4 history_color = lerp(float4(0.0, 1.0, 0.0, 1.0), float4(1.0, 0.0, 0.0, 1.0), blend_factor);
            float4 color = lerp(current, history_color, 0.3);
            storages[pc.out_idx].Store(int2(x, y), color);
            return;
        }

        float4 prev = images[pc.prev_idx].Sample(uv_reprojected);

        float4 min_color = float4(1.0);
        float4 max_color = float4(0.0);

        for (int offset_x = -1; offset_x <= 1; offset_x ++) {
            for (int offset_y = -1; offset_y <= 1; offset_y ++) {
                float2 uv_sample = (float2(x + offset_x, y + offset_y) + 0.5) / float2(global.res_x, global.res_y);

                uv_sample.x = clamp(uv_sample.x, 0.0, 1.0);
                uv_sample.y = clamp(uv_sample.y, 0.0, 1.0);

                float4 sample_color = images[pc.curr_idx].Sample(uv_sample);
                min_color = min(min_color, sample_color);
                max_color = max(max_color, sample_color);
            }
        }

        float4 prev_clamped = clamp(prev, min_color, max_color);
        storages[pc.out_idx].Store(int2(x, y), lerp(prev_clamped, current, blend_factor));
    }
}
