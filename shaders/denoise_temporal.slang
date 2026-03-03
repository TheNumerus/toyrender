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
    int moments_idx;
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

float get_depth(Sampler2D depth, float2 uv) {
    return depth.Sample(uv_to_abs(uv)).x;
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

    float m = get_depth(depth, uv);
    float t = get_depth(depth, uv + float2(0.0, -y_offset));
    float b = get_depth(depth, uv + float2(0.0, y_offset));
    float l = get_depth(depth, uv + float2(-x_offset, 0.0));
    float r = get_depth(depth, uv + float2(x_offset, 0.0));

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

float3 bilinear_load(RWTexture2D image, float2 uv) {
    float3 topLeft = image.Load(int2(floor(uv.x), floor(uv.y))).xyz;
    float3 topRight = image.Load(int2(floor(uv.x), ceil(uv.y))).xyz;
    float3 bottomLeft = image.Load(int2(ceil(uv.x), floor(uv.y))).xyz;
    float3 bottomRight = image.Load(int2(ceil(uv.x), ceil(uv.y))).xyz;

    float3 topMix = lerp(topLeft, topRight, fract(uv.y));
    float3 bottomMix = lerp(bottomLeft, bottomRight, fract(uv.y));

    return lerp(topMix, bottomMix, fract(uv.x));
}

[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 tid: SV_DispatchThreadID) {
    uint x = tid.x;
    uint y = tid.y;

    int2 size = int2(int(global.res_x), int(global.res_y));
    int2 drawSize = int2(int(global.draw_res_x), int(global.draw_res_y));

    float blend_factor = 0.05;
    float moment_blend_factor = 0.2;

    if (pc.clear == 1) {
        blend_factor = 1.0;
        moment_blend_factor = 1.0;
    }

    if (x <= drawSize.x && y <= drawSize.y) {
        float2 uv = (float2(x, y) + 0.5) / float2(global.draw_res_x, global.draw_res_y);

        float3 current = storages[pc.curr_idx].Load(int2(x, y)).xyz;

        float2 uv_reprojected;
        bool do_reproject = reproject(uv, uv_reprojected, drawSize);
        if (!do_reproject) {
            blend_factor = 1.0;
            moment_blend_factor = 1.0;
        }

        float4 prev = max(images[pc.prev_idx].Sample(uv_to_abs(uv_reprojected)), float4(0.000001));
        storages[pc.out_idx].Store(int2(x, y), lerp(prev, float4(current, 0.0), blend_factor));

        var moments = float3(0.0);
        int2 coords = int2(round((uv_reprojected * drawSize)));
        int2 pix = int2(x, y);
        if (do_reproject && blend_factor != 1.0) {
            
            //moments = storages[pc.moments_idx].Load(coords).xyz;

            moments = bilinear_load(storages[pc.moments_idx], uv_reprojected * drawSize - 0.5);
        }

        float luma = luminance(current);
        float luma2 = luma * luma;

        storages[pc.moments_idx].Store(int2(x, y), float4(lerp(moments.x, luma, moment_blend_factor), lerp(moments.y, luma2, moment_blend_factor), moments.z + 1.0, 0.0));
    }
}