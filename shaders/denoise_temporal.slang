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
    int prev_moments_idx;
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

float get_prev_linear_depth(Sampler2D depth, float2 uv) {
    float curr_depth = depth.Sample(uv_to_abs(uv)).x;

    float4 per_pos = mul(vp.proj_inverse, float4(uv * 2.0 - 1.0, curr_depth, 1.0));
    per_pos /= per_pos.w;
    float3 curr_pos = mul(vp.view_inverse, per_pos).xyz;
    float4 pos = mul(vp.proj[1], mul(vp.view[1], float4(curr_pos, 1.0)));

    return pos.z * pos.w;
}

float get_depth(Sampler2D depth, float2 uv) {
    return depth.Sample(uv_to_abs(uv)).x;
}

struct Weights {
    float topLeft = 0.0;
    float topRight = 0.0;
    float bottomLeft = 0.0;
    float bottomRight = 0.0;
}

Optional<Weights> reproject(float2 uv_in, out float2 uv_out, int2 size) {
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
        return none;
    }

    // try 2x2 reproject
    float lx = (floor((prev_uv.x * size.x) - 0.5) + 0.5) / size.x;
    float hx = (ceil((prev_uv.x * size.x) - 0.5) + 0.5) / size.x;

    float ly = (floor((prev_uv.y * size.y) - 0.5) + 0.5) / size.y;
    float hy = (ceil((prev_uv.y * size.y) - 0.5) + 0.5) / size.y;

    uv_out = prev_uv.xy;

    var matchFound = false;
    var weights = Weights();

    float prev_lin_depth = get_linear_depth(images[pc.prev_depth_idx], float2(lx, ly));
    if (abs(lin_depth - prev_lin_depth) <= 0.01) {
        matchFound = true;
        weights.bottomLeft = 1.0;
    }

    prev_lin_depth = get_linear_depth(images[pc.prev_depth_idx], float2(hx, ly));
    if (abs(lin_depth - prev_lin_depth) <= 0.01) {
        matchFound = true;
        weights.bottomRight = 1.0;
    }

    prev_lin_depth = get_linear_depth(images[pc.prev_depth_idx], float2(lx, hy));
    if (abs(lin_depth - prev_lin_depth) <= 0.01) {
        matchFound = true;
        weights.topLeft = 1.0;
    }

    prev_lin_depth = get_linear_depth(images[pc.prev_depth_idx], float2(hx, hy));
    if (abs(lin_depth - prev_lin_depth) <= 0.01) {
        matchFound = true;
        weights.topRight = 1.0;
    }

    if (matchFound) {
        return weights;
    }

    return none;
}

float luminance(float3 color) {
    return dot(color, float3(0.2126f, 0.7152f, 0.0722f));
}

float3 bilinear_load(RWTexture2D image, float2 uv, Weights weights) {
    float3 topLeft = image.Load(int2(floor(uv.x), ceil(uv.y))).xyz;
    float3 topRight = image.Load(int2(ceil(uv.x), ceil(uv.y))).xyz;
    float3 bottomLeft = image.Load(int2(floor(uv.x), floor(uv.y))).xyz;
    float3 bottomRight = image.Load(int2(ceil(uv.x), floor(uv.y))).xyz;

    float x = fract(uv.x);
    float y = fract(uv.y);

    float totalWeight =
        (1.0 - x) * (y) * weights.topLeft
        + (x) * (y) * weights.topRight
        + (1.0 - x) * (1.0 - y) * weights.bottomLeft
        + (x) * (1.0 - y) * weights.bottomRight;

    float3 total = topLeft * (1.0 - x) * (y) * weights.topLeft
        + topRight * (x) * (y) * weights.topRight
        + bottomLeft * (1.0 - x) * (1.0 - y) * weights.bottomLeft
        + bottomRight * (x) * (1.0 - y) * weights.bottomRight;

    return total/totalWeight;
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

    if (x >= drawSize.x || y >= drawSize.y) {
        return;
    }


    float2 uv = (float2(x, y) + 0.5) / float2(global.draw_res_x, global.draw_res_y);

    float4 current = storages[pc.curr_idx].Load(int2(x, y));

    if (current.w == 0.0) {
        return;
    }

    float2 uv_reprojected;
    float3 prev = float3(0.0);
    let weights = reproject(uv, uv_reprojected, drawSize);
    if (!weights.hasValue) {
        blend_factor = 1.0;
        moment_blend_factor = 1.0;
    } else {
        prev = max(bilinear_load(storages[pc.prev_idx], uv_reprojected * drawSize - 0.5, weights.value), float3(0.0));
    }

    storages[pc.out_idx].Store(int2(x, y), float4(lerp(prev, current.xyz, blend_factor), 0.0));

    var moments = float3(0.0);
    int2 coords = int2(round((uv_reprojected * drawSize)));
    int2 pix = int2(x, y);
    if (weights.hasValue && blend_factor != 1.0) {
        moments = bilinear_load(storages[pc.prev_moments_idx], uv_reprojected * drawSize - 0.5, weights.value);
    }

    float luma = luminance(current.xyz);
    float luma2 = luma * luma;

    storages[pc.moments_idx].Store(int2(x, y), float4(lerp(moments.x, luma, moment_blend_factor), lerp(moments.y, luma2, moment_blend_factor), moments.z + 1.0, 0.0));

}