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
    int level;
    int normal_idx;
    int depth_idx;
    int grad_idx;
    int src_idx;
    int out_idx;
};

[vk::push_constant]
uniform PushConsts pc;

int stride(int level) {
    return int(pow(2, level));
}

float luminance(float3 color) {
    return dot(color, float3(0.2126f, 0.7152f, 0.0722f));
}

const static float storages_weights[] = { 0.375, 0.25, 0.0625 };

float get_weight(int offset) {
    return storages_weights[abs(offset)];
}

const static float gauss_3x3[] = { 0.5, 0.25 };
float get_gauss_3x3_weight(int offset) {
    return gauss_3x3[abs(offset)];
}

float2 uv_to_abs(float2 uv) {
    return float2(uv.x * (global.draw_res_x / global.res_x), uv.y * (global.draw_res_y / global.res_y));
}

float to_linear_depth(float depth) {
    return vp.near * vp.far / (vp.far + depth * (vp.near - vp.far));
}

float get_depth(Sampler2D depth, float2 uv) {
    return depth.Sample(uv_to_abs(uv)).x;
}

float get_depth_gradient(uint3 localid) {
    float t = to_linear_depth(depths[localid.y][localid.x + 1]);

    float l = to_linear_depth(depths[localid.y + 1][localid.x]);
    float m = to_linear_depth(depths[localid.y + 1][localid.x + 1]);
    float r = to_linear_depth(depths[localid.y + 1][localid.x + 2]);

    float b = to_linear_depth(depths[localid.y + 2][localid.x + 1]);

    return length(float2(((l - m) + (m - r)) / 2.0, ((t - m) + (m - b)) / 2.0));
}

float get_depth_weight(float base, float sample_depth, float gradient, float dist) {
    float a = abs(base - sample_depth);
    float b = 5.0 * length(gradient * dist) + 0.00001;

    return -(a / b);
}

float get_luma_weight(float base, float sample_luma, float variance) {
    float a = abs(base - sample_luma);
    float b = 4.0 * sqrt(variance) + 0.00001;
    return -(a / b);
}

float get_blurred_variance(uint3 localid) {
    float sum = 0.0;
    float weight = 0.0;

    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            int final_y = int(localid.y) + y + 1;
            int final_x = int(localid.x) + x + 1;

            float val = depths[final_y][final_x];
            float weight_sample = get_gauss_3x3_weight(x) * get_gauss_3x3_weight(y);

            if (val < 0.0) {
                weight_sample = 0.0;
            }
            
            sum += val * weight_sample;
            weight += weight_sample;
        }
    }

    return sum / weight;
}

groupshared float depths[18][18];

void populateSharedVariance(uint3 localid, uint3 tid, int2 drawSize) {
    int2 start = tid.xy - localid.xy;

    int index = localid.x + localid.y * 16;
    // two iterations whould be enough
    for (int i_base = 0; i_base <= 1; i_base++) {
        let total_index = i_base * 16 * 16 + index;

        let offset = int2(total_index % 18, total_index / 18);
        int2 coords = start + offset - 1;
        coords = clamp(coords, 0, drawSize - 1);

        if (offset.y >= 18) {
            break;
        }

        depths[offset.y][offset.x] = storages[pc.src_idx].Load(coords).w;
    }

    GroupMemoryBarrierWithGroupSync();
}

[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 tid: SV_DispatchThreadID, uint3 localid: SV_GroupThreadID) {
    uint x_base = tid.x;
    uint y_base = tid.y;

    int2 size = int2(global.res_x, global.res_y);
    int2 drawSize = int2(global.draw_res_x, global.draw_res_y);

    float3 sum = float3(0.0);
    float weight_sum = 0.0;

    float new_variance_sum = 0.0;
    float new_variance_weight_sum = 0.0;

    float2 uv = float2((x_base + 0.5) / drawSize.x, (y_base + 0.5) / drawSize.y);

    float expDepth = storages[pc.depth_idx].Load(tid.xy).x;
    float gradient = storages[pc.grad_idx].Load(tid.xy).x;

    populateSharedVariance(localid, tid, drawSize);

    float blurred_variance = get_blurred_variance(localid);

    if (x_base >= drawSize.x || y_base >= drawSize.y) {
        return;
    }

    if (expDepth == 0.0) {
        return;
    }

    float4 result = storages[pc.src_idx].Load(int2(x_base, y_base));

    float luma_base = luminance(result.xyz);
    float3 normal_base = normalize(storages[pc.normal_idx].Load(int2(x_base, y_base)).xyz * 2.0 - 1.0);
    float depth_base = to_linear_depth(expDepth);

    for (int y = -2; y <= 2; y++) {
        int y_final = int(y_base) + y * stride(pc.level);
        for (int x = -2; x <= 2; x++) {
            int x_final = int(x_base) + x * stride(pc.level);
            if (y_final >= drawSize.y || y_final < 0 || x_final >= drawSize.x || x_final < 0) {
                continue;
            }
            let coords = int2(x_final, y_final);

            float3 sample_normal = normalize(storages[pc.normal_idx].Load(coords).xyz * 2.0 - 1.0);
            float sample_depth = storages[pc.depth_idx].Load(coords).x;
            sample_depth = to_linear_depth(sample_depth);

            float sample_dist = length(float2(x, y));

            float4 img_in = storages[pc.src_idx].Load(coords);
            float variance = img_in.a;

            float normal_weight = pow(max(dot(normal_base, sample_normal), 0.0), 128.0);
            float depth_weight = get_depth_weight(depth_base, sample_depth, gradient, sample_dist);
            float luma_weight = get_luma_weight(luminance(img_in.xyz), luma_base, blurred_variance);
            float depth_luma_exp = exp(depth_weight + luma_weight);

            float filter_weight = get_weight(x) * get_weight(y);
            float edge_stops = normal_weight * depth_luma_exp;

            float weight = max(filter_weight * edge_stops, 0.0000001);
            float variance_weight = max(pow(weight, 2.0), 0.0000001);

            sum += img_in.xyz * weight;
            weight_sum += weight;
            new_variance_sum += variance * variance_weight;
        }
    }
    float3 targetColor = max((sum / weight_sum).xyz, float3(0.0));
    float targetVariance = max(new_variance_sum / pow(weight_sum, 2.0), 0.0);

    float4 total = float4(targetColor, targetVariance);

    storages[pc.out_idx].Store(int2(x_base, y_base), total);
}
