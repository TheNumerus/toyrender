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
    int src_idx;
    int acc_idx;
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

float get_linear_depth(Sampler2D depth, float2 uv) {
    float curr_depth = depth.Sample(uv).x;

    float4 per_pos = mul(vp.proj_inverse, float4(uv * 2.0 - 1.0, curr_depth, 1.0));
    per_pos /= per_pos.w;
    float3 curr_pos = mul(vp.view_inverse, per_pos).xyz;
    float4 pos = mul(vp.proj[0] * vp.view[0], float4(curr_pos, 1.0));

    return pos.z * pos.w;
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

float get_blurred_variance(int2 uv, int2 size) {
    float sum = 0.0;
    float weight = 0.0;

    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            float val = storages[pc.src_idx].Load(int2(clamp(uv.x + x, 0, size.x - 1), clamp(uv.y + y, 0, size.y - 1))).w;
            float weight_sample = get_gauss_3x3_weight(x) * get_gauss_3x3_weight(y);
            sum += val * weight_sample;
            weight += weight_sample;
        }
    }

    return sum / weight;
}

float get_blurred_variance_1(int2 uv, int2 size) {
    float sum = 0.0;
    float weight = 0.0;

    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            float val = storages[pc.acc_idx].Load(int2(clamp(uv.x + x, 0, size.x - 1), clamp(uv.y + y, 0, size.y - 1))).w;
            float weight_sample = get_gauss_3x3_weight(x) * get_gauss_3x3_weight(y);
            sum += val * weight_sample;
            weight += weight_sample;
        }
    }

    return sum / weight;
}

[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 tid: SV_DispatchThreadID) {
    uint x_base = tid.x;
    uint y_base = tid.y;

    int width;
    int height;
    storages[pc.src_idx].GetDimensions(width, height);
    int2 size = int2(width, height);

    if (x_base <= size.x && y_base <= size.y) {
        float4 result = storages[pc.src_idx].Load(int2(x_base, y_base));
        float4 raw = storages[pc.acc_idx].Load(int2(x_base, y_base));

        float4 sum = float4(0.0);
        float weight_sum = 0.0;

        float new_variance_sum = 0.0;
        float new_variance_weight_sum = 0.0;

        float2 uv = float2((x_base + 0.5) / size.x, (y_base + 0.5) / size.y);

        if (images[pc.depth_idx].Sample(uv).x == 0.0) {
            return;
        }

        float luma_base = luminance(raw.xyz);
        float3 normal_base = normalize(images[pc.normal_idx].Sample(uv).xyz * 2.0 - 1.0);
        float depth_base = get_linear_depth(images[pc.depth_idx], uv);
        float gradient = get_depth_gradient(x_base, y_base, images[pc.depth_idx], size);
        float blurred_variance = get_blurred_variance(int2(x_base, y_base), size);

        for (int y = -2; y <= 2; y++) {
            for (int x = -2; x <= 2; x++) {
                int x_final = int(x_base) + x * stride(pc.level);
                int y_final = int(y_base) + y * stride(pc.level);

                if (x_final >= size.x || x_final < 0 || y_final >= size.y || y_final < 0) {
                    continue;
                }

                float2 sample_uv = float2((x_final + 0.5) / size.x, (y_final + 0.5) / size.y);
                float3 sample_normal = normalize(images[pc.normal_idx].Sample(sample_uv).xyz * 2.0 - 1.0);
                float sample_depth = get_linear_depth(images[pc.depth_idx], sample_uv);
                float sample_dist = distance(sample_uv, uv);

                float4 img_in;
                if (pc.level == 0) {
                    img_in = storages[pc.acc_idx].Load(int2(x_final, y_final));
                } else {
                    img_in = storages[pc.src_idx].Load(int2(x_final, y_final));
                }
                float variance = img_in.a;

                float normal_weight = pow(max(dot(normal_base, sample_normal), 0.0), 128.0);
                float depth_weight = get_depth_weight(depth_base, sample_depth, gradient, sample_dist);
                float luma_weight = get_luma_weight(luminance(img_in.xyz), luma_base, blurred_variance);

                float weight = max(get_weight(x) * get_weight(y) * normal_weight * depth_weight * luma_weight, 0.00001);
                float variance_weight = max(pow(get_weight(x) * get_weight(y), 2.0) * pow(normal_weight * depth_weight * luma_weight, 2.0), 0.00001);

                sum += img_in * weight;
                weight_sum += weight;
                new_variance_sum += variance * variance_weight;
                new_variance_weight_sum += get_weight(x) * get_weight(y) * pow(normal_weight * depth_weight * luma_weight, 2.0);
            }
        }

        float4 total = float4((sum / weight_sum).xyz, max(new_variance_sum / new_variance_weight_sum, 0.0));

        if (pc.level == 0) {
            storages[pc.acc_idx].Store(int2(x_base, y_base), total);
        }

        storages[pc.src_idx].Store(int2(x_base, y_base), total);
    }
}
