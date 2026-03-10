import Common;
import Sky;

#define PI 3.14159265359
#define PHI 1.618033988

[[vk::binding(0, 0)]]
ConstantBuffer<Global> global;

[[vk::binding(2, 0)]]
ConstantBuffer<ViewProj> vp;

[[vk::binding(3, 0)]]
ConstantBuffer<Environment> env;

[[vk::binding(0, 1)]]
Sampler2D images[];

[[vk::binding(1, 1)]]
RWTexture2D<float4> storages[];

struct PushConsts {
    int out_idx;
    int color_idx;
    int normal_idx;
    int depth_idx;
    int direct_idx;
    int indirect_idx;
    int sky_idx;
};

float3 light(float3 normal, float3 light_dir, float3 shadow)  {
    return max(dot(normal, light_dir) * shadow, 0.0);
}

float spec(float3 loc, float3 normal, float3 pos, float3 lightPos, float shadow, float power) {
    float3 view_dir = normalize(pos - loc);
    float3 ref_dir = view_dir - 2.0 * normal * dot(normal, view_dir);

    return pow(max(dot(ref_dir, normalize(lightPos - pos)), 0.0), power) * shadow;
}

float schlick(float f0, float3 l, float3 n) {
    return f0 + (1.0 - f0) * pow(1.0 - dot(l, n), 5.0);
}

[numthreads(16, 16, 1)]
[shader("compute")]
void main(uint3 tid: SV_DispatchThreadID, uniform PushConsts pc) {
    uint x = tid.x;
    uint y = tid.y;

    if (x >= global.draw_res_x || y >= global.draw_res_y) {
        return;
    }

    float2 uv = (float2(x, y) + 0.5) / float2(global.res_x, global.res_y);
    float2 drawUv = (float2(x, y) + 0.5) / float2(global.draw_res_x, global.draw_res_y);

    float3 light_dir = env.sun_dir_sun_angle.xyz;

    float ratio = global.res_x / global.res_y;

    float4 color = images[pc.color_idx].Sample(uv);
    float3 normal = images[pc.normal_idx].Sample(uv).xyz * 2.0 - 1.0;
    float depth = 1.0 - images[pc.depth_idx].Sample(uv).x;

    float4 per_pos = mul(vp.proj_inverse, float4(drawUv * 2.0 - 1.0, depth, 1.0));
    per_pos /= per_pos.w;
    float3 vertPos = mul(vp.view_inverse, per_pos).xyz;
    float3 loc = transpose(vp.view_inverse)[3].xyz;
    float3 view_dir = normalize(vertPos - loc);

    float3 sky_col;
    if (env.do_distort == 1) {
        sky_col = Toyrender::Sky::sample_distorted(view_dir, images[pc.sky_idx]) * pow(2.0, env.exposure) * env.sky_color_intensity.w;
    } else {
        sky_col = Toyrender::Sky::sample(view_dir, images[pc.sky_idx]) * pow(2.0, env.exposure) * env.sky_color_intensity.w;
    }

    if (color.a == 0.0) {
        storages[pc.out_idx].Store(int2(x, y), float4(sky_col, 1.0));
        return;
    }

    float3 direct = images[pc.direct_idx].Sample(uv).xyz * pow(2.0, env.exposure);
    float3 indirect = images[pc.indirect_idx].Sample(uv).xyz * pow(2.0, env.exposure);

    float3 diffuse_dir = light(normalize(normal), light_dir, direct) + indirect;
    float3 specular = float3(0.0);
    float3 lighted = ((diffuse_dir + specular));

    storages[pc.out_idx].Store(int2(x, y), float4(lighted, 1.0));
}
