import Common;
import Octa;

[[vk::binding(2, 0)]]
ConstantBuffer<ViewProj> vp;

[[vk::binding(3, 0)]]
ConstantBuffer<Environment> env;

[[vk::binding(4, 0)]]
StructuredBuffer<MeshTransform> mesh;

[[vk::binding(0, 1)]]
Sampler2D images[];

struct VSInput {
    float3 pos;
    float3 normal;
    float4 color;
    float2 uv;
};

struct FSInput {
    float4 pos : SV_Position;
    float3 normal;
    float3 color;
};

struct FSOutput {
    float4 color : SV_Target0;
};

struct PushConsts {
    float4x4 proj;
    float3 offset;
    int conv_idx;
    int use_sun;
};

struct PushConstsDiff {
    float4x4 proj;
    float3 offset;
    int conv_idx;
    int sun_idx;
};

[shader("vertex")]
FSInput vert(VSInput input, uniform PushConsts pc) {
    FSInput output;

    float4x4 view = vp.view[0];

    // fix the gizmo at center
    view._14 = 0.0;
    view._24 = 0.0;
    view._34 = 0.0;

    output.pos = mul(pc.proj, mul(view, float4(input.pos + float3(0.0, 0.0, 0.0), 1.0)) + float4(pc.offset, 0.0));
    output.color = input.color.xyz;

    output.normal = input.normal;

    return output;
}

[shader("fragment")]
FSOutput frag(FSInput input, uniform PushConsts pc) {
    FSOutput output;

    output.color = float4(input.normal, 1.0);

    float3x3 rot = float3x3(0.0);
    rot._11 = cos(env.rotation * 2.0 * PI);
    rot._12 = sin(env.rotation * 2.0 * PI);
    rot._21 = -sin(env.rotation * 2.0 * PI);
    rot._22 = cos(env.rotation * 2.0 * PI);
    rot._33 = 1.0;

    float2 oct = Toyrender::Octa::encode(mul(input.normal, rot));
    float2 octNorm = (oct + 1.0) * 0.5;

    int width, height;
    images[pc.conv_idx].GetDimensions(width, height);
    float convDim = float(width);

    octNorm *= (convDim - 1.0) / convDim;
    octNorm += 0.5 / convDim;
    float3 convRes = images[pc.conv_idx].Sample(octNorm).xyz * env.sky_color_intensity.w;
    output.color = float4(convRes, 1.0);

    if (pc.use_sun != 0) {
        output.color.rgb += max(dot(input.normal, env.sun_dir_sun_angle.xyz), 0.0) * env.sun_color_intensity.rgb * env.sun_color_intensity.w;
    }

    return output;
}

[shader("fragment")]
FSOutput fragDiff(FSInput input, uniform PushConstsDiff pc) {
    FSOutput output;

    output.color = float4(input.normal, 1.0);

    float3x3 rot = float3x3(0.0);
    rot._11 = cos(env.rotation * 2.0 * PI);
    rot._12 = sin(env.rotation * 2.0 * PI);
    rot._21 = -sin(env.rotation * 2.0 * PI);
    rot._22 = cos(env.rotation * 2.0 * PI);
    rot._33 = 1.0;

    float2 oct = Toyrender::Octa::encode(mul(input.normal, rot));
    float2 octNorm = (oct + 1.0) * 0.5;

    int width, height;
    images[pc.conv_idx].GetDimensions(width, height);
    float convDim = float(width);

    octNorm *= (convDim - 1.0) / convDim;
    octNorm += 0.5 / convDim;
    float3 convRes = images[pc.conv_idx].Sample(octNorm).xyz * env.sky_color_intensity.w;
    float3 convSunRes = images[pc.sun_idx].Sample(octNorm).xyz * env.sky_color_intensity.w;

    float3 sun = max(dot(input.normal, env.sun_dir_sun_angle.xyz), 0.0) * env.sun_color_intensity.rgb * env.sun_color_intensity.w;

    output.color = float4(abs(convSunRes - (convRes + sun)) * 1.0, 1.0);

    return output;
}
