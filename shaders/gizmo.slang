import Common;

[[vk::binding(2, 0)]]
ConstantBuffer<ViewProj> vp;

[[vk::binding(4, 0)]]
StructuredBuffer<MeshTransform> mesh;

struct VSInput {
    float3 pos;
    float3 normal;
    float4 color;
    float2 uv;
};

struct FSInput {
    float4 pos : SV_Position;
    float3 color;
};

struct FSOutput {
    float4 color : SV_Target0;
};

[shader("vertex")]
FSInput vert(VSInput input) {
    FSInput output;

    float4x4 view = vp.view[0];
    float4x4 proj = vp.proj[0];

    float3 dir = -view[2].xyz * 8.0;

    // fix the gizmo at center
    view._14 = 0.0;
    view._24 = 0.0;
    view._34 = 0.0;

    // remove jitter
    proj._13 = 0.0;
    proj._23 = 0.0;

    output.pos = mul(proj, mul(view, float4(input.pos + float3(0.0, 0.0, -0.5) + dir, 1.0)));
    output.color = input.color.xyz;

    return output;
}

[shader("vertex")]
FSInput vertArrow(VSInput input, uniform float4x4 rotation) {
    FSInput output;

    float4x4 view = vp.view[0];
    float4x4 proj = vp.proj[0];

    float3 dir = -view[2].xyz * 8.0;

    // fix the gizmo at center
    view._14 = 0.0;
    view._24 = 0.0;
    view._34 = 0.0;

    // remove jitter
    proj._13 = 0.0;
    proj._23 = 0.0;

    output.pos = mul(proj, mul(view, float4( mul(transpose(rotation), float4(input.pos, 0.0)).xyz + float3(0.0, 0.0, -0.5) + dir, 1.0)));
    output.color = input.color.xyz;

    return output;
}

[shader("fragment")]
FSOutput frag(FSInput input) {
    FSOutput output;

    output.color = float4(input.color, 1.0);

    return output;
}
