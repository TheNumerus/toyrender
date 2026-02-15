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
    float3 worldPos;
    float3 color;
    float2 uv;
    float3 normal;
};

struct FSOutput {
    float4 color : SV_Target0;
    float4 normal : SV_Target1;
};

[shader("vertex")]
FSInput vertMain(VSInput input, uint inst: SV_InstanceID, uniform uint first) {
    FSInput output;

    output.pos = mul(vp.proj[0], mul(vp.view[0], mul(mesh[inst + first].model, float4(input.pos, 1.0))));
    output.color = float3(1.0);
    output.worldPos = mul(mesh[inst + first].model, float4(input.pos, 1.0)).xyz;
    output.normal = normalize(mul(float3x3(transpose(mesh[inst + first].inverse)), input.normal));
    output.uv = input.uv;

    return output;
}

[shader("fragment")]
FSOutput fragMain(FSInput input) {
    FSOutput output;

    output.color = float4(input.color, 1.0);
    output.normal = float4((input.normal + 1.0) * 0.5, 1.0);

    return output;
}
