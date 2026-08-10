import Common;

[[vk::binding(2, 0)]]
ConstantBuffer<ViewProj> vp;

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
    float3 color;
    float3 mat;
    float2 uv;
    float3 normal;
};

struct FSOutput {
    float4 color : SV_Target0;
    float4 normal : SV_Target1;
    float4 mat : SV_Target2;
};

[shader("vertex")]
FSInput vertMain(VSInput input, uint inst: SV_InstanceID, uniform uint first) {
    FSInput output;

    output.pos = mul(vp.proj[0], mul(vp.view[0], mul(mesh[inst + first].model, float4(input.pos, 1.0))));
    output.color = mesh[inst + first].color_rough.xyz * input.color.xyz;
    output.mat = mesh[inst + first].color_rough.w;

    int is_mirrored = mesh[inst + first].is_mirrored;

    float3 normal = input.normal * -(float(is_mirrored) * 2.0 - 1.0);
    output.normal = normalize(mul(float3x3(transpose(mesh[inst + first].inverse)), normal));

    output.uv = input.uv;

    return output;
}

[shader("fragment")]
FSOutput fragMain(FSInput input, uint inst: SV_InstanceID, uniform uint first, bool ff: SV_IsFrontFace) {
    FSOutput output;

    output.color = float4(input.color, 1.0);
    output.normal = float4((input.normal + 1.0) * 0.5, 1.0);
    output.mat = float4(input.mat, 1.0);

    if (!ff) {
        output.normal = float4((-input.normal + 1.0) * 0.5, 1.0);
    }

    if (mesh[inst + first].color_idx != -1) {
        float4 tex = images[mesh[inst + first].color_idx].Sample(input.uv);
        if (tex.a < 0.5) {
            discard;
        }
        output.color *= tex;
    }

    if (mesh[inst + first].orm_idx != -1) {
        float4 tex = images[mesh[inst + first].orm_idx].Sample(input.uv);
        output.mat *= tex;
    }

    return output;
}
