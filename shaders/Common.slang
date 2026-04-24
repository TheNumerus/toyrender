module Common;

public struct Global {
    public int debug;
    public float res_x;
    public float res_y;
    public float draw_res_x;
    public float draw_res_y;
    public float time;
    public int frame_index;
    public float2 curr_jitter;
    public float2 prev_jitter;
};

public struct ViewProj {
    public float4x4 view[2];
    public float4x4 proj[2];
    public float4x4 view_inverse;
    public float4x4 proj_inverse;
    public float near;
    public float far;
};

public enum SkyMode {
    Shader = 0,
    SingleColor,
    Textured,
};

public struct Environment {
    public float4 sun_dir_sun_angle;
    public float4 sun_color_intensity;
    public float4 sky_color_intensity;
    public SkyMode mode;
    public float rotation;
    public float exposure;
    public int do_distort;
    public PointLight* lights;
    public int point_lights;
};

public struct MeshTransform {
    public float4x4 model;
    public float4x4 inverse;
    public int is_mirrored;
};

public enum DebugMode {
    None = 0,
    Direct,
    Indirect,
    Time,
    BaseColor,
    Normal,
    Depth,
    Disocclusion,
    DirectVariance,
    IndirectVariance,
    DenoiseDirect,
    DenoiseIndirect,
    Raw,
};

public struct Vertex {
    public float3 pos;
    public float3 normal;
    public float4 color;
    public float2 uv;
}

public struct MeshInstanceInfo {
    public float4x4 model_inverse;
    public float[12] *vertexData;
    public int *indexData;
}

public struct PointLight {
    public float4 color_intensity;
    public float4 pos_radius;
}
