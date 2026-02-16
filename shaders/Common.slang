module Common;

public struct Global {
    public float exposure;
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
};

public struct Environment {
    public float4 sun_dir_sun_angle;
    public float4 sun_color_sky_int;
    public float sun_int;
};

public struct MeshTransform {
    public float4x4 model;
    public float4x4 inverse;
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
};