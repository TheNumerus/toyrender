use crate::app::frame_stats::FrameStat;

macro_rules! stat {
    ($stat_name:ident, $t:ty,$id:literal,$display:literal) => {
        pub struct $stat_name;

        impl FrameStat for $stat_name {
            type Item = $t;
            fn id() -> &'static str {
                $id
            }
            fn display() -> &'static str {
                $display
            }
        }
    };
}

stat!(DrawCallStat, u32, "draw_calls", "Draw calls");
stat!(InstanceCountStat, u32, "instance_count", "Drawn objects");
stat!(CullPercentageStat, f32, "cull_percentage", "Culling percentage");
stat!(
    ResourcePrepareStat,
    f32,
    "resource_prepare",
    "Resource prepare time (ms)"
);
stat!(TlasTimeStat, f32, "tlas_time", "TLAS build time (ms)");
stat!(MeshTimeStat, f32, "mesh_time", "Mesh prepare time (ms)");
stat!(DescTimeStat, f32, "desc_time", "Descriptor prepare time (ms)");
stat!(CpuTimeStat, f32, "cpu_time", "CPU time (ms)");
stat!(RecordTimeStat, f32, "record_time", "Record time (ms)");
stat!(VramUsageStat, f32, "vram_usage", "VRAM used (MB)");
stat!(VramAllocatedStat, f32, "vram_allocated", "VRAM allocated (MB)");
stat!(RenderTargetStat, u32, "rt_count", "Render targets");
