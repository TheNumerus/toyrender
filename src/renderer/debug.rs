#[repr(i32)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum DebugMode {
    None = 0,
    Direct = 1,
    Indirect = 2,
    Time = 3,
    BaseColor = 4,
    Normal = 5,
    Depth = 6,
    DisOcclusion = 7,
    VarianceDirect = 8,
    VarianceIndirect = 9,
    DenoiseDirect = 10,
    DenoiseIndirect = 11,
}

impl TryFrom<i32> for DebugMode {
    type Error = &'static str;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(DebugMode::None),
            1 => Ok(DebugMode::Direct),
            2 => Ok(DebugMode::Indirect),
            3 => Ok(DebugMode::Time),
            4 => Ok(DebugMode::BaseColor),
            5 => Ok(DebugMode::Normal),
            6 => Ok(DebugMode::Depth),
            7 => Ok(DebugMode::DisOcclusion),
            8 => Ok(DebugMode::VarianceDirect),
            9 => Ok(DebugMode::VarianceIndirect),
            10 => Ok(DebugMode::DenoiseDirect),
            11 => Ok(DebugMode::DenoiseIndirect),
            _ => Err(""),
        }
    }
}

impl DebugMode {
    pub fn next(self) -> DebugMode {
        ((self as i32 + 1) % 12).try_into().unwrap()
    }
}
