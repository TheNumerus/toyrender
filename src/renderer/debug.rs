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
    Raw = 12,
}

impl TryFrom<i32> for DebugMode {
    type Error = &'static str;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0..Self::COUNT => Ok(unsafe { std::mem::transmute::<i32, DebugMode>(value) }),
            _ => Err(""),
        }
    }
}

impl DebugMode {
    pub const COUNT: i32 = 13;

    pub fn next(self) -> DebugMode {
        ((self as i32 + 1) % Self::COUNT).try_into().unwrap()
    }
}
