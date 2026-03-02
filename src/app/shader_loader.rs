use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::io::{Read, Seek};
use thiserror::Error;
use zip::ZipArchive;
use zip::result::ZipError;

#[derive(Debug)]
pub struct ShaderLoader {
    pub shaders: HashMap<String, SpirVFile>,
}

impl ShaderLoader {
    pub fn from_zip(mut zip: ZipArchive<impl Read + Seek>) -> Result<Self, ShaderLoaderError> {
        let mut shaders = HashMap::new();

        for index in 0..zip.len() {
            let mut file = zip.by_index(index)?;

            let mut content = Vec::with_capacity(file.size() as usize);
            file.read_to_end(&mut content)
                .map_err(|e| ShaderLoaderError::MangledFile(format!("Cannot read shader file: {e}").into()))?;
            shaders.insert(file.name().to_owned(), SpirVFile(content));
        }

        Ok(Self { shaders })
    }
}

pub struct SpirVFile(Vec<u8>);

impl Debug for SpirVFile {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpirVFile").field("size", &self.0.len()).finish()
    }
}

impl AsRef<[u8]> for SpirVFile {
    fn as_ref(&self) -> &[u8] {
        self.0.as_slice()
    }
}

#[derive(Debug, Error)]
pub enum ShaderLoaderError {
    #[error("Error reading file: {0}")]
    MangledFile(Cow<'static, str>),
    #[error("Zip error: {0}")]
    ZipError(ZipError),
}

impl From<ZipError> for ShaderLoaderError {
    fn from(value: ZipError) -> Self {
        Self::ZipError(value)
    }
}
