use log::warn;
use serde::Deserialize;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::io::{Read, Seek};
use thiserror::Error;
use zip::read::ZipFile;
use zip::result::ZipError;
use zip::ZipArchive;

#[derive(Debug)]
pub struct ShaderLoader {
    pub shaders: HashMap<String, SpirVFile>,
    pub manifest: Manifest,
}

impl ShaderLoader {
    pub fn from_zip(mut zip: ZipArchive<impl Read + Seek>) -> Result<Self, ShaderLoaderError> {
        let mut shaders = HashMap::new();

        let mut manifest = None;

        for index in 0..zip.len() {
            let mut file = zip.by_index(index)?;

            let (name, extension) = Self::get_zip_filename(&file)?;

            match (name.as_str(), extension.as_str()) {
                ("manifest", "toml") => {
                    manifest = Some(Self::build_manifest(&mut file)?);
                }
                (_, "spv") => {
                    let mut content = Vec::with_capacity(file.size() as usize);
                    file.read_to_end(&mut content)
                        .map_err(|e| ShaderLoaderError::MangledFile(format!("Cannot read shader file: {e}").into()))?;
                    shaders.insert(name.to_owned(), SpirVFile(content));
                }
                (_, _) => {
                    warn!("invalid file {name}.{extension} in shader archive");
                }
            }
        }

        let manifest = manifest.ok_or(ShaderLoaderError::MissingManifest)?;

        Self::validate(&shaders, &manifest)?;

        Ok(Self { shaders, manifest })
    }

    fn get_zip_filename(file: &ZipFile) -> Result<(String, String), ShaderLoaderError> {
        let name = file.name().split('.').collect::<Vec<_>>();

        match name[..] {
            [a, b] => Ok((a.to_owned(), b.to_owned())),
            _ => {
                let err = format!(
                    "invalid file name in zip: {}, needs file name AND extension",
                    file.name()
                );
                Err(ShaderLoaderError::MangledFile(err.into()))
            }
        }
    }

    fn build_manifest(file: &mut ZipFile) -> Result<Manifest, ShaderLoaderError> {
        let mut content = String::with_capacity(file.size() as usize);
        file.read_to_string(&mut content)
            .map_err(|e| ShaderLoaderError::MangledFile(format!("Cannot read manifest: {e}").into()))?;

        let stubs = toml::from_str(&content)
            .map_err(|e| ShaderLoaderError::MangledFile(format!("Error parsing manifest: {e}").into()))?;

        Ok(Manifest { stubs })
    }

    fn validate(shaders: &HashMap<String, SpirVFile>, manifest: &Manifest) -> Result<(), ShaderLoaderError> {
        let check = |name: &str| {
            if !shaders.contains_key(name) {
                Err(ShaderLoaderError::NameNotFound(name.to_owned()))
            } else {
                Ok(())
            }
        };

        for stub in manifest.stubs.values() {
            match stub {
                PipelineStub::Graphics { vertex, fragment, .. } => {
                    check(vertex)?;
                    check(fragment)?;
                }
                PipelineStub::Rt { raygen, closest, miss } => {
                    check(raygen)?;
                    check(closest)?;
                    check(miss)?;
                }
                PipelineStub::Compute { compute } => {
                    check(compute)?;
                }
            }
        }

        Ok(())
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

#[derive(Debug)]
pub struct Manifest {
    pub stubs: HashMap<String, PipelineStub>,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "kind")]
pub enum PipelineStub {
    #[serde(rename = "graphics")]
    Graphics {
        vertex: String,
        fragment: String,
        use_depth: bool,
    },
    #[serde(rename = "rt")]
    Rt {
        raygen: String,
        closest: String,
        miss: String,
    },
    #[serde(rename = "compute")]
    Compute { compute: String },
}

#[derive(Debug, Error)]
pub enum ShaderLoaderError {
    #[error("The manifest is missing")]
    MissingManifest,
    #[error("Shader '{0}' not found")]
    NameNotFound(String),
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
