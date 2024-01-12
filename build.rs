use std::io::{ErrorKind, Write};
use std::path::Path;
use zip::write::FileOptions;
use zip::ZipWriter;

fn main() -> Result<(), std::io::Error> {
    if let Err(e) = std::fs::create_dir("build") {
        if e.kind() != ErrorKind::AlreadyExists {
            panic!("{e}")
        }
    }

    for file in std::fs::read_dir("build")? {
        std::fs::remove_file(file?.path())?;
    }

    for file in std::fs::read_dir("shaders")? {
        let file = file?;

        let name = file.file_name();

        if file.path().extension().unwrap() != "glsl" {
            continue;
        }

        let final_name = format!("{}.spv", file.path().file_stem().unwrap().to_str().unwrap());

        let input = format!("shaders/{}", name.to_str().unwrap());
        let output = format!("build/{}", final_name);

        let res = std::process::Command::new("glslc")
            .args([&input, "-o", &output, "--target-env=vulkan1.3"])
            .output()
            .unwrap();

        if !res.status.success() {
            panic!("{}", String::from_utf8_lossy(&res.stderr));
        }

        println!("Compiling {} to {}", name.to_str().unwrap(), final_name);
    }

    let archive = std::fs::File::create("build/shaders.zip")?;
    let mut archive = ZipWriter::new(archive);

    for file in std::fs::read_dir("build")? {
        let file = file?;

        if file.path().extension().unwrap() != "spv" {
            continue;
        }

        archive.start_file(
            file.path().file_name().unwrap().to_string_lossy(),
            FileOptions::default(),
        )?;
        let content = std::fs::read(file.path())?;
        archive.write_all(content.as_slice())?;
    }

    archive.start_file("manifest.toml", FileOptions::default())?;
    let content = std::fs::read("shaders/manifest.toml")?;
    archive.write_all(content.as_slice())?;

    archive.finish()?;

    let manifest_dir_string = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let build_type = std::env::var("PROFILE").unwrap();
    let path = Path::new(&manifest_dir_string)
        .join("target")
        .join(build_type)
        .join("shaders.zip");
    std::fs::copy("build/shaders.zip", path)?;

    println!("cargo:rerun-if-changed=shaders");

    Ok(())
}
