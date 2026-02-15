use std::io::{ErrorKind, Write};
use std::path::Path;
use zip::ZipWriter;
use zip::write::FileOptions;

const ENTRY_REGEX: &str = r##"\[shader\("[a-z]+"\)\][\sa-zA-Z0-9]+ ([A-Za-z]+)\("##;

fn main() -> Result<(), std::io::Error> {
    if let Err(e) = std::fs::create_dir("build") {
        if e.kind() != ErrorKind::AlreadyExists {
            panic!("{e}")
        }
    }

    for file in std::fs::read_dir("build")? {
        std::fs::remove_file(file?.path())?;
    }

    let archive = std::fs::File::create("build/shaders.zip")?;
    let mut archive = ZipWriter::new(archive);

    let re = regex::Regex::new(ENTRY_REGEX).map_err(|e| std::io::Error::other(e))?;

    for file in std::fs::read_dir("shaders")? {
        let file = file?;
        let path = file.path();

        if path.is_dir() {
            continue;
        }

        let name = file.file_name();

        if path.extension().unwrap() != "slang" {
            continue;
        }

        let stem = path.file_stem().unwrap().to_str().unwrap();

        let input = format!("shaders/{}", name.to_str().unwrap());

        let content = std::fs::read_to_string(&input)?;

        for m in re.captures_iter(&content) {
            let entry = m.get(1).unwrap().as_str();

            let name = format!("{stem}|{entry}");

            let res = std::process::Command::new("slangc")
                .args([&input, "-target", "spirv", "-entry", entry])
                .output()
                .unwrap();

            if !res.status.success() {
                panic!("{}: {}", res.status, String::from_utf8_lossy(&res.stderr));
            }

            println!("Compiling {stem}|{entry}");

            archive.start_file(&name, FileOptions::default())?;
            archive.write_all(&res.stdout)?;
        }
    }

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
