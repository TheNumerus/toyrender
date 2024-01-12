use std::io::ErrorKind;

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

    println!("cargo:rerun-if-changed=shaders");

    Ok(())
}
