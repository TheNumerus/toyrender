use std::io::ErrorKind;

fn main() {
    if let Err(e) = std::fs::create_dir("build") {
        if e.kind() != ErrorKind::AlreadyExists {
            panic!("{e}")
        }
    }

    for file in std::fs::read_dir("shaders").unwrap() {
        let file = file.unwrap();

        let name = file.file_name();

        let final_name = format!("{}.spv", file.path().file_stem().unwrap().to_str().unwrap());

        let input = format!("shaders/{}", name.to_str().unwrap());
        let output = format!("build/{}", final_name);

        std::process::Command::new("glslc")
            .args([&input, "-o", &output])
            .spawn()
            .unwrap()
            .wait()
            .unwrap();

        println!("Compiling {} to {}", name.to_str().unwrap(), final_name);
    }

    println!("cargo:rerun-if-changed=shaders");
}
