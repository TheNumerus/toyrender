fn main() {
    // TODO build all
    std::process::Command::new("glslc").spawn().unwrap();
}
