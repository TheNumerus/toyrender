use clap::Parser;
use log::info;

mod app;
mod args;
mod camera;
mod err;
mod import;
mod input;
mod mesh;
mod renderer;
mod scene;
mod vulkan;

use crate::args::Args;

fn main() {
    env_logger::init();

    let args = Args::parse();
    let app = match app::App::create() {
        Ok(app) => app,
        Err(e) => {
            eprintln!("{e}");
            info!("Quitting app...");
            std::process::exit(1);
        }
    };

    match app.run(args) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("{e}");
        }
    };

    info!("Quitting app...");
}
