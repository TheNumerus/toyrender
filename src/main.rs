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
use err::AppError;
fn main() -> Result<(), AppError> {
    env_logger::init();

    let args = Args::parse();
    let app = app::App::create()?;
    app.run(args)?;

    info!("Quitting app...");

    Ok(())
}
