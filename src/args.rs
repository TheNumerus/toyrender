use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
pub struct Args {
    pub file_to_open: Option<PathBuf>,
    #[arg(short, long, default_value_t = false)]
    pub benchmark: bool,
}
