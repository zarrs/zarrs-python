use std::process::Command;
use std::{env, io};

fn main() -> Result<(), String> {
    if env::var("CARGO_BIN_NAME") == Err(env::VarError::NotPresent) {
        // If we’re compiling the library, generate stubs first
        if let Err(e) = Command::new("target/debug/stub_gen").spawn() {
            if e.kind() == io::ErrorKind::NotFound {
                eprintln!("Run `cargo build --bin stub_gen` first");
            } else {
                return Err(e.to_string());
            }
        };
    }
    Ok(())
}
