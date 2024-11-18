use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    maybe_generate_stubs()?;
    Ok(())
}

/// Generate stubs if we’re compiling the library
/// Returns `true` if stubs were generated
fn maybe_generate_stubs() -> Result<bool, Box<dyn std::error::Error>> {
    if env::var("CARGO_BIN_NAME") != Err(env::VarError::NotPresent) {
        return Ok(false);
    }

    // Find an existing `stub_gen` binary or exit silently
    let Some(bin_path) = ["debug", "release"]
        .into_iter()
        .filter_map(|mode| {
            let p = PathBuf::from(format!("target/{mode}/stub_gen"));
            p.exists().then_some(p)
        })
        .next()
    else {
        return Ok(false);
    };

    // If we’re compiling the library, generate stubs first
    Command::new(bin_path).spawn()?.wait()?;
    Ok(true)
}
