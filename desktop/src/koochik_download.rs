//! Download one private HF bundle at a resolved immutable revision.
use crate::koochik_bundle::{Manifest, prepare};
use anyhow::{Context, Result, ensure};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};
const REPO: &str = "Reza2kn/Gooya-Koochik-v2.0-exp-tract";
fn token() -> Result<String> {
    if let Ok(t) = std::env::var("HF_TOKEN") {
        if !t.trim().is_empty() {
            return Ok(t.trim().into());
        }
    }
    let home = std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .or_else(|| std::env::var_os("USERPROFILE"))
                .map(|p| PathBuf::from(p).join(".cache/huggingface"))
        })
        .context("Set HF_TOKEN to download the private Koochik model")?;
    Ok(fs::read_to_string(home.join("token"))
        .context(
            "Authenticate with Hugging Face or set HF_TOKEN to download the private Koochik model",
        )?
        .trim()
        .into())
}
fn curl(url: &str, dest: &Path, token: &str) -> Result<()> {
    ensure!(
        !token.is_empty()
            && token
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_'),
        "invalid Hugging Face token format"
    );
    let mut child = Command::new("curl")
        .args([
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--retry",
            "3",
            "--connect-timeout",
            "20",
            "--config",
            "-",
            "--output",
        ])
        .arg(dest)
        .arg(url)
        .stdin(Stdio::piped())
        .spawn()
        .context("curl is required to download models")?;
    child
        .stdin
        .take()
        .context("curl stdin")?
        .write_all(format!("header = \"Authorization: Bearer {token}\"\n").as_bytes())?;
    ensure!(
        child.wait()?.success(),
        "Hugging Face download failed for {}",
        dest.file_name().unwrap_or_default().to_string_lossy()
    );
    Ok(())
}
pub fn download(root: &Path, progress: impl Fn(usize, usize, &str)) -> Result<()> {
    fs::create_dir_all(root)?;
    let token = token()?;
    let suffix = std::process::id();
    let info = root.join(format!(".hf-info-{suffix}.json"));
    curl(
        &format!("https://huggingface.co/api/models/{REPO}/revision/main"),
        &info,
        &token,
    )?;
    let value: serde_json::Value = serde_json::from_slice(&fs::read(&info)?)?;
    let revision = value["sha"]
        .as_str()
        .context("HF did not return a model revision")?;
    ensure!(
        revision.len() == 40 && revision.bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid model revision"
    );
    let pending = root.join(format!(".manifest-{suffix}.json"));
    curl(
        &format!("https://huggingface.co/{REPO}/resolve/{revision}/manifest.json"),
        &pending,
        &token,
    )?;
    let manifest: Manifest = serde_json::from_slice(&fs::read(&pending)?)?;
    for (i, asset) in manifest.files.iter().enumerate() {
        ensure!(
            Path::new(&asset.path)
                .components()
                .all(|c| matches!(c, std::path::Component::Normal(_))),
            "unsafe asset path"
        );
        let target = root.join(&asset.path);
        fs::create_dir_all(target.parent().context("asset parent")?)?;
        progress(i, manifest.files.len(), &asset.path);
        let partial = target.with_extension(format!("download-{suffix}"));
        curl(
            &format!(
                "https://huggingface.co/{REPO}/resolve/{revision}/{}",
                asset.path
            ),
            &partial,
            &token,
        )?;
        ensure!(
            fs::metadata(&partial)?.len() == asset.bytes,
            "downloaded file size mismatch"
        );
        crate::koochik_bundle::verify(&partial, asset)?;
        fs::rename(partial, target)?;
    }
    // The engine verifies every checksum before expanding and loading the graph.
    fs::rename(pending, root.join("manifest.json"))?;
    prepare(root)?;
    fs::write(root.join("download-revision.txt"), revision)?;
    let _ = fs::remove_file(info);
    progress(manifest.files.len(), manifest.files.len(), "ready");
    Ok(())
}
