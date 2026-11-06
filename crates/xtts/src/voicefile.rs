//! `.voice` files: one cloned voice, without any audio.
//!
//! Layout, little-endian: magic `XTTSVOIC`, `u32` format version (1), `u32`
//! name length, the name in UTF-8, `u32` latents, `u32` latent width, `u32`
//! embedding size, then the GPT latents and the speaker embedding as F32.
//! A voice of XTTS v2 (32 x 1024 + 512) takes about 133 KB.

use crate::model::Voice;
use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, RwLock};
use std::time::{Duration, Instant};

pub const MAGIC: &[u8; 8] = b"XTTSVOIC";
pub const VERSION: u32 = 1;
const COND: (usize, usize) = (32, 1024);
const SPK: usize = 512;
const MAX_NAME: usize = 1024;

/// How voice names compare: case-insensitive, `_` or `-` stand for spaces.
pub fn normalize_name(s: &str) -> String {
    s.to_lowercase().replace(['_', '-'], " ")
}

pub fn write_voice(path: &Path, name: &str, voice: &Voice) -> Result<()> {
    if name.is_empty() || name.len() > MAX_NAME {
        bail!("bad voice name '{name}'");
    }
    let cond: Vec<f32> = voice
        .cond
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1()?;
    let spk: Vec<f32> = voice
        .spk
        .flatten_all()?
        .to_device(&Device::Cpu)?
        .to_vec1()?;
    if cond.len() != COND.0 * COND.1 || spk.len() != SPK {
        bail!(
            "voice has {} latent values and {} embedding values",
            cond.len(),
            spk.len()
        );
    }
    let mut b = Vec::with_capacity(32 + name.len() + 4 * (cond.len() + spk.len()));
    b.extend_from_slice(MAGIC);
    b.extend_from_slice(&VERSION.to_le_bytes());
    b.extend_from_slice(&(name.len() as u32).to_le_bytes());
    b.extend_from_slice(name.as_bytes());
    for v in [COND.0, COND.1, SPK] {
        b.extend_from_slice(&(v as u32).to_le_bytes());
    }
    for v in cond.iter().chain(&spk) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    // Written beside and renamed, so a running server never sees half a file.
    let tmp = path.with_extension("voice.tmp");
    let mut f = std::fs::File::create(&tmp).with_context(|| format!("creating {tmp:?}"))?;
    f.write_all(&b)?;
    drop(f);
    std::fs::rename(&tmp, path).with_context(|| format!("writing {path:?}"))?;
    Ok(())
}

fn u32_at(b: &[u8], at: usize) -> Result<usize> {
    let s = b.get(at..at + 4).context("truncated header")?;
    Ok(u32::from_le_bytes(s.try_into()?) as usize)
}

/// Name and offset of the tensors, from the first bytes of a file.
fn parse_header(b: &[u8]) -> Result<(String, usize)> {
    if b.get(..8) != Some(MAGIC.as_slice()) {
        bail!("not a .voice file");
    }
    let ver = u32_at(b, 8)? as u32;
    if ver != VERSION {
        bail!("unsupported .voice version {ver}");
    }
    let len = u32_at(b, 12)?;
    if len == 0 || len > MAX_NAME {
        bail!("bad name length {len}");
    }
    let name = std::str::from_utf8(b.get(16..16 + len).context("truncated name")?)?.to_string();
    let at = 16 + len;
    let dims = (u32_at(b, at)?, u32_at(b, at + 4)?, u32_at(b, at + 8)?);
    if dims != (COND.0, COND.1, SPK) {
        bail!("unexpected shape {dims:?}");
    }
    Ok((name, at + 12))
}

fn tensors_size() -> u64 {
    4 * (COND.0 * COND.1 + SPK) as u64
}

/// Only the name: reads the first bytes and checks the file's length.
pub fn read_name(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path).with_context(|| format!("opening {path:?}"))?;
    let mut b = [0u8; 16 + MAX_NAME + 12];
    let mut got = 0;
    while got < b.len() {
        let n = f.read(&mut b[got..])?;
        if n == 0 {
            break;
        }
        got += n;
    }
    let (name, off) = parse_header(&b[..got]).with_context(|| format!("{path:?}"))?;
    if f.metadata()?.len() != off as u64 + tensors_size() {
        bail!("{path:?}: wrong size");
    }
    Ok(name)
}

/// The whole voice: latents `[1, 32, 1024]` and embedding `[512]`.
pub fn read_voice(path: &Path, device: &Device) -> Result<(String, Voice)> {
    let b = std::fs::read(path).with_context(|| format!("reading {path:?}"))?;
    let (name, off) = parse_header(&b).with_context(|| format!("{path:?}"))?;
    if b.len() as u64 != off as u64 + tensors_size() {
        bail!("{path:?}: wrong size");
    }
    let vals: Vec<f32> = b[off..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let n = COND.0 * COND.1;
    let cond = Tensor::from_slice(&vals[..n], (1, COND.0, COND.1), device)?;
    let spk = Tensor::from_slice(&vals[n..], SPK, device)?;
    Ok((name, Voice { cond, spk }))
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        match e.file_type() {
            Ok(t) if t.is_dir() => walk(&p, out),
            Ok(_) if p.extension().is_some_and(|x| x == "voice") => out.push(p),
            _ => {}
        }
    }
}

#[derive(Default)]
struct Entries {
    /// Normalized name → (name, path), sorted by name.
    voices: std::collections::BTreeMap<String, (String, PathBuf)>,
    /// Paths seen so far with an unusable file, so they warn only once.
    bad: std::collections::HashSet<PathBuf>,
}

/// A folder of `.voice` files (subfolders too). At startup only the headers
/// are read; a voice's tensors are read when a request uses it. Unknown names
/// rescan the folder (at most every 2 s), so voices cloned while the server
/// runs appear by themselves.
pub struct VoiceDir {
    dir: PathBuf,
    entries: RwLock<Entries>,
    last_scan: Mutex<Instant>,
}

const RESCAN_EVERY: Duration = Duration::from_secs(2);

impl VoiceDir {
    pub fn open(dir: &Path) -> Result<Self> {
        if !dir.is_dir() {
            bail!("voices folder {dir:?} does not exist");
        }
        let d = Self {
            dir: dir.to_path_buf(),
            entries: RwLock::new(Entries::default()),
            last_scan: Mutex::new(Instant::now()),
        };
        d.scan();
        Ok(d)
    }

    fn scan(&self) {
        let mut paths = Vec::new();
        walk(&self.dir, &mut paths);
        paths.sort();
        let mut e = self.entries.write().unwrap();
        let known: std::collections::HashSet<PathBuf> =
            e.voices.values().map(|(_, p)| p.clone()).collect();
        // Forget the files that went away.
        let present: std::collections::HashSet<&PathBuf> = paths.iter().collect();
        e.voices.retain(|_, (_, p)| present.contains(p));
        for p in &paths {
            if known.contains(p) || e.bad.contains(p) {
                continue;
            }
            match read_name(p) {
                Ok(name) => {
                    let key = normalize_name(&name);
                    if let Some((_, other)) = e.voices.get(&key) {
                        tracing::warn!("voice '{name}' in {p:?} ignored: already in {other:?}");
                        e.bad.insert(p.clone());
                    } else {
                        e.voices.insert(key, (name, p.clone()));
                    }
                }
                Err(err) => {
                    tracing::warn!("skipping voice file: {err:#}");
                    e.bad.insert(p.clone());
                }
            }
        }
    }

    /// Rescans unless that was just done.
    pub fn refresh(&self) {
        let mut last = self.last_scan.lock().unwrap();
        if last.elapsed() >= RESCAN_EVERY {
            *last = Instant::now();
            drop(last);
            self.scan();
        }
    }

    /// Names, sorted.
    pub fn names(&self) -> Vec<String> {
        let e = self.entries.read().unwrap();
        e.voices.values().map(|(n, _)| n.clone()).collect()
    }

    /// The file of a voice, rescanning once if the name is unknown.
    pub fn path_of(&self, name: &str) -> Option<PathBuf> {
        let key = normalize_name(name);
        let get = || {
            let e = self.entries.read().unwrap();
            e.voices.get(&key).map(|(_, p)| p.clone())
        };
        get().or_else(|| {
            self.refresh();
            get()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn voice(seed: f32) -> Voice {
        let cond: Vec<f32> = (0..COND.0 * COND.1).map(|i| seed + i as f32).collect();
        let spk: Vec<f32> = (0..SPK).map(|i| seed - i as f32).collect();
        Voice {
            cond: Tensor::from_vec(cond, (1, COND.0, COND.1), &Device::Cpu).unwrap(),
            spk: Tensor::from_vec(spk, SPK, &Device::Cpu).unwrap(),
        }
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("xtts-voicefile-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// covers: REQ-CLN-002
    #[test]
    fn round_trip_is_exact() {
        let d = tmp("rt");
        let p = d.join("a.voice");
        let v = voice(0.5);
        write_voice(&p, "Test_Voice", &v).unwrap();
        assert_eq!(read_name(&p).unwrap(), "Test_Voice");
        let (name, w) = read_voice(&p, &Device::Cpu).unwrap();
        assert_eq!(name, "Test_Voice");
        let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(flat(&v.cond), flat(&w.cond));
        assert_eq!(flat(&v.spk), flat(&w.spk));
        assert_eq!(w.cond.dims(), &[1, 32, 1024]);
        assert_eq!(w.spk.dims(), &[512]);
        std::fs::remove_dir_all(&d).unwrap();
    }

    /// covers: REQ-CLN-002
    #[test]
    fn folder_skips_bad_files() {
        let d = tmp("dir");
        std::fs::create_dir_all(d.join("sub")).unwrap();
        write_voice(&d.join("sub/x.voice"), "voice_one", &voice(1.0)).unwrap();
        std::fs::write(d.join("broken.voice"), b"XTTSVOIC????").unwrap();
        std::fs::write(d.join("other.txt"), b"nothing").unwrap();
        let vd = VoiceDir::open(&d).unwrap();
        assert_eq!(vd.names(), vec!["voice_one".to_string()]);
        assert!(vd.path_of("VOICE-ONE").is_some());
        std::fs::remove_dir_all(&d).unwrap();
    }
}
