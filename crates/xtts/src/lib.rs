//! Coqui XTTS-v2 in Rust/Candle with GGUF weights.

pub mod config;
pub mod gguf;
pub mod gpt;
pub mod hifigan;
pub mod model;
pub mod sampling;
pub mod text;
mod text_tables;
pub mod weights;

pub use model::{Codes, StreamOptions, SynthOptions, Voice, Xtts};
pub use sampling::SamplingOptions;

/// Returns memory the CUDA allocator keeps cached after a request to the
/// driver (Candle allocates from the device's default stream-ordered pool,
/// which otherwise keeps the peak of past requests). No-op off CUDA.
pub fn release_cached_memory(device: &candle_core::Device) -> anyhow::Result<()> {
    #[cfg(feature = "cuda")]
    if let candle_core::Device::Cuda(d) = device {
        let stream = d.cuda_stream();
        stream.synchronize()?;
        let ctx = stream.context();
        if ctx.has_async_alloc() {
            // SAFETY: the device handle comes from a live context, and the
            // default pool lives as long as the device.
            unsafe {
                let pool = cudarc::driver::result::device::get_default_mem_pool(ctx.cu_device())?;
                cudarc::driver::result::mem_pool::trim_to(pool, 0)?;
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    let _ = device;
    Ok(())
}
