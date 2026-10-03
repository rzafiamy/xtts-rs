//! Prints the text chunks and token ids synthesis would use.
//! cargo run --release --example prep -- <model> <lang> "<text>"
fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let m = xtts::Xtts::load(
        std::path::Path::new(&a[1]),
        &candle_core::Device::Cpu,
        candle_core::DType::F32,
    )?;
    for c in m.prepare_text(&a[3], &a[2], &xtts::SynthOptions::default()) {
        println!(
            "{c:?} -> {:?}",
            xtts::text::clean(&c, &xtts::text::base_lang(&a[2]))
        );
        println!("  {:?}", m.tokenizer.encode(&c, &a[2])?);
    }
    Ok(())
}
