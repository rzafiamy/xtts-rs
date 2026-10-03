fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let key = args.get(2).map(|s| s.as_str());
    let ts = candle_core::pickle::read_all_with_key(&args[1], key)?;
    println!("{} tensors", ts.len());
    for (n, t) in ts.iter().take(8) {
        println!("{n} {:?} {:?}", t.dims(), t.dtype());
    }
    Ok(())
}
