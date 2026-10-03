//! VRAM of a bare Candle CUDA context with a matmul, a conv and a quantized
//! matmul run once (the floor any model pays).
fn main() -> anyhow::Result<()> {
    use candle_core::{Device, Module, Tensor};
    let d = Device::new_cuda(0)?;
    let a = Tensor::randn(0f32, 1., (64, 1024), &d)?;
    let b = a.matmul(&a.t()?)?;
    let c = a
        .unsqueeze(0)?
        .conv1d(&Tensor::randn(0f32, 1., (8, 64, 3), &d)?, 1, 1, 1, 1)?;
    let q = candle_core::quantized::QTensor::quantize(
        &Tensor::randn(0f32, 1., (1024, 1024), &d)?,
        candle_core::quantized::GgmlDType::Q8_0,
    )?;
    let m = candle_core::quantized::QMatMul::from_qtensor(q)?;
    let y = m.forward(&a.narrow(0, 0, 1)?)?;
    let z = candle_nn::ops::softmax_last_dim(&b)?;
    println!(
        "{} {} {} {}",
        b.sum_all()?,
        c.sum_all()?,
        y.sum_all()?,
        z.sum_all()?
    );
    std::thread::sleep(std::time::Duration::from_secs(3));
    Ok(())
}
