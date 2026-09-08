//! Executes an exported Koochik graph in tract and writes little-endian f32 outputs.
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::{fs, io::Write, path::PathBuf, time::Instant};
use tract_onnx::prelude::*;
#[derive(Deserialize)]
struct Input {
    shape: Vec<usize>,
    dtype: String,
    data: serde_json::Value,
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    if args.len() != 3 {
        bail!("usage: koochik_graph_probe GRAPH.onnx INPUTS.json OUTPUT.f32");
    }
    let inputs: Vec<Input> = serde_json::from_slice(&fs::read(&args[1])?)?;
    let tensors: Vec<Tensor> = inputs
        .iter()
        .map(|x| -> Result<Tensor> {
            Ok(match x.dtype.as_str() {
                "int64" => Tensor::from_shape(
                    &x.shape,
                    &serde_json::from_value::<Vec<i64>>(x.data.clone())?,
                )?,
                "float32" => Tensor::from_shape(
                    &x.shape,
                    &serde_json::from_value::<Vec<f32>>(x.data.clone())?,
                )?,
                "bool" => Tensor::from_shape(
                    &x.shape,
                    &serde_json::from_value::<Vec<bool>>(x.data.clone())?,
                )?,
                d => bail!("unsupported input type {d}"),
            })
        })
        .collect::<Result<_>>()?;
    let start = Instant::now();
    let mut model = tract_onnx::onnx()
        .model_for_path(&args[0])
        .context("parse Koochik ONNX")?;
    for (i, t) in tensors.iter().enumerate() {
        model.set_input_fact(i, InferenceFact::dt_shape(t.datum_type(), t.shape()))?;
    }
    let graph = model
        .into_optimized()
        .context("optimize Koochik graph")?
        .into_runnable()?;
    let load_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let outputs = graph.run(tensors.into_iter().map(|x| x.into_tvalue()).collect())?;
    let values = outputs[0].to_plain_array_view::<f32>()?;
    let mut f = fs::File::create(&args[2])?;
    for &x in values.iter() {
        f.write_all(&x.to_le_bytes())?;
    }
    println!(
        "{}",
        serde_json::json!({"runtime":"tract-0.23.4","shape":values.shape(),"load_seconds":load_seconds,"run_seconds":start.elapsed().as_secs_f64()})
    );
    Ok(())
}
