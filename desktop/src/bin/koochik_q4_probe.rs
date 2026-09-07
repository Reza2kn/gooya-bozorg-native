//! Native tract Q4_0 candidate. Always evaluates a serialized/reloaded NNEF artifact.
use anyhow::{Result, ensure};
use gooya_native_desktop::koochik;
use std::{fs, io::Write, path::PathBuf};
use tract_core::{ops::matmul::de_block_quant::BlockQuantTransform, transform::ModelTransform};
use tract_onnx::prelude::*;
fn main() -> Result<()> {
    let a: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    ensure!(
        a.len() == 3,
        "usage: koochik_q4_probe STEP.onnx INPUTS.json OUTPUT_DIR"
    );
    fs::create_dir_all(&a[2])?;
    let inputs = koochik::read_inputs(&a[1])?;
    let mut m = tract_onnx::onnx().model_for_path(&a[0])?;
    let dynamic = std::env::var_os("GOOYA_Q4_DYNAMIC").is_some();
    if !dynamic {
        for (i, t) in inputs.iter().enumerate() {
            m.set_input_fact(i, InferenceFact::dt_shape(t.datum_type(), t.shape()))?;
        }
    }
    let mut m = m.into_typed()?.into_decluttered()?;
    // tract 0.23.4's block transform expects the constant operand in slot zero.
    // Swap the operands and their axis mappings together; the contraction is unchanged.
    for id in m.eval_order()? {
        let node = m.node(id);
        if let Some(op) = node.op_as::<tract_core::ops::einsum::EinSum>() {
            if node.inputs.len() == 2
                && m.outlet_fact(node.inputs[1])?.konst.is_some()
                && m.outlet_fact(node.inputs[0])?.konst.is_none()
            {
                let mut op = op.clone();
                for axis in op.axes.iter_all_axes_mut() {
                    axis.inputs.swap(0, 1);
                }
                let swapped = [node.inputs[1], node.inputs[0]];
                TypedModelPatch::replace_single_op(&m, node, &swapped, op)?.apply(&mut m)?;
            }
        }
    }
    BlockQuantTransform.transform(&mut m)?;
    let path = a[2].join("step-q4.nnef.tar");
    tract_nnef::nnef().write_to_tar(&m, fs::File::create(&path)?)?;
    drop(m);
    let bytes = fs::metadata(&path)?.len();
    eprintln!("Serialized Q4 candidate: {bytes} bytes");
    if std::env::var_os("GOOYA_Q4_EXPORT_ONLY").is_some() {
        return Ok(());
    }
    let options = tract_core::runtime::RunOptions {
        executor: Some(tract_linalg::multithread::Executor::multithread(6)),
        ..Default::default()
    };
    let model = tract_nnef::nnef()
        .model_for_path(&path)?
        .into_optimized()?
        .into_runnable_with_options(&options)?;
    let out = model.run(inputs.iter().cloned().map(|x| x.into_tvalue()).collect())?;
    let values = out[0].to_plain_array_view::<f32>()?;
    let mut f = fs::File::create(a[2].join("logits.f32"))?;
    for &v in values.iter() {
        f.write_all(&v.to_le_bytes())?;
    }
    fs::write(
        a[2].join("export.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"runtime":"tract-0.23.4","quantization":"native Q4_0 matmul weights","artifact_bytes":bytes,"shape":values.shape(),"promotion":false}),
        )?,
    )?;
    let fixture_path = a[1]
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("decode.json"));
    if let Some(p) = fixture_path.filter(|p| p.exists()) {
        let fixture: koochik::DecodeFixture = serde_json::from_slice(&fs::read(p)?)?;
        let codes = koochik::decode(&koochik::Graph::from_plan(model), inputs, &fixture)?;
        let agreement = codes
            .iter()
            .zip(&fixture.source_codes)
            .filter(|(a, b)| a == b)
            .count() as f64
            / codes.len() as f64;
        fs::write(a[2].join("codes.json"), serde_json::to_vec(&codes)?)?;
        fs::write(
            a[2].join("full-parity.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"code_agreement":agreement,"passes_strict_code_gate":agreement>0.98,"promotion":false}),
            )?,
        )?;
        eprintln!("Full Q4 code agreement: {agreement}");
    }
    Ok(())
}
