//! Direct Core ML prediction for research bundles. No Python or ORT speech path.
// objc 0.2 macros emit their legacy cargo-clippy cfg in this module.
#![allow(unsafe_op_in_unsafe_fn, unexpected_cfgs)]
use anyhow::{Context, Result, ensure};
use objc::{
    msg_send,
    rc::StrongPtr,
    runtime::{Class, Object},
    sel, sel_impl,
};
use std::{
    ffi::{CStr, CString},
    path::Path,
    ptr,
    sync::Mutex,
};
use tract_onnx::prelude::*;
#[link(name = "CoreML", kind = "framework")]
unsafe extern "C" {}
#[link(name = "Foundation", kind = "framework")]
unsafe extern "C" {}
struct State {
    fp32: StrongPtr,
    fp16: Option<StrongPtr>,
    length: usize,
    output: StrongPtr,
    step: usize,
    steps: usize,
}
// MLModel predictions and lifetime changes are serialized by Model's mutex.
// The models have no thread-affine UI state and no Objective-C object escapes run().
unsafe impl Send for State {}
pub struct Model {
    state: Mutex<State>,
}
unsafe fn cls(name: &str) -> Result<&'static Class> {
    Class::get(name).with_context(|| format!("missing Objective-C class {name}"))
}
unsafe fn string(s: &str) -> Result<*mut Object> {
    let c = CString::new(s)?;
    Ok(msg_send![cls("NSString")?, stringWithUTF8String:c.as_ptr()])
}
unsafe fn error(e: *mut Object) -> String {
    if e.is_null() {
        return "Core ML returned nil without NSError".into();
    }
    let description: *mut Object = msg_send![e, localizedDescription];
    let text: *const std::ffi::c_char = msg_send![description, UTF8String];
    if text.is_null() {
        "Core ML error".into()
    } else {
        CStr::from_ptr(text).to_string_lossy().into_owned()
    }
}
unsafe fn numbers(object: *mut Object) -> Vec<usize> {
    let count: usize = msg_send![object, count];
    (0..count)
        .map(|i| {
            let n: *mut Object = msg_send![object,objectAtIndex:i];
            let v: u64 = msg_send![n, unsignedLongLongValue];
            v as usize
        })
        .collect()
}
unsafe fn load(path: &Path) -> Result<StrongPtr> {
    ensure!(
        path.extension().is_some_and(|s| s == "mlmodelc"),
        "Core ML requires a compiled .mlmodelc directory"
    );
    ensure!(path.is_dir(), "missing Core ML model {}", path.display());
    let url: *mut Object = msg_send![cls("NSURL")?, fileURLWithPath:string(path.to_str().context("non-UTF8 model path")?)?];
    let configuration: *mut Object = msg_send![cls("MLModelConfiguration")?, new];
    let configuration = StrongPtr::new(configuration);
    let _: () = msg_send![*configuration,setComputeUnits:2isize]; // MLComputeUnitsAll
    let mut e: *mut Object = ptr::null_mut();
    let model: *mut Object = msg_send![cls("MLModel")?,modelWithContentsOfURL:url configuration:*configuration error:&mut e];
    ensure!(!model.is_null(), "{}", error(e));
    Ok(StrongPtr::retain(model))
}
unsafe fn description(model: *mut Object) -> Result<(usize, StrongPtr)> {
    let d: *mut Object = msg_send![model, modelDescription];
    let inputs: *mut Object = msg_send![d, inputDescriptionsByName];
    let ids: *mut Object = msg_send![inputs,objectForKey:string("ids")?];
    ensure!(!ids.is_null(), "Core ML ids input missing");
    let constraint: *mut Object = msg_send![ids, multiArrayConstraint];
    let shape: *mut Object = msg_send![constraint, shape];
    let shape = numbers(shape);
    ensure!(
        shape.len() == 3 && shape[0] == 2 && shape[1] == 8,
        "unsupported Core ML input shape {shape:?}"
    );
    let outputs: *mut Object = msg_send![d, outputDescriptionsByName];
    let keys: *mut Object = msg_send![outputs, allKeys];
    let count: usize = msg_send![keys, count];
    ensure!(count == 1, "expected one Core ML output");
    let key: *mut Object = msg_send![keys,objectAtIndex:0usize];
    Ok((shape[2], StrongPtr::retain(key)))
}
unsafe fn array(shape: &[usize], dtype: isize) -> Result<StrongPtr> {
    let dims: *mut Object = msg_send![cls("NSMutableArray")?, array];
    for &d in shape {
        let n: *mut Object = msg_send![cls("NSNumber")?,numberWithUnsignedLongLong:d as u64];
        let _: () = msg_send![dims,addObject:n];
    }
    let allocated: *mut Object = msg_send![cls("MLMultiArray")?, alloc];
    let mut e: *mut Object = ptr::null_mut();
    let a: *mut Object = msg_send![allocated,initWithShape:dims dataType:dtype error:&mut e];
    ensure!(!a.is_null(), "{}", error(e));
    Ok(StrongPtr::new(a))
}
impl Model {
    pub fn from_env() -> Result<Self> {
        objc::rc::autoreleasepool(|| unsafe {
            let fp32 = load(Path::new(
                &std::env::var("GOOYA_KOOCHIK_COREML_FP32")
                    .context("set GOOYA_KOOCHIK_COREML_FP32 to a compiled model")?,
            ))?;
            let (length, output) = description(*fp32)?;
            ensure!((1..=2048).contains(&length), "unsupported Core ML capacity");
            let fp16 = if std::env::var_os("GOOYA_KOOCHIK_COREML_MIXED").is_some() {
                let m = load(Path::new(
                    &std::env::var("GOOYA_KOOCHIK_COREML_FP16")
                        .context("mixed Core ML requires FP16 model")?,
                ))?;
                ensure!(
                    description(*m)?.0 == length,
                    "Core ML precision models have different shapes"
                );
                Some(m)
            } else {
                None
            };
            eprintln!(
                "Core ML native: sequence capacity {length}, mixed={}",
                fp16.is_some()
            );
            Ok(Self {
                state: Mutex::new(State {
                    fp32,
                    fp16,
                    length,
                    output,
                    step: 0,
                    steps: 24,
                }),
            })
        })
    }
    pub fn begin_decode(&self, steps: usize) -> Result<()> {
        let mut s = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("Core ML lock poisoned"))?;
        ensure!(
            s.fp16.is_none() || steps == 24,
            "mixed Core ML policy requires 24 passes"
        );
        s.step = 0;
        s.steps = steps;
        Ok(())
    }
    pub fn run(&self, inputs: &[Tensor]) -> Result<TVec<TValue>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow::anyhow!("Core ML lock poisoned"))?;
        objc::rc::autoreleasepool(|| unsafe {
            ensure!(inputs.len() == 4, "Core ML expects four inputs");
            let shape = inputs[0].shape();
            ensure!(
                shape.len() == 3 && shape[0] == 2 && shape[1] == 8,
                "invalid speech shape"
            );
            let s = shape[2];
            let l = state.length;
            ensure!(s <= l, "phrase exceeds Core ML sequence capacity {l}");
            ensure!(
                inputs[1].shape() == [2, s]
                    && inputs[2].shape() == [2, 1, s, s]
                    && inputs[3].shape() == [2, s],
                "invalid Core ML conditioning shapes"
            );
            let ids = inputs[0].to_plain_array_view::<i64>()?;
            let mask = inputs[1].to_plain_array_view::<bool>()?;
            let attention = inputs[2].to_plain_array_view::<f32>()?;
            let pos = inputs[3].to_plain_array_view::<i64>()?;
            let ids = ids.as_slice().context("noncontiguous ids")?;
            let mask = mask.as_slice().context("noncontiguous mask")?;
            let attention = attention.as_slice().context("noncontiguous attention")?;
            let pos = pos.as_slice().context("noncontiguous positions")?;
            let mut pid = vec![1024i32; 2 * 8 * l];
            let mut pmask = vec![1i32; 2 * l];
            let mut ppos = vec![0i32; 2 * l];
            let mut patt = vec![f32::NEG_INFINITY; 2 * l * l];
            for b in 0..2 {
                for j in 0..s {
                    pmask[b * l + j] = mask[b * s + j] as i32;
                    ppos[b * l + j] = i32::try_from(pos[b * s + j])?;
                    patt[b * l * l + j * l..b * l * l + j * l + s]
                        .copy_from_slice(&attention[b * s * s + j * s..b * s * s + j * s + s]);
                }
                for j in s..l {
                    patt[b * l * l + j * l + j] = 0.;
                }
                for c in 0..8 {
                    for j in 0..s {
                        pid[(b * 8 + c) * l + j] = i32::try_from(ids[(b * 8 + c) * s + j])?;
                    }
                }
            }
            let arrays = [
                array(&[2, 8, l], 0x20000 | 32)?,
                array(&[2, l], 0x20000 | 32)?,
                array(&[2, 1, l, l], 0x10000 | 32)?,
                array(&[2, l], 0x20000 | 32)?,
            ];
            let bytes: [&[u8]; 4] = [
                std::slice::from_raw_parts(pid.as_ptr().cast(), pid.len() * 4),
                std::slice::from_raw_parts(pmask.as_ptr().cast(), pmask.len() * 4),
                std::slice::from_raw_parts(patt.as_ptr().cast(), patt.len() * 4),
                std::slice::from_raw_parts(ppos.as_ptr().cast(), ppos.len() * 4),
            ];
            let dictionary: *mut Object = msg_send![cls("NSMutableDictionary")?, dictionary];
            for (i, name) in ["ids", "mask", "attention", "pos"].iter().enumerate() {
                let strides: *mut Object = msg_send![*arrays[i], strides];
                let dims: *mut Object = msg_send![*arrays[i], shape];
                let dims = numbers(dims);
                let strides = numbers(strides);
                let mut expected = 1;
                for axis in (0..dims.len()).rev() {
                    ensure!(
                        strides[axis] == expected,
                        "noncontiguous Core ML input allocation"
                    );
                    expected *= dims[axis];
                }
                let pointer: *mut std::ffi::c_void = msg_send![*arrays[i], dataPointer];
                ensure!(!pointer.is_null(), "nil Core ML data");
                ptr::copy_nonoverlapping(bytes[i].as_ptr(), pointer.cast(), bytes[i].len());
                let value: *mut Object =
                    msg_send![cls("MLFeatureValue")?,featureValueWithMultiArray:*arrays[i]];
                let _: () = msg_send![dictionary,setObject:value forKey:string(name)?];
            }
            let allocated: *mut Object = msg_send![cls("MLDictionaryFeatureProvider")?, alloc];
            let mut e: *mut Object = ptr::null_mut();
            let provider: *mut Object =
                msg_send![allocated,initWithDictionary:dictionary error:&mut e];
            ensure!(!provider.is_null(), "{}", error(e));
            let provider = StrongPtr::new(provider);
            let use_fp16 = state.fp16.is_some() && state.step >= 8 && state.step < state.steps - 8;
            let model = if use_fp16 {
                **state.fp16.as_ref().unwrap()
            } else {
                *state.fp32
            };
            let result: *mut Object =
                msg_send![model,predictionFromFeatures:*provider error:&mut e];
            ensure!(!result.is_null(), "{}", error(e));
            let feature: *mut Object = msg_send![result,featureValueForName:*state.output];
            ensure!(!feature.is_null(), "Core ML output missing");
            let output: *mut Object = msg_send![feature, multiArrayValue];
            ensure!(!output.is_null(), "Core ML output is not a multi-array");
            let dtype: isize = msg_send![output, dataType];
            ensure!(
                dtype == (0x10000 | 32) || dtype == (0x10000 | 16),
                "Core ML output must be FP32 or FP16, got {dtype}"
            );
            let dims: *mut Object = msg_send![output, shape];
            ensure!(
                numbers(dims) == [2, 8, l, 1025],
                "unexpected Core ML logit shape"
            );
            // Core ML may pad the physical buffer; count is only logical elements.
            // Read strides and copy while the SDK guarantees buffer validity.
            let copied = std::rc::Rc::new(std::cell::RefCell::new(None));
            let destination = copied.clone();
            let handler =
                block::ConcreteBlock::new(move |bytes: *const std::ffi::c_void, size: isize| {
                    let read = (|| -> anyhow::Result<Vec<f32>> {
                        let stride: *mut Object = msg_send![output, strides];
                        let stride = numbers(stride);
                        ensure!(stride.len() == 4, "bad output strides");
                        let dimensions = [2, 8, l, 1025];
                        let mut span = 1usize;
                        for axis in 0..4 {
                            span = span
                                .checked_add(
                                    (dimensions[axis] - 1)
                                        .checked_mul(stride[axis])
                                        .context("Core ML stride overflow")?,
                                )
                                .context("Core ML span overflow")?;
                        }
                        let width = if dtype == (0x10000 | 16) { 2 } else { 4 };
                        let required = span
                            .checked_mul(width)
                            .context("Core ML byte span overflow")?;
                        ensure!(
                            !bytes.is_null() && size >= 0 && required <= size as usize,
                            "Core ML output buffer is smaller than its strided shape"
                        );
                        let mut values = Vec::with_capacity(2 * 8 * s * 1025);
                        for b in 0..2 {
                            for c in 0..8 {
                                for j in 0..s {
                                    let offset = b * stride[0] + c * stride[1] + j * stride[2];
                                    for k in 0..1025 {
                                        let index = offset + k * stride[3];
                                        values.push(if width == 2 {
                                            half::f16::from_bits(
                                                bytes.cast::<u16>().add(index).read_unaligned(),
                                            )
                                            .to_f32()
                                        } else {
                                            bytes.cast::<f32>().add(index).read_unaligned()
                                        });
                                    }
                                }
                            }
                        }
                        Ok(values)
                    })();
                    *destination.borrow_mut() = Some(read);
                })
                .copy();
            let _: () = msg_send![output, getBytesWithHandler: &*handler];
            let values = copied
                .borrow_mut()
                .take()
                .context("Core ML did not supply output bytes")??;
            state.step += 1;
            Ok(tvec![
                Tensor::from_shape(&[2, 8, s, 1025], &values)?.into_tvalue()
            ])
        })
    }
}
