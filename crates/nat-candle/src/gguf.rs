//! WP-1.4 / g3-gguf — GGUF export.
//!
//! Serializes a NAT model's weights into a **GGUF** container (lossless F32), the
//! format the llama.cpp / Ollama ecosystem reads. candle is GGUF-native, so this is
//! a clean fit: every parameter is written as an F32 tensor with NAT metadata
//! (architecture, vocab, width, zones). The file round-trips through candle's own
//! GGUF reader.
//!
//! Honest scope: this produces a **valid GGUF container of NAT's weights** — the
//! "GGUF round-trip" half of g3-gguf. Making it *execute* in stock Ollama also
//! requires conforming to a llama.cpp-recognized architecture, which the
//! zone-partitioned NAT graph is not; that runtime mapping is a separate effort.

use candle_core::quantized::gguf_file::Value;
use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{Device, Result, Tensor};
use std::path::Path;

/// Write named tensors + metadata to a GGUF file (each tensor stored as lossless
/// F32). Tensors are moved to CPU and made contiguous before quantization.
pub fn export(tensors: &[(String, Tensor)], metadata: &[(&str, Value)], path: &Path) -> Result<()> {
    let qts: Vec<(String, QTensor)> = tensors
        .iter()
        .map(|(n, t)| {
            let cpu = t.to_device(&Device::Cpu)?.contiguous()?;
            Ok((n.clone(), QTensor::quantize(&cpu, GgmlDType::F32)?))
        })
        .collect::<Result<_>>()?;
    let t_refs: Vec<(&str, &QTensor)> = qts.iter().map(|(n, q)| (n.as_str(), q)).collect();
    let m_refs: Vec<(&str, &Value)> = metadata.iter().map(|(k, v)| (*k, v)).collect();
    let mut f = std::fs::File::create(path).map_err(candle_core::Error::wrap)?;
    gguf_file::write(&mut f, &m_refs, &t_refs)
}

/// The tensor names in a GGUF file (round-trip verification).
pub fn tensor_names(path: &Path) -> Result<Vec<String>> {
    let mut f = std::fs::File::open(path).map_err(candle_core::Error::wrap)?;
    let content = gguf_file::Content::read(&mut f)?;
    Ok(content.tensor_infos.keys().cloned().collect())
}

/// Read one tensor back, dequantized (for value round-trip checks).
pub fn read_tensor(path: &Path, name: &str) -> Result<Tensor> {
    let mut f = std::fs::File::open(path).map_err(candle_core::Error::wrap)?;
    let content = gguf_file::Content::read(&mut f)?;
    content
        .tensor(&mut f, name, &Device::Cpu)?
        .dequantize(&Device::Cpu)
}

/// A string metadata value (sugar for building a metadata list).
pub fn s(v: &str) -> Value {
    Value::String(v.to_string())
}
/// A u32 metadata value.
pub fn u(v: usize) -> Value {
    Value::U32(v as u32)
}
