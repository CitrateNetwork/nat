//! Permissive source-code connector (HERMES-S1 WP-H5) — the deterministic
//! conversion half. Walks a (cloned) repository for source files and converts
//! them to `RawDoc` passages for the CX (codec) zone.
//!
//! The repo's license is supplied by the caller (Hermes vets the `LICENSE` file
//! and passes the SPDX id); the pipeline's `ALLOWED_LICENSES` allow-list is the
//! fail-closed gate. The *clone/fetch* is the agent's network-granted job — this
//! module is pure, dependency-free walking + splitting.
//!
//! Code layout (indentation, newlines) survives the pipeline: NORMALIZE is
//! code-aware (WP-D8) — it preserves line structure and indentation, so the CX zone
//! trains on code as it is written, not a flattened token soup.

use crate::RawDoc;
use std::path::{Path, PathBuf};

/// Recognized source-code file extensions.
pub const CODE_EXTENSIONS: &[&str] = &[
    "rs", "py", "js", "ts", "jsx", "tsx", "go", "c", "h", "cpp", "hpp", "cc", "java", "rb", "php",
    "swift", "kt", "scala", "sh", "sql", "lua", "hs", "ml", "ex", "exs", "clj", "cs",
];

/// Directories never walked (VCS, vendored, generated, build output).
pub const SKIP_DIRS: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    "dist",
    "build",
    "vendor",
    ".venv",
    "__pycache__",
    ".idea",
    ".vscode",
    "out",
    "obj",
    "bin",
    ".next",
    "coverage",
];

pub fn is_skip_dir(name: &str) -> bool {
    SKIP_DIRS.contains(&name)
}

pub fn is_code_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| CODE_EXTENSIONS.contains(&e))
        .unwrap_or(false)
}

/// Recursively collect source-file paths under `root`, skipping vendored/generated
/// directories. Returned in sorted order (deterministic).
pub fn walk_code_files(root: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    walk(root, &mut out)?;
    out.sort();
    Ok(out)
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            let name = entry.file_name().to_string_lossy().to_string();
            if !is_skip_dir(&name) {
                walk(&path, out)?;
            }
        } else if is_code_file(&path) {
            out.push(path);
        }
    }
    Ok(())
}

fn slug(rel: &str) -> String {
    rel.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect()
}

/// Convert one source file's content to `RawDoc` passages: accumulate whole lines
/// up to ~`target_chars` per passage (never split mid-line). Deterministic.
/// Provenance records `code/<source>/<relpath>` and the caller-supplied license.
pub fn file_to_rawdocs(
    source: &str,
    license: &str,
    fetch_date: &str,
    relpath: &str,
    content: &str,
    target_chars: usize,
) -> Vec<RawDoc> {
    let target = target_chars.max(1);
    let base = slug(relpath);
    let mut docs = Vec::new();
    let mut buf = String::new();
    let mut idx = 0u32;

    let flush = |buf: &mut String, idx: &mut u32, docs: &mut Vec<RawDoc>| {
        let text = buf.trim();
        if !text.is_empty() {
            docs.push(RawDoc {
                id: format!("code-{source}-{base}-p{:04}", *idx),
                source: format!("code/{source}/{relpath}"),
                license: license.to_string(),
                fetch_date: fetch_date.to_string(),
                text: text.to_string(),
                modality_refs: vec![],
            });
            *idx += 1;
        }
        buf.clear();
    };

    for line in content.lines() {
        buf.push_str(line);
        buf.push('\n');
        if buf.len() >= target {
            flush(&mut buf, &mut idx, &mut docs);
        }
    }
    flush(&mut buf, &mut idx, &mut docs);
    docs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extensions_and_skip_dirs() {
        assert!(is_code_file(Path::new("src/lib.rs")));
        assert!(is_code_file(Path::new("a/b/main.py")));
        assert!(!is_code_file(Path::new("README.md")));
        assert!(!is_code_file(Path::new("LICENSE")));
        assert!(is_skip_dir("node_modules"));
        assert!(is_skip_dir("target"));
        assert!(!is_skip_dir("src"));
    }

    #[test]
    fn file_splits_into_passages_with_provenance() {
        let content = (0..40)
            .map(|i| format!("let x{i} = compute({i});"))
            .collect::<Vec<_>>()
            .join("\n");
        let docs = file_to_rawdocs("demo", "MIT", "2026-06-22", "src/lib.rs", &content, 120);
        assert!(docs.len() > 1);
        for d in &docs {
            assert_eq!(d.license, "MIT");
            assert_eq!(d.source, "code/demo/src/lib.rs");
            assert!(d.id.starts_with("code-demo-src-lib-rs-p"));
        }
    }

    #[test]
    fn walk_finds_code_and_skips_vendored() {
        let root = std::env::temp_dir().join("nat_code_walk_test");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::create_dir_all(root.join("node_modules/pkg")).unwrap();
        std::fs::write(root.join("src/main.rs"), "fn main() {}").unwrap();
        std::fs::write(root.join("README.md"), "# docs").unwrap();
        std::fs::write(root.join("node_modules/pkg/index.js"), "module.exports={}").unwrap();

        let files = walk_code_files(&root).unwrap();
        let names: Vec<String> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert!(names.contains(&"main.rs".to_string()));
        assert!(!names.iter().any(|n| n == "index.js")); // node_modules skipped
        assert!(!names.iter().any(|n| n == "README.md")); // not code
        let _ = std::fs::remove_dir_all(&root);
    }
}
