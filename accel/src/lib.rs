use blake2::digest::{Update, VariableOutput};
use blake2::Blake2bVar;
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

const DEFAULT_SIMHASH_MAX_TOKENS: usize = 20000;
const DEFAULT_MINHASH_MAX_SHINGLES: usize = 20000;
const PRIME32: u64 = 4294967311;
const MINHASH_SENTINEL: u64 = PRIME32;
const ADLER_MOD: u32 = 65521;

#[derive(Debug)]
enum HashError {
    Init,
    Finalize,
}

impl std::fmt::Display for HashError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HashError::Init => write!(f, "blake2b init failed"),
            HashError::Finalize => write!(f, "blake2b finalize failed"),
        }
    }
}

fn is_ascii_alpha(b: u8) -> bool {
    (b'A'..=b'Z').contains(&b) || (b'a'..=b'z').contains(&b)
}

fn is_ascii_alnum_or_underscore(b: u8) -> bool {
    is_ascii_alpha(b) || (b'0'..=b'9').contains(&b) || b == b'_'
}

fn is_stopword(token: &[u8]) -> bool {
    matches!(
        token,
        b"the"
            | b"and"
            | b"for"
            | b"that"
            | b"with"
            | b"this"
            | b"from"
            | b"have"
            | b"your"
            | b"into"
            | b"such"
            | b"when"
            | b"where"
            | b"which"
            | b"while"
            | b"there"
            | b"their"
            | b"been"
            | b"than"
            | b"also"
            | b"more"
            | b"would"
            | b"could"
            | b"should"
            | b"will"
            | b"can"
            | b"about"
            | b"each"
            | b"other"
            | b"some"
            | b"most"
            | b"many"
            | b"very"
            | b"over"
            | b"under"
            | b"between"
            | b"these"
            | b"those"
            | b"them"
            | b"then"
            | b"here"
            | b"onto"
            | b"upon"
            | b"using"
            | b"used"
            | b"use"
            | b"based"
            | b"within"
            | b"across"
            | b"say"
            | b"her"
            | b"she"
    )
}

fn token_hash64(token: &[u8]) -> Result<u64, HashError> {
    let mut hasher = Blake2bVar::new(8).map_err(|_| HashError::Init)?;
    hasher.update(token);
    let mut out = [0u8; 8];
    hasher
        .finalize_variable(&mut out)
        .map_err(|_| HashError::Finalize)?;
    Ok(u64::from_le_bytes(out))
}

fn normalize_max_tokens(max_tokens: Option<i64>) -> Option<usize> {
    match max_tokens {
        None => Some(DEFAULT_SIMHASH_MAX_TOKENS),
        Some(val) if val <= 0 => Some(0),
        Some(val) => Some(usize::try_from(val).unwrap_or(usize::MAX)),
    }
}

fn normalize_max_shingles(max_shingles: Option<i64>) -> Option<usize> {
    match max_shingles {
        None => Some(DEFAULT_MINHASH_MAX_SHINGLES),
        Some(val) if val <= 0 => None,
        Some(val) => Some(usize::try_from(val).unwrap_or(usize::MAX)),
    }
}

fn simhash64_impl(text: &str, max_tokens: Option<usize>) -> Result<u64, HashError> {
    let limit = max_tokens.unwrap_or(DEFAULT_SIMHASH_MAX_TOKENS);
    if limit == 0 {
        return Ok(0);
    }

    let bytes = text.as_bytes();
    let mut v = [0i32; 64];
    let mut i = 0usize;
    let mut seen = 0usize;

    while i < bytes.len() {
        let b = bytes[i];
        if is_ascii_alpha(b) {
            let mut token: Vec<u8> = Vec::new();
            token.push(b.to_ascii_lowercase());
            i += 1;
            while i < bytes.len() {
                let b2 = bytes[i];
                if is_ascii_alnum_or_underscore(b2) {
                    token.push(b2.to_ascii_lowercase());
                    i += 1;
                } else {
                    break;
                }
            }
            if token.len() >= 4 && !is_stopword(&token) {
                let h = token_hash64(&token)?;
                for bit in 0..64 {
                    if (h >> bit) & 1 == 1 {
                        v[bit] += 1;
                    } else {
                        v[bit] -= 1;
                    }
                }
                seen += 1;
                if seen >= limit {
                    break;
                }
            }
        } else {
            i += 1;
        }
    }

    let mut out = 0u64;
    for (idx, val) in v.iter().enumerate() {
        if *val > 0 {
            out |= 1u64 << idx;
        }
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (text, max_tokens=None))]
fn simhash64(py: Python, text: &str, max_tokens: Option<i64>) -> PyResult<u64> {
    let text_owned = text.to_owned();
    let limit = normalize_max_tokens(max_tokens);
    let result = py.allow_threads(|| simhash64_impl(&text_owned, limit));
    result.map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

fn adler32(data: &[u8]) -> u32 {
    let mut s1: u32 = 1;
    let mut s2: u32 = 0;
    for &b in data {
        s1 = (s1 + b as u32) % ADLER_MOD;
        s2 = (s2 + s1) % ADLER_MOD;
    }
    (s2 << 16) | s1
}

#[pyfunction]
#[pyo3(signature = (text, k, coeffs, max_shingles=None))]
fn minhash_signature_with_coeffs(
    py: Python,
    text: &str,
    k: usize,
    coeffs: Vec<(u64, u64)>,
    max_shingles: Option<i64>,
) -> PyResult<Vec<u64>> {
    let text_owned = text.to_owned();
    let max_shingles = normalize_max_shingles(max_shingles);
    let result = py.allow_threads(|| {
        if k == 0 {
            return vec![MINHASH_SENTINEL; coeffs.len()];
        }

        let mut bytes = text_owned.as_bytes();
        if bytes.len() < k {
            return vec![MINHASH_SENTINEL; coeffs.len()];
        }

        if let Some(limit) = max_shingles {
            let byte_limit = limit.saturating_add(k.saturating_sub(1));
            if bytes.len() > byte_limit {
                bytes = &bytes[..byte_limit];
            }
        }

        let mut shingles = std::collections::HashSet::new();
        if bytes.len() >= k {
            for i in 0..=bytes.len() - k {
                let gram = &bytes[i..i + k];
                if !gram.iter().any(|&c| c > 32) {
                    continue;
                }
                shingles.insert(adler32(gram));
            }
        }

        if shingles.is_empty() {
            return vec![MINHASH_SENTINEL; coeffs.len()];
        }

        let mut sig = vec![MINHASH_SENTINEL; coeffs.len()];
        for x in shingles {
            let x64 = x as u64;
            for (idx, (a, b)) in coeffs.iter().enumerate() {
                let v = ((u128::from(*a) * u128::from(x64)) + u128::from(*b))
                    % u128::from(PRIME32);
                let v = v as u64;
                if v < sig[idx] {
                    sig[idx] = v;
                }
            }
        }

        sig
    });
    Ok(result)
}

#[pymodule]
fn _native(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let qc = PyModule::new(py, "qc")?;
    qc.add_function(wrap_pyfunction!(simhash64, &qc)?)?;
    qc.add_function(wrap_pyfunction!(minhash_signature_with_coeffs, &qc)?)?;
    m.add_submodule(&qc)?;
    Ok(())
}
