use std::sync::LazyLock;

use pyo3::prelude::*;
use numpy::pyo3::Python;
use numpy::{IntoPyArray, ToPyArray};

const S_BOX : [u8; 256] = [
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
];

static INV_S_BOX : LazyLock<[u8; 256]> = std::sync::LazyLock::new(compute_inv_sbox);

fn compute_inv_sbox() -> [u8; 256] {
    let mut inv_sbox = [0u8; 256];
    for idx in 0..=255 {
        inv_sbox[S_BOX[idx as usize] as usize] = idx;
    }
    inv_sbox
}



#[pyfunction]
#[pyo3(signature = (plaintext_byte, scores))]
fn sbox_scores_to_key_scores<'py>(py: Python<'py>, plaintext_byte : u8, scores : [f32; 256]) -> PyResult<Bound<'py, numpy::PyArray<f32, numpy::ndarray::Dim<[usize; 1]>>>> {

    // Index mapping for sbox -> key
    let mut sbox_to_key = [0u8; 256];

    // Perform xor only once
    let xor_inv_sbox = INV_S_BOX.iter().map(|b| b ^ plaintext_byte).collect::<Vec<u8>>();

    for sbox_value in 0..256 {
        sbox_to_key[sbox_value] = xor_inv_sbox[sbox_value];
    }

    // The key scores after mapping from sbox to key
    let mut keyscores = [0.0f32; 256];

    for byte in 0..256 {
        keyscores[sbox_to_key[byte] as usize] = scores[byte as usize];
    }


    Ok(keyscores.to_vec().to_pyarray(py))
}

fn map_score((scores,pt_byte) : (&mut [f32], u8)) -> Vec<f32> {
    let mut sbox_to_key = [0u8; 256];

    // Perform xor only once
    let xor_inv_sbox = INV_S_BOX.iter().map(|b| b ^ pt_byte).collect::<Vec<u8>>();

    for sbox_value in 0..256 {
        sbox_to_key[sbox_value] = xor_inv_sbox[sbox_value];
    }

    // The key scores after mapping from sbox to key
    let mut keyscores = [0.0f32; 256];

    for byte in 0..256 {
        keyscores[sbox_to_key[byte] as usize] = scores[byte as usize];
    }

    keyscores.to_vec()
}


use numpy::PyArrayMethods;
use rayon::prelude::*;

#[pyfunction]
#[pyo3(signature = (plaintext_bytes, scores))]
fn sbox_scores_to_keyscores_parallel<'py>(
    py: Python<'py>, 
    plaintext_bytes : Vec<u8>,
    scores : Bound<'py, numpy::PyArray<f32, numpy::ndarray::Dim<[usize; 2]>>>
) -> PyResult<Bound<'py, numpy::PyArray<f32, numpy::ndarray::Dim<[usize; 2]>>>> {

    // scores (500, 256)

    let mut scores = scores.to_vec().unwrap();

    let keyscores = scores.par_chunks_mut(256)
        .zip(plaintext_bytes)
        .map(map_score)
        .collect::<Vec<_>>();

    let pyarray = numpy::PyArray::from_vec2(py, &keyscores).unwrap();

    return Ok(pyarray);
}


#[pyfunction]
#[pyo3(signature = (plaintext_bytes))]
fn sbox_key_permutations<'py>(
    py: Python<'py>, 
    plaintext_bytes : Vec<u8>,
) -> PyResult<Bound<'py, numpy::PyArray<f32, numpy::ndarray::Dim<[usize; 2]>>>> {
    // Function for constructing permutation vectors in order to map sbox model output to
    // key during training where every trace has a different plaintext and key
    
    let keyscores = plaintext_bytes.par_iter()
        .map(|pt_byte| {
            let mut permutation = vec![0.0; 256];
            for idx in (0..256).into_iter() {
                // May seem inverted, but this is needed to map
                // from sbox to key using torch.gather
                let keyidx = INV_S_BOX[idx] ^ pt_byte;
                permutation[keyidx as usize] = idx as f32
            }
            permutation
        })
        .collect::<Vec<_>>();

    let pyarray = numpy::PyArray::from_vec2(py, &keyscores).unwrap();

    return Ok(pyarray);
}




#[pyfunction]
#[pyo3(signature = (plaintext_bytes, scores))]
fn sbox_scores_to_keyscores_2pt<'py>(
    py: Python<'py>,
    plaintext_bytes : Bound<'py, numpy::PyArray<f32, numpy::ndarray::Dim<[usize; 2]>>>,
    scores : Bound<'py, numpy::PyArray<f32, numpy::ndarray::Dim<[usize; 3]>>>
) -> PyResult<Bound<'py, numpy::PyArray<f32, numpy::ndarray::Dim<[usize; 3]>>>> {
    let outer_plaintexts = plaintext_bytes
        .to_vec()
        .unwrap()
        .into_par_iter().map(|f| f as u8)
        .collect::<Vec<_>>();
    let pt_half = outer_plaintexts.len() / 2;

    let mut outer_scores = scores.to_vec().unwrap();
    let score_half = outer_scores.len() / 2;

    let keyscores = outer_scores
        .par_chunks_mut(score_half)
        .zip(outer_plaintexts.par_chunks(pt_half))
        .map(|(scores, plaintexts)| 
            scores
                .par_chunks_mut(256)
                .zip(plaintexts)
                .map(|(score, pt_byte)|
                    map_score((score, *pt_byte))
            ).collect::<Vec<_>>()
        ).collect::<Vec<Vec<_>>>();

    let pyarray = numpy::PyArray::from_vec3(py, &keyscores).unwrap();

    return Ok(pyarray);
}

#[pymodule]
fn keyrank_rs(m : &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sbox_scores_to_keyscores_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(sbox_key_permutations, m)?)?;

    Ok(())   
}