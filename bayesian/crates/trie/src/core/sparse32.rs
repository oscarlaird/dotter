use crate::safe_float::Float;

pub(super) fn sparse_to_dense32(a: &[Float], nonzeros: u32, default: Float) -> [Float; 32] {
    let mut res = [default; 32];
    let mut mask_after_bit = 0;
    for i in 0..32 {
        if (nonzeros & (1 << i)) != 0 {
            res[i] = a[(mask_after_bit & nonzeros).count_ones() as usize];
        }
        mask_after_bit <<= 1;
        mask_after_bit |= 1;
    }
    res
}

pub(super) fn dense32_to_sparse(a: &[Float; 32], nonzeros: u32) -> Vec<Float> {
    // TODO: allocating a vec here is unwise
    let mut res = Vec::new();
    for i in 0..32 {
        if (nonzeros & (1 << i)) != 0 {
            res.push(a[i]);
        }
    }
    res
}
