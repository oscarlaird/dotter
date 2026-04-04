pub(super) fn dense_to_sparse16(a: &[f32], nonzeros: u16, default: f32) -> [f32; 16] {
    let mut res = [default; 16];
    let mut mask_after_bit = 0;
    for i in 0..16 {
        if (nonzeros & (1 << i)) != 0 {
            res[i] = a[(mask_after_bit & nonzeros).count_ones() as usize];
        }
        mask_after_bit <<= 1;
        mask_after_bit |= 1;
    }
    res
}

pub(super) fn sparse16_to_dense(a: &[f32; 16], nonzeros: u16) -> Vec<f32> {
    // TODO: allocating a vec here is unwise
    let mut res = Vec::new();
    for i in 0..16 {
        if (nonzeros & (1 << i)) != 0 {
            res.push(a[i]);
        }
    }
    res
}
