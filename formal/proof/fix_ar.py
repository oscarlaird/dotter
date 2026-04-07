import re

with open('Playground/correctness/_scratch_ar.lean', 'r') as f:
    content = f.read()

# Fix hmax -> bounds
content = content.replace('exact hash.hmax', 'exact hcast_h_bound')
content = content.replace('exact ps.hmax', 'exact hcast_ps_bound')
content = content.replace('exact right_hash.hmax', 'exact hcast_rh_bound')
content = content.replace('exact b.hmax', 'exact hcast_b_bound')

# Fix dif_pos issue in hi1
old_hi1 = """    have h_if : decide (hash.val * ps.val < 2 ^ UScalarTy.U128.numBits) = true := decide_eq_true h_bound
    rw [dif_pos h_if]
    refine congrArg ok ?_
    refine UScalar.eq_of_val_eq ?_
    rfl"""

new_hi1 = """    split_ifs with h_if
    · refine congrArg ok ?_
      refine UScalar.eq_of_val_eq ?_
      rfl
    · exfalso
      have h_true : decide (hash.val * ps.val < 2 ^ UScalarTy.U128.numBits) = true := decide_eq_true h_bound
      exact h_if h_true"""

content = content.replace(old_hi1, new_hi1)

# Fix dif_pos issue in hi2
old_hi2 = """    have h_if : decide (hash.val * ps.val + right_hash.val < 2 ^ UScalarTy.U128.numBits) = true := decide_eq_true h_bound
    rw [dif_pos h_if]
    refine congrArg ok ?_
    refine UScalar.eq_of_val_eq ?_
    rfl"""

new_hi2 = """    split_ifs with h_if
    · refine congrArg ok ?_
      refine UScalar.eq_of_val_eq ?_
      rfl
    · exfalso
      have h_true : decide (hash.val * ps.val + right_hash.val < 2 ^ UScalarTy.U128.numBits) = true := decide_eq_true h_bound
      exact h_if h_true"""

content = content.replace(old_hi2, new_hi2)

with open('Playground/correctness/_scratch_ar.lean', 'w') as f:
    f.write(content)

