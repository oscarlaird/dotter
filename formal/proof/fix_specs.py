import re

with open('Playground/correctness/rolling_hash.lean', 'r') as f:
    content = f.read()

# Fix the end of extend_right_spec
old_ext_end = "rw [hhash, hps2, hright]"
new_ext_end = """  rw [hhash, hps2, hright]
  exact ⟨rfl, by rw [hfast_mod_val]; exact Nat.mod_lt _ (by unfold M; omega)⟩"""
content = content.replace(old_ext_end, new_ext_end)

# Fix hextend in append_right_spec
old_hextend = """  have hextend : ∃ r, RollingHashKernel.extend_right hash (UScalar.cast UScalarTy.U64 b) 1#usize = ok r ∧
         u64Z r = extendRight (u64Z hash) (u64Z (UScalar.cast UScalarTy.U64 b)) 1 := by"""
new_hextend = """  have hextend : ∃ r, RollingHashKernel.extend_right hash (UScalar.cast UScalarTy.U64 b) 1#usize = ok r ∧
         u64Z r = extendRight (u64Z hash) (u64Z (UScalar.cast UScalarTy.U64 b)) 1 ∧ r.val < M := by"""
content = content.replace(old_hextend, new_hextend)

# Fix the rcases hextend
old_rcases = """  rcases hextend with ⟨r, hextend_eval, hextend_val⟩
  rw [hextend_eval]
  refine ⟨r, rfl, ?_⟩
  rw [hextend_val]"""
new_rcases = """  rcases hextend with ⟨r, hextend_eval, hextend_val, hextend_bound⟩
  rw [hextend_eval]
  refine ⟨r, rfl, ?_⟩
  rw [hextend_val]"""
content = content.replace(old_rcases, new_rcases)

# Fix the end of append_right_spec
old_app_end = "exact h_zmod_eq"
new_app_end = "exact ⟨by rw [h_zmod_eq], hextend_bound⟩"
content = content.replace(old_app_end, new_app_end)

with open('Playground/correctness/rolling_hash.lean', 'w') as f:
    f.write(content)

