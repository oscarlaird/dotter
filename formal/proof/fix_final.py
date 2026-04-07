with open('Playground/correctness/rolling_hash.lean', 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "rw [hhash, hps2, hright]" in line and "exact ⟨rfl" not in lines[i+1]:
        lines.insert(i+1, "  exact ⟨rfl, by rw [hfast_mod_val]; exact Nat.mod_lt _ (by unfold M; omega)⟩\n")
    if "exact ⟨by rw [h_zmod_eq], hextend_bound⟩" in line:
        lines[i] = "  exact ⟨by rw [h_zmod_eq]; rfl, hextend_bound⟩\n"

with open('Playground/correctness/rolling_hash.lean', 'w') as f:
    f.writelines(lines)
