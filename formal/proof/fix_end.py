import re

with open('Playground/correctness/rolling_hash.lean', 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "have hps2 : (ps.val : ZMod M) = B ^ right_length.val := hps_val" in line:
        if "rw [hhash, hps2, hright]" in lines[i+1]:
            lines[i+1] = "  rw [hhash, hps2, hright]\n"

    if "exact h_zmod_eq" in line and "append_right_spec" in "".join(lines[max(0, i-40):i]):
        lines[i] = "  exact ⟨by rw [h_zmod_eq]; rfl, hextend_bound⟩\n"

with open('Playground/correctness/rolling_hash.lean', 'w') as f:
    f.writelines(lines)

