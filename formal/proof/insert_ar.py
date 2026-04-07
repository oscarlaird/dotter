import re

with open('Playground/correctness/_scratch_ar.lean', 'r') as f:
    content = f.read()

# Extract from 'instance : NeZero M' down to 'end Playground.Correctness.RollingHash'
match = re.search(r'(instance : NeZero M := ⟨by decide⟩.*?)\nend Playground\.Correctness\.RollingHash', content, re.DOTALL)
if not match:
    print("Could not find content in _scratch_ar.lean")
    exit(1)

to_insert = match.group(1)

with open('Playground/correctness/rolling_hash.lean', 'r') as f:
    rh_content = f.read()

# Replace all occurrences of power_shift_exp_zero and similar empty/zero tests
# since we have full specs now? Actually, no, let's just keep them and insert before asUInt8
insertion_point = "noncomputable def asUInt8"
if insertion_point not in rh_content:
    print("Could not find insertion point")
    exit(1)

rh_content = rh_content.replace(insertion_point, to_insert + "\n" + insertion_point)

with open('Playground/correctness/rolling_hash.lean', 'w') as f:
    f.write(rh_content)

