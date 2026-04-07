import re

with open('Playground/correctness/rolling_hash.lean', 'r') as f:
    content = f.read()

# We want to move `noncomputable def asUInt8` and `hashBytesU8` and `hashBytes_map_asUInt8` to BEFORE `theorem extend_right_spec` or just `theorem append_right_spec`.
# Actually, the simplest is to just move the block from `noncomputable def asUInt8` down to `hashBytes_map_asUInt8` before the new theorems, which start with `instance : NeZero M`.
# Let's find the inserted block: starts with `instance : NeZero M := ⟨by decide⟩` and ends before `noncomputable def asUInt8`.
# It's currently right before `noncomputable def asUInt8`.
# Let's extract `noncomputable def asUInt8 ... hashBytes_map_asUInt8 (bs : List U8) :\n    hashBytes (bs.map asUInt8) = hashBytesU8 bs := by\n  simp [hashBytes, hashBytesU8, List.foldl_map]\n\n`

# Wait, `asUInt8` only depends on `U8` and `UInt8`. So we can put it right after the imports.

match = re.search(r'(noncomputable def asUInt8.*?hashBytes \([^)]*\) = hashBytesU8 bs := by\n  simp \[hashBytes, hashBytesU8, List\.foldl_map\]\n)', content, re.DOTALL)
if match:
    block = match.group(1)
    # Remove block
    content = content.replace(block, "")
    
    # Insert block before `instance : NeZero M`
    insert_idx = content.find('instance : NeZero M := ⟨by decide⟩')
    if insert_idx != -1:
        content = content[:insert_idx] + block + "\n" + content[insert_idx:]
    
    with open('Playground/correctness/rolling_hash.lean', 'w') as f:
        f.write(content)
else:
    print("Could not find asUInt8 block")

