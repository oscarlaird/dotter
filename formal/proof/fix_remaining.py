import sys

file_path = "Playground/correctness/rolling_hash.lean"
with open(file_path, "r") as f:
    content = f.read()

old_str2 = """    · rw [u64Z_zero]
      change u64Z 0#u64 = hashBytesU8 (data.val.take 0)
      have htake : data.val.take 0 = [] := List.take_zero
      rw [htake, hashBytes_nil]
      rfl"""

new_str2 = """    · rw [u64Z_zero]
      change (0 : Hash) = hashBytesU8 (data.val.take 0)
      have htake : data.val.take 0 = [] := List.take_zero
      rw [htake]
      unfold hashBytesU8
      rfl"""

if old_str2 in content:
    content = content.replace(old_str2, new_str2)
    print("Fixed hashBytes_nil")
else:
    print("Could not find hashBytes_nil string")

with open(file_path, "w") as f:
    f.write(content)
