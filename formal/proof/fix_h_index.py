import sys

file_path = "Playground/correctness/rolling_hash.lean"
with open(file_path, "r") as f:
    content = f.read()

old_str = """  have h_index : Slice.index_usize data i = ok (data.val[i.val]'h_ilt) := by
    unfold Slice.index_usize
    have h_get : data[i]? = some (data.val[i.val]'h_ilt) := by
      exact getElem?_pos ..
    simp [h_get]"""

new_str = """  have h_index : Slice.index_usize data i = ok (data.val[i.val]'h_ilt) := by
    unfold Slice.index_usize
    rw [Slice.getElem?_Usize_eq, getElem?_pos _ _ h_ilt]
    rfl"""

if old_str in content:
    content = content.replace(old_str, new_str)
    with open(file_path, "w") as f:
        f.write(content)
    print("Fixed h_index")
else:
    print("Could not find h_index string")

