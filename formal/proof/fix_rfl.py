import sys

file_path = "Playground/correctness/rolling_hash.lean"
with open(file_path, "r") as f:
    content = f.read()

old_str1 = """    rw [Slice.getElem?_Usize_eq]
    have h_list_get : data.val[i.val]? = some (data.val[i.val]'h_ilt) := getElem?_pos ..
    rw [h_list_get]
    rfl"""

new_str1 = """    rw [Slice.getElem?_Usize_eq]
    have h_list_get : data.val[i.val]? = some (data.val[i.val]'h_ilt) := getElem?_pos ..
    rw [h_list_get]"""

if old_str1 in content:
    content = content.replace(old_str1, new_str1)
    print("Fixed rfl")
else:
    print("Could not find rfl string")

with open(file_path, "w") as f:
    f.write(content)
