with open('Playground/correctness/rolling_hash.lean', 'r') as f:
    content = f.read()

# Fix Slice.index_usize
old_idx = """  have h_index : Slice.index_usize data i = ok (data.val[i.val]'h_ilt) := by
    unfold Slice.index_usize
    rw [dif_pos h_ilt]
    rfl"""
new_idx = """  have h_index : Slice.index_usize data i = ok (data.val[i.val]'h_ilt) := by
    unfold Slice.index_usize
    have h_get : data[i]? = some (data.val[i.val]'h_ilt) := by
      exact getElem?_pos ..
    simp [h_get]"""
content = content.replace(old_idx, new_idx)

# Wait, `getElem?_pos` might need explicit arguments or just `getElem?_pos data.val i.val h_ilt`?
# Actually, the easiest way to solve `match data[i]?` is `simp [getElem?_pos, h_ilt]`? No, let's just do `simp [h_ilt]`.
# Aeneas usually has a lemma `Slice.index_usize_eq` or something. If not, `have h : data[i]? = some _ := getElem?_pos _ _ h_ilt; rw [h]; rfl`.
# Wait! In Aeneas `data[i]?` is NOT `List.get?`, it's `Slice.get?`.
# Let's check `Aeneas` Slice index. Let's just use `simp` and see.

# Fix List.take_length
old_len = "have htake : data.val.take data.len.val = data.val := List.take_length data.val"
new_len = "have htake : data.val.take data.len.val = data.val := List.take_length"
content = content.replace(old_len, new_len)

# Fix List.take_zero
old_zero = """      have htake : data.val.take 0 = [] := List.take_zero
      rw [htake, hashBytes_nil]
      rfl"""
new_zero = """      change u64Z 0#u64 = hashBytesU8 (data.val.take 0)
      have htake : data.val.take 0 = [] := List.take_zero
      rw [htake, hashBytes_nil]
      rfl"""
content = content.replace(old_zero, new_zero)

with open('Playground/correctness/rolling_hash.lean', 'w') as f:
    f.write(content)
