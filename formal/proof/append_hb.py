import re

with open('Playground/correctness/rolling_hash.lean', 'r') as f:
    content = f.read()

new_content = """
lemma take_succ_eq_append_get {α : Type _} (l : List α) (i : Nat) (h : i < l.length) :
    l.take (i + 1) = l.take i ++ [l[i]] := by
  have hs : l.take i ++ [l[i]] = l.take (i + 1) := by simp
  exact hs.symm

lemma hashBytesU8_take_succ (bs : List U8) (i : Nat) (h : i < bs.length) :
    hashBytesU8 (bs.take (i + 1)) = appendRight (hashBytesU8 (bs.take i)) (asUInt8 (bs[i])) := by
  unfold hashBytesU8
  rw [take_succ_eq_append_get bs i h]
  rw [List.foldl_append]
  simp only [List.foldl_cons, List.foldl_nil]
  unfold appendRight extendRight
  ring

private lemma hash_bytes_loop_body_cont (data : Slice U8) (hash : U64) (i : Usize)
    (h_ilt : i.val < data.len.val) (hhash_bound : hash.val < M) :
    ∃ hash1 i3, RollingHashKernel.hash_bytes_loop.body data hash i = ok (cont (hash1, i3)) ∧
        u64Z hash1 = appendRight (u64Z hash) (asUInt8 (data.val[i.val]'h_ilt)) ∧
        hash1.val < M ∧ i3.val = i.val + 1 := by
  unfold RollingHashKernel.hash_bytes_loop.body
  rw [if_pos (by simpa using h_ilt)]
  have h_index : Slice.index_usize data i = ok (data.val[i.val]'h_ilt) := by
    unfold Slice.index_usize
    rw [dif_pos h_ilt]
    rfl
  rw [h_index]
  simp only [Bind.bind, Aeneas.Std.bind_ok, lift]
  have happend : ∃ r, RollingHashKernel.append_right hash (data.val[i.val]'h_ilt) = ok r ∧ u64Z r = appendRight (u64Z hash) (asUInt8 (data.val[i.val]'h_ilt)) ∧ r.val < M := by
    apply append_right_spec hash _ hhash_bound
  rcases happend with ⟨hash1, happend_eval, happend_val, happend_bound⟩
  rw [happend_eval]
  simp only [Aeneas.Std.bind_ok]
  have hi3 : (i + 1#usize : Result Usize) = ok (UScalar.ofNatCore (ty := UScalarTy.Usize) (i.val + 1) (by
      have : i.val < data.len.val := h_ilt
      have hmax : data.len.val < 2 ^ UScalarTy.Usize.numBits := data.len.hmax
      omega
    )) := by
    simp only [HAdd.hAdd, UScalar.add, UScalar.tryMk, UScalar.tryMkOpt, Result.ofOption, UScalar.check_bounds]
    split_ifs with h_if
    · refine congrArg ok ?_
      refine UScalar.eq_of_val_eq ?_
      rfl
    · exfalso
      have h_bound : i.val + 1 < 2 ^ System.Platform.numBits := by
        have : i.val < data.len.val := h_ilt
        have hmax : data.len.val < 2 ^ UScalarTy.Usize.numBits := data.len.hmax
        have heq : UScalarTy.Usize.numBits = System.Platform.numBits := UScalarTy.Usize_numBits_eq
        rw [heq] at hmax
        omega
      have h_true : decide (Add.add i.val 1 < 2 ^ System.Platform.numBits) = true := decide_eq_true h_bound
      exact h_if h_true
  rw [hi3]
  simp only [bind_ok]
  refine ⟨hash1, _, rfl, happend_val, happend_bound, rfl⟩

theorem hash_bytes_loop_spec_aux (fuel : Nat) (data : Slice U8) (hash : U64) (i : Usize)
    (hfuel : data.len.val - i.val = fuel) (hi : i.val ≤ data.len.val)
    (hhash_bound : hash.val < M)
    (hhash_val : u64Z hash = hashBytesU8 (data.val.take i.val)) :
    ∃ r, RollingHashKernel.hash_bytes_loop data hash i = ok r ∧
         u64Z r = hashBytesU8 data.val ∧ r.val < M := by
  induction fuel generalizing hash i with
  | zero =>
    have h_eq : i.val = data.len.val := by omega
    have h_ile : ¬(i.val < data.len.val) := by omega
    have h_body : RollingHashKernel.hash_bytes_loop.body data hash i = ok (done hash) := by
      unfold RollingHashKernel.hash_bytes_loop.body
      rw [if_neg (by simpa using h_ile)]
    refine ⟨hash, ?_, ?_, hhash_bound⟩
    · dsimp [RollingHashKernel.hash_bytes_loop]
      rw [loop]
      simp [h_body]
    · rw [hhash_val]
      have htake : data.val.take data.len.val = data.val := List.take_length data.val
      have heq2 : data.val.take i.val = data.val.take data.len.val := by rw [h_eq]
      rw [heq2, htake]
  | succ f ih =>
    have h_ilt : i.val < data.len.val := by omega
    have h_body := hash_bytes_loop_body_cont data hash i h_ilt hhash_bound
    rcases h_body with ⟨hash1, i3, hbody_eval, hbody_val, hbody_bound, hbody_i3⟩
    have hfuel1 : data.len.val - i3.val = f := by omega
    have hi3_le : i3.val ≤ data.len.val := by omega
    have hhash1_val : u64Z hash1 = hashBytesU8 (data.val.take i3.val) := by
      rw [hbody_val, hhash_val, hbody_i3]
      have h_take := hashBytesU8_take_succ data.val i.val h_ilt
      rw [h_take]
    have h_ih := ih hash1 i3 hfuel1 hi3_le hbody_bound hhash1_val
    rcases h_ih with ⟨r, h_eval, h_val, h_bound_r⟩
    refine ⟨r, ?_, h_val, h_bound_r⟩
    dsimp [RollingHashKernel.hash_bytes_loop]
    rw [loop]
    simp [hbody_eval]
    exact h_eval

theorem hash_bytes_spec (data : Slice U8) :
    ∃ r, RollingHashKernel.hash_bytes data = ok r ∧
         u64Z r = hashBytesU8 data.val ∧ r.val < M := by
  unfold RollingHashKernel.hash_bytes
  have hloop : ∃ r, RollingHashKernel.hash_bytes_loop data 0#u64 0#usize = ok r ∧ u64Z r = hashBytesU8 data.val ∧ r.val < M := by
    apply hash_bytes_loop_spec_aux data.len.val data 0#u64 0#usize
    · change data.len.val - 0 = data.len.val; exact Nat.sub_zero _
    · change 0 ≤ data.len.val; exact Nat.zero_le _
    · unfold M; decide
    · rw [u64Z_zero]
      have htake : data.val.take 0 = [] := List.take_zero
      rw [htake, hashBytes_nil]
      rfl
  rcases hloop with ⟨r, hloop_eval, hloop_val, hloop_bound⟩
  rw [hloop_eval]
  refine ⟨r, rfl, hloop_val, hloop_bound⟩

theorem hash_bytes_matches_spec (data : Slice U8) :
    ∃ r, RollingHashKernel.hash_bytes data = ok r ∧
         u64Z r = hashBytes (data.val.map asUInt8) := by
  have hspec := hash_bytes_spec data
  rcases hspec with ⟨r, heval, hval, _⟩
  refine ⟨r, heval, ?_⟩
  rw [hval]
  rw [hashBytes_map_asUInt8]

"""

content = content.replace("end Playground.Correctness.RollingHash", new_content + "\nend Playground.Correctness.RollingHash")

with open('Playground/correctness/rolling_hash.lean', 'w') as f:
    f.write(content)

