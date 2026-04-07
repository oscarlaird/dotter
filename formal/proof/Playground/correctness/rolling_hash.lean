/-
  Refinement: Aeneas `RollingHashKernel` vs `Playground.defs.rolling_hash`.

  `u64Z` maps a Rust `u64` bitvector to `Hash = ZMod M` via natural reduction mod `M`.

  **Proved here:** extracted `MOD` / `BASE` match `M` / `B`; `hash_bytes` on an empty slice
  returns `ok 0`, matching `hashBytes []`; `MOD` as `ok (eval_global MOD)`; closed `fast_mod`
  and `power_shift` steps giving `extend_right 0 0 0 = ok 0` and
  `append_right (0#u64) (0#u8) = ok 0`.

  **Still open for full list refinement:** relate `hash_bytes` on non-empty `Slice`/`List` to
  `hashBytes` (same `Result`/`BitVec` chain as general `fast_mod` / loop reasoning).
-/
import Playground.defs.rolling_hash
import Playground.extracted.rolling_hash
import Mathlib.Data.ZMod.Basic
import Mathlib.NumberTheory.LucasLehmer
import Aeneas
import Aeneas.Std.Scalar.Casts
open Aeneas Aeneas.Std Result ControlFlow Error

namespace Playground.Correctness.RollingHash

noncomputable def u64Z (u : U64) : Hash :=
  (UScalar.val u : ℕ)

private lemma loop_done_first {α β} (body : α → Result (ControlFlow α β)) (x : α) (y : β)
    (h : body x = ok (ControlFlow.done y)) : loop body x = ok y := by
  simp [loop, h]

theorem MOD_eq_ok : RollingHashKernel.MOD = ok (eval_global RollingHashKernel.MOD) := by
  unfold RollingHashKernel.MOD eval_global
  simp only [ok?]
  rfl

private theorem u128_and_mod61 (x : U128) :
    (x &&& (UScalar.ofNatCore M (by native_decide) : U128)).val = x.val % (2 ^ 61) := by
  change x.val &&& (2 ^ 61 - 1) = x.val % (2 ^ 61)
  simpa using (Nat.and_two_pow_sub_one_eq_mod x.val 61)

private lemma u64_shl_1_61 :
    (1#u64 <<< 61#i32 : Result U64) = ok (UScalar.ofNatCore (2 ^ 61) (by native_decide)) := by
  simp only [HShiftLeft.hShiftLeft, UScalar.shiftLeft_IScalar, UScalar.shiftLeft, reduceIte]
  refine congrArg ok ?_
  refine UScalar.eq_of_val_eq ?_
  native_decide

private lemma u64_MOD_do : (do
    let i ← 1#u64 <<< 61#i32
    i - 1#u64 : Result U64) = ok (UScalar.ofNatCore M (by native_decide)) := by
  simp [Bind.bind, u64_shl_1_61, bind_ok]
  simp only [HSub.hSub, UScalar.sub, reduceIte]
  refine congrArg ok ?_
  refine UScalar.eq_of_val_eq ?_
  native_decide

theorem fast_mod_spec_small (x : U128)
    (hsmall : (x.val &&& M) + (x.val >>> 61) < 2 ^ 64)
    (h2m : (x.val &&& M) + (x.val >>> 61) < 2 * M) :
    ∃ r, RollingHashKernel.fast_mod x = ok r ∧ r.val = x.val % M := by
  unfold RollingHashKernel.fast_mod RollingHashKernel.MOD
  simp_rw [u64_MOD_do]
  let n : Nat := (x.val &&& M) + (x.val >>> 61)
  have hn64 : n < 2 ^ 64 := by simpa [n] using hsmall
  have hn128 : n < 2 ^ UScalarTy.U128.numBits := by
    exact lt_trans hn64 (by native_decide)
  have hcast :
      (UScalar.cast UScalarTy.U64
        (UScalar.ofNatCore (ty := UScalarTy.U128) n hn128)).val = n := by
    apply UScalar.cast_val_mod_pow_of_inBounds_eq
    simpa using hn64
  have hcastEq :
      UScalar.cast UScalarTy.U64 (UScalar.ofNatCore (ty := UScalarTy.U128) n hn128) =
      UScalar.ofNatCore (ty := UScalarTy.U64) n (by simpa using hn64) := by
    refine UScalar.eq_of_val_eq ?_
    simpa using hcast
  have hmodEq : x.val % M = n % M := by
    have hm : x.val % M = (x.val / 2 ^ 61 + x.val % 2 ^ 61) % M := modEq_mersenne 61 x.val
    have hand : x.val &&& M = x.val % 2 ^ 61 := by
      simpa [M] using (Nat.and_two_pow_sub_one_eq_mod x.val 61)
    have hshift : x.val >>> 61 = x.val / 2 ^ 61 := by
      simpa [Nat.shiftRight_eq_div_pow]
    change x.val % M = ((x.val &&& M) + (x.val >>> 61)) % M
    rw [hand, hshift, Nat.add_comm]
    exact hm
  
  have hsh_bound : (↑x : Nat) >>> 61 < 2 ^ UScalarTy.U128.numBits := by
    have h1 : x.val >>> 61 ≤ n := by omega
    exact lt_of_le_of_lt h1 hn128
  
  have hi3 : (x >>> 61#i32 : Result U128) = ok (UScalar.ofNatCore (↑x >>> 61) hsh_bound) := by
    simp only [HShiftRight.hShiftRight, UScalar.shiftRight_IScalar, UScalar.shiftRight]
    refine congrArg ok ?_
    refine UScalar.eq_of_val_eq ?_
    rfl
  
  have hi2i3 : (x &&& UScalar.cast UScalarTy.U128 (UScalar.ofNatCore M u64_MOD_do._proof_1)) + UScalar.ofNatCore (↑x >>> 61) hsh_bound = ok (UScalar.ofNatCore (ty := UScalarTy.U128) n hn128) := by
    simp only [HAdd.hAdd, UScalar.add, UScalar.tryMk, UScalar.tryMkOpt, Result.ofOption]
    have h128bound : (x &&& UScalar.cast UScalarTy.U128 (UScalar.ofNatCore M u64_MOD_do._proof_1)).val + (UScalar.ofNatCore (↑x >>> 61) hsh_bound : U128).val < 2 ^ UScalarTy.U128.numBits := by
      have handval : (x &&& UScalar.cast UScalarTy.U128 (UScalar.ofNatCore M u64_MOD_do._proof_1)).val = x.val &&& M := by rfl
      have hshval : (UScalar.ofNatCore (↑x >>> 61) hsh_bound : U128).val = x.val >>> 61 := by rfl
      rw [handval, hshval]
      exact hn128
    simp only [UScalar.check_bounds]
    have h_if : decide (Add.add (x &&& UScalar.cast UScalarTy.U128 (UScalar.ofNatCore M u64_MOD_do._proof_1)).val (UScalar.ofNatCore (↑x >>> 61) hsh_bound : U128).val < 2 ^ UScalarTy.U128.numBits) = true := by
      exact decide_eq_true h128bound
    rw [dif_pos h_if]
    refine congrArg ok ?_
    refine UScalar.eq_of_val_eq ?_
    exact rfl

  simp [Bind.bind, bind_ok, lift, hi3, hi2i3]

  have hval1 : (UScalar.ofNatCore (ty := UScalarTy.U64) n (by simpa using hn64)).val = n := 
    UScalar.ofNatCore_val_eq (by simpa using hn64)

  by_cases hge : M ≤ n
  · have hn64sub : n - M < 2 ^ 64 := by omega
    refine ⟨UScalar.ofNatCore (n - M) (by simpa using hn64sub), ?_⟩
    have hsubmod : n % M = n - M := by
      have hlt : n - M < M := by omega
      have hdecomp : n = M + (n - M) := by omega
      rw [hdecomp, Nat.add_mod_left, Nat.mod_eq_of_lt hlt]
      omega
    refine ⟨?_, ?_⟩
    · rw [hcastEq]
      rw [hval1]
      rw [if_pos hge]
      simp only [HSub.hSub, UScalar.sub]
      have hval2 : (UScalar.ofNatCore (ty := UScalarTy.U64) M u64_MOD_do._proof_1).val = M := rfl
      have h_not_lt : ¬ n < M := by omega
      have h_not_lt2 : ¬ (UScalar.ofNatCore (ty := UScalarTy.U64) n (by simpa using hn64)).val < (UScalar.ofNatCore M u64_MOD_do._proof_1).val := by
        rw [hval1, hval2]; exact h_not_lt
      rw [if_neg h_not_lt2]
      refine congrArg ok ?_
      apply UScalar.eq_of_val_eq
      have h1 : (BitVec.ofNat 64 (n - M)).toNat = (n - M) % 2^64 := BitVec.toNat_ofNat (n - M) 64
      have h2 : (n - M) % 2^64 = n - M := Nat.mod_eq_of_lt (by omega)
      rw [hval1, hval2]
      exact Eq.trans h1 h2
    · rw [hmodEq, hsubmod]
      apply UScalar.ofNatCore_val_eq
  · refine ⟨UScalar.ofNatCore n (by simpa using hn64), ?_⟩
    refine ⟨?_, ?_⟩
    · rw [hcastEq]
      rw [hval1]
      rw [if_neg (not_le_of_gt (lt_of_not_ge hge))]
    · rw [hmodEq, Nat.mod_eq_of_lt (lt_of_not_ge hge)]
      apply UScalar.ofNatCore_val_eq

theorem fast_mod_zero : RollingHashKernel.fast_mod (0#u128) = ok (0#u64) := by
  unfold RollingHashKernel.fast_mod RollingHashKernel.MOD
  simp_rw [u64_MOD_do]
  simp [bind_ok, lift, bind_ok, lift, bind_ok]
  simp only [HShiftRight.hShiftRight, UScalar.shiftRight_IScalar, UScalar.shiftRight, reduceIte]
  simp [bind_ok, HAdd.hAdd, UScalar.add, bind_ok, lift, bind_ok, reduceIte]
  refine congrArg ok ?_
  refine UScalar.eq_of_val_eq ?_
  native_decide

theorem fast_mod_B_u128 :
    RollingHashKernel.fast_mod (UScalar.ofNatCore B (by native_decide)) = ok RollingHashKernel.BASE := by
  unfold RollingHashKernel.fast_mod RollingHashKernel.MOD B RollingHashKernel.BASE
  simp_rw [u64_MOD_do]
  simp [bind_ok, lift, bind_ok, lift, bind_ok]
  simp only [HShiftRight.hShiftRight, UScalar.shiftRight_IScalar, UScalar.shiftRight, reduceIte]
  simp [bind_ok, HAdd.hAdd, UScalar.add, bind_ok, lift, bind_ok, reduceIte]
  refine congrArg ok ?_
  refine UScalar.eq_of_val_eq ?_
  native_decide

private theorem fast_mod_mul_1_257 :
    RollingHashKernel.fast_mod (UScalar.ofNatCore (Mul.mul 1 257) (by native_decide)) =
      ok RollingHashKernel.BASE :=
  Eq.trans
    (congrArg RollingHashKernel.fast_mod (UScalar.eq_of_val_eq (by simp [B])))
    fast_mod_B_u128

private lemma power_shift_loop_body_zero :
    RollingHashKernel.power_shift_loop.body (0#usize) (1#u128) (0#usize) =
      ok (ControlFlow.done (1#u128)) := by
  simp [RollingHashKernel.power_shift_loop.body]

private lemma power_shift_zero_loop :
    RollingHashKernel.power_shift_loop (0#usize) (1#u128) (0#usize) = ok (1#u128) := by
  dsimp [RollingHashKernel.power_shift_loop]
  exact loop_done_first _ (1#u128, 0#usize) (1#u128) power_shift_loop_body_zero

private lemma power_shift_exp_zero :
    RollingHashKernel.power_shift (0#usize) = ok (1#u64) := by
  simp [RollingHashKernel.power_shift, power_shift_zero_loop, bind_ok]
  refine UScalar.eq_of_val_eq ?_
  simp [UScalar.cast_val_eq]

private lemma power_shift_loop_body_one_first :
    RollingHashKernel.power_shift_loop.body 1#usize 1#u128 0#usize =
      ok (cont (UScalar.cast .U128 RollingHashKernel.BASE, 1#usize)) := by
  simp [RollingHashKernel.power_shift_loop.body, RollingHashKernel.BASE, bind_ok, lift, bind_ok, HMul.hMul,
    UScalar.mul, bind_ok, UScalar.tryMk, UScalar.tryMkOpt, Result.ofOption]
  simp +arith [reduceIte, bind_ok]
  rw [fast_mod_mul_1_257]
  simp [bind_ok, HAdd.hAdd, UScalar.add, bind_ok]
  unfold RollingHashKernel.BASE
  simp [bind_ok, UScalar.tryMk, UScalar.tryMkOpt, Result.ofOption, bind_ok, reduceIte]
  simp_all only [UScalar.check_bounds, UScalarTy.Usize_numBits_eq, decide_true,
    Aeneas.SimpIfs.dite_true, bind_ok, UScalar.ofNatCore, Result.ofOption, ok.injEq, cont.injEq,
    Prod.mk.injEq]
  by_cases ht : Add.add 0 1 < 2 ^ System.Platform.numBits
  swap
  · exfalso
    exact ht (by native_decide)
  · simp [ht, bind_ok, Result.ofOption, bind_ok, reduceIte]

private lemma power_shift_loop_body_one_done :
    RollingHashKernel.power_shift_loop.body 1#usize (UScalar.cast .U128 RollingHashKernel.BASE) 1#usize =
      ok (done (UScalar.cast .U128 RollingHashKernel.BASE)) := by
  simp [RollingHashKernel.power_shift_loop.body]

private lemma power_shift_one_loop :
    RollingHashKernel.power_shift_loop 1#usize 1#u128 0#usize =
      ok (UScalar.cast .U128 RollingHashKernel.BASE) := by
  dsimp [RollingHashKernel.power_shift_loop, loop]
  simp [power_shift_loop_body_one_first, bind_ok, loop, power_shift_loop_body_one_done, bind_ok]

private lemma power_shift_exp_one :
    RollingHashKernel.power_shift 1#usize = ok RollingHashKernel.BASE := by
  simp [RollingHashKernel.power_shift, power_shift_one_loop, bind_ok]
  refine UScalar.eq_of_val_eq ?_
  simp [UScalar.cast_val_eq]

theorem extend_right_zero_zero_zero :
    RollingHashKernel.extend_right (0#u64) (0#u64) (0#usize) = ok (0#u64) := by
  simp [RollingHashKernel.extend_right, power_shift_exp_zero, bind_ok, lift, bind_ok, HMul.hMul, UScalar.mul,
    bind_ok, HAdd.hAdd, UScalar.add, bind_ok, UScalar.tryMk, UScalar.tryMkOpt, Result.ofOption]
  simp +arith [reduceIte, bind_ok]
  refine Eq.trans (congrArg RollingHashKernel.fast_mod ?_) fast_mod_zero
  refine UScalar.eq_of_val_eq ?_
  simp +arith

theorem append_right_zero_zero_byte :
    RollingHashKernel.append_right (0#u64) (0#u8) = ok (0#u64) := by
  simp [RollingHashKernel.append_right, lift, RollingHashKernel.extend_right, power_shift_exp_one, bind_ok, lift,
    bind_ok, HMul.hMul, UScalar.mul, bind_ok, HAdd.hAdd, UScalar.add, bind_ok, UScalar.tryMk, UScalar.tryMkOpt,
    Result.ofOption]
  simp +arith [reduceIte, bind_ok]
  refine Eq.trans (congrArg RollingHashKernel.fast_mod ?_) fast_mod_zero
  refine UScalar.eq_of_val_eq ?_
  simp +arith

noncomputable def asUInt8 (b : U8) : UInt8 := UInt8.ofNat b.val

noncomputable def hashBytesU8 (bs : List U8) : Hash :=
  bs.foldl (fun h b => h * B + byteHash (asUInt8 b)) 0

private theorem hashBytes_map_asUInt8 (bs : List U8) :
    hashBytes (bs.map asUInt8) = hashBytesU8 bs := by
  simp [hashBytes, hashBytesU8, List.foldl_map]

instance : NeZero M := ⟨by decide⟩

private lemma mod_eq_iff_zmod_eq {a b : Nat} : a % M = b % M ↔ (a : ZMod M) = (b : ZMod M) := by
  have ha : (a : ZMod M).val = a % M := ZMod.val_natCast M a
  have hb : (b : ZMod M).val = b % M := ZMod.val_natCast M b
  constructor
  · intro h; exact ZMod.val_injective M (by rwa [ha, hb])
  · intro h; rw [← ha, ← hb, h]

private lemma base_cast_val : UScalar.cast UScalarTy.U128 RollingHashKernel.BASE = UScalar.ofNatCore (ty := UScalarTy.U128) B (by native_decide) := by
  refine UScalar.eq_of_val_eq ?_
  simp [RollingHashKernel.BASE, B, UScalar.cast_val_eq]

private lemma cur_mul_base (cur : U128) (hcur : cur.val < M) :
    (cur * UScalar.ofNatCore (ty := UScalarTy.U128) B (by native_decide) : Result U128) = ok (UScalar.ofNatCore (ty := UScalarTy.U128) (cur.val * B) (by
      unfold M at hcur
      unfold B
      have h128 : UScalarTy.U128.numBits = 128 := rfl
      rw [h128]
      omega
    )) := by
  simp only [HMul.hMul, UScalar.mul, UScalar.tryMk, UScalar.tryMkOpt, Result.ofOption, UScalar.check_bounds]
  split_ifs with h_if
  · refine congrArg ok ?_
    refine UScalar.eq_of_val_eq ?_
    rfl
  · exfalso
    have h_bound : cur.val * (UScalar.ofNatCore (ty := UScalarTy.U128) B (by native_decide) : U128).val < 2 ^ UScalarTy.U128.numBits := by
      have hB : (UScalar.ofNatCore (ty := UScalarTy.U128) B (by native_decide) : U128).val = B := rfl
      rw [hB]
      unfold M at hcur
      unfold B
      have h128 : UScalarTy.U128.numBits = 128 := rfl
      rw [h128]
      omega
    have h_true : decide (Mul.mul cur.val (UScalar.ofNatCore (ty := UScalarTy.U128) B (by native_decide) : U128).val < 2 ^ UScalarTy.U128.numBits) = true := decide_eq_true h_bound
    exact h_if h_true

private lemma fast_mod_precond (cur : Nat) (hcur : cur < M) :
    ((cur * B) &&& M) + ((cur * B) >>> 61) < 2 * M ∧
    ((cur * B) &&& M) + ((cur * B) >>> 61) < 2 ^ 64 := by
  have hM : M = 2^61 - 1 := rfl
  have hand_eq : (cur * B) &&& M = (cur * B) % 2^61 := by
    change (cur * B) &&& (2^61 - 1) = (cur * B) % 2^61
    exact Nat.and_two_pow_sub_one_eq_mod (cur * B) 61
  have hshift_eq : (cur * B) >>> 61 = (cur * B) / 2^61 := by
    exact Nat.shiftRight_eq_div_pow (cur * B) 61
  rw [hand_eq, hshift_eq]
  have hand : (cur * B) % 2^61 ≤ M := by
    rw [hM]
    have : (cur * B) % 2^61 < 2^61 := Nat.mod_lt _ (by decide)
    omega
  have hshift : (cur * B) / 2^61 < 257 := by
    have : cur * B < 2^61 * 257 := by
      have h1 : cur < 2^61 := by unfold M at hcur; omega
      have hB : B = 257 := rfl
      rw [hB]
      exact Nat.mul_lt_mul_of_pos_right h1 (by decide)
    exact Nat.div_lt_of_lt_mul this
  constructor
  · unfold M at hand
    unfold M
    omega
  · unfold M at hand
    omega

private lemma power_shift_loop_body_cont (exp : Usize) (cur : U128) (i : Usize)
    (h_ilt : i.val < exp.val) (hcur : cur.val < M) :
    ∃ cur1 i4, RollingHashKernel.power_shift_loop.body exp cur i = ok (cont (cur1, i4)) ∧
        cur1.val % M = (cur.val * B) % M ∧ cur1.val < M ∧ i4.val = i.val + 1 := by
  unfold RollingHashKernel.power_shift_loop.body
  rw [if_pos (by simpa using h_ilt)]
  rw [base_cast_val]
  simp only [Bind.bind, Aeneas.Std.bind_ok, lift, cur_mul_base cur hcur]
  have h_bound : cur.val * B < 2 ^ UScalarTy.U128.numBits := by
    unfold M at hcur
    unfold B
    have h128 : UScalarTy.U128.numBits = 128 := rfl
    rw [h128]
    omega
  have hfast_mod : ∃ r, RollingHashKernel.fast_mod (UScalar.ofNatCore (ty := UScalarTy.U128) (cur.val * B) h_bound) = ok r ∧ r.val = (cur.val * B) % M := by
    apply fast_mod_spec_small
    · have h_pre := fast_mod_precond cur.val hcur
      have hval : (UScalar.ofNatCore (ty := UScalarTy.U128) (cur.val * B) h_bound : U128).val = cur.val * B := UScalar.ofNatCore_val_eq h_bound
      rw [hval]
      exact h_pre.2
    · have h_pre := fast_mod_precond cur.val hcur
      have hval : (UScalar.ofNatCore (ty := UScalarTy.U128) (cur.val * B) h_bound : U128).val = cur.val * B := UScalar.ofNatCore_val_eq h_bound
      rw [hval]
      exact h_pre.1
  rcases hfast_mod with ⟨r, hfast_mod_eval, hfast_mod_val⟩
  rw [hfast_mod_eval]
  simp only [Aeneas.Std.bind_ok]
  have h_bound_r : r.val < 2 ^ UScalarTy.U128.numBits := by
    have h128 : UScalarTy.U128.numBits = 128 := rfl
    rw [h128]
    have : r.val < 2 ^ 64 := r.hmax
    omega
  have hcast_r_val : UScalar.cast UScalarTy.U128 r = UScalar.ofNatCore (ty := UScalarTy.U128) r.val h_bound_r := by
    refine UScalar.eq_of_val_eq ?_
    simp [UScalar.cast_val_eq]
  rw [hcast_r_val]
  -- now i4 is i + 1
  have hi4 : (i + 1#usize : Result Usize) = ok (UScalar.ofNatCore (ty := UScalarTy.Usize) (i.val + 1) (by
      have : i.val < exp.val := h_ilt
      have hmax : exp.val < 2 ^ UScalarTy.Usize.numBits := exp.hmax
      omega
    )) := by
    simp only [HAdd.hAdd, UScalar.add, UScalar.tryMk, UScalar.tryMkOpt, Result.ofOption, UScalar.check_bounds]
    split_ifs with h_if
    · refine congrArg ok ?_
      refine UScalar.eq_of_val_eq ?_
      rfl
    · exfalso
      have h_bound : i.val + 1 < 2 ^ System.Platform.numBits := by
        have : i.val < exp.val := h_ilt
        have hmax : exp.val < 2 ^ UScalarTy.Usize.numBits := exp.hmax
        have heq : UScalarTy.Usize.numBits = System.Platform.numBits := UScalarTy.Usize_numBits_eq
        rw [heq] at hmax
        omega
      have h_true : decide (Add.add i.val 1 < 2 ^ System.Platform.numBits) = true := decide_eq_true h_bound
      exact h_if h_true
  rw [hi4]
  simp only [bind_ok]
  refine ⟨_, _, rfl, ?_, ?_, rfl⟩
  · have hr : (UScalar.ofNatCore (ty := UScalarTy.U128) r.val h_bound_r : U128).val = r.val := UScalar.ofNatCore_val_eq h_bound_r
    rw [hr, hfast_mod_val]
    exact Nat.mod_mod _ _
  · have hr : (UScalar.ofNatCore (ty := UScalarTy.U128) r.val h_bound_r : U128).val = r.val := UScalar.ofNatCore_val_eq h_bound_r
    rw [hr, hfast_mod_val]
    exact Nat.mod_lt _ (by unfold M; omega)

theorem power_shift_loop_spec_aux (fuel : Nat) (exp : Usize) (cur : U128) (i : Usize)
    (hfuel : exp.val - i.val = fuel) (hi : i.val ≤ exp.val)
    (hcur : cur.val < M) (hmod : cur.val % M = (B ^ i.val) % M) :
    ∃ r, RollingHashKernel.power_shift_loop exp cur i = ok r ∧ r.val % M = (B ^ exp.val) % M ∧ r.val < M := by
  induction fuel generalizing cur i with
  | zero =>
    have h_eq : i.val = exp.val := by omega
    have h_ile : ¬(i.val < exp.val) := by omega
    have h_body : RollingHashKernel.power_shift_loop.body exp cur i = ok (done cur) := by
      unfold RollingHashKernel.power_shift_loop.body
      rw [if_neg (by simpa using h_ile)]
    refine ⟨cur, ?_, ?_⟩
    · dsimp [RollingHashKernel.power_shift_loop]
      rw [loop]
      simp [h_body]
    · refine ⟨?_, hcur⟩
      rw [hmod, h_eq]
  | succ f ih =>
    have h_ilt : i.val < exp.val := by omega
    have h_body := power_shift_loop_body_cont exp cur i h_ilt hcur
    rcases h_body with ⟨cur1, i4, hbody_eval, hbody_mod, hbody_bound, hbody_i4⟩
    have hfuel1 : exp.val - i4.val = f := by omega
    have hi4_le : i4.val ≤ exp.val := by omega
    have hmod1 : cur1.val % M = (B ^ i4.val) % M := by
      rw [hbody_mod, hbody_i4]
      rw [mod_eq_iff_zmod_eq]
      have hmod_zmod : (cur.val : ZMod M) = (B : ZMod M) ^ i.val := by
        have := mod_eq_iff_zmod_eq.mp hmod
        push_cast at this
        exact this
      push_cast
      rw [hmod_zmod]
      ring
    have h_ih := ih cur1 i4 hfuel1 hi4_le hbody_bound hmod1
    rcases h_ih with ⟨r, h_eval, h_mod, h_bound_r⟩
    refine ⟨r, ?_, h_mod, h_bound_r⟩
    dsimp [RollingHashKernel.power_shift_loop]
    rw [loop]
    simp [hbody_eval]
    exact h_eval

theorem power_shift_spec (exp : Usize) :
    ∃ r, RollingHashKernel.power_shift exp = ok r ∧ u64Z r = B ^ exp.val ∧ r.val < M := by
  unfold RollingHashKernel.power_shift
  have hloop : ∃ cur, RollingHashKernel.power_shift_loop exp 1#u128 0#usize = ok cur ∧ cur.val % M = B ^ exp.val % M ∧ cur.val < M := by
    apply power_shift_loop_spec_aux exp.val exp 1#u128 0#usize
    · change exp.val - 0 = exp.val; exact Nat.sub_zero _
    · change 0 ≤ exp.val; exact Nat.zero_le _
    · unfold M; decide
    · rfl
  rcases hloop with ⟨cur, hloop_eval, hloop_mod, hloop_bound⟩
  rw [hloop_eval]
  simp only [Bind.bind, Aeneas.Std.bind_ok]
  have h64 : cur.val < 2 ^ 64 := by unfold M at hloop_bound; omega
  have hcast : UScalar.cast UScalarTy.U64 cur = UScalar.ofNatCore (ty := UScalarTy.U64) cur.val (by
      have h128 : UScalarTy.U64.numBits = 64 := rfl
      rw [h128]
      exact h64
    ) := by
    refine UScalar.eq_of_val_eq ?_
    change (UScalar.cast UScalarTy.U64 cur).val = cur.val
    apply UScalar.cast_val_mod_pow_of_inBounds_eq
    exact h64
  rw [hcast]
  refine ⟨_, rfl, ?_, ?_⟩
  · have hr_bound : cur.val < 2 ^ UScalarTy.U64.numBits := by
      have h128 : UScalarTy.U64.numBits = 64 := rfl
      rw [h128]
      exact h64
    have hr : (UScalar.ofNatCore (ty := UScalarTy.U64) cur.val hr_bound : U64).val = cur.val := UScalar.ofNatCore_val_eq hr_bound
    have h_goal : u64Z (UScalar.ofNatCore (ty := UScalarTy.U64) cur.val hr_bound) = (cur.val : ZMod M) := by
      unfold u64Z
      rw [hr]
    rw [h_goal]
    have hmod_zmod := mod_eq_iff_zmod_eq.mp hloop_mod
    push_cast at hmod_zmod
    exact hmod_zmod
  · have hr_bound : cur.val < 2 ^ UScalarTy.U64.numBits := by
      have h128 : UScalarTy.U64.numBits = 64 := rfl
      rw [h128]
      exact h64
    have hr : (UScalar.ofNatCore (ty := UScalarTy.U64) cur.val hr_bound : U64).val = cur.val := UScalar.ofNatCore_val_eq hr_bound
    rw [hr]
    exact hloop_bound

private lemma fast_mod_precond_extend (hash ps right_hash : Nat)
    (h1 : hash < M) (h2 : ps < M) (h3 : right_hash < M) :
    let val := hash * ps + right_hash;
    (val &&& M) + (val >>> 61) < 2 * M ∧
    (val &&& M) + (val >>> 61) < 2 ^ 64 := by
  intro val
  have hM : M = 2^61 - 1 := rfl
  have hand_eq : val &&& M = val % 2^61 := by
    change val &&& (2^61 - 1) = val % 2^61
    exact Nat.and_two_pow_sub_one_eq_mod val 61
  have hshift_eq : val >>> 61 = val / 2^61 := Nat.shiftRight_eq_div_pow val 61
  rw [hand_eq, hshift_eq]
  have hand : val % 2^61 ≤ M := by
    rw [hM]
    have : val % 2^61 < 2^61 := Nat.mod_lt _ (by decide)
    omega
  have hshift : val / 2^61 < M := by
    have hval_lt : val < M * 2^61 := by
      have : hash * ps + right_hash < M * M + M := by nlinarith
      have h1 : M * M + M = M * (M + 1) := by ring
      have h2 : M * (M + 1) = M * 2^61 := by
        rw [hM]
        have : 2^61 - 1 + 1 = 2^61 := by omega
        congr 1
      omega
    unfold M at *
    omega
  constructor
  · omega
  · unfold M at *; omega

theorem extend_right_spec (hash right_hash : U64) (right_length : Usize)
    (h1 : hash.val < M) (h2 : right_hash.val < M) :
    ∃ r, RollingHashKernel.extend_right hash right_hash right_length = ok r ∧
         u64Z r = extendRight (u64Z hash) (u64Z right_hash) right_length.val ∧ r.val < M := by
  unfold RollingHashKernel.extend_right
  have hps : ∃ ps, RollingHashKernel.power_shift right_length = ok ps ∧ u64Z ps = B ^ right_length.val ∧ ps.val < M := by
    apply power_shift_spec
  rcases hps with ⟨ps, hps_eval, hps_val, hps_bound⟩
  rw [hps_eval]
  simp only [Bind.bind, Aeneas.Std.bind_ok, lift]
  have h_bound_hi1 : hash.val * ps.val < 2 ^ UScalarTy.U128.numBits := by
    have hh : hash.val < 2^61 := by unfold M at h1; omega
    have hp : ps.val < 2^61 := by unfold M at hps_bound; omega
    have h128 : UScalarTy.U128.numBits = 128 := rfl
    rw [h128]
    nlinarith
  have hi1 : (UScalar.cast UScalarTy.U128 hash : U128) * (UScalar.cast UScalarTy.U128 ps : U128) =
    ok (UScalar.ofNatCore (ty := UScalarTy.U128) (hash.val * ps.val) h_bound_hi1) := by
    have hcast_h_bound : hash.val < 2 ^ UScalarTy.U128.numBits := by
      have hh : hash.val < 2^61 := by unfold M at h1; omega
      have h128 : UScalarTy.U128.numBits = 128 := rfl
      rw [h128]
      omega
    have hcast_ps_bound : ps.val < 2 ^ UScalarTy.U128.numBits := by
      have hp : ps.val < 2^61 := by unfold M at hps_bound; omega
      have h128 : UScalarTy.U128.numBits = 128 := rfl
      rw [h128]
      omega
    have hcast_h : (UScalar.cast UScalarTy.U128 hash : U128).val = hash.val := by
      have h1 : UScalar.cast UScalarTy.U128 hash = UScalar.ofNatCore (ty := UScalarTy.U128) hash.val hcast_h_bound := by
        refine UScalar.eq_of_val_eq ?_
        change (UScalar.cast UScalarTy.U128 hash).val = hash.val
        apply UScalar.cast_val_mod_pow_of_inBounds_eq
        exact hcast_h_bound
      rw [h1]
      exact UScalar.ofNatCore_val_eq hcast_h_bound
    have hcast_ps : (UScalar.cast UScalarTy.U128 ps : U128).val = ps.val := by
      have h1 : UScalar.cast UScalarTy.U128 ps = UScalar.ofNatCore (ty := UScalarTy.U128) ps.val hcast_ps_bound := by
        refine UScalar.eq_of_val_eq ?_
        change (UScalar.cast UScalarTy.U128 ps).val = ps.val
        apply UScalar.cast_val_mod_pow_of_inBounds_eq
        exact hcast_ps_bound
      rw [h1]
      exact UScalar.ofNatCore_val_eq hcast_ps_bound
    simp only [HMul.hMul, UScalar.mul, UScalar.tryMk, UScalar.tryMkOpt, Result.ofOption, UScalar.check_bounds]
    rw [hcast_h, hcast_ps]
    have h_bound : hash.val * ps.val < 2 ^ UScalarTy.U128.numBits := by
      have hh : hash.val < 2^61 := by unfold M at h1; omega
      have hp : ps.val < 2^61 := by unfold M at hps_bound; omega
      have h128 : UScalarTy.U128.numBits = 128 := rfl
      rw [h128]
      nlinarith
    have h_if : decide (hash.val * ps.val < 2 ^ UScalarTy.U128.numBits) = true := decide_eq_true h_bound
    split_ifs with h_if
    · refine congrArg ok ?_
      refine UScalar.eq_of_val_eq ?_
      rfl
    · exfalso
      exact h_if (decide_eq_true h_bound)
  rw [hi1]
  simp only [Aeneas.Std.bind_ok]
  have h_bound_hi2 : hash.val * ps.val + right_hash.val < 2 ^ UScalarTy.U128.numBits := by
    have hh : hash.val < 2^61 := by unfold M at h1; omega
    have hp : ps.val < 2^61 := by unfold M at hps_bound; omega
    have hr : right_hash.val < 2^61 := by unfold M at h2; omega
    have h128 : UScalarTy.U128.numBits = 128 := rfl
    rw [h128]
    nlinarith
  have hi2 : (UScalar.ofNatCore (ty := UScalarTy.U128) (hash.val * ps.val) h_bound_hi1) + (UScalar.cast UScalarTy.U128 right_hash : U128) =
    ok (UScalar.ofNatCore (ty := UScalarTy.U128) (hash.val * ps.val + right_hash.val) h_bound_hi2) := by
    have hcast_rh_bound : right_hash.val < 2 ^ UScalarTy.U128.numBits := by
      have hr : right_hash.val < 2^61 := by unfold M at h2; omega
      have h128 : UScalarTy.U128.numBits = 128 := rfl
      rw [h128]
      omega
    have hcast_rh : (UScalar.cast UScalarTy.U128 right_hash : U128).val = right_hash.val := by
      have h1 : UScalar.cast UScalarTy.U128 right_hash = UScalar.ofNatCore (ty := UScalarTy.U128) right_hash.val hcast_rh_bound := by
        refine UScalar.eq_of_val_eq ?_
        change (UScalar.cast UScalarTy.U128 right_hash).val = right_hash.val
        apply UScalar.cast_val_mod_pow_of_inBounds_eq
        exact hcast_rh_bound
      rw [h1]
      exact UScalar.ofNatCore_val_eq hcast_rh_bound
    simp only [HAdd.hAdd, UScalar.add, UScalar.tryMk, UScalar.tryMkOpt, Result.ofOption, UScalar.check_bounds]
    rw [hcast_rh]
    have hval : (UScalar.ofNatCore (ty := UScalarTy.U128) (hash.val * ps.val) h_bound_hi1 : U128).val = hash.val * ps.val := UScalar.ofNatCore_val_eq h_bound_hi1
    rw [hval]
    have h_bound : hash.val * ps.val + right_hash.val < 2 ^ UScalarTy.U128.numBits := by
      have hh : hash.val < 2^61 := by unfold M at h1; omega
      have hp : ps.val < 2^61 := by unfold M at hps_bound; omega
      have hr : right_hash.val < 2^61 := by unfold M at h2; omega
      have h128 : UScalarTy.U128.numBits = 128 := rfl
      rw [h128]
      nlinarith
    have h_if : decide (hash.val * ps.val + right_hash.val < 2 ^ UScalarTy.U128.numBits) = true := decide_eq_true h_bound
    split_ifs with h_if
    · refine congrArg ok ?_
      refine UScalar.eq_of_val_eq ?_
      rfl
    · exfalso
      exact h_if (decide_eq_true h_bound)
  rw [hi2]
  simp only [Aeneas.Std.bind_ok]
  have hfast_mod : ∃ r, RollingHashKernel.fast_mod (UScalar.ofNatCore (ty := UScalarTy.U128) (hash.val * ps.val + right_hash.val) h_bound_hi2) = ok r ∧ r.val = (hash.val * ps.val + right_hash.val) % M := by
    apply fast_mod_spec_small
    · have h_pre := fast_mod_precond_extend hash.val ps.val right_hash.val h1 hps_bound h2
      have hval : (UScalar.ofNatCore (ty := UScalarTy.U128) (hash.val * ps.val + right_hash.val) h_bound_hi2 : U128).val = hash.val * ps.val + right_hash.val := UScalar.ofNatCore_val_eq h_bound_hi2
      rw [hval]
      exact h_pre.2
    · have h_pre := fast_mod_precond_extend hash.val ps.val right_hash.val h1 hps_bound h2
      have hval : (UScalar.ofNatCore (ty := UScalarTy.U128) (hash.val * ps.val + right_hash.val) h_bound_hi2 : U128).val = hash.val * ps.val + right_hash.val := UScalar.ofNatCore_val_eq h_bound_hi2
      rw [hval]
      exact h_pre.1
  rcases hfast_mod with ⟨r, hfast_mod_eval, hfast_mod_val⟩
  rw [hfast_mod_eval]
  refine ⟨r, rfl, ?_⟩
  unfold extendRight
  have h_r_zmod : (r.val : ZMod M) = ((hash.val * ps.val + right_hash.val : Nat) : ZMod M) := by
    have h_mod_eq : r.val % M = (hash.val * ps.val + right_hash.val) % M := by
      rw [hfast_mod_val]
      exact Nat.mod_mod _ _
    exact mod_eq_iff_zmod_eq.mp h_mod_eq
  have h_goal_zmod : u64Z r = (r.val : ZMod M) := rfl
  rw [h_goal_zmod, h_r_zmod]
  have hmod_mul : ((hash.val * ps.val + right_hash.val : Nat) : ZMod M) = (hash.val : ZMod M) * (ps.val : ZMod M) + (right_hash.val : ZMod M) := by push_cast; rfl
  rw [hmod_mul]
  have hhash : (hash.val : ZMod M) = u64Z hash := rfl
  have hright : (right_hash.val : ZMod M) = u64Z right_hash := rfl
  have hps2 : (ps.val : ZMod M) = B ^ right_length.val := hps_val
  rw [hhash, hps2, hright]
  exact ⟨rfl, by rw [hfast_mod_val]; exact Nat.mod_lt _ (by unfold M; omega)⟩

theorem append_right_spec (hash : U64) (b : U8)
    (h1 : hash.val < M) :
    ∃ r, RollingHashKernel.append_right hash b = ok r ∧
         u64Z r = appendRight (u64Z hash) (asUInt8 b) ∧ r.val < M := by
  unfold RollingHashKernel.append_right
  simp only [Bind.bind, Aeneas.Std.bind_ok, lift]
  have hb_bound : b.val < M := by unfold M; have : b.val < 2^8 := b.hmax; omega
  have hcast_b_bound : b.val < 2^64 := by have : b.val < 2^8 := b.hmax; omega
  have hcast_b : (UScalar.cast UScalarTy.U64 b : U64).val = b.val := by
    apply UScalar.cast_val_mod_pow_of_inBounds_eq
    have h128 : UScalarTy.U64.numBits = 64 := rfl
    rw [h128]; exact hcast_b_bound
  have hcast_b_M : (UScalar.cast UScalarTy.U64 b : U64).val < M := by rw [hcast_b]; exact hb_bound
  have hextend : ∃ r, RollingHashKernel.extend_right hash (UScalar.cast UScalarTy.U64 b) 1#usize = ok r ∧
         u64Z r = extendRight (u64Z hash) (u64Z (UScalar.cast UScalarTy.U64 b)) 1 ∧ r.val < M := by
    apply extend_right_spec hash (UScalar.cast UScalarTy.U64 b) 1#usize h1 hcast_b_M
  rcases hextend with ⟨r, hextend_eval, hextend_val, hextend_bound⟩
  rw [hextend_eval]
  refine ⟨r, rfl, ?_⟩
  rw [hextend_val]
  unfold appendRight
  congr 2
  unfold u64Z byteHash asUInt8 UInt8.toNat
  have h_zmod_eq : ((UScalar.cast UScalarTy.U64 b : U64).val : ZMod M) = ((UInt8.ofNat b.val).toNat : ZMod M) := by
    have hcast_b : (UScalar.cast UScalarTy.U64 b : U64).val = b.val := by
      have h1 : UScalar.cast UScalarTy.U64 b = UScalar.ofNatCore (ty := UScalarTy.U64) b.val hcast_b_bound := by
        refine UScalar.eq_of_val_eq ?_
        change (UScalar.cast UScalarTy.U64 b).val = b.val
        apply UScalar.cast_val_mod_pow_of_inBounds_eq
        exact hcast_b_bound
      rw [h1]
      exact UScalar.ofNatCore_val_eq hcast_b_bound
    rw [hcast_b]
    have h8 : b.val < 256 := by have : b.val < 2^8 := b.hmax; exact this
    have htoNat : (UInt8.ofNat b.val).toNat = b.val := Nat.mod_eq_of_lt h8
    rw [htoNat]
  exact ⟨by rw [h_zmod_eq]; rfl, hextend_bound⟩


theorem MOD_eval_val : UScalar.val (eval_global RollingHashKernel.MOD) = M := by
  unfold RollingHashKernel.MOD eval_global
  simp only [ok?]
  rfl

theorem BASE_eval_val : UScalar.val RollingHashKernel.BASE = B := by
  simp [RollingHashKernel.BASE, B]

theorem BASE_z : u64Z RollingHashKernel.BASE = (B : Hash) := by
  simp [u64Z, BASE_eval_val, B]

theorem u64Z_zero : u64Z (0#u64) = 0 := by
  simp [u64Z]

theorem hashBytes_nil : hashBytes ([] : List UInt8) = 0 := by
  simp [hashBytes]

theorem hash_bytes_empty_eval :
    RollingHashKernel.hash_bytes (Slice.new U8) = ok (0#u64) := by
  dsimp [RollingHashKernel.hash_bytes, RollingHashKernel.hash_bytes_loop]
  exact loop_done_first _ (0#u64, 0#usize) (0#u64) rfl

theorem hash_bytes_empty_matches_spec :
    RollingHashKernel.hash_bytes (Slice.new U8) = ok (0#u64) ∧
      u64Z (0#u64) = hashBytes ([] : List UInt8) :=
  And.intro hash_bytes_empty_eval (by simp [u64Z_zero, hashBytes_nil])

theorem extendRight_zero_spec : extendRight (0 : Hash) (0 : Hash) 0 = 0 := by
  simp [extendRight, pow_zero, mul_one, add_zero]

theorem appendRight_zero_byte_spec :
    appendRight (0 : Hash) (0 : UInt8) = 0 := by
  simp [appendRight, extendRight, byteHash, UInt8.toNat, pow_one, add_zero]


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
    rw [Slice.getElem?_Usize_eq]
    have h_list_get : data.val[i.val]? = some (data.val[i.val]'h_ilt) := getElem?_pos ..
    rw [h_list_get]
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
      have htake : data.val.take data.len.val = data.val := List.take_length
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
      change (0 : Hash) = hashBytesU8 (data.val.take 0)
      have htake : data.val.take 0 = [] := List.take_zero
      rw [htake]
      unfold hashBytesU8
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


end Playground.Correctness.RollingHash
