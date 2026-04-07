import Playground.defs.rolling_hash
import Playground.extracted.rolling_hash
import Playground.correctness.rolling_hash
import Mathlib.Data.ZMod.Basic
import Mathlib.NumberTheory.LucasLehmer
import Aeneas
import Aeneas.Std.Scalar.Casts

open Aeneas Aeneas.Std Result ControlFlow Error
namespace Playground.Correctness.RollingHash

-- The task is to prove power_shift, extend_right, append_right.

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
         u64Z r = extendRight (u64Z hash) (u64Z right_hash) right_length.val := by
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

theorem append_right_spec (hash : U64) (b : U8)
    (h1 : hash.val < M) :
    ∃ r, RollingHashKernel.append_right hash b = ok r ∧
         u64Z r = appendRight (u64Z hash) (asUInt8 b) := by
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
         u64Z r = extendRight (u64Z hash) (u64Z (UScalar.cast UScalarTy.U64 b)) 1 := by
    apply extend_right_spec hash (UScalar.cast UScalarTy.U64 b) 1#usize h1 hcast_b_M
  rcases hextend with ⟨r, hextend_eval, hextend_val⟩
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
  exact h_zmod_eq

end Playground.Correctness.RollingHash