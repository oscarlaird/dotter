import Mathlib.Data.List.Basic
import Mathlib.Data.List.Infix
import Mathlib.Data.Real.Basic
import Mathlib.Topology.Algebra.InfiniteSum.Basic
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Topology.Instances.NNReal.Lemmas
import Mathlib.Topology.Algebra.InfiniteSum.Ring

variable {α : Type} [DecidableEq α]

abbrev StringAlg (α : Type) := List α

def IsPrefix {α : Type} [DecidableEq α] (s t : StringAlg α) : Prop := List.IsPrefix s t

/-- Proper prefix -/
def IsStrictPrefix {α : Type} [DecidableEq α] (s t : StringAlg α) : Prop := List.IsPrefix s t ∧ s ≠ t

lemma prefix_or_prefix_of_prefix {α : Type} [DecidableEq α] {l1 l2 l3 : StringAlg α}
  (h1 : IsPrefix l1 l3) (h2 : IsPrefix l2 l3) : IsPrefix l1 l2 ∨ IsPrefix l2 l1 :=
  List.prefix_or_prefix_of_prefix h1 h2

def PrefixCode {α : Type} [DecidableEq α] (C : Set (StringAlg α)) : Prop :=
  ∀ c₁ c₂, c₁ ∈ C → c₂ ∈ C → IsStrictPrefix c₁ c₂ → False

lemma prefix_code_unique_prefix {α : Type} [DecidableEq α] {C : Set (StringAlg α)}
  (hC : PrefixCode C) {x : StringAlg α} {c1 c2 : StringAlg α}
  (h1 : c1 ∈ C) (hp1 : IsPrefix c1 x) (h2 : c2 ∈ C) (hp2 : IsPrefix c2 x) : c1 = c2 := by
  have h_or := prefix_or_prefix_of_prefix hp1 hp2
  rcases h_or with h12 | h21
  · by_cases heq : c1 = c2
    · exact heq
    · exfalso
      exact hC c1 c2 h1 h2 ⟨h12, heq⟩
  · by_cases heq : c2 = c1
    · exact heq.symm
    · exfalso
      exact hC c2 c1 h2 h1 ⟨h21, heq⟩

def CompletePrefixCode {α : Type} [DecidableEq α] (C : Set (StringAlg α)) : Prop :=
  PrefixCode C ∧ ∀ x : StringAlg α, ∃ c ∈ C, IsPrefix x c ∨ IsPrefix c x

def Refinement {α : Type} [DecidableEq α] (A B : Set (StringAlg α)) : Prop :=
  ∀ a ∈ A, ∃ b ∈ B, IsPrefix b a

/-- x ≤ C means that x is a prefix of some element of C. -/
def PrefixOfCode {α : Type} [DecidableEq α] (x : StringAlg α) (C : Set (StringAlg α)) : Prop :=
  ∃ c ∈ C, IsPrefix x c

noncomputable instance decidablePrefixOfCode {α : Type} [DecidableEq α] {x : StringAlg α} {C : Set (StringAlg α)} : Decidable (PrefixOfCode x C) := Classical.dec _

/-- x ≥ C means that x is an extension of some element of C. -/
def ExtensionOfCode {α : Type} [DecidableEq α] (x : StringAlg α) (C : Set (StringAlg α)) : Prop :=
  ∃ c ∈ C, IsPrefix c x

/-- x > C means that x is a proper extension of some element of C. -/
def ProperExtensionOfCode {α : Type} [DecidableEq α] (x : StringAlg α) (C : Set (StringAlg α)) : Prop :=
  ∃ c ∈ C, IsStrictPrefix c x

class Alphabet (α : Type) where
  space : α
  stop : α
  space_ne_stop : space ≠ stop

/-- A string h is properly terminated if and only if h[i] = 'stop' iff i = |h| - 1. -/
def ProperlyTerminated {α : Type} [Alphabet α] [DecidableEq α] (h : StringAlg α) : Prop :=
  ∃ (h_not_empty : h ≠ []),
  h.getLast h_not_empty = Alphabet.stop ∧
  ∀ i (_h_lt : i < h.length - 1), ∃ h_get_lt, h.get ⟨i, h_get_lt⟩ ≠ Alphabet.stop

def H {α : Type} [Alphabet α] [DecidableEq α] : Set (StringAlg α) := { h | ProperlyTerminated h }

lemma H_is_prefix_code {α : Type} [Alphabet α] [DecidableEq α] : PrefixCode (H (α := α)) := by
  intro c1 c2 hc1 hc2 h_strict_pref
  -- To prove it's a prefix code, assume for contradiction c1 < c2
  -- Then c1 must end in `stop`, but it's a proper prefix of c2, so c2 has `stop` before its end.
  -- But properly terminated strings only have `stop` at the very end.
  rcases hc1 with ⟨c1_not_emp, hc1_last, hc1_no_stop⟩
  rcases hc2 with ⟨c2_not_emp, hc2_last, hc2_no_stop⟩
  have h_pref := h_strict_pref.1
  have h_neq := h_strict_pref.2
  -- Since c1 is a proper prefix of c2, its length is strictly less than c2's length
  have h_len_lt : c1.length < c2.length := by
    cases h_pref with | intro k hk =>
      have hk_not_emp : k ≠ [] := by
        intro h_k_emp
        subst h_k_emp
        rw [List.append_nil] at hk
        exact h_neq hk
      have h_len : c2.length = c1.length + k.length := by
        rw [← hk, List.length_append]
      have h_k_len : 0 < k.length := by
        cases k
        · contradiction
        · exact Nat.zero_lt_succ _
      omega

  have h_c1_len_pos : 0 < c1.length := by
    cases c1
    · contradiction
    · exact Nat.zero_lt_succ _

  have h_c2_get_stop : c2[c1.length - 1] = Alphabet.stop := by
    cases h_pref with | intro k hk =>
      have hc1_last_eq : c1.getLast c1_not_emp = c1[c1.length - 1] := by
        exact List.getLast_eq_getElem c1_not_emp
      rw [← hc1_last, hc1_last_eq]
      have h_get_eq : c1[c1.length - 1] = c2[c1.length - 1] := by
        have h_eq : c1.get ⟨c1.length - 1, by omega⟩ = (c1 ++ k).get ⟨c1.length - 1, by rw [List.length_append]; omega⟩ := by
          exact (List.getElem_append_left (by omega)).symm
        subst hk
        exact h_eq
      exact h_get_eq.symm

  have hc2_no_early_stop := hc2_no_stop (c1.length - 1) (by omega)
  rcases hc2_no_early_stop with ⟨h_get_lt, h_neq_stop⟩
  have h_get_eq' : c2.get ⟨c1.length - 1, h_get_lt⟩ = c2[c1.length - 1] := rfl
  rw [← h_get_eq'] at h_c2_get_stop
  contradiction

lemma H_is_complete_prefix_code {α : Type} [Alphabet α] [DecidableEq α] : CompletePrefixCode (H (α := α)) := by
  constructor
  · exact H_is_prefix_code
  · intro x
    -- Either x has 'stop', then there is a prefix of x in H (or x is in H)
    -- Or x does not have 'stop', then x is a prefix of some string in H (e.g., x ++ ['stop'])
    by_cases h_stop_cond : ∃ i, ∃ (h_lt : i < x.length), x.get ⟨i, h_lt⟩ = Alphabet.stop
    · rcases h_stop_cond with ⟨i, hi, hi_stop⟩
      -- Since there is a stop, we can split x at the first stop
      -- wait, does it have to be the *first* stop?
      -- A properly terminated string has exactly one stop at the very end.
      -- So we take the prefix of x up to the first stop.
      have h_ex_first : ∃ (i_first : ℕ) (h_first_lt : i_first < x.length),
        x.get ⟨i_first, h_first_lt⟩ = Alphabet.stop ∧
        ∀ j < i_first, ∀ (hj : j < x.length), x.get ⟨j, hj⟩ ≠ Alphabet.stop := by
          let P : ℕ → Prop := fun i => ∃ (h_lt : i < x.length), x.get ⟨i, h_lt⟩ = Alphabet.stop
          have h_ex : ∃ i, P i := ⟨i, hi, hi_stop⟩
          have _inst : DecidablePred P := fun i => Classical.dec _
          let i_first := Nat.find h_ex
          have h_find := Nat.find_spec h_ex
          have h_min : ∀ (m : ℕ), m < i_first → ¬P m := fun m hm => Nat.find_min h_ex hm
          rcases h_find with ⟨h_first_lt, h_first_stop⟩
          use i_first, h_first_lt
          constructor
          · exact h_first_stop
          · intro j hj hj_lt h_eq
            have h_P_j : P j := ⟨hj_lt, h_eq⟩
            exact h_min j hj h_P_j
      rcases h_ex_first with ⟨i_first, h_first_lt, h_first_stop, h_first_min⟩
      let pref := x.take (i_first + 1)
      have h_pref_term : ProperlyTerminated pref := by
        -- pref is not empty because its length is i_first + 1 > 0
        have h_pref_len : pref.length = i_first + 1 := by
          exact List.length_take_of_le (by omega)
        have h_pref_not_emp : pref ≠ [] := by
          intro h_emp
          have h_len_zero : pref.length = 0 := by rw [h_emp, List.length_nil]
          rw [h_pref_len] at h_len_zero
          contradiction
        use h_pref_not_emp
        constructor
        · -- pref.getLast = pref[i_first] = x[i_first] = stop
          have h_last : pref.getLast h_pref_not_emp = Alphabet.stop := by
            have h_get_eq : pref.getLast h_pref_not_emp = pref[pref.length - 1] := by
              exact List.getLast_eq_getElem h_pref_not_emp
            rw [h_get_eq]
            have h_idx_eq : pref.length - 1 = i_first := by omega
            have h_pref_get : pref[pref.length - 1] = pref[i_first]'(by omega) := by
              congr
            rw [h_pref_get]
            have h_x_get : pref[i_first]'(by omega) = x[i_first] := by
              -- x.take n getElem
              -- List.getElem_take
              apply List.getElem_take
            rw [h_x_get]
            exact h_first_stop
          exact h_last
        · -- for j < i_first, pref[j] = x[j] != stop
          intro j hj
          use (by omega)
          have h_j_lt : j < i_first := by omega
          have h_pref_get : pref.get ⟨j, by omega⟩ = x.get ⟨j, by omega⟩ := by
            -- List.getElem_take
            apply List.getElem_take
          rw [h_pref_get]
          exact h_first_min j h_j_lt (by omega)
      use pref
      constructor
      · exact h_pref_term
      · right
        -- The goal is `IsPrefix x c ∨ IsPrefix c x`. We want `IsPrefix c x` (i.e. `IsPrefix pref x`).
        -- IsPrefix c x means ∃ k, c ++ k = x. Here c is pref, and k is x.drop (i_first + 1)
        use x.drop (i_first + 1)
        exact List.take_append_drop (i_first + 1) x
    · -- x has no 'stop'. So x ++ [Alphabet.stop] is properly terminated and extends x
      have h_no_stop : ∀ i, ∀ (h_lt : i < x.length), x.get ⟨i, h_lt⟩ ≠ Alphabet.stop := by
        intro i h_lt h_eq
        exact h_stop_cond ⟨i, h_lt, h_eq⟩
      have h_term : ProperlyTerminated (x ++ [Alphabet.stop]) := by
        -- ProperlyTerminated consists of an existence of non-empty proof
        use (by intro h_emp; rw [List.append_eq_nil_iff] at h_emp; rcases h_emp with ⟨_, h_bad⟩; contradiction)
        constructor
        · have h_not_emp : x ++ [Alphabet.stop] ≠ [] := by intro h; rw [List.append_eq_nil_iff] at h; rcases h with ⟨_, h_bad⟩; contradiction
          have h_last : (x ++ [Alphabet.stop]).getLast h_not_emp = Alphabet.stop := by
            -- prove that the last element of appending [Alphabet.stop] is Alphabet.stop
            have h_append_eq : (x ++ [Alphabet.stop]).getLast h_not_emp = Alphabet.stop := by
              have h_not_emp2 : ([Alphabet.stop] : StringAlg α) ≠ [] := by intro h; contradiction
              have h_eq : (x ++ [Alphabet.stop]).getLast h_not_emp = [Alphabet.stop].getLast h_not_emp2 := by
                apply List.getLast_append
              rw [h_eq]
              rfl
            exact h_append_eq
          exact h_last
        · intro i h_lt
          -- If i is in the first part, it's x.get which is not stop.
          -- since i < (x ++ [stop]).length - 1, and length is x.length + 1
          -- i < x.length
          have h_i_lt : i < x.length := by
            rw [List.length_append, List.length_singleton] at h_lt
            omega
          use (by rw [List.length_append, List.length_singleton]; omega)
          have h_get_append : (x ++ [Alphabet.stop]).get ⟨i, by rw [List.length_append, List.length_singleton]; omega⟩ = x.get ⟨i, h_i_lt⟩ := by
            exact List.getElem_append_left h_i_lt
          rw [h_get_append]
          exact h_no_stop i h_i_lt
      use x ++ [Alphabet.stop]
      constructor
      · exact h_term
      · left
        -- x is a prefix of x ++ [stop]
        use [Alphabet.stop]

-- Section 1: Likelihood, Prior, Branch Prior, Sequences

def LikelihoodFn {α : Type} [Alphabet α] [DecidableEq α] := H (α := α) → NNReal

structure PriorFn {α : Type} [Alphabet α] [DecidableEq α] where
  fn : H (α := α) → NNReal

noncomputable def BranchPrior {α : Type} [Alphabet α] [DecidableEq α] (P : PriorFn (α := α)) (x : StringAlg α) : NNReal :=
  -- the sum is over properly terminated strings. Since h is finite and alphabet might not be, we use tsum
  tsum (fun (h : H (α := α)) => if PrefixOfCode x {h.val} then P.fn h else (0:NNReal))

structure LikelihoodSeq {α : Type} [Alphabet α] [DecidableEq α] where
  seq : ℕ → LikelihoodFn (α := α)
  init_one : ∀ h, seq 0 h = 1

def PriorSeq {α : Type} [Alphabet α] [DecidableEq α] := ℕ → PriorFn (α := α)

-- Delta Digest and Likelihood Delta Digest Sequence

structure DeltaDigest {α : Type} [Alphabet α] [DecidableEq α] where
  C : Set (StringAlg α)
  is_prefix_code : PrefixCode C
  Δ : C → NNReal

noncomputable def Truncation {α : Type} [Alphabet α] [DecidableEq α] (x : StringAlg α) (D : DeltaDigest (α := α)) (h_ext : ExtensionOfCode x D.C) : D.C :=
  ⟨Classical.choose h_ext, (Classical.choose_spec h_ext).1⟩

lemma Truncation_is_prefix {α : Type} [Alphabet α] [DecidableEq α] (x : StringAlg α) (D : DeltaDigest (α := α)) (h_ext : ExtensionOfCode x D.C) :
  IsPrefix (Truncation x D h_ext).val x := (Classical.choose_spec h_ext).2

lemma Truncation_mem {α : Type} [Alphabet α] [DecidableEq α] (x : StringAlg α) (D : DeltaDigest (α := α)) (h_ext : ExtensionOfCode x D.C) :
  (Truncation x D h_ext).val ∈ D.C := (Truncation x D h_ext).property

noncomputable def safe_Δ {α : Type} [Alphabet α] [DecidableEq α] (x : StringAlg α) (D : DeltaDigest (α := α)) : NNReal :=
  let _inst : Decidable (ExtensionOfCode x D.C) := Classical.dec _
  if h : ExtensionOfCode x D.C then D.Δ (Truncation x D h) else 1

structure LikelihoodDeltaDigestSeq {α : Type} [Alphabet α] [DecidableEq α] (L : LikelihoodSeq (α := α)) where
  seq : ℕ → DeltaDigest (α := α)
  covers_H : ∀ n, ∀ h : H (α := α), ExtensionOfCode h.val (seq n).C
  satisfies_update : ∀ n > 0, ∀ h : H (α := α),
    L.seq n h = L.seq (n - 1) h * (seq n).Δ (Truncation h.val (seq n) (covers_H n h))

structure PriorDeltaDigestSeq {α : Type} [Alphabet α] [DecidableEq α] (P : PriorSeq (α := α)) where
  seq : ℕ → DeltaDigest (α := α)
  covers_H : ∀ n, ∀ h : H (α := α), ExtensionOfCode h.val (seq n).C
  satisfies_update : ∀ n > 0, ∀ h : H (α := α),
    (P n).fn h = (P (n - 1)).fn h * (seq n).Δ (Truncation h.val (seq n) (covers_H n h))

-- Lemma: Eventually Constant Likelihoods

/-- x ≥ ⋁_{i=start}^N C_i -/
def IsBeyondFrontier {α : Type} [Alphabet α] [DecidableEq α] (x : StringAlg α) (D : ℕ → DeltaDigest (α := α)) (start_idx N : ℕ) : Prop :=
  ∀ i, start_idx ≤ i → i ≤ N → ExtensionOfCode x (D i).C

open Finset

lemma likelihood_update_Icc {α : Type} [Alphabet α] [DecidableEq α] {L : LikelihoodSeq (α := α)} {D : LikelihoodDeltaDigestSeq L}
  (h : H (α := α)) (start n : ℕ) (h_start : 1 ≤ start) (h_le : start ≤ n + 1) :
  L.seq n h = L.seq (start - 1) h * ∏ i ∈ Finset.Icc start n, (D.seq i).Δ (Truncation h.val (D.seq i) (D.covers_H i h)) := by
  induction n with
  | zero =>
    have h_start_eq : start = 1 := by omega
    subst h_start_eq
    have h_sub : 1 - 1 = 0 := rfl
    rw [h_sub]
    have h_empty : Finset.Icc 1 0 = ∅ := Finset.Icc_eq_empty_of_lt (by omega)
    rw [h_empty, Finset.prod_empty, mul_one]
  | succ n ih =>
    by_cases h_start_le : start ≤ n + 1
    · have ih_n := ih h_start_le
      have h_upd := D.satisfies_update (n + 1) (by omega) h
      rw [h_upd]
      have h_eq : n + 1 - 1 = n := rfl
      rw [h_eq]
      rw [ih_n]
      have h_prod : ∏ i ∈ Finset.Icc start (n + 1), (D.seq i).Δ (Truncation h.val (D.seq i) (D.covers_H i h)) =
        (∏ i ∈ Finset.Icc start n, (D.seq i).Δ (Truncation h.val (D.seq i) (D.covers_H i h))) *
        (D.seq (n + 1)).Δ (Truncation h.val (D.seq (n + 1)) (D.covers_H (n + 1) h)) := by
        exact Finset.prod_Icc_succ_top h_start_le _
      rw [h_prod]
      exact mul_assoc _ _ _
    · have h_start_eq : start = n + 2 := by omega
      subst h_start_eq
      have h_start_sub : n + 2 - 1 = n + 1 := rfl
      rw [h_start_sub]
      have h_empty : Finset.Icc (n + 2) (n + 1) = ∅ := Finset.Icc_eq_empty_of_lt (by omega)
      rw [h_empty, Finset.prod_empty, mul_one]

lemma prior_update_Icc {α : Type} [Alphabet α] [DecidableEq α] {P : PriorSeq (α := α)} {D : PriorDeltaDigestSeq P}
  (h : H (α := α)) (start n : ℕ) (h_start : 1 ≤ start) (h_le : start ≤ n + 1) :
  (P n).fn h = (P (start - 1)).fn h * ∏ i ∈ Finset.Icc start n, (D.seq i).Δ (Truncation h.val (D.seq i) (D.covers_H i h)) := by
  induction n with
  | zero =>
    have h_start_eq : start = 1 := by omega
    subst h_start_eq
    have h_sub : 1 - 1 = 0 := rfl
    rw [h_sub]
    have h_empty : Finset.Icc 1 0 = ∅ := Finset.Icc_eq_empty_of_lt (by omega)
    rw [h_empty, Finset.prod_empty, mul_one]
  | succ n ih =>
    by_cases h_start_le : start ≤ n + 1
    · have ih_n := ih h_start_le
      have h_upd := D.satisfies_update (n + 1) (by omega) h
      rw [h_upd]
      have h_eq : n + 1 - 1 = n := rfl
      rw [h_eq]
      rw [ih_n]
      have h_prod : ∏ i ∈ Finset.Icc start (n + 1), (D.seq i).Δ (Truncation h.val (D.seq i) (D.covers_H i h)) =
        (∏ i ∈ Finset.Icc start n, (D.seq i).Δ (Truncation h.val (D.seq i) (D.covers_H i h))) *
        (D.seq (n + 1)).Δ (Truncation h.val (D.seq (n + 1)) (D.covers_H (n + 1) h)) := by
        exact Finset.prod_Icc_succ_top h_start_le _
      rw [h_prod]
      exact mul_assoc _ _ _
    · have h_start_eq : start = n + 2 := by omega
      subst h_start_eq
      have h_start_sub : n + 2 - 1 = n + 1 := rfl
      rw [h_start_sub]
      have h_empty : Finset.Icc (n + 2) (n + 1) = ∅ := Finset.Icc_eq_empty_of_lt (by omega)
      rw [h_empty, Finset.prod_empty, mul_one]

lemma likelihood_update_safe_Δ {α : Type} [Alphabet α] [DecidableEq α] {L : LikelihoodSeq (α := α)} {D : LikelihoodDeltaDigestSeq L}
  {start N n : ℕ} (h_start : 1 ≤ start) (h_le : start ≤ n + 1) (h_n : n ≤ N)
  {f : StringAlg α} (h_f : IsBeyondFrontier f D.seq start N)
  (h : H (α := α)) (h_ext : IsPrefix f h.val) :
  L.seq n h = L.seq (start - 1) h * ∏ i ∈ Finset.Icc start n, safe_Δ f (D.seq i) := by
  have h_upd := likelihood_update_Icc (L := L) (D := D) h start n h_start h_le
  rw [h_upd]
  congr 1
  apply Finset.prod_congr rfl
  intro i hi
  have hi_mem := Finset.mem_Icc.mp hi
  have h_safe_eq : safe_Δ f (D.seq i) = (D.seq i).Δ (Truncation f (D.seq i) (h_f i hi_mem.1 (by omega))) := by
    unfold safe_Δ
    have h_ext_f : ExtensionOfCode f (D.seq i).C := h_f i hi_mem.1 (by omega)
    rw [dif_pos h_ext_f]
  rw [h_safe_eq]
  have h_trunc_eq : (Truncation h.val (D.seq i) (D.covers_H i h)).val =
    (Truncation f (D.seq i) (h_f i hi_mem.1 (by omega))).val := by
    have h1 := Truncation_mem h.val (D.seq i) (D.covers_H i h)
    have hp1 := Truncation_is_prefix h.val (D.seq i) (D.covers_H i h)
    have h2 := Truncation_mem f (D.seq i) (h_f i hi_mem.1 (by omega))
    have hp2 := Truncation_is_prefix f (D.seq i) (h_f i hi_mem.1 (by omega))
    have hp2_ext : IsPrefix (Truncation f (D.seq i) (h_f i hi_mem.1 (by omega))).val h.val := by
      exact List.IsPrefix.trans hp2 h_ext
    exact prefix_code_unique_prefix (D.seq i).is_prefix_code h1 hp1 h2 hp2_ext
  have h_trunc_subtype_eq : Truncation h.val (D.seq i) (D.covers_H i h) =
    Truncation f (D.seq i) (h_f i hi_mem.1 (by omega)) := by
    exact SetCoe.ext h_trunc_eq
  rw [h_trunc_subtype_eq]

lemma prior_update_safe_Δ {α : Type} [Alphabet α] [DecidableEq α] {P : PriorSeq (α := α)} {D : PriorDeltaDigestSeq P}
  {start N n : ℕ} (h_start : 1 ≤ start) (h_le : start ≤ n + 1) (h_n : n ≤ N)
  {f : StringAlg α} (h_f : IsBeyondFrontier f D.seq start N)
  (h : H (α := α)) (h_ext : IsPrefix f h.val) :
  (P n).fn h = (P (start - 1)).fn h * ∏ i ∈ Finset.Icc start n, safe_Δ f (D.seq i) := by
  have h_upd := prior_update_Icc (P := P) (D := D) h start n h_start h_le
  rw [h_upd]
  congr 1
  apply Finset.prod_congr rfl
  intro i hi
  have hi_mem := Finset.mem_Icc.mp hi
  have h_safe_eq : safe_Δ f (D.seq i) = (D.seq i).Δ (Truncation f (D.seq i) (h_f i hi_mem.1 (by omega))) := by
    unfold safe_Δ
    have h_ext_f : ExtensionOfCode f (D.seq i).C := h_f i hi_mem.1 (by omega)
    rw [dif_pos h_ext_f]
  rw [h_safe_eq]
  have h_trunc_eq : (Truncation h.val (D.seq i) (D.covers_H i h)).val =
    (Truncation f (D.seq i) (h_f i hi_mem.1 (by omega))).val := by
    have h1 := Truncation_mem h.val (D.seq i) (D.covers_H i h)
    have hp1 := Truncation_is_prefix h.val (D.seq i) (D.covers_H i h)
    have h2 := Truncation_mem f (D.seq i) (h_f i hi_mem.1 (by omega))
    have hp2 := Truncation_is_prefix f (D.seq i) (h_f i hi_mem.1 (by omega))
    have hp2_ext : IsPrefix (Truncation f (D.seq i) (h_f i hi_mem.1 (by omega))).val h.val := by
      exact List.IsPrefix.trans hp2 h_ext
    exact prefix_code_unique_prefix (D.seq i).is_prefix_code h1 hp1 h2 hp2_ext
  have h_trunc_subtype_eq : Truncation h.val (D.seq i) (D.covers_H i h) =
    Truncation f (D.seq i) (h_f i hi_mem.1 (by omega)) := by
    exact SetCoe.ext h_trunc_eq
  rw [h_trunc_subtype_eq]

lemma eventually_constant_likelihoods {α : Type} [Alphabet α] [DecidableEq α] {L : LikelihoodSeq (α := α)} {D : LikelihoodDeltaDigestSeq L}
  {N n : ℕ} (h_n : n ≤ N) {f : StringAlg α} (h_f : IsBeyondFrontier f D.seq 1 N)
  (h : H (α := α)) (h_ext : IsPrefix f h.val) :
  L.seq n h = ∏ i ∈ Icc 1 n, safe_Δ f (D.seq i) := by
  induction n with
  | zero =>
    have h_L_zero : L.seq 0 h = 1 := L.init_one h
    rw [h_L_zero]
    -- The product over Icc 1 0 is empty, so it's 1.
    have h_prod_empty : ∏ i ∈ Icc 1 0, safe_Δ f (D.seq i) = 1 := by
      exact prod_empty
    rw [h_prod_empty]
  | succ n ih =>
    have h_n_le_N : n ≤ N := by omega
    have ih_n := ih h_n_le_N
    -- L.seq (n+1) h = L.seq n h * Δ(Truncation h.val (seq (n+1)))
    have h_update := D.satisfies_update (n + 1) (by omega) h
    rw [h_update]
    have h_eq_n : n + 1 - 1 = n := rfl
    rw [h_eq_n]
    rw [ih_n]
    -- Now we need to show that ∏ i ∈ Icc 1 (n + 1) = ∏ i ∈ Icc 1 n * term(n+1)
    have h_prod : ∏ i ∈ Icc 1 (n + 1), safe_Δ f (D.seq i) =
      (∏ i ∈ Icc 1 n, safe_Δ f (D.seq i)) *
      safe_Δ f (D.seq (n + 1)) := by
      -- we can use prod_Icc_succ_top if 1 <= n + 1
      have h_one_le : 1 ≤ n + 1 := by omega
      exact Finset.prod_Icc_succ_top h_one_le _
    rw [h_prod]
    -- Now we need to show that Truncation h.val = Truncation f.
    -- We know f is a prefix of h.val, and f extends a codeword in D.seq (n+1).C.
    -- Since it's a prefix code, the prefix of h.val in D.seq (n+1).C must be the same as the prefix of f in D.seq (n+1).C.
    have h_trunc_eq : (Truncation h.val (D.seq (n + 1)) (D.covers_H (n + 1) h)).val =
      (Truncation f (D.seq (n + 1)) (h_f (n + 1) (by omega) h_n)).val := by
      have h1 := Truncation_mem h.val (D.seq (n + 1)) (D.covers_H (n + 1) h)
      have hp1 := Truncation_is_prefix h.val (D.seq (n + 1)) (D.covers_H (n + 1) h)
      have h2 := Truncation_mem f (D.seq (n + 1)) (h_f (n + 1) (by omega) h_n)
      have hp2 := Truncation_is_prefix f (D.seq (n + 1)) (h_f (n + 1) (by omega) h_n)
      have hp2_ext : IsPrefix (Truncation f (D.seq (n + 1)) (h_f (n + 1) (by omega) h_n)).val h.val := by
        exact List.IsPrefix.trans hp2 h_ext
      exact prefix_code_unique_prefix (D.seq (n + 1)).is_prefix_code h1 hp1 h2 hp2_ext
    have h_trunc_subtype_eq : Truncation h.val (D.seq (n + 1)) (D.covers_H (n + 1) h) =
      Truncation f (D.seq (n + 1)) (h_f (n + 1) (by omega) h_n) := by
      exact SetCoe.ext h_trunc_eq
    rw [h_trunc_subtype_eq]
    have h_safe_eq : safe_Δ f (D.seq (n + 1)) = (D.seq (n + 1)).Δ (Truncation f (D.seq (n + 1)) (h_f (n + 1) (by omega) h_n)) := by
      unfold safe_Δ
      have h_ext_f : ExtensionOfCode f (D.seq (n + 1)).C := h_f (n + 1) (by omega) h_n
      rw [dif_pos h_ext_f]
    rw [← h_safe_eq]

-- Unnormalized Posterior Function

noncomputable def UnnormalizedPosterior {α : Type} [Alphabet α] [DecidableEq α] (L : LikelihoodSeq (α := α)) (P : PriorSeq (α := α))
  (n m : ℕ) (x : StringAlg α) : NNReal :=
  tsum (fun (h : H (α := α)) => if PrefixOfCode x {h.val} then L.seq n h * (P m).fn h else (0:NNReal))

lemma root_posterior_zero {α : Type} [Alphabet α] [DecidableEq α] {L : LikelihoodSeq (α := α)} {P : PriorSeq (α := α)}
  {m : ℕ} (h_prior_sum : tsum (fun (h : H (α := α)) => (P m).fn h) = 1) :
  UnnormalizedPosterior L P 0 m [] = 1 := by
  unfold UnnormalizedPosterior
  have h_simp : (fun (h : H (α := α)) => if PrefixOfCode [] {h.val} then L.seq 0 h * (P m).fn h else (0:NNReal)) =
    (fun (h : H (α := α)) => (P m).fn h) := by
    ext h
    have h_pref : PrefixOfCode [] {h.val} := by
      use h.val
      constructor
      · rfl
      · exact List.nil_prefix
    have h_L_zero : L.seq 0 h = 1 := L.init_one h
    simp [h_pref, h_L_zero]
  rw [h_simp]
  exact h_prior_sum

lemma posterior_sum_children {α : Type} [Alphabet α] [Fintype α] [DecidableEq α] {L : LikelihoodSeq (α := α)} {P : PriorSeq (α := α)}
  {n m : ℕ} {x : StringAlg α} (hx : ¬ ProperlyTerminated x)
  (h_summable : ∀ a, Summable (fun (h : H (α := α)) => if PrefixOfCode (x ++ [a]) {h.val} then L.seq n h * (P m).fn h else (0:NNReal))) :
  UnnormalizedPosterior L P n m x = ∑ a : α, UnnormalizedPosterior L P n m (x ++ [a]) := by
  unfold UnnormalizedPosterior
  -- We want to show the sum over properly terminated strings prefixed by x is the sum over a of the sums over properly terminated strings prefixed by x ++ [a]
  -- Let S(a) be the set of ProperlyTerminated h that have x ++ [a] as prefix.
  -- Since h is properly terminated, if it has x as prefix but is not x, it must have x ++ [a] as prefix for some unique a.
  -- Since x is not properly terminated, any properly terminated h that has x as prefix cannot be exactly x.
  -- Because `UnnormalizedPosterior` uses `tsum`, we can use `tsum_sum` if we can relate the terms.
  have h_eq : ∀ h : H (α := α),
    (if PrefixOfCode x {h.val} then L.seq n h * (P m).fn h else (0:NNReal)) =
    ∑ a : α, (if PrefixOfCode (x ++ [a]) {h.val} then L.seq n h * (P m).fn h else (0:NNReal)) := by
    intro h
    by_cases h_pref : PrefixOfCode x {h.val}
    · -- x is a prefix of h.val
      -- Since h is properly terminated, and hx says x is not, x cannot be equal to h.val
      -- So x is a proper prefix of h.val
      -- Thus x ++ [a] is a prefix of h.val for exactly one a
      rw [if_pos h_pref]
      -- x is a prefix of h.val, so h.val = x ++ k for some k
      rcases h_pref with ⟨c, hc, h_is_pref⟩
      have h_c_eq : c = h.val := by exact Set.mem_singleton_iff.mp hc
      subst h_c_eq
      cases h_is_pref with | intro k hk =>
        -- k cannot be empty, otherwise x = h.val which is properly terminated, contradicting hx
        have h_k_not_emp : k ≠ [] := by
          intro h_k_emp
          subst h_k_emp
          rw [List.append_nil] at hk
          have h_x_eq_h : x = h.val := hk
          subst h_x_eq_h
          exact hx h.property
        -- Since k is not empty, it has a head `a` and tail `k'`
        cases k with
        | nil => contradiction
        | cons a k' =>
          -- h.val = x ++ (a :: k') = (x ++ [a]) ++ k'
          have h_h_val : h.val = (x ++ [a]) ++ k' := by
            have h_a_k : a :: k' = [a] ++ k' := rfl
            rw [h_a_k] at hk
            rw [← hk]
            exact (List.append_assoc x [a] k').symm
          -- This means x ++ [a] is a prefix of h.val
          have h_pref_a : PrefixOfCode (x ++ [a]) {h.val} := by
            use h.val
            constructor
            · rfl
            · use k'
              exact h_h_val.symm
          have h_sum_split : ∑ b : α, (if PrefixOfCode (x ++ [b]) {h.val} then L.seq n h * (P m).fn h else (0:NNReal)) =
            (if PrefixOfCode (x ++ [a]) {h.val} then L.seq n h * (P m).fn h else (0:NNReal)) +
            ∑ b ∈ Finset.univ \ {a}, (if PrefixOfCode (x ++ [b]) {h.val} then L.seq n h * (P m).fn h else (0:NNReal)) := by
            -- We can split the sum since a is in Finset.univ
            have h_mem : a ∈ Finset.univ := Finset.mem_univ a
            exact Finset.sum_eq_add_sum_diff_singleton h_mem _
          rw [h_sum_split]
          rw [if_pos h_pref_a]
          have h_sum_zero : ∑ b ∈ Finset.univ \ {a}, (if PrefixOfCode (x ++ [b]) {h.val} then L.seq n h * (P m).fn h else (0:NNReal)) = 0 := by
            apply Finset.sum_eq_zero
            intro b hb
            have h_b_neq_a : b ≠ a := by
              have h_not_mem := Finset.mem_sdiff.mp hb
              have h_not_in_sing := h_not_mem.2
              intro h_eq
              rw [h_eq] at h_not_in_sing
              exact h_not_in_sing (Finset.mem_singleton_self a)
            have h_not_pref_b : ¬ PrefixOfCode (x ++ [b]) {h.val} := by
              intro h_pref_b
              rcases h_pref_b with ⟨c, hc, h_is_pref_b⟩
              have h_c_eq_h : c = h.val := by exact Set.mem_singleton_iff.mp hc
              subst h_c_eq_h
              cases h_is_pref_b with | intro k_b hk_b =>
                have h_eq_val : (x ++ [a]) ++ k' = (x ++ [b]) ++ k_b := by
                  rw [← h_h_val]
                  exact hk_b.symm
                have h_eq_b : a = b := by
                  -- Because (x ++ [a]) ++ k' = (x ++ [b]) ++ k_b
                  -- We know the prefix of length x.length + 1 must match
                  -- So x ++ [a] = x ++ [b]
                  have h_pref_eq : List.take (x.length + 1) ((x ++ [a]) ++ k') = x ++ [a] := by
                    have h_len_eq : (x ++ [a]).length = x.length + 1 := by
                      rw [List.length_append, List.length_singleton]
                    rw [← h_len_eq]
                    have h_take := List.take_append_of_le_length (l₁ := x ++ [a]) (l₂ := k') (by exact le_refl _)
                    -- we know take_append_of_le_length takes exactly l1
                    exact List.take_left' (by rfl)
                  have h_pref_eq_b : List.take (x.length + 1) ((x ++ [b]) ++ k_b) = x ++ [b] := by
                    have h_len_eq_b : (x ++ [b]).length = x.length + 1 := by
                      rw [List.length_append, List.length_singleton]
                    rw [← h_len_eq_b]
                    have h_take_b := List.take_append_of_le_length (l₁ := x ++ [b]) (l₂ := k_b) (by exact le_refl _)
                    -- same
                    exact List.take_left' (by rfl)
                  rw [h_eq_val] at h_pref_eq
                  have h_x_a_eq_x_b : x ++ [a] = x ++ [b] := by
                    rw [← h_pref_eq, h_pref_eq_b]
                  have h_list_eq : [a] = [b] := List.append_inj_right' h_x_a_eq_x_b rfl
                  injection h_list_eq
                exact h_b_neq_a h_eq_b.symm
            rw [if_neg h_not_pref_b]
          rw [h_sum_zero]
          exact (add_zero _).symm
    · -- x is not a prefix, so x ++ [a] is also not a prefix
      rw [if_neg h_pref]
      -- we need to show that for all a, x ++ [a] is not a prefix
      have h_sum_zero : ∑ a : α, (if PrefixOfCode (x ++ [a]) {h.val} then L.seq n h * (P m).fn h else (0:NNReal)) = 0 := by
        apply Finset.sum_eq_zero
        intro a _ha
        have h_not_pref_a : ¬ PrefixOfCode (x ++ [a]) {h.val} := by
          intro h_pref_a
          rcases h_pref_a with ⟨c, hc, h_is_pref⟩
          have h_c_eq : c = h.val := by exact Set.mem_singleton_iff.mp hc
          subst h_c_eq
          have h_x_pref_h : IsPrefix x h.val := by
            -- if x ++ [a] is a prefix of h.val, then x is a prefix of h.val
            cases h_is_pref with | intro k hk =>
              use [a] ++ k
              have h_a_k : [a] ++ k = a :: k := rfl
              rw [h_a_k]
              rw [← hk]
              exact (List.append_assoc x [a] k).symm
          exact h_pref ⟨h.val, Set.mem_singleton h.val, h_x_pref_h⟩
        rw [if_neg h_not_pref_a]
      exact h_sum_zero.symm
  -- Because `UnnormalizedPosterior` uses `tsum`, we can use `tsum_sum` if we can relate the terms.
  have h_tsum_eq : tsum (fun (h : H (α := α)) => if PrefixOfCode x {h.val} then L.seq n h * (P m).fn h else (0:NNReal)) =
    tsum (fun (h : H (α := α)) => ∑ a : α, (if PrefixOfCode (x ++ [a]) {h.val} then L.seq n h * (P m).fn h else (0:NNReal))) := by
    exact congrArg tsum (funext h_eq)
  rw [h_tsum_eq]
  have h_tsum_sum : tsum (fun (h : H (α := α)) => ∑ a : α, (if PrefixOfCode (x ++ [a]) {h.val} then L.seq n h * (P m).fn h else (0:NNReal))) =
    ∑ a : α, tsum (fun (h : H (α := α)) => if PrefixOfCode (x ++ [a]) {h.val} then L.seq n h * (P m).fn h else (0:NNReal)) := by
    exact Summable.tsum_finsetSum (fun a _ => h_summable a)
  rw [h_tsum_sum]

-- Valid Node and Bayesian Trie

structure BayesianTrieNode {α : Type} [Alphabet α] [DecidableEq α] where
  val : StringAlg α
  n : ℕ
  m : ℕ
  Z : NNReal

def IsValidNode {α : Type} [Alphabet α] [DecidableEq α] (x : BayesianTrieNode (α := α)) (N M : ℕ)
  {L : LikelihoodSeq (α := α)} {P : PriorSeq (α := α)}
  (DL : LikelihoodDeltaDigestSeq L) (DP : PriorDeltaDigestSeq P) : Prop :=
  IsBeyondFrontier x.val DL.seq 1 N ∨
  (IsBeyondFrontier x.val DL.seq (x.n + 1) N ∧ IsBeyondFrontier x.val DP.seq (x.m + 1) M)

noncomputable instance {α : Type} [Alphabet α] [DecidableEq α] {x : StringAlg α} {n' : ℕ} {L : LikelihoodSeq (α := α)} {DL : LikelihoodDeltaDigestSeq L} : Decidable (IsBeyondFrontier x DL.seq 1 n') := Classical.dec _

lemma current_nodes_are_valid {α : Type} [Alphabet α] [DecidableEq α] (x : BayesianTrieNode (α := α))
  (N M : ℕ)
  {L : LikelihoodSeq (α := α)} {P : PriorSeq (α := α)}
  (DL : LikelihoodDeltaDigestSeq L) (DP : PriorDeltaDigestSeq P)
  (h_n : x.n = N) (h_m : x.m = M) :
  IsValidNode x N M DL DP := by
  right
  subst h_n h_m
  constructor
  · intro i h_start h_end
    -- x.n + 1 <= i <= x.n is impossible
    linarith
  · intro i h_start h_end
    linarith

def IsCorrectNode {α : Type} [Alphabet α] [DecidableEq α] (x : BayesianTrieNode (α := α))
  (L : LikelihoodSeq (α := α)) (P : PriorSeq (α := α)) : Prop :=
  x.Z = UnnormalizedPosterior L P x.n x.m x.val

structure BayesianTrie {α : Type} [Alphabet α] [DecidableEq α] where
  N : ℕ
  M : ℕ
  nodes : StringAlg α → BayesianTrieNode (α := α)
  val_eq : ∀ x, (nodes x).val = x

def IsValidTrie {α : Type} [Alphabet α] [DecidableEq α] (trie : BayesianTrie (α := α))
  {L : LikelihoodSeq (α := α)} {P : PriorSeq (α := α)}
  (DL : LikelihoodDeltaDigestSeq L) (DP : PriorDeltaDigestSeq P) : Prop :=
  ∀ x, IsValidNode (trie.nodes x) trie.N trie.M DL DP

def IsCorrectTrie {α : Type} [Alphabet α] [DecidableEq α] (trie : BayesianTrie (α := α))
  (L : LikelihoodSeq (α := α)) (P : PriorSeq (α := α)) : Prop :=
  ∀ x, IsCorrectNode (trie.nodes x) L P

-- Algorithms

noncomputable def AdvanceNodeTime {α : Type} [Alphabet α] [DecidableEq α] (x : BayesianTrieNode (α := α))
  (n' m' : ℕ)
  {L : LikelihoodSeq (α := α)} {P : PriorSeq (α := α)}
  (DL : LikelihoodDeltaDigestSeq L) (DP : PriorDeltaDigestSeq P)
  (h_valid : IsValidNode x n' m' DL DP) : BayesianTrieNode (α := α) :=
  let _inst : Decidable (IsBeyondFrontier x.val DL.seq 1 n') := Classical.dec _
  if h_const : (IsBeyondFrontier x.val DL.seq 1 n' : Prop) then
    let new_Z := BranchPrior (P m') x.val * ∏ i ∈ Finset.Icc 1 n', safe_Δ x.val (DL.seq i)
    { x with n := n', m := m', Z := new_Z }
  else
    -- x is not beyond the frontier up to n'. Since it is valid, it must be that
    -- it's beyond the frontier from x.n + 1 to n', and x.m + 1 to m'
    have h_valid' : (IsBeyondFrontier x.val DL.seq (x.n + 1) n' ∧ IsBeyondFrontier x.val DP.seq (x.m + 1) m') := by
      rcases h_valid with h1 | h2
      · contradiction
      · exact h2
    let new_Z := x.Z *
      (∏ i ∈ Finset.Icc (x.n + 1) n', safe_Δ x.val (DL.seq i)) *
      (∏ i ∈ Finset.Icc (x.m + 1) m', safe_Δ x.val (DP.seq i))
    { x with n := n', m := m', Z := new_Z }

lemma advance_node_time_validity {α : Type} [Alphabet α] [DecidableEq α] {x : BayesianTrieNode (α := α)}
  {n' m' : ℕ}
  {L : LikelihoodSeq (α := α)} {P : PriorSeq (α := α)}
  {DL : LikelihoodDeltaDigestSeq L} {DP : PriorDeltaDigestSeq P}
  (h_valid : IsValidNode x n' m' DL DP) :
  IsValidNode (AdvanceNodeTime x n' m' DL DP h_valid) n' m' DL DP := by
  unfold IsValidNode
  unfold AdvanceNodeTime
  -- dsimp to get inside the if
  split_ifs with h_const
  · -- it's constant
    right
    constructor
    · intro i h_start h_end
      -- n' + 1 <= i <= n' is impossible
      linarith
    · intro i h_start h_end
      linarith
  · -- it's not constant
    right
    constructor
    · intro i h_start h_end
      linarith
    · intro i h_start h_end
      linarith

lemma advance_node_time_correctness {α : Type} [Alphabet α] [DecidableEq α] {x : BayesianTrieNode (α := α)}
  {n' m' : ℕ}
  {L : LikelihoodSeq (α := α)} {P : PriorSeq (α := α)}
  {DL : LikelihoodDeltaDigestSeq L} {DP : PriorDeltaDigestSeq P}
  (h_valid : IsValidNode x n' m' DL DP)
  (h_correct : IsCorrectNode x L P)
  (h_n_le : x.n ≤ n') (h_m_le : x.m ≤ m') :
  IsCorrectNode (AdvanceNodeTime x n' m' DL DP h_valid) L P := by
  unfold IsCorrectNode at *
  unfold AdvanceNodeTime
  split_ifs with h_const
  · -- x is beyond the frontier, so its likelihood is constant and equal to the product of deltas up to n'.
    -- We can pull the product out of the tsum.
    dsimp
    unfold UnnormalizedPosterior
    have h_prod_eq : ∀ h : H (α := α), PrefixOfCode x.val {h.val} →
      L.seq n' h = ∏ i ∈ Finset.Icc 1 n', safe_Δ x.val (DL.seq i) := by
      intro h h_pref
      -- PrefixOfCode means x.val is a prefix of some element in {h.val}, which means x.val is a prefix of h.val
      rcases h_pref with ⟨c, hc, h_is_pref⟩
      have h_c_eq : c = h.val := by exact Set.mem_singleton_iff.mp hc
      subst h_c_eq
      exact eventually_constant_likelihoods (by exact le_refl _) h_const h h_is_pref
    have h_tsum_eq : tsum (fun h : H (α := α) => if PrefixOfCode x.val {h.val} then L.seq n' h * (P m').fn h else (0:NNReal)) =
      tsum (fun h : H (α := α) => if PrefixOfCode x.val {h.val} then (∏ i ∈ Finset.Icc 1 n', safe_Δ x.val (DL.seq i)) * (P m').fn h else (0:NNReal)) := by
      congr 1
      ext h
      by_cases h_pref : PrefixOfCode x.val {h.val}
      · rw [if_pos h_pref, if_pos h_pref]
        rw [h_prod_eq h h_pref]
      · rw [if_neg h_pref, if_neg h_pref]
    rw [h_tsum_eq]
    -- Now we can pull out the product because it doesn't depend on h
    -- It is a constant factor multiplied by (P m').fn h
    -- The definition of BranchPrior is exactly the tsum of (P m').fn h over prefixes
    -- so this is (∏ ...) * BranchPrior (P m') x.val
    have h_pull_out : tsum (fun h : H (α := α) => if PrefixOfCode x.val {h.val} then (∏ i ∈ Finset.Icc 1 n', safe_Δ x.val (DL.seq i)) * (P m').fn h else (0:NNReal)) =
      (∏ i ∈ Finset.Icc 1 n', safe_Δ x.val (DL.seq i)) * tsum (fun h : H (α := α) => if PrefixOfCode x.val {h.val} then (P m').fn h else (0:NNReal)) := by
      -- we can pull out the constant product
      have h_fun_eq : (fun h : H (α := α) => if PrefixOfCode x.val {h.val} then (∏ i ∈ Finset.Icc 1 n', safe_Δ x.val (DL.seq i)) * (P m').fn h else (0:NNReal)) =
        (fun h : H (α := α) => (∏ i ∈ Finset.Icc 1 n', safe_Δ x.val (DL.seq i)) * (if PrefixOfCode x.val {h.val} then (P m').fn h else (0:NNReal))) := by
        ext h
        by_cases h_pref : PrefixOfCode x.val {h.val}
        · rw [if_pos h_pref, if_pos h_pref]
        · rw [if_neg h_pref, if_neg h_pref, mul_zero]
      rw [h_fun_eq]
      have h_tsum_mul_left_inst : ∑' (h : H (α := α)), (∏ i ∈ Finset.Icc 1 n', safe_Δ x.val (DL.seq i)) * (if PrefixOfCode x.val {h.val} then (P m').fn h else 0) = (∏ i ∈ Finset.Icc 1 n', safe_Δ x.val (DL.seq i)) * ∑' (h : H (α := α)), if PrefixOfCode x.val {h.val} then (P m').fn h else 0 := by
        exact tsum_mul_left
      rw [h_tsum_mul_left_inst]
    rw [h_pull_out]
    -- Now rewrite BranchPrior definition
    have h_branch_prior : BranchPrior (P m') x.val = tsum (fun h : H (α := α) => if PrefixOfCode x.val {h.val} then (P m').fn h else (0:NNReal)) := rfl
    rw [h_branch_prior]
    exact mul_comm _ _
  · -- x is not beyond the frontier up to n'
    -- but we know it was valid up to x.n and x.m, so its likelihood is constant from x.n + 1 to n'
    -- and its prior is constant from x.m + 1 to m'
    dsimp
    unfold UnnormalizedPosterior
    have h_valid' : (IsBeyondFrontier x.val DL.seq (x.n + 1) n' ∧ IsBeyondFrontier x.val DP.seq (x.m + 1) m') := by
      rcases h_valid with h1 | h2
      · contradiction
      · exact h2
    -- using h_valid'.1 and h_valid'.2, we can show that for any prefix h of x,
    -- L.seq n' h = L.seq x.n h * ∏ i ∈ Icc (x.n + 1) n', Δ
    -- (P m').fn h = (P x.m).fn h * ∏ i ∈ Icc (x.m + 1) m', Δ
    have h_L_eq : ∀ h : H (α := α), PrefixOfCode x.val {h.val} →
      L.seq n' h = L.seq x.n h * ∏ i ∈ Finset.Icc (x.n + 1) n', safe_Δ x.val (DL.seq i) := by
      intro h h_pref
      rcases h_pref with ⟨c, hc, h_is_pref⟩
      have h_c_eq : c = h.val := by exact Set.mem_singleton_iff.mp hc
      subst h_c_eq
      have h_start_le : 1 ≤ x.n + 1 := by omega
      have h_n_le' : x.n + 1 ≤ n' + 1 := by omega
      have h_upd := likelihood_update_safe_Δ (start := x.n + 1) (N := n') (n := n') h_start_le h_n_le' (by exact le_refl _) h_valid'.1 h h_is_pref
      have h_x_n_eq : x.n + 1 - 1 = x.n := rfl
      rw [h_x_n_eq] at h_upd
      exact h_upd
    have h_P_eq : ∀ h : H (α := α), PrefixOfCode x.val {h.val} →
      (P m').fn h = (P x.m).fn h * ∏ i ∈ Finset.Icc (x.m + 1) m', safe_Δ x.val (DP.seq i) := by
      intro h h_pref
      rcases h_pref with ⟨c, hc, h_is_pref⟩
      have h_c_eq : c = h.val := by exact Set.mem_singleton_iff.mp hc
      subst h_c_eq
      have h_start_le : 1 ≤ x.m + 1 := by omega
      have h_m_le' : x.m + 1 ≤ m' + 1 := by omega
      have h_upd := prior_update_safe_Δ (start := x.m + 1) (N := m') (n := m') h_start_le h_m_le' (by exact le_refl _) h_valid'.2 h h_is_pref
      have h_x_m_eq : x.m + 1 - 1 = x.m := rfl
      rw [h_x_m_eq] at h_upd
      exact h_upd
    -- Then we can substitute these into the tsum, pull the products out, and use h_correct.
    have h_tsum_eq : tsum (fun h : H (α := α) => if PrefixOfCode x.val {h.val} then L.seq n' h * (P m').fn h else (0:NNReal)) =
      tsum (fun h : H (α := α) => if PrefixOfCode x.val {h.val} then (L.seq x.n h * (P x.m).fn h) * (∏ i ∈ Finset.Icc (x.n + 1) n', safe_Δ x.val (DL.seq i)) * (∏ i ∈ Finset.Icc (x.m + 1) m', safe_Δ x.val (DP.seq i)) else (0:NNReal)) := by
      congr 1
      ext h
      by_cases h_pref : PrefixOfCode x.val {h.val}
      · rw [if_pos h_pref, if_pos h_pref]
        rw [h_L_eq h h_pref, h_P_eq h h_pref]
        congr 1
        rw [mul_assoc (L.seq x.n h)]
        rw [mul_left_comm (∏ i ∈ Finset.Icc (x.n + 1) n', safe_Δ x.val (DL.seq i))]
        rw [← mul_assoc (L.seq x.n h)]
        rw [mul_assoc (L.seq x.n h * (P x.m).fn h)]
      · rw [if_neg h_pref, if_neg h_pref]
    rw [h_tsum_eq]
    have h_pull : tsum (fun h : H (α := α) => if PrefixOfCode x.val {h.val} then (L.seq x.n h * (P x.m).fn h) * (∏ i ∈ Finset.Icc (x.n + 1) n', safe_Δ x.val (DL.seq i)) * (∏ i ∈ Finset.Icc (x.m + 1) m', safe_Δ x.val (DP.seq i)) else (0:NNReal)) =
      tsum (fun h : H (α := α) => if PrefixOfCode x.val {h.val} then (L.seq x.n h * (P x.m).fn h) else (0:NNReal)) * (∏ i ∈ Finset.Icc (x.n + 1) n', safe_Δ x.val (DL.seq i)) * (∏ i ∈ Finset.Icc (x.m + 1) m', safe_Δ x.val (DP.seq i)) := by
      have h_fun_eq : (fun h : H (α := α) => if PrefixOfCode x.val {h.val} then (L.seq x.n h * (P x.m).fn h) * (∏ i ∈ Finset.Icc (x.n + 1) n', safe_Δ x.val (DL.seq i)) * (∏ i ∈ Finset.Icc (x.m + 1) m', safe_Δ x.val (DP.seq i)) else (0:NNReal)) =
        (fun h : H (α := α) => (if PrefixOfCode x.val {h.val} then (L.seq x.n h * (P x.m).fn h) else (0:NNReal)) * ((∏ i ∈ Finset.Icc (x.n + 1) n', safe_Δ x.val (DL.seq i)) * (∏ i ∈ Finset.Icc (x.m + 1) m', safe_Δ x.val (DP.seq i)))) := by
        ext h
        by_cases h_pref : PrefixOfCode x.val {h.val}
        · rw [if_pos h_pref, if_pos h_pref]
          exact mul_assoc _ _ _
        · rw [if_neg h_pref, if_neg h_pref, zero_mul]
      rw [h_fun_eq]
      have h_tsum_mul_right_inst : ∑' (h : H (α := α)), (if PrefixOfCode x.val {h.val} then L.seq x.n h * (P x.m).fn h else 0) * ((∏ i ∈ Finset.Icc (x.n + 1) n', safe_Δ x.val (DL.seq i)) * ∏ i ∈ Finset.Icc (x.m + 1) m', safe_Δ x.val (DP.seq i)) = (∑' (h : H (α := α)), if PrefixOfCode x.val {h.val} then L.seq x.n h * (P x.m).fn h else 0) * ((∏ i ∈ Finset.Icc (x.n + 1) n', safe_Δ x.val (DL.seq i)) * ∏ i ∈ Finset.Icc (x.m + 1) m', safe_Δ x.val (DP.seq i)) := by
        exact tsum_mul_right
      rw [h_tsum_mul_right_inst]
      rw [mul_assoc]
    rw [h_pull]
    have h_corr_eq : tsum (fun h : H (α := α) => if PrefixOfCode x.val {h.val} then (L.seq x.n h * (P x.m).fn h) else (0:NNReal)) = x.Z := by
      exact h_correct.symm
    rw [h_corr_eq]

lemma beyond_frontier_extend {α : Type} [Alphabet α] [DecidableEq α] (x c : StringAlg α) (D : ℕ → DeltaDigest (α := α)) (start N : ℕ)
  (hc : IsBeyondFrontier c D start N) (h_ext : IsPrefix c x) :
  IsBeyondFrontier x D start N := by
  intro i h_start h_end
  have ⟨codeword, hc_in, h_pref⟩ := hc i h_start h_end
  use codeword
  constructor
  · exact hc_in
  · exact List.IsPrefix.trans h_pref h_ext

lemma beyond_frontier_glue {α : Type} [Alphabet α] [DecidableEq α] (x : StringAlg α) (D : ℕ → DeltaDigest (α := α)) (a b c : ℕ)
  (h1 : IsBeyondFrontier x D a b) (h2 : IsBeyondFrontier x D (b + 1) c) :
  IsBeyondFrontier x D a c := by
  intro i h_start h_end
  by_cases h_le : i ≤ b
  · exact h1 i h_start h_le
  · exact h2 i (by omega) h_end

/-- AdvanceTrieTime returns a new trie. We specify its properties. -/
def AdvanceTrieTimeProp {α : Type} [Alphabet α] [Fintype α] [DecidableEq α]
  (old_trie : BayesianTrie (α := α)) (new_trie : BayesianTrie (α := α))
  (F : Set (StringAlg α)) (_h_complete : CompletePrefixCode F)
  (N' M' : ℕ)
  {L : LikelihoodSeq (α := α)} {P : PriorSeq (α := α)}
  (DL : LikelihoodDeltaDigestSeq L) (DP : PriorDeltaDigestSeq P)
  (h_F_valid : ∀ x ∈ F, IsValidNode (old_trie.nodes x) N' M' DL DP) : Prop :=
  new_trie.N = N' ∧
  new_trie.M = M' ∧
  -- nodes in frontier are advanced
  (∀ x (hx : x ∈ F), new_trie.nodes x = AdvanceNodeTime (old_trie.nodes x) N' M' DL DP (h_F_valid x hx)) ∧
  -- nodes beyond frontier are unchanged
  (∀ x, ProperExtensionOfCode x F → new_trie.nodes x = old_trie.nodes x) ∧
  -- nodes strictly within frontier update in reverse topological order (Z becomes sum of children)
  (∀ x, (∃ c ∈ F, IsStrictPrefix x c) →
    (new_trie.nodes x).n = N' ∧
    (new_trie.nodes x).m = M' ∧
    (new_trie.nodes x).Z = ∑ a : α, (new_trie.nodes (x ++ [a])).Z)

lemma advance_trie_time_validity {α : Type} [Alphabet α] [Fintype α] [DecidableEq α]
  (old_trie new_trie : BayesianTrie (α := α))
  (F : Set (StringAlg α)) (h_complete : CompletePrefixCode F)
  (N' M' : ℕ)
  {L : LikelihoodSeq (α := α)} {P : PriorSeq (α := α)}
  (DL : LikelihoodDeltaDigestSeq L) (DP : PriorDeltaDigestSeq P)
  (h_F_valid : ∀ x ∈ F, IsValidNode (old_trie.nodes x) N' M' DL DP)
  (h_prop : AdvanceTrieTimeProp old_trie new_trie F h_complete N' M' DL DP h_F_valid)
  (_h_old_valid : IsValidTrie old_trie DL DP)
  (h_ext_valid : ∀ x, ProperExtensionOfCode x F → IsValidNode (old_trie.nodes x) N' M' DL DP) :
  IsValidTrie new_trie DL DP := by
  unfold IsValidTrie
  intro x
  rcases h_prop with ⟨h_N, h_M, h_F, h_ext, h_pref⟩
  rw [h_N, h_M]
  -- Trichotomy: x in F, x > F, or x < F
  have h_cases : x ∈ F ∨ ProperExtensionOfCode x F ∨ (∃ c ∈ F, IsStrictPrefix x c) := by
    have ⟨c, hc_in, h_c_pref⟩ := h_complete.2 x
    rcases h_c_pref with h_x_c | h_c_x
    · by_cases h_eq : x = c
      · left; subst h_eq; exact hc_in
      · right; right; use c; exact ⟨hc_in, h_x_c, h_eq⟩
    · by_cases h_eq : c = x
      · left; subst h_eq; exact hc_in
      · right; left; use c; exact ⟨hc_in, h_c_x, h_eq⟩
  rcases h_cases with hx_F | hx_ext | hx_pref
  · -- x in F
    rw [h_F x hx_F]
    exact advance_node_time_validity (h_F_valid x hx_F)
  · -- x > F
    rw [h_ext x hx_ext]
    exact h_ext_valid x hx_ext
  · -- x < F
    have h_upd := h_pref x hx_pref
    unfold IsValidNode
    right
    constructor
    · rw [h_upd.1]
      -- IsBeyondFrontier x DL.seq (N' + 1) N' is vacuously true
      intro i h_start h_end
      linarith
    · rw [h_upd.2.1]
      intro i h_start h_end
      linarith

lemma advance_trie_time_correctness {α : Type} [Alphabet α] [Fintype α] [DecidableEq α]
  (old_trie new_trie : BayesianTrie (α := α))
  (F : Set (StringAlg α)) (h_complete : CompletePrefixCode F)
  (N' M' : ℕ) (h_N_le : old_trie.N ≤ N') (h_M_le : old_trie.M ≤ M')
  {L : LikelihoodSeq (α := α)} {P : PriorSeq (α := α)}
  (DL : LikelihoodDeltaDigestSeq L) (DP : PriorDeltaDigestSeq P)
  (h_F_valid : ∀ x ∈ F, IsValidNode (old_trie.nodes x) N' M' DL DP)
  (h_prop : AdvanceTrieTimeProp old_trie new_trie F h_complete N' M' DL DP h_F_valid)
  (h_old_n_le : ∀ x, (old_trie.nodes x).n ≤ old_trie.N)
  (h_old_m_le : ∀ x, (old_trie.nodes x).m ≤ old_trie.M)
  (h_old_correct : IsCorrectTrie old_trie L P)
  (h_summable : ∀ x, ¬ ProperlyTerminated x → ∀ a : α,
    Summable (fun h : H (α := α) => if PrefixOfCode (x ++ [a]) {h.val} then L.seq N' h * (P M').fn h else (0:NNReal)))
  (h_not_term : ∀ x, (∃ c ∈ F, IsStrictPrefix x c) → ¬ ProperlyTerminated x)
  (h_children_correct : ∀ x, (∃ c ∈ F, IsStrictPrefix x c) → ∀ a : α,
    (new_trie.nodes (x ++ [a])).n = N' ∧
    (new_trie.nodes (x ++ [a])).m = M' ∧
    IsCorrectNode (new_trie.nodes (x ++ [a])) L P) :
  IsCorrectTrie new_trie L P := by
  unfold IsCorrectTrie
  intro x
  rcases h_prop with ⟨h_N, h_M, h_F, h_ext, h_pref⟩
  have h_cases : x ∈ F ∨ ProperExtensionOfCode x F ∨ (∃ c ∈ F, IsStrictPrefix x c) := by
    have ⟨c, hc_in, h_c_pref⟩ := h_complete.2 x
    rcases h_c_pref with h_x_c | h_c_x
    · by_cases h_eq : x = c
      · left; subst h_eq; exact hc_in
      · right; right; use c; exact ⟨hc_in, h_x_c, h_eq⟩
    · by_cases h_eq : c = x
      · left; subst h_eq; exact hc_in
      · right; left; use c; exact ⟨hc_in, h_c_x, h_eq⟩
  rcases h_cases with hx_F | hx_ext | hx_pref
  · -- x in F
    have h_old_c := h_old_correct x
    have h_val : (old_trie.nodes x).val = x := old_trie.val_eq x
    have h_n_le : (old_trie.nodes x).n ≤ N' := by
      exact le_trans (h_old_n_le x) h_N_le
    have h_m_le : (old_trie.nodes x).m ≤ M' := by
      exact le_trans (h_old_m_le x) h_M_le
    rw [h_F x hx_F]
    have h_corr := advance_node_time_correctness (h_F_valid x hx_F) h_old_c h_n_le h_m_le
    -- advance_node_time_correctness requires h_n_le and h_m_le
    exact h_corr
  · -- x > F
    rw [h_ext x hx_ext]
    exact h_old_correct x
  · -- x < F
    have h_upd := h_pref x hx_pref
    have h_child_corr := h_children_correct x hx_pref
    unfold IsCorrectNode at *
    rw [h_upd.1, h_upd.2.1, h_upd.2.2]
    have h_val : (new_trie.nodes x).val = x := new_trie.val_eq x
    rw [h_val]
    have h_post := posterior_sum_children (hx := h_not_term x hx_pref) (h_summable x (h_not_term x hx_pref))
    rw [h_post]
    apply Finset.sum_congr rfl
    intro a _ha
    have h_a_corr := h_child_corr a
    have h_a_val : (new_trie.nodes (x ++ [a])).val = x ++ [a] := new_trie.val_eq (x ++ [a])
    rw [h_a_corr.1, h_a_corr.2.1, h_a_val] at h_a_corr
    exact h_a_corr.2.2
