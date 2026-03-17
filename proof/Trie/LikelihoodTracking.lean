import Trie.Core

open scoped BigOperators
open Finset

namespace Trie

variable {α : Type} [Alphabet α] [DecidableEq α]

/-- A trie node augmented with the unpushed likelihood `l` and local likelihood time `n_lt`. -/
structure LikelihoodTrackingNode where
  val : StringAlg α
  l : NNReal
  n_lt : ℕ

/-- A trie augmented with a global likelihood-tracking time `N_lt`. -/
structure LikelihoodTrackingTrie where
  N_lt : ℕ
  nodes : StringAlg α → LikelihoodTrackingNode (α := α)
  val_eq : ∀ x, (nodes x).val = x

/-- The prefixes of a string, viewed as the finite set of its ancestors in the trie. -/
def Prefixes (x : StringAlg α) : Finset (StringAlg α) :=
  x.inits.toFinset

omit [Alphabet α] in
lemma mem_Prefixes {x y : StringAlg α} :
    y ∈ Prefixes x ↔ IsPrefix y x := by
  unfold Prefixes IsPrefix
  rw [List.mem_toFinset, List.mem_inits]

omit [Alphabet α] in
lemma self_mem_Prefixes (x : StringAlg α) :
    x ∈ Prefixes x := by
  exact mem_Prefixes.mpr ⟨[], by simp⟩

omit [Alphabet α] in
lemma prefix_length_le {x y : StringAlg α} (h : IsPrefix x y) :
    x.length ≤ y.length := by
  rcases h with ⟨z, rfl⟩
  simp

omit [Alphabet α] in
lemma prefix_eq_of_length_eq {x y : StringAlg α}
    (h : IsPrefix x y) (hlen : x.length = y.length) : x = y := by
  rcases h with ⟨z, rfl⟩
  simp at hlen
  subst z
  simp

/-- Product of unpushed likelihoods over all ancestors of `x`, including `x`. -/
def AncestorLikelihoodProduct
    (B : LikelihoodTrackingTrie (α := α))
    (x : StringAlg α) : NNReal :=
  ∏ y ∈ Prefixes x, (B.nodes y).l

/-- Every node's local likelihood time is bounded by the global time. -/
def LikelihoodTimesBounded (B : LikelihoodTrackingTrie (α := α)) : Prop :=
  ∀ x, (B.nodes x).n_lt ≤ B.N_lt

/--
Push correctness:
for nodes beyond the cumulative frontier from their local time to the current global time,
the ancestor product of unpushed likelihoods equals the corresponding product of deltas.
-/
def PushCorrect
    {L : LikelihoodSeq (α := α)}
    (DL : LikelihoodDeltaDigestSeq L)
    (B : LikelihoodTrackingTrie (α := α)) : Prop :=
  ∀ x,
    IsBeyondFrontier x DL.seq ((B.nodes x).n_lt + 1) B.N_lt →
      AncestorLikelihoodProduct B x =
        ∏ i ∈ Icc ((B.nodes x).n_lt + 1) B.N_lt, safe_Δ x (DL.seq i)

/-- `x` has no strict ancestors carrying unpushed likelihood. -/
def NoUnpushedAncestors
    (B : LikelihoodTrackingTrie (α := α))
    (x : StringAlg α) : Prop :=
  ∀ a, IsStrictPrefix a x → (B.nodes a).l = 1

/-- `child` is an immediate child of `parent`. -/
def IsChild (parent child : StringAlg α) : Prop :=
  ∃ a : α, child = parent ++ [a]

omit [Alphabet α] in
lemma child_has_prefix {parent child : StringAlg α} (h : IsChild parent child) :
    IsPrefix parent child := by
  rcases h with ⟨a, rfl⟩
  exact ⟨[a], by simp⟩

omit [Alphabet α] [DecidableEq α] in
lemma child_length {parent child : StringAlg α} (h : IsChild parent child) :
    child.length = parent.length + 1 := by
  rcases h with ⟨a, rfl⟩
  simp

omit [Alphabet α] in
lemma exists_child_prefix_of_strictPrefix {x y : StringAlg α}
    (hxy : IsStrictPrefix x y) :
    ∃ z, IsChild x z ∧ IsPrefix z y := by
  rcases hxy.1 with ⟨tail, rfl⟩
  have htail_ne : tail ≠ [] := by
    intro hnil
    apply hxy.2
    simp [hnil]
  cases tail with
  | nil =>
      contradiction
  | cons a rest =>
      refine ⟨x ++ [a], ?_, ?_⟩
      · exact ⟨a, rfl⟩
      · exact ⟨rest, by simp⟩

omit [Alphabet α] in
lemma child_prefix_unique {x c y z : StringAlg α}
    (hy : IsChild c y) (hz : IsChild c z)
    (hyx : IsPrefix y x) (hzx : IsPrefix z x) :
    y = z := by
  have hy_or_hz := prefix_or_prefix_of_prefix hyx hzx
  have hlen : y.length = z.length := by
    rw [child_length hy, child_length hz]
  rcases hy_or_hz with hyz | hzy
  · exact prefix_eq_of_length_eq hyz hlen
  · exact (prefix_eq_of_length_eq hzy hlen.symm).symm

omit [Alphabet α] in
lemma not_isChild_of_prefix {x y : StringAlg α} (hyx : IsPrefix y x) :
    ¬ IsChild x y := by
  intro hchild
  have hlen_le : y.length ≤ x.length := prefix_length_le hyx
  have hlen_eq : y.length = x.length + 1 := child_length hchild
  omega

def NextLikelihoodDigest
    {L : LikelihoodSeq (α := α)}
    (B : LikelihoodTrackingTrie (α := α))
    (DL : LikelihoodDeltaDigestSeq L) : DeltaDigest (α := α) :=
  DL.seq (B.N_lt + 1)

def NextLikelihoodFrontier
    {L : LikelihoodSeq (α := α)}
    (B : LikelihoodTrackingTrie (α := α))
    (DL : LikelihoodDeltaDigestSeq L) : Set (StringAlg α) :=
  (NextLikelihoodDigest B DL).C

lemma safe_Δ_eq_of_mem_prefix
    (D : DeltaDigest (α := α))
    {x c : StringAlg α}
    (hc : c ∈ D.C)
    (hcx : IsPrefix c x) :
    safe_Δ x D = D.Δ ⟨c, hc⟩ := by
  unfold safe_Δ
  let h_ext : ExtensionOfCode x D.C := ⟨c, hc, hcx⟩
  rw [dif_pos h_ext]
  have h_trunc_eq : (Truncation x D h_ext).val = c := by
    have h1 := Truncation_mem x D h_ext
    have hp1 := Truncation_is_prefix x D h_ext
    exact prefix_code_unique_prefix D.is_prefix_code h1 hp1 hc hcx
  have h_subtype_eq : Truncation x D h_ext = ⟨c, hc⟩ := by
    exact SetCoe.ext h_trunc_eq
  rw [h_subtype_eq]

lemma safe_Δ_next_eq_of_prefix
    {L : LikelihoodSeq (α := α)}
    (B : LikelihoodTrackingTrie (α := α))
    (DL : LikelihoodDeltaDigestSeq L)
    {x c : StringAlg α}
    (hc : c ∈ NextLikelihoodFrontier B DL)
    (hcx : IsPrefix c x) :
    safe_Δ x (NextLikelihoodDigest B DL) = safe_Δ c (NextLikelihoodDigest B DL) := by
  rw [safe_Δ_eq_of_mem_prefix (NextLikelihoodDigest B DL) hc hcx]
  symm
  exact safe_Δ_eq_of_mem_prefix (NextLikelihoodDigest B DL) hc ⟨[], by simp⟩

noncomputable def pushLikelihood
    (B : LikelihoodTrackingTrie (α := α))
    (x : StringAlg α) : LikelihoodTrackingTrie (α := α) where
  N_lt := B.N_lt
  nodes y := by
    classical
    let node := B.nodes y
    exact
      if hy : y = x then
        { node with l := 1, n_lt := B.N_lt }
      else if hchild : IsChild x y then
        { node with l := node.l * (B.nodes x).l }
      else
        node
  val_eq y := by
    classical
    dsimp
    split_ifs <;> simp [B.val_eq y]

noncomputable def applyLikelihoodDelta
    {L : LikelihoodSeq (α := α)}
    (B : LikelihoodTrackingTrie (α := α))
    (DL : LikelihoodDeltaDigestSeq L) : LikelihoodTrackingTrie (α := α) where
  N_lt := B.N_lt + 1
  nodes x := by
    classical
    let node := B.nodes x
    exact
      if hx : x ∈ NextLikelihoodFrontier B DL then
        { node with l := node.l * safe_Δ x (NextLikelihoodDigest B DL) }
      else
        node
  val_eq x := by
    classical
    dsimp
    split_ifs <;> simp [B.val_eq x]

omit [Alphabet α] in
lemma pushLikelihood_n_lt_of_self
    (B : LikelihoodTrackingTrie (α := α))
    (x : StringAlg α) :
    ((pushLikelihood B x).nodes x).n_lt = B.N_lt := by
  classical
  simp [pushLikelihood]

omit [Alphabet α] in
lemma pushLikelihood_n_lt_of_ne
    (B : LikelihoodTrackingTrie (α := α))
    {x y : StringAlg α}
    (hy : y ≠ x) :
    ((pushLikelihood B x).nodes y).n_lt = (B.nodes y).n_lt := by
  classical
  by_cases hchild : IsChild x y
  · simp [pushLikelihood, hy, hchild]
  · simp [pushLikelihood, hy, hchild]

omit [Alphabet α] in
lemma pushLikelihood_l_of_self
    (B : LikelihoodTrackingTrie (α := α))
    (x : StringAlg α) :
    ((pushLikelihood B x).nodes x).l = 1 := by
  classical
  simp [pushLikelihood]

omit [Alphabet α] in
lemma pushLikelihood_l_of_child
    (B : LikelihoodTrackingTrie (α := α))
    {x y : StringAlg α}
    (hchild : IsChild x y) :
    ((pushLikelihood B x).nodes y).l = (B.nodes y).l * (B.nodes x).l := by
  classical
  have hy : y ≠ x := by
    intro h
    subst h
    exact not_isChild_of_prefix (x := y) (y := y) ⟨[], by simp⟩ hchild
  simp [pushLikelihood, hy, hchild]

omit [Alphabet α] in
lemma pushLikelihood_l_of_other
    (B : LikelihoodTrackingTrie (α := α))
    {x y : StringAlg α}
    (hy : y ≠ x)
    (hchild : ¬ IsChild x y) :
    ((pushLikelihood B x).nodes y).l = (B.nodes y).l := by
  classical
  simp [pushLikelihood, hy, hchild]

lemma applyLikelihoodDelta_n_lt
    {L : LikelihoodSeq (α := α)}
    (B : LikelihoodTrackingTrie (α := α))
    (DL : LikelihoodDeltaDigestSeq L)
    (x : StringAlg α) :
    ((applyLikelihoodDelta B DL).nodes x).n_lt = (B.nodes x).n_lt := by
  classical
  by_cases hx : x ∈ NextLikelihoodFrontier B DL
  · simp [applyLikelihoodDelta, hx]
  · simp [applyLikelihoodDelta, hx]

lemma applyLikelihoodDelta_l_of_frontier
    {L : LikelihoodSeq (α := α)}
    (B : LikelihoodTrackingTrie (α := α))
    (DL : LikelihoodDeltaDigestSeq L)
    {x : StringAlg α}
    (hx : x ∈ NextLikelihoodFrontier B DL) :
    ((applyLikelihoodDelta B DL).nodes x).l
      = (B.nodes x).l * safe_Δ x (NextLikelihoodDigest B DL) := by
  classical
  simp [applyLikelihoodDelta, hx]

lemma applyLikelihoodDelta_l_of_not_frontier
    {L : LikelihoodSeq (α := α)}
    (B : LikelihoodTrackingTrie (α := α))
    (DL : LikelihoodDeltaDigestSeq L)
    {x : StringAlg α}
    (hx : x ∉ NextLikelihoodFrontier B DL) :
    ((applyLikelihoodDelta B DL).nodes x).l = (B.nodes x).l := by
  classical
  simp [applyLikelihoodDelta, hx]

lemma nextLikelihoodFrontier_unique_prefix
    {L : LikelihoodSeq (α := α)}
    (B : LikelihoodTrackingTrie (α := α))
    (DL : LikelihoodDeltaDigestSeq L)
    {x c d : StringAlg α}
    (hc : c ∈ NextLikelihoodFrontier B DL)
    (hd : d ∈ NextLikelihoodFrontier B DL)
    (hcx : IsPrefix c x)
    (hdx : IsPrefix d x) :
    c = d := by
  exact prefix_code_unique_prefix (NextLikelihoodDigest B DL).is_prefix_code hc hcx hd hdx

omit [Alphabet α] in
lemma pushLikelihood_ancestorProduct_self
    (B : LikelihoodTrackingTrie (α := α))
    {x : StringAlg α}
    (h_anc : NoUnpushedAncestors B x) :
    AncestorLikelihoodProduct (pushLikelihood B x) x = 1 := by
  classical
  unfold AncestorLikelihoodProduct
  refine Finset.prod_eq_one ?_
  intro y hy
  by_cases hyx : y = x
  · rw [hyx]
    simpa using pushLikelihood_l_of_self B x
  · have hy_pref : IsPrefix y x := mem_Prefixes.mp hy
    have hy_strict : IsStrictPrefix y x := ⟨hy_pref, hyx⟩
    have hy_one : (B.nodes y).l = 1 := h_anc y hy_strict
    have hy_not_child : ¬ IsChild x y := not_isChild_of_prefix hy_pref
    rw [pushLikelihood_l_of_other B hyx hy_not_child, hy_one]

omit [Alphabet α] in
lemma pushLikelihood_ancestorProduct_of_not_prefix
    (B : LikelihoodTrackingTrie (α := α))
    {x z : StringAlg α}
    (hxz : ¬ IsPrefix x z) :
    AncestorLikelihoodProduct (pushLikelihood B x) z = AncestorLikelihoodProduct B z := by
  classical
  unfold AncestorLikelihoodProduct
  apply Finset.prod_congr rfl
  intro y hy
  have hyz : IsPrefix y z := mem_Prefixes.mp hy
  have hy_ne_x : y ≠ x := by
    intro hy_eq
    apply hxz
    simpa [hy_eq] using hyz
  have hy_not_child : ¬ IsChild x y := by
    intro hchild
    apply hxz
    exact List.IsPrefix.trans (child_has_prefix hchild) hyz
  exact pushLikelihood_l_of_other B hy_ne_x hy_not_child

omit [Alphabet α] in
lemma pushLikelihood_ancestorProduct_of_strictPrefix
    (B : LikelihoodTrackingTrie (α := α))
    {x z : StringAlg α}
    (hxz : IsStrictPrefix x z) :
    AncestorLikelihoodProduct (pushLikelihood B x) z = AncestorLikelihoodProduct B z := by
  classical
  obtain ⟨c, hc_child, hcz⟩ := exists_child_prefix_of_strictPrefix hxz
  let P : Finset (StringAlg α) := Prefixes z
  let R : Finset (StringAlg α) := (P.erase x).erase c
  have hx_mem : x ∈ P := mem_Prefixes.mpr hxz.1
  have hc_mem : c ∈ P := mem_Prefixes.mpr hcz
  have hc_ne_x : c ≠ x := by
    rcases hc_child with ⟨a, rfl⟩
    simp
  have hc_mem_erase : c ∈ P.erase x := Finset.mem_erase.mpr ⟨hc_ne_x, hc_mem⟩
  have h_rest :
      ∏ y ∈ R, ((pushLikelihood B x).nodes y).l
        = ∏ y ∈ R, (B.nodes y).l := by
    apply Finset.prod_congr rfl
    intro y hy
    have hy_ne_c : y ≠ c := (Finset.mem_erase.mp hy).1
    have hy_mem_erase_x : y ∈ P.erase x := (Finset.mem_erase.mp hy).2
    have hy_ne_x : y ≠ x := (Finset.mem_erase.mp hy_mem_erase_x).1
    have hy_mem_P : y ∈ P := (Finset.mem_erase.mp hy_mem_erase_x).2
    have hyz : IsPrefix y z := mem_Prefixes.mp hy_mem_P
    have hy_not_child : ¬ IsChild x y := by
      intro hy_child
      have hy_eq_c : y = c := child_prefix_unique hy_child hc_child hyz hcz
      exact hy_ne_c hy_eq_c
    exact pushLikelihood_l_of_other B hy_ne_x hy_not_child
  have h_after :
      AncestorLikelihoodProduct (pushLikelihood B x) z
        = ((pushLikelihood B x).nodes x).l *
            (((pushLikelihood B x).nodes c).l *
              ∏ y ∈ R, ((pushLikelihood B x).nodes y).l) := by
    unfold AncestorLikelihoodProduct
    change ∏ y ∈ P, ((pushLikelihood B x).nodes y).l =
      ((pushLikelihood B x).nodes x).l *
        (((pushLikelihood B x).nodes c).l * ∏ y ∈ R, ((pushLikelihood B x).nodes y).l)
    calc
      ∏ y ∈ P, ((pushLikelihood B x).nodes y).l
          = ∏ y ∈ insert x (P.erase x), ((pushLikelihood B x).nodes y).l := by
              rw [Finset.insert_erase hx_mem]
      _ = ((pushLikelihood B x).nodes x).l *
            ∏ y ∈ P.erase x, ((pushLikelihood B x).nodes y).l := by
              rw [Finset.prod_insert]
              simp
      _ = ((pushLikelihood B x).nodes x).l *
            (∏ y ∈ insert c ((P.erase x).erase c), ((pushLikelihood B x).nodes y).l) := by
              rw [Finset.insert_erase hc_mem_erase]
      _ = ((pushLikelihood B x).nodes x).l *
            (((pushLikelihood B x).nodes c).l *
              ∏ y ∈ (P.erase x).erase c, ((pushLikelihood B x).nodes y).l) := by
              rw [Finset.prod_insert]
              simp
  have h_before :
      AncestorLikelihoodProduct B z
        = (B.nodes x).l * ((B.nodes c).l * ∏ y ∈ R, (B.nodes y).l) := by
    unfold AncestorLikelihoodProduct
    change ∏ y ∈ P, (B.nodes y).l =
      (B.nodes x).l * ((B.nodes c).l * ∏ y ∈ R, (B.nodes y).l)
    calc
      ∏ y ∈ P, (B.nodes y).l
          = ∏ y ∈ insert x (P.erase x), (B.nodes y).l := by
              rw [Finset.insert_erase hx_mem]
      _ = (B.nodes x).l * ∏ y ∈ P.erase x, (B.nodes y).l := by
              rw [Finset.prod_insert]
              simp
      _ = (B.nodes x).l * (∏ y ∈ insert c ((P.erase x).erase c), (B.nodes y).l) := by
              rw [Finset.insert_erase hc_mem_erase]
      _ = (B.nodes x).l * ((B.nodes c).l * ∏ y ∈ (P.erase x).erase c, (B.nodes y).l) := by
              rw [Finset.prod_insert]
              simp
  calc
    AncestorLikelihoodProduct (pushLikelihood B x) z
        = ((pushLikelihood B x).nodes x).l *
            (((pushLikelihood B x).nodes c).l *
              ∏ y ∈ R, ((pushLikelihood B x).nodes y).l) := h_after
    _ = 1 * (((B.nodes c).l * (B.nodes x).l) * ∏ y ∈ R, (B.nodes y).l) := by
          rw [pushLikelihood_l_of_self B x, pushLikelihood_l_of_child B hc_child, h_rest]
    _ = (B.nodes x).l * ((B.nodes c).l * ∏ y ∈ R, (B.nodes y).l) := by
          simp [mul_assoc, mul_comm]
    _ = AncestorLikelihoodProduct B z := h_before.symm

omit [Alphabet α] in
theorem pushLikelihood_preserves_timesBounded
    (B : LikelihoodTrackingTrie (α := α))
    {x : StringAlg α}
    (h_bound : LikelihoodTimesBounded B) :
    LikelihoodTimesBounded (pushLikelihood B x) := by
  intro y
  by_cases hy : y = x
  · subst hy
    simp [pushLikelihood]
  · rw [pushLikelihood_n_lt_of_ne B hy]
    exact h_bound y

theorem pushLikelihood_preserves_pushCorrect
    {L : LikelihoodSeq (α := α)}
    {DL : LikelihoodDeltaDigestSeq L}
    {B : LikelihoodTrackingTrie (α := α)}
    {x : StringAlg α}
    (h_push : PushCorrect DL B)
    (h_anc : NoUnpushedAncestors B x) :
    PushCorrect DL (pushLikelihood B x) := by
  intro z hz
  by_cases hz_eq : z = x
  · subst z
    have h_left : AncestorLikelihoodProduct (pushLikelihood B x) x = 1 :=
      pushLikelihood_ancestorProduct_self B h_anc
    have h_right :
        ∏ i ∈ Icc (((pushLikelihood B x).nodes x).n_lt + 1) (pushLikelihood B x).N_lt,
          safe_Δ x (DL.seq i) = 1 := by
      rw [pushLikelihood_n_lt_of_self B x]
      simp [pushLikelihood]
    rw [h_left, h_right]
  · have hn : ((pushLikelihood B x).nodes z).n_lt = (B.nodes z).n_lt :=
      pushLikelihood_n_lt_of_ne B hz_eq
    have h_old :
        IsBeyondFrontier z DL.seq ((B.nodes z).n_lt + 1) B.N_lt := by
      intro i hi_start hi_end
      exact hz i (by simpa [hn] using hi_start) (by simpa [pushLikelihood] using hi_end)
    by_cases hxz : IsPrefix x z
    · have h_prod :
          AncestorLikelihoodProduct (pushLikelihood B x) z = AncestorLikelihoodProduct B z :=
          pushLikelihood_ancestorProduct_of_strictPrefix B ⟨hxz, fun h => hz_eq h.symm⟩
      rw [h_prod, hn]
      simpa [pushLikelihood] using h_push z h_old
    · have h_prod :
          AncestorLikelihoodProduct (pushLikelihood B x) z = AncestorLikelihoodProduct B z :=
          pushLikelihood_ancestorProduct_of_not_prefix B hxz
      rw [h_prod, hn]
      simpa [pushLikelihood] using h_push z h_old

theorem applyLikelihoodDelta_preserves_timesBounded
    {L : LikelihoodSeq (α := α)}
    (B : LikelihoodTrackingTrie (α := α))
    (DL : LikelihoodDeltaDigestSeq L)
    (h_bound : LikelihoodTimesBounded B) :
    LikelihoodTimesBounded (applyLikelihoodDelta B DL) := by
  intro x
  rw [applyLikelihoodDelta_n_lt B DL x]
  exact Nat.le_trans (h_bound x) (Nat.le_succ _)

theorem applyLikelihoodDelta_preserves_pushCorrect
    {L : LikelihoodSeq (α := α)}
    {DL : LikelihoodDeltaDigestSeq L}
    {B : LikelihoodTrackingTrie (α := α)}
    (h_bound : LikelihoodTimesBounded B)
    (h_push : PushCorrect DL B) :
    PushCorrect DL (applyLikelihoodDelta B DL) := by
  intro z hz
  classical
  have hn : ((applyLikelihoodDelta B DL).nodes z).n_lt = (B.nodes z).n_lt :=
    applyLikelihoodDelta_n_lt B DL z
  have h_old :
      IsBeyondFrontier z DL.seq ((B.nodes z).n_lt + 1) B.N_lt := by
    intro i hi_start hi_end
    exact hz i (by simpa [hn] using hi_start)
      (Nat.le_trans hi_end (by simp [applyLikelihoodDelta]))
  have h_start_le : ((applyLikelihoodDelta B DL).nodes z).n_lt + 1 ≤ B.N_lt + 1 := by
    simpa [hn] using Nat.succ_le_succ (h_bound z)
  have h_ext_next : ExtensionOfCode z (NextLikelihoodFrontier B DL) := by
    simpa [NextLikelihoodFrontier, NextLikelihoodDigest, applyLikelihoodDelta, hn]
      using hz (B.N_lt + 1) h_start_le (by simp [applyLikelihoodDelta])
  let c : StringAlg α := Classical.choose h_ext_next
  have hc : c ∈ NextLikelihoodFrontier B DL := (Classical.choose_spec h_ext_next).1
  have hcz : IsPrefix c z := (Classical.choose_spec h_ext_next).2
  let P : Finset (StringAlg α) := Prefixes z
  let R : Finset (StringAlg α) := P.erase c
  have hc_mem : c ∈ P := mem_Prefixes.mpr hcz
  have h_rest :
      ∏ y ∈ R, ((applyLikelihoodDelta B DL).nodes y).l
        = ∏ y ∈ R, (B.nodes y).l := by
    apply Finset.prod_congr rfl
    intro y hy
    have hy_ne_c : y ≠ c := (Finset.mem_erase.mp hy).1
    have hy_mem_P : y ∈ P := (Finset.mem_erase.mp hy).2
    have hyz : IsPrefix y z := mem_Prefixes.mp hy_mem_P
    have hy_not_frontier : y ∉ NextLikelihoodFrontier B DL := by
      intro hy_frontier
      have hy_eq_c : y = c :=
        nextLikelihoodFrontier_unique_prefix B DL hy_frontier hc hyz hcz
      exact hy_ne_c hy_eq_c
    exact applyLikelihoodDelta_l_of_not_frontier B DL hy_not_frontier
  have h_after :
      AncestorLikelihoodProduct (applyLikelihoodDelta B DL) z
        = ((applyLikelihoodDelta B DL).nodes c).l *
            ∏ y ∈ R, ((applyLikelihoodDelta B DL).nodes y).l := by
    unfold AncestorLikelihoodProduct
    change ∏ y ∈ P, ((applyLikelihoodDelta B DL).nodes y).l =
      ((applyLikelihoodDelta B DL).nodes c).l * ∏ y ∈ R, ((applyLikelihoodDelta B DL).nodes y).l
    calc
      ∏ y ∈ P, ((applyLikelihoodDelta B DL).nodes y).l
          = ∏ y ∈ insert c (P.erase c), ((applyLikelihoodDelta B DL).nodes y).l := by
              rw [Finset.insert_erase hc_mem]
      _ = ((applyLikelihoodDelta B DL).nodes c).l *
            ∏ y ∈ P.erase c, ((applyLikelihoodDelta B DL).nodes y).l := by
              rw [Finset.prod_insert]
              simp
  have h_before :
      AncestorLikelihoodProduct B z
        = (B.nodes c).l * ∏ y ∈ R, (B.nodes y).l := by
    unfold AncestorLikelihoodProduct
    change ∏ y ∈ P, (B.nodes y).l = (B.nodes c).l * ∏ y ∈ R, (B.nodes y).l
    calc
      ∏ y ∈ P, (B.nodes y).l
          = ∏ y ∈ insert c (P.erase c), (B.nodes y).l := by
              rw [Finset.insert_erase hc_mem]
      _ = (B.nodes c).l * ∏ y ∈ P.erase c, (B.nodes y).l := by
              rw [Finset.prod_insert]
              simp
  calc
    AncestorLikelihoodProduct (applyLikelihoodDelta B DL) z
        = ((applyLikelihoodDelta B DL).nodes c).l *
            ∏ y ∈ R, ((applyLikelihoodDelta B DL).nodes y).l := h_after
    _ = ((B.nodes c).l * safe_Δ c (NextLikelihoodDigest B DL)) *
          ∏ y ∈ R, (B.nodes y).l := by
          rw [applyLikelihoodDelta_l_of_frontier B DL hc, h_rest]
    _ = ((B.nodes c).l * ∏ y ∈ R, (B.nodes y).l) *
          safe_Δ z (NextLikelihoodDigest B DL) := by
          have hdelta :
              safe_Δ c (NextLikelihoodDigest B DL) = safe_Δ z (NextLikelihoodDigest B DL) := by
            rw [safe_Δ_next_eq_of_prefix B DL hc hcz]
          rw [hdelta]
          simp [mul_assoc, mul_comm]
    _ = AncestorLikelihoodProduct B z * safe_Δ z (NextLikelihoodDigest B DL) := by
          rw [h_before]
    _ = (∏ i ∈ Icc ((B.nodes z).n_lt + 1) B.N_lt, safe_Δ z (DL.seq i)) *
          safe_Δ z (DL.seq (B.N_lt + 1)) := by
          rw [h_push z h_old]
          rfl
    _ = ∏ i ∈ Icc ((B.nodes z).n_lt + 1) (B.N_lt + 1), safe_Δ z (DL.seq i) := by
          symm
          exact Finset.prod_Icc_succ_top (Nat.succ_le_succ (h_bound z)) _
    _ = ∏ i ∈ Icc (((applyLikelihoodDelta B DL).nodes z).n_lt + 1)
          (applyLikelihoodDelta B DL).N_lt, safe_Δ z (DL.seq i) := by
          rw [applyLikelihoodDelta_n_lt B DL z]
          simp [applyLikelihoodDelta]

end Trie
