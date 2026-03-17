import Trie.Core

open Finset

variable {α : Type} [Alphabet α] [DecidableEq α]

/-- Bayesian trie node augmented with an unpushed likelihood factor `l`. -/
structure AugBayesianTrieNode where
  val : StringAlg α
  n : ℕ
  m : ℕ
  Z : NNReal
  l : NNReal

/-- Bayesian trie augmented with unpushed likelihood tracking. -/
structure AugBayesianTrie where
  N : ℕ
  M : ℕ
  nodes : StringAlg α → AugBayesianTrieNode (α := α)
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
lemma prefix_eq_of_length_eq {x y : StringAlg α} (h : IsPrefix x y)
  (hlen : x.length = y.length) : x = y := by
  rcases h with ⟨z, rfl⟩
  simp at hlen
  subst z
  simp

omit [Alphabet α] in
lemma Prefixes_subset {x y : StringAlg α} (h : IsPrefix x y) :
  Prefixes x ⊆ Prefixes y := by
  intro z hz
  exact mem_Prefixes.mpr (List.IsPrefix.trans (mem_Prefixes.mp hz) h)

/-- Product of unpushed likelihood factors over all ancestors of `x`, including `x`. -/
def AncestorLikelihoodProduct
  (B : AugBayesianTrie (α := α))
  (x : StringAlg α) : NNReal :=
  ∏ y ∈ Prefixes x, (B.nodes y).l

/-- Every node's local likelihood time is at most the global time. -/
def LikelihoodTimesBounded (B : AugBayesianTrie (α := α)) : Prop :=
  ∀ x, (B.nodes x).n ≤ B.N

/-- Every node is beyond the likelihood frontier from its local time to the global time. -/
def LikelihoodValidTrie
  {L : LikelihoodSeq (α := α)}
  (DL : LikelihoodDeltaDigestSeq L)
  (B : AugBayesianTrie (α := α)) : Prop :=
  ∀ x, IsBeyondFrontier x DL.seq ((B.nodes x).n + 1) B.N

/-- The likelihood digest used by the next update step. -/
def NextLikelihoodDigest
  {L : LikelihoodSeq (α := α)}
  (B : AugBayesianTrie (α := α))
  (DL : LikelihoodDeltaDigestSeq L) : DeltaDigest (α := α) :=
  DL.seq (B.N + 1)

/-- The likelihood frontier at the next time step. -/
def NextFrontier
  {L : LikelihoodSeq (α := α)}
  (B : AugBayesianTrie (α := α))
  (DL : LikelihoodDeltaDigestSeq L) : Set (StringAlg α) :=
  (NextLikelihoodDigest B DL).C

/-- `child` is an immediate child of `parent` in the trie. -/
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
  (B : AugBayesianTrie (α := α))
  (DL : LikelihoodDeltaDigestSeq L)
  {x c : StringAlg α}
  (hc : c ∈ NextFrontier B DL)
  (hcx : IsPrefix c x) :
  safe_Δ x (NextLikelihoodDigest B DL) = safe_Δ c (NextLikelihoodDigest B DL) := by
  rw [safe_Δ_eq_of_mem_prefix (NextLikelihoodDigest B DL) hc hcx]
  symm
  exact safe_Δ_eq_of_mem_prefix (NextLikelihoodDigest B DL) hc ⟨[], by simp⟩

/--
Push correctness (Definition in likelihood-tracking section):
if all likelihood changes below a node are a branch-constant factor, then that
factor is equal to the product of unpushed likelihoods across its ancestors.
-/
def PushCorrect
  {L : LikelihoodSeq (α := α)}
  (DL : LikelihoodDeltaDigestSeq L)
  (B : AugBayesianTrie (α := α)) : Prop :=
  ∀ x,
    IsBeyondFrontier x DL.seq ((B.nodes x).n + 1) B.N →
      AncestorLikelihoodProduct B x
        = ∏ i ∈ Finset.Icc ((B.nodes x).n + 1) B.N, safe_Δ x (DL.seq i)

/-- Nodes strictly below the next frontier. -/
def IsStrictPrefixOfNextFrontier
  {L : LikelihoodSeq (α := α)}
  (DL : LikelihoodDeltaDigestSeq L)
  (B : AugBayesianTrie (α := α))
  (x : StringAlg α) : Prop :=
  ∃ c, c ∈ NextFrontier B DL ∧ IsStrictPrefix x c

/-- Immediate children of the next frontier. -/
def IsChildOfNextFrontier
  {L : LikelihoodSeq (α := α)}
  (DL : LikelihoodDeltaDigestSeq L)
  (B : AugBayesianTrie (α := α))
  (x : StringAlg α) : Prop :=
  ∃ c, c ∈ NextFrontier B DL ∧ IsChild c x

lemma strictPrefixOfNextFrontier_not_mem
  {L : LikelihoodSeq (α := α)}
  (B : AugBayesianTrie (α := α))
  (DL : LikelihoodDeltaDigestSeq L)
  {x : StringAlg α}
  (hx : IsStrictPrefixOfNextFrontier DL B x) :
  x ∉ NextFrontier B DL := by
  intro hx_mem
  rcases hx with ⟨c, hc, hxc⟩
  exact (NextLikelihoodDigest B DL).is_prefix_code x c hx_mem hc hxc

lemma childOfNextFrontier_not_mem
  {L : LikelihoodSeq (α := α)}
  (B : AugBayesianTrie (α := α))
  (DL : LikelihoodDeltaDigestSeq L)
  {x : StringAlg α}
  (hx : IsChildOfNextFrontier DL B x) :
  x ∉ NextFrontier B DL := by
  intro hx_mem
  rcases hx with ⟨c, hc, hcx⟩
  rcases hcx with ⟨a, rfl⟩
  have hneq : c ≠ c ++ [a] := by
    intro h
    have hlen := congrArg List.length h
    simp at hlen
  exact (NextLikelihoodDigest B DL).is_prefix_code c (c ++ [a]) hc hx_mem
    ⟨child_has_prefix ⟨a, rfl⟩, hneq⟩

/-- The historical likelihood factor gathered at the new frontier node `x`. -/
noncomputable def FrontierLikelihoodFactor
  {L : LikelihoodSeq (α := α)}
  (B : AugBayesianTrie (α := α))
  (DL : LikelihoodDeltaDigestSeq L)
  (x : StringAlg α) : NNReal :=
  AncestorLikelihoodProduct B x * safe_Δ x (NextLikelihoodDigest B DL)

/-- First loop of the chapter: push old unpushed likelihoods down to the next frontier. -/
noncomputable def PushBelowFrontier
  {L : LikelihoodSeq (α := α)}
  (B : AugBayesianTrie (α := α))
  (DL : LikelihoodDeltaDigestSeq L) : AugBayesianTrie (α := α) where
  N := B.N
  M := B.M
  nodes x := by
    classical
    let node := B.nodes x
    exact
      if hx : x ∈ NextFrontier B DL then
        { node with l := AncestorLikelihoodProduct B x }
      else if hx : IsStrictPrefixOfNextFrontier DL B x then
        { node with l := 1 }
      else
        node
  val_eq x := by
    classical
    dsimp
    split_ifs <;> simp [B.val_eq x]

/-- State immediately before `AdvanceTrieTime`: frontier nodes hold the new delta factor. -/
noncomputable def CreateUnpushedLikelihood
  {L : LikelihoodSeq (α := α)}
  (B : AugBayesianTrie (α := α))
  (DL : LikelihoodDeltaDigestSeq L) : AugBayesianTrie (α := α) where
  N := B.N
  M := B.M
  nodes x := by
    classical
    let node := B.nodes x
    exact
      if hx : x ∈ NextFrontier B DL then
        { node with l := FrontierLikelihoodFactor B DL x }
      else if hx : IsStrictPrefixOfNextFrontier DL B x then
        { node with l := 1 }
      else
        node
  val_eq x := by
    classical
    dsimp
    split_ifs <;> simp [B.val_eq x]

/--
Final state after the chapter's single likelihood update:
frontier nodes are reset to `1`, and their accumulated factor is pushed one step to children.
-/
noncomputable def SingleLikelihoodUpdate
  {L : LikelihoodSeq (α := α)}
  (B : AugBayesianTrie (α := α))
  (DL : LikelihoodDeltaDigestSeq L) : AugBayesianTrie (α := α) where
  N := B.N + 1
  M := B.M
  nodes x := by
    classical
    let node := B.nodes x
    exact
      if hx : x ∈ NextFrontier B DL then
        { node with n := B.N + 1, l := 1 }
      else if hx : IsChildOfNextFrontier DL B x then
        let c := Classical.choose hx
        { node with l := node.l * FrontierLikelihoodFactor B DL c }
      else if hx : IsStrictPrefixOfNextFrontier DL B x then
        { node with l := 1 }
      else
        node
  val_eq x := by
    classical
    dsimp
    split_ifs <;> simp [B.val_eq x]

lemma pushBelowFrontier_l_of_frontier
  {L : LikelihoodSeq (α := α)}
  (B : AugBayesianTrie (α := α))
  (DL : LikelihoodDeltaDigestSeq L)
  {x : StringAlg α}
  (hx : x ∈ NextFrontier B DL) :
  ((PushBelowFrontier B DL).nodes x).l = AncestorLikelihoodProduct B x := by
  classical
  simp [PushBelowFrontier, hx]

lemma createUnpushedLikelihood_l_of_frontier
  {L : LikelihoodSeq (α := α)}
  (B : AugBayesianTrie (α := α))
  (DL : LikelihoodDeltaDigestSeq L)
  {x : StringAlg α}
  (hx : x ∈ NextFrontier B DL) :
  ((CreateUnpushedLikelihood B DL).nodes x).l = FrontierLikelihoodFactor B DL x := by
  classical
  simp [CreateUnpushedLikelihood, hx]

lemma singleLikelihoodUpdate_l_of_frontier
  {L : LikelihoodSeq (α := α)}
  (B : AugBayesianTrie (α := α))
  (DL : LikelihoodDeltaDigestSeq L)
  {x : StringAlg α}
  (hx : x ∈ NextFrontier B DL) :
  ((SingleLikelihoodUpdate B DL).nodes x).l = 1 := by
  classical
  simp [SingleLikelihoodUpdate, hx]

lemma singleLikelihoodUpdate_n_of_frontier
  {L : LikelihoodSeq (α := α)}
  (B : AugBayesianTrie (α := α))
  (DL : LikelihoodDeltaDigestSeq L)
  {x : StringAlg α}
  (hx : x ∈ NextFrontier B DL) :
  ((SingleLikelihoodUpdate B DL).nodes x).n = B.N + 1 := by
  classical
  simp [SingleLikelihoodUpdate, hx]

lemma singleLikelihoodUpdate_n_of_not_frontier
  {L : LikelihoodSeq (α := α)}
  (B : AugBayesianTrie (α := α))
  (DL : LikelihoodDeltaDigestSeq L)
  {x : StringAlg α}
  (hx : x ∉ NextFrontier B DL) :
  ((SingleLikelihoodUpdate B DL).nodes x).n = (B.nodes x).n := by
  classical
  by_cases hxChild : IsChildOfNextFrontier DL B x
  · simp [SingleLikelihoodUpdate, hx, hxChild]
  · by_cases hxPref : IsStrictPrefixOfNextFrontier DL B x
    · simp [SingleLikelihoodUpdate, hx, hxChild, hxPref]
    · simp [SingleLikelihoodUpdate, hx, hxChild, hxPref]

lemma singleLikelihoodUpdate_l_of_strictPrefix
  {L : LikelihoodSeq (α := α)}
  (B : AugBayesianTrie (α := α))
  (DL : LikelihoodDeltaDigestSeq L)
  {x : StringAlg α}
  (hx : IsStrictPrefixOfNextFrontier DL B x) :
  ((SingleLikelihoodUpdate B DL).nodes x).l = 1 := by
  classical
  have hx_not_child : ¬ IsChildOfNextFrontier DL B x := by
    intro hx_child
    rcases hx with ⟨c, hc, hxc⟩
    rcases hx_child with ⟨d, hd, hdx⟩
    have hd_eq : d = c := prefix_code_unique_prefix (NextLikelihoodDigest B DL).is_prefix_code
      hd (List.IsPrefix.trans (child_has_prefix hdx) hxc.1) hc ⟨[], by simp⟩
    have hlen_le : x.length ≤ c.length := prefix_length_le hxc.1
    have hlen_eq : x.length = c.length + 1 := by
      rw [← hd_eq]
      simpa using child_length hdx
    omega
  simp [SingleLikelihoodUpdate, hx, hx_not_child, strictPrefixOfNextFrontier_not_mem B DL hx]

lemma singleLikelihoodUpdate_l_of_child
  {L : LikelihoodSeq (α := α)}
  (B : AugBayesianTrie (α := α))
  (DL : LikelihoodDeltaDigestSeq L)
  {x : StringAlg α}
  (hx : IsChildOfNextFrontier DL B x) :
  ((SingleLikelihoodUpdate B DL).nodes x).l
    = (B.nodes x).l * FrontierLikelihoodFactor B DL (Classical.choose hx) := by
  classical
  simp [SingleLikelihoodUpdate, hx, childOfNextFrontier_not_mem B DL hx]

lemma singleLikelihoodUpdate_l_on_prefix_of_frontier
  {L : LikelihoodSeq (α := α)}
  (B : AugBayesianTrie (α := α))
  (DL : LikelihoodDeltaDigestSeq L)
  {x y : StringAlg α}
  (hx : x ∈ NextFrontier B DL)
  (hy : IsPrefix y x) :
  ((SingleLikelihoodUpdate B DL).nodes y).l = 1 := by
  by_cases h_eq : y = x
  · subst h_eq
    exact singleLikelihoodUpdate_l_of_frontier B DL hx
  · exact singleLikelihoodUpdate_l_of_strictPrefix B DL ⟨x, hx, ⟨hy, h_eq⟩⟩

lemma ancestorLikelihoodProduct_frontier_after_update
  {L : LikelihoodSeq (α := α)}
  (B : AugBayesianTrie (α := α))
  (DL : LikelihoodDeltaDigestSeq L)
  {x : StringAlg α}
  (hx : x ∈ NextFrontier B DL) :
  AncestorLikelihoodProduct (SingleLikelihoodUpdate B DL) x = 1 := by
  unfold AncestorLikelihoodProduct
  refine Finset.prod_eq_one ?_
  intro y hy
  exact singleLikelihoodUpdate_l_on_prefix_of_frontier B DL hx (mem_Prefixes.mp hy)

lemma nextFrontier_unique_prefix
  {L : LikelihoodSeq (α := α)}
  (B : AugBayesianTrie (α := α))
  (DL : LikelihoodDeltaDigestSeq L)
  {x c d : StringAlg α}
  (hc : c ∈ NextFrontier B DL)
  (hd : d ∈ NextFrontier B DL)
  (hcx : IsPrefix c x)
  (hdx : IsPrefix d x) :
  c = d := by
  exact prefix_code_unique_prefix (NextLikelihoodDigest B DL).is_prefix_code hc hcx hd hdx

lemma frontier_factor_eq_historical_product
  {L : LikelihoodSeq (α := α)}
  {B : AugBayesianTrie (α := α)}
  {DL : LikelihoodDeltaDigestSeq L}
  (h_time : LikelihoodTimesBounded B)
  (h_valid : LikelihoodValidTrie DL B)
  (h_push : PushCorrect DL B)
  {x : StringAlg α} :
  FrontierLikelihoodFactor B DL x
    = ∏ i ∈ Finset.Icc ((B.nodes x).n + 1) (B.N + 1), safe_Δ x (DL.seq i) := by
  have h_old := h_push x (h_valid x)
  have h_split :
      ∏ i ∈ Finset.Icc ((B.nodes x).n + 1) (B.N + 1), safe_Δ x (DL.seq i)
        =
      (∏ i ∈ Finset.Icc ((B.nodes x).n + 1) B.N, safe_Δ x (DL.seq i)) *
        safe_Δ x (DL.seq (B.N + 1)) := by
    exact Finset.prod_Icc_succ_top (Nat.succ_le_succ (h_time x)) _
  calc
    FrontierLikelihoodFactor B DL x
      = AncestorLikelihoodProduct B x * safe_Δ x (DL.seq (B.N + 1)) := by
          rfl
    _ = (∏ i ∈ Finset.Icc ((B.nodes x).n + 1) B.N, safe_Δ x (DL.seq i)) *
          safe_Δ x (DL.seq (B.N + 1)) := by
          rw [h_old]
    _ = ∏ i ∈ Finset.Icc ((B.nodes x).n + 1) (B.N + 1), safe_Δ x (DL.seq i) := by
          exact h_split.symm

theorem singleLikelihoodUpdate_pushCorrectness
  {L : LikelihoodSeq (α := α)}
  {B : AugBayesianTrie (α := α)}
  {DL : LikelihoodDeltaDigestSeq L}
  (h_time : LikelihoodTimesBounded B)
  (h_push : PushCorrect DL B) :
  PushCorrect DL (SingleLikelihoodUpdate B DL) := by
  intro x h_new
  by_cases hx : x ∈ NextFrontier B DL
  · have h_left : AncestorLikelihoodProduct (SingleLikelihoodUpdate B DL) x = 1 :=
      ancestorLikelihoodProduct_frontier_after_update B DL hx
    have h_right :
        ∏ i ∈ Finset.Icc (((SingleLikelihoodUpdate B DL).nodes x).n + 1)
            (SingleLikelihoodUpdate B DL).N, safe_Δ x (DL.seq i) = 1 := by
      rw [singleLikelihoodUpdate_n_of_frontier B DL hx]
      simp [SingleLikelihoodUpdate]
    rw [h_left, h_right]
  · have h_n_same : ((SingleLikelihoodUpdate B DL).nodes x).n = (B.nodes x).n :=
      singleLikelihoodUpdate_n_of_not_frontier B DL hx
    have h_old_beyond : IsBeyondFrontier x DL.seq ((B.nodes x).n + 1) B.N := by
      intro i hi_start hi_end
      have hi_end' : i ≤ (SingleLikelihoodUpdate B DL).N := by
        exact Nat.le_trans hi_end (by simp [SingleLikelihoodUpdate])
      have hi_start' : ((SingleLikelihoodUpdate B DL).nodes x).n + 1 ≤ i := by
        simpa [h_n_same] using hi_start
      exact h_new i hi_start' hi_end'
    have h_start_le : (B.nodes x).n + 1 ≤ B.N + 1 := Nat.succ_le_succ (h_time x)
    have h_ext_next : ExtensionOfCode x (NextFrontier B DL) := by
      have h_top : IsBeyondFrontier x DL.seq (((SingleLikelihoodUpdate B DL).nodes x).n + 1)
          (SingleLikelihoodUpdate B DL).N := h_new
      have h_start' : ((SingleLikelihoodUpdate B DL).nodes x).n + 1 ≤ B.N + 1 := by
        rw [h_n_same]
        exact h_start_le
      simpa [NextFrontier, NextLikelihoodDigest, SingleLikelihoodUpdate]
        using h_top (B.N + 1) h_start' (by simp [SingleLikelihoodUpdate])
    let c : StringAlg α := Classical.choose h_ext_next
    have hc : c ∈ NextFrontier B DL := (Classical.choose_spec h_ext_next).1
    have hcx : IsPrefix c x := (Classical.choose_spec h_ext_next).2
    have hcx_ne : c ≠ x := by
      intro h_eq
      apply hx
      simpa [h_eq] using hc
    have hcx_strict : IsStrictPrefix c x := ⟨hcx, hcx_ne⟩
    obtain ⟨y, hy_child, hyx⟩ := exists_child_prefix_of_strictPrefix hcx_strict
    let s : Finset (StringAlg α) := Prefixes x \ Prefixes c
    have hy_mem_x : y ∈ Prefixes x := mem_Prefixes.mpr hyx
    have hy_not_mem_c : y ∉ Prefixes c := by
      intro hy_mem_c
      have hyc : IsPrefix y c := mem_Prefixes.mp hy_mem_c
      have hlen_le := prefix_length_le hyc
      rw [child_length hy_child] at hlen_le
      omega
    have hy_mem_s : y ∈ s := by
      exact Finset.mem_sdiff.mpr ⟨hy_mem_x, hy_not_mem_c⟩
    have hsubset : Prefixes c ⊆ Prefixes x := Prefixes_subset hcx
    have hdisj : Disjoint (Prefixes c) s := by
      refine Finset.disjoint_left.mpr ?_
      intro z hz1 hz2
      exact (Finset.mem_sdiff.mp hz2).2 hz1
    have h_union : Prefixes c ∪ s = Prefixes x := by
      exact Finset.union_sdiff_of_subset hsubset
    have h_old_split :
        AncestorLikelihoodProduct B x
          = AncestorLikelihoodProduct B c * (∏ z ∈ s, (B.nodes z).l) := by
      calc
        AncestorLikelihoodProduct B x
            = (∏ z ∈ Prefixes x, (B.nodes z).l) := by rfl
        _ = (∏ z ∈ Prefixes c ∪ s, (B.nodes z).l) := by
              rw [h_union]
        _ = (∏ z ∈ Prefixes c, (B.nodes z).l) * (∏ z ∈ s, (B.nodes z).l) := by
              rw [Finset.prod_union hdisj]
        _ = AncestorLikelihoodProduct B c * (∏ z ∈ s, (B.nodes z).l) := by
              unfold AncestorLikelihoodProduct
              ac_rfl
    have h_new_split :
        AncestorLikelihoodProduct (SingleLikelihoodUpdate B DL) x
          = AncestorLikelihoodProduct (SingleLikelihoodUpdate B DL) c *
              (∏ z ∈ s, ((SingleLikelihoodUpdate B DL).nodes z).l) := by
      calc
        AncestorLikelihoodProduct (SingleLikelihoodUpdate B DL) x
            = (∏ z ∈ Prefixes x, ((SingleLikelihoodUpdate B DL).nodes z).l) := by rfl
        _ = (∏ z ∈ Prefixes c ∪ s, ((SingleLikelihoodUpdate B DL).nodes z).l) := by
              rw [h_union]
        _ = (∏ z ∈ Prefixes c, ((SingleLikelihoodUpdate B DL).nodes z).l) *
              (∏ z ∈ s, ((SingleLikelihoodUpdate B DL).nodes z).l) := by
              rw [Finset.prod_union hdisj]
        _ = AncestorLikelihoodProduct (SingleLikelihoodUpdate B DL) c *
              (∏ z ∈ s, ((SingleLikelihoodUpdate B DL).nodes z).l) := by
              unfold AncestorLikelihoodProduct
              ac_rfl
    have hy_frontier_child : IsChildOfNextFrontier DL B y := ⟨c, hc, hy_child⟩
    have h_erase_eq :
        ∀ z ∈ s.erase y,
          ((SingleLikelihoodUpdate B DL).nodes z).l = (B.nodes z).l := by
      intro z hz
      have hz_mem_s : z ∈ s := Finset.mem_of_mem_erase hz
      have hzx : IsPrefix z x := mem_Prefixes.mp ((Finset.mem_sdiff.mp hz_mem_s).1)
      have hz_not_mem_c : z ∉ Prefixes c := (Finset.mem_sdiff.mp hz_mem_s).2
      have hz_ne_y : z ≠ y := by
        exact Finset.ne_of_mem_erase hz
      have hz_not_frontier : z ∉ NextFrontier B DL := by
        intro hz_frontier
        have hzc : z = c := nextFrontier_unique_prefix (B := B) (DL := DL)
          (x := x) (c := z) (d := c) hz_frontier hc hzx hcx
        exact hz_not_mem_c (hzc ▸ self_mem_Prefixes c)
      have hz_not_child : ¬ IsChildOfNextFrontier DL B z := by
        intro hz_child
        rcases hz_child with ⟨d, hd, hdz⟩
        have hdx : IsPrefix d x := List.IsPrefix.trans (child_has_prefix hdz) hzx
        have hd_eq : d = c := nextFrontier_unique_prefix (B := B) (DL := DL)
          (x := x) (c := d) (d := c) hd hc hdx hcx
        subst hd_eq
        have hz_eq_y : z = y := child_prefix_unique hdz hy_child hzx hyx
        exact hz_ne_y hz_eq_y
      have hz_not_strict :
          ¬ IsStrictPrefixOfNextFrontier DL B z := by
        intro hz_strict
        rcases hz_strict with ⟨d, hd, hzd⟩
        have hcz : IsPrefix c z := by
          rcases prefix_or_prefix_of_prefix hcx hzx with h1 | h2
          · exact h1
          · exfalso
            exact hz_not_mem_c (mem_Prefixes.mpr h2)
        have hcd : IsPrefix c d := List.IsPrefix.trans hcz hzd.1
        have hd_eq : d = c := nextFrontier_unique_prefix (B := B) (DL := DL)
          (x := d) (c := d) (d := c) hd hc ⟨[], by simp⟩ hcd
        subst hd_eq
        exact hz_not_mem_c (mem_Prefixes.mpr hzd.1)
      classical
      simp [SingleLikelihoodUpdate, hz_not_frontier, hz_not_child, hz_not_strict]
    have h_insert : insert y (s.erase y) = s := Finset.insert_erase hy_mem_s
    have h_old_s :
        ∏ z ∈ s, (B.nodes z).l
          = (B.nodes y).l * (∏ z ∈ s.erase y, (B.nodes z).l) := by
      calc
        (∏ z ∈ s, (B.nodes z).l)
            = (∏ z ∈ insert y (s.erase y), (B.nodes z).l) := by
                rw [h_insert]
        _ = (B.nodes y).l * (∏ z ∈ s.erase y, (B.nodes z).l) := by
              rw [Finset.prod_insert]
              simp
    have h_new_s :
        ∏ z ∈ s, ((SingleLikelihoodUpdate B DL).nodes z).l
          = FrontierLikelihoodFactor B DL c * (∏ z ∈ s, (B.nodes z).l) := by
      have hchoose : Classical.choose hy_frontier_child = c := by
        exact prefix_code_unique_prefix (NextLikelihoodDigest B DL).is_prefix_code
          (Classical.choose_spec hy_frontier_child).1
          (child_has_prefix (Classical.choose_spec hy_frontier_child).2)
          hc
          (child_has_prefix hy_child)
      calc
        (∏ z ∈ s, ((SingleLikelihoodUpdate B DL).nodes z).l)
            = (∏ z ∈ insert y (s.erase y), ((SingleLikelihoodUpdate B DL).nodes z).l) := by
                rw [h_insert]
        _ = ((SingleLikelihoodUpdate B DL).nodes y).l *
              (∏ z ∈ s.erase y, ((SingleLikelihoodUpdate B DL).nodes z).l) := by
                rw [Finset.prod_insert]
                simp
        _ = ((B.nodes y).l * FrontierLikelihoodFactor B DL c) *
              (∏ z ∈ s.erase y, (B.nodes z).l) := by
                rw [singleLikelihoodUpdate_l_of_child B DL hy_frontier_child]
                rw [hchoose]
                have hprod_eq :
                    (∏ z ∈ s.erase y, ((SingleLikelihoodUpdate B DL).nodes z).l)
                      = (∏ z ∈ s.erase y, (B.nodes z).l) := by
                  apply Finset.prod_congr rfl
                  intro z hz
                  exact h_erase_eq z hz
                rw [hprod_eq]
        _ = FrontierLikelihoodFactor B DL c *
              ((B.nodes y).l * (∏ z ∈ s.erase y, (B.nodes z).l)) := by
                ac_rfl
        _ = FrontierLikelihoodFactor B DL c * (∏ z ∈ s, (B.nodes z).l) := by
              rw [h_old_s]
    have h_frontier_prod :
        AncestorLikelihoodProduct (SingleLikelihoodUpdate B DL) x
          = AncestorLikelihoodProduct B x * safe_Δ x (DL.seq (B.N + 1)) := by
      calc
        AncestorLikelihoodProduct (SingleLikelihoodUpdate B DL) x
            = AncestorLikelihoodProduct (SingleLikelihoodUpdate B DL) c *
                (∏ z ∈ s, ((SingleLikelihoodUpdate B DL).nodes z).l) := h_new_split
        _ = 1 * (∏ z ∈ s, ((SingleLikelihoodUpdate B DL).nodes z).l) := by
              rw [ancestorLikelihoodProduct_frontier_after_update B DL hc]
        _ = ∏ z ∈ s, ((SingleLikelihoodUpdate B DL).nodes z).l := by simp
        _ = FrontierLikelihoodFactor B DL c * (∏ z ∈ s, (B.nodes z).l) := h_new_s
        _ = (AncestorLikelihoodProduct B c * safe_Δ c (DL.seq (B.N + 1))) *
              (∏ z ∈ s, (B.nodes z).l) := by
                rfl
        _ = (AncestorLikelihoodProduct B c * (∏ z ∈ s, (B.nodes z).l)) *
              safe_Δ x (DL.seq (B.N + 1)) := by
                have hdelta : safe_Δ c (DL.seq (B.N + 1)) = safe_Δ x (DL.seq (B.N + 1)) := by
                  simpa [NextLikelihoodDigest] using (safe_Δ_next_eq_of_prefix B DL hc hcx).symm
                rw [hdelta]
                ac_rfl
        _ = AncestorLikelihoodProduct B x * safe_Δ x (DL.seq (B.N + 1)) := by
              rw [h_old_split]
    have h_old_prod := h_push x h_old_beyond
    have h_hist_split :
        AncestorLikelihoodProduct B x * safe_Δ x (DL.seq (B.N + 1))
          = ∏ i ∈ Finset.Icc ((B.nodes x).n + 1) (B.N + 1), safe_Δ x (DL.seq i) := by
      rw [h_old_prod]
      symm
      exact Finset.prod_Icc_succ_top (Nat.succ_le_succ (h_time x)) _
    calc
      AncestorLikelihoodProduct (SingleLikelihoodUpdate B DL) x
          = AncestorLikelihoodProduct B x * safe_Δ x (DL.seq (B.N + 1)) := h_frontier_prod
      _ = ∏ i ∈ Finset.Icc ((B.nodes x).n + 1) (B.N + 1), safe_Δ x (DL.seq i) := h_hist_split
      _ = ∏ i ∈ Finset.Icc (((SingleLikelihoodUpdate B DL).nodes x).n + 1)
            (SingleLikelihoodUpdate B DL).N, safe_Δ x (DL.seq i) := by
              rw [singleLikelihoodUpdate_n_of_not_frontier B DL hx]
              simp [SingleLikelihoodUpdate]

lemma equivalence_unpushed_likelihood
  {L : LikelihoodSeq (α := α)}
  {B : AugBayesianTrie (α := α)}
  {DL : LikelihoodDeltaDigestSeq L}
  (h_time : LikelihoodTimesBounded B)
  (h_valid : LikelihoodValidTrie DL B)
  (h_push : PushCorrect DL B)
  {x : StringAlg α}
  (hx : x ∈ NextFrontier B DL) :
  ((CreateUnpushedLikelihood B DL).nodes x).l
    = ∏ i ∈ Finset.Icc ((B.nodes x).n + 1) (B.N + 1), safe_Δ x (DL.seq i) := by
  rw [createUnpushedLikelihood_l_of_frontier B DL hx]
  exact frontier_factor_eq_historical_product h_time h_valid h_push

/--
Corollary used for efficient `AdvanceNodeTime` implementation:
for frontier nodes, replacing the historical product of likelihood deltas by
one multiplication with `x.l` is mathematically exact.
-/
theorem efficient_advanceNodeTime_likelihood_factor
  {L : LikelihoodSeq (α := α)}
  {B : AugBayesianTrie (α := α)}
  {DL : LikelihoodDeltaDigestSeq L}
  (h_time : LikelihoodTimesBounded B)
  (h_valid : LikelihoodValidTrie DL B)
  (h_push : PushCorrect DL B)
  {x : StringAlg α}
  (hx : x ∈ NextFrontier B DL) :
  (B.nodes x).Z * ((CreateUnpushedLikelihood B DL).nodes x).l
    = (B.nodes x).Z *
      (∏ i ∈ Finset.Icc ((B.nodes x).n + 1) (B.N + 1), safe_Δ x (DL.seq i)) := by
  rw [equivalence_unpushed_likelihood h_time h_valid h_push hx]
