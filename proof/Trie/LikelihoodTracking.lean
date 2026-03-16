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

/-- Product of unpushed likelihood factors along an abstract ancestor set. -/
def AncestorLikelihoodProduct
  (B : AugBayesianTrie (α := α))
  (Anc : StringAlg α → Finset (StringAlg α))
  (x : StringAlg α) : NNReal :=
  ∏ y ∈ Anc x, (B.nodes y).l

/--
Push correctness (Definition in likelihood-tracking section):
if all likelihood changes below a node are a branch-constant factor, then that
factor is equal to the product of unpushed likelihoods across its ancestors.
-/
def PushCorrect
  {L : LikelihoodSeq (α := α)}
  (DL : LikelihoodDeltaDigestSeq L)
  (B : AugBayesianTrie (α := α))
  (Anc : StringAlg α → Finset (StringAlg α)) : Prop :=
  ∀ x,
    IsBeyondFrontier x DL.seq ((B.nodes x).n + 1) B.N →
      AncestorLikelihoodProduct B Anc x
        = ∏ i ∈ Finset.Icc ((B.nodes x).n + 1) B.N, safe_Δ x (DL.seq i)

/--
Abstract contract for one likelihood step. This mirrors the math chapter:
we advance from time `N` to `N+1`, preserve prior time, and maintain push correctness.
-/
structure SingleLikelihoodUpdateSpec
  {L : LikelihoodSeq (α := α)}
  (DL : LikelihoodDeltaDigestSeq L)
  (oldB newB : AugBayesianTrie (α := α))
  (Anc : StringAlg α → Finset (StringAlg α))
  (Cnext : Set (StringAlg α)) : Prop where
  next_time : newB.N = oldB.N + 1
  same_prior_time : newB.M = oldB.M
  push_correct_next : PushCorrect DL newB Anc
  frontier_factor :
    ∀ x, x ∈ Cnext →
      (newB.nodes x).l
        = ∏ i ∈ Finset.Icc ((oldB.nodes x).n + 1) (oldB.N + 1), safe_Δ x (DL.seq i)

theorem singleLikelihoodUpdate_pushCorrectness
  {L : LikelihoodSeq (α := α)}
  {DL : LikelihoodDeltaDigestSeq L}
  {oldB newB : AugBayesianTrie (α := α)}
  {Anc : StringAlg α → Finset (StringAlg α)}
  {Cnext : Set (StringAlg α)}
  (h_step : SingleLikelihoodUpdateSpec DL oldB newB Anc Cnext) :
  PushCorrect DL newB Anc :=
  h_step.push_correct_next

lemma equivalence_unpushed_likelihood
  {L : LikelihoodSeq (α := α)}
  {DL : LikelihoodDeltaDigestSeq L}
  {oldB newB : AugBayesianTrie (α := α)}
  {Anc : StringAlg α → Finset (StringAlg α)}
  {Cnext : Set (StringAlg α)}
  (h_step : SingleLikelihoodUpdateSpec DL oldB newB Anc Cnext)
  {x : StringAlg α}
  (hx : x ∈ Cnext) :
  (newB.nodes x).l
    = ∏ i ∈ Finset.Icc ((oldB.nodes x).n + 1) (oldB.N + 1), safe_Δ x (DL.seq i) :=
  h_step.frontier_factor x hx

/--
Corollary used for efficient `AdvanceNodeTime` implementation:
for frontier nodes, replacing the historical product of likelihood deltas by
one multiplication with `x.l` is mathematically exact.
-/
theorem efficient_advanceNodeTime_likelihood_factor
  {L : LikelihoodSeq (α := α)}
  {DL : LikelihoodDeltaDigestSeq L}
  {oldB newB : AugBayesianTrie (α := α)}
  {Anc : StringAlg α → Finset (StringAlg α)}
  {Cnext : Set (StringAlg α)}
  (h_step : SingleLikelihoodUpdateSpec DL oldB newB Anc Cnext)
  {x : StringAlg α}
  (hx : x ∈ Cnext) :
  (oldB.nodes x).Z * (newB.nodes x).l
    = (oldB.nodes x).Z *
      (∏ i ∈ Finset.Icc ((oldB.nodes x).n + 1) (oldB.N + 1), safe_Δ x (DL.seq i)) := by
  rw [equivalence_unpushed_likelihood h_step hx]
