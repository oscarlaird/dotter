import Trie.Core
import Trie.TokenPrior
import Mathlib.Topology.Algebra.InfiniteSum.ENNReal

open scoped BigOperators
open Finset

namespace Trie

variable {α : Type} [Alphabet α] [DecidableEq α]

/--
The historical product of prior deltas is exactly a branch-prior factor.
- This is the multiplicative form used in the prior-tracking section.
-/
theorem priorDeltaFactorization
    {P : PriorSeq (α := α)} {DP : PriorDeltaDigestSeq P}
    {m m' : ℕ} (hmm' : m ≤ m') {x : StringAlg α}
    (hx : IsBeyondFrontier x DP.seq (m + 1) m') :
    BranchPrior (P m') x
      =
    BranchPrior (P m) x * ∏ i ∈ Icc (m + 1) m', safe_Δ x (DP.seq i) := by
  unfold BranchPrior
  have h_tsum_eq :
      (∑' h : H (α := α), if PrefixOfCode x {h.val} then (P m').fn h else 0)
        =
      ∑' h : H (α := α),
        if PrefixOfCode x {h.val}
        then (P m).fn h * ∏ i ∈ Icc (m + 1) m', safe_Δ x (DP.seq i)
        else 0 := by
    apply tsum_congr
    intro h
    by_cases hp : PrefixOfCode x {h.val}
    · rw [if_pos hp, if_pos hp]
      rcases hp with ⟨c, hc, hpref⟩
      have hc_eq : c = h.val := Set.mem_singleton_iff.mp hc
      subst hc_eq
      have h_start_le : 1 ≤ m + 1 := by omega
      have h_start_end : m + 1 ≤ m' + 1 := by omega
      have h_upd :=
        prior_update_safe_Δ (P := P) (D := DP)
          (start := m + 1) (N := m') (n := m')
          h_start_le h_start_end (by omega) hx h hpref
      simpa using h_upd
    · simp [hp]
  rw [h_tsum_eq]
  have h_fun_eq :
      (fun h : H (α := α) =>
        if PrefixOfCode x {h.val}
        then (P m).fn h * ∏ i ∈ Icc (m + 1) m', safe_Δ x (DP.seq i)
        else 0)
      =
      (fun h : H (α := α) =>
        (if PrefixOfCode x {h.val} then (P m).fn h else 0) *
          (∏ i ∈ Icc (m + 1) m', safe_Δ x (DP.seq i))) := by
    funext h
    by_cases hp : PrefixOfCode x {h.val} <;> simp [hp]
  rw [h_fun_eq, NNReal.tsum_mul_right]

/--
Division form of `priorDeltaFactorization`.
-/
theorem priorDeltaQuotient
    {P : PriorSeq (α := α)} {DP : PriorDeltaDigestSeq P}
    {m m' : ℕ} (hmm' : m ≤ m') {x : StringAlg α}
    (hx : IsBeyondFrontier x DP.seq (m + 1) m')
    (hpos : BranchPrior (P m) x ≠ 0) :
    ∏ i ∈ Icc (m + 1) m', safe_Δ x (DP.seq i)
      =
    BranchPrior (P m') x / BranchPrior (P m) x := by
  apply (eq_div_iff hpos).2
  simpa [mul_comm] using (priorDeltaFactorization hmm' hx).symm

/-- A tracked token context storing a token branch prior at a given time. -/
structure TrackedTokenContext where
  seq : TokenSeq α
  tokBranchPrior : NNReal
  m_tok : ℕ

/-- A tracked string node storing a branch prior at a given time. -/
structure TrackedStringNode where
  val : StringAlg α
  branchPrior : ENNReal
  m_prior : ℕ

/-- Correctness predicate for stored token branch priors. -/
def TokenContextCorrect
    (S : TokenPriorSetup α)
    (m : ℕ)
    (ctx : TrackedTokenContext (α := α)) : Prop :=
  ctx.tokBranchPrior = tokBranchPrior S.cond ctx.seq ∧
  ctx.m_tok = m

/-- Correctness predicate for stored string branch priors. -/
def StringNodeCorrect
    (S : TokenPriorSetup α)
    (m : ℕ)
    (node : TrackedStringNode (α := α)) : Prop :=
  node.branchPrior = ((BranchPrior S.derivedPrior node.val : NNReal) : ENNReal) ∧
  node.m_prior = m

/-- The root token context update. -/
def computeTokenBranchPriorRoot
    (m : ℕ) : TrackedTokenContext (α := α) where
  seq := []
  tokBranchPrior := 1
  m_tok := m

/--
One-step token-branch-prior update:
assuming the parent token context is already correct at time `m`,
extend it by one token.
-/
def computeTokenBranchPriorStep
    (S : TokenPriorSetup α)
    (parent : TrackedTokenContext (α := α))
    (t : TokenString α)
    (m : ℕ) : TrackedTokenContext (α := α) where
  seq := parent.seq ++ [t]
  tokBranchPrior := parent.tokBranchPrior * S.cond parent.seq t
  m_tok := m

theorem computeTokenBranchPriorRoot_correct
    (S : TokenPriorSetup α)
    (m : ℕ) :
    TokenContextCorrect S m (computeTokenBranchPriorRoot (α := α) m) := by
  constructor
  · rfl
  · rfl

theorem computeTokenBranchPriorStep_correct
    (S : TokenPriorSetup α)
    {m : ℕ}
    {parent : TrackedTokenContext (α := α)}
    (hparent : TokenContextCorrect S m parent)
    (t : TokenString α) :
    TokenContextCorrect S m (computeTokenBranchPriorStep S parent t m) := by
  rcases hparent with ⟨hprior, htime⟩
  constructor
  · simp [computeTokenBranchPriorStep, hprior, tokBranchPrior_append_one]
  · simp [computeTokenBranchPriorStep]

/-- The token fan from the prior-tracking section. -/
noncomputable def tokenFan
    (S : TokenPriorSetup α) (b : TokenSeq α) (x : StringAlg α) : ENNReal :=
  ∑' t : TokenString α,
    if List.IsPrefix x (literalizeTokenSeq (b ++ [t])) then ↑(S.cond b t) else 0

/--
Ideal branch-prior computation from a finite set of tracked token contexts:
for the root it is `1`, and otherwise it is the finite sum over relevant contexts.
-/
noncomputable def branchPriorFromContexts
    (S : TokenPriorSetup α)
    (contexts : Finset (TokenSeq α))
    (x : StringAlg α) : ENNReal :=
  if x = [] then 1
  else
    ∑ b ∈ contexts,
      if IsStrictPrefix (literalizeTokenSeq b) x
      then ↑(tokBranchPrior S.cond b) * tokenFan S b x
      else 0

/--
Algorithmic branch-prior computation using stored token-branch-prior values.
-/
noncomputable def computeBranchPrior
    (S : TokenPriorSetup α)
    (contexts : Finset (TokenSeq α))
    (ctxNode : TokenSeq α → TrackedTokenContext (α := α))
    (x : StringAlg α)
    (m : ℕ) : TrackedStringNode (α := α) where
  val := x
  branchPrior :=
    if x = [] then 1
    else
      ∑ b ∈ contexts,
        if IsStrictPrefix (literalizeTokenSeq b) x
        then ↑((ctxNode b).tokBranchPrior) * tokenFan S b x
        else 0
  m_prior := m

lemma cross_inner_eq_tokenFan
    (S : TokenPriorSetup α) (b : TokenSeq α) (x : StringAlg α) :
    (∑' t : TokenString α,
      if (IsStrictPrefix (literalizeTokenSeq b) x ∧
          List.IsPrefix x (literalizeTokenSeq (b ++ [t])))
      then ((tokBranchPrior S.cond b * S.cond b t : NNReal) : ENNReal)
      else 0)
      =
    if IsStrictPrefix (literalizeTokenSeq b) x
    then ↑(tokBranchPrior S.cond b) * tokenFan S b x
    else 0 := by
  by_cases hstrict : IsStrictPrefix (literalizeTokenSeq b) x
  · rw [if_pos hstrict]
    calc
      (∑' t : TokenString α,
        if (IsStrictPrefix (literalizeTokenSeq b) x ∧
            List.IsPrefix x (literalizeTokenSeq (b ++ [t])))
        then ((tokBranchPrior S.cond b * S.cond b t : NNReal) : ENNReal)
        else 0)
          =
      ∑' t : TokenString α,
        ((tokBranchPrior S.cond b : NNReal) : ENNReal) *
          (if List.IsPrefix x (literalizeTokenSeq (b ++ [t]))
           then ↑(S.cond b t)
           else 0) := by
            apply tsum_congr
            intro t
            simp [hstrict]
      _ = ((tokBranchPrior S.cond b : NNReal) : ENNReal) *
            ∑' t : TokenString α,
              (if List.IsPrefix x (literalizeTokenSeq (b ++ [t]))
               then ↑(S.cond b t)
               else 0) := by
                rw [ENNReal.tsum_mul_left]
      _ = ↑(tokBranchPrior S.cond b) * tokenFan S b x := by
            simp [tokenFan]
  · rw [if_neg hstrict]
    rw [ENNReal.tsum_eq_zero]
    intro t
    simp [hstrict]

/--
Correctness of the ideal finite dynamic program for branch priors.
-/
theorem branchPriorFromContexts_correct
    (S : TokenPriorSetup α)
    (frontier : Finset (StringAlg α))
    (contexts : Finset (TokenSeq α))
    (hcomplete :
      ∀ ⦃x : StringAlg α⦄, x ∈ frontier →
        ∀ ⦃b : TokenSeq α⦄, IsStrictPrefix (literalizeTokenSeq b) x → b ∈ contexts)
    {x : StringAlg α} (hx : x ∈ frontier) :
    branchPriorFromContexts S contexts x = ((BranchPrior S.derivedPrior x : NNReal) : ENNReal) := by
  by_cases hxnil : x = []
  · subst hxnil
    simp [branchPriorFromContexts, S.root_branch]
  · have hpath := S.token_prior_path_summation x
    simp [hxnil] at hpath
    calc
      branchPriorFromContexts S contexts x
          = ∑ b ∈ contexts,
              if IsStrictPrefix (literalizeTokenSeq b) x
              then ↑(tokBranchPrior S.cond b) * tokenFan S b x
              else 0 := by
                simp [branchPriorFromContexts, hxnil]
      _ = ∑' b : TokenSeq α,
            if IsStrictPrefix (literalizeTokenSeq b) x
            then ↑(tokBranchPrior S.cond b) * tokenFan S b x
            else 0 := by
              symm
              exact tsum_eq_sum (s := contexts) (fun b hb => by
                by_cases hstrict : IsStrictPrefix (literalizeTokenSeq b) x
                · have : b ∈ contexts := hcomplete hx hstrict
                  exact (hb this).elim
                · simp [hstrict])
      _ = ∑' b : TokenSeq α,
            ∑' t : TokenString α,
              if (IsStrictPrefix (literalizeTokenSeq b) x ∧
                  List.IsPrefix x (literalizeTokenSeq (b ++ [t])))
              then ((tokBranchPrior S.cond b * S.cond b t : NNReal) : ENNReal)
              else 0 := by
                apply tsum_congr
                intro b
                symm
                exact cross_inner_eq_tokenFan S b x
      _ = ((BranchPrior S.derivedPrior x : NNReal) : ENNReal) := by
            exact hpath.symm

/--
Correctness of the branch-prior algorithm:
if every relevant token context already stores the correct token branch prior at time `m`,
then the computed string value is the correct branch prior at time `m`.
-/
theorem computeBranchPrior_correct
    (S : TokenPriorSetup α)
    (frontier : Finset (StringAlg α))
    (contexts : Finset (TokenSeq α))
    (ctxNode : TokenSeq α → TrackedTokenContext (α := α))
    (m : ℕ)
    (hcomplete :
      ∀ ⦃x : StringAlg α⦄, x ∈ frontier →
        ∀ ⦃b : TokenSeq α⦄, IsStrictPrefix (literalizeTokenSeq b) x → b ∈ contexts)
    (hctx :
      ∀ ⦃b : TokenSeq α⦄, b ∈ contexts →
        (ctxNode b).seq = b ∧ TokenContextCorrect S m (ctxNode b))
    {x : StringAlg α} (hx : x ∈ frontier) :
    StringNodeCorrect S m (computeBranchPrior S contexts ctxNode x m) := by
  constructor
  · by_cases hxnil : x = []
    · simp [computeBranchPrior, hxnil, S.root_branch]
    · calc
        (computeBranchPrior S contexts ctxNode x m).branchPrior
            = ∑ b ∈ contexts,
                if IsStrictPrefix (literalizeTokenSeq b) x
                then ↑((ctxNode b).tokBranchPrior) * tokenFan S b x
                else 0 := by
                  simp [computeBranchPrior, hxnil]
        _ = ∑ b ∈ contexts,
              if IsStrictPrefix (literalizeTokenSeq b) x
              then ↑(tokBranchPrior S.cond b) * tokenFan S b x
              else 0 := by
                apply Finset.sum_congr rfl
                intro b hb
                rcases hctx hb with ⟨hseq, hcorr⟩
                rcases hcorr with ⟨hprior, htime⟩
                simp [hseq, hprior]
        _ = branchPriorFromContexts S contexts x := by
              simp [branchPriorFromContexts, hxnil]
        _ = ((BranchPrior S.derivedPrior x : NNReal) : ENNReal) := by
              exact branchPriorFromContexts_correct S frontier contexts hcomplete hx
  · simp [computeBranchPrior]

end Trie
