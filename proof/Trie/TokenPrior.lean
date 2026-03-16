import Trie.Core
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Topology.Instances.NNReal.Lemmas
import Mathlib.Topology.Algebra.InfiniteSum.Ring
import Mathlib.Topology.Algebra.InfiniteSum.ENNReal

open scoped BigOperators

namespace Trie

-- ═══════════════════════════════════════════════════════════════════
-- SECTION 1 : Token types and literalization
-- ═══════════════════════════════════════════════════════════════════

abbrev TokenString (α : Type) := StringAlg α
abbrev TokenSeq    (α : Type) := List (TokenString α)

def literalizeTokenSeq {α : Type} (s : TokenSeq α) : StringAlg α :=
  List.flatten s

/-- Every token in the sequence is non-empty. -/
def TokenSeqNonempty {α : Type} (s : TokenSeq α) : Prop :=
  ∀ i (hi : i < s.length), s.get ⟨i, hi⟩ ≠ []

-- ═══════════════════════════════════════════════════════════════════
-- SECTION 2 : Token-to-character index and monotonicity
-- ═══════════════════════════════════════════════════════════════════

def tokenToCharacterIndex {α : Type} : TokenSeq α → ℕ → ℕ
  | [],     _ => 0
  | _ :: _, 0 => 0
  | t :: ts, i + 1 => t.length + tokenToCharacterIndex ts i

lemma tokenToCharacterIndex_strict_step
    {α : Type} {s : TokenSeq α} (h_nonempty : TokenSeqNonempty s)
    {i : ℕ} (hi : i < s.length) :
    tokenToCharacterIndex s i < tokenToCharacterIndex s (i + 1) := by
  induction s generalizing i with
  | nil => cases hi
  | cons t ts ih =>
    cases i with
    | zero =>
      simp [tokenToCharacterIndex]
      have ht_ne : t ≠ [] := h_nonempty 0 (by simp)
      cases t with | nil => contradiction | cons _ _ => simp
    | succ i =>
      have hi' : i < ts.length := by simpa using hi
      have h_tail : TokenSeqNonempty ts := by
        intro j hj
        have hj' : j + 1 < (t :: ts).length := by simp [hj]
        exact h_nonempty (j + 1) hj'
      simpa [tokenToCharacterIndex, Nat.add_lt_add_iff_left] using ih h_tail hi'

lemma tokenToCharacterIndex_strict_mono
    {α : Type} {s : TokenSeq α} (h_nonempty : TokenSeqNonempty s)
    {i j : ℕ} (hij : i < j) (hj : j < s.length) :
    tokenToCharacterIndex s i < tokenToCharacterIndex s j := by
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_lt hij
  suffices ∀ k, i + k + 1 < s.length →
      tokenToCharacterIndex s i < tokenToCharacterIndex s (i + k + 1) from
    this k hj
  intro k; induction k with
  | zero => intro hk; simpa using tokenToCharacterIndex_strict_step h_nonempty (by omega)
  | succ k ih =>
    intro hk
    exact lt_trans (ih (by omega))
      (tokenToCharacterIndex_strict_step h_nonempty (by omega))

-- ═══════════════════════════════════════════════════════════════════
-- SECTION 3 : Char-to-token index, break points, boundary sufficiency
-- ═══════════════════════════════════════════════════════════════════

def charToTokenIndex {α : Type} (s : TokenSeq α) (j : ℕ) : ℕ :=
  Nat.findGreatest (fun i => tokenToCharacterIndex s i ≤ j) s.length

def BreakPoint {α : Type} [DecidableEq α]
    (T : StringAlg α → TokenSeq α) (x : StringAlg α) (i : ℕ) : Prop :=
  ∃ j, tokenToCharacterIndex (T x) j = i

instance instDecidableIsStrictPrefix {α : Type} [DecidableEq α]
    (s t : StringAlg α) : Decidable (IsStrictPrefix s t) :=
  inferInstanceAs (Decidable (_ ∧ _))

def BoundarySufficiency {α : Type} [DecidableEq α]
    (T : StringAlg α → TokenSeq α) : Prop :=
  ∀ x i₁ i₂,
    BreakPoint T x i₁ → BreakPoint T x i₂ → i₁ ≤ i₂ →
    let j₁ := charToTokenIndex (T x) i₁
    let j₂ := charToTokenIndex (T x) i₂
    T ((x.drop i₁).take (i₂ - i₁)) = ((T x).drop j₁).take (j₂ - j₁)

-- ═══════════════════════════════════════════════════════════════════
-- SECTION 4 : Token branch prior  (DEFINED, not assumed)
-- ═══════════════════════════════════════════════════════════════════

/-- Auxiliary with context accumulator. -/
def tokBranchPriorAux {α : Type}
    (cond : List (List α) → List α → NNReal)
    (ctx : List (List α)) : List (List α) → NNReal
  | []        => 1
  | t :: rest => cond ctx t * tokBranchPriorAux cond (ctx ++ [t]) rest

/-- Token branch prior: ∏_{i < |s|} cond(s[:i], s[i]).  Chapter definition. -/
def tokBranchPrior {α : Type}
    (cond : List (List α) → List α → NNReal) (s : List (List α)) : NNReal :=
  tokBranchPriorAux cond [] s

@[simp] lemma tokBranchPrior_nil {α : Type}
    (cond : List (List α) → List α → NNReal) :
    tokBranchPrior cond [] = 1 := rfl

lemma tokBranchPriorAux_append {α : Type}
    (cond : List (List α) → List α → NNReal)
    (ctx b : List (List α)) (t : List α) :
    tokBranchPriorAux cond ctx (b ++ [t]) =
      tokBranchPriorAux cond ctx b * cond (ctx ++ b) t := by
  induction b generalizing ctx with
  | nil => simp [tokBranchPriorAux]
  | cons t' b' ih =>
    simp only [List.cons_append, tokBranchPriorAux]
    rw [ih]; ring_nf
    congr 1; congr 1
    simp [List.append_assoc]

lemma tokBranchPrior_append_one {α : Type}
    (cond : List (List α) → List α → NNReal)
    (b : List (List α)) (t : List α) :
    tokBranchPrior cond (b ++ [t]) = tokBranchPrior cond b * cond b t := by
  simp [tokBranchPrior, tokBranchPriorAux_append]

-- ═══════════════════════════════════════════════════════════════════
-- SECTION 5 : Token prior setup  (parameters + one axiom)
-- ═══════════════════════════════════════════════════════════════════

/--
Token prior setup from the chapter.

**Parameters** (given data — the chapter also takes these as inputs):
- `T`    : tokenization function
- `cond` : conditional token prior  P(t | context)

**Derived** (not fields — computed from the parameters):
- `tokBranchPrior`  : product of conditional priors
- `derivedPrior`    : string prior  P(h) = tokBranchPrior(T h)

**Single axiom** `prefix_mass`:
The token model assigns total probability  tokBranchPrior(b)
to all properly-terminated strings whose tokenization begins with `b`.
The chapter assumes this implicitly (it is the well-calibration of the
autoregressive token model).
-/
structure TokenPriorSetup (α : Type) [Alphabet α] [DecidableEq α] where
  T               : H (α := α) → TokenSeq α
  T_right_inv     : ∀ h : H (α := α), literalizeTokenSeq (T h) = h.val
  T_nonempty      : ∀ h : H (α := α), TokenSeqNonempty (T h)
  T_seq_nonempty  : ∀ h : H (α := α), T h ≠ []
  cond            : TokenSeq α → TokenString α → NNReal
  prefix_mass     : ∀ b : TokenSeq α,
    (∑' h : H (α := α),
      if List.IsPrefix b (T h) then tokBranchPrior cond (T h) else 0)
    = tokBranchPrior cond b

variable {α : Type} [Alphabet α] [DecidableEq α]

namespace TokenPriorSetup

-- ---------------------------------------------------------------
-- Derived objects
-- ---------------------------------------------------------------

/-- String prior derived from the token model: P(h) := tokBranchPrior(T h). -/
noncomputable def derivedPrior (S : TokenPriorSetup α) : PriorFn (α := α) where
  fn h := tokBranchPrior S.cond (S.T h)

-- ---------------------------------------------------------------
-- SECTION 6 : root_branch — proved from prefix_mass
-- ---------------------------------------------------------------

theorem root_branch (S : TokenPriorSetup α) :
    BranchPrior S.derivedPrior [] = 1 := by
  unfold BranchPrior derivedPrior
  -- PrefixOfCode [] {h.val} is always true ([] is a prefix of everything)
  have h_simp : ∀ h : H (α := α), PrefixOfCode (α := α) [] {h.val} := by
    intro h; exact ⟨h.val, rfl, List.nil_prefix⟩
  simp_rw [if_pos (h_simp _)]
  -- Now goal: tsum (fun h => tokBranchPrior S.cond (S.T h)) = 1
  -- By prefix_mass with b = []:
  have pm := S.prefix_mass []
  simp [List.nil_prefix] at pm
  exact pm

-- ---------------------------------------------------------------
-- SECTION 7 : First crossing index
-- ---------------------------------------------------------------

/-- Predicate: "a is a prefix of the literalization of the first j+1 tokens of c". -/
def CrossesByIndex (a : StringAlg α) (c : TokenSeq α) (j : ℕ) : Prop :=
  List.IsPrefix a (literalizeTokenSeq (c.take (j + 1)))

instance (a : StringAlg α) (c : TokenSeq α) (j : ℕ) :
    Decidable (CrossesByIndex a c j) := by
  unfold CrossesByIndex literalizeTokenSeq; exact inferInstance

/-- Existence of a crossing index for any h ≥ a with a ≠ []. -/
lemma exists_crossing_index (S : TokenPriorSetup α)
    (a : StringAlg α) (h : H (α := α))
    (ha : a ≠ []) (hpref : List.IsPrefix a h.val) :
    ∃ j, CrossesByIndex a (S.T h) j := by
  use (S.T h).length - 1
  unfold CrossesByIndex
  have hne := S.T_seq_nonempty h
  have htake : (S.T h).take ((S.T h).length - 1 + 1) = S.T h := by
    rw [Nat.sub_one_add_one_eq_of_pos (List.length_pos_of_ne_nil hne)]
    exact List.take_length
  rw [htake]
  rwa [S.T_right_inv]

/-- The first crossing index: smallest j with a ≤ lit(T(h).take(j+1)). -/
noncomputable def firstCrossingIdx (S : TokenPriorSetup α)
    (a : StringAlg α) (h : H (α := α))
    (ha : a ≠ []) (hpref : List.IsPrefix a h.val) : ℕ :=
  Nat.find (exists_crossing_index S a h ha hpref)

lemma firstCrossingIdx_spec (S : TokenPriorSetup α)
    (a : StringAlg α) (h : H (α := α))
    (ha : a ≠ []) (hpref : List.IsPrefix a h.val) :
    CrossesByIndex a (S.T h) (firstCrossingIdx S a h ha hpref) :=
  Nat.find_spec _

lemma firstCrossingIdx_min (S : TokenPriorSetup α)
    (a : StringAlg α) (h : H (α := α))
    (ha : a ≠ []) (hpref : List.IsPrefix a h.val)
    {j : ℕ} (hj : j < firstCrossingIdx S a h ha hpref) :
    ¬ CrossesByIndex a (S.T h) j :=
  Nat.find_min _ hj

/-- The first crossing index is within bounds. -/
lemma firstCrossingIdx_lt (S : TokenPriorSetup α)
    (a : StringAlg α) (h : H (α := α))
    (ha : a ≠ []) (hpref : List.IsPrefix a h.val) :
    firstCrossingIdx S a h ha hpref < (S.T h).length := by
  by_contra h_ge
  push_neg at h_ge
  -- The first crossing index j satisfies CrossesByIndex
  have hspec := firstCrossingIdx_spec S a h ha hpref
  unfold CrossesByIndex at hspec
  -- If j ≥ length, then take (j+1) = T h, so this just says a ≤ lit(T h) = h.val
  -- But also, the smallest such j should be ≤ length - 1.
  -- Since j ≥ length, and we know j = length - 1 works (from exists_crossing_index),
  -- j must be ≤ length - 1 < length. Contradiction.
  have hne := S.T_seq_nonempty h
  have hlen_pos : 0 < (S.T h).length := List.length_pos_of_ne_nil hne
  have : firstCrossingIdx S a h ha hpref ≤ (S.T h).length - 1 := by
    apply Nat.find_min'
    unfold CrossesByIndex
    rw [Nat.sub_one_add_one_eq_of_pos hlen_pos, List.take_length]
    rwa [S.T_right_inv]
  omega

/-- Literalization of a token-list prefix is a string prefix. -/
lemma literalize_take_prefix {α : Type} (c : List (List α)) (j : ℕ) :
    List.IsPrefix (literalizeTokenSeq (c.take j)) (literalizeTokenSeq c) := by
  unfold literalizeTokenSeq
  exact List.IsPrefix.flatten (List.take_prefix j c)

/-- Context b = T(h).take j gives a strict string-prefix of a. -/
lemma firstCrossingIdx_strict (S : TokenPriorSetup α)
    (a : StringAlg α) (h : H (α := α))
    (ha : a ≠ []) (hpref : List.IsPrefix a h.val) :
    IsStrictPrefix
      (literalizeTokenSeq ((S.T h).take (firstCrossingIdx S a h ha hpref)))
      a := by
  set j := firstCrossingIdx S a h ha hpref with hj_def
  set b := (S.T h).take j with hb_def
  constructor
  · -- literalize(b) is a prefix of a
    by_contra h_not
    have h_both_pref_hval : List.IsPrefix (literalizeTokenSeq b) h.val := by
      rw [← S.T_right_inv h]; exact literalize_take_prefix (S.T h) j
    have h_or := List.prefix_or_prefix_of_prefix hpref h_both_pref_hval
    cases h_or with
    | inl h_a_le_b =>
      by_cases hj0 : j = 0
      · rw [hb_def, hj0] at h_a_le_b
        simp [literalizeTokenSeq] at h_a_le_b
        exact ha h_a_le_b
      · have hcross : CrossesByIndex a (S.T h) (j - 1) := by
          unfold CrossesByIndex
          have hj_eq : j - 1 + 1 = j := Nat.sub_one_add_one_eq_of_pos (Nat.pos_of_ne_zero hj0)
          rw [hj_eq]
          exact h_a_le_b
        exact absurd hcross (firstCrossingIdx_min S a h ha hpref (Nat.sub_one_lt hj0))
    | inr h_b_le_a => exact h_not h_b_le_a
  · -- literalize(b) ≠ a  (strict)
    intro heq
    by_cases hj0 : j = 0
    · rw [hb_def, hj0] at heq; simp [literalizeTokenSeq] at heq; exact ha heq
    · have hcross : CrossesByIndex a (S.T h) (j - 1) := by
        unfold CrossesByIndex
        have hj_eq : j - 1 + 1 = j := Nat.sub_one_add_one_eq_of_pos (Nat.pos_of_ne_zero hj0)
        rw [hj_eq]
        rw [heq]
      exact absurd hcross (firstCrossingIdx_min S a h ha hpref (Nat.sub_one_lt hj0))

/-- The crossing token sequence b ++ [t] is a prefix of T(h). -/
lemma firstCrossing_isPrefix (S : TokenPriorSetup α)
    (a : StringAlg α) (h : H (α := α))
    (ha : a ≠ []) (hpref : List.IsPrefix a h.val) :
    let j := firstCrossingIdx S a h ha hpref
    List.IsPrefix ((S.T h).take (j + 1)) (S.T h) :=
  List.take_prefix _ _

-- ---------------------------------------------------------------
-- SECTION 8 : Helper lemmas for the main theorem
-- ---------------------------------------------------------------

/-- Take monotonicity for List prefix. -/
private lemma list_take_prefix {β : Type} (l : List β) :
    ∀ (i j : ℕ), i ≤ j → List.IsPrefix (l.take i) (l.take j) := by
  induction l with
  | nil => simp
  | cons a l ih =>
    intro i j hij
    rcases i with _ | i
    · exact List.nil_prefix
    · rcases j with _ | j
      · omega
      · simp only [List.take_succ_cons]
        obtain ⟨r, hr⟩ := ih i j (by omega)
        exact ⟨r, by simp [hr]⟩

/-- Soundness: crossing + starts_with implies a <+: h.val. -/
private lemma crossing_sound (S : TokenPriorSetup α)
    (a : StringAlg α) (h : H (α := α))
    (b : TokenSeq α) (t : TokenString α)
    (hcross_pref : List.IsPrefix a (literalizeTokenSeq (b ++ [t])))
    (hstart : List.IsPrefix (b ++ [t]) (S.T h)) :
    PrefixOfCode a {h.val} :=
  ⟨h.val, rfl, List.IsPrefix.trans hcross_pref
    (by rw [← S.T_right_inv h]; exact List.IsPrefix.flatten hstart)⟩

/-- If b <+: l then l.take(b.length) = b. -/
private lemma take_of_prefix {β : Type} {b l : List β}
    (h : List.IsPrefix b l) : l.take b.length = b := by
  rcases h with ⟨r, rfl⟩; simp

/-- The crossing pair is uniquely determined: b ++ [t] = T(h).take(j+1). -/
private lemma crossing_eq_take (S : TokenPriorSetup α)
    (a : StringAlg α) (h : H (α := α))
    (ha : a ≠ []) (hpref : List.IsPrefix a h.val)
    (b : TokenSeq α) (t : TokenString α)
    (hcross_str : IsStrictPrefix (literalizeTokenSeq b) a)
    (hcross_pref : List.IsPrefix a (literalizeTokenSeq (b ++ [t])))
    (hstart : List.IsPrefix (b ++ [t]) (S.T h)) :
    b ++ [t] = (S.T h).take (S.firstCrossingIdx a h ha hpref + 1) := by
  set j := S.firstCrossingIdx a h ha hpref
  -- Step 1: CrossesByIndex at b.length  →  j ≤ b.length (by minimality)
  have hb_pref : List.IsPrefix b (S.T h) :=
    List.IsPrefix.trans (List.prefix_append b [t]) hstart
  have hcross_idx : CrossesByIndex a (S.T h) b.length := by
    unfold CrossesByIndex
    have h_take_eq : (S.T h).take (b.length + 1) = b ++ [t] := by
      have := take_of_prefix hstart; simp at this; exact this
    rw [h_take_eq]; exact hcross_pref
  have hj_le : j ≤ b.length := Nat.find_min' _ hcross_idx
  -- Step 2: b.length ≤ j  (if b.length > j, then a ≤ lit(b), contradicting lit(b) < a)
  have hb_le_j : b.length ≤ j := by
    by_contra h_gt; push_neg at h_gt
    have hb_eq : (S.T h).take b.length = b := take_of_prefix hb_pref
    have h_j1_le_b : List.IsPrefix ((S.T h).take (j + 1)) b := by
      rw [← hb_eq]; exact list_take_prefix _ _ _ (by omega)
    have h_a_le_litb : List.IsPrefix a (literalizeTokenSeq b) :=
      List.IsPrefix.trans
        (firstCrossingIdx_spec S a h ha hpref)
        (List.IsPrefix.flatten h_j1_le_b)
    exact hcross_str.2
      (List.IsPrefix.eq_of_length hcross_str.1
        (le_antisymm hcross_str.1.length_le h_a_le_litb.length_le))
  -- Step 3: b.length = j, so b ++ [t] = T(h).take(j+1)
  have h_len_eq : b.length = j := le_antisymm hb_le_j hj_le
  have hbt_eq : (S.T h).take (j + 1) = b ++ [t] := by
    have h := take_of_prefix hstart; simp at h; rw [← h_len_eq]; exact h
  exact hbt_eq.symm

-- ---------------------------------------------------------------
-- SECTION 9 : Token prior path summation — the main theorem
-- ---------------------------------------------------------------

/--
**Token Prior Path Summation** (piecewise form).

For `a = []` (root):  `P^{br}(a) = 1`.

For `a ≠ []` (non-root):
$$
P^{br}(a)=\sum_{\mathbf{b},\,T^{-1}(\mathbf{b}) < a}
  P^{br}_{tok}(\mathbf{b})\,
  \sum_{\mathbf{t},\, a \le T^{-1}(\mathbf{b+t})}
  P(\mathbf{t}\mid\mathbf{b})
$$

The outer/inner sums are expressed as `tsum` over all token sequences /
tokens with indicator conditions.  Only finitely many terms are non-zero
for any fixed `a`.
-/
theorem token_prior_path_summation (S : TokenPriorSetup α) :
    ∀ a : StringAlg α,
      BranchPrior S.derivedPrior a =
        if a = [] then 1
        else ∑' (b : TokenSeq α) (t : TokenString α),
          if (IsStrictPrefix (literalizeTokenSeq b) a ∧
              List.IsPrefix a (literalizeTokenSeq (b ++ [t])))
          then tokBranchPrior S.cond b * S.cond b t
          else 0 := by
  intro a
  by_cases ha : a = []
  · subst ha; simp [S.root_branch]
  · simp only [ha, ite_false]
    -- ================================================================
    -- The proof lifts to ENNReal, uses tsum_comm, and casts back.
    -- ================================================================
    -- Abbreviation for crossing condition (does not depend on h)
    let cross (b : TokenSeq α) (t : TokenString α) : Prop :=
      IsStrictPrefix (literalizeTokenSeq b) a ∧
      List.IsPrefix a (literalizeTokenSeq (b ++ [t]))
    -- The "joint indicator" in ENNReal
    let φ : H (α := α) → TokenSeq α → TokenString α → ENNReal :=
      fun hh b t =>
        if (cross b t ∧ List.IsPrefix (b ++ [t]) (S.T hh))
        then ↑(tokBranchPrior S.cond (S.T hh))
        else 0
    -- ---- KEY LEMMA: ∀ h, LHS indicator = ∑' b t, φ h b t ----
    have key : ∀ hh : H (α := α),
        (if PrefixOfCode a {hh.val}
         then (↑(tokBranchPrior S.cond (S.T hh)) : ENNReal) else 0)
        = ∑' (b : TokenSeq α) (t : TokenString α), φ hh b t := by
      intro hh
      by_cases hpoc : PrefixOfCode a {hh.val}
      · -- PrefixOfCode holds → exactly one (b,t) contributes
        simp only [if_pos hpoc]
        obtain ⟨_, rfl, hpref⟩ := hpoc
        set j := S.firstCrossingIdx a hh ha hpref
        set bt := (S.T hh).take (j + 1) with hbt_def
        have hj_lt := firstCrossingIdx_lt S a hh ha hpref
        have hbt_len : bt.length = j + 1 := List.length_take_of_le (by omega)
        have hbt_ne : bt ≠ [] := by intro h_eq; simp [h_eq] at hbt_len
        set b0 := bt.dropLast with hb0_def
        set t0 := bt.getLast hbt_ne with ht0_def
        have hbt_split : bt = b0 ++ [t0] := (List.dropLast_append_getLast hbt_ne).symm
        have hb0_eq : b0 = (S.T hh).take j := by
          rw [hb0_def, hbt_def, List.dropLast_eq_take, hbt_len]
          simp [List.take_take]
        -- Crossing condition holds for (b0, t0)
        have hcross : cross b0 t0 := by
          exact ⟨by rw [hb0_eq]; exact firstCrossingIdx_strict S a hh ha hpref,
                 by rw [← hbt_split]; exact firstCrossingIdx_spec S a hh ha hpref⟩
        have hstart' : List.IsPrefix (b0 ++ [t0]) (S.T hh) := hbt_split ▸ List.take_prefix _ _
        -- φ at (b0, t0) = P(hh)
        have hφ_val : φ hh b0 t0 = ↑(tokBranchPrior S.cond (S.T hh)) :=
          if_pos ⟨hcross, hstart'⟩
        -- φ at all other (b, t) = 0
        have hφ_zero : ∀ b t, φ hh b t ≠ 0 → b = b0 ∧ t = t0 := by
          intro b t hne
          simp only [φ] at hne
          split at hne
          · next hcond =>
            obtain ⟨⟨hcs, hcp⟩, hst⟩ := hcond
            have h_eq := crossing_eq_take S a hh ha hpref b t hcs hcp hst
            -- h_eq : b ++ [t] = (S.T hh).take (j + 1) = bt
            rw [← hbt_def] at h_eq
            rw [hbt_split] at h_eq
            -- From h_eq : b ++ [t] = b0 ++ [t0], extract b = b0 and t = t0
            have h_last : t = t0 := by
              have := congr_arg List.getLast? h_eq; simp at this; exact this
            have h_first : b = b0 := by
              have := congr_arg List.dropLast h_eq; simp at this; exact this
            exact ⟨h_first, h_last⟩
          · exact absurd rfl hne
        -- The tsum has a single nonzero term at (b0, t0)
        symm
        -- All terms with b ≠ b0 or t ≠ t0 are zero
        have hφ_eq : ∀ b t, φ hh b t = if b = b0 ∧ t = t0 then
            ↑(tokBranchPrior S.cond (S.T hh)) else 0 := by
          intro b t
          by_cases h_eq : b = b0 ∧ t = t0
          · rw [if_pos h_eq, h_eq.1, h_eq.2, hφ_val]
          · rw [if_neg h_eq]
            by_contra hne
            exact h_eq (hφ_zero b t hne)
        simp_rw [hφ_eq]
        -- ∑' b, ∑' t, if b = b0 ∧ t = t0 then v else 0 = v
        -- The inner tsum: for fixed b, ∑' t, if b = b0 ∧ t = t0 then v else 0
        --   = if b = b0 then ∑' t, (if t = t0 then v else 0) else 0
        --   = if b = b0 then v else 0
        -- The outer tsum: ∑' b, if b = b0 then v else 0 = v
        have h_inner : ∀ b, ∑' t, (if b = b0 ∧ t = t0 then
            (↑(tokBranchPrior S.cond (S.T hh)) : ENNReal) else 0)
          = if b = b0 then ↑(tokBranchPrior S.cond (S.T hh)) else 0 := by
          intro b
          by_cases hb : b = b0
          · subst hb; simp [tsum_ite_eq]
          · simp [hb]
        simp_rw [h_inner]
        exact tsum_ite_eq b0 _
      · -- ¬PrefixOfCode → all terms are 0
        simp only [if_neg hpoc]
        have : ∀ b t, φ hh b t = 0 := by
          intro b t; simp only [φ]
          split
          · next hcond => exact absurd (crossing_sound S a hh b t hcond.1.2 hcond.2) hpoc
          · rfl
        simp [this]
    -- ---- FACTOR LEMMA: ∀ b t, ∑' h, φ h b t = crossing ? tokBranchPrior(b++[t]) : 0 ----
    have factor : ∀ (b : TokenSeq α) (t : TokenString α),
        ∑' hh, φ hh b t =
          if cross b t
          then ↑(tokBranchPrior S.cond (b ++ [t]))
          else 0 := by
      intro b t
      by_cases hc : cross b t
      · simp only [if_pos hc]
        have hφ_eq : ∀ hh : H (α := α), φ hh b t =
            if List.IsPrefix (b ++ [t]) (S.T hh)
            then ↑(tokBranchPrior S.cond (S.T hh)) else 0 := by
          intro hh; simp only [φ, hc, true_and]
        simp_rw [hφ_eq]
        have h_pm := S.prefix_mass (b ++ [t])
        -- Push cast inside tsum: ∑'ₕ (if .. then ↑v else 0) = ↑(∑'ₕ if .. then v else 0)
        have h_cast_eq : ∀ hh : H (α := α),
            (if (b ++ [t]) <+: S.T hh
             then (↑(tokBranchPrior S.cond (S.T hh)) : ENNReal) else 0)
            = (↑(if (b ++ [t]) <+: S.T hh
              then tokBranchPrior S.cond (S.T hh) else (0 : NNReal)) : ENNReal) := by
          intro hh; split <;> simp
        simp_rw [h_cast_eq]
        -- ENNReal tsum always exists, so we can compute directly.
        -- We want: ∑'ₕ (↑g(h) : ENNReal) = ↑(tokBranchPrior(b++[t]))
        -- Since the ENNReal tsum = sup of partial sums, and h_pm gives the NNReal value,
        -- we use the fact that if the ENNReal tsum is finite, it equals the NNReal tsum cast.
        -- A cleaner approach: show the ENNReal tsum ≤ ↑(tokBranchPrior(b++[t]))
        -- and ≥ ↑(tokBranchPrior(b++[t])).
        -- Instead, use h_pm directly by rewriting through tsum.
        have h_eq_nnreal : (∑' hh : H (α := α),
            if (b ++ [t]) <+: S.T hh
            then tokBranchPrior S.cond (S.T hh) else (0 : NNReal))
          = tokBranchPrior S.cond (b ++ [t]) := h_pm
        -- Now lift: ENNReal tsum = ↑(NNReal tsum) when summable
        -- Use: the ENNReal tsum is always ≥ any partial sum, and ≤ ↑(NNReal total)
        -- For the equality, we use ENNReal.tsum_coe_ne_top_iff_summable backwards
        -- Actually simplest: use ENNReal.toNNReal_tsum and the fact that sum < ⊤
        -- Or just note that for indicator functions on NNReal, summability is easy.
        -- The function is ≤ 1 pointwise (since tokBranchPrior ≤ 1 by prefix_mass with b=[])
        -- Actually let's just compute: ↑(∑' h, g h) = ∑' h, ↑(g h) when summable
        -- Summability: follows from h_pm (if the tsum equals a finite value, f is summable)
        sorry
      · simp only [if_neg hc]
        have : ∀ hh, φ hh b t = 0 := by
          intro hh; simp only [φ]; split
          · next h => exact absurd h.1 hc
          · rfl
        simp [this]
    -- ---- CHAIN THE EQUALITIES ----
    -- Cast to ENNReal and use tsum_comm
    suffices h_ennreal :
        (↑(BranchPrior S.derivedPrior a) : ENNReal) =
        ↑(∑' (b : TokenSeq α) (t : TokenString α),
          if (IsStrictPrefix (literalizeTokenSeq b) a ∧
              List.IsPrefix a (literalizeTokenSeq (b ++ [t])))
          then tokBranchPrior S.cond b * S.cond b t
          else 0) by
      exact_mod_cast h_ennreal
    -- Unfold BranchPrior and derivedPrior
    simp only [BranchPrior, derivedPrior]
    -- Step 1: Apply KEY LEMMA pointwise
    conv_lhs => ext hh; rw [key hh]
    -- Now LHS = ∑' hh, ∑' b, ∑' t, φ hh b t
    -- Step 2: Swap h with (b, t) using ENNReal.tsum_comm
    conv_lhs =>
      rw [show (∑' hh, ∑' b, ∑' t, φ hh b t) = ∑' b, ∑' t, ∑' hh, φ hh b t from by
        calc ∑' hh, ∑' b, ∑' t, φ hh b t
            = ∑' b, ∑' hh, ∑' t, φ hh b t := ENNReal.tsum_comm
          _ = ∑' b, ∑' t, ∑' hh, φ hh b t := by
              congr 1; ext b; exact ENNReal.tsum_comm]
    -- Step 3: Apply FACTOR LEMMA
    conv_lhs => ext b; ext t; rw [factor b t]
    -- Now = ∑' b t, if cross then ↑(tokBranchPrior(b++[t])) else 0
    -- Step 4: Rewrite tokBranchPrior(b++[t]) = tokBranchPrior(b) * cond(b, t)
    congr 1; ext b; congr 1; ext t
    split
    · next hc =>
      push_cast; rw [tokBranchPrior_append_one]
    · rfl

end TokenPriorSetup
end Trie
