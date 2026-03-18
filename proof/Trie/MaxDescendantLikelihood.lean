import Trie.Core
import Mathlib.Topology.Algebra.InfiniteSum.ENNReal

namespace Trie

variable {α β : Type} [DecidableEq α] [DecidableEq β]

/--
Abstract interface for the "Maximum Descendant Likelihood" section.

The chapter uses tokenization facts that are developed informally in the LaTeX.
This structure makes every such fact explicit, so the Lean development has no
hidden assumptions.
-/
structure MaxDescendantLikelihoodSetup (α β : Type) [DecidableEq α] [DecidableEq β] where
  H : Set (StringAlg α)
  likelihood : StringAlg α → ENNReal
  tokens : StringAlg α → StringAlg β
  Canonical : StringAlg β → Prop
  decode : StringAlg β → StringAlg α
  tokenPrefixOfString : StringAlg β → StringAlg α → Prop
  tokenTrunc : StringAlg α → StringAlg β → StringAlg β
  encodePrefix : StringAlg α → ℕ → StringAlg β
  encodePrefix_canonical :
    ∀ f i, Canonical (encodePrefix f i)
  encodePrefix_prefix :
    ∀ f i, i ≤ f.length → tokenPrefixOfString (encodePrefix f i) f
  identical_token_string_forward :
    ∀ {a h},
      Canonical a →
      IsPrefix (decode a) h →
      IsPrefix a (tokens h) →
      tokenTrunc (decode a) (tokens h) = a →
      IsPrefix a (tokens h)
  identical_token_string_backward :
    ∀ {a h},
      Canonical a →
      IsPrefix a (tokens h) →
      IsPrefix (decode a) h ∧
        IsPrefix a (tokens h) ∧
        tokenTrunc (decode a) (tokens h) = a
  tokenTrunc_prefix_token :
    ∀ x c, IsPrefix (tokenTrunc x c) c
  tokenTrunc_prefix_string :
    ∀ x c, tokenPrefixOfString (tokenTrunc x c) x
  tokenTrunc_canonical :
    ∀ x c, Canonical (tokenTrunc x c)
  tokenTrunc_trans :
    ∀ {x f c},
      IsPrefix x f →
      tokenTrunc x (tokenTrunc f c) = tokenTrunc x c
  canonical_prefixes_iff :
    ∀ {f b},
      Canonical b →
      (tokenPrefixOfString b f ↔ ∃ i ≤ f.length, encodePrefix f i = b)

namespace MaxDescendantLikelihoodSetup

variable (S : MaxDescendantLikelihoodSetup α β)

/-- The strings contributing to `R(a, x)`. -/
def admissible
    (a : StringAlg β)
    (x h : StringAlg α) : Prop :=
  h ∈ S.H ∧
    IsPrefix x h ∧
    IsPrefix a (S.tokens h) ∧
    S.tokenTrunc x (S.tokens h) = a

/-- The value set whose supremum is the double-path maximum likelihood. -/
def doublePathValues
    (a : StringAlg β)
    (x : StringAlg α) : Set ENNReal :=
  {r | ∃ h, S.admissible a x h ∧ S.likelihood h = r}

/--
`R(a, x)` is the maximum likelihood of a descendant whose tokenization extends `a`
and truncates back to `a` at `x`. Empty maxima are interpreted as `0` via `sSup`.
-/
noncomputable def R
    (a : StringAlg β)
    (x : StringAlg α) : ENNReal :=
  sSup (S.doublePathValues a x)

/-- The special case `R(a, decode a)`. -/
def tokenDescendantValues
    (a : StringAlg β) : Set ENNReal :=
  {r | ∃ h, h ∈ S.H ∧ IsPrefix a (S.tokens h) ∧ S.likelihood h = r}

/-- The frontier-indexed value set from the main theorem. -/
def frontierValueSet
    (F : Set (StringAlg α))
    (a : StringAlg β)
    (x : StringAlg α) : Set ENNReal :=
  {r | ∃ f ∈ F, IsPrefix x f ∧
      ∃ b, S.Canonical b ∧ IsPrefix a b ∧ S.tokenPrefixOfString b f ∧
        S.tokenTrunc x b = a ∧ S.R b f = r}

/-- The reindexed frontier value set, using string prefixes `f[:i]`. -/
def frontierReindexedValueSet
    (F : Set (StringAlg α))
    (a : StringAlg β)
    (x : StringAlg α) : Set ENNReal :=
  {r | ∃ f ∈ F, IsPrefix x f ∧
      ∃ i ≤ f.length,
        S.tokenTrunc x (S.encodePrefix f i) = a ∧
        S.R (S.encodePrefix f i) f = r}

/--
The mathematical meaning of the MTDL up-propagation algorithm: the value stored at
`(a, x)` after processing the frontier.
-/
def upPropagationValueSet
    (F : Set (StringAlg α))
    (Qfront : StringAlg β → StringAlg α → ENNReal)
    (a : StringAlg β)
    (x : StringAlg α) : Set ENNReal :=
  {r | ∃ f ∈ F, IsPrefix x f ∧
      ∃ i ≤ f.length,
        S.tokenTrunc x (S.encodePrefix f i) = a ∧
        Qfront (S.encodePrefix f i) f = r}

noncomputable def Q
    (F : Set (StringAlg α))
    (Qfront : StringAlg β → StringAlg α → ENNReal)
    (a : StringAlg β)
    (x : StringAlg α) : ENNReal :=
  sSup (S.upPropagationValueSet F Qfront a x)

lemma prefix_antisymm
    {x y : StringAlg α}
    (hxy : IsPrefix x y)
    (hyx : IsPrefix y x) :
    x = y := by
  rcases hxy with ⟨u, rfl⟩
  rcases hyx with ⟨v, hv⟩
  have hsum : x.length + u.length + v.length = x.length := by
    simpa [List.length_append, Nat.add_assoc] using congrArg List.length hv
  have hsum' : x.length + (u.length + v.length) = x.length + 0 := by
    simpa [Nat.add_assoc] using hsum
  have huv : u.length + v.length = 0 := Nat.add_left_cancel hsum'
  have hu_len : u.length = 0 := by omega
  cases u with
  | nil =>
      simp
  | cons a u =>
      simp at hu_len

/-- LaTeX lemma "Identical Token and String Prefixes". -/
theorem identicalTokenAndStringPrefixes
    {a : StringAlg β}
    (ha : S.Canonical a) :
    S.R a (S.decode a) = sSup (S.tokenDescendantValues a) := by
  show sSup (S.doublePathValues a (S.decode a)) = sSup (S.tokenDescendantValues a)
  apply congrArg sSup
  ext r
  constructor
  · intro hr
    rcases hr with ⟨h, hadm, hlh⟩
    rcases hadm with ⟨hH, hxh, hatok, htrunc⟩
    exact ⟨h, hH, S.identical_token_string_forward ha hxh hatok htrunc, hlh⟩
  · intro hr
    rcases hr with ⟨h, hH, hatok, hlh⟩
    have hadm :
        IsPrefix (S.decode a) h ∧
          IsPrefix a (S.tokens h) ∧
          S.tokenTrunc (S.decode a) (S.tokens h) = a :=
      S.identical_token_string_backward ha hatok
    exact ⟨h, ⟨hH, hadm.1, hadm.2.1, hadm.2.2⟩, hlh⟩

/-- LaTeX theorem "Double Path Maximum Likelihood from the Frontier". -/
theorem doublePathMaximumLikelihoodFromTheFrontier
    {F : Set (StringAlg α)}
    {a : StringAlg β}
    {x : StringAlg α}
    (hF : PrefixCode F)
    (hxF : PrefixOfCode x F)
    (hHF : Refinement S.H F) :
    S.R a x = sSup (S.frontierValueSet F a x) := by
  show sSup (S.doublePathValues a x) = sSup (S.frontierValueSet F a x)
  apply le_antisymm
  · refine sSup_le ?_
    intro r hr
    rcases hr with ⟨h, ⟨hH, hxh, hatok, htrunc⟩, hlh⟩
    rcases hHF h hH with ⟨f, hfF, hfh⟩
    rcases hxF with ⟨g, hgF, hxg⟩
    have hfx_or_hxf : IsPrefix f x ∨ IsPrefix x f :=
      prefix_or_prefix_of_prefix hfh hxh
    have hxf : IsPrefix x f := by
      rcases hfx_or_hxf with hfx | hxf
      · by_cases h_eq : f = x
        · subst h_eq
          exact ⟨[], by simp⟩
        · have hfg : IsPrefix f g := List.IsPrefix.trans hfx hxg
          have hne : f ≠ g := by
            intro hfg_eq
            apply h_eq
            exact prefix_antisymm hfx (hfg_eq ▸ hxg)
          exact False.elim (hF f g hfF hgF ⟨hfg, hne⟩)
      · exact hxf
    let b := S.tokenTrunc f (S.tokens h)
    have hbcanon : S.Canonical b := S.tokenTrunc_canonical f (S.tokens h)
    have hbf : S.tokenPrefixOfString b f := S.tokenTrunc_prefix_string f (S.tokens h)
    have hxa : S.tokenTrunc x b = a := by
      dsimp [b]
      rw [S.tokenTrunc_trans hxf]
      exact htrunc
    have hab : IsPrefix a b := by
      have hb : IsPrefix (S.tokenTrunc x b) b := S.tokenTrunc_prefix_token x b
      simpa [hxa] using hb
    have hr_le : r ≤ S.R b f := by
      rw [← hlh]
      exact le_sSup ⟨h, ⟨hH, hfh, S.tokenTrunc_prefix_token f (S.tokens h), rfl⟩, rfl⟩
    have hr_mem : S.R b f ∈ S.frontierValueSet F a x := by
      refine ⟨f, hfF, hxf, b, hbcanon, hab, hbf, hxa, rfl⟩
    exact le_trans hr_le (le_sSup hr_mem)
  · refine sSup_le ?_
    intro r hr
    rcases hr with ⟨f, hfF, hxf, b, hbcanon, hab, hbf, hxa, hr_eq⟩
    rw [← hr_eq]
    refine sSup_le ?_
    intro s hs
    rcases hs with ⟨h, ⟨hH, hfh, hbTok, hfb⟩, hlh⟩
    rw [← hlh]
    have hxa' : S.tokenTrunc x (S.tokens h) = a := by
      calc
        S.tokenTrunc x (S.tokens h)
            = S.tokenTrunc x (S.tokenTrunc f (S.tokens h)) := by
                symm
                exact S.tokenTrunc_trans hxf
        _ = S.tokenTrunc x b := by rw [hfb]
        _ = a := hxa
    exact le_sSup ⟨h, ⟨hH, List.IsPrefix.trans hxf hfh, List.IsPrefix.trans hab hbTok, hxa'⟩, rfl⟩

/-- LaTeX lemma "Canonical token prefixes of a string". -/
theorem canonicalTokenPrefixesOfAString
    {f : StringAlg α}
    {b : StringAlg β} :
    S.Canonical b ∧ S.tokenPrefixOfString b f
      ↔ ∃ i ≤ f.length, S.encodePrefix f i = b := by
  constructor
  · intro h
    exact (S.canonical_prefixes_iff h.1).mp h.2
  · intro h
    rcases h with ⟨i, hi, rfl⟩
    exact ⟨S.encodePrefix_canonical f i, S.encodePrefix_prefix f i hi⟩

/-- LaTeX corollary "Double Path Maximum Likelihood from the Frontier, Reindexed". -/
theorem doublePathMaximumLikelihoodFromTheFrontierReindexed
    {F : Set (StringAlg α)}
    {a : StringAlg β}
    {x : StringAlg α}
    (hF : PrefixCode F)
    (hxF : PrefixOfCode x F)
    (hHF : Refinement S.H F) :
    S.R a x = sSup (S.frontierReindexedValueSet F a x) := by
  rw [S.doublePathMaximumLikelihoodFromTheFrontier hF hxF hHF]
  apply congrArg sSup
  ext r
  constructor
  · intro hr
    rcases hr with ⟨f, hfF, hxf, b, hbcanon, hab, hbf, hxa, hrb⟩
    rcases (S.canonicalTokenPrefixesOfAString).mp ⟨hbcanon, hbf⟩ with ⟨i, hi, henc⟩
    exact ⟨f, hfF, hxf, i, hi, henc ▸ hxa, henc ▸ hrb⟩
  · intro hr
    rcases hr with ⟨f, hfF, hxf, i, hi, hxa, hrb⟩
    have hab : IsPrefix a (S.encodePrefix f i) := by
      have hprefix : IsPrefix (S.tokenTrunc x (S.encodePrefix f i)) (S.encodePrefix f i) :=
        S.tokenTrunc_prefix_token x (S.encodePrefix f i)
      simpa [hxa] using hprefix
    exact ⟨f, hfF, hxf, S.encodePrefix f i, S.encodePrefix_canonical f i, hab,
      S.encodePrefix_prefix f i hi, hxa, hrb⟩

/-- LaTeX theorem "MTDL Up Propagation Correctness". -/
theorem mtdlUpPropagationCorrectness
    {F : Set (StringAlg α)}
    {Qfront : StringAlg β → StringAlg α → ENNReal}
    {a : StringAlg β}
    {x : StringAlg α}
    (hF : PrefixCode F)
    (hxF : PrefixOfCode x F)
    (hHF : Refinement S.H F)
    (hfront :
      ∀ f ∈ F, ∀ i ≤ f.length,
        Qfront (S.encodePrefix f i) f = S.R (S.encodePrefix f i) f) :
    S.Q F Qfront a x = S.R a x := by
  rw [S.doublePathMaximumLikelihoodFromTheFrontierReindexed hF hxF hHF]
  show sSup (S.upPropagationValueSet F Qfront a x) =
      sSup (S.frontierReindexedValueSet F a x)
  apply congrArg sSup
  ext r
  constructor
  · intro hr
    rcases hr with ⟨f, hfF, hxf, i, hi, hxa, hq⟩
    exact ⟨f, hfF, hxf, i, hi, hxa, (hfront f hfF i hi).symm.trans hq⟩
  · intro hr
    rcases hr with ⟨f, hfF, hxf, i, hi, hxa, hq⟩
    exact ⟨f, hfF, hxf, i, hi, hxa, (hfront f hfF i hi).trans hq⟩

end MaxDescendantLikelihoodSetup

end Trie
