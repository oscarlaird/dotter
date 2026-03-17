import Trie.Core
import Mathlib.Logic.Relation
import Mathlib.Topology.Algebra.InfiniteSum.ENNReal

namespace Trie

open scoped BigOperators

variable {α : Type} [DecidableEq α]

/-- A token-ancestor tracking state is a trie node together with an array index. -/
abbrev MDLState (α : Type) := StringAlg α × ℕ

/--
When we move from a node to its parent, the same token-ancestor summary is represented
by this shifted index. This is the abstract form of the update rule used by
`TrackMDLFTA` in the chapter.
-/
def shiftIndex
    (finalTokenLength : StringAlg α → ℕ)
    (x : StringAlg α) (i : ℕ) : ℕ :=
  if i = 0 then finalTokenLength x - 1 else i - 1

/-- One upward propagation step in the MDLFTA array. -/
inductive MDLStep
    (finalTokenLength : StringAlg α → ℕ) :
    MDLState α → MDLState α → Prop where
  | mk (parent : StringAlg α) (a : α) (i : ℕ) :
      MDLStep finalTokenLength (parent ++ [a], i)
        (parent, shiftIndex finalTokenLength (parent ++ [a]) i)

/-- Reachability by repeated upward propagation. -/
abbrev MDLReach
    (finalTokenLength : StringAlg α → ℕ) :
    MDLState α → MDLState α → Prop :=
  Relation.ReflTransGen (MDLStep finalTokenLength)

/-- The subtype of states whose local contributions propagate to `s`. -/
abbrev Reachers
    (finalTokenLength : StringAlg α → ℕ)
    (s : MDLState α) :=
  {t : MDLState α // MDLReach finalTokenLength t s}

/-- Exact MDLFTA quantity: supremum of all local contributions that reach the target state. -/
def mdlContribSet
    (finalTokenLength : StringAlg α → ℕ)
    (score : MDLState α → ENNReal)
    (s : MDLState α) : Set ENNReal :=
  {r | ∃ t : Reachers finalTokenLength s, score t.1 = r}

noncomputable def exactMDL
    (finalTokenLength : StringAlg α → ℕ)
    (score : MDLState α → ENNReal)
    (s : MDLState α) : ENNReal :=
  sSup (mdlContribSet finalTokenLength score s)

/-- Updating exactly one local contribution by taking its maximum with `v`. -/
def updateLocal
    (score : MDLState α → ENNReal)
    (s : MDLState α)
    (v : ENNReal) : MDLState α → ENNReal :=
  fun t => if t = s then max (score t) v else score t

/--
Closed-form version of the single-source propagation algorithm:
states on the propagation chain from `s` are updated by taking the max with `v`,
all others are unchanged.
-/
noncomputable def trackValue
    (finalTokenLength : StringAlg α → ℕ)
    (M : MDLState α → ENNReal)
    (s : MDLState α)
    (v : ENNReal) : MDLState α → ENNReal :=
  by
    classical
    exact fun t => if MDLReach finalTokenLength s t then max (M t) v else M t

omit [DecidableEq α] in
lemma exactMDL_mono_local
    {finalTokenLength : StringAlg α → ℕ}
    {score₁ score₂ : MDLState α → ENNReal}
    (hmono : ∀ s, score₁ s ≤ score₂ s)
    (t : MDLState α) :
    exactMDL finalTokenLength score₁ t ≤ exactMDL finalTokenLength score₂ t := by
  show sSup (mdlContribSet finalTokenLength score₁ t : Set ENNReal)
      ≤ sSup (mdlContribSet finalTokenLength score₂ t : Set ENNReal)
  refine sSup_le ?_
  intro r hr
  rcases hr with ⟨u, rfl⟩
  have hmem : score₂ u.1 ∈ mdlContribSet finalTokenLength score₂ t := by
    exact ⟨u, rfl⟩
  exact le_trans (hmono u.1) (le_sSup hmem)

omit [DecidableEq α] in
lemma exactMDL_ge_local
    {finalTokenLength : StringAlg α → ℕ}
    {score : MDLState α → ENNReal}
    {s t : MDLState α}
    (hst : MDLReach finalTokenLength s t) :
    score s ≤ exactMDL finalTokenLength score t := by
  show score s ≤ sSup (mdlContribSet finalTokenLength score t : Set ENNReal)
  have hmem : score s ∈ mdlContribSet finalTokenLength score t := by
    exact ⟨(⟨s, hst⟩ : Reachers finalTokenLength t), rfl⟩
  exact le_sSup hmem

omit [DecidableEq α] in
/-- Exact MDLFTA values are monotone upward along the propagation relation. -/
theorem exactMDL_monotone_of_reach
    {finalTokenLength : StringAlg α → ℕ}
    {score : MDLState α → ENNReal}
    {s t : MDLState α}
    (hst : MDLReach finalTokenLength s t) :
    exactMDL finalTokenLength score s ≤ exactMDL finalTokenLength score t := by
  show sSup (mdlContribSet finalTokenLength score s : Set ENNReal)
      ≤ sSup (mdlContribSet finalTokenLength score t : Set ENNReal)
  refine sSup_le ?_
  intro r hr
  rcases hr with ⟨u, hu_eq⟩
  rw [← hu_eq]
  exact exactMDL_ge_local (Relation.ReflTransGen.trans u.2 hst)

omit [DecidableEq α] in
theorem exactMDL_monotone_of_step
    {finalTokenLength : StringAlg α → ℕ}
    {score : MDLState α → ENNReal}
    {s t : MDLState α}
    (hst : MDLStep finalTokenLength s t) :
    exactMDL finalTokenLength score s ≤ exactMDL finalTokenLength score t := by
  exact exactMDL_monotone_of_reach (Relation.ReflTransGen.single hst)

lemma exactMDL_update_unaffected
    {finalTokenLength : StringAlg α → ℕ}
    {score : MDLState α → ENNReal}
    {s t : MDLState α}
    {v : ENNReal}
    (hnot : ¬ MDLReach finalTokenLength s t) :
    exactMDL finalTokenLength (updateLocal score s v) t
      = exactMDL finalTokenLength score t := by
  show sSup (mdlContribSet finalTokenLength (updateLocal score s v) t : Set ENNReal)
      = sSup (mdlContribSet finalTokenLength score t : Set ENNReal)
  apply le_antisymm
  · refine sSup_le ?_
    intro r hr
    rcases hr with ⟨u, hr_eq⟩
    have hu_ne : u.1 ≠ s := by
      intro h_eq
      apply hnot
      simpa [h_eq] using u.2
    rw [show updateLocal score s v u.1 = score u.1 by simp [updateLocal, hu_ne]] at hr_eq
    rw [← hr_eq]
    have hmem : score u.1 ∈ mdlContribSet finalTokenLength score t := by
      exact ⟨u, rfl⟩
    exact le_sSup hmem
  · refine sSup_le ?_
    intro r hr
    rcases hr with ⟨u, hr_eq⟩
    rw [← hr_eq]
    have hu_ne : u.1 ≠ s := by
      intro h_eq
      apply hnot
      simpa [h_eq] using u.2
    have hsame : updateLocal score s v u.1 = score u.1 := by
      simp [updateLocal, hu_ne]
    rw [← hsame]
    exact le_sSup ⟨u, rfl⟩

lemma exactMDL_update_affected
    {finalTokenLength : StringAlg α → ℕ}
    {score : MDLState α → ENNReal}
    {s t : MDLState α}
    {v : ENNReal}
    (hst : MDLReach finalTokenLength s t) :
    exactMDL finalTokenLength (updateLocal score s v) t
      = max (exactMDL finalTokenLength score t) v := by
  show sSup (mdlContribSet finalTokenLength (updateLocal score s v) t : Set ENNReal)
      = max (sSup (mdlContribSet finalTokenLength score t : Set ENNReal)) v
  apply le_antisymm
  · refine sSup_le ?_
    intro r hr
    rcases hr with ⟨u, hr_eq⟩
    by_cases hu : u.1 = s
    · have h_upd : updateLocal score s v u.1 = max (score s) v := by
        simp [updateLocal, hu]
      rw [h_upd] at hr_eq
      rw [← hr_eq]
      have h_old_le : score s ≤ sSup (mdlContribSet finalTokenLength score t : Set ENNReal) :=
        exactMDL_ge_local hst
      exact max_le_iff.mpr ⟨le_trans h_old_le (le_max_left _ _), le_max_right _ _⟩
    · rw [show updateLocal score s v u.1 = score u.1 by simp [updateLocal, hu]] at hr_eq
      rw [← hr_eq]
      have hmem : score u.1 ∈ mdlContribSet finalTokenLength score t := by
        exact ⟨u, rfl⟩
      exact le_trans (le_sSup hmem) (le_max_left _ _)
  · have h_old_le :
        exactMDL finalTokenLength score t
          ≤ exactMDL finalTokenLength (updateLocal score s v) t := by
      refine exactMDL_mono_local ?_ t
      intro u
      by_cases hu : u = s
      · simp [updateLocal, hu]
      · simp [updateLocal, hu]
    have h_v_le :
        v ≤ exactMDL finalTokenLength (updateLocal score s v) t := by
      have h_at_s :
          updateLocal score s v s
            ≤ exactMDL finalTokenLength (updateLocal score s v) t :=
        exactMDL_ge_local hst
      exact le_trans (by simp [updateLocal]) h_at_s
    exact max_le_iff.mpr ⟨h_old_le, h_v_le⟩

/--
Early stopping principle used by the implementation:
if a state on the propagation chain already dominates the new value,
then every state above it is unchanged by the update.
-/
theorem exactMDL_update_stable_above
    {finalTokenLength : StringAlg α → ℕ}
    {score : MDLState α → ENNReal}
    {s t u : MDLState α}
    {v : ENNReal}
    (hst : MDLReach finalTokenLength s t)
    (htu : MDLReach finalTokenLength t u)
    (hbound : v ≤ exactMDL finalTokenLength score t) :
    exactMDL finalTokenLength (updateLocal score s v) u
      = exactMDL finalTokenLength score u := by
  have hsu : MDLReach finalTokenLength s u := Relation.ReflTransGen.trans hst htu
  have hmono :
      exactMDL finalTokenLength score t ≤ exactMDL finalTokenLength score u :=
    exactMDL_monotone_of_reach htu
  have hbound' : v ≤ exactMDL finalTokenLength score u := le_trans hbound hmono
  rw [exactMDL_update_affected hsu, max_eq_left hbound']

/--
The closed-form propagation update exactly matches the mathematically correct
single-source update of the local contribution function.
-/
theorem trackValue_matches_exact
    {finalTokenLength : StringAlg α → ℕ}
    {score : MDLState α → ENNReal}
    {s : MDLState α}
    {v : ENNReal} :
    trackValue finalTokenLength (exactMDL finalTokenLength score) s v
      =
    exactMDL finalTokenLength (updateLocal score s v) := by
  funext t
  by_cases h : MDLReach finalTokenLength s t
  · simp [trackValue, h, exactMDL_update_affected]
  · simp [trackValue, h, exactMDL_update_unaffected]

end Trie
