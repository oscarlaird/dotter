import Trie.Core
import Mathlib.Data.List.MinMax

open scoped BigOperators
open Finset

namespace Trie

variable {α : Type} [DecidableEq α]

/--
Minimal model of a likelihood update:
- `truncate x` picks a prefix of `x`
- `likelihood` assigns a real value to each node/string
-/
structure LUpdate where
  truncate : StringAlg α → StringAlg α
  likelihood : StringAlg α → ℝ
  truncate_prefix : ∀ x, IsPrefix (truncate x) x

def LUpdate.lookup (U : LUpdate (α := α)) (x : StringAlg α) : ℝ :=
  U.likelihood (U.truncate x)

def semanticMergeLookup {k : ℕ}
    (U : Fin k → LUpdate (α := α))
    (x : StringAlg α) : ℝ :=
  ∑ i, (U i).lookup x

/-- Truncation of `x` by the `i`-th update. -/
def truncAt {k : ℕ}
    (U : Fin k → LUpdate (α := α))
    (i : Fin k)
    (x : StringAlg α) : StringAlg α :=
  (U i).truncate x

/-- A pivot truncation chosen by argmax of truncation length (with `[]` fallback). -/
noncomputable def chosenPivot {k : ℕ}
    (U : Fin k → LUpdate (α := α))
    (x : StringAlg α) : StringAlg α :=
  match (List.finRange k).argmax (fun i => (truncAt U i x).length) with
  | some i => truncAt U i x
  | none => []

/-- Direct contribution: updates truncating exactly at the pivot. -/
def directContribution {k : ℕ}
    (U : Fin k → LUpdate (α := α))
    (x pivot : StringAlg α) : ℝ :=
  ∑ i ∈ Finset.univ.filter (fun i : Fin k => truncAt U i x = pivot), (U i).lookup x

/-- Carried contribution: updates truncating strictly above/below the pivot. -/
def carriedContribution {k : ℕ}
    (U : Fin k → LUpdate (α := α))
    (x pivot : StringAlg α) : ℝ :=
  ∑ i ∈ Finset.univ.filter (fun i : Fin k => truncAt U i x ≠ pivot), (U i).lookup x

/-- Algorithmic merge lookup = direct part + carried part around a pivot. -/
def mergeLookupWithPivot {k : ℕ}
    (U : Fin k → LUpdate (α := α))
    (x pivot : StringAlg α) : ℝ :=
  directContribution U x pivot + carriedContribution U x pivot

/-- Algorithm with an explicit pivot choice (argmax in our model). -/
noncomputable def mergeLookupAlgorithm {k : ℕ}
    (U : Fin k → LUpdate (α := α))
    (x : StringAlg α) : ℝ :=
  mergeLookupWithPivot U x (chosenPivot U x)

theorem mergeLookupWithPivot_eq_semantic {k : ℕ}
    (U : Fin k → LUpdate (α := α))
    (x pivot : StringAlg α) :
    mergeLookupWithPivot U x pivot = semanticMergeLookup U x := by
  classical
  simp [mergeLookupWithPivot, directContribution, carriedContribution, semanticMergeLookup,
    Finset.sum_filter_add_sum_filter_not] 
  /- `simp` with `sum_filter_add_sum_filter_not` performs the partition identity. -/

/-- Correctness of the encoded merge algorithm. -/
theorem mergeLookupAlgorithm_correct {k : ℕ}
    (U : Fin k → LUpdate (α := α))
    (x : StringAlg α) :
    mergeLookupAlgorithm U x = semanticMergeLookup U x := by
  unfold mergeLookupAlgorithm
  exact mergeLookupWithPivot_eq_semantic U x (chosenPivot U x)

end Trie

