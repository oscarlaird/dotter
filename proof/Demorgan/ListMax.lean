import Mathlib

/-!
# Linear-scan list maximum

We implement `listMax : List Nat → Nat` as a single left-to-right fold
and prove full correctness:

* `listMax_upper_bound`  — every element of the input is `≤` the result
* `listMax_mem_or_empty` — the result is an element of the list,
                           or the list is empty (and the result is `0`)

## Why `foldl max 0`?

A left fold over `max` is literally the imperative loop

    let mut best := 0
    for x in l do best := max best x
    return best

translated directly into a pure recursive function.  Lean can run this
code *and* reason about it using the same definition.
-/

/-- The largest element of `l`, or `0` for the empty list.
    Runs in `O(n)` with a single left-to-right pass. -/
def listMax (l : List Nat) : Nat :=
  l.foldl max 0

/-! ## Internal lemmas

We need two structural facts about `foldl max` before stating the main
theorems.  They are `private` — callers should only use the public API.
-/
section internal

/-- The accumulator never decreases: folding over any suffix can only
    push the running maximum up, never down. -/
private theorem foldl_max_acc_le (acc : Nat) (l : List Nat) :
    acc ≤ l.foldl max acc := by
  induction l generalizing acc with
  | nil        => rfl
  | cons h t ih =>
    simp only [List.foldl_cons]
    exact (le_max_left acc h).trans (ih _)

/-- Every member of `l` is `≤` the fold result, for *any* starting
    accumulator.  This is the key inductive fact behind `listMax_upper_bound`. -/
private theorem foldl_max_upper_bound (acc : Nat) (l : List Nat) :
    ∀ x ∈ l, x ≤ l.foldl max acc := by
  induction l generalizing acc with
  | nil        => simp
  | cons h t ih =>
    intro x hx
    simp only [List.foldl_cons, List.mem_cons] at *
    rcases hx with rfl | hx
    · -- x is the head: x = h ≤ max acc h ≤ foldl max (max acc h) t
      exact (le_max_right acc x).trans (foldl_max_acc_le _ t)
    · -- x is in the tail: use the induction hypothesis
      exact ih _ x hx

/-- The fold result either stayed at `acc` (all elements were smaller),
    or it equals some element that was actually in the list.
    This is the key fact behind `listMax_mem_or_empty`. -/
private theorem foldl_max_eq_acc_or_mem (acc : Nat) (l : List Nat) :
    l.foldl max acc = acc ∨ l.foldl max acc ∈ l := by
  induction l generalizing acc with
  | nil        => exact Or.inl rfl
  | cons h t ih =>
    simp only [List.foldl_cons, List.mem_cons]
    rcases ih (max acc h) with heq | hmem
    · -- the fold over the tail stayed at (max acc h); unwrap it
      rw [heq]
      rcases Nat.lt_or_ge acc h with hlt | hle
      · exact Or.inr (Or.inl (max_eq_right hlt.le))  -- h became the new max
      · exact Or.inl (max_eq_left hle)               -- acc was already ≥ h
    · -- the fold over the tail picked up some element of t
      exact Or.inr (Or.inr hmem)

end internal

/-! ## Main theorems -/

/-- **Correctness (upper bound)**: `listMax l` is an upper bound —
    every element of `l` is `≤ listMax l`. -/
theorem listMax_upper_bound (l : List Nat) : ∀ x ∈ l, x ≤ listMax l :=
  foldl_max_upper_bound 0 l

/-- **Correctness (attainment)**: `listMax l` is actually achieved —
    it equals some element of `l`, unless `l` is empty. -/
theorem listMax_mem_or_empty (l : List Nat) : listMax l ∈ l ∨ l = [] := by
  cases l with
  | nil        => exact Or.inr rfl
  | cons h t   =>
    refine Or.inl ?_
    -- Unfold one step: listMax (h::t) = foldl max (max 0 h) t = foldl max h t
    simp only [listMax, List.foldl_cons, max_eq_right (Nat.zero_le h)]
    -- Now apply foldl_max_eq_acc_or_mem with acc = h
    rcases foldl_max_eq_acc_or_mem h t with heq | hmem
    · -- fold stayed at h — the head itself is the maximum
      rw [heq]; exact List.Mem.head _
    · -- fold picked up some element of t
      exact List.mem_cons_of_mem h hmem

/-! ## Sanity checks -/

#eval listMax []           -- 0
#eval listMax [3, 1, 4, 1, 5, 9, 2, 6]  -- 9
#eval listMax [7]          -- 7
#eval listMax [2, 2, 2]    -- 2
