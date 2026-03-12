import Mathlib

/-!
# Reversing a list twice is the identity

We prove `l.reverse.reverse = l` in two ways:

1. **One-liner** — `List.reverse_reverse` is already in Mathlib and tagged
   `@[simp]`, so `simp` closes it instantly.

2. **From first principles** — we build the proof bottom-up from a single
   helper lemma about how `reverse` interacts with `++`, to understand
   *why* the theorem is true.
-/

-- ──────────────────────────────────────────────────────────────
-- § 1  The Mathlib proof
-- ──────────────────────────────────────────────────────────────
-- `List.reverse_reverse` is in Mathlib and marked @[simp], so both
-- of these work:

theorem reverse_twice_cite (l : List α) :
    l.reverse.reverse = l :=
  List.reverse_reverse l       -- cite the Mathlib lemma directly

theorem reverse_twice_simp (l : List α) :
    l.reverse.reverse = l := by
  simp                          -- @[simp] finds it automatically


-- ──────────────────────────────────────────────────────────────
-- § 2  From first principles
-- ──────────────────────────────────────────────────────────────
-- Why is it true?  The key structural fact is that `reverse`
-- *reverses the order of concatenation*:
--
--   (l ++ r).reverse = r.reverse ++ l.reverse
--
-- Once we have that, the double-reverse proof is a short calculation.

/-- Reversing a concatenation swaps and reverses the two halves. -/
private theorem reverse_append (l r : List α) :
    (l ++ r).reverse = r.reverse ++ l.reverse := by
  induction l with
  | nil =>
    -- ([] ++ r).reverse = r.reverse = r.reverse ++ [].reverse
    simp
  | cons h t ih =>
    -- ((h :: t) ++ r).reverse
    --   = (h :: (t ++ r)).reverse
    --   = (t ++ r).reverse ++ [h]       (by reverse_cons)
    --   = (r.reverse ++ t.reverse) ++ [h]  (by ih)
    --   = r.reverse ++ (t.reverse ++ [h])  (by append_assoc)
    --   = r.reverse ++ (h :: t).reverse    (by reverse_cons)
    simp [List.reverse_cons, ih, List.append_assoc]

/-- Reversing twice recovers the original list. -/
theorem reverse_twice (l : List α) :
    l.reverse.reverse = l := by
  induction l with
  | nil =>
    rfl
  | cons h t ih =>
    -- Goal: (h :: t).reverse.reverse = h :: t
    --
    -- Step 1: unfold the inner reverse
    --   (h :: t).reverse = t.reverse ++ [h]
    --
    -- Step 2: apply reverse_append to the outer reverse
    --   (t.reverse ++ [h]).reverse = [h].reverse ++ t.reverse.reverse
    --                               = [h] ++ t          (by ih)
    --                               = h :: t
    rw [List.reverse_cons, reverse_append, ih]
    rfl


-- ──────────────────────────────────────────────────────────────
-- § 3  Sanity checks
-- ──────────────────────────────────────────────────────────────

#eval [1, 2, 3].reverse.reverse    -- [1, 2, 3]
#eval ([] : List Nat).reverse      -- []
#eval ["a", "b", "c"].reverse      -- ["c", "b", "a"]
