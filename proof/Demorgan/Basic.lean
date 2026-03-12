/-
  De Morgan's Laws for Booleans — a Lean 4 learning project
  ==========================================================

  De Morgan's laws:
    Law 1:  !(a && b)  =  !a || !b
    Law 2:  !(a || b)  =  !a && !b

  This file proves both laws in several different ways, going from
  the most automatic to the most explicit, so you can understand
  how Lean works at each level.

  Read the sections in order.  Each section introduces one new idea.
-/


-- ──────────────────────────────────────────────────────────────
-- § 0  Warm-up: Bool basics
-- ──────────────────────────────────────────────────────────────
-- `Bool` has exactly two values: `true` and `false`.
-- Key operations:
--   !b        negation  (NOT)
--   a && b    conjunction (AND)
--   a || b    disjunction (OR)
--
-- Run `lake env lean --run Demorgan/Basic.lean` to evaluate #eval lines,
-- or just use the Lean VS Code extension (red/green squiggles = errors).

#check Bool          -- Bool : Type
#eval  !true         -- false
#eval  true && false -- false
#eval  true || false -- true


-- ──────────────────────────────────────────────────────────────
-- § 1  Proof by `decide` — automated truth-table check
-- ──────────────────────────────────────────────────────────────
-- `decide` is a tactic that fully evaluates a decidable statement.
-- For Bool it works exactly like checking all rows of a truth table.
-- Lean verifies every combination of (a, b) ∈ {true,false}² internally.
--
-- One catch: `decide` requires a *closed* term — no free variables.
-- `revert a b` moves the variables back into the goal, turning it into
-- the closed statement `∀ a b : Bool, ...`, which `decide` can enumerate.

theorem demorgan_and_v1 (a b : Bool) :
    !(a && b) = (!a || !b) := by
  revert a b; decide

theorem demorgan_or_v1 (a b : Bool) :
    !(a || b) = (!a && !b) := by
  revert a b; decide


-- ──────────────────────────────────────────────────────────────
-- § 2  Proof by `cases` — doing the truth-table by hand
-- ──────────────────────────────────────────────────────────────
-- `decide` hides the work.  Let's do it explicitly step by step.
--
-- `cases a` splits the current goal into TWO sub-goals:
--   • sub-goal 1:  a = false  (bullet  · )
--   • sub-goal 2:  a = true   (bullet  · )
-- `cases b` inside each branch does the same for b → four sub-goals.
-- After substituting concrete values both sides of = reduce by
-- computation, so `rfl` ("reflexivity") closes each one.

theorem demorgan_and_v2 (a b : Bool) :
    !(a && b) = (!a || !b) := by
  cases a
  · -- a = false
    cases b
    · rfl   -- !(false && false) = (!false || !false)  ↝  true  = true  ✓
    · rfl   -- !(false && true)  = (!false || !true)   ↝  true  = true  ✓
  · -- a = true
    cases b
    · rfl   -- !(true && false) = (!true || !false)    ↝  true  = true  ✓
    · rfl   -- !(true && true)  = (!true || !true)     ↝  false = false ✓

theorem demorgan_or_v2 (a b : Bool) :
    !(a || b) = (!a && !b) := by
  cases a
  · cases b
    · rfl   -- !(false || false) = (!false && !false)  ↝  true  = true  ✓
    · rfl   -- !(false || true)  = (!false && !true)   ↝  false = false ✓
  · cases b
    · rfl   -- !(true || false)  = (!true && !false)   ↝  false = false ✓
    · rfl   -- !(true || true)   = (!true && !true)    ↝  false = false ✓


-- ──────────────────────────────────────────────────────────────
-- § 3  Proof by `match` — term mode (no `by`)
-- ──────────────────────────────────────────────────────────────
-- Every tactic proof has an equivalent *term-mode* proof: a pure
-- expression whose *type* is the theorem statement.
-- `match` on two booleans gives four branches; `rfl` closes each.
-- This style looks more like a normal functional program.

theorem demorgan_and_v3 (a b : Bool) :
    !(a && b) = (!a || !b) :=
  match a, b with
  | false, false => rfl
  | false, true  => rfl
  | true,  false => rfl
  | true,  true  => rfl

theorem demorgan_or_v3 (a b : Bool) :
    !(a || b) = (!a && !b) :=
  match a, b with
  | false, false => rfl
  | false, true  => rfl
  | true,  false => rfl
  | true,  true  => rfl


-- ──────────────────────────────────────────────────────────────
-- § 4  Compact case-split with `<;>`
-- ──────────────────────────────────────────────────────────────
-- `t1 <;> t2` runs `t2` on *every* sub-goal produced by `t1`.
-- So `cases a <;> cases b <;> rfl` splits all four cases at once
-- and closes each with rfl in a single line.

theorem demorgan_and_v4 (a b : Bool) :
    !(a && b) = (!a || !b) := by
  cases a <;> cases b <;> rfl

theorem demorgan_or_v4 (a b : Bool) :
    !(a || b) = (!a && !b) := by
  cases a <;> cases b <;> rfl


-- ──────────────────────────────────────────────────────────────
-- § 5  De Morgan's laws for `Prop`
-- ──────────────────────────────────────────────────────────────
-- `Bool` is a *data type* with runtime values.
-- `Prop` is the type of *logical propositions* — things you prove.
-- The analogue of `=` for propositions is `↔` (if and only if).
-- The connectives become:  ∧ (and)  ∨ (or)  ¬ (not).
--
-- New tactics used here:
--   constructor   — split an ↔ or ∧ into its two halves
--   intro h       — introduce hypothesis h from the goal
--   exact e       — close the goal with expression e
--   by_cases h:p  — classical case-split: assume p, or assume ¬p
--   left / right  — choose a branch of ∨
--   obtain        — pattern-match on a hypothesis (∧ or ∃)

theorem demorgan_and_prop (p q : Prop) :
    ¬(p ∧ q) ↔ (¬p ∨ ¬q) := by
  constructor
  · -- forward: ¬(p ∧ q) → ¬p ∨ ¬q
    intro h
    by_cases hp : p
    · -- p holds, so q must fail
      right
      intro hq
      exact h ⟨hp, hq⟩
    · -- p fails immediately
      left
      exact hp
  · -- backward: ¬p ∨ ¬q → ¬(p ∧ q)
    intro h
    intro ⟨hp, hq⟩
    cases h with
    | inl hnp => exact hnp hp
    | inr hnq => exact hnq hq

-- The OR version has a simpler proof: no classical case-split needed.
theorem demorgan_or_prop (p q : Prop) :
    ¬(p ∨ q) ↔ (¬p ∧ ¬q) :=
  ⟨fun h => ⟨fun hp => h (Or.inl hp), fun hq => h (Or.inr hq)⟩,
   fun ⟨hnp, hnq⟩ h => h.elim hnp hnq⟩


-- ──────────────────────────────────────────────────────────────
-- § 6  Exercises  🎯
-- ──────────────────────────────────────────────────────────────
-- Replace each `sorry` with a real proof.
-- Tactics you know so far: decide · cases · rfl · match · <;>
--                          constructor · intro · exact · left/right
--
-- Hint for ex4: the proof structure mirrors demorgan_or_prop above.

-- ex1: double negation cancels out
theorem ex1_double_neg (a : Bool) : !!a = a := by
  sorry

-- ex2: AND is commutative
theorem ex2_and_comm (a b : Bool) : (a && b) = (b && a) := by
  sorry

-- ex3: OR is commutative
theorem ex3_or_comm (a b : Bool) : (a || b) = (b || a) := by
  sorry

-- ex4: De Morgan's AND law for Prop, proved without using demorgan_and_prop
theorem ex4_demorgan_and_manual (p q : Prop) :
    ¬(p ∧ q) ↔ (¬p ∨ ¬q) := by
  sorry
