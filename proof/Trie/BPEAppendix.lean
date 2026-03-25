import Trie.TokenPrior

namespace Trie

variable {α : Type} [DecidableEq α]

abbrev BpeToken (α : Type) := TokenString α
abbrev BpeSeq (α : Type) := TokenSeq α

structure MergeStep (α : Type) where
  left : BpeToken α
  right : BpeToken α
  merged : BpeToken α
  merged_eq : merged = left ++ right

def decompress (step : MergeStep α) : BpeSeq α → BpeSeq α
  | [] => []
  | x :: rest =>
      if x = step.merged then
        step.left :: step.right :: decompress step rest
      else
        x :: decompress step rest

def Reduced (step : MergeStep α) : BpeSeq α → Prop
  | [] => True
  | [_] => True
  | x :: y :: rest => ¬ (x = step.left ∧ y = step.right) ∧ Reduced step (y :: rest)

def OneCharTokens (s : BpeSeq α) : Prop :=
  ∀ t ∈ s, t.length = 1

def BpeCanonical : List (MergeStep α) → BpeSeq α → Prop
  | [], s => OneCharTokens s
  | step :: rest, s => Reduced step s ∧ BpeCanonical rest (decompress step s)

lemma decompress_preserves_literalization (step : MergeStep α) :
    ∀ s : BpeSeq α, literalizeTokenSeq (decompress step s) = literalizeTokenSeq s
  | [] => by simp [decompress, literalizeTokenSeq]
  | x :: xs => by
      by_cases hx : x = step.merged
      · subst x
        calc
          literalizeTokenSeq (decompress step (step.merged :: xs))
              = step.left ++ step.right ++ literalizeTokenSeq (decompress step xs) := by
                  simp [decompress, literalizeTokenSeq]
          _ = step.left ++ step.right ++ literalizeTokenSeq xs := by
                  rw [decompress_preserves_literalization step xs]
          _ = step.merged ++ literalizeTokenSeq xs := by
                  rw [step.merged_eq]
          _ = literalizeTokenSeq (step.merged :: xs) := by
                  simp [literalizeTokenSeq]
      · calc
          literalizeTokenSeq (decompress step (x :: xs))
              = x ++ literalizeTokenSeq (decompress step xs) := by
                  simp [decompress, hx, literalizeTokenSeq]
          _ = x ++ literalizeTokenSeq xs := by
                  rw [decompress_preserves_literalization step xs]
          _ = literalizeTokenSeq (x :: xs) := by
                  simp [literalizeTokenSeq]

lemma decompress_append (step : MergeStep α) :
    ∀ s t : BpeSeq α, decompress step (s ++ t) = decompress step s ++ decompress step t
  | [], t => by simp [decompress]
  | x :: xs, t => by
      by_cases hx : x = step.merged
      · subst x
        simp [decompress, decompress_append, List.cons_append]
      · simp [decompress, hx, decompress_append, List.cons_append]

lemma decompress_nonempty (step : MergeStep α) {s : BpeSeq α} (hs : s ≠ []) :
    decompress step s ≠ [] := by
  cases s with
  | nil => contradiction
  | cons x xs =>
      by_cases hx : x = step.merged
      · subst x
        simp [decompress]
      · simp [decompress, hx]

omit [DecidableEq α] in
lemma oneCharTokens_append_overlap :
    ∀ a b c : BpeSeq α, OneCharTokens (a ++ b) → OneCharTokens (b ++ c) → OneCharTokens (a ++ b ++ c)
  | [], b, c, hab, hbc => by
      simpa using hbc
  | x :: xs, b, c, hab, hbc => by
      intro t ht
      have hsplit : t = x ∨ t ∈ xs ++ b ++ c := by
        simpa [List.mem_append, List.mem_cons, List.append_assoc] using ht
      rcases hsplit with rfl | ht'
      · exact hab _ (by simp)
      ·
        have habTail : OneCharTokens (xs ++ b) := by
          intro u hu
          exact hab u (by simp [hu])
        exact oneCharTokens_append_overlap xs b c habTail hbc t ht'

omit [DecidableEq α] in
lemma reduced_append_overlap (step : MergeStep α) :
    ∀ a b c : BpeSeq α, b ≠ [] →
      Reduced step (a ++ b) →
      Reduced step (b ++ c) →
      Reduced step (a ++ b ++ c)
  | [], b, c, _, _, hbc => by simpa using hbc
  | x :: xs, b, c, hb, hab, hbc => by
      cases xs with
      | nil =>
          cases b with
          | nil => contradiction
          | cons y ys =>
              rcases hab with ⟨hxy, _⟩
              simpa [Reduced, List.append_assoc] using And.intro hxy hbc
      | cons y ys =>
          rcases hab with ⟨hxy, htail⟩
          have hb' : b ≠ [] := hb
          simpa [Reduced, List.cons_append, List.append_assoc] using
            And.intro hxy (reduced_append_overlap step (y :: ys) b c hb' htail hbc)

theorem bpe_overlap_concatenation :
    ∀ sys : List (MergeStep α),
      ∀ a b c : BpeSeq α, b ≠ [] →
        BpeCanonical sys (a ++ b) →
        BpeCanonical sys (b ++ c) →
        BpeCanonical sys (a ++ b ++ c)
  | [], a, b, c, _, hab, hbc => by
      exact oneCharTokens_append_overlap a b c hab hbc
  | step :: rest, a, b, c, hb, hab, hbc => by
      have hab' := hab
      have hbc' := hbc
      rcases hab' with ⟨hredAB, hcanAB⟩
      rcases hbc' with ⟨hredBC, hcanBC⟩
      have hbD : decompress step b ≠ [] := decompress_nonempty step hb
      have hprev :=
        bpe_overlap_concatenation rest
          (decompress step a) (decompress step b) (decompress step c) hbD
          (by simpa [decompress_append] using hcanAB)
          (by simpa [decompress_append] using hcanBC)
      constructor
      · simpa [List.append_assoc] using reduced_append_overlap step a b c hb hredAB hredBC
      · simpa [decompress_append, List.append_assoc] using hprev

end Trie
