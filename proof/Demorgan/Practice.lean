#check true

theorem demorgan_and_v1 (a b : Bool) :
    !(a && b) = (!a || !b) := by
    revert a b;
    decide;

theorem demorgan_and_v2 (a b : Bool) :
    !(a && b) = (!a || !b) := by
    cases a
    ·
      cases b
      · rfl
      · rfl
    ·
      cases b
      · rfl
      · rfl

theorem demorgan_and_v3 (a b : Bool) :
    !(a && b) = (!a || !b) :=
    match a, b with
    | false, false => rfl
    | false, true => rfl
    | true, false => rfl
    | true, true => rfl

theorem demorgan_and_v4 (a b : Bool) :
  !(a && b) = (!a || !b) := by
  cases a <;> cases b <;> rfl
