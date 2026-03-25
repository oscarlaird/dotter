import AdderCore
import Aeneas

open Aeneas Aeneas.Std Result ControlFlow Error

namespace RustAdderProof

theorem wrapping_add_even_val (x y : Std.U32)
    (hx : x.val % 2 = 0)
    (hy : y.val % 2 = 0) :
    (core.num.U32.wrapping_add x y).val % 2 = 0 := by
  rw [core.num.U32.wrapping_add_val_eq]
  have hdiv : 2 ∣ UScalar.size UScalarTy.U32 := by
    rw [UScalar.size_UScalarTyU32]
    simp [U32.size, U32.numBits]
  rw [Nat.mod_mod_of_dvd _ hdiv]
  omega

theorem add_u32_preserves_evenness (x y : Std.U32)
    (hx : x.val % 2 = 0)
    (hy : y.val % 2 = 0) :
    ∃ z, RustAdder.add_u32 x y = ok z ∧ z.val % 2 = 0 := by
  refine ⟨core.num.U32.wrapping_add x y, ?_, ?_⟩
  · simp [RustAdder.add_u32]
  · exact wrapping_add_even_val x y hx hy

end RustAdderProof
