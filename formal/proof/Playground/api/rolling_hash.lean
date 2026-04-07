/-
  Rolling hash — main properties (proofs live in `Playground.lemmas.rolling_hash`).
-/
import Playground.defs.rolling_hash
import Playground.lemmas.rolling_hash

theorem extendRight_concat_nil (xs : List UInt8) :
    hashBytes (xs ++ []) = extendRight (hashBytes xs) (hashBytes []) 0 :=
  extendRight_concat_nil_impl xs

theorem extendRight_concat (xs ys : List UInt8) :
    hashBytes (xs ++ ys) = extendRight (hashBytes xs) (hashBytes ys) ys.length :=
  extendRight_concat_impl xs ys
