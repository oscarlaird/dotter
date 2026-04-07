/-
  Auxiliary definitions and proofs for rolling hash.
-/
import Playground.defs.rolling_hash

private def hashBytesFrom (h : Hash) (bs : List UInt8) : Hash :=
  bs.foldl (fun h b => h * B + byteHash b) h

private theorem hashBytes_append_fold (xs ys : List UInt8) :
    hashBytes (xs ++ ys) = hashBytesFrom (hashBytes xs) ys := by
  simp [hashBytes, hashBytesFrom, List.foldl_append]

private theorem hashBytesFrom_extendRight (h : Hash) (ys : List UInt8) :
    hashBytesFrom h ys = extendRight h (hashBytes ys) ys.length := by
  induction ys generalizing h with
  | nil => simp [hashBytesFrom, extendRight, hashBytes]
  | cons b bs ih =>
    change hashBytesFrom (h * B + byteHash b) bs =
        extendRight h (hashBytes (b :: bs)) (List.length (b :: bs))
    rw [ih (h * B + byteHash b)]
    have hc : hashBytes (b :: bs) = hashBytesFrom (byteHash b) bs := by
      simp [hashBytes, hashBytesFrom, List.foldl]
    rw [hc]
    rw [ih (byteHash b)]
    simp only [extendRight, List.length, pow_succ']
    ring

theorem extendRight_concat_nil_impl (xs : List UInt8) :
    hashBytes (xs ++ []) = extendRight (hashBytes xs) (hashBytes []) 0 := by
  simp [hashBytes, extendRight]

theorem extendRight_concat_impl (xs ys : List UInt8) :
    hashBytes (xs ++ ys) = extendRight (hashBytes xs) (hashBytes ys) ys.length := by
  rw [hashBytes_append_fold, hashBytesFrom_extendRight]
