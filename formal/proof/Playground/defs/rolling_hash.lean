/-
  Rolling polynomial hash — definitions only (`bayesian/src/rolling_hash.rs`).
-/
import Mathlib

def M : Nat := 2^61 - 1
def B : Nat := 257

abbrev Hash := ZMod M

def byteHash (b : UInt8) : Hash :=
  b.toNat

def hashBytes (bs : List UInt8) : Hash :=
  bs.foldl (fun h b => h * B + byteHash b) 0

def extendRight (h rh : Hash) (rlen : Nat) : Hash :=
  h * B ^ rlen + rh

def appendRight (h : Hash) (b : UInt8) : Hash :=
  extendRight h (byteHash b) 1
