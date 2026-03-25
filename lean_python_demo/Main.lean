def add (a b : Int) : Int :=
  a + b

def usage : String :=
  "Usage: lean_hello_bin hello | lean_hello_bin <a> <b>"

def main (args : List String) : IO UInt32 := do
  match args with
  | ["hello"] =>
      IO.println "Hello from Lean!"
      pure 0
  | [a, b] =>
      match a.toInt?, b.toInt? with
      | some x, some y =>
          IO.println s!"{add x y}"
          pure 0
      | _, _ =>
          IO.eprintln "Both arguments must be integers."
          pure 1
  | _ =>
      IO.eprintln usage
      pure 1
