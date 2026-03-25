structure Vec2 where
  x : Float
  y : Float

def sumVec2Spec (v : Vec2) : Float :=
  v.x + v.y

@[export sum_vec2_xy_standalone]
def sumVec2XYStandalone (x y : Float) : Float :=
  x + y

theorem sumVec2XYStandalone_eq_spec (x y : Float) :
    sumVec2XYStandalone x y = sumVec2Spec { x := x, y := y } := by
  rfl
