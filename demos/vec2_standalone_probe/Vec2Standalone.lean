structure Vec2 where
  x : Float
  y : Float

def sumVec2 (v : Vec2) : Float :=
  v.x + v.y

@[export sum_vec2_boxed]
def sumVec2Boxed (v : Vec2) : Float :=
  sumVec2 v

@[export sum_vec2_xy]
def sumVec2XY (x y : Float) : Float :=
  sumVec2 { x := x, y := y }
