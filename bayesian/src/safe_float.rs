use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

#[cfg(debug_assertions)]
pub(crate) type Float = SafeFloat;
#[cfg(not(debug_assertions))]
pub(crate) type Float = f32;
#[cfg(debug_assertions)]
pub(crate) const ZERO: Float = SafeFloat(0.0);
#[cfg(not(debug_assertions))]
pub(crate) const ZERO: Float = 0.0;

#[derive(Copy, Clone, Default, PartialEq, PartialOrd)]
#[repr(transparent)]
pub(crate) struct SafeFloat(f32);

impl SafeFloat {
    pub(crate) const NAN: Self = Self(f32::NAN);
    pub(crate) const NEG_INFINITY: Self = Self(f32::NEG_INFINITY);
    #[inline]
    pub(crate) fn new(value: f32) -> Self {
        assert!(!value.is_nan(), "SafeFloat::new received NaN");
        Self(value)
    }

    #[inline]
    fn assert_valid_input(self, op: &'static str) {
        assert!(!self.0.is_nan(), "SafeFloat::{op} received NaN input");
    }

    #[inline]
    fn assert_valid_inputs(self, rhs: Self, op: &'static str) {
        self.assert_valid_input(op);
        rhs.assert_valid_input(op);
    }

    #[inline]
    fn from_op(value: f32, op: &'static str) -> Self {
        assert!(!value.is_nan(), "SafeFloat::{op} produced NaN");
        Self(value)
    }

    #[inline]
    pub(crate) fn to_f32(self) -> f32 {
        self.assert_valid_input("to_f32");
        self.0
    }

    #[inline]
    pub(crate) fn exp(self) -> Self {
        self.assert_valid_input("exp");
        Self::from_op(self.0.exp(), "exp")
    }

    #[inline]
    pub(crate) fn ln(self) -> Self {
        self.assert_valid_input("ln");
        Self::from_op(self.0.ln(), "ln")
    }

    #[inline]
    pub(crate) fn max(self, other: Self) -> Self {
        self.assert_valid_inputs(other, "max");
        Self::from_op(self.0.max(other.0), "max")
    }

    #[inline]
    pub(crate) fn is_finite(self) -> bool {
        self.0.is_finite()
    }

    #[inline]
    pub(crate) fn is_nan(self) -> bool {
        self.0.is_nan()
    }
}

impl From<f32> for SafeFloat {
    #[inline]
    fn from(value: f32) -> Self {
        Self::new(value)
    }
}

impl From<SafeFloat> for f32 {
    #[inline]
    fn from(value: SafeFloat) -> Self {
        value.0
    }
}

impl fmt::Debug for SafeFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl fmt::Display for SafeFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

macro_rules! impl_binop {
    ($trait:ident, $method:ident, $name:literal) => {
        impl $trait for SafeFloat {
            type Output = Self;

            #[inline]
            fn $method(self, rhs: Self) -> Self::Output {
                self.assert_valid_inputs(rhs, $name);
                Self::from_op(self.0.$method(rhs.0), $name)
            }
        }

        impl $trait<&SafeFloat> for SafeFloat {
            type Output = Self;

            #[inline]
            fn $method(self, rhs: &SafeFloat) -> Self::Output {
                self.$method(*rhs)
            }
        }

        impl $trait<SafeFloat> for &SafeFloat {
            type Output = SafeFloat;

            #[inline]
            fn $method(self, rhs: SafeFloat) -> Self::Output {
                (*self).$method(rhs)
            }
        }

        impl $trait<&SafeFloat> for &SafeFloat {
            type Output = SafeFloat;

            #[inline]
            fn $method(self, rhs: &SafeFloat) -> Self::Output {
                (*self).$method(*rhs)
            }
        }
    };
}

macro_rules! impl_assign_op {
    ($trait:ident, $method:ident, $op_trait:ident, $op_method:ident) => {
        impl $trait for SafeFloat {
            #[inline]
            fn $method(&mut self, rhs: Self) {
                *self = <Self as $op_trait>::$op_method(*self, rhs);
            }
        }
    };
}

impl_binop!(Add, add, "add");
impl_binop!(Sub, sub, "sub");
impl_binop!(Mul, mul, "mul");
impl_binop!(Div, div, "div");
impl_assign_op!(AddAssign, add_assign, Add, add);

impl Neg for SafeFloat {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        self.assert_valid_input("neg");
        Self::from_op(-self.0, "neg")
    }
}

#[cfg(debug_assertions)]
#[inline]
pub(crate) fn into_f32(value: Float) -> f32 {
    value.to_f32()
}

#[cfg(not(debug_assertions))]
#[inline]
pub(crate) fn into_f32(value: Float) -> f32 {
    value
}
