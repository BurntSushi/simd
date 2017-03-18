#![allow(dead_code)]

use std::ops;

use stdsimd as SS;

use super::*;
use super::bitcast;

/// Boolean type for 64-bit integers.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy, Clone)]
pub struct bool64i(i64);
/// Boolean type for 64-bit floats.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy, Clone)]
pub struct bool64f(i64);
/// A SIMD vector of 2 `u64`s.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct u64x2(SS::u64x2);
/// A SIMD vector of 2 `i64`s.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct i64x2(SS::i64x2);
/// A SIMD vector of 2 `f64`s.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct f64x2(SS::f64x2);
/// A SIMD boolean vector for length-2 vectors of 64-bit integers.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct bool64ix2(SS::i64x2);
/// A SIMD boolean vector for length-2 vectors of 64-bit floats.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct bool64fx2(SS::i64x2);

simd! {
    bool64ix2: i64x2 = i64, u64x2 = u64, bool64ix2 = bool64i;
    bool64fx2: f64x2 = f64, bool64fx2 = bool64f;
}
basic_impls! {
    u64x2: u64, bool64ix2, 2, x0 | x1;
    i64x2: i64, bool64ix2, 2, x0 | x1;
    f64x2: f64, bool64fx2, 2, x0 | x1;
}

mod common {
    use super::*;
    // naive for now
    #[inline]
    pub fn bool64ix2_all(x: bool64ix2) -> bool {
        x.extract(0) && x.extract(1)
    }
    #[inline]
    pub fn bool64ix2_any(x: bool64ix2) -> bool {
        x.extract(0) || x.extract(1)
    }
    #[inline]
    pub fn bool64fx2_all(x: bool64fx2) -> bool {
        x.extract(0) && x.extract(1)
    }
    #[inline]
    pub fn bool64fx2_any(x: bool64fx2) -> bool {
        x.extract(0) || x.extract(1)
    }}
bool_impls! {
    bool64ix2: bool64i, i64x2, i64, 2, bool64ix2_all, bool64ix2_any, x0 | x1
        [/// Convert `self` to a boolean vector for interacting with floating point vectors.
         to_f -> bool64fx2];

    bool64fx2: bool64f, i64x2, i64, 2, bool64fx2_all, bool64fx2_any, x0 | x1
        [/// Convert `self` to a boolean vector for interacting with integer vectors.
         to_i -> bool64ix2];
}

impl u64x2 {
    /// Convert each lane to a signed integer.
    #[inline]
    pub fn to_i64(self) -> i64x2 {
        i64x2(self.0.as_i64x2())
    }
    /// Convert each lane to a 64-bit float.
    #[inline]
    pub fn to_f64(self) -> f64x2 {
        f64x2(self.0.as_f64x2())
    }
}
impl i64x2 {
    /// Convert each lane to an unsigned integer.
    #[inline]
    pub fn to_u64(self) -> u64x2 {
        u64x2(self.0.as_u64x2())
    }
    /// Convert each lane to a 64-bit float.
    #[inline]
    pub fn to_f64(self) -> f64x2 {
        f64x2(self.0.as_f64x2())
    }
}
impl f64x2 {
    /// Convert each lane to a signed integer.
    #[inline]
    pub fn to_i64(self) -> i64x2 {
        i64x2(self.0.as_i64x2())
    }
    /// Convert each lane to an unsigned integer.
    #[inline]
    pub fn to_u64(self) -> u64x2 {
        u64x2(self.0.as_u64x2())
    }

    /// Convert each lane to a 32-bit float.
    #[inline]
    pub fn to_f32(self) -> f32x4 {
        let x = self.0.as_f32x2();
        f32x4::new(x.extract(0), x.extract(1), 0.0, 0.0)
    }
}

neg_impls!{
    0,
    i64x2,
}
neg_impls! {
    0.0,
    f64x2,
}
macro_rules! not_impls {
    ($($ty: ident,)*) => {
        $(impl ops::Not for $ty {
            type Output = Self;
            fn not(self) -> Self {
                $ty::splat(!0) ^ self
            }
        })*
    }
}
not_impls! {
    i64x2,
    u64x2,
}

macro_rules! operators {
    ($($trayt: ident ($method: ident): $($ty: ident),*;)*) => {
        $(
            $(impl ops::$trayt for $ty {
                type Output = Self;
                #[inline]
                fn $method(self, x: Self) -> Self {
                    $ty((self.0).$method(x.0))
                }
            })*
        )*
    }
}
operators! {
    Add (add):
        i64x2, u64x2,
        f64x2;
    Sub (sub):
        i64x2, u64x2,
        f64x2;
    Mul (mul):
        i64x2, u64x2,
        f64x2;
    Div (div): f64x2;

    BitAnd (bitand):
        i64x2, u64x2,
        bool64ix2,
        bool64fx2;
    BitOr (bitor):
        i64x2, u64x2,
        bool64ix2,
        bool64fx2;
    BitXor (bitxor):
        i64x2, u64x2,
        bool64ix2,
        bool64fx2;
}

macro_rules! shift_one {
    ($ty: ident, $($by: ident),*) => {
        $(
            impl ops::Shl<$by> for $ty {
                type Output = Self;
                #[inline]
                fn shl(self, other: $by) -> Self {
                    $ty(self.0.shl(other))
                }
            }
            impl ops::Shr<$by> for $ty {
                type Output = Self;
                #[inline]
                fn shr(self, other: $by) -> Self {
                    $ty(self.0.shr(other))
                }
            }
        )*
    }
}

macro_rules! shift {
    ($($ty: ident),*) => {
        $(shift_one! {
            $ty,
            u8, u16, u32, u64, usize,
            i8, i16, i32, i64, isize
        })*
    }
}
shift! {
    i64x2, u64x2
}
