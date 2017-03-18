#![allow(dead_code)]

use std::ops;

use SS;

#[allow(unused_imports)]
use super::{
	Simd,
    u32x4, i32x4, u16x8, i16x8, u8x16, i8x16, f32x4,
    bool32ix4, bool16ix8, bool8ix16, bool32fx4,
    bool8i, bool16i, bool32i, bool32f,
    Unalign, bitcast,
};
use super::sixty_four::*;
#[cfg(all(target_feature = "avx"))]
use super::x86::avx::common;

#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct u64x4(SS::u64x4);
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct i64x4(SS::i64x4);
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct f64x4(SS::f64x4);
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct bool64ix4(SS::i64x4);
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct bool64fx4(SS::i64x4);

#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct u32x8(SS::u32x8);
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct i32x8(SS::i32x8);
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct f32x8(SS::f32x8);
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct bool32ix8(SS::i32x8);
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct bool32fx8(SS::i32x8);

#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct u16x16(SS::u16x16);
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct i16x16(SS::i16x16);
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct bool16ix16(SS::i16x16);

#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct u8x32(SS::u8x32);
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct i8x32(SS::i8x32);
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct bool8ix32(SS::i8x32);

simd! {
    bool8ix32: i8x32 = i8, u8x32 = u8, bool8ix32 = bool8i;
    bool16ix16: i16x16 = i16, u16x16 = u16, bool16ix16 = bool16i;
    bool32ix8: i32x8 = i32, u32x8 = u32, bool32ix8 = bool32i;
    bool64ix4: i64x4 = i64, u64x4 = u64, bool64ix4 = bool64i;

    bool32fx8: f32x8 = f32, bool32fx8 = bool32f;
    bool64fx4: f64x4 = f64, bool64fx4 = bool64f;
}

basic_impls! {
    u64x4: u64, bool64ix4, 4, x0, x1 | x2, x3;
    i64x4: i64, bool64ix4, 4, x0, x1 | x2, x3;
    f64x4: f64, bool64fx4, 4, x0, x1 | x2, x3;

    u32x8: u32, bool32ix8, 8, x0, x1, x2, x3 | x4, x5, x6, x7;
    i32x8: i32, bool32ix8, 8, x0, x1, x2, x3 | x4, x5, x6, x7;
    f32x8: f32, bool32fx8, 8, x0, x1, x2, x3 | x4, x5, x6, x7;

    u16x16: u16, bool16ix16, 16, x0, x1, x2, x3, x4, x5, x6, x7 | x8, x9, x10, x11, x12, x13, x14, x15;
    i16x16: i16, bool16ix16, 16, x0, x1, x2, x3, x4, x5, x6, x7 | x8, x9, x10, x11, x12, x13, x14, x15;

    u8x32: u8, bool8ix32, 32, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15 |
        x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31;
    i8x32: i8, bool8ix32, 32, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15 |
        x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31;
}

#[cfg(all(not(target_feature = "avx")))]
#[doc(hidden)]
mod common {
    use super::*;
    // implementation via SSE vectors
    macro_rules! bools {
        ($($ty: ty, $all: ident, $any: ident;)*) => {
            $(
                #[inline]
                pub fn $all(x: $ty) -> bool {
                    x.low().all() && x.high().all()
                }
                #[inline]
                pub fn $any(x: $ty) -> bool {
                    x.low().any() || x.high().any()
                }
            )*
        }
    }

    bools! {
        bool64ix4, bool64ix4_all, bool64ix4_any;
        bool64fx4, bool64fx4_all, bool64fx4_any;
        bool32ix8, bool32ix8_all, bool32ix8_any;
        bool32fx8, bool32fx8_all, bool32fx8_any;
        bool16ix16, bool16ix16_all, bool16ix16_any;
        bool8ix32, bool8ix32_all, bool8ix32_any;
    }

}

bool_impls! {
    bool64ix4: bool64i, i64x4, i64, 4, bool64ix4_all, bool64ix4_any, x0, x1 | x2, x3
        [/// Convert `self` to a boolean vector for interacting with floating point vectors.
         to_f -> bool64fx4];

    bool64fx4: bool64f, i64x4, i64, 4, bool64fx4_all, bool64fx4_any, x0, x1 | x2, x3
        [/// Convert `self` to a boolean vector for interacting with integer vectors.
         to_i -> bool64ix4];

    bool32ix8: bool32i, i32x8, i32, 8, bool32ix8_all, bool32ix8_any, x0, x1, x2, x3 | x4, x5, x6, x7
        [/// Convert `self` to a boolean vector for interacting with floating point vectors.
         to_f -> bool32fx8];

    bool32fx8: bool32f, i32x8, i32, 8, bool32fx8_all, bool32fx8_any, x0, x1, x2, x3 | x4, x5, x6, x7
        [/// Convert `self` to a boolean vector for interacting with integer vectors.
         to_i -> bool32ix8];

    bool16ix16: bool16i, i16x16, i16, 16, bool16ix16_all, bool16ix16_any,
            x0, x1, x2, x3, x4, x5, x6, x7 | x8, x9, x10, x11, x12, x13, x14, x15 [];

    bool8ix32: bool8i, i8x32, i8, 32, bool8ix32_all, bool8ix32_any,
        x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15 |
        x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31 [];
}

pub trait LowHigh128 {
    type Half: Simd;
    /// Extract the low 128 bit part.
    fn low(self) -> Self::Half;
    /// Extract the high 128 bit part.
    fn high(self) -> Self::Half;
}

macro_rules! expr { ($x:expr) => ($x) } // HACK
macro_rules! low_high_impls {
    ($(
        $name: ident, $half: ident, $($first: tt),+ ... $($last: tt),+;
        )*) => {
        $(impl LowHigh128 for $name {
            type Half = $half;
            #[inline]
            fn low(self) -> Self::Half {
                $half::new($( expr!(self.extract($first)), )*)
            }

            #[inline]
            fn high(self) -> Self::Half {
                $half::new($( expr!(self.extract($last)), )*)
            }
        })*
    }
}

low_high_impls! {
    u64x4, u64x2, 0, 1 ... 2, 3;
    i64x4, i64x2, 0, 1 ... 2, 3;
    f64x4, f64x2, 0, 1 ... 2, 3;

    u32x8, u32x4, 0, 1, 2, 3 ... 4, 5, 6, 7;
    i32x8, i32x4, 0, 1, 2, 3 ... 4, 5, 6, 7;
    f32x8, f32x4, 0, 1, 2, 3 ... 4, 5, 6, 7;

    u16x16, u16x8, 0, 1, 2, 3, 4, 5, 6, 7 ... 8, 9, 10, 11, 12, 13, 14, 15;
    i16x16, i16x8, 0, 1, 2, 3, 4, 5, 6, 7 ... 8, 9, 10, 11, 12, 13, 14, 15;

    u8x32, u8x16,  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ...
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31;
    i8x32, i8x16,  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ...
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31;

}

macro_rules! bool_low_high_impls {
    ($(
        $name: ident: $half: ident;
        )*) => {
        $(impl LowHigh128 for $name {
            type Half = $half;
            /// Extract the low 128 bit part.
            #[inline]
            fn low(self) -> Self::Half {
                Self::Half::from_repr(self.to_repr().low())
            }

            /// Extract the high 128 bit part.
            #[inline]
            fn high(self) -> Self::Half {
                Self::Half::from_repr(self.to_repr().high())
            }
        })*
    }
}

bool_low_high_impls! {
    bool64fx4: bool64fx2;
    bool32fx8: bool32fx4;

    bool64ix4: bool64ix2;
    bool32ix8: bool32ix4;
    bool16ix16: bool16ix8;
    bool8ix32: bool8ix16;
}

impl u64x4 {
    /// Convert each lane to a signed integer.
    #[inline]
    pub fn to_i64(self) -> i64x4 {
        i64x4(self.0.as_i64x4())
    }
    /// Convert each lane to a 64-bit float.
    #[inline]
    pub fn to_f64(self) -> f64x4 {
        f64x4(self.0.as_f64x4())
    }
}

impl i64x4 {
    /// Convert each lane to an unsigned integer.
    #[inline]
    pub fn to_u64(self) -> u64x4 {
        u64x4(self.0.as_u64x4())
    }
    /// Convert each lane to a 64-bit float.
    #[inline]
    pub fn to_f64(self) -> f64x4 {
        f64x4(self.0.as_f64x4())
    }
}

impl f64x4 {
    /// Convert each lane to a signed integer.
    #[inline]
    pub fn to_i64(self) -> i64x4 {
        i64x4(self.0.as_i64x4())
    }
    /// Convert each lane to an unsigned integer.
    #[inline]
    pub fn to_u64(self) -> u64x4 {
        u64x4(self.0.as_u64x4())
    }
}

impl u32x8 {
    /// Convert each lane to a signed integer.
    #[inline]
    pub fn to_i32(self) -> i32x8 {
        i32x8(self.0.as_i32x8())
    }
    /// Convert each lane to a 32-bit float.
    #[inline]
    pub fn to_f32(self) -> f32x8 {
        f32x8(self.0.as_f32x8())
    }
}

impl i32x8 {
    /// Convert each lane to an unsigned integer.
    #[inline]
    pub fn to_u32(self) -> u32x8 {
        u32x8(self.0.as_u32x8())
    }
    /// Convert each lane to a 32-bit float.
    #[inline]
    pub fn to_f32(self) -> f32x8 {
        f32x8(self.0.as_f32x8())
    }
}

impl f32x8 {
    /// Convert each lane to a signed integer.
    #[inline]
    pub fn to_i32(self) -> i32x8 {
        i32x8(self.0.as_i32x8())
    }
    /// Convert each lane to an unsigned integer.
    #[inline]
    pub fn to_u32(self) -> u32x8 {
        u32x8(self.0.as_u32x8())
    }
}

impl i16x16 {
    /// Convert each lane to an unsigned integer.
    #[inline]
    pub fn to_u16(self) -> u16x16 {
        u16x16(self.0.as_u16x16())
    }
}

impl u16x16 {
    /// Convert each lane to a signed integer.
    #[inline]
    pub fn to_i16(self) -> i16x16 {
        i16x16(self.0.as_i16x16())
    }
}

impl i8x32 {
    /// Convert each lane to an unsigned integer.
    #[inline]
    pub fn to_u8(self) -> u8x32 {
        u8x32(self.0.as_u8x32())
    }
}

impl u8x32 {
    /// Convert each lane to a signed integer.
    #[inline]
    pub fn to_i8(self) -> i8x32 {
        i8x32(self.0.as_i8x32())
    }
}

operators! {
    Add (add):
        i8x32, u8x32, i16x16, u16x16, i32x8, u32x8, i64x4, u64x4,
        f64x4, f32x8;
    Sub (sub):
        i8x32, u8x32, i16x16, u16x16, i32x8, u32x8, i64x4, u64x4,
        f64x4, f32x8;
    Mul (mul):
        i8x32, u8x32, i16x16, u16x16, i32x8, u32x8, i64x4, u64x4,
        f64x4, f32x8;
    Div (div): f64x4, f32x8;

    BitAnd (bitand):
        i8x32, u8x32, i16x16, u16x16, i32x8, u32x8, i64x4, u64x4,
        bool64ix4, bool32ix8, bool16ix16,
        bool64fx4, bool32fx8;
    BitOr (bitor):
        i8x32, u8x32, i16x16, u16x16, i32x8, u32x8, i64x4, u64x4,
        bool64ix4, bool32ix8, bool16ix16,
        bool64fx4, bool32fx8;
    BitXor (bitxor):
        i8x32, u8x32, i16x16, u16x16, i32x8, u32x8, i64x4, u64x4,
        bool64ix4, bool32ix8, bool16ix16,
        bool64fx4, bool32fx8;
}

neg_impls!{
    0,
    i64x4,
    i32x8,
    i16x16,
    i8x32,
}

neg_impls! {
    0.0,
    f64x4,
    f32x8,
}

not_impls! {
    i64x4,
    u64x4,
    i32x8,
    u32x8,
    i16x16,
    u16x16,
    i8x32,
    u8x32,
}

shift! {
    i64x4,
    u64x4,
    i32x8,
    u32x8,
    i16x16,
    u16x16,
    i8x32,
    u8x32
}
