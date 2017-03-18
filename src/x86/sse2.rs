use SS;

use super::super::*;
use bitcast;

pub use sixty_four::{f64x2, i64x2, u64x2, bool64ix2, bool64fx2};

#[doc(hidden)]
pub mod common {
    use super::super::super::*;
    use std::mem;

    #[inline]
    pub fn f32x4_sqrt(x: f32x4) -> f32x4 {
        f32x4::from(SS::_mm_sqrt_ps(x.0))
    }
    #[inline]
    pub fn f32x4_approx_rsqrt(x: f32x4) -> f32x4 {
        f32x4::from(SS::_mm_rsqrt_ps(x.0))
    }
    #[inline]
    pub fn f32x4_approx_reciprocal(x: f32x4) -> f32x4 {
        f32x4::from(SS::_mm_rcp_ps(x.0))
    }
    #[inline]
    pub fn f32x4_max(x: f32x4, y: f32x4) -> f32x4 {
        f32x4::from(SS::_mm_max_ps(x.0, y.0))
    }
    #[inline]
    pub fn f32x4_min(x: f32x4, y: f32x4) -> f32x4 {
        f32x4::from(SS::_mm_min_ps(x.0, y.0))
    }

    macro_rules! bools {
        ($($ty: ty, $all: ident, $any: ident, $movemask: ident, $width: expr;)*) => {
            $(
                #[inline]
                pub fn $all(x: $ty) -> bool {
                    unsafe {
                        SS::$movemask(mem::transmute(x)) == (1 << $width) - 1
                    }
                }
                #[inline]
                pub fn $any(x: $ty) -> bool {
                    unsafe {
                        SS::$movemask(mem::transmute(x)) != 0
                    }
                }
            )*
        }
    }

    bools! {
        bool32fx4, bool32fx4_all, bool32fx4_any, _mm_movemask_ps, 4;
        bool8ix16, bool8ix16_all, bool8ix16_any, _mm_movemask_epi8, 16;
        bool16ix8, bool16ix8_all, bool16ix8_any, _mm_movemask_epi8, 16;
        bool32ix4, bool32ix4_all, bool32ix4_any, _mm_movemask_epi8, 16;
    }
}

// 32 bit floats

pub trait Sse2F32x4 {
    fn to_f64(self) -> f64x2;
    fn move_mask(self) -> u32;
}
impl Sse2F32x4 for f32x4 {
    #[inline]
    fn to_f64(self) -> f64x2 {
        f64x2::from(
            SS::f32x2::new(self.extract(0), self.extract(1)).as_f64x2())
    }
    fn move_mask(self) -> u32 {
        SS::_mm_movemask_ps(self.into()) as u32
    }
}
pub trait Sse2Bool32fx4 {
    fn move_mask(self) -> u32;
}
impl Sse2Bool32fx4 for bool32fx4 {
    #[inline]
    fn move_mask(self) -> u32 {
        let x: f32x4 = bitcast(self);
        SS::_mm_movemask_ps(x.0) as u32
    }
}

// 64 bit floats

pub trait Sse2F64x2 {
    fn move_mask(self) -> u32;
    fn sqrt(self) -> Self;
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;
}
impl Sse2F64x2 for f64x2 {
    #[inline]
    fn move_mask(self) -> u32 {
        SS::_mm_movemask_pd(self.into()) as u32
    }

    #[inline]
    fn sqrt(self) -> Self {
        f64x2::from(SS::_mm_sqrt_pd(self.into()))
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        f64x2::from(SS::_mm_max_pd(self.into(), other.into()))
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        f64x2::from(SS::_mm_min_pd(self.into(), other.into()))
    }
}

pub trait Sse2Bool64fx2 {
    fn move_mask(self) -> u32;
}
impl Sse2Bool64fx2 for bool64fx2 {
    #[inline]
    fn move_mask(self) -> u32 {
        let x: f64x2 = bitcast(self.to_repr());
        SS::_mm_movemask_pd(x.into()) as u32
    }
}

// 64 bit ints

pub trait Sse2U64x2 {}
impl Sse2U64x2 for u64x2 {}

pub trait Sse2I64x2 {}
impl Sse2I64x2 for i64x2 {}

pub trait Sse2Bool64ix2 {}
impl Sse2Bool64ix2 for bool64ix2 {}

// 32 bit ints

pub trait Sse2U32x4 {
    fn low_mul(self, other: Self) -> u64x2;
}
impl Sse2U32x4 for u32x4 {
    #[inline]
    fn low_mul(self, other: Self) -> u64x2 {
        u64x2::from(SS::_mm_mul_epu32(self.into(), other.into()))
    }
}

pub trait Sse2I32x4 {
    fn packs(self, other: Self) -> i16x8;
}
impl Sse2I32x4 for i32x4 {
    #[inline]
    fn packs(self, other: Self) -> i16x8 {
        i16x8::from(SS::_mm_packs_epi32(self.into(), other.into()))
    }
}

pub trait Sse2Bool32ix4 {}
impl Sse2Bool32ix4 for bool32ix4 {}

// 16 bit ints

pub trait Sse2U16x8 {
    fn adds(self, other: Self) -> Self;
    fn subs(self, other: Self) -> Self;
    fn avg(self, other: Self) -> Self;
    fn mulhi(self, other: Self) -> Self;
}
impl Sse2U16x8 for u16x8 {
    #[inline]
    fn adds(self, other: Self) -> Self {
        u16x8::from(SS::_mm_adds_epu16(self.into(), other.into()))
    }
    #[inline]
    fn subs(self, other: Self) -> Self {
        u16x8::from(SS::_mm_subs_epu16(self.into(), other.into()))
    }

    #[inline]
    fn avg(self, other: Self) -> Self {
        u16x8::from(SS::_mm_avg_epu16(self.into(), other.into()))
    }

    #[inline]
    fn mulhi(self, other: Self) -> Self {
        u16x8::from(SS::_mm_mulhi_epu16(self.into(), other.into()))
    }
}

pub trait Sse2I16x8 {
    fn adds(self, other: Self) -> Self;
    fn subs(self, other: Self) -> Self;
    fn madd(self, other: Self) -> i32x4;
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;
    fn mulhi(self, other: Self) -> Self;
    fn packs(self, other: Self) -> i8x16;
    fn packus(self, other: Self) -> u8x16;
}
impl Sse2I16x8 for i16x8 {
    #[inline]
    fn adds(self, other: Self) -> Self {
        i16x8::from(SS::_mm_adds_epi16(self.into(), other.into()))
    }
    #[inline]
    fn subs(self, other: Self) -> Self {
        i16x8::from(SS::_mm_subs_epi16(self.into(), other.into()))
    }

    #[inline]
    fn madd(self, other: Self) -> i32x4 {
        i32x4::from(SS::_mm_madd_epi16(self.into(), other.into()))
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        i16x8::from(SS::_mm_max_epi16(self.into(), other.into()))
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        i16x8::from(SS::_mm_min_epi16(self.into(), other.into()))
    }

    #[inline]
    fn mulhi(self, other: Self) -> Self {
        i16x8::from(SS::_mm_mulhi_epi16(self.into(), other.into()))
    }

    #[inline]
    fn packs(self, other: Self) -> i8x16 {
        i8x16::from(SS::_mm_packs_epi16(self.into(), other.into()))
    }
    #[inline]
    fn packus(self, other: Self) -> u8x16 {
        u8x16::from(SS::_mm_packus_epi16(self.into(), other.into()))
    }
}

pub trait Sse2Bool16ix8 {}
impl Sse2Bool16ix8 for bool16ix8 {}

// 8 bit ints

pub trait Sse2U8x16 {
    fn move_mask(self) -> u32;
    fn adds(self, other: Self) -> Self;
    fn subs(self, other: Self) -> Self;
    fn avg(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;
    fn sad(self, other: Self) -> u64x2;
}
impl Sse2U8x16 for u8x16 {
    #[inline]
    fn move_mask(self) -> u32 {
        let x: SS::u8x16 = self.into();
        SS::_mm_movemask_epi8(x.into()) as u32
    }

    #[inline]
    fn adds(self, other: Self) -> Self {
        u8x16::from(SS::_mm_adds_epu8(self.into(), other.into()))
    }
    #[inline]
    fn subs(self, other: Self) -> Self {
        u8x16::from(SS::_mm_subs_epu8(self.into(), other.into()))
    }

    #[inline]
    fn avg(self, other: Self) -> Self {
        u8x16::from(SS::_mm_avg_epu8(self.into(), other.into()))
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        u8x16::from(SS::_mm_max_epu8(self.into(), other.into()))
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        u8x16::from(SS::_mm_min_epu8(self.into(), other.into()))
    }

    #[inline]
    fn sad(self, other: Self) -> u64x2 {
        u64x2::from(SS::_mm_sad_epu8(self.into(), other.into()))
    }
}

pub trait Sse2I8x16 {
    fn move_mask(self) -> u32;
    fn adds(self, other: Self) -> Self;
    fn subs(self, other: Self) -> Self;
}
impl Sse2I8x16 for i8x16 {
    #[inline]
    fn move_mask(self) -> u32 {
        SS::_mm_movemask_epi8(self.into()) as u32
    }

    #[inline]
    fn adds(self, other: Self) -> Self {
        i8x16::from(SS::_mm_adds_epi8(self.into(), other.into()))
    }
    #[inline]
    fn subs(self, other: Self) -> Self {
        i8x16::from(SS::_mm_subs_epi8(self.into(), other.into()))
    }
}

pub trait Sse2Bool8ix16 {
    fn move_mask(self) -> u32;
}
impl Sse2Bool8ix16 for bool8ix16 {
    #[inline]
    fn move_mask(self) -> u32 {
        let x: SS::i8x16 = self.to_repr().into();
        SS::_mm_movemask_epi8(x.into()) as u32
    }
}
