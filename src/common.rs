use super::*;
#[allow(unused_imports)]
use super::bitcast;
use std::ops;

#[cfg(any(target_arch = "x86",
          target_arch = "x86_64"))]
use x86::sse2::common;
#[cfg(any(target_arch = "arm"))]
use arm::neon::common;
#[cfg(any(target_arch = "aarch64"))]
use aarch64::neon::common;

macro_rules! basic_impls {
    ($(
        $name: ident:
        $elem: ident, $bool: ident, $length: expr, $($first: ident),* | $($last: ident),*;
        )*) => {
        $(impl $name {
            /// Create a new instance.
            #[inline]
            pub fn new($($first: $elem),*, $($last: $elem),*) -> $name {
                $name(SS::$name::new($($first),*, $($last),*))
            }

            /// Create a new instance where every lane has value `x`.
            #[inline]
            pub fn splat(x: $elem) -> $name {
                $name(SS::$name::splat(x))
            }

            /// Compare for equality.
            #[inline]
            pub fn eq(self, other: Self) -> $bool {
                $bool(self.0.eq(other.0))
            }
            /// Compare for equality.
            #[inline]
            pub fn ne(self, other: Self) -> $bool {
                $bool(self.0.ne(other.0))
            }
            /// Compare for equality.
            #[inline]
            pub fn lt(self, other: Self) -> $bool {
                $bool(self.0.lt(other.0))
            }
            /// Compare for equality.
            #[inline]
            pub fn le(self, other: Self) -> $bool {
                $bool(self.0.le(other.0))
            }
            /// Compare for equality.
            #[inline]
            pub fn gt(self, other: Self) -> $bool {
                $bool(self.0.gt(other.0))
            }
            /// Compare for equality.
            #[inline]
            pub fn ge(self, other: Self) -> $bool {
                $bool(self.0.ge(other.0))
            }

            /// Extract the value of the `idx`th lane of `self`.
            ///
            /// # Panics
            ///
            /// `extract` will panic if `idx` is out of bounds.
            #[inline]
            pub fn extract(self, idx: u32) -> $elem {
                self.0.extract(idx)
            }
            /// Return a new vector where the `idx`th lane is replaced
            /// by `elem`.
            ///
            /// # Panics
            ///
            /// `replace` will panic if `idx` is out of bounds.
            #[inline]
            pub fn replace(self, idx: u32, elem: $elem) -> Self {
                $name(self.0.replace(idx, elem))
            }

            /// Load a new value from the `idx`th position of `array`.
            ///
            /// This is equivalent to the following, but is possibly
            /// more efficient:
            ///
            /// ```rust,ignore
            /// Self::new(array[idx], array[idx + 1], ...)
            /// ```
            ///
            /// # Panics
            ///
            /// `load` will panic if `idx` is out of bounds in
            /// `array`, or if `array[idx..]` is too short.
            #[inline]
            pub fn load(array: &[$elem], idx: usize) -> Self {
                $name(::SS::$name::load(array, idx))
            }

            /// Store the elements of `self` to `array`, starting at
            /// the `idx`th position.
            ///
            /// This is equivalent to the following, but is possibly
            /// more efficient:
            ///
            /// ```rust,ignore
            /// array[i] = self.extract(0);
            /// array[i + 1] = self.extract(1);
            /// // ...
            /// ```
            ///
            /// # Panics
            ///
            /// `store` will panic if `idx` is out of bounds in
            /// `array`, or if `array[idx...]` is too short.
            #[inline]
            pub fn store(self, array: &mut [$elem], idx: usize) {
                self.0.store(array, idx);
            }
        }

        impl From<SS::$name> for $name {
            fn from(x: SS::$name) -> $name {
                $name(x)
            }
        }

        impl From<$name> for SS::$name {
            fn from(x: $name) -> SS::$name {
                x.0
            }
        }

        )*
    }
}

basic_impls! {
    u32x4: u32, bool32ix4, 4, x0, x1 | x2, x3;
    i32x4: i32, bool32ix4, 4, x0, x1 | x2, x3;
    f32x4: f32, bool32fx4, 4, x0, x1 | x2, x3;

    u16x8: u16, bool16ix8, 8, x0, x1, x2, x3 | x4, x5, x6, x7;
    i16x8: i16, bool16ix8, 8, x0, x1, x2, x3 | x4, x5, x6, x7;

    u8x16: u8, bool8ix16, 16, x0, x1, x2, x3, x4, x5, x6, x7 | x8, x9, x10, x11, x12, x13, x14, x15;
    i8x16: i8, bool8ix16, 16, x0, x1, x2, x3, x4, x5, x6, x7 | x8, x9, x10, x11, x12, x13, x14, x15;
}

macro_rules! bool_impls {
    ($(
        $name: ident:
        $elem: ident, $repr: ident, $repr_elem: ident, $length: expr, $all: ident, $any: ident,
        $($first: ident),* | $($last: ident),*
        [$(#[$cvt_meta: meta] $cvt: ident -> $cvt_to: ident),*];
        )*) => {
        $(impl $name {
            /// Convert to integer representation.
            #[inline]
            pub fn to_repr(self) -> $repr {
                $repr(self.0)
            }
            /// Convert from integer representation.
            #[inline]
            #[inline]
            pub fn from_repr(x: $repr) -> Self {
                $name(x.0)
            }

            /// Create a new instance.
            #[inline]
            pub fn new($($first: bool),*, $($last: bool),*) -> $name {
                $name(
                    SS::$repr::splat(0)
                    -
                    SS::$repr::new(
                        $(($first as $repr_elem)),*,
                        $(($last as $repr_elem)),*))
            }

            /// Create a new instance where every lane has value `x`.
            #[inline]
            pub fn splat(x: bool) -> $name {
                let x = if x {!(0 as $repr_elem)} else {0};
                $name(SS::$repr::splat(x))
            }

            /// Extract the value of the `idx`th lane of `self`.
            ///
            /// # Panics
            ///
            /// `extract` will panic if `idx` is out of bounds.
            #[inline]
            pub fn extract(self, idx: u32) -> bool {
                self.0.extract(idx) != 0
            }
            /// Return a new vector where the `idx`th lane is replaced
            /// by `elem`.
            ///
            /// # Panics
            ///
            /// `replace` will panic if `idx` is out of bounds.
            #[inline]
            pub fn replace(self, idx: u32, elem: bool) -> Self {
                let x = if elem {!(0 as $repr_elem)} else {0};
                $name(self.0.replace(idx, x))
            }
            /// Select between elements of `then` and `else_`, based on
            /// the corresponding element of `self`.
            ///
            /// This is equivalent to the following, but is possibly
            /// more efficient:
            ///
            /// ```rust,ignore
            /// T::new(if self.extract(0) { then.extract(0) } else { else_.extract(0) },
            ///        if self.extract(1) { then.extract(1) } else { else_.extract(1) },
            ///        ...)
            /// ```
            #[inline]
            pub fn select<T: Simd<Bool = $name>>(self, then: T, else_: T) -> T {
                let then: $repr = bitcast(then);
                let else_: $repr = bitcast(else_);
                bitcast((then & self.to_repr()) | (else_ & (!self).to_repr()))
            }

            /// Check if every element of `self` is true.
            ///
            /// This is equivalent to the following, but is possibly
            /// more efficient:
            ///
            /// ```rust,ignore
            /// self.extract(0) && self.extract(1) && ...
            /// ```
            #[inline]
            pub fn all(self) -> bool {
                common::$all(self)
            }
            /// Check if any element of `self` is true.
            ///
            /// This is equivalent to the following, but is possibly
            /// more efficient:
            ///
            /// ```rust,ignore
            /// self.extract(0) || self.extract(1) || ...
            /// ```
            #[inline]
            pub fn any(self) -> bool {
                common::$any(self)
            }

            $(
                #[$cvt_meta]
                #[inline]
                pub fn $cvt(self) -> $cvt_to {
                    bitcast(self)
                }
            )*
        }
          impl ops::Not for $name {
              type Output = Self;

              #[inline]
              fn not(self) -> Self {
                  $name(SS::$repr::splat(!(0 as $repr_elem)) ^ self.0)
              }
          }
      )*
    }
}

bool_impls! {
    bool32ix4: bool32i, i32x4, i32, 4, bool32ix4_all, bool32ix4_any, x0, x1 | x2, x3
        [/// Convert `self` to a boolean vector for interacting with floating point vectors.
         to_f -> bool32fx4];
    bool32fx4: bool32f, i32x4, i32, 4, bool32fx4_all, bool32fx4_any, x0, x1 | x2, x3
        [/// Convert `self` to a boolean vector for interacting with integer vectors.
         to_i -> bool32ix4];

    bool16ix8: bool16i, i16x8, i16, 8, bool16ix8_all, bool16ix8_any, x0, x1, x2, x3 | x4, x5, x6, x7 [];

    bool8ix16: bool8i, i8x16, i8, 16, bool8ix16_all, bool8ix16_any, x0, x1, x2, x3, x4, x5, x6, x7 | x8, x9, x10, x11, x12, x13, x14, x15 [];
}

impl u32x4 {
    /// Convert each lane to a signed integer.
    #[inline]
    pub fn to_i32(self) -> i32x4 {
        i32x4(SS::i32x4::from(self.0))
    }
    /// Convert each lane to a 32-bit float.
    #[inline]
    pub fn to_f32(self) -> f32x4 {
        f32x4(self.0.as_f32x4())
    }
}
impl i32x4 {
    /// Convert each lane to an unsigned integer.
    #[inline]
    pub fn to_u32(self) -> u32x4 {
        u32x4(SS::u32x4::from(self.0))
    }
    /// Convert each lane to a 32-bit float.
    #[inline]
    pub fn to_f32(self) -> f32x4 {
        f32x4(self.0.as_f32x4())
    }
}
impl f32x4 {
    /// Compute the square root of each lane.
    #[inline]
    pub fn sqrt(self) -> Self {
        common::f32x4_sqrt(self)
    }
    /// Compute an approximation to the reciprocal of the square root
    /// of `self`, that is, `f32::splat(1.0) / self.sqrt()`.
    ///
    /// The accuracy of this approximation is platform dependent.
    #[inline]
    pub fn approx_rsqrt(self) -> Self {
        common::f32x4_approx_rsqrt(self)
    }
    /// Compute an approximation to the reciprocal of `self`, that is,
    /// `f32::splat(1.0) / self`.
    ///
    /// The accuracy of this approximation is platform dependent.
    #[inline]
    pub fn approx_reciprocal(self) -> Self {
        common::f32x4_approx_reciprocal(self)
    }
    /// Compute the lane-wise maximum of `self` and `other`.
    ///
    /// This is equivalent to the following, but is possibly more
    /// efficient:
    ///
    /// ```rust,ignore
    /// f32x4::new(self.extract(0).max(other.extract(0)),
    ///            self.extract(1).max(other.extract(1)),
    ///            ...)
    /// ```
    #[inline]
    pub fn max(self, other: Self) -> Self {
        common::f32x4_max(self, other)
    }
    /// Compute the lane-wise minimum of `self` and `other`.
    ///
    /// This is equivalent to the following, but is possibly more
    /// efficient:
    ///
    /// ```rust,ignore
    /// f32x4::new(self.extract(0).min(other.extract(0)),
    ///            self.extract(1).min(other.extract(1)),
    ///            ...)
    /// ```
    #[inline]
    pub fn min(self, other: Self) -> Self {
        common::f32x4_min(self, other)
    }
    /// Convert each lane to a signed integer.
    #[inline]
    pub fn to_i32(self) -> i32x4 {
        i32x4(self.0.as_i32x4())
    }
    /// Convert each lane to an unsigned integer.
    #[inline]
    pub fn to_u32(self) -> u32x4 {
        u32x4(self.0.as_u32x4())
    }
}

impl i16x8 {
    /// Convert each lane to an unsigned integer.
    #[inline]
    pub fn to_u16(self) -> u16x8 {
        u16x8(self.0.as_u16x8())
    }
}
impl u16x8 {
    /// Convert each lane to a signed integer.
    #[inline]
    pub fn to_i16(self) -> i16x8 {
        i16x8(self.0.as_i16x8())
    }
}

impl i8x16 {
    /// Convert each lane to an unsigned integer.
    #[inline]
    pub fn to_u8(self) -> u8x16 {
        u8x16(self.0.as_u8x16())
    }
}
impl u8x16 {
    /// Convert each lane to a signed integer.
    #[inline]
    pub fn to_i8(self) -> i8x16 {
        i8x16(self.0.as_i8x16())
    }
}


macro_rules! neg_impls {
    ($zero: expr, $($ty: ident,)*) => {
        $(impl ops::Neg for $ty {
            type Output = Self;
            fn neg(self) -> Self {
                $ty::splat($zero) - self
            }
        })*
    }
}
neg_impls!{
    0,
    i32x4,
    i16x8,
    i8x16,
}
neg_impls! {
    0.0,
    f32x4,
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
    i32x4,
    i16x8,
    i8x16,
    u32x4,
    u16x8,
    u8x16,
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
        i8x16, u8x16, i16x8, u16x8, i32x4, u32x4,
        f32x4;
    Sub (sub):
        i8x16, u8x16, i16x8, u16x8, i32x4, u32x4,
        f32x4;
    Mul (mul):
        i8x16, u8x16, i16x8, u16x8, i32x4, u32x4,
        f32x4;
    Div (div): f32x4;

    BitAnd (bitand):
        i8x16, u8x16, i16x8, u16x8, i32x4, u32x4,
        bool8ix16, bool16ix8, bool32ix4,
        bool32fx4;
    BitOr (bitor):
        i8x16, u8x16, i16x8, u16x8, i32x4, u32x4,
        bool8ix16, bool16ix8, bool32ix4,
        bool32fx4;
    BitXor (bitxor):
        i8x16, u8x16, i16x8, u16x8, i32x4, u32x4,
        bool8ix16, bool16ix8, bool32ix4,
        bool32fx4;
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
    i8x16, u8x16, i16x8, u16x8, i32x4, u32x4
}
