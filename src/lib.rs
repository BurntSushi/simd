//! `simd` offers a basic interface to the SIMD functionality of CPUs.

#![feature(cfg_target_feature)]
#![allow(non_camel_case_types)]

#[cfg(feature = "with-serde")]
extern crate serde;
#[cfg(feature = "with-serde")]
#[macro_use]
extern crate serde_derive;
extern crate stdsimd;

use stdsimd as SS;

/// Boolean type for 8-bit integers.
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct bool8i(i8);
/// Boolean type for 16-bit integers.
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct bool16i(i16);
/// Boolean type for 32-bit integers.
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct bool32i(i32);
/// Boolean type for 32-bit floats.
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct bool32f(i32);

macro_rules! bool {
    ($($name: ident, $inner: ty;)*) => {
        $(
            impl From<bool> for $name {
                #[inline]
                fn from(b: bool) -> $name {
                    $name(-(b as $inner))
                }
            }
            impl From<$name> for bool {
                #[inline]
                fn from(b: $name) -> bool {
                    b.0 != 0
                }
            }
            )*
    }
}
bool! {
    bool8i, i8;
    bool16i, i16;
    bool32i, i32;
    bool32f, i32;
}

/// Types that are SIMD vectors.
pub unsafe trait Simd {
    /// The corresponding boolean vector type.
    type Bool: Simd;
    /// The element that this vector stores.
    type Elem;
}

/// A SIMD vector of 4 `u32`s.
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct u32x4(SS::u32x4);
/// A SIMD vector of 4 `i32`s.
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct i32x4(SS::i32x4);
/// A SIMD vector of 4 `f32`s.
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct f32x4(SS::f32x4);
/// A SIMD boolean vector for length-4 vectors of 32-bit integers.
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct bool32ix4(SS::i32x4);
/// A SIMD boolean vector for length-4 vectors of 32-bit floats.
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct bool32fx4(SS::i32x4);

#[allow(dead_code)]
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
struct u32x2(SS::u32x2);
#[allow(dead_code)]
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
struct i32x2(SS::i32x2);
#[allow(dead_code)]
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
struct f32x2(SS::f32x2);
#[allow(dead_code)]
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
struct bool32ix2(SS::i32x2);
#[allow(dead_code)]
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
struct bool32fx2(SS::i32x2);

/// A SIMD vector of 8 `u16`s.
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct u16x8(SS::u16x8);
/// A SIMD vector of 8 `i16`s.
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct i16x8(SS::i16x8);
/// A SIMD boolean vector for length-8 vectors of 16-bit integers.
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct bool16ix8(SS::i16x8);

/// A SIMD vector of 16 `u8`s.
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct u8x16(SS::u8x16);
/// A SIMD vector of 16 `i8`s.
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct i8x16(SS::i8x16);
/// A SIMD boolean vector for length-16 vectors of 8-bit integers.
#[cfg_attr(feature = "with-serde", derive(Serialize, Deserialize))]
#[derive(Debug, Copy)]
pub struct bool8ix16(SS::i8x16);


macro_rules! simd {
    ($($bool: ty: $($ty: ty = $elem: ty),*;)*) => {
        $($(unsafe impl Simd for $ty {
            type Bool = $bool;
            type Elem = $elem;
        }
            impl Clone for $ty { #[inline] fn clone(&self) -> Self { *self } }
            )*)*}
}
simd! {
    bool8ix16: i8x16 = i8, u8x16 = u8, bool8ix16 = bool8i;
    bool16ix8: i16x8 = i16, u16x8 = u16, bool16ix8 = bool16i;
    bool32ix4: i32x4 = i32, u32x4 = u32, bool32ix4 = bool32i;
    bool32fx4: f32x4 = f32, bool32fx4 = bool32f;

    bool32ix2: i32x2 = i32, u32x2 = u32, bool32ix2 = bool32i;
    bool32fx2: f32x2 = f32, bool32fx2 = bool32f;
}

#[allow(dead_code)]
#[inline]
fn bitcast<T: Simd, U: Simd>(x: T) -> U {
    assert_eq!(std::mem::size_of::<T>(),
               std::mem::size_of::<U>());
    unsafe {std::mem::transmute_copy(&x)}
}

#[repr(packed)]
#[derive(Debug, Copy, Clone)]
struct Unalign<T>(T);

#[macro_use]
mod common;
mod sixty_four;
mod v256;

#[cfg(any(feature = "doc",
          target_arch = "x86",
          target_arch = "x86_64"))]
pub mod x86;
#[cfg(any(feature = "doc", target_arch = "arm"))]
pub mod arm;
#[cfg(any(feature = "doc", target_arch = "aarch64"))]
pub mod aarch64;
