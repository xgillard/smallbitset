// Copyright 2021 Xavier Gillard
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! This crate provides a series of allocation free set of integers capable of 
//! holding small integer values. These sets are all implemented as bitsets 
//! and are therefore only capable of storing integers with large values.
//!
//! # Warning
//! These collections willingly only target the case of small integer values.
//! Hence they will not try to grow so as to accomodate more values depending
//! on your need. 

/// This macro generates an implementation for the desired types. It simply 
/// avoid code duplication for structs that cannot otherwise be generic.
macro_rules! small_set {
    ($name:ident, $iter:ident, $inner:ty, $capa: expr, $tests:ident) => {
        /// This structure encapsulates a small allocation free set of integers.
        /// Because it is implemented as a fixed size bitset, it can only
        /// accomodate values in the range 0..$capa. 
        ///
        /// Because of this restriction on the range of allowed values, all the
        /// items that can be stored in this set are of type 'u8'. 
        #[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Hash)]
        pub struct $name($inner);

        impl $name {
            /// This method creates an empty set 
            pub fn empty() -> Self { Self(0) }
            /// This method returns the complete set
            pub fn full() -> Self { Self(!0) }
            /// This method creates a singleton set holding the single value 'x'.
            pub fn singleton(x: u8) -> Self { Self(1 << x) }
            /// This method returns the union of two sets
            pub fn union(self, other: Self) -> Self { Self(self.0 | other.0) }
            /// This method returns the intersection of two sets
            pub fn inter(self, other: Self) -> Self { Self(self.0 & other.0) }
            /// This method returns the difference of two sets.
            pub fn diff (self, other: Self) -> Self { Self(self.0 &!other.0) }
            /// This method retuns the complement of the current set
            pub fn complement(self) -> Self { Self(!self.0) }
            /// This method returns the set obtained by adding the singleton x
            /// to the current set
            pub fn insert(&mut self, x: u8) -> Self { 
               self.union(Self::singleton(x))
            }
            /// This method returns the set obtained by removing the singleton 
            /// x from the current set
            pub fn remove(self, x: u8)-> Self {
               self.diff(Self::singleton(x))
            }
            /// Returns true iff the set is empty 
            pub fn is_empty(self) -> bool { 
                self.0 == 0 
            }
            /// Returns the number of items in this set
            pub fn len(self) -> usize { 
                self.0.count_ones() as usize 
            }
            /// Returns true iff the current set contains the given item x
            pub fn contains(self, x: u8) -> bool {
                !self.inter(Self::singleton(x)).is_empty()
            }
            /// Returns the capcity of this set
            pub const fn capacity(self) -> usize {
                $capa as usize
            }
            /// Returns an iterator over the elements of this set
            pub fn iter(self) -> $iter {
                $iter::new(self)
            }
        }
        /// This structure provides a convenient iterator over the items of 
        /// a $name .
        #[derive(Copy, Clone, Debug)]
        pub struct $iter {
            /// The set iterated upon
            set: $name,
            /// The current position in the iteration (aka a cursor)
            pos: u8
        }
        impl $iter {
            /// Creates a new $iter to iterate over a given $name.
            pub fn new(set: $name) -> Self { Self {set, pos: 0 } }
        }
        impl ::std::iter::Iterator for $iter {
            type Item = u8;

            fn next(&mut self) -> Option<Self::Item> {
                while self.pos < $capa {
                    if self.set.contains(self.pos) {
                        let result = Some(self.pos);
                        self.pos += 1;
                        return result;
                    } else {
                        self.pos += 1;
                    }
                }
                None
            }
        }

        impl ::std::iter::IntoIterator for $name {
            type Item = u8;
            type IntoIter = $iter;

            fn into_iter(self) -> Self::IntoIter {
                self.iter()
            }
        }

        impl ::std::fmt::Display for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
                ::std::write!(f, "{{")?;
                let mut it   = self.iter();
                let mut next = it.next();
                while let Some(v) = next {
                    next = it.next();
                    if next.is_some() {
                        ::std::write!(f, "{}, ", v)?;
                    } else {
                        ::std::write!(f, "{}", v)?;
                    }
                }
                ::std::write!(f, "}}")
            }
        }

        impl From<$name> for $inner {
            fn from(x: $name) -> $inner {
                x.0
            }
        }
        impl From<$inner> for $name {
            fn from(x: $inner) -> $name {
                $name(x)
            }
        }

        #[cfg(test)]
        mod $tests {
            use super::*;

            // --- EMPTY SET -------------------------------------------------
            #[test]
            fn emptyset_is_empty() {
                let x = $name::empty();
                assert!(x.is_empty());
            }
            #[test]
            fn emptyset_len() {
                let x = $name::empty();
                assert_eq!(0, x.len());
            }
            #[test]
            fn emptyset_iter_count(){
                let x = $name::empty();
                assert_eq!(0, x.iter().count());
            }
            #[test]
            fn emptyset_contains_no_item() {
                let x = $name::empty();
                for i in 0..$capa {
                    assert!(!x.contains(i));
                }
            }
            #[test]
            fn emptyset_complement() {
                assert_eq!($name::empty().complement(), $name::full());
                assert_eq!($name::empty().complement().complement(), $name::empty());
            }

            // --- FULL SET --------------------------------------------------
            #[test]
            fn fullset_is_not_empty() {
                let x = $name::full();
                assert!(!x.is_empty());
            }
            #[test]
            fn fullset_len() {
                let x = $name::full();
                assert_eq!($capa, x.len());
            }
            #[test]
            fn fullset_iter_count(){
                let x = $name::full();
                assert_eq!($capa, x.iter().count());
            }
            #[test]
            fn fullset_contains_all_items() {
                let x = $name::full();
                for i in 0..$capa {
                    assert!(x.contains(i));
                }
            }
            #[test]
            fn fullset_complement() {
                assert_eq!($name::full().complement(), $name::empty());
                assert_eq!($name::full().complement().complement(), $name::full());
            }
            
            // --- SINGLETON --------------------------------------------------
            #[test]
            fn singleton_is_not_empty() {
                let x = $name::singleton(0);
                assert!(!x.is_empty());
            }
            #[test]
            fn singleton_len() {
                let x = $name::singleton(1);
                assert_eq!(1, x.len());
            }
            #[test]
            fn singleton_iter_count(){
                let x = $name::singleton(2);
                assert_eq!(1, x.iter().count());
            }
            #[test]
            fn singleton_contains_one_single_item() {
                let x = $name::singleton(3);
                for i in 0..$capa {
                    if i == 3 {
                        assert!(x.contains(i));
                    } else {
                        assert!(!x.contains(i));
                    }
                }
            }
            #[test]
            fn singleton_complement() {
                let x = $name::singleton(4);
                assert_eq!(x.complement().complement(), x);
            }

            // --- OTHER METHODS ---------------------------------------------
            #[test]
            fn test_union() {
                let _124   = $name::empty().insert(1).insert(2).insert(4);
                let _035   = $name::empty().insert(0).insert(3).insert(5);

                let _01235 = $name::empty()
                    .insert(0).insert(1).insert(2)
                    .insert(3).insert(4).insert(5);

                assert_eq!(_01235, _124.union(_035));
            }
            #[test]
            fn union_with_is_idempotent() {
                let x = $name::singleton(4);
                assert_eq!(x, x.union(x));
            }
            #[test]
            fn test_inter() {
                let _124   = $name::empty().insert(1).insert(2).insert(4);
                let _025   = $name::empty().insert(0).insert(2).insert(5);

                assert_eq!($name::singleton(2), _124.inter(_025));
            }
            #[test]
            fn diff_removes_existing_item() {
                let _02 = $name::singleton(0).insert(2);
                
                assert_eq!($name::singleton(0), _02.diff($name::singleton(2)));
            }
            #[test]
            fn diff_removes_all_existing_item() {
                let _02  = $name::singleton(0).insert(2);
                let delta= $name::singleton(0).insert(2);
                
                assert_eq!($name::empty(), _02.diff(delta));
            }
            #[test]
            fn diff_leaves_non_existing_items() {
                let _02  = $name::singleton(0).insert(2);
                let delta= $name::singleton(5).insert(6);
                
                assert_eq!(_02, _02.diff(delta));
            }
            #[test]
            fn insert_contains() {
                let mut x = $name::empty();
                assert!(!x.contains(1));
                assert!(!x.contains(2));
                x = x.insert(1);
                assert!(x.contains(1));
                assert!(!x.contains(2));
                // duplicate add has no effect
                x = x.insert(1);
                assert!(x.contains(1));
                assert!(!x.contains(2));
            }
            #[test]
            fn remove_contains() {
                let mut x = $name::singleton(4);
                assert!(!x.contains(1));
                assert!(!x.contains(2));
                assert!(x.contains(4));
                // removing non existing has no effect
                x = x.remove(1);
                assert!(!x.contains(1));
                assert!(!x.contains(2));
                assert!(x.contains(4));
                // removing existing
                x = x.remove(4);
                assert!(!x.contains(1));
                assert!(!x.contains(2));
                assert!(!x.contains(4));
                // duplicate remove has no effect
                x = x.remove(4);
                assert!(!x.contains(1));
                assert!(!x.contains(2));
                assert!(!x.contains(4));
            }
            #[test]
            fn test_iter() {
                let x = $name::singleton(1).insert(2).insert(3);
                let v = x.iter().collect::<Vec<u8>>();
                assert_eq!(vec![1, 2, 3], v);
            }
            #[test]
            fn test_into_iter() {
                let x = $name::singleton(1).insert(2).insert(3);
                let mut v = vec![];
                for i in x {
                    v.push(i);
                }
                assert_eq!(vec![1, 2, 3], v);
            }
            #[test]
            fn test_capacity() {
                assert_eq!($name::empty().capacity(), $capa);
            }


            #[test]
            fn test_display_empty(){
                let set = $name::empty();
                assert_eq!("{}", format!("{}", set));
            }
            #[test]
            fn test_display_singleton(){
                let set = $name::singleton(4);
                assert_eq!("{4}", format!("{}", set));
            }
            #[test]
            fn test_display_multi(){
                let set = $name::singleton(1).insert(4).insert(6);
                assert_eq!("{1, 4, 6}", format!("{}", set));
            }
        }
    }
}

// This is where we declare and define the structures for the various types of
// sets.
small_set!(Set8,    Set8Iter,    u8,      8, test_8);
small_set!(Set16,   Set16Iter,   u16,    16, test_16);
small_set!(Set32,   Set32Iter,   u32,    32, test_32);
small_set!(Set64,   Set64Iter,   u64,    64, test_64);
small_set!(Set128,  Set128Iter,  u128,  128, test128);

mod custom;
pub use custom::*;

pub type MutSet8   = BitSet<1>;
pub type MutSet16  = BitSet<2>;
pub type MutSet32  = BitSet<4>;
pub type MutSet64  = BitSet<8>;
pub type MutSet128 = BitSet<16>;
pub type MutSet256 = BitSet<32>;