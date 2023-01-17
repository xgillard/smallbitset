use std::{ops::{Shl, ShlAssign}, fmt::Debug};

use num::{PrimInt, Unsigned};

#[macro_export]
macro_rules! bitset {
    ($name:ident, $capa:expr, $block_size:expr, $block_type:ty) => {
        /// This structure implemts a bitset with a maximum capacity of 
        /// $capa bits. The structure requires no dynamic allocation and it is
        /// therefore fully copiable
        #[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
        pub struct $name {
            /// the datablocks
            blocks: [$block_type; Self::nb_blocks($capa, $block_size)]
        }

        impl $name {
            /// Creates an empty set
            pub const fn empty() -> Self {
                Self {blocks: [0; Self::nb_blocks($capa, $block_size)]}
            }
            /// Creates a set where all items are present
            pub const fn full() -> Self {
                Self {blocks: [!0; Self::nb_blocks($capa, $block_size)]}
            }
            /// Creates a singleton set comprising one single item
            pub const fn singleton(x: usize) -> Self {
                Self::empty().add(x)
            }
            /// Returns the capacity of the bitset
            pub const fn capacity(self) -> usize {
                $capa
            }
            /// Returns the lenght of the set. That is the number of itemes present
            /// (value of the bit is 1) in the set
            pub fn len(self) -> usize {
                self.blocks.iter().map(|block| block.count_ones() as usize).sum()
            }
            /// Returns true iff the set is empty
            pub fn is_empty(self) -> bool {
                self.blocks.iter().all(|block| *block == 0)
            }
            /// Returns true iff the set contains item 'x'
            pub const fn contains(self, x: usize) -> bool {
                let block = self.block(x);
                let pos   = self.pos_in_block(x);
                let mask  = 1 << pos;
                (self.blocks[block] & mask) != 0
            }
            /// Returns a new set where item x is present in the set
            pub const fn add(mut self, x: usize) -> Self {
                let block           = self.block(x);
                let pos             = self.pos_in_block(x);
                let mask            = 1 << pos;
                self.blocks[block] |= mask;
                self
            }
            /// Adds the item x to this set
            pub fn add_inplace(&mut self, x: usize) -> &mut Self {
                let block           = self.block(x);
                let pos             = self.pos_in_block(x);
                let mask            = 1 << pos;
                self.blocks[block] |= mask;
                self
            }
            /// Returns a new set where item x is absent from the set
            pub const fn remove(mut self, x: usize) -> Self {
                let block           = self.block(x);
                let pos             = self.pos_in_block(x);
                let mask            = 1 << pos;
                self.blocks[block] &= !mask;
                self
            }
            /// Removes x from the current set
            pub fn remove_inplace(&mut self, x: usize) -> &mut Self {
                let block           = self.block(x);
                let pos             = self.pos_in_block(x);
                let mask            = 1 << pos;
                self.blocks[block] &= !mask;
                self
            }
            /// Returns the union of both sets
            pub fn union(mut self, other: Self) -> Self {
                for (block, otherblock) in self.blocks.iter_mut().zip(other.blocks.iter()) {
                    *block |= *otherblock;
                }
                self
            }
            /// Updates this set so that it contains the union of self and other
            pub fn union_inplace(&mut self, other: &Self) -> &mut Self {
                for (block, otherblock) in self.blocks.iter_mut().zip(other.blocks.iter()) {
                    *block |= *otherblock;
                }
                self
            }
            /// Returns a set which is the intersection of both sets
            pub fn inter(mut self, other: Self) -> Self {
                for (block, otherblock) in self.blocks.iter_mut().zip(other.blocks.iter()) {
                    *block &= *otherblock;
                }
                self
            }
            /// Updates this set so that it contains the intersection of self and other
            pub fn inter_inplace(&mut self, other: &Self) -> &mut Self {
                for (block, otherblock) in self.blocks.iter_mut().zip(other.blocks.iter()) {
                    *block &= *otherblock;
                }
                self
            }
            /// Returns a set which is the difference of the two sets
            pub fn diff(mut self, other: Self) -> Self {
                for (block, otherblock) in self.blocks.iter_mut().zip(other.blocks.iter()) {
                    *block &= !*otherblock;
                }
                self
            }
            /// Updates this set so that it contains the difference of self and other
            pub fn diff_inplace(&mut self, other: &Self) -> &mut Self {
                for (block, otherblock) in self.blocks.iter_mut().zip(other.blocks.iter()) {
                    *block &= !*otherblock;
                }
                self
            }
            /// Returns a set which is the exclusive or of the two sets
            pub fn symmetric_difference(mut self, other: Self) -> Self {
                for (block, otherblock) in self.blocks.iter_mut().zip(other.blocks.iter()) {
                    *block ^= *otherblock;
                }
                self
            }
            /// Updates this set so that it contains the exclusive or of self and other
            pub fn symmetric_difference_inplace(&mut self, other: &Self) -> &mut Self {
                for (block, otherblock) in self.blocks.iter_mut().zip(other.blocks.iter()) {
                    *block ^= *otherblock;
                }
                self
            }
            /// Flips all the bits of self (all members are removed, all absent are added)
            pub fn flip(mut self) -> Self {
                for block in self.blocks.iter_mut() {
                    *block = !*block;
                }
                self
            }
            /// Updates this set so that it contains the negation of self
            pub fn flip_inplace(&mut self) -> &mut Self {
                for block in self.blocks.iter_mut() {
                    *block = !*block;
                }
                self
            }
            /// Returns true iff this set contains all the elements of the other set.
            /// (self is a superset of other).
            pub fn contains_all(self, other: Self) -> bool {
                self.blocks.iter().zip(other.blocks.iter())
                    .all(|(x, y)| (*x & *y) == *y)
            }
            /// Return true iff the two sets are disjoint
            pub fn disjoint(self, other: Self) -> bool {
                self.blocks.iter().zip(other.blocks.iter())
                    .all(|(x, y)| (*x & *y) == 0)
            }
            /// Returns true iff the intersects with other set
            pub fn intersects(&self, other: Self) -> bool {
                !self.disjoint(other)
            }
            /// Returns an iterator that goes over all the items present in this set
            pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
                self.ones()
            }
            /// Returns an iterator that goes over all the items present in this set
            pub fn ones(&self) -> impl Iterator<Item = usize> + '_ {
                let iter = self.blocks.iter().copied();
                $crate::BitsBuilder::<$block_size>::build(iter)
            }
            /// Returns an iterator that goes over all the items absent from this set
            pub fn zeroes(&self) -> impl Iterator<Item = usize> + '_ {
                let iter = self.blocks.iter().copied().map(|x| !x);
                $crate::BitsBuilder::<$block_size>::build(iter)
            }
            /// Returns an iterator that goes over all the items absent from this set
            pub fn subsets(&self, of_size: usize) -> impl Iterator<Item = Self> + '_ {
                let iter = self.ones();
                $crate::SubsetsOfSize::<$name>::new(of_size, iter)
            }

            /// Returns the next value when considering the bitset as a large integer value 
            /// and incrementing it by one.
            pub const fn inc(mut self) -> Self {
                let mut i = 0;
                let mut cont = true;

                while cont {
                    let (block, carry) = self.blocks[i].overflowing_add(1);
                    self.blocks[i] = block;
                    i += 1;
                    cont = carry;
                }
                self
            }

            /// Consider the bitset as a large integer value and increments it by one
            pub fn inc_inplace(&mut self) -> &mut Self {
                for b in self.blocks.iter_mut() {
                    let (block, carry) = b.overflowing_add(1);
                    *b = block;
                    if !carry { break }
                }
                self
            }

            /// Utility function that returns the number of blocks to use in this structure
            const fn nb_blocks(capa: usize, block_sz: usize) -> usize {
                capa / block_sz
            }
            /// Returns the index of the block that can contain the item x
            const fn block(self, x: usize) -> usize {
                x / $block_size
            }
            /// Return the position of item x in its block
            const fn pos_in_block(self, x: usize) -> usize {
                x % $block_size
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(fmt, "{{")?;
                let mut first = true;
                
                for x in self.ones() {
                    if first {
                        first = false;
                    } else {
                        write!(fmt, ", ")?;
                    }
                    write!(fmt, "{}", x)?;
                }

                write!(fmt, "}}")
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::empty()
            }
        }

        impl std::ops::BitAnd for $name {
            type Output = Self;

            fn bitand(self, rhs: Self) -> Self::Output {
                self.inter(rhs)
            }
        }
        impl std::ops::BitAndAssign for $name {
            fn bitand_assign(&mut self, rhs: Self) {
                self.inter_inplace(&rhs);
            }
        }
        impl std::ops::BitOr for $name {
            type Output = Self;

            fn bitor(self, rhs: Self) -> Self::Output {
                self.union(rhs)
            }
        }
        impl std::ops::BitOrAssign for $name {
            fn bitor_assign(&mut self, rhs: Self) {
                self.union_inplace(&rhs);
            }
        }
        impl std::ops::BitXor for $name {
            type Output = Self;

            fn bitxor(self, rhs: Self) -> Self::Output {
                self.symmetric_difference(rhs)
            }
        }
        impl std::ops::BitXorAssign for $name {
            fn bitxor_assign(&mut self, rhs: Self) {
                self.symmetric_difference_inplace(&rhs);
            }
        }
        impl std::ops::Not for $name {
            type Output = Self;

            fn not(self) -> Self::Output {
                self.flip()   
            }
        }

        impl $crate::SubsetsOfSize<$name> {
            /// This function is used to remap a smaller bitset (one that only account
            /// for the lowest possible set of values only) onto the set of values that
            /// were initially given.
            fn remap(&self, bs: $name) -> $name {
                let mut out = $name::empty();
                for b in bs.iter() {
                    out.add_inplace(self.mapping[b]);
                }
                out
            }
        }
        
        impl $crate::SubsetsOfSize<$name> {
            /// Creates a new instance
            pub fn new(k: usize, among: impl Iterator<Item = usize>) -> Self {
                let mapping = among.collect::<Vec<_>>();
                let combinations = $crate::nb_combinations(k, mapping.len());
                
                let mut current = $name::default();
                for i in 0..k {
                    current.add_inplace(i);
                }
        
                Self {
                    current,
                    of_size: k,
                    counted: 0,
                    max_count: combinations,
                    mapping
                }
            }
        }

        impl Iterator for $crate::SubsetsOfSize<$name> {
            type Item = $name;

            fn next(&mut self) -> Option<Self::Item> {
                let mut found = None;
                while found.is_none() && self.counted < self.max_count {
                    if self.current.len() == self.of_size {
                        self.counted += 1;
                        found = Some(self.remap(self.current));
                    }
                    self.current.inc_inplace();
                }
                found
            }
        }
    };
}

/// This structure is really only meant to facilitate the writing
/// of iterator methods on bitsets. This way, only the const param
/// can be specified while leaving all the other types inferred by 
/// the compiler. (That's kind of type curry-ing).
#[derive(Debug, Clone, Copy)]
pub struct BitsBuilder<const BITS: u32>;
impl <const BITS: u32> BitsBuilder<BITS> {
    /// Creates a new iterator over the bits of the given blocks
    pub fn build<T, I>(iter: I) -> Bits<BITS, T, I> 
    where T: PrimInt + Unsigned + Shl + ShlAssign, 
          I: Iterator<Item = T>
    {
        Bits::new(iter)
    }
}

/// This iterator goes over all the bits whose value are 1 in a biset
#[derive(Debug, Clone)]
pub struct Bits<const BITS: u32, T, I> 
where T: PrimInt + Unsigned + Shl + ShlAssign, 
      I: Iterator<Item = T>
{
    /// The current block
    head: Option<T>,
    /// The rest of the blocks that haven't been touched yet
    tail: I,
    /// The portion of the value of the next item dependent 
    /// of the position of the current block
    base: u32,
    /// The offset of the next tested item in its block
    offset : u32, 
    /// The number of items that remain to cover in the current block
    /// 
    /// # Note
    /// This seemed like a smart thing to do since the LLVM assembly
    /// has a decicated primitive 'ctpop' whose sole purpose is to 
    /// count the on bits in an integer value.
    rem : u32,
    /// A mask that targets the test of the next bit in the current block
    mask: T,
}
impl <const BITS: u32, T, I> Bits<BITS, T, I>
where T: PrimInt + Unsigned + Shl + ShlAssign, 
      I: Iterator<Item = T>
{
    /// Create a new iterator on the bits present in the integer blocks
    /// that are iterated over by 'it'.
    pub fn new(mut it: I) -> Self {
        let head = it.next();
        let tail = it;
        let mask = T::one();
        let base = 0;
        let offset = 0;
        let rem = if let Some(x) = head { x.count_ones() } else { 0 };

        Self { head, tail, base, offset, rem, mask }
    }
}
impl <const BITS: u32, T, I> Iterator for Bits<BITS, T, I>
where T: PrimInt + Unsigned + Shl + ShlAssign, 
      I: Iterator<Item = T>
{
    type Item = usize;

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, tail_ub) = self.tail.size_hint();
        let lb = self.rem as usize;
        let ub = tail_ub.map(|x| x * BITS as usize + lb);

        (lb, ub)
    }

    fn next(&mut self) -> Option<Self::Item> {
        while self.head.is_some() {
            if self.rem == 0 {
                self.head   = self.tail.next();
                self.mask   = T::one();
                self.base  += BITS;
                self.offset = 0;
                self.rem    = if let Some(x) = self.head { x.count_ones() } else { 0 }; 
            } else {
                break;
            }
        }
        //
        if let Some(x) = self.head {
            // we can be sure that x is a non empty block
            while (x & self.mask).is_zero() {
                self.offset += 1;
                self.mask  <<= T::one();
            }
            
            // this is the result of our computation
            let result = self.base + self.offset;
            
            // move to next bit
            self.rem -= 1;
            self.offset += 1;
            self.mask  <<= T::one();
            
            Some(result as usize)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
/// This is an iterator that will iterate over all the subsets of a given size
/// of a given bitset
pub struct SubsetsOfSize<T> {
    /// the current value of the bitset considered as a large integer
    pub current: T,
    /// the size of the subsets we want to enumerate
    pub of_size: usize,
    /// the number of subsets we have counted already
    pub counted: usize,
    /// the maximum number of such sets that can be generated
    pub max_count: usize,
    /// a mapping between the original values and those in the smaller subset which
    /// is being used to generate values during iteration
    pub mapping: Vec<usize>
}

/// This will count the number of possible cases. Given that we want to 
/// draw k items among a set of n, without considering the repetitions
/// and without considerations for the order, we are computing the number
/// of possible combinations $C^n_k = \frac{n!}{k! (n-k)!}$. 
pub fn nb_combinations(k: usize, among_n: usize) -> usize {
    if k == among_n { 
        1 
    } else {
        let mut num  = 1;
        let mut denom = 1;

        for i in 1..=k {
            denom *= i;
            num   *= among_n - (i - 1);
        }

        num / denom
    }
}

bitset!(Set8,     8,   8, u8);
bitset!(Set16,   16,  16, u16);
bitset!(Set32,   32,  32, u32);
bitset!(Set64,   64,  64, u64);
bitset!(Set128, 128, 128, u128);
bitset!(Set256, 256, 128, u128);

macro_rules! test {
    ($name: ident, $capa: expr) => {
        paste::paste! {
            #[cfg(test)]
            mod [<test_ $name:snake:lower>] {
                use super::*;

                #[test]
                fn default_is_emptyset() {
                    let x = $name::default();
                    assert!(x.is_empty());
                }

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
                fn emptyset_ones_count(){
                    let x = $name::empty();
                    assert_eq!(0, x.ones().count());
                }
                #[test]
                fn emptyset_zeroes_count(){
                    let x = $name::empty();
                    assert_eq!($capa, x.zeroes().count());
                }
                #[test]
                fn emptyset_contains_no_item() {
                    let x = $name::empty();
                    for i in 0..$capa {
                        assert!(!x.contains(i));
                    }
                }
                #[test]
                fn emptyset_flip() {
                    assert_eq!($name::empty().flip(), $name::full());
                    assert_eq!($name::empty().flip().flip(), $name::empty());
                }
                #[test]
                fn emptyset_flip_inplace() {
                    assert_eq!($name::empty().flip_inplace(), &$name::full());
                    assert_eq!($name::empty().flip_inplace().flip_inplace(), &$name::empty());
                }
                #[test]
                fn emptyset_not() {
                    assert_eq!(!$name::empty(), $name::full());
                    assert_eq!(!!$name::empty(), $name::empty());
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
                fn fullset_ones_count(){
                    let x = $name::full();
                    assert_eq!($capa, x.ones().count());
                }
                #[test]
                fn fullset_zeroes_count(){
                    let x = $name::full();
                    assert_eq!(0, x.zeroes().count());
                }
                #[test]
                fn fullset_contains_all_items() {
                    let x = $name::full();
                    for i in 0..$capa {
                        assert!(x.contains(i));
                    }
                }
                #[test]
                fn fullset_flip() {
                    assert_eq!($name::full().flip(), $name::empty());
                    assert_eq!($name::full().flip().flip(), $name::full());
                }
                #[test]
                fn fullset_flip_inplace() {
                    assert_eq!($name::full().flip_inplace(), &$name::empty());
                    assert_eq!($name::full().flip_inplace().flip_inplace(), &$name::full());
                }
                #[test]
                fn fullset_not() {
                    assert_eq!(!$name::full(), $name::empty());
                    assert_eq!(!!$name::full(), $name::full());
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
                fn singleton_ones_count(){
                    let x = $name::singleton(2);
                    assert_eq!(1, x.ones().count());
                }
                #[test]
                fn singleton_zeroes_count(){
                    let x = $name::singleton(2);
                    assert_eq!($capa - 1, x.zeroes().count());
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
                fn singleton_flip() {
                    let x = $name::singleton(4);
                    assert_eq!(x.flip().flip(), x);
                }
                #[test]
                fn singleton_flip_inplace() {
                    let mut x = $name::singleton(4);
                    let y = x;
                    assert_eq!(x.flip_inplace().flip_inplace(), &y);
                }
                #[test]
                fn singleton_not() {
                    let x = $name::singleton(4);
                    assert_eq!(!!x, x);
                }

                // --- OTHER METHODS ---------------------------------------------
                #[test]
                fn test_add() {
                    let a = $name::empty();
                    let b = a.add(4);
                    assert_eq!(b, $name::singleton(4));

                    // adding the same item has no effect
                    let b = b.add(4);
                    assert_eq!(b, $name::singleton(4));
                    let b = b.add(4);
                    assert_eq!(b, $name::singleton(4));
                    let b = b.add(4);
                    assert_eq!(b, $name::singleton(4));

                    assert_eq!(1, b.len());
                }
                #[test]
                fn test_add_inplace() {
                    let mut a = $name::empty();
                    a.add_inplace(4);
                    assert_eq!(a, $name::singleton(4));

                    // adding the same item has no effect
                    a.add_inplace(4);
                    assert_eq!(a, $name::singleton(4));
                    a.add_inplace(4);
                    assert_eq!(a, $name::singleton(4));
                    a.add_inplace(4);
                    assert_eq!(a, $name::singleton(4));

                    assert_eq!(1, a.len());
                }
                #[test]
                fn test_remove() {
                    let a = $name::singleton(4);
                    let b = a.remove(4);
                    assert_eq!(b, $name::empty());

                    // removing absent item has no effect
                    let b = b.remove(4);
                    assert_eq!(b, $name::empty());
                    let b = b.remove(4);
                    assert_eq!(b, $name::empty());
                    let b = b.remove(4);
                    assert_eq!(b, $name::empty());

                    assert_eq!(0, b.len());
                }
                #[test]
                fn test_remove_inplace() {
                    let mut a = $name::singleton(4);
                    a.remove_inplace(4);
                    assert_eq!(a, $name::empty());

                    // removing absent item has no effect
                    a.remove_inplace(4);
                    assert_eq!(a, $name::empty());
                    a.remove_inplace(4);
                    assert_eq!(a, $name::empty());
                    a.remove_inplace(4);
                    assert_eq!(a, $name::empty());

                    assert_eq!(0, a.len());
                }
                #[test]
                fn test_union() {
                    let _124   = $name::empty().add(1).add(2).add(4);
                    let _035   = $name::empty().add(0).add(3).add(5);

                    let _01235 = $name::empty()
                        .add(0).add(1).add(2)
                        .add(3).add(4).add(5);

                    assert_eq!(_01235, _124.union(_035));
                }
                #[test]
                fn test_bitor() {
                    let _124   = $name::empty().add(1).add(2).add(4);
                    let _035   = $name::empty().add(0).add(3).add(5);

                    let _01235 = $name::empty()
                        .add(0).add(1).add(2)
                        .add(3).add(4).add(5);

                    assert_eq!(_01235, _124 | _035);
                }
                #[test]
                fn test_union_inplace() {
                    let mut _124 = $name::empty().add(1).add(2).add(4);
                    let _035 = $name::empty().add(0).add(3).add(5);

                    let _01235 = $name::empty()
                        .add(0).add(1).add(2)
                        .add(3).add(4).add(5);

                    assert_eq!(&_01235, _124.union_inplace(&_035));
                }
                #[test]
                fn test_bitor_assign() {
                    let mut _124   = $name::empty().add(1).add(2).add(4);
                    let _035   = $name::empty().add(0).add(3).add(5);

                    let _01235 = $name::empty()
                        .add(0).add(1).add(2)
                        .add(3).add(4).add(5);

                    _124 |= _035;
                    assert_eq!(_01235, _124);
                }
                #[test]
                fn union_with_self_is_idempotent() {
                    let x = $name::singleton(4);
                    assert_eq!(x, x.union(x));
                }
                #[test]
                fn test_inter() {
                    let _124   = $name::empty().add(1).add(2).add(4);
                    let _025   = $name::empty().add(0).add(2).add(5);

                    assert_eq!($name::singleton(2), _124.inter(_025));
                }
                #[test]
                fn test_inter_inplace() {
                    let mut _124 = $name::empty().add(1).add(2).add(4);
                    let _025 = $name::empty().add(0).add(2).add(5);
                    _124.inter_inplace(&_025);
                    assert_eq!($name::singleton(2), _124);
                }
                #[test]
                fn test_bitand() {
                    let _124   = $name::empty().add(1).add(2).add(4);
                    let _025   = $name::empty().add(0).add(2).add(5);

                    assert_eq!($name::singleton(2), _124 & _025);
                }
                #[test]
                fn test_bitand_assign() {
                    let mut _124 = $name::empty().add(1).add(2).add(4);
                    let _025 = $name::empty().add(0).add(2).add(5);
                    _124 &= _025;
                    assert_eq!($name::singleton(2), _124);
                }
                #[test]
                fn diff_removes_existing_item() {
                    let _02 = $name::singleton(0).add(2);
                    
                    assert_eq!($name::singleton(0), _02.diff($name::singleton(2)));
                }
                #[test]
                fn diff_inplace_removes_existing_item() {
                    let mut _02 = $name::singleton(0).add(2);
                    
                    assert_eq!(&$name::singleton(0), _02.diff_inplace(&$name::singleton(2)));
                }
                #[test]
                fn diff_removes_all_existing_item() {
                    let _02  = $name::singleton(0).add(2);
                    let delta= $name::singleton(0).add(2);
                    
                    assert_eq!($name::empty(), _02.diff(delta));
                }
                #[test]
                fn diff_inplace_removes_all_existing_item() {
                    let mut _02  = $name::singleton(0).add(2);
                    let delta= $name::singleton(0).add(2);
                    
                    assert_eq!(&$name::empty(), _02.diff_inplace(&delta));
                }
                #[test]
                fn diff_leaves_non_existing_items() {
                    let _02  = $name::singleton(0).add(2);
                    let delta= $name::singleton(5).add(6);
                    
                    assert_eq!(_02, _02.diff(delta));
                }
                #[test]
                fn diff_inplace_leaves_non_existing_items() {
                    let _02   = $name::singleton(0).add(2);
                    let mut alpha = $name::singleton(0).add(2);
                    let delta = $name::singleton(5).add(6);
                    
                    assert_eq!(&_02, alpha.diff_inplace(&delta));
                }


                #[test]
                fn symdiff_removes_existing_item() {
                    let _02 = $name::singleton(0).add(2);
                    
                    assert_eq!($name::singleton(0), _02.symmetric_difference($name::singleton(2)));
                }
                #[test]
                fn symdiff_inplace_removes_existing_item() {
                    let mut _02 = $name::singleton(0).add(2);
                    
                    assert_eq!(&$name::singleton(0), _02.symmetric_difference_inplace(&$name::singleton(2)));
                }
                #[test]
                fn bitxor_removes_existing_item() {
                    let _02 = $name::singleton(0).add(2);
                    
                    assert_eq!($name::singleton(0), _02 ^ $name::singleton(2));
                }
                #[test]
                fn bitxorassign_removes_existing_item() {
                    let mut _02 = $name::singleton(0).add(2);
                    _02 ^= $name::singleton(2);
                    assert_eq!($name::singleton(0), _02);
                }

                #[test]
                fn symdiff_removes_all_existing_item() {
                    let _02  = $name::singleton(0).add(2);
                    let delta= $name::singleton(0).add(2);
                    
                    assert_eq!($name::empty(), _02.symmetric_difference(delta));
                }
                #[test]
                fn bitxor_removes_all_existing_item() {
                    let _02  = $name::singleton(0).add(2);
                    let delta= $name::singleton(0).add(2);
                    
                    assert_eq!($name::empty(), _02 ^ delta);
                }
                #[test]
                fn symdiff_inplace_removes_all_existing_item() {
                    let mut _02  = $name::singleton(0).add(2);
                    let delta= $name::singleton(0).add(2);
                    
                    assert_eq!(&$name::empty(), _02.symmetric_difference_inplace(&delta));
                }
                #[test]
                fn bitxor_assign_removes_all_existing_items() {
                    let mut _02  = $name::singleton(0).add(2);
                    let delta= $name::singleton(0).add(2);
                    _02 ^= delta;
                    assert_eq!($name::empty(), _02);
                }
                #[test]
                fn symdiff_incorporates_non_existing_items() {
                    let _02  = $name::singleton(0).add(2);
                    let delta= $name::singleton(5).add(6);
                    
                    assert_eq!(_02.union(delta), _02.symmetric_difference(delta));
                }
                #[test]
                fn bitxor_incorporates_non_existing_items() {
                    let _02  = $name::singleton(0).add(2);
                    let delta= $name::singleton(5).add(6);
                    
                    assert_eq!(_02.union(delta), _02 ^ delta);
                }
                #[test]
                fn symdiff_inplace_leaves_non_existing_items() {
                    let _02   = $name::singleton(0).add(2);
                    let mut alpha = $name::singleton(0).add(2);
                    let delta = $name::singleton(5).add(6);
                    
                    assert_eq!(&_02.union(delta), alpha.symmetric_difference_inplace(&delta));
                }
                #[test]
                fn bitxorassign_leaves_non_existing_items() {
                    let _02   = $name::singleton(0).add(2);
                    let mut alpha = $name::singleton(0).add(2);
                    let delta = $name::singleton(5).add(6);
                    alpha ^= delta;
                    assert_eq!(_02.union(delta), alpha);
                }

                #[test]
                fn add_contains() {
                    let mut x = $name::empty();
                    assert!(!x.contains(1));
                    assert!(!x.contains(2));
                    x = x.add(1);
                    assert!(x.contains(1));
                    assert!(!x.contains(2));
                    // duplicate add has no effect
                    x = x.add(1);
                    assert!(x.contains(1));
                    assert!(!x.contains(2));
                }
                #[test]
                fn add_inplace_contains() {
                    let mut x = $name::empty();
                    assert!(!x.contains(1));
                    assert!(!x.contains(2));
                    x.add_inplace(1);
                    assert!(x.contains(1));
                    assert!(!x.contains(2));
                    // duplicate add has no effect
                    x.add_inplace(1);
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
                fn remove_inplace_contains() {
                    let mut x = $name::singleton(4);
                    assert!(!x.contains(1));
                    assert!(!x.contains(2));
                    assert!(x.contains(4));
                    // removing non existing has no effect
                    x.remove_inplace(1);
                    assert!(!x.contains(1));
                    assert!(!x.contains(2));
                    assert!(x.contains(4));
                    // removing existing
                    x.remove_inplace(4);
                    assert!(!x.contains(1));
                    assert!(!x.contains(2));
                    assert!(!x.contains(4));
                    // duplicate remove has no effect
                    x.remove_inplace(4);
                    assert!(!x.contains(1));
                    assert!(!x.contains(2));
                    assert!(!x.contains(4));
                }
                #[test]
                fn test_capacity() {
                    assert_eq!($name::empty().capacity(), $capa);
                }

                // --- ITERATORS -----------------------
                #[test]
                fn test_iter() {
                    let x = $name::singleton(1).add(2).add(3);
                    let v = x.iter().collect::<Vec<_>>();
                    assert_eq!(vec![1, 2, 3], v);
                }
                #[test]
                fn test_ones() {
                    let x = $name::singleton(1).add(2).add(3);
                    let v = x.ones().collect::<Vec<_>>();
                    assert_eq!(vec![1, 2, 3], v);
                }
                #[test]
                fn test_zeroes() {
                    let x = $name::singleton(1).add(2).add(3).flip();
                    let v = x.zeroes().collect::<Vec<_>>();
                    assert_eq!(vec![1, 2, 3], v);
                }

                #[test]
                fn test_subsets_small() {
                    let x = $name::singleton(1).add(2).add(3).add(4).add(5);
                    let v = x.subsets(3).collect::<Vec<_>>();

                    let r = vec![
                        $name::empty().add(1).add(2).add(3), // 00111
                        $name::empty().add(1).add(2).add(4), // 01011
                        $name::empty().add(1).add(3).add(4), // 01101
                        $name::empty().add(2).add(3).add(4), // 01110
                        $name::empty().add(1).add(2).add(5), // 10011
                        $name::empty().add(1).add(3).add(5), // 10101
                        $name::empty().add(2).add(3).add(5), // 10110
                        $name::empty().add(1).add(4).add(5), // 11001
                        $name::empty().add(2).add(4).add(5), // 11010
                        $name::empty().add(3).add(4).add(5), // 11100
                    ];
                    assert_eq!(r, v);
                }
                #[test]
                fn test_subsets_zero() {
                    let x = $name::singleton(1).add(2).add(3).add(4).add(5);
                    let v = x.subsets(0).collect::<Vec<_>>();

                    let r = vec![
                        $name::empty()
                    ];

                    assert_eq!(r, v);
                }
                #[test]
                fn test_subsets_full() {
                    let x = $name::full();
                    let v = x.subsets($capa).collect::<Vec<_>>();

                    let r = vec![
                        $name::full()
                    ];
                    
                    assert_eq!(r, v);
                }

                #[test]
                fn fullset_contains_all_empty() {
                    let x = $name::full();
                    let y = $name::empty();

                    assert!(x.contains_all(y));
                }
                #[test]
                fn fullset_contains_all_fullset() {
                    let x = $name::full();
                    let y = $name::full();

                    assert!(x.contains_all(y));
                }
                #[test]
                fn fullset_contains_all_singleton() {
                    let x = $name::full();
                    
                    let y = $name::singleton(0);
                    assert!(x.contains_all(y));

                    let y = $name::singleton(2);
                    assert!(x.contains_all(y));

                    let y = $name::singleton(3);
                    assert!(x.contains_all(y));

                    let y = $name::singleton(4);
                    assert!(x.contains_all(y));

                    let y = $name::singleton(5);
                    assert!(x.contains_all(y));

                    let y = $name::singleton(6);
                    assert!(x.contains_all(y));

                    let y = $name::singleton(7);
                    assert!(x.contains_all(y));
                }

                #[test]
                fn fullset_contains_all() {
                    let x = $name::full();
                    let mut y = $name::singleton(0);
                    y.add_inplace(1);
                    y.add_inplace(2);
                    y.add_inplace(3);
                    y.add_inplace(4);
                    y.add_inplace(5);
                    y.add_inplace(6);
                    y.add_inplace(7);

                    assert!(x.contains_all(y));
                }

                #[test]
                fn emptyset_contains_all_empty() {
                    let x = $name::empty();
                    let y = $name::empty();

                    assert!(x.contains_all(y));
                }
                #[test]
                fn emptyset_contains_all_fullset() {
                    let x = $name::empty();
                    let y = $name::full();

                    assert!(!x.contains_all(y));
                }
                #[test]
                fn emptyset_contains_all_singleton() {
                    let x = $name::empty();
                    
                    let y = $name::singleton(0);
                    assert!(!x.contains_all(y));

                    let y = $name::singleton(2);
                    assert!(!x.contains_all(y));

                    let y = $name::singleton(3);
                    assert!(!x.contains_all(y));

                    let y = $name::singleton(4);
                    assert!(!x.contains_all(y));

                    let y = $name::singleton(5);
                    assert!(!x.contains_all(y));

                    let y = $name::singleton(6);
                    assert!(!x.contains_all(y));

                    let y = $name::singleton(7);
                    assert!(!x.contains_all(y));
                }

                #[test]
                fn emptyset_contains_all() {
                    let x = $name::empty();
                    let mut y = $name::singleton(0);
                    y.add_inplace(1);
                    y.add_inplace(2);
                    y.add_inplace(7);

                    assert!(!x.contains_all(y));
                }

                #[test]
                fn contains_all_empty() {
                    let mut x = $name::empty();
                    x.add_inplace(1);
                    x.add_inplace(2);
                    x.add_inplace(3);
                    x.add_inplace(7);

                    let y = $name::empty();
                    assert!(x.contains_all(y));
                }
                #[test]
                fn contains_all_fullset() {
                    let mut x = $name::empty();
                    x.add_inplace(2);
                    x.add_inplace(3);
                    x.add_inplace(4);
                    x.add_inplace(7);

                    let y = $name::full();
                    assert!(!x.contains_all(y));
                }
                #[test]
                fn contains_all_singleton() {
                    let mut x = $name::empty();
                    x.add_inplace(1);
                    x.add_inplace(2);
                    x.add_inplace(3);
                    x.add_inplace(4);
                    
                    let y = $name::singleton(0);
                    assert!(!x.contains_all(y));

                    let y = $name::singleton(1);
                    assert!(x.contains_all(y));

                    let y = $name::singleton(2);
                    assert!(x.contains_all(y));

                    let y = $name::singleton(3);
                    assert!(x.contains_all(y));

                    let y = $name::singleton(4);
                    assert!(x.contains_all(y));
                }

                #[test]
                fn contains_all() {
                    let mut x = $name::empty();
                    x.add_inplace(1);
                    x.add_inplace(2);
                    x.add_inplace(3);
                    x.add_inplace(4);
                    x.add_inplace(7);

                    let mut y = $name::empty();
                    y.add_inplace(1);
                    y.add_inplace(2);
                    y.add_inplace(3);
                    y.add_inplace(4);
                    y.add_inplace(5);
                    y.add_inplace(6);
                    y.add_inplace(7);

                    assert!(y.contains_all(x));
                }
                #[test]
                fn contains_all_partial_overlap() {
                    let mut x = $name::empty();
                    x.add_inplace(1);
                    x.add_inplace(2);
                    x.add_inplace(4);

                    let mut y = $name::singleton(0);
                    y.add_inplace(1);
                    y.add_inplace(2);
                    y.add_inplace(3);
                    y.add_inplace(4);
                    y.add_inplace(5);

                    assert!(!x.contains_all(y));
                }

                #[test]
                fn emptyset_disjoint_from_fullset() {
                    let x = $name::empty();
                    let y = $name::full();
                    assert!(x.disjoint(y));
                }
                #[test]
                fn fullset_disjoint_from_emptyset() {
                    let x = $name::empty();
                    let y = $name::full();
                    assert!(y.disjoint(x));
                }
                #[test]
                fn set_disjoint_from_complement() {
                    let x = $name::singleton(4);
                    let mut y = x;
                    y.flip_inplace();
                    assert!(x.disjoint(y));
                }
                #[test]
                fn disjoint_if_disjoint() {
                    let x = $name::singleton(3);
                    let y = $name::singleton(4);
                    assert!(x.disjoint(y));
                }
                #[test]
                fn not_disjoint_if_partial_overlap() {
                    let mut x = $name::singleton(3);
                    x.add_inplace(0);
                    let mut y = $name::singleton(4);
                    y.add_inplace(0);
                    assert!(!x.disjoint(y));
                }

                #[test]
                fn emptyset_does_not_intersect_fullset() {
                    let x = $name::empty();
                    let y = $name::full();
                    assert!(!x.intersects(y));
                }
                #[test]
                fn fullset_does_not_intersect_emptyset() {
                    let x = $name::empty();
                    let y = $name::full();
                    assert!(!y.intersects(x));
                }
                #[test]
                fn set_does_not_intersect_complement() {
                    let x = $name::singleton(4);
                    let mut y = x;
                    y.flip_inplace();
                    assert!(!x.intersects(y));
                }
                #[test]
                fn do_not_intersect_if_disjoint() {
                    let x = $name::singleton(3);
                    let y = $name::singleton(4);
                    assert!(!x.intersects(y));
                }
                #[test]
                fn intersect_if_partial_overlap() {
                    let mut x = $name::singleton(3);
                    x.add_inplace(0);
                    let mut y = $name::singleton(4);
                    y.add_inplace(0);
                    assert!(x.intersects(y));
                }
                #[test]
                fn intersect_self() {
                    let x = $name::singleton(6);
                    assert!(x.intersects(x));
                }

                // zeroes
                #[test]
                fn no_zeroes_in_fullset() {
                    let x = $name::full();
                    assert_eq!(0, x.zeroes().count());
                }
                #[test]
                fn all_zeroes_in_emptyset() {
                    let x = $name::empty();
                    assert_eq!(x.capacity(), x.zeroes().count());
                }
                #[test]
                fn almost_all_zeroes_in_sinleton() {
                    let x = $name::singleton(6);
                    assert_eq!(x.capacity()-1, x.zeroes().count());
                }

                // --- DISPLAY -----------------------
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
                    let set = $name::singleton(1).add(4).add(6);
                    assert_eq!("{1, 4, 6}", format!("{}", set));
                }
            }
        }
    };
}

test!(Set8,     8);
test!(Set16,   16);
test!(Set32,   32);
test!(Set64,   64);
test!(Set128, 128);
test!(Set256, 256);