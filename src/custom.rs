use std::{slice::Iter, usize, ops::{Not, BitAndAssign, BitOr, BitOrAssign, BitAnd, Shl, ShlAssign}, fmt::Display};

use num::{PrimInt, Unsigned};

/// This structure encapsulates a small allocation free set of integers.
/// Because it is implemented as a fixed size bitset, it can only
/// accomodate values in the range 0..$BITS. 
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct BitSet<T, const BITS: usize, const BLOCKS: usize> 
    where T: PrimInt + Unsigned + BitAnd + BitAndAssign + BitOr + BitOrAssign + Shl + ShlAssign
{
    blocks: [T; BLOCKS]
}
impl <T, const BITS: usize, const BLOCKS: usize> BitSet<T, BITS, BLOCKS> 
    where T: PrimInt + Unsigned + BitAnd + BitAndAssign + BitOr + BitOrAssign + Shl + ShlAssign
{
    /// This method creates an empty set 
    pub fn empty() -> Self {
        Self{blocks: [T::zero(); BLOCKS]}
    }
    /// This method returns the complete set
    pub fn all() -> Self {
        let mut me = Self::empty();
        me.complement();
        me
    }
    /// This method creates a singleton set holding the single value 'x'.
    pub fn singleton(x: usize) -> Self {
        let mut res = Self::empty();
        res.add(x);
        res
    }
    /// Returns true iff the set contains item i
    pub fn contains(&self, i: usize) -> bool {
        let block  = i/Self::bits_per_block();
        let offset = i%Self::bits_per_block();
        let mask   = T::one() << offset;
        (self.blocks[block] & mask) > T::zero()
    }
    /// Adds element i to the set
    pub fn add(&mut self, i: usize) {
        let block  = i/Self::bits_per_block();
        let offset = i%Self::bits_per_block();
        let mask   = T::one() << offset;
        self.blocks[block] |= mask;
    }
    /// Removes element i from the set
    pub fn remove(&mut self, i: usize) {
        let block  = i/Self::bits_per_block();
        let offset = i%Self::bits_per_block();
        let mask   = !(T::one() << offset);
        self.blocks[block] &= mask;
    }
    /// Complements (filp all bits of) the set
    pub fn complement(&mut self) {
        self.blocks.iter_mut().for_each(|x| *x = !*x)
    }
    /// Keeps the intersection of self and other
    pub fn inter(&mut self, other: &Self) {
        self.blocks.iter_mut()
            .zip(other.blocks.iter())
            .for_each(|(x, y)| *x&=*y)
    }
    /// Keeps the union of self and other
    pub fn union(&mut self, other: &Self) {
        self.blocks.iter_mut()
            .zip(other.blocks.iter().copied())
            .for_each(|(x, y)| *x|=y)
    }
    /// Keeps the difference of self and other
    pub fn diff(&mut self, other: &Self) {
        self.blocks.iter_mut()
            .zip(other.blocks.iter().copied())
            .for_each(|(x, y)| *x&=!y)
    }
    /// Returns true iff the set contains all items from the other set
    pub fn contains_all(&self, other: &Self) -> bool {
        self.blocks.iter().copied()
            .zip(other.blocks.iter().copied())
            .all(|(x, y)| x&y == y)
    }
    /// Returns true iff the set is totally disjoint from the other set
    pub fn disjoint(&self, other: &Self) -> bool {
        self.blocks.iter().copied()
            .zip(other.blocks.iter().copied())
            .all(|(x, y)| x&y == T::zero())
    }
    /// Returns true iff the intersects with other set
    pub fn intersect(&self, other: &Self) -> bool {
        self.disjoint(other).not()
    }
    /// Returns an iterator over all bits of the set
    pub fn iter(&self) -> impl Iterator<Item=usize> + '_ {
        self.ones()
    }
    /// Returns the number of items in the set
    pub fn len(&self) -> usize {
        let x: u32 = self.blocks.iter().map(|x| x.count_ones()).sum();
        x as usize
    }
    /// Returns true iff the set is empty (all zeroes)
    pub fn is_empty(&self) -> bool {
        self.len() == 0
     }
    /// Iterates over the bits of the set
    pub fn bits(&self) -> impl Iterator<Item=bool> + '_ {
        BitIter::<T, BITS, BLOCKS>::new(self.blocks.iter())
    }
    /// Iterates over the ones in the set
    pub fn ones(&self) -> impl Iterator<Item=usize> + '_ {
        self.bits()
            .enumerate()
            .filter_map(|(i, x)| if x {Some(i)} else {None})
    }
    /// Iterates over the zeroes of the set
    pub fn zeroes(&self) -> impl Iterator<Item=usize> + '_ {
        self.bits()
            .enumerate()
            .filter_map(|(i, x)| if !x {Some(i)} else {None})
    }
    /// Returns the number of bits that can be contained in this bitset
    pub fn capacity() -> usize {
        BITS
    }
    #[inline]
    fn bits_per_block() -> usize {
        bits_per_block(BITS, BLOCKS)
    }
}

const fn bits_per_block(bits: usize, blocks: usize) -> usize {
    bits / blocks
}

#[derive(Debug, Clone)]
pub struct BitIter<'a, T, const BITS: usize, const BLOCKS: usize> 
    where T: PrimInt + Unsigned + BitAnd + BitAndAssign + BitOr + BitOrAssign + Shl + ShlAssign
{
    blocks : Iter<'a, T>,
    current: Option<T>,
    bit    : T,
    pos    : usize
}
impl <'a, T, const BITS: usize, const BLOCKS: usize> BitIter<'a, T, BITS, BLOCKS> 
    where T: PrimInt + Unsigned + BitAnd + BitAndAssign + BitOr + BitOrAssign + Shl + ShlAssign
{
    pub fn new(blocks: Iter<'a, T>) -> Self {
        Self {
            blocks, 
            current: None,
            bit    : T::one(),
            pos    : 0
        }
    }
}
impl <T, const BITS: usize, const BLOCKS: usize> Iterator for BitIter<'_, T, BITS, BLOCKS> 
    where T: PrimInt + Unsigned + BitAnd + BitAndAssign + BitOr + BitOrAssign + Shl + ShlAssign
{
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos % bits_per_block(BITS, BLOCKS) == 0 {
            self.current = self.blocks.next().copied();
            self.bit     = T::one();
        }
        let res    = self.current.map(|i| (i & self.bit) > T::zero() );
        self.bit <<= T::one();
        self.pos  += 1;
        res
    }
}
impl <T, const BITS: usize, const BLOCKS: usize> Display for BitSet<T, BITS, BLOCKS> 
    where T: PrimInt + Unsigned + BitAnd + BitAndAssign + BitOr + BitOrAssign + Shl + ShlAssign
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

#[allow(clippy::just_underscores_and_digits)]
#[cfg(test)]
mod tests_one_block {
    use super::*;
    type MutBitSet = BitSet::<u8, 8, 1>;

    // --- EMPTY SET -------------------------------------------------
    #[test]
    fn emptyset_is_empty() {
        let x = MutBitSet::empty();
        assert!(x.is_empty());
    }
    #[test]
    fn emptyset_len() {
        let x = MutBitSet::empty();
        assert_eq!(0, x.len());
    }
    #[test]
    fn emptyset_iter_count(){
        let x = MutBitSet::empty();
        assert_eq!(0, x.iter().count());
    }
    #[test]
    fn emptyset_contains_no_item() {
        let x = MutBitSet::empty();
        for i in 0..MutBitSet::capacity() {
            assert!(!x.contains(i));
        }
    }
    #[test]
    fn emptyset_complement() {
        let mut bs1 = MutBitSet::empty();
        bs1.complement();
        assert_eq!(bs1, MutBitSet::all());

        bs1.complement();
        assert_eq!(bs1, MutBitSet::empty());
    }

    // --- FULL SET --------------------------------------------------
    #[test]
    fn fullset_is_not_empty() {
        let x = MutBitSet::all();
        assert!(!x.is_empty());
    }
    #[test]
    fn fullset_len() {
        let x = MutBitSet::all();
        assert_eq!(MutBitSet::capacity(), x.len());
    }
    #[test]
    fn fullset_iter_count(){
        let x = MutBitSet::all();
        assert_eq!(MutBitSet::capacity(), x.iter().count());
    }
    #[test]
    fn fullset_contains_all_items() {
        let x = MutBitSet::all();
        for i in 0..MutBitSet::capacity() {
            assert!(x.contains(i));
        }
    }
    #[test]
    fn fullset_complement() {
        let mut bs = MutBitSet::all();
        bs.complement();
        assert_eq!(bs, MutBitSet::empty());

        bs.complement();
        assert_eq!(bs, MutBitSet::all());
    }
    
    // --- SINGLETON --------------------------------------------------
    #[test]
    fn singleton_is_not_empty() {
        let x = MutBitSet::singleton(0);
        assert!(!x.is_empty());
    }
    #[test]
    fn singleton_len() {
        let x = MutBitSet::singleton(1);
        assert_eq!(1, x.len());
    }
    #[test]
    fn singleton_iter_count(){
        let x = MutBitSet::singleton(2);
        assert_eq!(1, x.iter().count());
    }
    #[test]
    fn singleton_contains_one_single_item() {
        let x = MutBitSet::singleton(3);
        for i in 0..MutBitSet::capacity() {
            if i == 3 {
                assert!(x.contains(i));
            } else {
                assert!(!x.contains(i));
            }
        }
    }
    #[test]
    fn singleton_complement() {
        let mut x = MutBitSet::singleton(4);
        x.complement();
        x.complement();
        assert_eq!(x, MutBitSet::singleton(4));
    }

    // --- OTHER METHODS ---------------------------------------------
    #[test]
    fn test_union() {
        let mut _124   = MutBitSet::empty();
        let mut _035   = MutBitSet::empty();
        let mut _01235 = MutBitSet::empty();

        _124.add(1);
        _124.add(2);
        _124.add(4);

        _035.add(0);
        _035.add(3);
        _035.add(5);

        _01235.add(0);
        _01235.add(1);
        _01235.add(2);
        _01235.add(3);
        _01235.add(4);
        _01235.add(5);

        let mut union = _124;
        union.union(&_035);

        assert_eq!(_01235, union);
    }

    #[test]
    fn test_inter() {
        let mut _124   = MutBitSet::empty();
        let mut _025   = MutBitSet::empty();

        _124.add(1);
        _124.add(2);
        _124.add(4);

        _025.add(0);
        _025.add(2);
        _025.add(5);

        let mut inter = _124;
        inter.inter(&_025);

        assert_eq!(MutBitSet::singleton(2), inter);
    }
    #[test]
    fn diff_removes_existing_item() {
        let mut _02 = MutBitSet::singleton(0);
        _02.add(2);

        let mut diff = _02;
        diff.diff(&MutBitSet::singleton(2));

        assert_eq!(MutBitSet::singleton(0), diff);
    }
    #[test]
    fn diff_removes_all_existing_item() {
        let mut _02  = MutBitSet::singleton(0);
        let mut delta= MutBitSet::singleton(0);

        _02.add(2);
        delta.add(2);

        let mut empty = _02;
        empty.diff(&delta);
        
        assert_eq!(MutBitSet::empty(), empty);
    }
    #[test]
    fn diff_leaves_non_existing_items() {
        let mut _02  = MutBitSet::singleton(0);
        let mut delta= MutBitSet::singleton(0);

        _02.add(2);
        delta.add(6);

        let mut diff = _02;
        diff.diff(&delta);
        
        assert_eq!(MutBitSet::singleton(2), diff);
    }
    #[test]
    fn insert_contains() {
        let mut x = MutBitSet::empty();
        assert!(!x.contains(1));
        assert!(!x.contains(2));
        x.add(1);

        assert!(x.contains(1));
        assert!(!x.contains(2));
        
        // duplicate add has no effect
        x.add(1);
        assert!(x.contains(1));
        assert!(!x.contains(2));
    }
    #[test]
    fn remove_contains() {
        let mut x = MutBitSet::singleton(4);
        assert!(!x.contains(1));
        assert!(!x.contains(2));
        assert!(x.contains(4));

        // removing non existing has no effect
        x.remove(1);
        assert!(!x.contains(1));
        assert!(!x.contains(2));
        assert!(x.contains(4));
        
        // removing existing
        x.remove(4);
        assert!(!x.contains(1));
        assert!(!x.contains(2));
        assert!(!x.contains(4));
        
        // duplicate remove has no effect
        x.remove(4);
        assert!(!x.contains(1));
        assert!(!x.contains(2));
        assert!(!x.contains(4));
    }
    #[test]
    fn test_iter() {
        let mut x = MutBitSet::singleton(1);
        x.add(2);
        x.add(3);

        let v = x.ones().collect::<Vec<_>>();
        assert_eq!(vec![1, 2, 3], v);
    }
    #[test]
    fn test_into_iter() {
        let mut x = MutBitSet::singleton(1);
        x.add(2);
        x.add(3);

        let mut v = vec![];
        for i in x.ones() {
            v.push(i);
        }
        assert_eq!(vec![1, 2, 3], v);
    }

    #[test]
    fn test_display_empty(){
        let set = MutBitSet::empty();
        assert_eq!("{}", format!("{}", set));
    }
    #[test]
    fn test_display_singleton(){
        let set = MutBitSet::singleton(4);
        assert_eq!("{4}", format!("{}", set));
    }
    #[test]
    fn test_display_multi(){
        let mut set = MutBitSet::singleton(1);
        set.add(4);
        set.add(6);
        assert_eq!("{1, 4, 6}", format!("{}", set));
    }
}

#[allow(clippy::just_underscores_and_digits)]
#[cfg(test)]
mod tests_multiple_blocks {
    use super::*;
    type MutBitSet = BitSet::<u128, 256, 2>;

    // --- EMPTY SET -------------------------------------------------
    #[test]
    fn emptyset_is_empty() {
        let x = MutBitSet::empty();
        assert!(x.is_empty());
    }
    #[test]
    fn emptyset_len() {
        let x = MutBitSet::empty();
        assert_eq!(0, x.len());
    }
    #[test]
    fn emptyset_iter_count(){
        let x = MutBitSet::empty();
        assert_eq!(0, x.iter().count());
    }
    #[test]
    fn emptyset_contains_no_item() {
        let x = MutBitSet::empty();
        for i in 0..MutBitSet::capacity() {
            assert!(!x.contains(i));
        }
    }
    #[test]
    fn emptyset_complement() {
        let mut bs1 = MutBitSet::empty();
        bs1.complement();
        assert_eq!(bs1, MutBitSet::all());

        bs1.complement();
        assert_eq!(bs1, MutBitSet::empty());
    }

    // --- FULL SET --------------------------------------------------
    #[test]
    fn fullset_is_not_empty() {
        let x = MutBitSet::all();
        assert!(!x.is_empty());
    }
    #[test]
    fn fullset_len() {
        let x = MutBitSet::all();
        assert_eq!(MutBitSet::capacity(), x.len());
    }
    #[test]
    fn fullset_iter_count(){
        let x = MutBitSet::all();
        assert_eq!(MutBitSet::capacity(), x.iter().count());
    }
    #[test]
    fn fullset_contains_all_items() {
        let x = MutBitSet::all();
        for i in 0..MutBitSet::capacity() {
            assert!(x.contains(i));
        }
    }
    #[test]
    fn fullset_complement() {
        let mut bs = MutBitSet::all();
        bs.complement();
        assert_eq!(bs, MutBitSet::empty());

        bs.complement();
        assert_eq!(bs, MutBitSet::all());
    }
    
    // --- SINGLETON --------------------------------------------------
    #[test]
    fn singleton_is_not_empty() {
        let x = MutBitSet::singleton(0);
        assert!(!x.is_empty());
    }
    #[test]
    fn singleton_len() {
        let x = MutBitSet::singleton(1);
        assert_eq!(1, x.len());
    }
    #[test]
    fn singleton_iter_count(){
        let x = MutBitSet::singleton(2);
        assert_eq!(1, x.iter().count());
    }
    #[test]
    fn singleton_contains_one_single_item() {
        let x = MutBitSet::singleton(3);
        for i in 0..MutBitSet::capacity() {
            if i == 3 {
                assert!(x.contains(i));
            } else {
                assert!(!x.contains(i));
            }
        }
    }
    #[test]
    fn singleton_complement() {
        let mut x = MutBitSet::singleton(4);
        x.complement();
        x.complement();
        assert_eq!(x, MutBitSet::singleton(4));
    }

    // --- OTHER METHODS ---------------------------------------------
    #[test]
    fn test_union() {
        let mut _124_127  = MutBitSet::empty();
        let mut _035_254  = MutBitSet::empty();
        let mut _01235_127_254 = MutBitSet::empty();

        _124_127.add(1);
        _124_127.add(2);
        _124_127.add(4);
        _124_127.add(127);

        _035_254.add(0);
        _035_254.add(3);
        _035_254.add(5);
        _035_254.add(254);

        _01235_127_254.add(0);
        _01235_127_254.add(1);
        _01235_127_254.add(2);
        _01235_127_254.add(3);
        _01235_127_254.add(4);
        _01235_127_254.add(5);
        _01235_127_254.add(127);
        _01235_127_254.add(254);

        let mut union = _124_127;
        union.union(&_035_254);

        assert_eq!(_01235_127_254, union);
    }

    #[test]
    fn test_inter() {
        let mut _124_127  = MutBitSet::empty();
        let mut _025_254  = MutBitSet::empty();

        _124_127.add(1);
        _124_127.add(2);
        _124_127.add(4);
        _124_127.add(127);

        _025_254.add(0);
        _025_254.add(2);
        _025_254.add(5);
        _025_254.add(254);

        let mut inter = _124_127;
        inter.inter(&_025_254);

        assert_eq!(MutBitSet::singleton(2), inter);
    }
    #[test]
    fn diff_removes_existing_item() {
        let mut _234 = MutBitSet::singleton(0);
        _234.add(234);

        let mut diff = _234;
        diff.diff(&MutBitSet::singleton(234));

        assert_eq!(MutBitSet::singleton(0), diff);
    }
    #[test]
    fn diff_removes_all_existing_item() {
        let mut _135  = MutBitSet::singleton(135);
        let mut delta= MutBitSet::singleton(0);

        _135.add(135);
        delta.add(135);

        let mut empty = _135;
        empty.diff(&delta);
        
        assert_eq!(MutBitSet::empty(), empty);
    }
    #[test]
    fn diff_leaves_non_existing_items() {
        let mut _135  = MutBitSet::singleton(135);
        let mut delta= MutBitSet::singleton(135);

        _135.add(2);
        delta.add(6);

        let mut diff = _135;
        diff.diff(&delta);
        
        assert_eq!(MutBitSet::singleton(2), diff);
    }
    #[test]
    fn insert_contains() {
        let mut x = MutBitSet::empty();
        assert!(!x.contains(1));
        assert!(!x.contains(2));
        x.add(1);

        assert!(x.contains(1));
        assert!(!x.contains(2));
        
        // duplicate add has no effect
        x.add(1);
        assert!(x.contains(1));
        assert!(!x.contains(2));

        // insertion in next block
        assert!(!x.contains(155));
        x.add(155);
        assert!(x.contains(155));
    }
    #[test]
    fn remove_contains() {
        let mut x = MutBitSet::singleton(4);
        assert!(!x.contains(1));
        assert!(!x.contains(2));
        assert!(x.contains(4));

        // removing non existing has no effect
        x.remove(1);
        assert!(!x.contains(1));
        assert!(!x.contains(2));
        assert!(x.contains(4));
        
        // removing existing
        x.remove(4);
        assert!(!x.contains(1));
        assert!(!x.contains(2));
        assert!(!x.contains(4));
        
        // duplicate remove has no effect
        x.remove(4);
        assert!(!x.contains(1));
        assert!(!x.contains(2));
        assert!(!x.contains(4));

        // in next block
        x.add(155);
        assert!(x.contains(155));
        
        x.remove(155);
        assert!(!x.contains(155));
        x.remove(155);
        assert!(!x.contains(155));
    }
    #[test]
    fn test_iter() {
        let mut x = MutBitSet::singleton(1);
        x.add(2);
        x.add(3);
        x.add(254);

        let v = x.ones().collect::<Vec<_>>();
        assert_eq!(vec![1, 2, 3, 254], v);
    }
    #[test]
    fn test_into_iter() {
        let mut x = MutBitSet::singleton(1);
        x.add(2);
        x.add(3);
        x.add(254);

        let mut v = vec![];
        for i in x.ones() {
            v.push(i);
        }
        assert_eq!(vec![1, 2, 3, 254], v);
    }

    #[test]
    fn test_display_empty(){
        let set = MutBitSet::empty();
        assert_eq!("{}", format!("{}", set));
    }
    #[test]
    fn test_display_singleton(){
        let set = MutBitSet::singleton(154);
        assert_eq!("{154}", format!("{}", set));
    }
    #[test]
    fn test_display_multi(){
        let mut set = MutBitSet::singleton(1);
        set.add(4);
        set.add(6);
        set.add(228);
        assert_eq!("{1, 4, 6, 228}", format!("{}", set));
    }
}
