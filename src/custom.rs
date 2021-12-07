use std::{slice::Iter, usize, ops::Not, fmt::Display};

/// This structure encapsulates a small allocation free set of integers.
/// Because it is implemented as a fixed size bitset, it can only
/// accomodate values in the range 0..$capa/8. 
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct BitSet<const CAPA: usize> {
    blocks: [u8; CAPA]
}
impl <const CAPA: usize> BitSet<CAPA> {
    /// This method creates an empty set 
    pub fn empty() -> Self {
        Self{blocks: [0_u8; CAPA]}
    }
    /// This method returns the complete set
    pub fn all() -> Self {
        Self{blocks: [255_u8; CAPA]}
    }
    /// This method creates a singleton set holding the single value 'x'.
    pub fn singleton(x: usize) -> Self {
        let mut res = Self::empty();
        res.add(x);
        res
    }
    /// Returns true iff the set contains item i
    pub fn contains(&self, i: usize) -> bool {
        let block  = i/8;
        let offset = i%8;
        let mask   = 1_u8 << offset;
        (self.blocks[block] & mask) > 0
    }
    /// Adds element i to the set
    pub fn add(&mut self, i: usize) {
        let block  = i/8;
        let offset = i%8;
        let mask   = 1_u8 << offset;
        self.blocks[block] |= mask;
    }
    /// Removes element i from the set
    pub fn remove(&mut self, i: usize) {
        let block  = i/8;
        let offset = i%8;
        let mask   = !(1_u8 << offset);
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
            .for_each(|(x, y)| *x&=y)
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
            .all(|(x, y)| x&y == 0)
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
        BitIter::new(self.blocks.iter())
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
}

#[derive(Debug, Clone)]
pub struct BitIter<'a> {
    blocks : Iter<'a, u8>,
    current: Option<u8>,
    bit    : u8,
    pos    : usize
}
impl <'a> BitIter<'a> {
    pub fn new(blocks: Iter<'a, u8>) -> Self {
        Self {
            blocks, 
            current: None,
            bit    : 1_u8,
            pos    : 0
        }
    }
}
impl Iterator for BitIter<'_> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos % 8 == 0 {
            self.current = self.blocks.next().copied();
            self.bit     = 1_u8;
        }
        let res    = self.current.map(|i| (i & self.bit) > 0_u8 );
        self.bit <<= 1;
        self.pos  += 1;
        res
    }
}
impl <const CAPA: usize> Display for BitSet<CAPA> {
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

#[cfg(test)]
mod tests {
    use super::*;
    type BitSet8 = BitSet::<1>;

    // --- EMPTY SET -------------------------------------------------
    #[test]
    fn emptyset_is_empty() {
        let x = BitSet8::empty();
        assert!(x.is_empty());
    }
    #[test]
    fn emptyset_len() {
        let x = BitSet8::empty();
        assert_eq!(0, x.len());
    }
    #[test]
    fn emptyset_iter_count(){
        let x = BitSet8::empty();
        assert_eq!(0, x.iter().count());
    }
    #[test]
    fn emptyset_contains_no_item() {
        let x = BitSet8::empty();
        for i in 0..8{
            assert!(!x.contains(i));
        }
    }
    #[test]
    fn emptyset_complement() {
        let mut bs1 = BitSet8::empty();
        bs1.complement();
        assert_eq!(bs1, BitSet8::all());

        bs1.complement();
        assert_eq!(bs1, BitSet8::empty());
    }

    // --- FULL SET --------------------------------------------------
    #[test]
    fn fullset_is_not_empty() {
        let x = BitSet8::all();
        assert!(!x.is_empty());
    }
    #[test]
    fn fullset_len() {
        let x = BitSet8::all();
        assert_eq!(8, x.len());
    }
    #[test]
    fn fullset_iter_count(){
        let x = BitSet8::all();
        assert_eq!(8, x.iter().count());
    }
    #[test]
    fn fullset_contains_all_items() {
        let x = BitSet8::all();
        for i in 0..8 {
            assert!(x.contains(i));
        }
    }
    #[test]
    fn fullset_complement() {
        let mut bs = BitSet8::all();
        bs.complement();
        assert_eq!(bs, BitSet8::empty());

        bs.complement();
        assert_eq!(bs, BitSet8::all());
    }
    
    // --- SINGLETON --------------------------------------------------
    #[test]
    fn singleton_is_not_empty() {
        let x = BitSet8::singleton(0);
        assert!(!x.is_empty());
    }
    #[test]
    fn singleton_len() {
        let x = BitSet8::singleton(1);
        assert_eq!(1, x.len());
    }
    #[test]
    fn singleton_iter_count(){
        let x = BitSet8::singleton(2);
        assert_eq!(1, x.iter().count());
    }
    #[test]
    fn singleton_contains_one_single_item() {
        let x = BitSet8::singleton(3);
        for i in 0..8 {
            if i == 3 {
                assert!(x.contains(i));
            } else {
                assert!(!x.contains(i));
            }
        }
    }
    #[test]
    fn singleton_complement() {
        let mut x = BitSet8::singleton(4);
        x.complement();
        x.complement();
        assert_eq!(x, BitSet8::singleton(4));
    }

    // --- OTHER METHODS ---------------------------------------------
    #[test]
    fn test_union() {
        let mut _124   = BitSet8::empty();
        let mut _035   = BitSet8::empty();
        let mut _01235 = BitSet8::empty();

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
        let mut _124   = BitSet8::empty();
        let mut _025   = BitSet8::empty();

        _124.add(1);
        _124.add(2);
        _124.add(4);

        _025.add(0);
        _025.add(2);
        _025.add(5);

        let mut inter = _124;
        inter.inter(&_025);

        assert_eq!(BitSet8::singleton(2), inter);
    }
    #[test]
    fn diff_removes_existing_item() {
        let mut _02 = BitSet8::singleton(0);
        _02.add(2);

        let mut diff = _02;
        diff.diff(&BitSet8::singleton(2));

        assert_eq!(BitSet8::singleton(0), diff);
    }
    #[test]
    fn diff_removes_all_existing_item() {
        let mut _02  = BitSet8::singleton(0);
        let mut delta= BitSet8::singleton(0);

        _02.add(2);
        delta.add(2);

        let mut empty = _02;
        empty.diff(&delta);
        
        assert_eq!(BitSet8::empty(), empty);
    }
    #[test]
    fn diff_leaves_non_existing_items() {
        let mut _02  = BitSet8::singleton(0);
        let mut delta= BitSet8::singleton(0);

        _02.add(2);
        delta.add(6);

        let mut diff = _02;
        diff.diff(&delta);
        
        assert_eq!(BitSet8::singleton(2), diff);
    }
    #[test]
    fn insert_contains() {
        let mut x = BitSet8::empty();
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
        let mut x = BitSet8::singleton(4);
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
        let mut x = BitSet8::singleton(1);
        x.add(2);
        x.add(3);

        let v = x.ones().collect::<Vec<_>>();
        assert_eq!(vec![1, 2, 3], v);
    }
    #[test]
    fn test_into_iter() {
        let mut x = BitSet8::singleton(1);
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
        let set = BitSet8::empty();
        assert_eq!("{}", format!("{}", set));
    }
    #[test]
    fn test_display_singleton(){
        let set = BitSet8::singleton(4);
        assert_eq!("{4}", format!("{}", set));
    }
    #[test]
    fn test_display_multi(){
        let mut set = BitSet8::singleton(1);
        set.add(4);
        set.add(6);
        assert_eq!("{1, 4, 6}", format!("{}", set));
    }
}