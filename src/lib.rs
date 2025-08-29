#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]

mod arena;
pub mod linked_hash_map;

extern crate alloc;

#[cfg(feature = "std")]
type RandomState = std::hash::RandomState;
#[cfg(not(feature = "std"))]
type RandomState = hashbrown::DefaultHashBuilder;

/// A hash map that maintains the relative order of entries, implemented as a
/// doubly-linked list backed by a hash table for O(1) lookups.
///
/// This is the main type alias using the default hasher. For custom hashers,
/// use [`linked_hash_map::LinkedHashMap`] directly.
///
/// # Examples
///
/// ```
/// use tether_map::LinkedHashMap;
///
/// let mut map = LinkedHashMap::new();
/// map.insert("a", 1);
/// map.insert("b", 2);
///
/// // Maintains relative order
/// let entries: Vec<_> = map.iter().collect();
/// assert_eq!(entries, [(&"a", &1), (&"b", &2)]);
/// ```
pub type LinkedHashMap<K, V> = crate::linked_hash_map::LinkedHashMap<K, V, RandomState>;
use core::num::NonZeroU32;

pub use linked_hash_map::CursorMut;
pub use linked_hash_map::Entry;
pub use linked_hash_map::IntoIter;
pub use linked_hash_map::Iter;
pub use linked_hash_map::OccupiedEntry;
pub use linked_hash_map::VacantEntry;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
/// A pointer type used to identify entries in the linked hash map.
///
/// This is an opaque handle that can be used to directly access entries
/// without key lookup. It provides O(1) access to entries. It is
/// **non-generational**, meaning that once an entry is removed, the pointer may
/// be re-used for a new entry.
///
/// # Examples
///
/// ```
/// use tether_map::Entry;
/// use tether_map::LinkedHashMap;
/// use tether_map::Ptr;
///
/// let mut map = LinkedHashMap::new();
/// let ptr = match map.entry("key") {
///     Entry::Vacant(entry) => entry.insert_tail(42).0,
///     Entry::Occupied(entry) => entry.ptr(),
/// };
///
/// // Use the pointer for direct access
/// assert_eq!(map.ptr_get(ptr), Some(&42));
/// ```
pub struct Ptr(NonZeroU32);

impl core::fmt::Debug for Ptr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Ptr({})", self.0.get() - 1)
    }
}

impl Ptr {
    pub(crate) fn unchecked_from(index: usize) -> Self {
        debug_assert!(
            index < u32::MAX as usize,
            "Index too large to fit in Ptr: {index}"
        );
        Ptr(NonZeroU32::new((index as u32).saturating_add(1)).unwrap())
    }

    pub(crate) fn unchecked_get(self) -> usize {
        self.0.get() as usize - 1
    }
}
