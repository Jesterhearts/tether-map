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
/// A pointer type used to identify entries in the linked hash map.
///
/// This is an opaque handle that can be used to directly access entries
/// without key lookup. It provides O(1) access to entries.
///
/// By default, `Ptr` is **non-generational**, meaning that once an entry is
/// removed, the pointer may be re-used for a new entry. With the `generational`
/// feature enabled, `Ptr` includes generation tracking that will panic or
/// return None when attempting to use a stale pointer after its entry has been
/// removed.
///
/// # Examples
///
/// ```
/// # #[cfg(not(feature = "generational"))] {
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
///
/// // Remove the entry
/// map.remove(&"key");
///
/// // Using the stale pointer is a logic error but will not panic
/// assert_eq!(map.ptr_get(ptr), None);
///
/// // Insert a new entry, which may reuse the same Ptr value
/// map.insert("key", 100);
///
/// // The old pointer is stale, but may point to the new entry by coincidence
/// // This may work or not depending on whether the same Ptr value was reused:
/// // assert_eq!(map.ptr_get(ptr), Some(100));
///
/// # }
/// ```
///
/// With the `generational` feature enabled, using a stale pointer will return
/// None or panic:
///
/// ```
/// # #[cfg(feature = "generational")] {
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
/// // Remove the entry
/// map.remove(&"key");
///
/// // Using the stale pointer will return None or panic
/// assert_eq!(map.ptr_get(ptr), None);
///
/// // Insert a new entry, which may reuse the same Ptr value
/// map.insert("key", 100);
/// // The old pointer is stale, so this will definitely return None
/// assert_eq!(map.ptr_get(ptr), None);
///
/// # }
/// ```
pub struct Ptr {
    inner: NonZeroU32,
    #[cfg(feature = "generational")]
    generation: u32,
}

impl core::fmt::Debug for Ptr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        #[cfg(not(feature = "generational"))]
        {
            write!(f, "Ptr({})", self.inner.get() - 1)
        }
        #[cfg(feature = "generational")]
        write!(f, "Ptr({}@{})", self.inner.get() - 1, self.generation)
    }
}

impl Ptr {
    pub(crate) fn unchecked_from(
        index: usize,
        #[cfg(feature = "generational")] generation: u32,
    ) -> Self {
        debug_assert!(
            index < u32::MAX as usize,
            "Index too large to fit in Ptr: {index}"
        );
        Ptr {
            inner: NonZeroU32::new((index as u32).saturating_add(1)).unwrap(),
            #[cfg(feature = "generational")]
            generation,
        }
    }

    pub(crate) fn unchecked_get(self) -> usize {
        self.inner.get() as usize - 1
    }
}
