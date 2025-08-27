#![doc = include_str!("../README.md")]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(not(feature = "iter-mut"), forbid(unsafe_code))]
#![deny(missing_docs)]

mod arena;
pub mod linked_hash_map;

extern crate alloc;

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
/// use tether_map::{
///     Entry,
///     LinkedHashMap,
///     Ptr,
/// };
///
/// let mut map = LinkedHashMap::new();
/// let ptr = match map.entry("key") {
///     Entry::Vacant(entry) => entry.insert_tail(42),
///     Entry::Occupied(entry) => entry.ptr(),
/// };
///
/// // Use the pointer for direct access
/// assert_eq!(map.ptr_get(ptr), Some(&42));
/// ```
pub struct Ptr(NonZeroU32);

impl core::fmt::Debug for Ptr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if *self == Ptr::null() {
            write!(f, "Ptr(null)")
        } else {
            write!(f, "Ptr({})", self.0.get() - 1)
        }
    }
}

impl Default for Ptr {
    fn default() -> Self {
        Ptr::null()
    }
}

impl Ptr {
    /// Creates a null pointer.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tether_map::Ptr;
    /// let ptr = Ptr::null();
    /// assert!(ptr.is_null());
    /// ```
    pub fn null() -> Self {
        Ptr(NonZeroU32::new(u32::MAX).unwrap())
    }

    /// Returns true if this is a null pointer.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tether_map::{LinkedHashMap, Ptr};
    /// let null_ptr = Ptr::null();
    /// assert!(null_ptr.is_null());
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert_tail("key", 42);
    /// let ptr = map.get_ptr(&"key").unwrap();
    /// assert!(!ptr.is_null());
    /// ```
    pub fn is_null(&self) -> bool {
        *self == Ptr::null()
    }

    pub(crate) fn unchecked_from(index: usize) -> Self {
        debug_assert!(
            index < u32::MAX as usize - 1,
            "Index too large to fit in Ptr: {index}"
        );
        Ptr(NonZeroU32::new((index as u32).saturating_add(1)).unwrap())
    }

    pub(crate) fn unchecked_get(self) -> usize {
        self.0.get() as usize - 1
    }

    /// Converts to Option, returning None for null pointers.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tether_map::{LinkedHashMap, Ptr};
    /// let null_ptr = Ptr::null();
    /// assert_eq!(null_ptr.optional(), None);
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert_tail("key", 42);
    /// let ptr = map.get_ptr(&"key").unwrap();
    /// assert_eq!(ptr.optional(), Some(ptr));
    /// ```
    pub fn optional(self) -> Option<Ptr> {
        if self.is_null() { None } else { Some(self) }
    }

    /// Returns this pointer if non-null, otherwise returns the provided
    /// alternative.
    ///
    /// # Examples
    ///
    /// ```
    /// # use tether_map::{LinkedHashMap, Ptr};
    /// let mut map = LinkedHashMap::new();
    /// map.insert_tail("key", 42);
    ///
    /// let null_ptr = Ptr::null();
    /// let valid_ptr = map.get_ptr(&"key").unwrap();
    ///
    /// assert_eq!(null_ptr.or(valid_ptr), valid_ptr);
    /// assert_eq!(valid_ptr.or(null_ptr), valid_ptr);
    /// ```
    pub fn or(&self, other: Ptr) -> Ptr {
        if self.is_null() { other } else { *self }
    }
}

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

pub use linked_hash_map::{
    CursorMut,
    Entry,
    IntoIter,
    Iter,
    OccupiedEntry,
    VacantEntry,
};
