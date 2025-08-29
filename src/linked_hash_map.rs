//! Linked hash map implementation.
//!
//! This module provides the core [`LinkedHashMap`] type and related
//! functionality. The linked hash map maintains relative order while providing
//! O(1) access, insertion, and removal operations.
//!
//! # Examples
//!
//! ```
//! use tether_map::linked_hash_map::LinkedHashMap;
//!
//! let mut map = LinkedHashMap::new();
//! map.insert("first", 1);
//! map.insert("second", 2);
//!
//! // Iteration preserves insertion order
//! let entries: Vec<_> = map.iter().collect();
//! assert_eq!(entries, [(&"first", &1), (&"second", &2)]);
//! ```

use alloc::vec::Vec;
use core::clone::Clone;
use core::cmp::Eq;
use core::hash::BuildHasher;
use core::hash::Hash;
use core::marker::PhantomData;
use core::ops::Index;
use core::ops::IndexMut;
use core::panic;
use core::ptr::NonNull;
use core::ptr::{
    self,
};

use hashbrown::HashTable;
use hashbrown::hash_table;

use crate::Ptr;
use crate::RandomState;
use crate::arena::Arena;
use crate::arena::FreedSlot;
use crate::arena::LLSlot;

/// A hash map that maintains relative order using a doubly-linked list.
///
/// This data structure combines the O(1) lookup performance of a hash table
/// with the ability to iterate over entries in relative order. It supports
/// arbitrary key-value pairs where keys implement `Hash + Eq`.
///
/// The generic parameters are:
/// - `K`: Key type, must implement `Hash + Eq`
/// - `T`: Value type
/// - `S`: Hash builder type, defaults to the standard hasher
///
/// # Examples
///
/// ```
/// use tether_map::linked_hash_map::LinkedHashMap;
///
/// let mut map = LinkedHashMap::new();
/// map.insert("apple", 5);
/// map.insert("banana", 3);
/// map.insert("cherry", 8);
///
/// // Iterate in insertion order
/// for (key, value) in map.iter() {
///     println!("{}: {}", key, value);
/// }
/// // Prints: apple: 5, banana: 3, cherry: 8
/// ```
pub struct LinkedHashMap<K, T, S = RandomState> {
    head: *mut LLSlot<K, T>,
    tail: *mut LLSlot<K, T>,
    nodes: Arena<K, T>,
    table: HashTable<NonNull<LLSlot<K, T>>>,
    hasher: S,
}

impl<K: Hash + Eq + Clone, T: Clone> Clone for LinkedHashMap<K, T> {
    fn clone(&self) -> Self {
        let mut new_map = LinkedHashMap::with_capacity(self.len());
        let mut cursor = new_map.head_cursor_mut();
        for (key, value) in self.iter() {
            cursor.insert_after_move_to(key.clone(), value.clone());
        }
        new_map
    }
}

/// Represents an entry that was removed from the linked hash map.
///
/// Contains the key-value pair along with the pointers to the previous
/// and next entries in the linked list, allowing for potential reinsertion
/// at the same position.
///
/// # Examples
///
/// ```
/// use tether_map::LinkedHashMap;
///
/// let mut map = LinkedHashMap::new();
/// map.insert("key", 42);
/// let ptr = map.get_ptr(&"key").unwrap();
///
/// let removed = map.remove_ptr(ptr).unwrap();
/// assert_eq!(removed.key, "key");
/// assert_eq!(removed.value, 42);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemovedEntry<K, T> {
    /// The key of the removed entry
    pub key: K,
    /// The value of the removed entry
    pub value: T,
    /// Pointer to the previous entry in the linked list
    pub prev: Option<Ptr>,
    /// Pointer to the next entry in the linked list
    pub next: Option<Ptr>,
}

impl<K: core::fmt::Debug, T: core::fmt::Debug, S> core::fmt::Debug for LinkedHashMap<K, T, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        #[derive(Debug)]
        #[allow(dead_code)]
        struct Entry<'a, K, V> {
            key: &'a K,
            value: &'a V,
            previous: &'a K,
            next: &'a K,
        }

        let mut entries = Vec::with_capacity(self.len());

        for ptr in self.table.iter() {
            // SAFETY: All pointers come from our own arena
            let entry = unsafe { ptr.as_ref() };
            let next_key = unsafe { &entry.links.next().as_ref().data.assume_init_ref().key };
            let prev_key = unsafe { &entry.links.prev().as_ref().data.assume_init_ref().key };
            let key = unsafe { &entry.data.assume_init_ref().key };
            let value = unsafe { &entry.data.assume_init_ref().value };

            entries.push(Entry {
                key,
                value,
                previous: prev_key,
                next: next_key,
            });
        }

        // SAFETY: head & tail are valid pointers into our own arena.
        f.debug_struct("LinkedHashMap")
            .field("len", &self.len())
            .field("head", &self.head.addr())
            .field("tail", &self.tail.addr())
            .field("entries", &entries)
            .finish()?;

        Ok(())
    }
}

impl<K, T, S: BuildHasher + Default> Default for LinkedHashMap<K, T, S> {
    fn default() -> Self {
        LinkedHashMap::with_capacity_and_hasher(0, S::default())
    }
}

impl<K, T> LinkedHashMap<K, T> {
    /// Creates a new linked hash map with the specified capacity.
    ///
    /// The map will be able to hold at least `capacity` elements without
    /// reallocating. If `capacity` is 0, the map will not allocate.
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map: LinkedHashMap<&str, i32> = LinkedHashMap::with_capacity(10);
    /// assert_eq!(map.len(), 0);
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        LinkedHashMap {
            head: core::ptr::null_mut(),
            tail: core::ptr::null_mut(),
            nodes: Arena::with_capacity(capacity),
            table: HashTable::with_capacity(capacity),
            hasher: RandomState::default(),
        }
    }

    /// Creates a new, empty linked hash map.
    ///
    /// The map is initially created with a capacity of 0, so it will not
    /// allocate until the first element is inserted.
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map: LinkedHashMap<&str, i32> = LinkedHashMap::new();
    /// assert!(map.is_empty());
    /// map.insert("key", 42);
    /// assert!(!map.is_empty());
    /// ```
    pub fn new() -> Self {
        Self::with_capacity(0)
    }
}

impl<K, T, S> LinkedHashMap<K, T, S> {
    /// Creates a new linked hash map with the specified capacity and hasher.
    ///
    /// The map will use the given hasher to hash keys and will be able to
    /// hold at least `capacity` elements without reallocating.
    ///
    /// # Examples
    ///
    /// ```
    /// # use hashbrown::DefaultHashBuilder as RandomState;
    /// use tether_map::linked_hash_map::LinkedHashMap;
    ///
    /// let hasher = RandomState::default();
    /// let mut map: LinkedHashMap<&str, i32, _> = LinkedHashMap::with_capacity_and_hasher(10, hasher);
    /// map.insert("key", 42);
    /// ```
    pub fn with_capacity_and_hasher(capacity: usize, hasher: S) -> Self {
        LinkedHashMap {
            head: core::ptr::null_mut(),
            tail: core::ptr::null_mut(),
            nodes: Arena::with_capacity(capacity),
            table: HashTable::with_capacity(capacity),
            hasher,
        }
    }

    /// Moves the entry at `moved` to be immediately after the entry at `after`
    /// in the linked list.
    ///
    /// If `moved` is already immediately after `after`, returns `None` and no
    /// change is made. Both pointers must refer to valid entries in the
    /// map.
    ///
    /// # Arguments
    ///
    /// * `moved` - The pointer to the entry to move
    /// * `after` - The pointer to the entry after which `moved` will be placed
    ///
    /// # Returns
    ///
    /// * `Some(())` if the move was successful
    /// * `None` if the move was unnecessary (already in correct position) or if
    ///   either pointer is invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// let (ptr1, _) = map.insert_tail_full("a", 1);
    /// let (ptr2, _) = map.insert_tail_full("b", 2);
    /// let (ptr3, _) = map.insert_tail_full("c", 3);
    ///
    /// // Move "c" to be after "a"
    /// map.move_after(ptr3, ptr1);
    ///
    /// // Order is now: a, c, b
    /// let entries: Vec<_> = map.iter().collect();
    /// assert_eq!(entries, [(&"a", &1), (&"c", &3), (&"b", &2)]);
    /// ```
    pub fn move_after(&mut self, moved: Ptr, after: Ptr) -> Option<()> {
        if moved == after {
            return None;
        }

        let moved = self.nodes.map_ptr(moved)?;
        let after = self.nodes.map_ptr(after)?;

        // SAFETY: Both pointers are occupied, and not identical per the above
        // checks.
        unsafe { self.move_after_internal(moved, after) }
    }

    /// # Safety
    ///
    /// Both pointers must be occupied, meaning they are part of this map's
    /// arena. They must not be equal.
    unsafe fn move_after_internal(
        &mut self,
        mut moved: NonNull<LLSlot<K, T>>,
        mut after: NonNull<LLSlot<K, T>>,
    ) -> Option<()> {
        debug_assert_ne!(moved, after);

        // SAFETY: Per contract of this function, moved/after are occupied.
        let (mut needs_next, mut needs_prev, mut also_needs_prev) = unsafe {
            let moved_mut = moved.as_mut();

            if after.as_ptr() == self.tail && moved.as_ptr() == self.head {
                self.tail = moved.as_ptr();
                self.head = moved_mut.links.next().as_ptr();
                return Some(());
            }

            let after_mut = after.as_mut();

            if after_mut.links.next() == moved {
                return None;
            }

            let needs_next = moved_mut.links.prev();
            let needs_prev = moved_mut.links.next();
            let also_needs_prev = after_mut.links.next();

            *moved_mut.links.next_mut() = also_needs_prev;
            *moved_mut.links.prev_mut() = after;
            *after_mut.links.next_mut() = moved;

            (needs_next, needs_prev, also_needs_prev)
        };

        if also_needs_prev != after {
            // SAFETY: We do not have non occupied pointers links in our list.
            unsafe {
                *also_needs_prev.as_mut().links.prev_mut() = moved;
            }
        }

        if needs_next != moved {
            // SAFETY: We do not have non occupied pointers links in our list.
            unsafe {
                *needs_next.as_mut().links.next_mut() = needs_prev;
            }
        }

        if needs_prev != moved {
            // SAFETY: We do not have non occupied pointers links in our list.
            unsafe {
                *needs_prev.as_mut().links.prev_mut() = needs_next;
            }
        }

        if self.head == moved.as_ptr() {
            self.head = needs_prev.as_ptr();
        }
        if self.tail == moved.as_ptr() {
            self.tail = needs_next.as_ptr();
        }

        if self.tail == after.as_ptr() {
            self.tail = moved.as_ptr();
        }

        Some(())
    }

    /// Moves the entry at `moved` to be immediately before the entry at
    /// `before` in the linked list.
    ///
    /// If `moved` is already immediately before `before`, returns `None` and no
    /// change is made. Both pointers must refer to valid entries in the
    /// map.
    ///
    /// # Arguments
    ///
    /// * `moved` - The pointer to the entry to move
    /// * `before` - The pointer to the entry before which `moved` will be
    ///   placed
    ///
    /// # Returns
    ///
    /// * `Some(())` if the move was successful
    /// * `None` if the move was unnecessary (already in correct position) or if
    ///   either pointer is invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// let (ptr1, _) = map.insert_tail_full("a", 1);
    /// let (ptr2, _) = map.insert_tail_full("b", 2);
    /// let (ptr3, _) = map.insert_tail_full("c", 3);
    ///
    /// // Move "a" to be before "c"
    /// map.move_before(ptr1, ptr3);
    ///
    /// // Order is now: b, a, c
    /// let entries: Vec<_> = map.iter().collect();
    /// assert_eq!(entries, [(&"b", &2), (&"a", &1), (&"c", &3)]);
    /// ```
    pub fn move_before(&mut self, moved: Ptr, before: Ptr) -> Option<()> {
        if moved == before {
            return None;
        }

        let moved = self.nodes.map_ptr(moved)?;
        let before = self.nodes.map_ptr(before)?;

        // SAFETY: Both pointers are occupied, and not identical per the above
        // checks.
        unsafe { self.move_before_internal(moved, before) }
    }

    /// # Safety
    ///
    /// Both pointers must be occupied, meaning they are part of this map's
    /// arena. They must not be equal.
    unsafe fn move_before_internal(
        &mut self,
        mut moved: NonNull<LLSlot<K, T>>,
        mut before: NonNull<LLSlot<K, T>>,
    ) -> Option<()> {
        debug_assert_ne!(moved, before);

        // SAFETY: Per contract of this function, moved/after are occupied.
        let (mut needs_next, mut needs_prev, mut also_needs_next) = unsafe {
            let moved_mut = moved.as_mut();

            if before.as_ptr() == self.head && moved.as_ptr() == self.tail {
                self.head = moved.as_ptr();
                self.tail = moved_mut.links.prev().as_ptr();
                return Some(());
            }

            let before_mut = before.as_mut();

            if before_mut.links.prev() == moved {
                return None;
            }

            let needs_next = moved_mut.links.prev();
            let needs_prev = moved_mut.links.next();
            let also_needs_next = before_mut.links.prev();

            *moved_mut.links.next_mut() = before;
            *moved_mut.links.prev_mut() = also_needs_next;
            *before_mut.links.prev_mut() = moved;

            (needs_next, needs_prev, also_needs_next)
        };

        if also_needs_next != before {
            // SAFETY: We do not have non occupied pointers links in our list.
            unsafe {
                *also_needs_next.as_mut().links.next_mut() = moved;
            }
        }

        if needs_next != moved {
            // SAFETY: We do not have non occupied pointers links in our list.
            unsafe {
                *needs_next.as_mut().links.next_mut() = needs_prev;
            }
        }

        if needs_prev != moved {
            // SAFETY: We do not have non occupied pointers links in our list.
            unsafe {
                *needs_prev.as_mut().links.prev_mut() = needs_next;
            }
        }

        if self.head == moved.as_ptr() {
            self.head = needs_prev.as_ptr();
        }
        if self.tail == moved.as_ptr() {
            self.tail = needs_next.as_ptr();
        }

        if self.head == before.as_ptr() {
            self.head = moved.as_ptr();
        }

        Some(())
    }

    /// Links an entry as the new head of the linked list.
    ///
    /// The entry at `ptr` must already exist in the map but not be part of the
    /// linked list. This is typically used when inserting a new entry at
    /// the head position.
    ///
    /// # Arguments
    ///
    /// * `ptr` - The pointer to the entry to link as the new head
    ///
    /// # Returns
    ///
    /// * `Some(())` if the linking was successful
    /// * `None` if the pointer is invalid
    pub fn link_as_head(&mut self, ptr: Ptr) -> Option<()> {
        let node = self.nodes.map_ptr(ptr)?;
        // SAFETY: We just checked that ptr is occupied.
        unsafe { self.link_node_internal(node, self.tail, self.head, true) }
    }

    /// Links an entry as the new tail of the linked list.
    ///
    /// The entry at `ptr` must already exist in the map but not be part of the
    /// linked list. This is typically used when inserting a new entry at
    /// the tail position.
    ///
    /// # Arguments
    ///
    /// * `ptr` - The pointer to the entry to link as the new tail
    ///
    /// # Returns
    ///
    /// * `Some(())` if the linking was successful
    /// * `None` if the pointer is invalid
    pub fn link_as_tail(&mut self, ptr: Ptr) -> Option<()> {
        let node = self.nodes.map_ptr(ptr)?;
        // SAFETY: We just checked that ptr is occupied .
        unsafe { self.link_node_internal(node, self.tail, self.head, false) }
    }

    /// Links an entry into the doubly-linked list at a specific position after
    /// another entry.
    pub fn link_after(&mut self, ptr: Ptr, prev: Ptr) -> Option<()> {
        let node = self.nodes.map_ptr(ptr)?;
        let prev = self
            .nodes
            .map_ptr(prev)
            .map_or(core::ptr::null_mut(), |p| p.as_ptr());
        // SAFETY: ptr is occupied and non-null per the above checks. If prev is
        // non-null, it must also be occupied since it came from our own arena.
        unsafe { self.link_node_internal(node, prev, core::ptr::null_mut(), false) }
    }

    /// Links an entry into the doubly-linked list at a specific position before
    /// another entry.
    pub fn link_before(&mut self, ptr: Ptr, next: Ptr) -> Option<()> {
        let node = self.nodes.map_ptr(ptr)?;
        let next = self
            .nodes
            .map_ptr(next)
            .map_or(core::ptr::null_mut(), |p| p.as_ptr());
        // SAFETY: ptr is occupied and non-null per the above checks. If next is
        // non-null, it must also be occupied since it came from our own arena.
        unsafe { self.link_node_internal(node, core::ptr::null_mut(), next, true) }
    }

    /// # Safety
    ///
    /// `node` must be occupied (i.e. part of this map's arena). If prev and
    /// next are non-null, they must also be occupied.
    unsafe fn link_node_internal(
        &mut self,
        mut node: NonNull<LLSlot<K, T>>,
        prev: *mut LLSlot<K, T>,
        next: *mut LLSlot<K, T>,
        as_head: bool,
    ) -> Option<()> {
        if self.head.is_null() && self.tail.is_null() {
            self.head = node.as_ptr();
            self.tail = node.as_ptr();
            // SAFETY: Node is occupied per contract.
            unsafe {
                *node.as_mut().links.next_mut() = node;
            };
            // SAFETY: Node is occupied per contract.
            unsafe {
                *node.as_mut().links.prev_mut() = node;
            }
            return Some(());
        }

        if prev.is_null() && next.is_null() {
            return None;
        }

        debug_assert!(
            !prev.is_null() || !next.is_null(),
            "Either prev or next must be non-null"
        );

        let mut prev = if prev.is_null() {
            // SAFETY: We know that either prev or next is non-null per the above
            // check. We just checked if prev was null, so next must be valid. It must also
            // be occupied per function contract.
            unsafe { next.as_ref().unwrap().links.prev() }
        } else {
            NonNull::new(prev).unwrap()
        };

        let mut next = if next.is_null() {
            // SAFETY: We know that either prev or next is non-null per the above
            // check. We just checked if next was null, so prev must be valid. It must also
            // be occupied per function contract.
            unsafe { prev.as_ref().links.next() }
        } else {
            NonNull::new(next).unwrap()
        };

        // SAFETY: We know from the above checks that prev, next, and node are occupied.
        // prev.next is either null or occupied, since prev is from our arena.
        // Similarly, next.prev is either null or occupied, since next is from our
        // arena.
        unsafe {
            *prev.as_mut().links.next_mut() = node;
            *next.as_mut().links.prev_mut() = node;
            *node.as_mut().links.prev_mut() = prev;
            *node.as_mut().links.next_mut() = next;
        }

        if as_head && self.head == next.as_ptr() {
            self.head = node.as_ptr();
        } else if !as_head && self.tail == prev.as_ptr() {
            self.tail = node.as_ptr();
        }

        Some(())
    }

    /// Moves an entry to the tail (end) of the linked list.
    ///
    /// This is equivalent to calling `move_after(moved, tail_ptr())`.
    ///
    /// # Arguments
    ///
    /// * `moved` - The pointer to the entry to move to the tail
    ///
    /// # Returns
    ///
    /// * `Some(())` if the move was successful
    /// * `None` if the entry is already at the tail or if the pointer is
    ///   invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// let (ptr1, _) = map.insert_tail_full("a", 1);
    /// let (ptr2, _) = map.insert_tail_full("b", 2);
    /// let (ptr3, _) = map.insert_tail_full("c", 3);
    ///
    /// // Move "a" to the tail
    /// map.move_to_tail(ptr1);
    ///
    /// // Order is now: b, c, a
    /// let entries: Vec<_> = map.iter().collect();
    /// assert_eq!(entries, [(&"b", &2), (&"c", &3), (&"a", &1)]);
    /// ```
    pub fn move_to_tail(&mut self, moved: Ptr) -> Option<()> {
        self.move_after(moved, self.tail_ptr()?)
    }

    /// Moves an entry to the head (beginning) of the linked list.
    ///
    /// This is equivalent to calling `move_before(moved, head_ptr())`.
    ///
    /// # Arguments
    ///
    /// * `moved` - The pointer to the entry to move to the head
    ///
    /// # Returns
    ///
    /// * `Some(())` if the move was successful
    /// * `None` if the entry is already at the head or if the pointer is
    ///   invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// let (ptr1, _) = map.insert_tail_full("a", 1);
    /// let (ptr2, _) = map.insert_tail_full("b", 2);
    /// let (ptr3, _) = map.insert_tail_full("c", 3);
    ///
    /// // Move "c" to the head
    /// map.move_to_head(ptr3);
    ///
    /// // Order is now: c, a, b
    /// let entries: Vec<_> = map.iter().collect();
    /// assert_eq!(entries, [(&"c", &3), (&"a", &1), (&"b", &2)]);
    /// ```
    pub fn move_to_head(&mut self, moved: Ptr) -> Option<()> {
        self.move_before(moved, self.head_ptr()?)
    }

    /// Creates a mutable cursor positioned at the entry with the given pointer.
    ///
    /// A cursor provides a way to navigate and modify the linked list structure
    /// while maintaining efficient access to specific entries.
    ///
    /// # Arguments
    ///
    /// * `ptr` - The pointer to the entry where the cursor should be positioned
    ///
    /// # Returns
    ///
    /// A `CursorMut` positioned at the specified entry. If the pointer is
    /// invalid, the cursor will be positioned at a null entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// let (ptr, _) = map.insert_tail_full("a", 1);
    ///
    /// let mut cursor = map.ptr_cursor_mut(ptr);
    /// if let Some((key, value)) = cursor.current_mut() {
    ///     *value = 42;
    /// }
    /// ```
    pub fn ptr_cursor_mut(&'_ mut self, ptr: Ptr) -> CursorMut<'_, K, T, S> {
        let ptr = self
            .nodes
            .map_ptr(ptr)
            .map_or(core::ptr::null_mut(), |p| p.as_ptr());
        CursorMut { ptr, map: self }
    }

    /// Returns the pointer to the next entry in the linked list.
    ///
    /// # Arguments
    ///
    /// * `ptr` - The pointer to the current entry
    ///
    /// # Returns
    ///
    /// * `Some(next_ptr)` if there is a next entry
    /// * `None` if this is the last entry or if the pointer is invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// let (ptr1, _) = map.insert_tail_full("a", 1);
    /// let (ptr2, _) = map.insert_tail_full("b", 2);
    ///
    /// // Get the next entry after "a"
    /// let next = map.next_ptr(ptr1).unwrap();
    /// assert_eq!(map.ptr_get(next), Some(&2));
    /// ```
    pub fn next_ptr(&self, ptr: Ptr) -> Option<Ptr> {
        let ptr = self.nodes.map_ptr(ptr)?;
        // SAFETY: We just checked that ptr is occupied.
        unsafe { Some(ptr.as_ref().links.next().as_ref().this) }
    }

    /// Returns the pointer to the previous entry in the linked list.
    ///
    /// # Arguments
    ///
    /// * `ptr` - The pointer to the current entry
    ///
    /// # Returns
    ///
    /// * `Some(prev_ptr)` if there is a previous entry
    /// * `None` if this is the first entry or if the pointer is invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// let (ptr1, _) = map.insert_tail_full("a", 1);
    /// let (ptr2, _) = map.insert_tail_full("b", 2);
    ///
    /// // Get the previous entry before "b"
    /// let prev = map.prev_ptr(ptr2).unwrap();
    /// assert_eq!(map.ptr_get(prev), Some(&1));
    /// ```
    pub fn prev_ptr(&self, ptr: Ptr) -> Option<Ptr> {
        let ptr = self.nodes.map_ptr(ptr)?;
        // SAFETY: We just checked that ptr is occupied.
        unsafe { Some(ptr.as_ref().links.prev().as_ref().this) }
    }

    /// Creates a mutable cursor positioned at the head (first entry) of the
    /// linked list.
    ///
    /// # Returns
    ///
    /// A `CursorMut` positioned at the head entry. If the map is empty,
    /// the cursor will be positioned at a null entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert_tail("a", 1);
    /// map.insert_tail("b", 2);
    ///
    /// let mut cursor = map.head_cursor_mut();
    /// if let Some((key, value)) = cursor.current_mut() {
    ///     assert_eq!(key, &"a");
    ///     *value = 42;
    /// }
    /// ```
    pub fn head_cursor_mut(&'_ mut self) -> CursorMut<'_, K, T, S> {
        CursorMut {
            ptr: self.head,
            map: self,
        }
    }

    /// Creates a mutable cursor positioned at the tail (last entry) of the
    /// linked list.
    ///
    /// # Returns
    ///
    /// A `CursorMut` positioned at the tail entry. If the map is empty,
    /// the cursor will be positioned at a null entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert_tail("a", 1);
    /// map.insert_tail("b", 2);
    ///
    /// let mut cursor = map.tail_cursor_mut();
    /// if let Some((key, value)) = cursor.current_mut() {
    ///     assert_eq!(key, &"b");
    ///     *value = 42;
    /// }
    /// ```
    pub fn tail_cursor_mut(&'_ mut self) -> CursorMut<'_, K, T, S> {
        CursorMut {
            ptr: self.tail,
            map: self,
        }
    }

    /// Returns the pointer to the head (first entry) of the linked list.
    pub fn head_ptr(&self) -> Option<Ptr> {
        // SAFETY: head is either null or a valid pointer into our own arena.
        unsafe { self.head.as_ref().map(|p| p.this) }
    }

    /// Returns the pointer to the tail (last entry) of the linked list.
    pub fn tail_ptr(&self) -> Option<Ptr> {
        // SAFETY: tail is either null or a valid pointer into our own arena.
        unsafe { self.tail.as_ref().map(|p| p.this) }
    }

    /// Returns a reference to the value associated with the given pointer.
    pub fn ptr_get(&self, ptr: Ptr) -> Option<&T> {
        // SAFETY: We just retrieved the pointer from our own arena.
        self.nodes
            .map_ptr(ptr)
            .map(|p| unsafe { &p.as_ref().data.assume_init_ref().value })
    }

    /// Returns a reference to the key-value pair associated with the given
    /// pointer.
    pub fn ptr_get_entry(&self, ptr: Ptr) -> Option<(&K, &T)> {
        self.nodes.map_ptr(ptr).map(|p| {
            // SAFETY: We just retrieved the pointer from our own arena.
            let r = unsafe { p.as_ref().data.assume_init_ref() };
            (&r.key, &r.value)
        })
    }

    /// Returns a mutable reference to the key-value pair associated with the
    /// given pointer.
    pub fn ptr_get_entry_mut(&mut self, ptr: Ptr) -> Option<(&K, &mut T)> {
        self.nodes.map_ptr(ptr).map(|mut p| {
            // SAFETY: We just retrieved the pointer from our own arena.
            unsafe { p.as_mut().data.assume_init_mut().key_value_mut() }
        })
    }

    /// Returns a mutable reference to the value associated with the given
    /// pointer.
    pub fn ptr_get_mut(&mut self, ptr: Ptr) -> Option<&mut T> {
        self.nodes.map_ptr(ptr).map(|mut p| {
            // SAFETY: We just retrieved the pointer from our own arena.
            let r = unsafe { p.as_mut().data.assume_init_mut() };
            &mut r.value
        })
    }

    /// Returns a reference to the key associated with the given pointer.
    pub fn ptr_get_key(&self, ptr: Ptr) -> Option<&K> {
        // SAFETY: We just retrieved the pointer from our own arena.
        self.nodes
            .map_ptr(ptr)
            .map(|p| unsafe { &p.as_ref().data.assume_init_ref().key })
    }

    /// Returns the number of elements in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut a = LinkedHashMap::new();
    /// assert_eq!(a.len(), 0);
    /// a.insert(1, "a");
    /// assert_eq!(a.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Returns `true` if the map contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut a = LinkedHashMap::new();
    /// assert!(a.is_empty());
    /// a.insert(1, "a");
    /// assert!(!a.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clears the map, removing all key-value pairs.
    ///
    /// Keeps the allocated memory for reuse.
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut a = LinkedHashMap::new();
    /// a.insert(1, "a");
    /// a.clear();
    /// assert!(a.is_empty());
    /// ```
    pub fn clear(&mut self) {
        for node in self.table.drain() {
            // SAFETY: We only store valid pointers into our own arena
            unsafe {
                self.nodes.free_and_unlink(node);
            }
        }
        self.head = ptr::null_mut();
        self.tail = ptr::null_mut();
    }

    /// Returns an iterator over the key-value pairs of the map, in relative
    /// order.
    ///
    /// The iterator element type is `(&'a K, &'a V)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// for (key, val) in map.iter() {
    ///     println!("key: {} val: {}", key, val);
    /// }
    /// ```
    pub fn iter<'s>(&'s self) -> Iter<'s, K, T, S> {
        Iter {
            forward_ptr: self.head,
            reverse_ptr: self.tail,
            _map: core::marker::PhantomData,
        }
    }

    /// Checks if the map contains an entry for the given pointer.
    ///
    /// # Arguments
    ///
    /// * `ptr` - The pointer to check for existence in the map
    ///
    /// # Returns
    ///
    /// * `true` if the map contains an entry for the pointer
    /// * `false` otherwise
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// let (ptr, _) = map.insert_tail_full("a", 1);
    /// assert!(map.contains_ptr(ptr));
    /// map.remove_ptr(ptr);
    /// assert!(!map.contains_ptr(ptr));
    /// ```
    pub fn contains_ptr(&self, ptr: Ptr) -> bool {
        self.nodes.is_occupied(ptr)
    }

    /// Returns an iterator over the keys of the map in their relative order.
    ///
    /// The keys are returned in the order they were inserted into the map,
    /// which is maintained by the underlying doubly-linked list.
    ///
    /// # Returns
    ///
    /// An iterator that yields `&K` values in relative order.
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// let keys: Vec<_> = map.keys().collect();
    /// assert_eq!(keys, [&"a", &"b", &"c"]);
    /// ```
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.iter().map(|(k, _)| k)
    }

    /// Returns an iterator over the values of the map in their relative order.
    ///
    /// The values are returned in the order their corresponding keys were
    /// inserted into the map, which is maintained by the underlying
    /// doubly-linked list.
    ///
    /// # Returns
    ///
    /// An iterator that yields `&T` values in relative order.
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// let values: Vec<_> = map.values().collect();
    /// assert_eq!(values, [&1, &2, &3]);
    /// ```
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.iter().map(|(_, v)| v)
    }

    /// Returns a mutable iterator over the values of the map in their relative
    /// order.
    ///
    /// The values are returned in the order their corresponding keys were
    /// inserted into the map, which is maintained by the underlying
    /// doubly-linked list. This method allows for in-place mutation of the
    /// values.
    ///
    /// # Returns
    ///
    /// A mutable iterator that yields `&mut T` values in relative order.
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// for value in map.values_mut() {
    ///     *value *= 2;
    /// }
    ///
    /// let values: Vec<_> = map.values().collect();
    /// assert_eq!(values, [&2, &4, &6]);
    /// ```
    pub fn values_mut<'s>(&'s mut self) -> ValuesMut<'s, K, T> {
        ValuesMut {
            iter: self.iter_mut(),
        }
    }

    /// Returns a mutable iterator over the key-value pairs of the map in their
    /// relative order.
    ///
    /// The key-value pairs are returned in the order they were inserted into
    /// the map, which is maintained by the underlying doubly-linked list.
    /// This method allows for in-place mutation of the values while keeping
    /// the keys immutable.
    ///
    /// # Returns
    ///
    /// A mutable iterator that yields `(&K, &mut T)` pairs in relative order.
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// for (key, value) in map.iter_mut() {
    ///     if key == &"b" {
    ///         *value *= 10;
    ///     }
    /// }
    ///
    /// assert_eq!(map.get(&"b"), Some(&20));
    /// ```
    pub fn iter_mut<'s>(&'s mut self) -> IterMut<'s, K, T> {
        IterMut {
            forward_ptr: self.head,
            reverse_ptr: self.tail,
            _nodes: core::marker::PhantomData,
        }
    }
}

impl<K, T, S> PartialEq for LinkedHashMap<K, T, S>
where
    K: Hash + Eq,
    T: PartialEq,
    S: BuildHasher,
{
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }

        self.iter()
            .all(|(key, value)| other.get(key).is_some_and(|v| *value == *v))
    }
}

impl<K, T, S> Eq for LinkedHashMap<K, T, S>
where
    K: Hash + Eq,
    T: Eq,
    S: BuildHasher,
{
}

impl<K, T, S> FromIterator<(K, T)> for LinkedHashMap<K, T, S>
where
    K: Hash + Eq,
    S: BuildHasher + Default,
{
    fn from_iter<I: IntoIterator<Item = (K, T)>>(iter: I) -> Self {
        let mut map = Self::default();
        map.extend(iter);
        map
    }
}

impl<K, T, S> Extend<(K, T)> for LinkedHashMap<K, T, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    fn extend<I: IntoIterator<Item = (K, T)>>(&mut self, iter: I) {
        for (key, value) in iter {
            self.insert(key, value);
        }
    }
}

impl<'a, K, T, S> Extend<(&'a K, &'a T)> for LinkedHashMap<K, T, S>
where
    K: Hash + Eq + Clone,
    T: Clone,
    S: BuildHasher,
{
    fn extend<I: IntoIterator<Item = (&'a K, &'a T)>>(&mut self, iter: I) {
        for (key, value) in iter {
            self.insert(key.clone(), value.clone());
        }
    }
}

impl<K, T, S> IntoIterator for LinkedHashMap<K, T, S> {
    type IntoIter = IntoIter<K, T>;
    type Item = (K, T);

    fn into_iter(mut self) -> Self::IntoIter {
        let nodes = core::mem::replace(&mut self.nodes, Arena::new());
        self.table.clear(); // Clear the table to avoid double-free

        IntoIter {
            nodes,
            forward_ptr: self.head,
            reverse_ptr: self.tail,
        }
    }
}

impl<K: Hash + Eq, T, S: BuildHasher> LinkedHashMap<K, T, S> {
    /// Shrinks the capacity of the map as much as possible.
    pub fn shrink_to_fit(&mut self) {
        self.table.shrink_to_fit(|k| {
            // SAFETY: We only store valid pointers into our own arena
            unsafe { self.hasher.hash_one(&k.as_ref().data.assume_init_ref().key) }
        });
    }

    /// Removes the entry at the tail (end) of the linked list.
    ///
    /// This is equivalent to calling `remove_ptr(tail_ptr())`, but is more
    /// performant.
    pub fn remove_tail(&mut self) -> Option<RemovedEntry<K, T>> {
        let ptr = NonNull::new(self.tail)?;

        // SAFETY: We know our tail is not null and from our arena.
        unsafe { Some(self.remove_ptr_internal(ptr)) }
    }

    /// Removes the entry at the head (beginning) of the linked list.
    ///
    /// This is equivalent to calling `remove_ptr(head_ptr())`, but is more
    /// performant.
    pub fn remove_head(&mut self) -> Option<RemovedEntry<K, T>> {
        let ptr = NonNull::new(self.head)?;

        // SAFETY: We know our head is not null and from our arena.
        unsafe { Some(self.remove_ptr_internal(ptr)) }
    }

    /// Removes the entry at the given pointer from the map.
    ///
    /// If the pointer is invalid or does not correspond to an occupied entry,
    /// returns `None`. Otherwise, removes the entry and returns a
    /// `RemovedEntry` containing the key, value, and neighboring pointers.
    pub fn remove_ptr(&mut self, ptr: Ptr) -> Option<RemovedEntry<K, T>> {
        let ptr = self.nodes.map_ptr(ptr)?;

        // SAFETY: We just checked that ptr is occupied.
        unsafe { Some(self.remove_ptr_internal(ptr)) }
    }

    /// # Safety
    ///
    /// `ptr` must be occupied (i.e. part of this map's arena).
    unsafe fn remove_ptr_internal(&mut self, ptr: NonNull<LLSlot<K, T>>) -> RemovedEntry<K, T> {
        // SAFETY: ptr is occupied per the function contract, so we know it is from our
        // own arena.
        let hash = unsafe {
            self.hasher
                .hash_one(&ptr.as_ref().data.assume_init_ref().key)
        };

        match self.table.find_entry(hash, move |k| *k == ptr) {
            Ok(occupied) => {
                occupied.remove();
            }
            Err(_) => {
                #[cold]
                #[inline(never)]
                fn die() -> ! {
                    panic!("Pointer not found in table");
                }
                die()
            }
        };

        let ptr_raw = ptr.as_ptr();
        // SAFETY: ptr is occupied per the function contract, so we know it is from our
        // own arena. After this call, ptr is invalidated, and we update
        // head/tail below in finish_removal which does not examine ptr's data.
        let FreedSlot {
            data,
            prev,
            next,
            prev_raw,
            next_raw,
            ..
        } = unsafe { self.nodes.free_and_unlink(ptr) };

        if self.head == ptr_raw {
            self.head = next_raw;
        }
        if self.tail == ptr_raw {
            self.tail = prev_raw;
        }

        RemovedEntry {
            key: data.key,
            value: data.value,
            prev,
            next,
        }
    }

    /// Retains only the entries specified by the predicate.
    ///
    /// In other words, removes all entries for which `f(&key, &mut value)`
    /// returns `false`. The entries are visited in relative order, and the
    /// predicate is allowed to modify the values.
    ///
    /// # Arguments
    ///
    /// * `f` - A closure that returns `true` for entries that should be kept.
    ///   It receives references to the key and a mutable reference to the
    ///   value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    /// map.insert("d", 4);
    ///
    /// // Keep only entries with even values, and double them
    /// map.retain(|_key, value| {
    ///     if *value % 2 == 0 {
    ///         *value *= 2;
    ///         true
    ///     } else {
    ///         false
    ///     }
    /// });
    ///
    /// // Only "b" and "d" remain, with values doubled
    /// assert_eq!(map.len(), 2);
    /// assert_eq!(map.get(&"b"), Some(&4));
    /// assert_eq!(map.get(&"d"), Some(&8));
    /// ```
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&K, &mut T) -> bool,
    {
        let Some(mut ptr) = NonNull::new(self.head) else {
            return;
        };

        let mut seen = 0;
        let len = self.len();
        while seen < len {
            seen += 1;
            // SAFETY: We do not have non occupied pointer links in our list.
            let next = unsafe { ptr.as_ref().links.next() };
            let should_remove = {
                let node = unsafe { ptr.as_mut().data.assume_init_mut() };
                let (key, value) = node.key_value_mut();
                !f(key, value)
            };

            if should_remove {
                // SAFETY: We do not have non occupied pointer links in our list.
                unsafe { self.remove_ptr_internal(ptr) };
            }
            ptr = next;
        }
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the map did not have this key present, `None` is returned and the
    /// entry is inserted at the tail (most recently inserted position).
    ///
    /// If the map did have this key present, the value is updated and the old
    /// value is returned. The entry is **not** moved in the linked list.
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// assert_eq!(map.insert(37, "a"), None);
    /// assert_eq!(map.is_empty(), false);
    ///
    /// map.insert(37, "b");
    /// assert_eq!(map.insert(37, "c"), Some("b"));
    /// assert_eq!(map.get(&37), Some(&"c"));
    /// ```
    pub fn insert(&mut self, key: K, value: T) -> Option<T> {
        let entry = self.entry(key);
        match entry {
            Entry::Occupied(occupied_entry) => {
                let old = occupied_entry.insert_no_move(value);
                Some(old)
            }
            Entry::Vacant(vacant_entry) => {
                vacant_entry.insert_tail(value);
                None
            }
        }
    }

    /// Inserts a key-value pair and returns the pointer and any previous value.
    ///
    /// This method provides the same functionality as `insert` but returns
    /// additional information: the pointer to the inserted/updated entry
    /// and any previous value that was replaced. The insertion position
    /// depends on whether the key already exists:
    ///
    /// - If the key is new, the entry is inserted at the tail
    /// - If the key exists, the value is updated in-place without moving the
    ///   entry
    ///
    /// # Arguments
    ///
    /// * `key` - The key to insert or update
    /// * `value` - The value to insert
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - `Ptr` - The pointer to the inserted or updated entry
    /// - `Option<T>` - `Some(old_value)` if the key existed, `None` if it was
    ///   newly inserted
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    ///
    /// // Insert new entry
    /// let (ptr1, old) = map.insert_full("key1", 10);
    /// assert_eq!(old, None); // No previous value
    ///
    /// // Update existing entry
    /// let (ptr2, old) = map.insert_full("key1", 20);
    /// assert_eq!(old, Some(10)); // Previous value returned
    /// assert_eq!(ptr1, ptr2); // Same pointer since key existed
    ///
    /// // Use the pointer for direct access
    /// assert_eq!(map.ptr_get(ptr1), Some(&20));
    /// ```
    pub fn insert_full(&mut self, key: K, value: T) -> (Ptr, Option<T>) {
        let entry = self.entry(key);
        match entry {
            Entry::Occupied(occupied_entry) => {
                let ptr = occupied_entry.ptr();
                let old = occupied_entry.insert_no_move(value);
                (ptr, Some(old))
            }
            Entry::Vacant(vacant_entry) => {
                let (ptr, _) = vacant_entry.insert_tail(value);
                (ptr, None)
            }
        }
    }

    /// Inserts a key-value pair at the tail of the linked list.
    ///
    /// If the key already exists, updates the value and moves the entry
    /// to the tail position, returning the old value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert_tail("first", 1);
    /// map.insert_tail("second", 2);
    ///
    /// let keys: Vec<_> = map.keys().cloned().collect();
    /// assert_eq!(keys, ["first", "second"]);
    /// ```
    pub fn insert_tail(&mut self, key: K, value: T) -> Option<T> {
        let entry = self.entry(key);
        match entry {
            Entry::Occupied(occupied_entry) => {
                let ptr = occupied_entry.ptr();
                let old = occupied_entry.insert_no_move(value);
                self.move_to_tail(ptr);
                Some(old)
            }
            Entry::Vacant(vacant_entry) => {
                vacant_entry.insert_tail(value);
                None
            }
        }
    }

    /// Inserts a key-value pair at the tail and returns the pointer and any
    /// previous value.
    ///
    /// This method is similar to `insert_full` but with explicit tail
    /// positioning behavior:
    ///
    /// - If the key is new, the entry is inserted at the tail
    /// - If the key exists, the value is updated AND the entry is moved to the
    ///   tail
    ///
    /// This is useful when you want to ensure that updated entries are moved to
    /// the most recent position, implementing LRU (Least Recently Used)
    /// cache behavior.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to insert or update
    /// * `value` - The value to insert
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - `Ptr` - The pointer to the inserted or updated entry
    /// - `Option<T>` - `Some(old_value)` if the key existed, `None` if it was
    ///   newly inserted
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// // Update existing entry - it moves to tail
    /// let (ptr, old) = map.insert_tail_full("a", 10);
    /// assert_eq!(old, Some(1));
    ///
    /// // Order is now: b, c, a (with updated value)
    /// let entries: Vec<_> = map.iter().collect();
    /// assert_eq!(entries, [(&"b", &2), (&"c", &3), (&"a", &10)]);
    /// ```
    pub fn insert_tail_full(&mut self, key: K, value: T) -> (Ptr, Option<T>) {
        let entry = self.entry(key);
        match entry {
            Entry::Occupied(occupied_entry) => {
                let ptr = occupied_entry.ptr();
                let old = occupied_entry.insert_no_move(value);
                self.move_to_tail(ptr);
                (ptr, Some(old))
            }
            Entry::Vacant(vacant_entry) => {
                let (ptr, _) = vacant_entry.insert_tail(value);
                (ptr, None)
            }
        }
    }

    /// Inserts a key-value pair at the head of the linked list.
    ///
    /// If the key already exists, updates the value and moves the entry
    /// to the head position, returning the old value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert_head("first", 1);
    /// map.insert_head("second", 2);
    ///
    /// let keys: Vec<_> = map.keys().cloned().collect();
    /// assert_eq!(keys, ["second", "first"]); // second is now first
    /// ```
    pub fn insert_head(&mut self, key: K, value: T) -> Option<T> {
        let entry = self.entry(key);
        match entry {
            Entry::Occupied(occupied_entry) => {
                let ptr = occupied_entry.ptr();
                let old = occupied_entry.insert_no_move(value);
                self.move_to_head(ptr);
                Some(old)
            }
            Entry::Vacant(vacant_entry) => {
                vacant_entry.insert_head(value);
                None
            }
        }
    }

    /// Inserts a key-value pair at the head and returns the pointer and any
    /// previous value.
    ///
    /// This method is similar to `insert_full` but with explicit head
    /// positioning behavior:
    ///
    /// - If the key is new, the entry is inserted at the head
    /// - If the key exists, the value is updated AND the entry is moved to the
    ///   head
    ///
    /// This is useful when you want to ensure that updated entries are moved to
    /// the most recent position at the beginning of the list, implementing
    /// MRU (Most Recently Used) behavior or priority queuing where recent
    /// updates should be processed first.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to insert or update
    /// * `value` - The value to insert
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - `Ptr` - The pointer to the inserted or updated entry
    /// - `Option<T>` - `Some(old_value)` if the key existed, `None` if it was
    ///   newly inserted
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    /// map.insert("c", 3);
    ///
    /// // Update existing entry - it moves to head
    /// let (ptr, old) = map.insert_head_full("b", 20);
    /// assert_eq!(old, Some(2));
    ///
    /// // Order is now: b (with updated value), a, c
    /// let entries: Vec<_> = map.iter().collect();
    /// assert_eq!(entries, [(&"b", &20), (&"a", &1), (&"c", &3)]);
    /// ```
    pub fn insert_head_full(&mut self, key: K, value: T) -> (Ptr, Option<T>) {
        let entry = self.entry(key);
        match entry {
            Entry::Occupied(occupied_entry) => {
                let ptr = occupied_entry.ptr();
                let old = occupied_entry.insert_no_move(value);
                self.move_to_head(ptr);
                (ptr, Some(old))
            }
            Entry::Vacant(vacant_entry) => {
                let ptr = vacant_entry.insert_head(value);
                (ptr, None)
            }
        }
    }

    /// Creates a mutable cursor positioned at the entry with the given key.
    ///
    /// This method performs a hash table lookup to find the entry with the
    /// specified key, then returns a cursor positioned at that entry.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to search for
    ///
    /// # Returns
    ///
    /// A `CursorMut` positioned at the entry with the given key. If the key is
    /// not found, the cursor will be positioned at a null entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert_tail("a", 1);
    /// map.insert_tail("b", 2);
    ///
    /// let mut cursor = map.key_cursor_mut(&"a");
    /// if let Some((key, value)) = cursor.current_mut() {
    ///     assert_eq!(key, &"a");
    ///     *value = 42;
    /// }
    /// ```
    pub fn key_cursor_mut(&'_ mut self, key: &K) -> CursorMut<'_, K, T, S> {
        let hash = self.hasher.hash_one(key);
        // SAFETY: We only store valid pointers into our own arena
        let ptr = self
            .table
            .find(
                hash,
                |k| unsafe { &k.as_ref().data.assume_init_ref().key } == key,
            )
            .copied()
            .map_or(core::ptr::null_mut(), |p| p.as_ptr());
        CursorMut { ptr, map: self }
    }

    /// Gets the given key's corresponding entry in the map for in-place
    /// manipulation.
    ///
    /// This method provides an efficient way to insert, update, or
    /// conditionally modify entries without performing multiple lookups.
    /// The entry API is particularly useful for complex operations that
    /// depend on whether a key exists.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to get the entry for
    ///
    /// # Returns
    ///
    /// An `Entry` enum which can be either:
    /// - `Entry::Occupied` if the key exists, providing access to the existing
    ///   value
    /// - `Entry::Vacant` if the key doesn't exist, allowing insertion of a new
    ///   value
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::Entry;
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    ///
    /// // Insert a value if key doesn't exist
    /// match map.entry("key") {
    ///     Entry::Vacant(entry) => {
    ///         entry.insert_tail(42);
    ///     }
    ///     Entry::Occupied(_) => {
    ///         // Key already exists
    ///     }
    /// }
    ///
    /// // Update existing value or insert default
    /// let value = match map.entry("counter") {
    ///     Entry::Occupied(mut entry) => {
    ///         *entry.get_mut() += 1;
    ///         *entry.get()
    ///     }
    ///     Entry::Vacant(entry) => *entry.insert_tail(1).1,
    /// };
    /// ```
    pub fn entry(&'_ mut self, key: K) -> Entry<'_, K, T> {
        let hash = self.hasher.hash_one(&key);
        match self.table.entry(
            hash,
            |k| {
                // SAFETY: We only store valid pointers into our own arena
                unsafe { k.as_ref().data.assume_init_ref().key == key }
            },
            |e| {
                // SAFETY: We only store valid pointers into our own arena
                unsafe { self.hasher.hash_one(&e.as_ref().data.assume_init_ref().key) }
            },
        ) {
            hash_table::Entry::Occupied(entry) => Entry::Occupied(OccupiedEntry {
                arena: &mut self.nodes,
                head: &mut self.head,
                tail: &mut self.tail,
                entry,
            }),
            hash_table::Entry::Vacant(entry) => Entry::Vacant(VacantEntry {
                entry,
                key,
                nodes: &mut self.nodes,
                head: &mut self.head,
                tail: &mut self.tail,
            }),
        }
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.remove(&1), Some("a"));
    /// assert_eq!(map.remove(&1), None);
    /// ```
    pub fn remove(&mut self, key: &K) -> Option<T> {
        self.remove_entry(key).map(|(_, v)| v)
    }

    /// Removes a key from the map, returning the stored key and value if the
    /// key was previously in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.remove_entry(&1), Some((1, "a")));
    /// assert_eq!(map.remove(&1), None);
    /// ```
    pub fn remove_entry(&mut self, key: &K) -> Option<(K, T)> {
        let (_, removed) = self.remove_with_ptr(key)?;
        Some((removed.key, removed.value))
    }

    /// Removes a key from the map, returning the pointer to the removed entry
    /// and the stored key, value, and neighboring pointers if the key was  
    /// previously in the map.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to remove from the map
    ///
    /// # Returns
    ///
    /// * `Some((Ptr, RemovedEntry<K, T>))` if the key was found and removed,
    ///   where `Ptr` is the pointer to the removed entry and `RemovedEntry<K
    /// , T>` contains the key, value, and neighboring pointers
    /// * `None` if the key was not found in the map
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// let (ptr, _) = map.insert_tail_full("a", 1);
    ///
    /// // Remove by key
    /// let result = map.remove_with_ptr(&"a");
    /// assert!(result.is_some());
    /// let (removed_ptr, removed_entry) = result.unwrap();
    /// assert_eq!(removed_ptr, ptr);
    /// assert_eq!(removed_entry.key, "a");
    /// assert_eq!(removed_entry.value, 1);
    /// assert_eq!(removed_entry.prev, None);
    /// assert_eq!(removed_entry.next, None);
    /// // Removing a non-existent key returns None
    /// assert_eq!(map.remove_with_ptr(&"b"), None);
    /// ```
    pub fn remove_with_ptr(&mut self, key: &K) -> Option<(Ptr, RemovedEntry<K, T>)> {
        if self.is_empty() {
            return None;
        }

        let hash = self.hasher.hash_one(key);
        match self.table.find_entry(hash, |k| {
            // SAFETY: We only store valid pointers into our own arena
            unsafe { &k.as_ref().data.assume_init_ref().key == key }
        }) {
            Ok(occupied) => {
                let ptr = occupied.remove().0;
                let ptr_raw = ptr.as_ptr();

                // SAFETY: We only store valid pointers into our own arena.
                let FreedSlot {
                    prev,
                    next,
                    data,
                    this,
                    prev_raw,
                    next_raw,
                } = unsafe { self.nodes.free_and_unlink(ptr) };

                if self.head == ptr_raw {
                    self.head = next_raw;
                }
                if self.tail == ptr_raw {
                    self.tail = prev_raw;
                }

                Some((
                    this,
                    RemovedEntry {
                        key: data.key,
                        value: data.value,
                        prev,
                        next,
                    },
                ))
            }
            Err(_) => None,
        }
    }

    /// Returns the pointer to the entry with the given key.
    ///
    /// This method performs a hash table lookup to find the entry with the
    /// specified key and returns its pointer. The pointer can then be used
    /// for direct access operations or cursor positioning without
    /// additional key lookups.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to search for
    ///
    /// # Returns
    ///
    /// * `Some(ptr)` if the key exists in the map
    /// * `None` if the key is not found
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// let (inserted_ptr, _) = map.insert_tail_full("key", 42);
    ///
    /// // Get pointer for the key
    /// let found_ptr = map.get_ptr(&"key").unwrap();
    /// assert_eq!(inserted_ptr, found_ptr);
    ///
    /// // Use the pointer for direct access
    /// assert_eq!(map.ptr_get(found_ptr), Some(&42));
    ///
    /// // Non-existent key returns None
    /// assert_eq!(map.get_ptr(&"missing"), None);
    /// ```
    pub fn get_ptr(&self, key: &K) -> Option<Ptr> {
        let hash = self.hasher.hash_one(key);
        self.table
            .find(hash, |k| {
                // SAFETY: We only store valid pointers into our own arena
                unsafe { &k.as_ref().data.assume_init_ref().key == key }
            })
            .map(|ptr| {
                // SAFETY: We only store valid pointers into our own arena
                unsafe { ptr.as_ref().this }
            })
    }

    /// Returns a reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// `Hash` and `Eq` on the borrowed form must match those for the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.get(&1), Some(&"a"));
    /// assert_eq!(map.get(&2), None);
    /// ```
    pub fn get(&self, key: &K) -> Option<&T> {
        self.table
            .find(self.hasher.hash_one(key), |k| {
                // SAFETY: We only store valid pointers into our own arena
                unsafe { &k.as_ref().data.assume_init_ref().key == key }
            })
            .map(|&ptr| {
                // SAFETY: We only store valid pointers into our own arena
                unsafe { &ptr.as_ref().data.assume_init_ref().value }
            })
    }

    /// Returns a mutable reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// `Hash` and `Eq` on the borrowed form must match those for the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert(1, "a");
    /// if let Some(x) = map.get_mut(&1) {
    ///     *x = "b";
    /// }
    /// assert_eq!(map.get(&1), Some(&"b"));
    /// ```
    pub fn get_mut(&mut self, key: &K) -> Option<&mut T> {
        self.table
            .find_mut(self.hasher.hash_one(key), |k| {
                // SAFETY: We only store valid pointers into our own arena
                unsafe { &k.as_ref().data.assume_init_ref().key == key }
            })
            .map(|ptr| {
                // SAFETY: We only store valid pointers into our own arena
                unsafe { &mut ptr.as_mut().data.assume_init_mut().value }
            })
    }

    /// Returns `true` if the map contains a value for the specified key.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// `Hash` and `Eq` on the borrowed form must match those for the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert(1, "a");
    /// assert_eq!(map.contains_key(&1), true);
    /// assert_eq!(map.contains_key(&2), false);
    /// ```
    pub fn contains_key(&self, key: &K) -> bool {
        self.get_ptr(key).is_some()
    }
}

#[derive(Debug)]
/// A cursor for navigating and modifying a linked hash map.
///
/// A `CursorMut` is like an iterator, except that it can freely seek
/// back-and-forth and can safely mutate the map during iteration. This is
/// because the lifetime is tied to the map, not the cursor itself.
///
/// Cursors always point to an element in the map. The `CursorMut` is positioned
/// at a specific entry and allows for insertion and removal operations relative
/// to that position.
///
/// # Examples
///
/// ```
/// use tether_map::LinkedHashMap;
///
/// let mut map = LinkedHashMap::new();
/// map.insert("a", 1);
/// map.insert("b", 2);
/// map.insert("c", 3);
///
/// let mut cursor = map.head_cursor_mut();
/// if let Some((key, value)) = cursor.current_mut() {
///     *value *= 10;
/// }
/// assert_eq!(map.get(&"a"), Some(&10));
/// ```
pub struct CursorMut<'m, K, T, S> {
    ptr: *mut LLSlot<K, T>,
    map: &'m mut LinkedHashMap<K, T, S>,
}

impl<'m, K: Hash + Eq, T, S: BuildHasher> CursorMut<'m, K, T, S> {
    /// Inserts a key-value pair after the cursor's current position and
    /// moves the cursor to the inserted or updated entry.
    pub fn insert_after_move_to(&mut self, key: K, value: T) -> Option<T> {
        let ptr = if self.ptr.is_null() {
            self.map.tail
        } else {
            self.ptr
        };

        match self.map.entry(key) {
            Entry::Occupied(occupied_entry) => {
                let mut map_ptr = *occupied_entry.entry.get();
                if map_ptr.as_ptr() != ptr {
                    // SAFETY: Both `map_ptr` and `ptr` are guaranteed to be valid pointers within
                    // the map's arena because they come from either existing entries or the
                    // head/tail. We know our map is non-empty because we found
                    // an occupied entry, so we know tail is non-null, meaning the or above cannot
                    // yield null. We have also checked that they are not equal.
                    unsafe {
                        self.map
                            .move_after_internal(map_ptr, NonNull::new(ptr).unwrap())
                    };
                }
                self.ptr = map_ptr.as_ptr();
                // SAFETY: map_ptr was drawn from our own list of valid pointers.
                unsafe {
                    Some(core::mem::replace(
                        &mut map_ptr.as_mut().data.assume_init_mut().value,
                        value,
                    ))
                }
            }
            Entry::Vacant(vacant_entry) => {
                // SAFETY: ptr is either null or from our own arena.
                unsafe {
                    self.ptr = vacant_entry.insert_after_internal(value, ptr).as_ptr();
                }
                None
            }
        }
    }

    /// Inserts a key-value pair before the cursor's current position and
    /// moves the cursor to the inserted or updated entry.
    pub fn insert_before_move_to(&mut self, key: K, value: T) -> Option<T> {
        let ptr = if self.ptr.is_null() {
            self.map.head
        } else {
            self.ptr
        };
        match self.map.entry(key) {
            Entry::Occupied(occupied_entry) => {
                let mut map_ptr = *occupied_entry.entry.get();
                if map_ptr.as_ptr() != ptr {
                    // SAFETY: Both `map_ptr` and `ptr` are guaranteed to be valid pointers within
                    // the map's arena because they come from either existing entries or the
                    // head/tail. We know our map is non-empty because we found
                    // an occupied entry, so we know head is non-null, meaning the or above cannot
                    // yield null. We have also checked that they are not equal.
                    unsafe {
                        self.map
                            .move_before_internal(map_ptr, NonNull::new(ptr).unwrap())
                    };
                }
                self.ptr = map_ptr.as_ptr();
                // SAFETY: map_ptr was drawn from our own list of valid pointers.
                unsafe {
                    Some(core::mem::replace(
                        &mut map_ptr.as_mut().data.assume_init_mut().value,
                        value,
                    ))
                }
            }
            Entry::Vacant(vacant_entry) => {
                // SAFETY: ptr is either null or from our own arena.
                unsafe {
                    self.ptr = vacant_entry.insert_before_internal(value, ptr).as_ptr();
                }
                None
            }
        }
    }

    /// Returns the pointer to the entry with the given key.
    ///
    /// This is a convenience method that delegates to the underlying map's
    /// `get_ptr` method.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to search for
    ///
    /// # Returns
    ///
    /// * `Some(ptr)` if the key exists in the map
    /// * `None` if the key is not found
    pub fn get_ptr(&self, key: &K) -> Option<Ptr> {
        self.map.get_ptr(key)
    }
}

impl<'m, K: Hash + Eq, T, S: BuildHasher> CursorMut<'m, K, T, S> {
    /// Removes the entry before the cursor's current position and returns it.
    pub fn remove_prev(&mut self) -> Option<RemovedEntry<K, T>> {
        // SAFETY: We only store valid pointers into our own arena or null pointers, and
        // we check for null.
        unsafe {
            let prev = self.ptr.as_ref().map(|slot| slot.links.prev())?;
            Some(self.map.remove_ptr_internal(prev))
        }
    }

    /// Removes the entry after the cursor's current position and returns it.
    pub fn remove_next(&mut self) -> Option<RemovedEntry<K, T>> {
        // SAFETY: We only store valid pointers into our own arena or null pointers, and
        // we check for null.
        unsafe {
            let next = self.ptr.as_ref().map(|slot| slot.links.next())?;
            Some(self.map.remove_ptr_internal(next))
        }
    }

    /// Removes the entry at the cursor's current position and returns it.
    pub fn remove(self) -> Option<RemovedEntry<K, T>> {
        // SAFETY: We only store valid pointers into our own arena or null pointers, and
        // we check for null.
        unsafe { Some(self.map.remove_ptr_internal(NonNull::new(self.ptr)?)) }
    }
}

impl<'m, K, T, S> CursorMut<'m, K, T, S> {
    /// Returns an iterator starting from the cursor's current position.
    pub fn iter(&self) -> Iter<'m, K, T, S> {
        Iter {
            forward_ptr: self.ptr,
            reverse_ptr: self.map.tail,
            _map: core::marker::PhantomData,
        }
    }

    /// Moves the cursor to the next entry in the linked list. The internal
    /// linked list is **circular**, so moving next from the tail wraps around
    /// to the head.
    pub fn move_next(&mut self) {
        // SAFETY: We only store valid pointers into our own arena or null pointers.
        unsafe {
            self.ptr = self
                .ptr
                .as_ref()
                .map_or(core::ptr::null_mut(), |slot| slot.links.next().as_ptr())
        }
    }

    /// Moves the cursor to the previous entry in the linked list. The internal
    /// linked list is **circular**, so moving previous from the head wraps
    /// around to the tail.
    pub fn move_prev(&mut self) {
        // SAFETY: We only store valid pointers into our own arena or null pointers.
        unsafe {
            self.ptr = self
                .ptr
                .as_ref()
                .map_or(core::ptr::null_mut(), |slot| slot.links.prev().as_ptr())
        }
    }

    /// Gets the current pointer of the cursor.
    pub fn ptr(&self) -> Option<Ptr> {
        // SAFETY: We only store valid pointers into our own arena or null pointers.
        unsafe { self.ptr.as_ref().map(|slot| slot.this) }
    }

    /// Checks if the cursor is currently at the tail of the linked list.
    pub fn at_tail(&self) -> bool {
        self.ptr == self.map.tail
    }

    /// Checks if the cursor is currently at the head of the linked list.
    pub fn at_head(&self) -> bool {
        self.ptr == self.map.head
    }

    /// Returns the entry at the cursor's current position.
    pub fn current(&self) -> Option<(&K, &T)> {
        // SAFETY: We only store valid pointers into our own arena or null pointers.
        unsafe {
            let data = &self.ptr.as_ref()?.data.assume_init_ref();
            Some((&data.key, &data.value))
        }
    }

    /// Returns a mutable reference to the key-value pair at the cursor's
    /// current position.
    ///
    /// The key reference is immutable while the value reference is mutable,
    /// allowing modification of the value while preserving the key's
    /// integrity.
    ///
    /// # Returns
    ///
    /// * `Some((&K, &mut T))` if the cursor is positioned at a valid entry
    /// * `None` if the cursor is positioned at a null entry
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert("key", 42);
    ///
    /// let mut cursor = map.head_cursor_mut();
    /// if let Some((key, value)) = cursor.current_mut() {
    ///     assert_eq!(key, &"key");
    ///     *value = 100;
    /// }
    /// assert_eq!(map.get(&"key"), Some(&100));
    /// ```
    pub fn current_mut(&mut self) -> Option<(&K, &mut T)> {
        // SAFETY: We only store valid pointers into our own arena or null pointers.
        unsafe {
            let data = self.ptr.as_mut()?.data.assume_init_mut();
            Some(data.key_value_mut())
        }
    }

    /// Returns the pointer to the next entry in the linked list from the
    /// cursor's position.
    ///
    /// # Returns
    ///
    /// * `Some(next_ptr)` if there is a next entry
    /// * `None` if the cursor is at the last entry or positioned at a null
    ///   entry
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert_tail("a", 1);
    /// map.insert_tail("b", 2);
    ///
    /// let cursor = map.head_cursor_mut();
    /// if let Some(next_ptr) = cursor.next_ptr() {
    ///     assert_eq!(map.ptr_get(next_ptr), Some(&2));
    /// }
    /// ```
    pub fn next_ptr(&self) -> Option<Ptr> {
        // SAFETY: We only store valid pointers into our own arena or null pointers.
        unsafe {
            self.ptr
                .as_ref()
                .map(|slot| slot.links.next().as_ref().this)
        }
    }

    /// Returns a reference to the key-value pair of the next entry in the
    /// linked list.
    ///
    /// This is a convenience method that combines `next_ptr()` and accessing
    /// the entry.
    ///
    /// # Returns
    ///
    /// * `Some((&K, &T))` if there is a next entry
    /// * `None` if the cursor is at the last entry or positioned at a null
    ///   entry
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert_tail("a", 1);
    /// map.insert_tail("b", 2);
    ///
    /// let cursor = map.head_cursor_mut();
    /// if let Some((key, value)) = cursor.next() {
    ///     assert_eq!(key, &"b");
    ///     assert_eq!(value, &2);
    /// }
    /// ```
    pub fn next(&self) -> Option<(&K, &T)> {
        let ptr = self.next_ptr()?;
        self.map.ptr_get_entry(ptr)
    }

    /// Returns a mutable reference to the key-value pair of the next entry in
    /// the linked list.
    ///
    /// This is a convenience method that combines `next_ptr()` and accessing
    /// the entry mutably. The key reference is immutable while the value
    /// reference is mutable.
    ///
    /// # Returns
    ///
    /// * `Some((&K, &mut T))` if there is a next entry
    /// * `None` if the cursor is at the last entry or positioned at a null
    ///   entry
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert_tail("a", 1);
    /// map.insert_tail("b", 2);
    ///
    /// let mut cursor = map.head_cursor_mut();
    /// if let Some((key, value)) = cursor.next_mut() {
    ///     assert_eq!(key, &"b");
    ///     *value = 20;
    /// }
    /// assert_eq!(map.get(&"b"), Some(&20));
    /// ```
    pub fn next_mut(&mut self) -> Option<(&K, &mut T)> {
        let ptr = self.next_ptr()?;
        self.map.ptr_get_entry_mut(ptr)
    }

    /// Returns the pointer to the previous entry in the linked list from the
    /// cursor's position.
    ///
    /// # Returns
    ///
    /// * `Some(prev_ptr)` if there is a previous entry
    /// * `None` if the cursor is at the first entry or positioned at a null
    ///   entry
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert_tail("a", 1);
    /// map.insert_tail("b", 2);
    ///
    /// let cursor = map.tail_cursor_mut();
    /// if let Some(prev_ptr) = cursor.prev_ptr() {
    ///     assert_eq!(map.ptr_get(prev_ptr), Some(&1));
    /// }
    /// ```
    pub fn prev_ptr(&self) -> Option<Ptr> {
        // SAFETY: We only store valid pointers into our own arena or null pointers.
        unsafe {
            self.ptr
                .as_ref()
                .map(|slot| slot.links.prev().as_ref().this)
        }
    }

    /// Returns a reference to the key-value pair of the previous entry in the
    /// linked list.
    ///
    /// This is a convenience method that combines `prev_ptr()` and accessing
    /// the entry.
    ///
    /// # Returns
    ///
    /// * `Some((&K, &T))` if there is a previous entry
    /// * `None` if the cursor is at the first entry or positioned at a null
    ///   entry
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert_tail("a", 1);
    /// map.insert_tail("b", 2);
    ///
    /// let cursor = map.tail_cursor_mut();
    /// if let Some((key, value)) = cursor.prev() {
    ///     assert_eq!(key, &"a");
    ///     assert_eq!(value, &1);
    /// }
    /// ```
    pub fn prev(&self) -> Option<(&K, &T)> {
        let ptr = self.prev_ptr()?;
        self.map.ptr_get_entry(ptr)
    }

    /// Returns a mutable reference to the key-value pair of the previous entry
    /// in the linked list.
    ///
    /// This is a convenience method that combines `prev_ptr()` and accessing
    /// the entry mutably. The key reference is immutable while the value
    /// reference is mutable.
    ///
    /// # Returns
    ///
    /// * `Some((&K, &mut T))` if there is a previous entry
    /// * `None` if the cursor is at the first entry or positioned at a null
    ///   entry
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert_tail("a", 1);
    /// map.insert_tail("b", 2);
    ///
    /// let mut cursor = map.tail_cursor_mut();
    /// if let Some((key, value)) = cursor.prev_mut() {
    ///     assert_eq!(key, &"a");
    ///     *value = 10;
    /// }
    /// assert_eq!(map.get(&"a"), Some(&10));
    /// ```
    pub fn prev_mut(&mut self) -> Option<(&K, &mut T)> {
        let ptr = self.prev_ptr()?;
        self.map.ptr_get_entry_mut(ptr)
    }
}

impl<K, T, S> Index<&K> for LinkedHashMap<K, T, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    type Output = T;

    fn index(&self, key: &K) -> &Self::Output {
        self.get(key).expect("no entry found for key")
    }
}

impl<K, T, S> IndexMut<&K> for LinkedHashMap<K, T, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    fn index_mut(&mut self, key: &K) -> &mut Self::Output {
        self.get_mut(key).expect("no entry found for key")
    }
}

impl<K, T, S> Index<Ptr> for LinkedHashMap<K, T, S> {
    type Output = T;

    fn index(&self, index: Ptr) -> &Self::Output {
        &self.nodes[index].value
    }
}

impl<K, T, S> IndexMut<Ptr> for LinkedHashMap<K, T, S> {
    fn index_mut(&mut self, index: Ptr) -> &mut Self::Output {
        &mut self.nodes[index].value
    }
}

/// A view into a single entry in a map, which may either be vacant or occupied.
///
/// This enum is constructed from the [`entry`] method on [`LinkedHashMap`].
///
/// [`entry`]: LinkedHashMap::entry
///
/// # Examples
///
/// ```
/// use tether_map::Entry;
/// use tether_map::LinkedHashMap;
///
/// let mut map = LinkedHashMap::new();
///
/// match map.entry("key") {
///     Entry::Vacant(entry) => {
///         entry.insert_tail("value");
///     }
///     Entry::Occupied(entry) => {
///         println!("Key already exists: {}", entry.get());
///     }
/// }
/// ```
pub enum Entry<'a, K, V> {
    /// An occupied entry.
    Occupied(OccupiedEntry<'a, K, V>),

    /// A vacant entry.
    Vacant(VacantEntry<'a, K, V>),
}

impl<'a, K, V> Entry<'a, K, V>
where
    K: Hash + Eq,
{
    /// Ensures a value is in the entry by inserting the provided default if
    /// vacant, and returns a mutable reference to the value in the entry.
    ///
    /// When inserting, the new entry is linked at the tail (end) of the list,
    /// matching the behavior of `insert`/`insert_tail` for new keys.
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(v) => v.insert_tail(default).1,
        }
    }

    /// If the entry is occupied, applies the provided function to the value in
    /// place. Returns the entry for further chaining.
    pub fn and_modify<F>(self, f: F) -> Self
    where
        F: FnOnce(&mut V),
    {
        if let Entry::Occupied(mut e) = self {
            f(e.get_mut());
            Entry::Occupied(e)
        } else {
            self
        }
    }
}

/// A view into an occupied entry in a `LinkedHashMap`.
///
/// It is part of the [`Entry`] enum.
///
/// # Examples
///
/// ```
/// use tether_map::Entry;
/// use tether_map::LinkedHashMap;
///
/// let mut map = LinkedHashMap::new();
/// map.insert("key", "value");
///
/// if let Entry::Occupied(entry) = map.entry("key") {
///     println!("Found key: {}, value: {}", entry.key(), entry.get());
/// }
/// ```
pub struct OccupiedEntry<'a, K, T> {
    entry: hash_table::OccupiedEntry<'a, NonNull<LLSlot<K, T>>>,
    head: &'a mut *mut LLSlot<K, T>,
    tail: &'a mut *mut LLSlot<K, T>,
    arena: &'a mut Arena<K, T>,
}

impl<'a, K, T> OccupiedEntry<'a, K, T> {
    /// Returns a reference to the value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::Entry;
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert("key", 42);
    ///
    /// match map.entry("key") {
    ///     Entry::Occupied(entry) => {
    ///         assert_eq!(entry.get(), &42);
    ///     }
    ///     Entry::Vacant(_) => unreachable!(),
    /// }
    /// ```
    pub fn get(&self) -> &T {
        // SAFETY: Node was obtained from our own arena and is guaranteed to be valid
        unsafe { &self.entry.get().as_ref().data.assume_init_ref().value }
    }

    /// Returns a mutable reference to the value in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::Entry;
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert("key", 42);
    ///
    /// match map.entry("key") {
    ///     Entry::Occupied(mut entry) => {
    ///         *entry.get_mut() = 100;
    ///     }
    ///     Entry::Vacant(_) => unreachable!(),
    /// }
    /// assert_eq!(map.get(&"key"), Some(&100));
    /// ```
    pub fn get_mut(&mut self) -> &mut T {
        // SAFETY: Node was obtained from our own arena and is guaranteed to be valid
        unsafe { &mut self.entry.get_mut().as_mut().data.assume_init_mut().value }
    }

    /// Consumes the occupied entry and returns a mutable reference to the
    /// value.
    ///
    /// The returned reference is tied to the lifetime of the original map
    /// borrow.
    pub fn into_mut(self) -> &'a mut T {
        let OccupiedEntry { entry, .. } = self;
        unsafe {
            // SAFETY: Node was obtained from our own arena and is guaranteed to be valid,
            // we tie the lifetime to the arena so it cannot outlive the arena.
            &mut entry.into_mut().as_mut().data.assume_init_mut().value
        }
    }

    /// Replaces the entry's value and returns the old value without moving the
    /// entry's position.
    ///
    /// Unlike `insert()`, this method does not affect the entry's position in
    /// the linked list.
    ///
    /// # Arguments
    ///
    /// * `value` - The new value to insert
    ///
    /// # Returns
    ///
    /// The previous value that was replaced
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::Entry;
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert("a", 1);
    /// map.insert("b", 2);
    ///
    /// match map.entry("a") {
    ///     Entry::Occupied(entry) => {
    ///         let old = entry.insert_no_move(10);
    ///         assert_eq!(old, 1);
    ///     }
    ///     Entry::Vacant(_) => unreachable!(),
    /// }
    ///
    /// // Order remains: a, b ("a" was not moved)
    /// let entries: Vec<_> = map.iter().collect();
    /// assert_eq!(entries, [(&"a", &10), (&"b", &2)]);
    /// ```
    pub fn insert_no_move(mut self, value: T) -> T {
        // SAFETY: Node was obtained from our own arena and is guaranteed to be valid
        unsafe {
            core::mem::replace(
                &mut self.entry.get_mut().as_mut().data.assume_init_mut().value,
                value,
            )
        }
    }

    /// Returns the pointer to this entry.
    ///
    /// The pointer can be used for direct access operations or cursor
    /// positioning.
    ///
    /// # Returns
    ///
    /// The pointer to the entry
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::Entry;
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert("key", 42);
    ///
    /// match map.entry("key") {
    ///     Entry::Occupied(entry) => {
    ///         let ptr = entry.ptr();
    ///         assert_eq!(map.ptr_get(ptr), Some(&42));
    ///     }
    ///     Entry::Vacant(_) => unreachable!(),
    /// }
    /// ```
    pub fn ptr(&self) -> Ptr {
        // SAFETY: Node was obtained from our own arena and is guaranteed to be valid
        unsafe { self.entry.get().as_ref().this }
    }

    /// Returns a reference to the key in the entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::Entry;
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert("key", 42);
    ///
    /// match map.entry("key") {
    ///     Entry::Occupied(entry) => {
    ///         assert_eq!(entry.key(), &"key");
    ///     }
    ///     Entry::Vacant(_) => unreachable!(),
    /// }
    /// ```
    pub fn key(&self) -> &K {
        // SAFETY: Node was obtained from our own arena and is guaranteed to be valid
        unsafe { &self.entry.get().as_ref().data.assume_init_ref().key }
    }

    /// Replaces the entry's value and returns the old value.
    ///
    /// This is equivalent to `insert_no_move()` and does not move the entry's
    /// position.
    ///
    /// # Arguments
    ///
    /// * `value` - The new value to insert
    ///
    /// # Returns
    ///
    /// The previous value that was replaced
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::Entry;
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert("key", 42);
    ///
    /// match map.entry("key") {
    ///     Entry::Occupied(entry) => {
    ///         let old = entry.insert(100);
    ///         assert_eq!(old, 42);
    ///     }
    ///     Entry::Vacant(_) => unreachable!(),
    /// }
    /// assert_eq!(map.get(&"key"), Some(&100));
    /// ```
    pub fn insert(self, value: T) -> T {
        self.insert_no_move(value)
    }

    /// Removes the entry from the map and returns the key-value pair.
    ///
    /// This consumes the occupied entry and requires that both the key and
    /// value types implement `Default` for safe removal from the underlying
    /// storage.
    ///
    /// # Returns
    ///
    /// A tuple containing the key and value that were removed
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::Entry;
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert("key", 42);
    ///
    /// match map.entry("key") {
    ///     Entry::Occupied(entry) => {
    ///         let (key, value) = entry.remove_entry();
    ///         assert_eq!(key, "key");
    ///         assert_eq!(value, 42);
    ///     }
    ///     Entry::Vacant(_) => unreachable!(),
    /// }
    /// assert_eq!(map.len(), 0);
    /// ```
    pub fn remove_entry(self) -> (K, T) {
        // SAFETY: We only store valid pointers into our own arena, and self.node was
        // drawn from the known-good list in our own hashtable. We do not access the
        // data in self.node after this.
        let entry = self.entry.remove().0;
        let entry_raw = entry.as_ptr();
        let FreedSlot {
            data,
            prev_raw,
            next_raw,
            ..
        } = unsafe { self.arena.free_and_unlink(entry) };
        if *self.head == entry_raw {
            *self.head = next_raw;
        }
        if *self.tail == entry_raw {
            *self.tail = prev_raw;
        }

        (data.key, data.value)
    }

    /// Removes the entry from the map and returns the value.
    ///
    /// This consumes the occupied entry and requires that both the key and
    /// value types implement `Default` for safe removal from the underlying
    /// storage.
    ///
    /// # Returns
    ///
    /// The value that was removed
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::Entry;
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert("key", 42);
    ///
    /// match map.entry("key") {
    ///     Entry::Occupied(entry) => {
    ///         let value = entry.remove();
    ///         assert_eq!(value, 42);
    ///     }
    ///     Entry::Vacant(_) => unreachable!(),
    /// }
    /// assert_eq!(map.len(), 0);
    /// ```
    pub fn remove(self) -> T {
        self.remove_entry().1
    }
}

/// A view into a vacant entry in a `LinkedHashMap`.
///
/// It is part of the [`Entry`] enum.
///
/// # Examples
///
/// ```
/// use tether_map::Entry;
/// use tether_map::LinkedHashMap;
///
/// let mut map = LinkedHashMap::new();
///
/// if let Entry::Vacant(entry) = map.entry("key") {
///     entry.insert_tail("value");
/// }
/// assert_eq!(map.get(&"key"), Some(&"value"));
/// ```
pub struct VacantEntry<'a, K, T> {
    key: K,
    entry: hash_table::VacantEntry<'a, NonNull<LLSlot<K, T>>>,
    nodes: &'a mut Arena<K, T>,
    head: &'a mut *mut LLSlot<K, T>,
    tail: &'a mut *mut LLSlot<K, T>,
}

impl<'a, K: Hash + Eq, T> VacantEntry<'a, K, T> {
    /// Inserts a new entry at the tail (end) of the linked list.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to insert
    ///
    /// # Returns
    ///
    /// The pointer to the newly inserted entry
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::Entry;
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    ///
    /// match map.entry("new_key") {
    ///     Entry::Vacant(entry) => {
    ///         let value = entry.insert_tail(42).1;
    ///         assert_eq!(*value, 42);
    ///     }
    ///     Entry::Occupied(_) => unreachable!(),
    /// }
    /// ```
    pub fn insert_tail(self, value: T) -> (Ptr, &'a mut T) {
        let after = *self.tail;
        // SAFETY: `after` came from self.tail, which is either null or valid.
        unsafe {
            let mut ptr = self.insert_after_internal(value, after);
            let external_ptr = ptr.as_ref().this;
            (external_ptr, &mut ptr.as_mut().data.assume_init_mut().value)
        }
    }

    /// Inserts a new entry without linking it to the doubly-linked list.
    ///
    /// This is a low-level method that creates an entry in the hash table but
    /// does not link it into the ordered list. The entry will exist in the
    /// map but won't appear in iteration order until it's properly linked. It
    /// can still be accessed either by the returned pointer or by its key.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to insert
    ///
    /// # Returns
    ///
    /// The pointer to the newly created but unlinked entry
    ///
    /// # Note
    ///
    /// This method is primarily for advanced usage. Most users should use
    /// `insert_tail()`, `insert_head()`, or similar methods instead. This is
    /// most useful when you have an entry and want to store data in the map,
    /// but also need to operate on the linked list structure separately without
    /// including the new entry in the list yet. In that case, you can create
    /// the entry with `push_unlinked()` and then later link it in using
    /// methods like `link_as_head()`, or `link_as_tail()`.
    pub fn push_unlinked(self, value: T) -> (Ptr, &'a mut T) {
        let mut ptr = self.nodes.alloc_circular(self.key, value);
        self.entry.insert(ptr);
        // SAFETY: We just allocated ptr above, so it must be valid. We tie the returned
        // mutable reference to the lifetime of self.nodes, so it cannot outlive the
        // arena.
        unsafe {
            let external_ptr = ptr.as_ref().this;
            (external_ptr, &mut ptr.as_mut().data.assume_init_mut().value)
        }
    }

    /// Inserts a new entry immediately after the specified entry.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to insert
    /// * `after` - The pointer to the entry after which to insert
    ///
    /// # Returns
    ///
    /// The pointer to the newly inserted entry
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::Entry;
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// let (ptr1, _) = map.insert_tail_full("first", 1);
    /// map.insert_tail("third", 3);
    ///
    /// match map.entry("second") {
    ///     Entry::Vacant(entry) => {
    ///         entry.insert_after(2, ptr1);
    ///     }
    ///     Entry::Occupied(_) => unreachable!(),
    /// }
    ///
    /// // Order is now: first, second, third
    /// let entries: Vec<_> = map.iter().collect();
    /// assert_eq!(entries, [(&"first", &1), (&"second", &2), (&"third", &3)]);
    /// ```
    pub fn insert_after(self, value: T, after: Ptr) -> (Ptr, &'a mut T) {
        let after = self.nodes.map_ptr(after).map_or(*self.tail, |p| p.as_ptr());

        // SAFETY: `after` was either obtained from self.tail, which is either null or
        // valid, or it was mapped from a user-provided Ptr using self.nodes.map_ptr,
        // which returns None if the Ptr is invalid, so we fall back to self.tail in
        // that case.
        unsafe {
            let mut ptr = self.insert_after_internal(value, after);
            let external_ptr = ptr.as_ref().this;
            (external_ptr, &mut ptr.as_mut().data.assume_init_mut().value)
        }
    }

    // SAFETY: `after` must be either null or a valid pointer into self.nodes.
    unsafe fn insert_after_internal(
        self,
        value: T,
        after: *mut LLSlot<K, T>,
    ) -> NonNull<LLSlot<K, T>> {
        if after.is_null() {
            debug_assert_eq!(*self.head, core::ptr::null_mut());
            debug_assert_eq!(after, core::ptr::null_mut());
            let ptr = self.nodes.alloc_circular(self.key, value);

            *self.head = ptr.as_ptr();
            *self.tail = *self.head;
            self.entry.insert(ptr);
            ptr
        } else {
            let mut after = NonNull::new(after).expect("after pointer is null");
            // SAFETY: Per contract, after is valid, so its next pointer must also be valid.
            let mut after_next = unsafe { after.as_ref().links.next() };
            let ptr = self.nodes.alloc(self.key, value, after, after_next);

            // SAFETY: after and after_next are valid, so updating their pointers is safe.
            unsafe {
                *after.as_mut().links.next_mut() = ptr;
                *after_next.as_mut().links.prev_mut() = ptr;
            }
            self.entry.insert(ptr);

            if *self.tail == after.as_ptr() {
                *self.tail = ptr.as_ptr();
            }
            ptr
        }
    }

    /// Inserts a new entry at the head (beginning) of the linked list.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to insert
    ///
    /// # Returns
    ///
    /// The pointer to the newly inserted entry
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::Entry;
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert_tail("second", 2);
    ///
    /// match map.entry("first") {
    ///     Entry::Vacant(entry) => {
    ///         let ptr = entry.insert_head(1);
    ///         assert_eq!(map.ptr_get(ptr), Some(&1));
    ///     }
    ///     Entry::Occupied(_) => unreachable!(),
    /// }
    ///
    /// // Order is now: first, second
    /// let entries: Vec<_> = map.iter().collect();
    /// assert_eq!(entries, [(&"first", &1), (&"second", &2)]);
    /// ```
    pub fn insert_head(self, value: T) -> Ptr {
        let ptr = *self.head;
        // SAFETY: `ptr` came from self.head, which is either null or valid.
        unsafe { self.insert_before_internal(value, ptr).as_ref().this }
    }

    /// Inserts a new entry immediately before the specified entry.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to insert
    /// * `before` - The pointer to the entry before which to insert
    ///
    /// # Returns
    ///
    /// The pointer to the newly inserted entry
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::Entry;
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    /// map.insert_tail("first", 1);
    /// let (ptr3, _) = map.insert_tail_full("third", 3);
    ///
    /// match map.entry("second") {
    ///     Entry::Vacant(entry) => {
    ///         entry.insert_before(2, ptr3);
    ///     }
    ///     Entry::Occupied(_) => unreachable!(),
    /// }
    ///
    /// // Order is now: first, second, third
    /// let entries: Vec<_> = map.iter().collect();
    /// assert_eq!(entries, [(&"first", &1), (&"second", &2), (&"third", &3)]);
    /// ```
    pub fn insert_before(self, value: T, before: Ptr) -> (Ptr, &'a mut T) {
        let before = self
            .nodes
            .map_ptr(before)
            .map_or(*self.head, |p| p.as_ptr());

        // SAFETY: `before` was either obtained from self.head, which is either null or
        // valid, or it was mapped from a user-provided Ptr using self.nodes.map_ptr
        // which returns None if the Ptr is invalid, so we fall back to self.head in
        // that case.
        unsafe {
            let mut ptr = self.insert_before_internal(value, before);
            let external_ptr = ptr.as_ref().this;
            (external_ptr, &mut ptr.as_mut().data.assume_init_mut().value)
        }
    }

    /// # Safety
    ///
    /// `before` must be either null or a valid pointer into self.nodes.
    unsafe fn insert_before_internal(
        self,
        value: T,
        before: *mut LLSlot<K, T>,
    ) -> NonNull<LLSlot<K, T>> {
        if before.is_null() {
            debug_assert_eq!(*self.tail, core::ptr::null_mut());
            debug_assert_eq!(before, core::ptr::null_mut());
            let ptr = self.nodes.alloc_circular(self.key, value);

            *self.head = ptr.as_ptr();
            *self.tail = *self.head;
            self.entry.insert(ptr);
            ptr
        } else {
            let mut before = NonNull::new(before).expect("before pointer is null");
            // SAFETY: Per contract, before is valid, so its prev pointer must also be
            // valid.
            let mut before_prev = unsafe { before.as_ref().links.prev() };
            let ptr = self.nodes.alloc(self.key, value, before_prev, before);

            // SAFETY: before and before_prev are valid, so updating their pointers is safe.
            unsafe {
                *before.as_mut().links.prev_mut() = ptr;
                *before_prev.as_mut().links.next_mut() = ptr;
            }
            self.entry.insert(ptr);

            if *self.head == before.as_ptr() {
                *self.head = ptr.as_ptr();
            }
            ptr
        }
    }

    /// Consumes this vacant entry and returns the key.
    ///
    /// This method allows you to retrieve the key without inserting a value,
    /// which can be useful when the insertion is conditional.
    ///
    /// # Returns
    ///
    /// The key that would have been inserted
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::Entry;
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map: LinkedHashMap<&str, i32> = LinkedHashMap::new();
    ///
    /// match map.entry("key") {
    ///     Entry::Vacant(entry) => {
    ///         let key = entry.into_key();
    ///         assert_eq!(key, "key");
    ///         // Entry was not inserted
    ///     }
    ///     Entry::Occupied(_) => unreachable!(),
    /// }
    /// assert_eq!(map.len(), 0);
    /// ```
    pub fn into_key(self) -> K {
        self.key
    }

    /// Returns a reference to the key that would be inserted.
    ///
    /// # Examples
    ///
    /// ```
    /// use tether_map::Entry;
    /// use tether_map::LinkedHashMap;
    ///
    /// let mut map = LinkedHashMap::new();
    ///
    /// match map.entry("key") {
    ///     Entry::Vacant(entry) => {
    ///         assert_eq!(entry.key(), &"key");
    ///         entry.insert_tail(42);
    ///     }
    ///     Entry::Occupied(_) => unreachable!(),
    /// }
    /// ```
    pub fn key(&self) -> &K {
        &self.key
    }
}

#[derive(Debug, Clone, Copy)]
/// An iterator over the entries of a `LinkedHashMap`.
///
/// This struct is created by the [`iter`] method on [`LinkedHashMap`]. See its
/// documentation for more.
///
/// [`iter`]: LinkedHashMap::iter
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
/// for (key, value) in map.iter() {
///     println!("{}: {}", key, value);
/// }
/// ```
pub struct Iter<'a, K, T, S> {
    forward_ptr: *mut LLSlot<K, T>,
    reverse_ptr: *mut LLSlot<K, T>,
    _map: PhantomData<&'a LinkedHashMap<K, T, S>>,
}

impl<'a, K, T, S> Iterator for Iter<'a, K, T, S> {
    type Item = (&'a K, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.forward_ptr.is_null() || self.reverse_ptr.is_null() {
            return None;
        }

        let ptr = self.forward_ptr;
        // SAFETY: We are iterating over our own pointers which we know are valid.
        // We tie the lifetime to our immutable borrow of the map.
        let as_ref = unsafe { ptr.as_ref().unwrap() };
        if self.forward_ptr == self.reverse_ptr {
            self.forward_ptr = core::ptr::null_mut();
            self.reverse_ptr = core::ptr::null_mut();
        } else {
            // SAFETY: We are iterating over our own pointers which we know are valid.
            self.forward_ptr = unsafe { as_ref.links.next().as_ptr() };
        }

        // SAFETY: We are iterating over our own pointers which we know are valid.
        unsafe {
            let data = as_ref.data.assume_init_ref();
            Some((&data.key, &data.value))
        }
    }
}

impl<'a, K, T, S> DoubleEndedIterator for Iter<'a, K, T, S> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.forward_ptr.is_null() || self.reverse_ptr.is_null() {
            return None;
        }

        let ptr = self.reverse_ptr;
        // SAFETY: We are iterating over our own pointers which we know are valid.
        // We tie the lifetime to our immutable borrow of the map.
        let as_ref = unsafe { ptr.as_ref().unwrap() };
        if self.reverse_ptr == self.forward_ptr {
            self.reverse_ptr = core::ptr::null_mut();
            self.forward_ptr = core::ptr::null_mut();
        } else {
            // SAFETY: We are iterating over our own pointers which we know are valid.
            self.reverse_ptr = unsafe { as_ref.links.prev().as_ptr() };
        }

        // SAFETY: We are iterating over our own pointers which we know are valid.
        unsafe {
            let data = as_ref.data.assume_init_ref();
            Some((&data.key, &data.value))
        }
    }
}

#[derive(Debug)]
/// An owning iterator over the entries of a `LinkedHashMap`.
///
/// This struct is created by the [`into_iter`] method on [`LinkedHashMap`]
/// (provided by the [`IntoIterator`] trait). See its documentation for more.
///
/// [`into_iter`]: IntoIterator::into_iter
/// [`IntoIterator`]: core::iter::IntoIterator
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
/// for (key, value) in map {
///     println!("{}: {}", key, value);
/// }
/// ```
pub struct IntoIter<K, T> {
    nodes: Arena<K, T>,
    forward_ptr: *mut LLSlot<K, T>,
    reverse_ptr: *mut LLSlot<K, T>,
}

impl<K, T> Iterator for IntoIter<K, T> {
    type Item = (K, T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.forward_ptr.is_null() || self.reverse_ptr.is_null() {
            return None;
        }

        let ptr = self.forward_ptr;
        if self.forward_ptr == self.reverse_ptr {
            self.forward_ptr = core::ptr::null_mut();
            self.reverse_ptr = core::ptr::null_mut();
        } else {
            // SAFETY: We only store valid pointers into our own arena.
            self.forward_ptr = unsafe { ptr.as_ref().unwrap().links.next().as_ptr() };
        }

        // SAFETY: We only store valid pointers into our own arena. We do not access the
        // pointer after this call.
        let data = unsafe { self.nodes.free_and_unlink(NonNull::new(ptr).unwrap()).data };
        Some((data.key, data.value))
    }
}

impl<K, T> DoubleEndedIterator for IntoIter<K, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.forward_ptr.is_null() || self.reverse_ptr.is_null() {
            return None;
        }

        let ptr = self.reverse_ptr;
        if self.reverse_ptr == self.forward_ptr {
            self.reverse_ptr = core::ptr::null_mut();
            self.forward_ptr = core::ptr::null_mut();
        } else {
            // SAFETY: We only store valid pointers into our own arena.
            self.reverse_ptr = unsafe { ptr.as_ref().unwrap().links.prev().as_ptr() };
        }

        // SAFETY: We only store valid pointers into our own arena. We do not access the
        // pointer after this call.
        let data = unsafe { self.nodes.free_and_unlink(NonNull::new(ptr).unwrap()).data };
        Some((data.key, data.value))
    }
}

#[derive(Debug)]
/// A mutable iterator over the entries of a `LinkedHashMap`.
///
/// This struct is created by the [`iter_mut`] method on [`LinkedHashMap`]. See
/// its documentation for more.
///
/// [`iter_mut`]: LinkedHashMap::iter_mut
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
/// for (key, value) in map.iter_mut() {
///     *value *= 2;
/// }
///
/// assert_eq!(map.get(&"a"), Some(&2));
/// assert_eq!(map.get(&"b"), Some(&4));
/// ```
pub struct IterMut<'a, K, T> {
    forward_ptr: *mut LLSlot<K, T>,
    reverse_ptr: *mut LLSlot<K, T>,
    _nodes: PhantomData<&'a mut Arena<K, T>>,
}

#[derive(Debug)]
/// A mutable iterator over the values of a `LinkedHashMap`.
///
/// This iterator yields `&mut T` values in the order they were inserted into
/// the map. It is created by the [`values_mut`] method on `LinkedHashMap`.
///
/// [`values_mut`]: LinkedHashMap::values_mut
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
/// for value in map.values_mut() {
///     *value *= 10;
/// }
///
/// assert_eq!(map.get(&"a"), Some(&10));
/// assert_eq!(map.get(&"b"), Some(&20));
/// ```
pub struct ValuesMut<'a, K, T> {
    iter: IterMut<'a, K, T>,
}

impl<'a, K, T> Iterator for IterMut<'a, K, T> {
    type Item = (&'a K, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.forward_ptr.is_null() || self.reverse_ptr.is_null() {
            return None;
        }

        // SAFETY: We yield exactly one item per ptr. Our ptrs are unique. We trust the
        // pointers we are iterating over came from our arena. We tie the lifetime to
        // our mutable borrow of the arena.
        let node_mut = unsafe { self.forward_ptr.as_mut().unwrap() };
        if self.forward_ptr == self.reverse_ptr {
            self.forward_ptr = core::ptr::null_mut();
            self.reverse_ptr = core::ptr::null_mut();
        } else {
            // SAFETY: We are iterating over our own pointers which we know are valid.
            self.forward_ptr = unsafe { node_mut.links.next().as_ptr() };
        }

        // SAFETY: See above.
        unsafe { Some(node_mut.data.assume_init_mut().key_value_mut()) }
    }
}

impl<'a, K, T> DoubleEndedIterator for IterMut<'a, K, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.forward_ptr.is_null() || self.reverse_ptr.is_null() {
            return None;
        }

        // SAFETY: We yield exactly one item per ptr. Our ptrs are unique. We trust the
        // pointers we are iterating over came from our arena. We tie the lifetime to
        // our mutable borrow of the arena.
        let node_mut = unsafe { self.reverse_ptr.as_mut().unwrap() };
        if self.reverse_ptr == self.forward_ptr {
            self.reverse_ptr = core::ptr::null_mut();
            self.forward_ptr = core::ptr::null_mut();
        } else {
            // SAFETY: We are iterating over our own pointers which we know are valid.
            self.reverse_ptr = unsafe { node_mut.links.prev().as_ptr() };
        }

        // SAFETY: See above.
        unsafe { Some(node_mut.data.assume_init_mut().key_value_mut()) }
    }
}

impl<'a, K, T> Iterator for ValuesMut<'a, K, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(_, v)| v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, K, T> DoubleEndedIterator for ValuesMut<'a, K, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|(_, v)| v)
    }
}

#[cfg(test)]
mod tests {
    use alloc::format;
    use alloc::string::ToString;
    use alloc::vec;
    use core::assert_eq;
    use core::panic;

    use super::*;
    use crate::LinkedHashMap;

    #[test]
    fn test_new_and_default() {
        let map: LinkedHashMap<i32, Vec<i32>> = LinkedHashMap::default();
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
        assert_eq!(map.head_ptr(), None);
        assert_eq!(map.tail_ptr(), None);
    }

    #[test]
    fn test_with_capacity() {
        let map: LinkedHashMap<i32, Vec<i32>> = LinkedHashMap::with_capacity(10);
        assert!(map.is_empty());
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn test_clear() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(1, vec![1]);
        map.insert_tail(2, vec![2]);

        assert_eq!(map.len(), 2);
        assert!(!map.is_empty());

        map.clear();

        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        assert_eq!(map.head_ptr(), None);
        assert_eq!(map.tail_ptr(), None);
    }

    #[test]
    fn test_insert_tail() {
        let mut map = LinkedHashMap::default();

        let result = map.insert_tail(1, vec![1]);
        assert_eq!(result, None);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&1), Some(&vec![1]));
        assert_eq!(map.head_ptr(), map.tail_ptr());

        let result = map.insert_tail(2, vec![2]);
        assert_eq!(result, None);
        assert_eq!(map.len(), 2);
        assert_ne!(map.head_ptr(), map.tail_ptr());

        map.insert_tail(3, vec![3]);
        assert_eq!(map.len(), 3);

        let items: Vec<_> = map.iter().collect();
        assert_eq!(items, vec![(&1, &vec![1]), (&2, &vec![2]), (&3, &vec![3])]);

        let result = map.insert_tail(2, vec![2]);
        assert_eq!(result, Some(vec![2]));
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&2), Some(&vec![2]));

        let items: Vec<_> = map.iter().collect();
        assert_eq!(items, vec![(&1, &vec![1]), (&3, &vec![3]), (&2, &vec![2])]);
    }

    #[test]
    fn test_insert_head() {
        let mut map = LinkedHashMap::default();

        let result = map.insert_head(1, vec![1]);
        assert_eq!(result, None);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&1), Some(&vec![1]));
        assert_eq!(map.head_ptr(), map.tail_ptr());

        let result = map.insert_head(2, vec![2]);
        assert_eq!(result, None);
        assert_eq!(map.len(), 2);
        assert_ne!(map.head_ptr(), map.tail_ptr());

        map.insert_head(3, vec![3]);
        assert_eq!(map.len(), 3);

        let items: Vec<_> = map.iter().collect();
        assert_eq!(items, vec![(&3, &vec![3]), (&2, &vec![2]), (&1, &vec![1])]);

        let result = map.insert_head(2, vec![2]);
        assert_eq!(result, Some(vec![2]));
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&2), Some(&vec![2]));

        let items: Vec<_> = map.iter().collect();
        assert_eq!(items, vec![(&2, &vec![2]), (&3, &vec![3]), (&1, &vec![1])]);
    }

    #[test]
    fn test_mixed_insertion() {
        let mut map = LinkedHashMap::default();

        map.insert_tail(1, "one");
        map.insert_head(2, "two");
        map.insert_tail(3, "three");
        map.insert_head(4, "four");

        let items: Vec<_> = map.iter().collect();
        assert_eq!(
            items,
            vec![(&4, &"four"), (&2, &"two"), (&1, &"one"), (&3, &"three")]
        );
        assert_eq!(map.len(), 4);
    }

    #[test]
    fn test_get_operations() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(1, vec![1]);
        map.insert_tail(2, vec![2]);
        map.insert_tail(3, vec![3]);

        assert_eq!(map.get(&1), Some(&vec![1]));
        assert_eq!(map.get(&2), Some(&vec![2]));
        assert_eq!(map.get(&3), Some(&vec![3]));
        assert_eq!(map.get(&4), None);

        let value = map.get_mut(&2).unwrap();
        *value = vec![2];
        assert_eq!(map.get(&2), Some(&vec![2]));

        assert!(map.contains_key(&1));
        assert!(map.contains_key(&2));
        assert!(map.contains_key(&3));
        assert!(!map.contains_key(&4));
        assert!(!map.contains_key(&0));
    }

    #[test]
    fn test_get_ptr_operations() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(1, vec![1]);
        map.insert_tail(2, vec![2]);

        let ptr1 = map.get_ptr(&1).unwrap();
        let ptr2 = map.get_ptr(&2).unwrap();
        assert_ne!(ptr1, ptr2);
        assert_eq!(map.get_ptr(&99), None);

        assert_eq!(map.ptr_get(ptr1), Some(&vec![1]));
        assert_eq!(map.ptr_get(ptr2), Some(&vec![2]));

        let value = map.ptr_get_mut(ptr1).unwrap();
        *value = vec![1];
        assert_eq!(map.ptr_get(ptr1), Some(&vec![1]));

        let (key, value) = map.ptr_get_entry(ptr1).unwrap();
        assert_eq!(key, &1);
        assert_eq!(value, &vec![1]);

        let (key, value) = map.ptr_get_entry_mut(ptr2).unwrap();
        assert_eq!(key, &2);
        *value = vec![2];
        assert_eq!(map.get(&2), Some(&vec![2]));

        assert_eq!(map.ptr_get_key(ptr1), Some(&1));
        assert_eq!(map.ptr_get_key(ptr2), Some(&2));

        assert!(map.contains_ptr(ptr1));
        assert!(map.contains_ptr(ptr2));
    }

    #[test]
    fn test_index_operations() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(1, vec![1]);
        map.insert_tail(2, vec![2]);

        let ptr1 = map.get_ptr(&1).unwrap();
        let ptr2 = map.get_ptr(&2).unwrap();

        assert_eq!(&map[ptr1], &vec![1]);
        assert_eq!(&map[ptr2], &vec![2]);

        map[ptr1] = vec![1];
        assert_eq!(&map[ptr1], &vec![1]);
        assert_eq!(map.get(&1), Some(&vec![1]));
    }

    #[test]
    fn test_remove_by_key() {
        let mut map: LinkedHashMap<i32, Vec<i32>> = LinkedHashMap::default();
        map.insert_tail(1, vec![1]);
        map.insert_tail(2, vec![2]);
        map.insert_tail(3, vec![3]);

        let removed = map.remove_with_ptr(&2).unwrap().1;
        assert_eq!(
            removed,
            RemovedEntry {
                key: 2,
                value: vec![2],
                prev: map.get_ptr(&1),
                next: map.get_ptr(&3),
            }
        );
        assert_eq!(map.len(), 2);
        assert!(!map.contains_key(&2));

        let items: Vec<_> = map.iter().collect();
        assert_eq!(items, vec![(&1, &vec![1]), (&3, &vec![3])]);

        let removed = map.remove_with_ptr(&1).unwrap().1;
        assert_eq!(
            removed,
            RemovedEntry {
                key: 1,
                value: vec![1],
                prev: map.get_ptr(&3),
                next: map.get_ptr(&3),
            }
        );
        assert_eq!(map.len(), 1);
        assert_eq!(map.head_ptr(), map.tail_ptr());

        let removed = map.remove_with_ptr(&3).unwrap().1;
        assert_eq!(
            removed,
            RemovedEntry {
                key: 3,
                value: vec![3],
                prev: None,
                next: None,
            }
        );
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());
        assert_eq!(map.head_ptr(), None);
        assert_eq!(map.tail_ptr(), None);

        let removed = map.remove(&1);
        assert_eq!(removed, None);
    }

    #[test]
    fn test_remove_by_ptr() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(1, vec![1]);
        map.insert_tail(2, vec![2]);
        map.insert_tail(3, vec![3]);

        let removed = map.remove_ptr(map.get_ptr(&2).unwrap());
        assert_eq!(
            removed,
            Some(RemovedEntry {
                key: 2,
                value: vec![2],
                prev: map.get_ptr(&1),
                next: map.get_ptr(&3),
            })
        );
        assert_eq!(map.len(), 2);
        assert!(!map.contains_key(&2));

        let removed = map.remove_ptr(map.get_ptr(&1).unwrap());
        assert_eq!(
            removed,
            Some(RemovedEntry {
                key: 1,
                value: vec![1],
                prev: map.get_ptr(&3),
                next: map.get_ptr(&3),
            })
        );
        assert_eq!(map.len(), 1);
        assert_eq!(map.head_ptr(), map.get_ptr(&3));
        assert_eq!(map.tail_ptr(), map.get_ptr(&3));

        let removed = map.remove_ptr(map.get_ptr(&3).unwrap());
        assert_eq!(
            removed,
            Some(RemovedEntry {
                key: 3,
                value: vec![3],
                prev: None,
                next: None,
            })
        );
        assert!(map.is_empty());
    }

    #[test]
    fn test_remove_single_element() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(42, vec![42]);

        assert_eq!(map.len(), 1);
        assert_eq!(map.head_ptr(), map.tail_ptr());

        let removed = map.remove_with_ptr(&42).unwrap().1;
        assert_eq!(
            removed,
            RemovedEntry {
                key: 42,
                value: vec![42],
                prev: None,
                next: None,
            }
        );
        assert!(map.is_empty());
        assert_eq!(map.head_ptr(), None);
        assert_eq!(map.tail_ptr(), None);
    }

    #[test]
    fn test_remove_head_and_tail() {
        let mut map = LinkedHashMap::default();
        for i in 1..=5 {
            map.insert_tail(i, vec![1]);
        }

        let removed = map.remove_with_ptr(&1).unwrap().1;
        assert_eq!(
            removed,
            RemovedEntry {
                key: 1,
                value: vec![1],
                prev: map.get_ptr(&5),
                next: map.get_ptr(&2),
            }
        );
        assert_eq!(map.tail_ptr(), map.get_ptr(&5));

        let removed = map.remove_with_ptr(&5).unwrap().1;
        assert_eq!(
            removed,
            RemovedEntry {
                key: 5,
                value: vec![1],
                prev: map.get_ptr(&4),
                next: map.get_ptr(&2),
            }
        );
        assert_eq!(map.len(), 3);

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![2, 3, 4]);
    }

    #[test]
    fn test_move_to_head() {
        let mut map = LinkedHashMap::default();
        for i in 1..=4 {
            map.insert_tail(i, format!("value{}", i));
        }

        let ptr3 = map.get_ptr(&3).unwrap();

        map.move_to_head(ptr3);

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![3, 1, 2, 4]);
        assert_eq!(map.head_ptr(), Some(ptr3));

        let old_head = map.head_ptr().unwrap();
        map.move_to_head(old_head);
        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![3, 1, 2, 4]);

        let ptr4 = map.get_ptr(&4).unwrap();
        map.move_to_head(ptr4).unwrap();

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![4, 3, 1, 2]);
        assert_eq!(map.head_ptr(), Some(ptr4));
    }

    #[test]
    fn test_move_to_tail() {
        let mut map = LinkedHashMap::default();
        for i in 1..=4 {
            map.insert_tail(i, format!("value{}", i));
        }

        let ptr2 = map.get_ptr(&2).unwrap();

        map.move_to_tail(ptr2);

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![1, 3, 4, 2]);
        assert_eq!(map.tail_ptr(), Some(ptr2));

        let old_tail = map.tail_ptr().unwrap();
        map.move_to_tail(old_tail);
        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![1, 3, 4, 2]);

        let ptr1 = map.get_ptr(&1).unwrap();
        map.move_to_tail(ptr1);

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![3, 4, 2, 1]);
        assert_eq!(map.tail_ptr(), Some(ptr1));
    }

    #[test]
    fn test_move_after() {
        let mut map = LinkedHashMap::default();
        for i in 1..=5 {
            map.insert_tail(i, format!("value{}", i));
        }

        let ptr1 = map.get_ptr(&1).unwrap();
        let ptr3 = map.get_ptr(&3).unwrap();
        let ptr5 = map.get_ptr(&5).unwrap();

        map.move_after(ptr5, ptr1);
        assert_eq!(map.next_ptr(ptr1), Some(ptr5));
        assert_eq!(map.prev_ptr(ptr5), Some(ptr1));
        assert_eq!(map.next_ptr(ptr5), map.get_ptr(&2));
        assert_eq!(map.prev_ptr(map.get_ptr(&2).unwrap()), Some(ptr5));
        assert_eq!(map.next_ptr(ptr3), map.get_ptr(&4));
        assert_eq!(map.prev_ptr(map.get_ptr(&4).unwrap()), Some(ptr3));

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![1, 5, 2, 3, 4]);

        let ptr2 = map.get_ptr(&2).unwrap();
        let ptr4 = map.get_ptr(&4).unwrap();
        map.move_after(ptr2, ptr4);
        assert_eq!(map.next_ptr(ptr4), Some(ptr2));
        assert_eq!(map.prev_ptr(ptr2), Some(ptr4));
        assert_eq!(map.next_ptr(ptr5), Some(ptr3));
        assert_eq!(map.prev_ptr(ptr3), Some(ptr5));

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![1, 5, 3, 4, 2]);

        map.move_after(ptr3, ptr3);
        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![1, 5, 3, 4, 2]);

        map.move_after(ptr4, ptr3);
        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![1, 5, 3, 4, 2]);
    }

    #[test]
    fn test_move_before() {
        let mut map = LinkedHashMap::default();
        for i in 1..=5 {
            map.insert_tail(i, format!("value{}", i));
        }

        let ptr1 = map.get_ptr(&1).unwrap();
        let ptr3 = map.get_ptr(&3).unwrap();
        let ptr5 = map.get_ptr(&5).unwrap();

        map.move_before(ptr5, ptr3);

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![1, 2, 5, 3, 4]);

        let ptr4 = map.get_ptr(&4).unwrap();
        map.move_before(ptr1, ptr4);

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![2, 5, 3, 1, 4]);

        map.move_before(ptr3, ptr3);
        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![2, 5, 3, 1, 4]);

        let ptr2 = map.get_ptr(&2).unwrap();
        map.move_before(ptr2, ptr5);
        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![2, 5, 3, 1, 4]);
    }

    #[test]
    fn test_pointer_navigation() {
        let mut map = LinkedHashMap::default();
        for i in 1..=3 {
            map.insert_tail(i, format!("value{}", i));
        }

        let ptr1 = map.get_ptr(&1).unwrap();
        let ptr2 = map.get_ptr(&2).unwrap();
        let ptr3 = map.get_ptr(&3).unwrap();

        assert_eq!(map.next_ptr(ptr1), Some(ptr2));
        assert_eq!(map.next_ptr(ptr2), Some(ptr3));
        assert_eq!(map.next_ptr(ptr3), Some(ptr1));

        assert_eq!(map.prev_ptr(ptr1), Some(ptr3));
        assert_eq!(map.prev_ptr(ptr2), Some(ptr1));
        assert_eq!(map.prev_ptr(ptr3), Some(ptr2));
    }

    #[test]
    fn test_move_operations_edge_cases() {
        let mut map = LinkedHashMap::default();

        map.insert_tail(1, "one");
        let ptr1 = map.get_ptr(&1).unwrap();

        map.move_to_head(ptr1);
        assert_eq!(map.len(), 1);
        assert_eq!(map.head_ptr(), Some(ptr1));
        assert_eq!(map.tail_ptr(), Some(ptr1));

        map.move_to_tail(ptr1);
        assert_eq!(map.len(), 1);
        assert_eq!(map.head_ptr(), Some(ptr1));
        assert_eq!(map.tail_ptr(), Some(ptr1));
    }

    #[test]
    fn test_iter() {
        let mut map = LinkedHashMap::default();

        let items: Vec<_> = map.iter().collect();
        assert_eq!(items, vec![]);

        for i in 1..=4 {
            map.insert_tail(i, vec![i]);
        }

        let items: Vec<_> = map.iter().collect();
        assert_eq!(
            items,
            vec![
                (&1, &vec![1]),
                (&2, &vec![2]),
                (&3, &vec![3]),
                (&4, &vec![4])
            ]
        );

        map.insert_head(0, vec![0]);
        let items: Vec<_> = map.iter().collect();
        assert_eq!(
            items,
            vec![
                (&0, &vec![0]),
                (&1, &vec![1]),
                (&2, &vec![2]),
                (&3, &vec![3]),
                (&4, &vec![4])
            ]
        );
    }

    #[test]
    fn test_iter_rev() {
        let mut map = LinkedHashMap::default();

        let items: Vec<_> = map.iter().rev().collect();
        assert_eq!(items, vec![]);

        for i in 1..=4 {
            map.insert_tail(i, format!("value{}", i));
        }

        let items: Vec<_> = map.iter().rev().collect();
        assert_eq!(
            items,
            vec![
                (&4, &"value4".to_string()),
                (&3, &"value3".to_string()),
                (&2, &"value2".to_string()),
                (&1, &"value1".to_string())
            ]
        );

        map.insert_head(0, "value0".to_string());
        let items: Vec<_> = map.iter().rev().collect();
        assert_eq!(
            items,
            vec![
                (&4, &"value4".to_string()),
                (&3, &"value3".to_string()),
                (&2, &"value2".to_string()),
                (&1, &"value1".to_string()),
                (&0, &"value0".to_string())
            ]
        );
    }

    #[test]
    fn test_into_iter() {
        let mut map = LinkedHashMap::default();

        for i in 1..=4 {
            map.insert_tail(i, format!("value{}", i));
        }

        let items: Vec<_> = map.into_iter().collect();
        assert_eq!(
            items,
            vec![
                (1, "value1".to_string()),
                (2, "value2".to_string()),
                (3, "value3".to_string()),
                (4, "value4".to_string())
            ]
        );
    }

    #[test]
    fn test_into_iter_rev() {
        let mut map = LinkedHashMap::default();

        for i in 1..=4 {
            map.insert_tail(i, format!("value{}", i));
        }

        let items: Vec<_> = map.into_iter().rev().collect();
        assert_eq!(
            items,
            vec![
                (4, "value4".to_string()),
                (3, "value3".to_string()),
                (2, "value2".to_string()),
                (1, "value1".to_string())
            ]
        );
    }

    #[test]
    fn test_iteration_after_modifications() {
        let mut map = LinkedHashMap::default();
        for i in 1..=5 {
            map.insert_tail(i, format!("value{}", i));
        }

        map.remove(&3);

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![1, 2, 4, 5]);

        let ptr2 = map.get_ptr(&2).unwrap();
        map.move_to_tail(ptr2);

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![1, 4, 5, 2]);

        let items: Vec<_> = map.iter().rev().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![2, 5, 4, 1]);
    }

    #[test]
    fn test_empty_iteration() {
        let map: LinkedHashMap<i32, Vec<i32>> = LinkedHashMap::default();

        assert_eq!(map.iter().count(), 0);
        assert_eq!(map.iter().rev().count(), 0);

        let empty_map: LinkedHashMap<i32, Vec<i32>> = LinkedHashMap::default();
        assert_eq!(empty_map.into_iter().count(), 0);

        let empty_map: LinkedHashMap<i32, Vec<i32>> = LinkedHashMap::default();
        assert_eq!(empty_map.into_iter().rev().count(), 0);
    }

    #[test]
    fn test_single_element_iteration() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(42, "answer".to_string());

        let items: Vec<_> = map.iter().collect();
        assert_eq!(items, vec![(&42, &"answer".to_string())]);

        let items: Vec<_> = map.iter().rev().collect();
        assert_eq!(items, vec![(&42, &"answer".to_string())]);
    }

    #[test]
    fn test_entry_api_vacant() {
        let mut map = LinkedHashMap::default();

        match map.entry(1) {
            Entry::Vacant(entry) => {
                assert_eq!(entry.key(), &1);
                let value = entry.insert_tail(vec![1]).1;
                assert_eq!(value, &vec![1]);
            }
            Entry::Occupied(_) => panic!("Expected vacant entry"),
        }

        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&1), Some(&vec![1]));

        match map.entry(2) {
            Entry::Vacant(entry) => {
                let key = entry.into_key();
                assert_eq!(key, 2);
            }
            Entry::Occupied(_) => panic!("Expected vacant entry"),
        }

        assert_eq!(map.len(), 1);
        assert!(!map.contains_key(&2));
    }

    #[test]
    fn test_entry_api_occupied() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(1, vec![1]);
        map.insert_tail(2, vec![2]);

        match map.entry(1) {
            Entry::Occupied(entry) => {
                assert_eq!(entry.key(), &1);
                assert_eq!(entry.get(), &vec![1]);
            }
            Entry::Vacant(_) => panic!("Expected occupied entry"),
        }

        match map.entry(2) {
            Entry::Occupied(mut entry) => {
                let value = entry.get_mut();
                *value = vec![2];
            }
            Entry::Vacant(_) => panic!("Expected occupied entry"),
        }

        assert_eq!(map.get(&2), Some(&vec![2]));

        match map.entry(1) {
            Entry::Occupied(entry) => {
                let old_value = entry.insert_no_move(vec![1]);
                assert_eq!(old_value, vec![1]);
            }
            Entry::Vacant(_) => panic!("Expected occupied entry"),
        }

        assert_eq!(map.get(&1), Some(&vec![1]));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_entry_api_mixed_operations() {
        let mut map = LinkedHashMap::default();

        for i in 1..=3 {
            match map.entry(i) {
                Entry::Vacant(entry) => {
                    entry.insert_tail(format!("value{}", i));
                }
                Entry::Occupied(_) => panic!("Unexpected occupied entry"),
            }
        }

        assert_eq!(map.len(), 3);

        match map.entry(2) {
            Entry::Occupied(entry) => {
                entry.insert_no_move("updated".to_string());
            }
            Entry::Vacant(_) => panic!("Expected occupied entry"),
        }

        let items: Vec<_> = map.iter().collect();
        assert_eq!(
            items,
            vec![
                (&1, &"value1".to_string()),
                (&2, &"updated".to_string()),
                (&3, &"value3".to_string())
            ]
        );
    }

    #[test]
    fn test_cursor_mut_basic_operations() {
        let mut map = LinkedHashMap::default();
        for i in 1..=4 {
            map.insert_tail(i, format!("value{}", i));
        }

        let mut cursor = map.head_cursor_mut();
        assert_eq!(cursor.current(), Some((&1, &"value1".to_string())));

        cursor.move_next();
        assert_eq!(cursor.current(), Some((&2, &"value2".to_string())));

        cursor.move_next();
        assert_eq!(cursor.current(), Some((&3, &"value3".to_string())));

        cursor.move_prev();
        assert_eq!(cursor.current(), Some((&2, &"value2".to_string())));

        let mut cursor = map.tail_cursor_mut();
        assert_eq!(cursor.current(), Some((&4, &"value4".to_string())));

        cursor.move_prev();
        assert_eq!(cursor.current(), Some((&3, &"value3".to_string())));
    }

    #[test]
    fn test_cursor_mut_current_operations() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(1, vec![1]);
        map.insert_tail(2, vec![2]);

        let mut cursor = map.key_cursor_mut(&1);

        if let Some((key, value)) = cursor.current_mut() {
            assert_eq!(key, &1);
            *value = vec![1];
        }

        assert_eq!(map.get(&1), Some(&vec![1]));
    }

    #[test]
    fn test_cursor_mut_next_prev_operations() {
        let mut map = LinkedHashMap::default();
        for i in 1..=3 {
            map.insert_tail(i, format!("value{}", i));
        }

        let mut cursor = map.head_cursor_mut();

        assert_eq!(cursor.next(), Some((&2, &"value2".to_string())));

        if let Some((key, value)) = cursor.next_mut() {
            assert_eq!(key, &2);
            *value = "VALUE2".to_string();
        }

        cursor.move_next();
        cursor.move_next();

        assert_eq!(cursor.prev(), Some((&2, &"VALUE2".to_string())));

        if let Some((key, value)) = cursor.prev_mut() {
            assert_eq!(key, &2);
            *value = "value2_updated".to_string();
        }

        assert_eq!(map.get(&2), Some(&"value2_updated".to_string()));
    }

    #[test]
    fn test_cursor_mut_insert_after_move_to() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(1, vec![1]);
        map.insert_tail(3, vec![3]);

        let mut cursor = map.key_cursor_mut(&1);

        let old_value = cursor.insert_after_move_to(2, vec![2]);
        assert_eq!(old_value, None);

        let items: Vec<_> = cursor.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![2, 3]);

        let old_value = cursor.insert_after_move_to(2, vec![2]);
        assert_eq!(old_value, Some(vec![2]));

        assert_eq!(map.get(&2), Some(&vec![2]));
    }

    #[test]
    fn test_cursor_mut_insert_before_move_to() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(1, vec![1]);
        map.insert_tail(3, vec![3]);

        let mut cursor = map.key_cursor_mut(&3);

        let old_value = cursor.insert_before_move_to(2, vec![2]);
        assert_eq!(old_value, None);

        let items: Vec<_> = cursor.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![2, 3]);

        let old_value = cursor.insert_before_move_to(2, vec![2]);
        assert_eq!(old_value, Some(vec![2]));

        assert_eq!(map.get(&2), Some(&vec![2]));
    }

    #[test]
    fn test_cursor_mut_remove_operations() {
        let mut map = LinkedHashMap::default();
        for i in 1..=5 {
            map.insert_tail(i, format!("value{}", i));
        }

        let mut cursor = map.key_cursor_mut(&3);

        let removed = cursor.remove_next();
        assert_eq!(
            removed,
            Some(RemovedEntry {
                key: 4,
                value: "value4".to_string(),
                prev: cursor.get_ptr(&3),
                next: cursor.get_ptr(&5),
            })
        );

        let removed = cursor.remove_prev();
        assert_eq!(
            removed,
            Some(RemovedEntry {
                key: 2,
                value: "value2".to_string(),
                prev: cursor.get_ptr(&1),
                next: cursor.get_ptr(&3),
            })
        );

        let removed = cursor.remove();
        assert_eq!(
            removed,
            Some(RemovedEntry {
                key: 3,
                value: "value3".to_string(),
                prev: map.get_ptr(&1),
                next: map.get_ptr(&5),
            })
        );
        assert!(!map.contains_key(&3));

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items, vec![1, 5]);
    }

    #[test]
    fn test_cursor_mut_remove_entry() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(1, vec![1]);
        map.insert_tail(2, vec![2]);

        let cursor = map.key_cursor_mut(&1);
        let removed = cursor.remove();
        assert_eq!(
            removed,
            Some(RemovedEntry {
                key: 1,
                value: vec![1],
                prev: map.get_ptr(&2),
                next: map.get_ptr(&2),
            })
        );

        assert_eq!(map.len(), 1);
        assert!(!map.contains_key(&1));
        assert_eq!(map.get(&2), Some(&vec![2]));
    }

    #[test]
    fn test_cursor_mut_empty_map() {
        let mut map: LinkedHashMap<i32, Vec<i32>> = LinkedHashMap::default();

        let mut cursor = map.head_cursor_mut();
        assert_eq!(cursor.current(), None);
        assert_eq!(cursor.next(), None);
        assert_eq!(cursor.prev(), None);
        assert_eq!(cursor.next_ptr(), None);
        assert_eq!(cursor.prev_ptr(), None);

        let old_value = cursor.insert_after_move_to(1, vec![1]);
        assert_eq!(old_value, None);
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&1), Some(&vec![1]));
    }

    #[test]
    fn test_complex_movement_patterns() {
        let mut map = LinkedHashMap::default();
        for i in 1..=10 {
            map.insert_tail(i, i);
        }

        let ptr5 = map.get_ptr(&5).unwrap();
        let ptr2 = map.get_ptr(&2).unwrap();
        let ptr8 = map.get_ptr(&8).unwrap();

        map.move_after(ptr5, ptr8);

        map.move_before(ptr2, ptr5);

        map.move_to_head(ptr8);

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(items[0], 8);
        assert!(items.contains(&2));
        assert!(items.contains(&5));
        assert_eq!(map.len(), 10);

        for i in 1..=10 {
            assert!(map.contains_key(&i));
        }
    }

    #[test]
    fn test_iteration_consistency_after_modifications() {
        let mut map = LinkedHashMap::default();
        for i in 1..=5 {
            map.insert_tail(i, i * 10);
        }

        map.remove(&3);
        if let Some(ptr) = map.get_ptr(&1) {
            map.move_to_tail(ptr);
        }
        map.insert_head(0, 0);

        let forward: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        let backward: Vec<_> = map.iter().rev().map(|(k, _)| *k).collect();
        let mut backward_rev = backward.clone();
        backward_rev.reverse();

        assert_eq!(forward, backward_rev);

        let map_clone = map.clone();
        let consumed: Vec<_> = map_clone.into_iter().map(|(k, _)| k).collect();
        assert_eq!(forward, consumed);

        let consumed_rev: Vec<_> = map.into_iter().rev().map(|(k, _)| k).collect();
        assert_eq!(backward, consumed_rev);
    }

    #[test]
    fn test_shrink_to_fit() {
        let mut map = LinkedHashMap::with_capacity(100);

        for i in 1..=5 {
            map.insert_tail(i, format!("value{}", i));
        }

        map.shrink_to_fit();

        assert_eq!(map.len(), 5);
        for i in 1..=5 {
            assert_eq!(map.get(&i), Some(&format!("value{}", i)));
        }

        map.insert_tail(6, "value6".to_string());
        assert_eq!(map.len(), 6);
    }

    #[test]
    fn test_cursor_mut_with_ptr() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(1, vec![1]);
        map.insert_tail(2, vec![2]);

        let ptr1 = map.get_ptr(&1).unwrap();
        let mut cursor = map.ptr_cursor_mut(ptr1);

        assert_eq!(cursor.ptr(), Some(ptr1));
        assert_eq!(cursor.current(), Some((&1, &vec![1])));

        cursor.move_next();
        assert_eq!(cursor.current(), Some((&2, &vec![2])));
        assert_ne!(cursor.ptr(), Some(ptr1));
    }

    #[test]
    fn test_cursor_mut_nonexistent_key() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(1, vec![1]);

        let mut cursor = map.key_cursor_mut(&999);
        assert_eq!(cursor.current(), None);
        assert_eq!(cursor.ptr(), None);

        let old_value = cursor.insert_after_move_to(999, vec![999]);
        assert_eq!(old_value, None);
        assert_eq!(map.get(&999), Some(&vec![999]));
    }

    #[test]
    fn test_comprehensive_ordering_invariants() {
        let mut map = LinkedHashMap::default();

        for i in 1..=5 {
            map.insert_tail(i, i);
        }

        map.insert_head(0, 0);
        map.remove(&3);
        if let Some(ptr) = map.get_ptr(&4) {
            map.move_to_head(ptr);
        }
        map.insert_tail(6, 6);
        if let Some(ptr) = map.get_ptr(&1) {
            map.move_to_tail(ptr);
        }

        let items: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        let head_key = map.ptr_get_entry(map.head_ptr().unwrap()).map(|(k, _)| *k);
        let tail_key = map.ptr_get_entry(map.tail_ptr().unwrap()).map(|(k, _)| *k);

        assert_eq!(head_key, Some(items[0]));
        assert_eq!(tail_key, Some(items[items.len() - 1]));

        let mut forward_ptrs = Vec::new();
        let mut current_ptr = map.head_ptr().unwrap();
        let mut looped = false;
        while !looped {
            forward_ptrs.push(current_ptr);
            current_ptr = map.next_ptr(current_ptr).unwrap();
            looped = current_ptr == map.head_ptr().unwrap();
        }

        let mut backward_ptrs = Vec::new();
        let mut current_ptr = map.tail_ptr().unwrap();
        let mut looped = false;
        while !looped {
            backward_ptrs.push(current_ptr);
            current_ptr = map.prev_ptr(current_ptr).unwrap();
            looped = current_ptr == map.tail_ptr().unwrap();
        }

        backward_ptrs.reverse();
        assert_eq!(forward_ptrs, backward_ptrs);
        assert_eq!(forward_ptrs.len(), map.len());
    }

    #[test]
    fn test_iter_mut_basic_iteration() {
        let mut map = LinkedHashMap::default();
        for i in 1..=4 {
            map.insert_tail(i, i * 10);
        }

        let mut iter = map.iter_mut();

        let (k1, v1) = iter.next().unwrap();
        assert_eq!(*k1, 1);
        assert_eq!(*v1, 10);
        *v1 = 100;

        let (k2, v2) = iter.next().unwrap();
        assert_eq!(*k2, 2);
        assert_eq!(*v2, 20);
        *v2 = 200;

        let (k3, v3) = iter.next().unwrap();
        assert_eq!(*k3, 3);
        assert_eq!(*v3, 30);

        let (k4, v4) = iter.next().unwrap();
        assert_eq!(*k4, 4);
        assert_eq!(*v4, 40);

        assert!(iter.next().is_none());

        assert_eq!(map.get(&1), Some(&100));
        assert_eq!(map.get(&2), Some(&200));
        assert_eq!(map.get(&3), Some(&30));
        assert_eq!(map.get(&4), Some(&40));
    }

    #[test]
    fn test_iter_mut_backward_iteration() {
        let mut map = LinkedHashMap::default();
        for i in 1..=4 {
            map.insert_tail(i, format!("value{}", i));
        }

        let mut iter = map.iter_mut();

        let (k4, v4) = iter.next_back().unwrap();
        assert_eq!(*k4, 4);
        assert_eq!(*v4, "value4");
        *v4 = "VALUE4".to_string();

        let (k3, v3) = iter.next_back().unwrap();
        assert_eq!(*k3, 3);
        assert_eq!(*v3, "value3");

        let (k2, v2) = iter.next_back().unwrap();
        assert_eq!(*k2, 2);
        assert_eq!(*v2, "value2");
        *v2 = "VALUE2".to_string();

        let (k1, v1) = iter.next_back().unwrap();
        assert_eq!(*k1, 1);
        assert_eq!(*v1, "value1");

        assert!(iter.next_back().is_none());

        assert_eq!(map.get(&1), Some(&"value1".to_string()));
        assert_eq!(map.get(&2), Some(&"VALUE2".to_string()));
        assert_eq!(map.get(&3), Some(&"value3".to_string()));
        assert_eq!(map.get(&4), Some(&"VALUE4".to_string()));
    }

    #[test]
    fn test_iter_mut_bidirectional_iteration() {
        let mut map = LinkedHashMap::default();
        for i in 1..=6 {
            map.insert_tail(i, i * 10);
        }

        let mut iter = map.iter_mut();

        let (k1, v1) = iter.next().unwrap();
        assert_eq!(*k1, 1);
        *v1 = 11;

        let (k6, v6) = iter.next_back().unwrap();
        assert_eq!(*k6, 6);
        *v6 = 66;

        let (k2, v2) = iter.next().unwrap();
        assert_eq!(*k2, 2);
        *v2 = 22;

        let (k5, v5) = iter.next_back().unwrap();
        assert_eq!(*k5, 5);
        *v5 = 55;

        let (k3, v3) = iter.next().unwrap();
        assert_eq!(*k3, 3);
        *v3 = 33;

        let (k4, v4) = iter.next_back().unwrap();
        assert_eq!(*k4, 4);
        *v4 = 44;

        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);

        assert_eq!(map.get(&1), Some(&11));
        assert_eq!(map.get(&2), Some(&22));
        assert_eq!(map.get(&3), Some(&33));
        assert_eq!(map.get(&4), Some(&44));
        assert_eq!(map.get(&5), Some(&55));
        assert_eq!(map.get(&6), Some(&66));
    }

    #[test]
    fn test_iter_mut_empty_map() {
        use alloc::string::String;
        let mut map: LinkedHashMap<i32, String> = LinkedHashMap::default();

        let mut iter = map.iter_mut();
        assert!(iter.next().is_none());
        assert!(iter.next_back().is_none());

        assert!(map.is_empty());
    }

    #[test]
    fn test_iter_mut_single_element() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(42, "answer".to_string());

        let mut iter = map.iter_mut();

        let (key, value) = iter.next().unwrap();
        assert_eq!(*key, 42);
        assert_eq!(*value, "answer");
        *value = "ANSWER".to_string();

        assert!(iter.next().is_none());
        assert!(iter.next_back().is_none());

        assert_eq!(map.get(&42), Some(&"ANSWER".to_string()));
    }

    #[test]
    fn test_iter_mut_single_element_backward() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(42, vec![1, 2, 3]);

        let mut iter = map.iter_mut();

        let (key, value) = iter.next_back().unwrap();
        assert_eq!(*key, 42);
        assert_eq!(*value, vec![1, 2, 3]);
        value.push(4);

        assert!(iter.next().is_none());
        assert!(iter.next_back().is_none());

        assert_eq!(map.get(&42), Some(&vec![1, 2, 3, 4]));
    }

    #[test]
    fn test_iter_mut_modification_patterns() {
        let mut map = LinkedHashMap::default();
        for i in 1..=5 {
            map.insert_tail(i, vec![i]);
        }

        for (key, value) in map.iter_mut() {
            if *key % 2 == 0 {
                value.push(*key * 10);
            } else {
                value.clear();
                value.push(*key * 100);
            }
        }

        assert_eq!(map.get(&1), Some(&vec![100]));
        assert_eq!(map.get(&2), Some(&vec![2, 20]));
        assert_eq!(map.get(&3), Some(&vec![300]));
        assert_eq!(map.get(&4), Some(&vec![4, 40]));
        assert_eq!(map.get(&5), Some(&vec![500]));
    }

    #[test]
    fn test_iter_mut_complex_value_modifications() {
        let mut map = LinkedHashMap::default();
        map.insert_tail("first", vec!["a", "b"]);
        map.insert_tail("second", vec!["c", "d", "e"]);
        map.insert_tail("third", vec!["f"]);

        for (key, value) in map.iter_mut() {
            match *key {
                "first" => {
                    value.reverse();
                    value.push("new");
                }
                "second" => {
                    value.retain(|&s| s != "d");
                }
                "third" => {
                    value.extend_from_slice(&["g", "h"]);
                }
                _ => {}
            }
        }

        assert_eq!(map.get(&"first"), Some(&vec!["b", "a", "new"]));
        assert_eq!(map.get(&"second"), Some(&vec!["c", "e"]));
        assert_eq!(map.get(&"third"), Some(&vec!["f", "g", "h"]));
    }

    #[test]
    fn test_values_mut_iterator() {
        let mut map = LinkedHashMap::default();
        for i in 1..=4 {
            map.insert_tail(i, i * 10);
        }

        let mut values: Vec<_> = map.values_mut().collect();

        for value in values.iter_mut() {
            **value += 5;
        }

        for value in map.values_mut() {
            *value *= 2;
        }

        assert_eq!(map.get(&1), Some(&30));
        assert_eq!(map.get(&2), Some(&50));
        assert_eq!(map.get(&3), Some(&70));
        assert_eq!(map.get(&4), Some(&90));
    }

    #[test]
    fn test_values_mut_backward_iteration() {
        let mut map = LinkedHashMap::default();
        for i in 1..=3 {
            map.insert_tail(i, format!("value{}", i));
        }

        let values: Vec<_> = map.values_mut().rev().collect();

        assert_eq!(values.len(), 3);
        assert_eq!(*values[0], "value3");
        assert_eq!(*values[1], "value2");
        assert_eq!(*values[2], "value1");

        for (i, value) in values.into_iter().enumerate() {
            *value = format!("modified{}", i);
        }

        assert_eq!(map.get(&1), Some(&"modified2".to_string()));
        assert_eq!(map.get(&2), Some(&"modified1".to_string()));
        assert_eq!(map.get(&3), Some(&"modified0".to_string()));
    }

    #[test]
    fn test_iter_mut_with_complex_ordering() {
        let mut map = LinkedHashMap::default();
        for i in 1..=5 {
            map.insert_tail(i, i);
        }

        let ptr3 = map.get_ptr(&3).unwrap();
        map.move_to_head(ptr3);
        map.insert_head(0, 0);
        map.remove(&4);

        let expected_keys = [0, 3, 1, 2, 5];
        let mut iter = map.iter_mut();

        for expected_key in expected_keys.iter() {
            let (key, value) = iter.next().unwrap();
            assert_eq!(*key, *expected_key);
            *value = *expected_key * 100;
        }

        assert!(iter.next().is_none());

        assert_eq!(map.get(&0), Some(&0));
        assert_eq!(map.get(&3), Some(&300));
        assert_eq!(map.get(&1), Some(&100));
        assert_eq!(map.get(&2), Some(&200));
        assert_eq!(map.get(&5), Some(&500));
    }

    #[test]
    fn test_iter_mut_exhausted_iterator_behavior() {
        let mut map = LinkedHashMap::default();
        map.insert_tail(1, "one");
        map.insert_tail(2, "two");

        let mut iter = map.iter_mut();

        assert!(iter.next().is_some());
        assert!(iter.next().is_some());
        assert!(iter.next().is_none());

        assert!(iter.next().is_none());
        assert!(iter.next_back().is_none());
    }

    #[test]
    fn test_clone() {
        let mut map = LinkedHashMap::default();
        map.insert_tail("a", 1);
        map.insert_tail("b", 2);
        map.insert_tail("c", 3);

        let cloned = map.clone();

        assert_eq!(map.len(), cloned.len());
        assert_eq!(
            map.iter().collect::<Vec<_>>(),
            cloned.iter().collect::<Vec<_>>()
        );

        // Verify they are independent
        map.insert_tail("d", 4);
        assert_ne!(map.len(), cloned.len());
    }

    #[test]
    fn test_partial_eq() {
        let mut map1 = LinkedHashMap::default();
        let mut map2 = LinkedHashMap::default();

        // Empty maps are equal
        assert_eq!(map1, map2);

        // Add same elements in same order
        map1.insert_tail("a", 1);
        map1.insert_tail("b", 2);
        map2.insert_tail("a", 1);
        map2.insert_tail("b", 2);
        assert_eq!(map1, map2);

        // Different values
        map2.insert_tail("a", 3);
        assert_ne!(map1, map2);

        // Different lengths
        map1.insert_tail("c", 3);
        assert_ne!(map1, map2);

        // Same content but different order (should be equal since PartialEq doesn't
        // care about order)
        let mut map3 = LinkedHashMap::default();
        map3.insert_tail("b", 2);
        map3.insert_tail("a", 1);
        map3.insert_tail("c", 3);

        let mut map4 = LinkedHashMap::default();
        map4.insert_tail("a", 1);
        map4.insert_tail("b", 2);
        map4.insert_tail("c", 3);

        assert_eq!(map3, map4);
    }

    #[test]
    fn test_from_iterator() {
        let vec = vec![("a", 1), ("b", 2), ("c", 3)];
        let map: LinkedHashMap<&str, i32> = vec.into_iter().collect();

        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&"a"), Some(&1));
        assert_eq!(map.get(&"b"), Some(&2));
        assert_eq!(map.get(&"c"), Some(&3));

        let entries: Vec<_> = map.iter().collect();
        assert_eq!(entries, vec![(&"a", &1), (&"b", &2), (&"c", &3)]);
    }

    #[test]
    fn test_extend_from_iterator() {
        let mut map = LinkedHashMap::default();
        map.insert_tail("existing", 0);

        let vec = vec![("a", 1), ("b", 2), ("c", 3)];
        map.extend(vec);

        assert_eq!(map.len(), 4);
        let entries: Vec<_> = map.iter().collect();
        assert_eq!(
            entries,
            vec![(&"existing", &0), (&"a", &1), (&"b", &2), (&"c", &3)]
        );
    }

    #[test]
    fn test_extend_from_references() {
        let mut map = LinkedHashMap::default();
        map.insert_tail("existing", 0);

        let vec = vec![("a", 1), ("b", 2), ("c", 3)];
        map.extend(vec);

        assert_eq!(map.len(), 4);
        let entries: Vec<_> = map.iter().collect();
        assert_eq!(
            entries,
            vec![(&"existing", &0), (&"a", &1), (&"b", &2), (&"c", &3)]
        );
    }

    #[test]
    fn test_with_capacity_and_hasher() {
        use crate::RandomState;
        let hasher = RandomState::default();
        let mut map: crate::linked_hash_map::LinkedHashMap<&str, i32, _> =
            LinkedHashMap::with_capacity_and_hasher(10, hasher);

        assert_eq!(map.len(), 0);
        assert!(map.is_empty());

        map.insert_tail("key", 42);
        assert_eq!(map.get(&"key"), Some(&42));
    }

    #[test]
    fn test_link_operations() {
        let mut map = LinkedHashMap::default();
        let (ptr1, _) = map.insert_tail_full("first", 1);
        let (ptr2, _) = map.insert_tail_full("second", 2);

        let removed = map.remove_ptr(ptr2).unwrap();
        assert_eq!(removed.key, "second");

        let (ptr3, _) = map.insert_tail_full("third", 3);
        assert!(map.link_after(ptr3, ptr1).is_some());

        let (ptr4, _) = map.insert_tail_full("fourth", 4);
        assert!(map.link_before(ptr4, ptr1).is_some());
    }

    #[test]
    fn test_ptr_operations_comprehensive() {
        let mut map = LinkedHashMap::default();
        let (ptr1, _) = map.insert_tail_full("a", 1);
        let (ptr2, _) = map.insert_tail_full("b", 2);
        let (ptr3, _) = map.insert_tail_full("c", 3);

        assert_eq!(map.next_ptr(ptr1), Some(ptr2));
        assert_eq!(map.next_ptr(ptr2), Some(ptr3));
        assert_eq!(map.next_ptr(ptr3), Some(ptr1));

        assert_eq!(map.prev_ptr(ptr1), Some(ptr3));
        assert_eq!(map.prev_ptr(ptr2), Some(ptr1));
        assert_eq!(map.prev_ptr(ptr3), Some(ptr2));

        assert_eq!(map.ptr_get(ptr1), Some(&1));
        assert_eq!(map.ptr_get_key(ptr2), Some(&"b"));
        assert_eq!(map.ptr_get_entry(ptr3), Some((&"c", &3)));

        *map.ptr_get_mut(ptr1).unwrap() = 10;
        assert_eq!(map.ptr_get(ptr1), Some(&10));

        let (key, value) = map.ptr_get_entry_mut(ptr2).unwrap();
        assert_eq!(key, &"b");
        *value = 20;
        assert_eq!(map.ptr_get(ptr2), Some(&20));
    }

    #[test]
    fn test_cursors() {
        let mut map = LinkedHashMap::default();
        map.insert_tail_full("a", 1);
        let (ptr2, _) = map.insert_tail_full("b", 2);
        let (ptr3, _) = map.insert_tail_full("c", 3);

        let mut cursor = map.ptr_cursor_mut(ptr2);
        if let Some((key, value)) = cursor.current_mut() {
            assert_eq!(key, &"b");
            *value = 20;
        }
        assert_eq!(map.ptr_get(ptr2), Some(&20));

        let mut cursor = map.key_cursor_mut(&"c");
        if let Some((key, value)) = cursor.current_mut() {
            assert_eq!(key, &"c");
            *value = 30;
        }
        assert_eq!(map.ptr_get(ptr3), Some(&30));

        let cursor = map.head_cursor_mut();
        if let Some((key, _)) = cursor.current() {
            assert_eq!(key, &"a");
        }

        let cursor = map.tail_cursor_mut();
        if let Some((key, _)) = cursor.current() {
            assert_eq!(key, &"c");
        }
    }

    #[test]
    fn test_remove_operations_comprehensive() {
        let mut map = LinkedHashMap::default();
        map.insert_tail_full("a", 1);
        let (ptr2, _) = map.insert_tail_full("b", 2);
        map.insert_tail_full("c", 3);

        let removed = map.remove_head().unwrap();
        assert_eq!(removed.key, "a");
        assert_eq!(removed.value, 1);
        assert_eq!(map.len(), 2);

        let removed = map.remove_tail().unwrap();
        assert_eq!(removed.key, "c");
        assert_eq!(removed.value, 3);
        assert_eq!(map.len(), 1);

        let (removed_ptr, removed_entry) = map.remove_with_ptr(&"b").unwrap();
        assert_eq!(removed_ptr, ptr2);
        assert_eq!(removed_entry.key, "b");
        assert_eq!(removed_entry.value, 2);
        assert_eq!(map.len(), 0);

        assert_eq!(map.remove_head(), None);
        assert_eq!(map.remove_tail(), None);
        assert_eq!(map.remove_with_ptr(&"nonexistent"), None);
    }

    #[test]
    fn test_values_and_keys_iterators() {
        let mut map = LinkedHashMap::default();
        map.insert_tail("a", 1);
        map.insert_tail("b", 2);
        map.insert_tail("c", 3);

        let keys: Vec<_> = map.keys().cloned().collect();
        assert_eq!(keys, vec!["a", "b", "c"]);

        let values: Vec<_> = map.values().cloned().collect();
        assert_eq!(values, vec![1, 2, 3]);

        for value in map.values_mut() {
            *value *= 2;
        }

        let values: Vec<_> = map.values().cloned().collect();
        assert_eq!(values, vec![2, 4, 6]);
    }

    #[test]
    fn test_empty_map_edge_cases() {
        let mut map: LinkedHashMap<&str, i32> = LinkedHashMap::default();

        assert_eq!(map.head_ptr(), None);
        assert_eq!(map.tail_ptr(), None);
        assert_eq!(map.remove(&"nonexistent"), None);
        assert_eq!(map.remove_entry(&"nonexistent"), None);
        assert_eq!(map.get_ptr(&"nonexistent"), None);
        assert!(!map.contains_ptr(Ptr::unchecked_from(0)));

        assert_eq!(map.iter().count(), 0);
        assert_eq!(map.iter_mut().count(), 0);
        assert_eq!(map.keys().count(), 0);
        assert_eq!(map.values().count(), 0);
        assert_eq!(map.values_mut().count(), 0);

        map.retain(|_, _| true);
        assert!(map.is_empty());

        map.shrink_to_fit();
    }

    #[test]
    fn test_link_as_head_and_tail_with_unlinked_push() {
        let mut map = LinkedHashMap::default();

        let ptr_x = match map.entry("x") {
            Entry::Vacant(v) => v.push_unlinked(10).0,
            _ => unreachable!(),
        };

        assert_eq!(map.head_ptr(), None);
        assert_eq!(map.tail_ptr(), None);
        assert_eq!(map.iter().count(), 0);

        assert!(map.link_as_head(ptr_x).is_some());
        assert_eq!(map.head_ptr(), Some(ptr_x));
        assert_eq!(map.tail_ptr(), Some(ptr_x));

        let items: Vec<_> = map.iter().collect();
        assert_eq!(items, vec![(&"x", &10)]);

        let ptr_y = match map.entry("y") {
            Entry::Vacant(v) => v.push_unlinked(20).0,
            _ => unreachable!(),
        };
        assert!(map.link_as_tail(ptr_y).is_some());

        let items: Vec<_> = map.iter().collect();
        assert_eq!(items, vec![(&"x", &10), (&"y", &20)]);
        assert_eq!(map.tail_ptr(), Some(ptr_y));
    }

    #[test]
    fn test_link_after_and_before_with_unlinked_nodes() {
        let mut map = LinkedHashMap::default();
        let (ptr_a, _) = map.insert_tail_full("a", 1);
        let (_ptr_c, _) = map.insert_tail_full("c", 3);

        let ptr_b = match map.entry("b") {
            Entry::Vacant(v) => v.push_unlinked(2).0,
            _ => unreachable!(),
        };
        assert!(map.link_after(ptr_b, ptr_a).is_some());
        let keys: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec!["a", "b", "c"]);

        let ptr_0 = match map.entry("0") {
            Entry::Vacant(v) => v.push_unlinked(0).0,
            _ => unreachable!(),
        };
        assert!(map.link_before(ptr_0, map.head_ptr().unwrap()).is_some());
        let keys: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec!["0", "a", "b", "c"]);
    }

    #[test]
    fn test_entry_or_insert_and_and_modify() {
        let mut map = LinkedHashMap::default();

        {
            let v_ref = map.entry("k").or_insert(1);
            assert_eq!(*v_ref, 1);
        }
        assert_eq!(map.get(&"k"), Some(&1));

        {
            let e = map.entry("k").and_modify(|v| *v *= 10);
            let v_ref = e.or_insert(999);
            assert_eq!(*v_ref, 10);
        }
        assert_eq!(map.get(&"k"), Some(&10));
    }

    #[test]
    fn test_retain_filter_and_mutation() {
        let mut map = LinkedHashMap::default();
        for i in 1..=6 {
            map.insert_tail(i, i);
        }

        map.retain(|k, v| {
            *v *= 10;
            k % 2 == 0
        });

        let keys: Vec<_> = map.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec![2, 4, 6]);
        assert_eq!(map.get(&2), Some(&20));
        assert_eq!(map.get(&4), Some(&40));
        assert_eq!(map.get(&6), Some(&60));
        assert_eq!(map.len(), 3);
    }

    #[test]
    #[should_panic(expected = "no entry found for key")]
    fn test_index_key_panic_on_missing() {
        let map: LinkedHashMap<&str, i32> = LinkedHashMap::default();
        let _ = map[&"missing"];
    }

    #[test]
    #[should_panic(expected = "no entry found for key")]
    fn test_index_key_mut_panic_on_missing() {
        let mut map: LinkedHashMap<&str, i32> = LinkedHashMap::default();
        map[&"missing"] = 1;
    }

    #[test]
    #[should_panic(expected = "Invalid Ptr")]
    fn test_index_ptr_panic_after_removal() {
        let mut map = LinkedHashMap::default();
        let (ptr, _) = map.insert_tail_full("a", 1);
        let _ = map.remove_ptr(ptr).unwrap();
        let _ = map[ptr];
    }
}
