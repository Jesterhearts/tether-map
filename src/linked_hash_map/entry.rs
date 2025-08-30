use core::hash::Hash;

use hashbrown::hash_table;

use crate::Ptr;
use crate::arena::ActiveSlotRef;
use crate::arena::Arena;
use crate::arena::FreedSlot;
use crate::linked_hash_map::HeadTail;

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
    #[inline]
    pub fn or_insert(self, default: V) -> &'a mut V {
        match self {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(v) => v.insert_tail(default).1,
        }
    }

    /// If the entry is occupied, applies the provided function to the value in
    /// place. Returns the entry for further chaining.
    #[inline]
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
    pub(crate) entry: hash_table::OccupiedEntry<'a, ActiveSlotRef<K, T>>,
    pub(crate) head_tail: &'a mut Option<HeadTail<K, T>>,
    pub(crate) arena: &'a mut Arena<K, T>,
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
    #[inline]
    pub fn get(&self) -> &T {
        &self.entry.get().data(self.arena).value
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
    #[inline]
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.entry.get_mut().data_mut(self.arena).value
    }

    /// Consumes the occupied entry and returns a mutable reference to the
    /// value.
    ///
    /// The returned reference is tied to the lifetime of the original map
    /// borrow.
    #[inline]
    pub fn into_mut(self) -> &'a mut T {
        let OccupiedEntry { entry, .. } = self;
        &mut entry.into_mut().data_mut(self.arena).value
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
    #[inline]
    pub fn insert_no_move(mut self, value: T) -> T {
        core::mem::replace(&mut self.entry.get_mut().data_mut(self.arena).value, value)
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
    #[inline]
    pub fn ptr(&self) -> Ptr {
        self.entry.get().this(self.arena)
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
    #[inline]
    pub fn key(&self) -> &K {
        &self.entry.get().data(self.arena).key
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
    #[inline]
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
        let entry = self.entry.remove().0;
        // SAFETY: We just removed `entry` from the hash table, and we don't deref it
        // after this.
        let FreedSlot {
            data, prev_next, ..
        } = unsafe { self.arena.free_and_unlink(entry) };

        if let Some((prev, next)) = prev_next {
            if let Some(head_tail) = self.head_tail {
                if head_tail.head == entry {
                    head_tail.head = next;
                }
                if head_tail.tail == entry {
                    head_tail.tail = prev;
                }
            }
        } else if self
            .head_tail
            .as_ref()
            .is_some_and(|ht| ht.head == entry || ht.tail == entry)
        {
            *self.head_tail = None;
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
    #[inline]
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
    pub(crate) key: K,
    pub(crate) entry: hash_table::VacantEntry<'a, ActiveSlotRef<K, T>>,
    pub(crate) nodes: &'a mut Arena<K, T>,
    pub(crate) head_tail: &'a mut Option<HeadTail<K, T>>,
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
    #[inline]
    pub fn insert_tail(self, value: T) -> (Ptr, &'a mut T) {
        let after = self.head_tail.as_ref().map(|ht| ht.tail);
        let (_, ptr, data) = self.insert_after_internal(value, after);
        (ptr, data)
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
    #[inline]
    pub fn insert_unlinked(self, value: T) -> (Ptr, &'a mut T) {
        let mut ptr = self.nodes.alloc_circular(self.key, value);
        self.entry.insert(ptr);
        (ptr.this(self.nodes), &mut ptr.data_mut(self.nodes).value)
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
    #[inline]
    pub fn insert_after(self, value: T, after: Ptr) -> (Ptr, &'a mut T) {
        let after = self
            .nodes
            .map_ptr(after)
            .or(self.head_tail.as_ref().map(|ht| ht.tail));

        let (_, ptr, data) = self.insert_after_internal(value, after);
        (ptr, data)
    }

    pub(crate) fn insert_after_internal(
        self,
        value: T,
        after: Option<ActiveSlotRef<K, T>>,
    ) -> (ActiveSlotRef<K, T>, Ptr, &'a mut T) {
        if let Some(mut after) = after {
            let mut after_next = after.next(self.nodes);
            let mut ptr = self.nodes.alloc(self.key, value, after, after_next);

            *after.next_mut(self.nodes) = ptr;
            *after_next.prev_mut(self.nodes) = ptr;
            self.entry.insert(ptr);

            if let Some(head_tail) = self.head_tail.as_mut()
                && head_tail.tail == after
            {
                head_tail.tail = ptr;
            }

            (
                ptr,
                ptr.this(self.nodes),
                &mut ptr.data_mut(self.nodes).value,
            )
        } else {
            debug_assert_eq!(*self.head_tail, None);
            let mut ptr = self.nodes.alloc_circular(self.key, value);

            *self.head_tail = Some(HeadTail {
                head: ptr,
                tail: ptr,
            });
            self.entry.insert(ptr);
            (
                ptr,
                ptr.this(self.nodes),
                &mut ptr.data_mut(self.nodes).value,
            )
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
    ///         let ptr = entry.insert_head(1).0;
    ///         assert_eq!(map.ptr_get(ptr), Some(&1));
    ///     }
    ///     Entry::Occupied(_) => unreachable!(),
    /// }
    ///
    /// // Order is now: first, second
    /// let entries: Vec<_> = map.iter().collect();
    /// assert_eq!(entries, [(&"first", &1), (&"second", &2)]);
    /// ```
    #[inline]
    pub fn insert_head(self, value: T) -> (Ptr, &'a mut T) {
        let ptr = self.head_tail.as_ref().map(|ht| ht.head);
        let (_, ptr, data) = self.insert_before_internal(value, ptr);
        (ptr, data)
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
    #[inline]
    pub fn insert_before(self, value: T, before: Ptr) -> (Ptr, &'a mut T) {
        let before = self
            .nodes
            .map_ptr(before)
            .or(self.head_tail.as_ref().map(|ht| ht.head));

        let (_, ptr, data) = self.insert_before_internal(value, before);
        (ptr, data)
    }

    /// # Safety
    ///
    /// `before` must be either null or a valid pointer into self.nodes.
    pub(crate) fn insert_before_internal(
        self,
        value: T,
        before: Option<ActiveSlotRef<K, T>>,
    ) -> (ActiveSlotRef<K, T>, Ptr, &'a mut T) {
        if let Some(mut before) = before {
            let mut before_prev = before.prev(self.nodes);
            let mut ptr = self.nodes.alloc(self.key, value, before_prev, before);

            *before.prev_mut(self.nodes) = ptr;
            *before_prev.next_mut(self.nodes) = ptr;
            self.entry.insert(ptr);

            if let Some(head_tail) = self.head_tail.as_mut()
                && head_tail.head == before
            {
                head_tail.head = ptr;
            }

            (
                ptr,
                ptr.this(self.nodes),
                &mut ptr.data_mut(self.nodes).value,
            )
        } else {
            debug_assert_eq!(*self.head_tail, None);
            let mut ptr = self.nodes.alloc_circular(self.key, value);

            *self.head_tail = Some(HeadTail {
                head: ptr,
                tail: ptr,
            });
            self.entry.insert(ptr);
            (
                ptr,
                ptr.this(self.nodes),
                &mut ptr.data_mut(self.nodes).value,
            )
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
    #[inline]
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
    #[inline]
    pub fn key(&self) -> &K {
        &self.key
    }
}
