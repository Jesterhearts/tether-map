use core::hash::BuildHasher;
use core::hash::Hash;

use crate::Ptr;
use crate::arena::ActiveSlotRef;
use crate::linked_hash_map::Entry;
use crate::linked_hash_map::Iter;
use crate::linked_hash_map::LinkedHashMap;
use crate::linked_hash_map::RemovedEntry;

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
    pub(crate) ptr: Option<ActiveSlotRef<K, T>>,
    pub(crate) map: &'m mut LinkedHashMap<K, T, S>,
}

impl<'m, K: Hash + Eq, T, S: BuildHasher> CursorMut<'m, K, T, S> {
    /// Inserts a key-value pair after the cursor's current position and
    /// moves the cursor to the inserted or updated entry.
    #[inline]
    pub fn insert_after_move_to(&mut self, key: K, value: T) -> Option<T> {
        let ptr = if self.ptr.is_none() {
            self.map.head_tail.as_ref().map(|ht| ht.tail)
        } else {
            self.ptr
        };

        match self.map.entry(key) {
            Entry::Occupied(occupied_entry) => {
                let mut map_ptr = *occupied_entry.entry.get();
                if Some(map_ptr) != ptr {
                    self.map.move_after_internal(map_ptr, ptr.unwrap());
                }
                self.ptr = Some(map_ptr);
                Some(core::mem::replace(
                    &mut map_ptr.data_mut(&mut self.map.nodes).value,
                    value,
                ))
            }
            Entry::Vacant(vacant_entry) => {
                self.ptr = Some(vacant_entry.insert_after_internal(value, ptr).0);
                None
            }
        }
    }

    /// Inserts a key-value pair before the cursor's current position and
    /// moves the cursor to the inserted or updated entry.
    #[inline]
    pub fn insert_before_move_to(&mut self, key: K, value: T) -> Option<T> {
        let ptr = if self.ptr.is_none() {
            self.map.head_tail.as_ref().map(|ht| ht.head)
        } else {
            self.ptr
        };

        match self.map.entry(key) {
            Entry::Occupied(occupied_entry) => {
                let mut map_ptr = *occupied_entry.entry.get();
                if Some(map_ptr) != ptr {
                    self.map.move_before_internal(map_ptr, ptr.unwrap());
                }
                self.ptr = Some(map_ptr);
                Some(core::mem::replace(
                    &mut map_ptr.data_mut(&mut self.map.nodes).value,
                    value,
                ))
            }
            Entry::Vacant(vacant_entry) => {
                self.ptr = Some(vacant_entry.insert_before_internal(value, ptr).0);
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
    #[inline]
    pub fn get_ptr(&self, key: &K) -> Option<Ptr> {
        self.map.get_ptr(key)
    }
}

impl<'m, K: Hash + Eq, T, S: BuildHasher> CursorMut<'m, K, T, S> {
    /// Removes the entry before the cursor's current position and returns it.
    #[inline]
    pub fn remove_prev(&mut self) -> Option<(Ptr, RemovedEntry<K, T>)> {
        let prev = self.ptr.map(|slot| slot.prev(&self.map.nodes))?;
        if Some(prev) == self.ptr {
            // Only one element in the list, so removing prev would make the
            // cursor invalid. We choose to keep the cursor at the same position
            // (which will be None after removal).
            self.ptr = None;
        }
        Some(self.map.remove_ptr_internal(prev))
    }

    /// Removes the entry after the cursor's current position and returns it.
    #[inline]
    pub fn remove_next(&mut self) -> Option<(Ptr, RemovedEntry<K, T>)> {
        let next = self.ptr.map(|slot| slot.next(&self.map.nodes))?;
        if Some(next) == self.ptr {
            self.ptr = None;
        }
        Some(self.map.remove_ptr_internal(next))
    }

    /// Removes the entry at the cursor's current position and returns it.
    #[inline]
    pub fn remove(self) -> Option<(Ptr, RemovedEntry<K, T>)> {
        let ptr = self.ptr?;
        Some(self.map.remove_ptr_internal(ptr))
    }
}

impl<'m, K, T, S> CursorMut<'m, K, T, S> {
    /// Returns an iterator starting from the cursor's current position.
    #[inline]
    pub fn iter(&self) -> Iter<'_, K, T> {
        Iter {
            forward_ptr: self.ptr,
            reverse_ptr: self.ptr.map(|slot| slot.prev(&self.map.nodes)),
            nodes: &self.map.nodes,
        }
    }

    /// Moves the cursor to the next entry in the linked list. The internal
    /// linked list is **circular**, so moving next from the tail wraps around
    /// to the head.
    #[inline]
    pub fn move_next(&mut self) {
        self.ptr = self.ptr.map(|slot| slot.next(&self.map.nodes));
    }

    /// Moves the cursor to the previous entry in the linked list. The internal
    /// linked list is **circular**, so moving previous from the head wraps
    /// around to the tail.
    #[inline]
    pub fn move_prev(&mut self) {
        self.ptr = self.ptr.map(|slot| slot.prev(&self.map.nodes));
    }

    /// Gets the current pointer of the cursor.
    #[inline]
    pub fn ptr(&self) -> Option<Ptr> {
        self.ptr.map(|slot| slot.this(&self.map.nodes))
    }

    /// Checks if the cursor is currently at the tail of the linked list.
    #[inline]
    pub fn at_tail(&self) -> bool {
        self.ptr == self.map.head_tail.as_ref().map(|ht| ht.tail)
    }

    /// Checks if the cursor is currently at the head of the linked list.
    #[inline]
    pub fn at_head(&self) -> bool {
        self.ptr == self.map.head_tail.as_ref().map(|ht| ht.head)
    }

    /// Returns the entry at the cursor's current position.
    #[inline]
    pub fn current(&self) -> Option<(&K, &T)> {
        let ptr = self.ptr?;
        let data = ptr.data(&self.map.nodes);
        Some((&data.key, &data.value))
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
    #[inline]
    pub fn current_mut(&mut self) -> Option<(&K, &mut T)> {
        let mut ptr = self.ptr?;
        let data = ptr.data_mut(&mut self.map.nodes);
        Some((&data.key, &mut data.value))
    }

    /// Returns the pointer to the next entry in the linked list from the
    /// cursor's position.
    ///
    /// # Returns
    ///
    /// * `Some(next_ptr)` if there is a next entry
    /// * `None` if the cursor is positioned at a null entry
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
    #[inline]
    pub fn next_ptr(&self) -> Option<Ptr> {
        self.ptr
            .map(|slot| slot.next(&self.map.nodes).this(&self.map.nodes))
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
    /// * `None` if the cursor is positioned at a null entry
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
    #[inline]
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
    /// * `None` if the cursor is positioned at a null entry
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
    #[inline]
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
    /// * `None` if the cursor positioned at a null entry
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
    #[inline]
    pub fn prev_ptr(&self) -> Option<Ptr> {
        self.ptr
            .map(|slot| slot.prev(&self.map.nodes).this(&self.map.nodes))
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
    /// * `None` if the cursor is positioned at a null entry
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
    #[inline]
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
    /// * `None` if the cursor is positioned at a null entry
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
    #[inline]
    pub fn prev_mut(&mut self) -> Option<(&K, &mut T)> {
        let ptr = self.prev_ptr()?;
        self.map.ptr_get_entry_mut(ptr)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use crate::LinkedHashMap;

    #[test]
    fn test_cursor_insert_after_move_to_empty_map() {
        let mut map = LinkedHashMap::new();
        {
            let mut cursor = map.head_cursor_mut();
            assert_eq!(cursor.insert_after_move_to("first", 1), None);
            assert_eq!(cursor.current(), Some((&"first", &1)));
            assert!(cursor.at_head());
            assert!(cursor.at_tail());
        }
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_cursor_insert_after_move_to_existing_key() {
        let mut map = LinkedHashMap::new();
        map.insert("a", 1);
        map.insert("b", 2);

        let mut cursor = map.head_cursor_mut();
        let old_value = cursor.insert_after_move_to("aa", 10);

        assert_eq!(old_value, None);
        assert_eq!(cursor.current(), Some((&"aa", &10)));
        assert_eq!(map.len(), 3);

        let collected: Vec<_> = map.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(collected, vec![("a", 1), ("aa", 10), ("b", 2)]);
    }

    #[test]
    fn test_cursor_insert_after_move_to_new_key() {
        let mut map = LinkedHashMap::new();
        map.insert("a", 1);
        map.insert("b", 2);

        let mut cursor = map.head_cursor_mut();
        assert_eq!(cursor.insert_after_move_to("c", 3), None);

        assert_eq!(cursor.current(), Some((&"c", &3)));
        assert_eq!(map.len(), 3);

        let collected: Vec<_> = map.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(collected, vec![("a", 1), ("c", 3), ("b", 2)]);
    }

    #[test]
    fn test_cursor_insert_before_move_to_empty_map() {
        let mut map = LinkedHashMap::new();
        {
            let mut cursor = map.tail_cursor_mut();
            assert_eq!(cursor.insert_before_move_to("first", 1), None);
            assert_eq!(cursor.current(), Some((&"first", &1)));
            assert!(cursor.at_head());
            assert!(cursor.at_tail());
        }
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_cursor_insert_before_move_to_existing_key() {
        let mut map = LinkedHashMap::new();
        map.insert("a", 1);
        map.insert("b", 2);

        let mut cursor = map.tail_cursor_mut();
        let old_value = cursor.insert_before_move_to("a", 10);

        assert_eq!(old_value, Some(1));
        assert_eq!(cursor.current(), Some((&"a", &10)));
        assert_eq!(map.len(), 2);

        let collected: Vec<_> = map.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(collected, vec![("a", 10), ("b", 2)]);
    }

    #[test]
    fn test_cursor_insert_before_move_to_new_key() {
        let mut map = LinkedHashMap::new();
        map.insert("a", 1);
        map.insert("b", 2);

        let mut cursor = map.tail_cursor_mut();
        assert_eq!(cursor.insert_before_move_to("c", 3), None);

        assert_eq!(cursor.current(), Some((&"c", &3)));
        assert_eq!(map.len(), 3);

        let collected: Vec<_> = map.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(collected, vec![("a", 1), ("c", 3), ("b", 2)]);
    }

    #[test]
    fn test_cursor_remove_prev() {
        let mut map = LinkedHashMap::new();
        map.insert("a", 1);
        map.insert("b", 2);
        map.insert("c", 3);

        {
            let mut cursor = map.tail_cursor_mut();
            let (_ptr, removed) = cursor.remove_prev().unwrap();

            assert_eq!(removed.key, "b");
            assert_eq!(removed.value, 2);
            assert_eq!(cursor.current(), Some((&"c", &3)));
        }
        assert_eq!(map.len(), 2);
        let collected: Vec<_> = map.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(collected, vec![("a", 1), ("c", 3)]);
    }

    #[test]
    fn test_cursor_remove_next() {
        let mut map = LinkedHashMap::new();
        map.insert("a", 1);
        map.insert("b", 2);
        map.insert("c", 3);

        {
            let mut cursor = map.head_cursor_mut();
            let (_ptr, removed) = cursor.remove_next().unwrap();

            assert_eq!(removed.key, "b");
            assert_eq!(removed.value, 2);
            assert_eq!(cursor.current(), Some((&"a", &1)));
        }
        assert_eq!(map.len(), 2);
        let collected: Vec<_> = map.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(collected, vec![("a", 1), ("c", 3)]);
    }

    #[test]
    fn test_cursor_remove_current() {
        let mut map = LinkedHashMap::new();
        map.insert("a", 1);
        map.insert("b", 2);
        map.insert("c", 3);

        {
            let cursor = map.key_cursor_mut(&"b");
            let (_ptr, removed) = cursor.remove().unwrap();

            assert_eq!(removed.key, "b");
            assert_eq!(removed.value, 2);
        }
        assert_eq!(map.len(), 2);
        let collected: Vec<_> = map.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(collected, vec![("a", 1), ("c", 3)]);
    }

    #[test]
    fn test_cursor_remove_operations_empty_cases() {
        let mut map = LinkedHashMap::new();
        map.insert("only", 1);

        let mut cursor = map.head_cursor_mut();
        assert_ne!(cursor.remove_prev(), None);
        assert!(cursor.remove_next().is_none());

        let cursor = cursor.remove();
        assert!(cursor.is_none());
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn test_cursor_navigation_move_next_prev() {
        let mut map = LinkedHashMap::new();
        map.insert("a", 1);
        map.insert("b", 2);
        map.insert("c", 3);

        let mut cursor = map.head_cursor_mut();
        assert_eq!(cursor.current(), Some((&"a", &1)));
        assert!(cursor.at_head());

        cursor.move_next();
        assert_eq!(cursor.current(), Some((&"b", &2)));
        assert!(!cursor.at_head());
        assert!(!cursor.at_tail());

        cursor.move_next();
        assert_eq!(cursor.current(), Some((&"c", &3)));
        assert!(cursor.at_tail());

        cursor.move_next();
        assert_eq!(cursor.current(), Some((&"a", &1)));
        assert!(cursor.at_head());

        cursor.move_prev();
        assert_eq!(cursor.current(), Some((&"c", &3)));
        assert!(cursor.at_tail());

        cursor.move_prev();
        assert_eq!(cursor.current(), Some((&"b", &2)));
        assert!(!cursor.at_head());
        assert!(!cursor.at_tail());
    }

    #[test]
    fn test_cursor_current_and_current_mut() {
        let mut map = LinkedHashMap::new();
        map.insert("test", 42);

        let mut cursor = map.head_cursor_mut();
        assert_eq!(cursor.current(), Some((&"test", &42)));

        if let Some((key, value)) = cursor.current_mut() {
            assert_eq!(key, &"test");
            *value = 100;
        }

        assert_eq!(cursor.current(), Some((&"test", &100)));
        assert_eq!(map.get(&"test"), Some(&100));
    }

    #[test]
    fn test_cursor_next_and_next_mut() {
        let mut map = LinkedHashMap::new();
        map.insert("a", 1);
        map.insert("b", 2);

        let mut cursor = map.head_cursor_mut();
        assert_eq!(cursor.next(), Some((&"b", &2)));

        if let Some((key, value)) = cursor.next_mut() {
            assert_eq!(key, &"b");
            *value = 20;
        }

        assert_eq!(cursor.next(), Some((&"b", &20)));
        assert_eq!(map.get(&"b"), Some(&20));
    }

    #[test]
    fn test_cursor_prev_and_prev_mut() {
        let mut map = LinkedHashMap::new();
        map.insert("a", 1);
        map.insert("b", 2);

        let mut cursor = map.tail_cursor_mut();
        assert_eq!(cursor.prev(), Some((&"a", &1)));

        if let Some((key, value)) = cursor.prev_mut() {
            assert_eq!(key, &"a");
            *value = 10;
        }

        assert_eq!(cursor.prev(), Some((&"a", &10)));
        assert_eq!(map.get(&"a"), Some(&10));
    }

    #[test]
    fn test_cursor_ptr_operations() {
        let mut map = LinkedHashMap::new();
        map.insert("a", 1);
        map.insert("b", 2);
        map.insert("c", 3);

        let current_ptr;
        let next_ptr;
        let prev_ptr;
        {
            let cursor = map.head_cursor_mut();
            current_ptr = cursor.ptr().unwrap();
            next_ptr = cursor.next_ptr().unwrap();
            prev_ptr = cursor.prev_ptr().unwrap();

            assert_eq!(cursor.get_ptr(&"b"), Some(next_ptr));
            assert_eq!(cursor.get_ptr(&"nonexistent"), None);
        }
        assert_eq!(map.ptr_get(current_ptr), Some(&1));
        assert_eq!(map.ptr_get(next_ptr), Some(&2));
        assert_eq!(map.ptr_get(prev_ptr), Some(&3));
    }

    #[test]
    fn test_cursor_at_head_tail_single_element() {
        let mut map = LinkedHashMap::new();
        map.insert("only", 1);

        let cursor = map.head_cursor_mut();
        assert!(cursor.at_head());
        assert!(cursor.at_tail());
        assert_eq!(cursor.current(), Some((&"only", &1)));
    }

    #[test]
    fn test_cursor_iter_from_position() {
        let mut map = LinkedHashMap::new();
        map.insert("a", 1);
        map.insert("b", 2);
        map.insert("c", 3);
        map.insert("d", 4);

        let cursor = map.key_cursor_mut(&"b");
        let collected: Vec<_> = cursor.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(collected, vec![("b", 2), ("c", 3), ("d", 4), ("a", 1)]);
    }

    #[test]
    fn test_cursor_complex_insertion_sequence() {
        let mut map = LinkedHashMap::new();

        let mut cursor = map.head_cursor_mut();
        cursor.insert_after_move_to("b", 2);
        cursor.insert_before_move_to("a", 1);
        cursor.insert_after_move_to("c", 3);

        let collected: Vec<_> = map.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(collected, vec![("a", 1), ("c", 3), ("b", 2)]);
    }

    #[test]
    fn test_cursor_mixed_operations() {
        let mut map = LinkedHashMap::new();
        map.insert("a", 1);
        map.insert("b", 2);
        map.insert("c", 3);
        map.insert("d", 4);

        {
            let mut cursor = map.key_cursor_mut(&"b");

            cursor.insert_after_move_to("e", 5);
            assert_eq!(cursor.current(), Some((&"e", &5)));

            cursor.remove_prev();

            cursor.move_next();
            let (_ptr, removed) = cursor.remove_next().unwrap();
            assert_eq!(removed.key, "d");
        }
        assert_eq!(map.len(), 3);
        let final_order: Vec<_> = map.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(final_order, vec![("a", 1), ("e", 5), ("c", 3)]);
    }

    #[test]
    fn test_cursor_boundary_conditions() {
        let mut map = LinkedHashMap::new();
        map.insert("a", 1);
        map.insert("b", 2);

        let mut cursor = map.head_cursor_mut();

        cursor.move_prev();
        assert!(cursor.at_tail());
        assert_eq!(cursor.current(), Some((&"b", &2)));

        cursor.move_next();
        assert!(cursor.at_head());
        assert_eq!(cursor.current(), Some((&"a", &1)));

        let tail_cursor = map.tail_cursor_mut();
        assert!(tail_cursor.at_tail());
        assert_eq!(tail_cursor.current(), Some((&"b", &2)));
    }

    #[test]
    fn test_cursor_empty_map_operations() {
        let mut map: LinkedHashMap<&str, i32> = LinkedHashMap::new();
        let cursor = map.head_cursor_mut();

        assert!(cursor.current().is_none());
        assert!(cursor.next().is_none());
        assert!(cursor.prev().is_none());
        assert!(cursor.ptr().is_none());
        assert!(cursor.next_ptr().is_none());
        assert!(cursor.prev_ptr().is_none());
        assert!(cursor.at_head());
        assert!(cursor.at_tail());
    }

    #[test]
    fn test_cursor_modification_during_iteration() {
        let mut map = LinkedHashMap::new();
        map.insert("a", 1);
        map.insert("b", 2);
        map.insert("c", 3);

        let mut cursor = map.head_cursor_mut();

        while let Some((key, value)) = cursor.current_mut() {
            *value *= 10;
            if *key == "b" {
                cursor.insert_after_move_to("x", 99);
                break;
            }
            cursor.move_next();
        }

        let final_state: Vec<_> = map.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(final_state, vec![("a", 10), ("b", 20), ("x", 99), ("c", 3)]);
    }

    #[test]
    fn test_cursor_position_consistency_after_operations() {
        let mut map = LinkedHashMap::new();
        map.insert("a", 1);
        map.insert("b", 2);
        map.insert("c", 3);

        {
            let mut cursor = map.key_cursor_mut(&"b");
            let original_ptr = cursor.ptr().unwrap();

            cursor.insert_after_move_to("d", 4);
            let new_ptr = cursor.ptr().unwrap();
            assert_ne!(original_ptr, new_ptr);
            assert_eq!(cursor.current(), Some((&"d", &4)));

            cursor.insert_before_move_to("e", 5);
            assert_eq!(cursor.current(), Some((&"e", &5)));
        }
        let final_order: Vec<_> = map.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(
            final_order,
            vec![("a", 1), ("b", 2), ("e", 5), ("d", 4), ("c", 3)]
        );
    }
}
