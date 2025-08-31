use core::hint::unreachable_unchecked;
use core::marker::PhantomData;

use crate::arena::ActiveSlotRef;
use crate::arena::Arena;
use crate::arena::ArenaContainer;
use crate::arena::Links;

#[derive(Debug, Clone, Copy)]
/// An iterator over the entries of a `LinkedHashMap`.
///
/// This struct is created by the [`iter`] method on [`LinkedHashMap`]. See its
/// documentation for more.
///
/// [`iter`]: crate::LinkedHashMap::iter
/// [`LinkedHashMap`]: crate::LinkedHashMap
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
pub struct Iter<'a, K, T> {
    pub(crate) forward_ptr: Option<ActiveSlotRef<K, T>>,
    pub(crate) reverse_ptr: Option<ActiveSlotRef<K, T>>,
    pub(crate) nodes: &'a Arena<K, T>,
}

impl<'a, K, T> Iterator for Iter<'a, K, T> {
    type Item = (&'a K, &'a T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let ptr = self.forward_ptr?;
        if self.forward_ptr == self.reverse_ptr {
            self.forward_ptr = None;
            self.reverse_ptr = None;
        } else {
            self.forward_ptr = Some(ptr.next(self.nodes));
        }

        let data = ptr.data(self.nodes);

        Some((&data.key, &data.value))
    }
}

impl<'a, K, T> DoubleEndedIterator for Iter<'a, K, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let ptr = self.reverse_ptr?;
        if self.reverse_ptr == self.forward_ptr {
            self.reverse_ptr = None;
            self.forward_ptr = None;
        } else {
            self.reverse_ptr = Some(ptr.prev(self.nodes));
        }

        let data = ptr.data(self.nodes);

        Some((&data.key, &data.value))
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
/// [`LinkedHashMap`]: crate::LinkedHashMap
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
    pub(crate) nodes: ArenaContainer<K, T>,
    pub(crate) forward_ptr: Option<ActiveSlotRef<K, T>>,
    pub(crate) reverse_ptr: Option<ActiveSlotRef<K, T>>,
}

impl<K, T> Iterator for IntoIter<K, T> {
    type Item = (K, T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let ptr = self.forward_ptr?;
        if self.forward_ptr == self.reverse_ptr {
            self.forward_ptr = None;
            self.reverse_ptr = None;
        } else {
            self.forward_ptr = Some(ptr.next(&self.nodes));
        }

        // SAFETY: We do not access the pointer after this call.
        let data = unsafe { self.nodes.free_and_unlink(ptr).data };
        Some((data.key, data.value))
    }
}

impl<K, T> DoubleEndedIterator for IntoIter<K, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let ptr = self.reverse_ptr?;
        if self.reverse_ptr == self.forward_ptr {
            self.reverse_ptr = None;
            self.forward_ptr = None;
        } else {
            self.reverse_ptr = Some(ptr.prev(&self.nodes));
        }

        // SAFETY: We do not access the pointer after this call.
        let data = unsafe { self.nodes.free_and_unlink(ptr).data };
        Some((data.key, data.value))
    }
}

#[derive(Debug)]
/// A mutable iterator over the entries of a `LinkedHashMap`.
///
/// This struct is created by the [`iter_mut`] method on [`LinkedHashMap`]. See
/// its documentation for more.
///
/// [`iter_mut`]: crate::LinkedHashMap::iter_mut
/// [`LinkedHashMap`]: crate::LinkedHashMap
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
    pub(crate) forward_ptr: Option<ActiveSlotRef<K, T>>,
    pub(crate) reverse_ptr: Option<ActiveSlotRef<K, T>>,
    pub(crate) _nodes: PhantomData<&'a mut Arena<K, T>>,
}

#[derive(Debug)]
/// A mutable iterator over the values of a `LinkedHashMap`.
///
/// This iterator yields `&mut T` values in the order they were inserted into
/// the map. It is created by the [`values_mut`] method on [`LinkedHashMap`].
///
/// [`values_mut`]: crate::LinkedHashMap::values_mut
/// [`LinkedHashMap`]: crate::LinkedHashMap
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
    pub(crate) iter: IterMut<'a, K, T>,
}

impl<'a, K, T> Iterator for IterMut<'a, K, T> {
    type Item = (&'a K, &'a mut T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // SAFETY: We yield exactly one item per ptr. Our ptrs are unique. We trust the
        // pointers we are iterating over came from our arena. We tie the lifetime to
        // our mutable borrow of the arena.
        let node_mut = unsafe { self.forward_ptr?.as_ptr().as_mut() };
        if self.forward_ptr == self.reverse_ptr {
            self.forward_ptr = None;
            self.reverse_ptr = None;
        } else {
            // SAFETY: We are iterating over our own pointers which we know are valid.
            self.forward_ptr = unsafe {
                Some(match node_mut.links {
                    Links::Occupied { next, .. } => next,
                    Links::Vacant { .. } => unreachable_unchecked(),
                })
            };
        }

        // SAFETY: See above.
        unsafe {
            let data = node_mut.data.assume_init_mut();
            Some((&data.key, &mut data.value))
        }
    }
}

impl<'a, K, T> DoubleEndedIterator for IterMut<'a, K, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        // SAFETY: We yield exactly one item per ptr. Our ptrs are unique. We trust the
        // pointers we are iterating over came from our arena. We tie the lifetime to
        // our mutable borrow of the arena.
        let node_mut = unsafe { self.reverse_ptr?.as_ptr().as_mut() };
        if self.reverse_ptr == self.forward_ptr {
            self.reverse_ptr = None;
            self.forward_ptr = None;
        } else {
            // SAFETY: We are iterating over our own pointers which we know are valid.
            self.reverse_ptr = unsafe {
                Some(match node_mut.links {
                    Links::Occupied { prev, .. } => prev,
                    Links::Vacant { .. } => unreachable_unchecked(),
                })
            };
        }

        // SAFETY: See above.
        unsafe {
            let data = node_mut.data.assume_init_mut();
            Some((&data.key, &mut data.value))
        }
    }
}

impl<'a, K, T> Iterator for ValuesMut<'a, K, T> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(_, v)| v)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, K, T> DoubleEndedIterator for ValuesMut<'a, K, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|(_, v)| v)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use crate::LinkedHashMap;

    #[test]
    fn test_iter_empty() {
        let map: LinkedHashMap<i32, &str> = LinkedHashMap::new();
        let mut iter = map.iter();
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn test_iter_single_element() {
        let mut map = LinkedHashMap::new();
        map.insert(1, "one");

        let mut iter = map.iter();
        assert_eq!(iter.next(), Some((&1, &"one")));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn test_iter_multiple_elements_forward() {
        let mut map = LinkedHashMap::new();
        map.insert(1, "one");
        map.insert(2, "two");
        map.insert(3, "three");

        let mut iter = map.iter();
        assert_eq!(iter.next(), Some((&1, &"one")));
        assert_eq!(iter.next(), Some((&2, &"two")));
        assert_eq!(iter.next(), Some((&3, &"three")));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iter_multiple_elements_backward() {
        let mut map = LinkedHashMap::new();
        map.insert(1, "one");
        map.insert(2, "two");
        map.insert(3, "three");

        let mut iter = map.iter();
        assert_eq!(iter.next_back(), Some((&3, &"three")));
        assert_eq!(iter.next_back(), Some((&2, &"two")));
        assert_eq!(iter.next_back(), Some((&1, &"one")));
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn test_iter_bidirectional() {
        let mut map = LinkedHashMap::new();
        map.insert(1, "one");
        map.insert(2, "two");
        map.insert(3, "three");
        map.insert(4, "four");

        let mut iter = map.iter();
        assert_eq!(iter.next(), Some((&1, &"one")));
        assert_eq!(iter.next_back(), Some((&4, &"four")));
        assert_eq!(iter.next(), Some((&2, &"two")));
        assert_eq!(iter.next_back(), Some((&3, &"three")));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn test_iter_meet_in_middle() {
        let mut map = LinkedHashMap::new();
        map.insert(1, "one");
        map.insert(2, "two");
        map.insert(3, "three");

        let mut iter = map.iter();
        assert_eq!(iter.next(), Some((&1, &"one")));
        assert_eq!(iter.next_back(), Some((&3, &"three")));
        assert_eq!(iter.next(), Some((&2, &"two")));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn test_iter_collect() {
        let mut map = LinkedHashMap::new();
        map.insert("a", 1);
        map.insert("b", 2);
        map.insert("c", 3);

        let collected: Vec<_> = map.iter().collect();
        assert_eq!(collected, vec![(&"a", &1), (&"b", &2), (&"c", &3)]);

        let collected_rev: Vec<_> = map.iter().rev().collect();
        assert_eq!(collected_rev, vec![(&"c", &3), (&"b", &2), (&"a", &1)]);
    }

    #[test]
    fn test_into_iter_empty() {
        let map: LinkedHashMap<i32, &str> = LinkedHashMap::new();
        let mut iter = map.into_iter();
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn test_into_iter_single_element() {
        let mut map = LinkedHashMap::new();
        map.insert(1, "one");

        let mut iter = map.into_iter();
        assert_eq!(iter.next(), Some((1, "one")));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn test_into_iter_multiple_elements_forward() {
        let mut map = LinkedHashMap::new();
        map.insert(1, "one");
        map.insert(2, "two");
        map.insert(3, "three");

        let mut iter = map.into_iter();
        assert_eq!(iter.next(), Some((1, "one")));
        assert_eq!(iter.next(), Some((2, "two")));
        assert_eq!(iter.next(), Some((3, "three")));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_into_iter_multiple_elements_backward() {
        let mut map = LinkedHashMap::new();
        map.insert(1, "one");
        map.insert(2, "two");
        map.insert(3, "three");

        let mut iter = map.into_iter();
        assert_eq!(iter.next_back(), Some((3, "three")));
        assert_eq!(iter.next_back(), Some((2, "two")));
        assert_eq!(iter.next_back(), Some((1, "one")));
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn test_into_iter_bidirectional() {
        let mut map = LinkedHashMap::new();
        map.insert(1, "one");
        map.insert(2, "two");
        map.insert(3, "three");
        map.insert(4, "four");

        let mut iter = map.into_iter();
        assert_eq!(iter.next(), Some((1, "one")));
        assert_eq!(iter.next_back(), Some((4, "four")));
        assert_eq!(iter.next(), Some((2, "two")));
        assert_eq!(iter.next_back(), Some((3, "three")));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn test_into_iter_collect() {
        let mut map = LinkedHashMap::new();
        map.insert("a", 1);
        map.insert("b", 2);
        map.insert("c", 3);

        let collected: Vec<_> = map.into_iter().collect();
        assert_eq!(collected, vec![("a", 1), ("b", 2), ("c", 3)]);
    }

    #[test]
    fn test_iter_mut_empty() {
        let mut map: LinkedHashMap<i32, &str> = LinkedHashMap::new();
        let mut iter = map.iter_mut();
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn test_iter_mut_single_element() {
        let mut map = LinkedHashMap::new();
        map.insert(1, 100);

        {
            let mut iter = map.iter_mut();
            if let Some((key, value)) = iter.next() {
                assert_eq!(key, &1);
                *value *= 2;
            }
            assert_eq!(iter.next(), None);
        }

        assert_eq!(map.get(&1), Some(&200));
    }

    #[test]
    fn test_iter_mut_multiple_elements_forward() {
        let mut map = LinkedHashMap::new();
        map.insert(1, 10);
        map.insert(2, 20);
        map.insert(3, 30);

        {
            let iter = map.iter_mut();
            for (_, value) in iter {
                *value *= 2;
            }
        }

        assert_eq!(map.get(&1), Some(&20));
        assert_eq!(map.get(&2), Some(&40));
        assert_eq!(map.get(&3), Some(&60));
    }

    #[test]
    fn test_iter_mut_multiple_elements_backward() {
        let mut map = LinkedHashMap::new();
        map.insert(1, 10);
        map.insert(2, 20);
        map.insert(3, 30);

        {
            let mut iter = map.iter_mut();
            while let Some((_, value)) = iter.next_back() {
                *value += 1;
            }
        }

        assert_eq!(map.get(&1), Some(&11));
        assert_eq!(map.get(&2), Some(&21));
        assert_eq!(map.get(&3), Some(&31));
    }

    #[test]
    fn test_iter_mut_bidirectional() {
        let mut map = LinkedHashMap::new();
        map.insert(1, 10);
        map.insert(2, 20);
        map.insert(3, 30);
        map.insert(4, 40);

        {
            let mut iter = map.iter_mut();
            if let Some((_, value)) = iter.next() {
                *value = 100;
            }
            if let Some((_, value)) = iter.next_back() {
                *value = 400;
            }
            if let Some((_, value)) = iter.next() {
                *value = 200;
            }
            if let Some((_, value)) = iter.next_back() {
                *value = 300;
            }
        }

        assert_eq!(map.get(&1), Some(&100));
        assert_eq!(map.get(&2), Some(&200));
        assert_eq!(map.get(&3), Some(&300));
        assert_eq!(map.get(&4), Some(&400));
    }

    #[test]
    fn test_values_mut_empty() {
        let mut map: LinkedHashMap<i32, &str> = LinkedHashMap::new();
        let mut iter = map.values_mut();
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn test_values_mut_single_element() {
        let mut map = LinkedHashMap::new();
        map.insert("key", 42);

        {
            let mut iter = map.values_mut();
            if let Some(value) = iter.next() {
                *value = 99;
            }
            assert_eq!(iter.next(), None);
        }

        assert_eq!(map.get(&"key"), Some(&99));
    }

    #[test]
    fn test_values_mut_multiple_elements_forward() {
        let mut map = LinkedHashMap::new();
        map.insert("a", 1);
        map.insert("b", 2);
        map.insert("c", 3);

        {
            for value in map.values_mut() {
                *value *= 10;
            }
        }

        assert_eq!(map.get(&"a"), Some(&10));
        assert_eq!(map.get(&"b"), Some(&20));
        assert_eq!(map.get(&"c"), Some(&30));
    }

    #[test]
    fn test_values_mut_multiple_elements_backward() {
        let mut map = LinkedHashMap::new();
        map.insert("a", 1);
        map.insert("b", 2);
        map.insert("c", 3);

        {
            let mut iter = map.values_mut();
            while let Some(value) = iter.next_back() {
                *value += 100;
            }
        }

        assert_eq!(map.get(&"a"), Some(&101));
        assert_eq!(map.get(&"b"), Some(&102));
        assert_eq!(map.get(&"c"), Some(&103));
    }

    #[test]
    fn test_values_mut_bidirectional() {
        let mut map = LinkedHashMap::new();
        map.insert(1, 10);
        map.insert(2, 20);
        map.insert(3, 30);
        map.insert(4, 40);

        {
            let mut iter = map.values_mut();
            if let Some(value) = iter.next() {
                *value = 1000;
            }
            if let Some(value) = iter.next_back() {
                *value = 4000;
            }
            if let Some(value) = iter.next() {
                *value = 2000;
            }
            if let Some(value) = iter.next_back() {
                *value = 3000;
            }
        }

        assert_eq!(map.get(&1), Some(&1000));
        assert_eq!(map.get(&2), Some(&2000));
        assert_eq!(map.get(&3), Some(&3000));
        assert_eq!(map.get(&4), Some(&4000));
    }

    #[test]
    fn test_values_mut_collect() {
        let mut map = LinkedHashMap::new();
        map.insert("x", 1);
        map.insert("y", 2);
        map.insert("z", 3);

        let original_values: Vec<_> = map.values_mut().map(|v| *v).collect();
        assert_eq!(original_values, vec![1, 2, 3]);

        for value in map.values_mut() {
            *value *= 5;
        }

        let modified_values: Vec<_> = map.values_mut().map(|v| *v).collect();
        assert_eq!(modified_values, vec![5, 10, 15]);
    }

    #[test]
    fn test_iterator_size_hints() {
        let mut map = LinkedHashMap::new();
        map.insert(1, "a");
        map.insert(2, "b");
        map.insert(3, "c");

        let iter = map.iter();
        assert_eq!(iter.size_hint(), (0, None));

        let values_iter = map.values_mut();
        assert_eq!(values_iter.size_hint(), (0, None));
    }

    #[test]
    fn test_iterator_preservation_after_modification() {
        let mut map = LinkedHashMap::new();
        map.insert(1, 10);
        map.insert(2, 20);
        map.insert(3, 30);

        let collected_before: Vec<_> = map.iter().map(|(k, v)| (*k, *v)).collect();

        map.insert(4, 40);

        let collected_after: Vec<_> = map.iter().map(|(k, v)| (*k, *v)).collect();

        assert_eq!(collected_before, vec![(1, 10), (2, 20), (3, 30)]);
        assert_eq!(collected_after, vec![(1, 10), (2, 20), (3, 30), (4, 40)]);
    }
}
