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
pub struct Iter<'a, K, T> {
    pub(crate) forward_ptr: Option<ActiveSlotRef<K, T>>,
    pub(crate) reverse_ptr: Option<ActiveSlotRef<K, T>>,
    pub(crate) nodes: &'a Arena<K, T>,
}

impl<'a, K, T> Iterator for Iter<'a, K, T> {
    type Item = (&'a K, &'a T);

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
    pub(crate) forward_ptr: Option<ActiveSlotRef<K, T>>,
    pub(crate) reverse_ptr: Option<ActiveSlotRef<K, T>>,
    pub(crate) _nodes: PhantomData<&'a mut Arena<K, T>>,
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
    pub(crate) iter: IterMut<'a, K, T>,
}

impl<'a, K, T> Iterator for IterMut<'a, K, T> {
    type Item = (&'a K, &'a mut T);

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
