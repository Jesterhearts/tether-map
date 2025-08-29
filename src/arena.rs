use alloc::vec::Vec;
use core::clone::Clone;
use core::fmt::Debug;
use core::hint::unreachable_unchecked;
use core::mem::MaybeUninit;
use core::ops::Index;
use core::ops::IndexMut;
use core::ptr::NonNull;

use bumpalo::Bump;

use crate::Ptr;

#[derive(Debug, Clone, Copy)]
pub(crate) struct LLData<K, T> {
    pub(crate) key: K,
    pub(crate) value: T,
}

impl<K, T> LLData<K, T> {
    pub(crate) fn key_value_mut(&mut self) -> (&K, &mut T) {
        (&self.key, &mut self.value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Links<K, T> {
    Active {
        prev: NonNull<LLSlot<K, T>>,
        next: NonNull<LLSlot<K, T>>,
    },
    Free(Option<NonNull<LLSlot<K, T>>>),
}

impl<K, T> Links<K, T> {
    /// # Safety
    ///
    /// The caller must ensure that the Links is in the Active state.
    pub(crate) unsafe fn next(&self) -> NonNull<LLSlot<K, T>> {
        match self {
            Links::Active { next, .. } => *next,
            Links::Free(_) => unsafe { unreachable_unchecked() },
        }
    }

    /// # Safety
    ///
    /// The caller must ensure that the Links is in the Active state.
    pub(crate) unsafe fn next_mut(&mut self) -> &mut NonNull<LLSlot<K, T>> {
        match self {
            Links::Active { next, .. } => next,
            Links::Free(_) => unsafe { unreachable_unchecked() },
        }
    }

    /// # Safety
    ///
    /// The caller must ensure that the Links is in the Active state.
    pub(crate) unsafe fn prev(&self) -> NonNull<LLSlot<K, T>> {
        match self {
            Links::Active { prev, .. } => *prev,
            Links::Free(_) => unsafe { unreachable_unchecked() },
        }
    }

    /// # Safety
    ///
    /// The caller must ensure that the Links is in the Active state.
    pub(crate) unsafe fn prev_mut(&mut self) -> &mut NonNull<LLSlot<K, T>> {
        match self {
            Links::Active { prev, .. } => prev,
            Links::Free(_) => unsafe { unreachable_unchecked() },
        }
    }

    /// # Safety
    ///
    /// The caller must ensure that the Links is in the Free state.
    pub(crate) unsafe fn free(&self) -> Option<NonNull<LLSlot<K, T>>> {
        match self {
            Links::Free(next) => *next,
            Links::Active { .. } => unsafe { unreachable_unchecked() },
        }
    }
}

pub(crate) struct LLSlot<K, T> {
    pub(crate) this: Ptr,
    pub(crate) data: MaybeUninit<LLData<K, T>>,
    pub(crate) links: Links<K, T>,
}

impl<K, T> Debug for LLSlot<K, T>
where
    K: Debug,
    T: Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LLSlot")
            .field("this", &self.this)
            .field("links", &self.links)
            .finish()
    }
}

pub(crate) struct FreedSlot<K, T> {
    pub(crate) data: LLData<K, T>,
    pub(crate) prev: Option<Ptr>,
    pub(crate) next: Option<Ptr>,
    pub(crate) this: Ptr,
    pub(crate) prev_raw: *mut LLSlot<K, T>,
    pub(crate) next_raw: *mut LLSlot<K, T>,
}

#[derive(Debug)]
pub(crate) struct Arena<K, T> {
    bump: Bump,
    slots: Vec<NonNull<LLSlot<K, T>>>,
    free_head: Option<NonNull<LLSlot<K, T>>>,
}

impl<K, T> Drop for Arena<K, T> {
    fn drop(&mut self) {
        for mut slot in self.slots.drain(..) {
            // SAFETY: We own all slots, so they are valid. We check for free before
            // dropping data.
            unsafe {
                match slot.as_ref().links {
                    Links::Free(_) => continue,
                    Links::Active { .. } => {
                        slot.as_mut().data.assume_init_drop();
                    }
                }
            }
        }
    }
}

impl<K, T> Arena<K, T> {
    pub(crate) fn new() -> Self {
        Arena {
            bump: Bump::new(),
            slots: Vec::new(),
            free_head: None,
        }
    }

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Arena {
            bump: Bump::with_capacity(capacity * size_of::<LLSlot<K, T>>()),
            slots: Vec::with_capacity(capacity),
            free_head: None,
        }
    }

    pub(crate) fn alloc_circular(&mut self, key: K, value: T) -> NonNull<LLSlot<K, T>> {
        if let Some(mut free_head) = self.free_head {
            // SAFETY: We have nodes in our free list, so free_head is valid.
            let ptr = unsafe { free_head.as_mut() };
            let new_node = free_head;
            // SAFETY: Node is free, so links.free is valid
            self.free_head = unsafe { ptr.links.free() };

            ptr.data.write(LLData { key, value });
            ptr.links = Links::Active {
                prev: new_node,
                next: new_node,
            };

            new_node
        } else {
            let this = Ptr::unchecked_from(self.slots.len());
            let slot = self
                .bump
                .alloc_layout(core::alloc::Layout::new::<LLSlot<K, T>>())
                .cast();

            // SAFETY: We just allocated this slot, so it is valid to write to.
            unsafe {
                slot.write(LLSlot {
                    links: Links::Active {
                        prev: slot,
                        next: slot,
                    },
                    this,
                    data: MaybeUninit::new(LLData { key, value }),
                });
            }

            self.slots.push(slot);
            slot
        }
    }

    pub(crate) fn alloc(
        &mut self,
        key: K,
        value: T,
        prev: NonNull<LLSlot<K, T>>,
        next: NonNull<LLSlot<K, T>>,
    ) -> NonNull<LLSlot<K, T>> {
        if let Some(mut free_head) = self.free_head {
            // SAFETY: We have nodes in our free list, so free_head is valid.
            let ptr = unsafe { free_head.as_mut() };
            let new_node = free_head;
            // SAFETY: Node is free, so links.free is valid
            self.free_head = unsafe { ptr.links.free() };
            ptr.data.write(LLData { key, value });

            ptr.links = Links::Active { prev, next };

            new_node
        } else {
            let this = Ptr::unchecked_from(self.slots.len());
            let new_node = self.bump.alloc(LLSlot {
                links: Links::Active { prev, next },
                this,
                data: MaybeUninit::new(LLData { key, value }),
            });

            let new_node = NonNull::from_mut(new_node);
            self.slots.push(new_node);
            new_node
        }
    }

    pub(crate) fn is_occupied(&self, ptr: Ptr) -> bool {
        // SAFETY: We check that only store valid Ptrs in slots, so if the index is
        // out of bounds, we return false. If it is in bounds, we check if it is free.
        unsafe {
            self.slots
                .get(ptr.unchecked_get())
                .is_some_and(|ptr| matches!(ptr.as_ref().links, Links::Active { .. }))
        }
    }

    pub(crate) fn map_ptr(&self, ptr: Ptr) -> Option<NonNull<LLSlot<K, T>>> {
        // SAFETY: We check that only store valid Ptrs in slots, so if the index is
        // out of bounds, we return None. If it is in bounds, we return the pointer if
        // it is occupied.
        unsafe {
            self.slots
                .get(ptr.unchecked_get())
                .copied()
                .filter(|ptr| matches!(ptr.as_ref().links, Links::Active { .. }))
        }
    }

    /// # Safety
    ///
    /// The provided pointer must be valid and currently occupied.
    #[inline]
    pub(crate) unsafe fn free_and_unlink(
        &mut self,
        mut ptr: NonNull<LLSlot<K, T>>,
    ) -> FreedSlot<K, T> {
        // SAFETY: We know this pointer is valid and occupied per contract. We mark the
        // slot as free and return the data. We never read from data again after
        // this until after it is overwritten in alloc or alloc_circular.
        unsafe {
            let (data, this, mut prev, mut next) = {
                let links = Links::Free(self.free_head);
                self.free_head = Some(ptr);

                let ref_mut = ptr.as_mut();

                let prev = ref_mut.links.prev();
                let next = ref_mut.links.next();
                ref_mut.links = links;

                (
                    MaybeUninit::assume_init_read(&ref_mut.data),
                    ref_mut.this,
                    prev,
                    next,
                )
            };

            let (prev_external, prev_raw) = if prev != ptr {
                *prev.as_mut().links.next_mut() = next;
                (Some(prev.as_ref().this), prev.as_ptr())
            } else {
                (None, core::ptr::null_mut())
            };

            let (next_external, next_raw) = if next != ptr {
                *next.as_mut().links.prev_mut() = prev;
                (Some(next.as_ref().this), next.as_ptr())
            } else {
                (None, core::ptr::null_mut())
            };

            FreedSlot {
                data,
                prev: prev_external,
                next: next_external,
                prev_raw,
                next_raw,
                this,
            }
        }
    }
}

impl<K, T> Index<Ptr> for Arena<K, T> {
    type Output = LLData<K, T>;

    fn index(&self, index: Ptr) -> &Self::Output {
        // SAFETY: We know this pointer comes from this arena, and we check that it is
        // occupied before returning a reference to the data.
        unsafe {
            let ptr = self.slots[index.unchecked_get()].as_ref();
            match ptr.links {
                Links::Free(_) => {
                    #[cold]
                    #[inline(never)]
                    fn die() -> ! {
                        panic!("Invalid Ptr");
                    }
                    die();
                }
                Links::Active { .. } => ptr.data.assume_init_ref(),
            }
        }
    }
}

impl<K, T> IndexMut<Ptr> for Arena<K, T> {
    fn index_mut(&mut self, index: Ptr) -> &mut Self::Output {
        // SAFETY: We know this pointer comes from this arena, and we check that it is
        // occupied before returning a mutable reference to the data. The returned
        // lifetime is tied to our lifetime, so we won't have aliasing issues.
        unsafe {
            let ptr = self.slots[index.unchecked_get()].as_mut();
            match ptr.links {
                Links::Free(_) => {
                    #[cold]
                    #[inline(never)]
                    fn die() -> ! {
                        panic!("Invalid Ptr");
                    }
                    die();
                }
                Links::Active { .. } => ptr.data.assume_init_mut(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::format;
    use alloc::string::String;
    use alloc::string::ToString;
    use alloc::vec::Vec;
    use core::assert_eq;

    use super::*;

    #[test]
    fn test_arena_alloc_circular_basic() {
        let mut arena = Arena::new();
        let ptr = arena.alloc_circular(42, "hello".to_string());

        unsafe {
            let slot = ptr.as_ref();
            assert_eq!(slot.data.assume_init_ref().key, 42);
            assert_eq!(slot.data.assume_init_ref().value, "hello");
            assert_eq!(
                slot.links,
                Links::Active {
                    prev: ptr,
                    next: ptr
                }
            );
        }
    }

    #[test]
    fn test_arena_alloc_circular_multiple() {
        let mut arena = Arena::new();
        let ptr1 = arena.alloc_circular(1, "first".to_string());
        let ptr2 = arena.alloc_circular(2, "second".to_string());

        assert_ne!(ptr1.as_ptr(), ptr2.as_ptr());

        unsafe {
            let slot1 = ptr1.as_ref();
            let slot2 = ptr2.as_ref();

            assert_eq!(slot1.data.assume_init_ref().key, 1);
            assert_eq!(slot2.data.assume_init_ref().key, 2);

            assert_eq!(
                slot1.links,
                Links::Active {
                    prev: ptr1,
                    next: ptr1
                }
            );
            assert_eq!(
                slot2.links,
                Links::Active {
                    prev: ptr2,
                    next: ptr2
                }
            );
        }
    }

    #[test]
    fn test_arena_alloc_basic() {
        let mut arena = Arena::new();
        let ptr1 = arena.alloc_circular(1, "first".to_string());
        let ptr2 = arena.alloc(2, "second".to_string(), ptr1, ptr1);

        unsafe {
            let slot2 = ptr2.as_ref();

            assert_eq!(
                slot2.links,
                Links::Active {
                    prev: ptr1,
                    next: ptr1
                }
            );
            assert_eq!(slot2.data.assume_init_ref().key, 2);
            assert_eq!(slot2.data.assume_init_ref().value, "second");
        }
    }

    #[test]
    fn test_arena_is_occupied_after_alloc() {
        let mut arena = Arena::new();
        let ptr = arena.alloc_circular(42, "test".to_string());

        unsafe {
            let slot = ptr.as_ref();
            assert!(arena.is_occupied(slot.this));
        }
    }

    #[test]
    fn test_arena_is_occupied_out_of_bounds() {
        let arena: Arena<i32, String> = Arena::new();
        let invalid_ptr = Ptr::unchecked_from(100);
        assert!(!arena.is_occupied(invalid_ptr));
    }

    #[test]
    fn test_arena_map_ptr_valid() {
        let mut arena = Arena::new();
        let ptr = arena.alloc_circular(42, "test".to_string());

        unsafe {
            let slot = ptr.as_ref();
            let mapped = arena.map_ptr(slot.this);
            assert!(mapped.is_some());
            assert_eq!(mapped.unwrap().as_ptr(), ptr.as_ptr());
        }
    }

    #[test]
    fn test_arena_map_ptr_invalid() {
        let arena: Arena<i32, String> = Arena::new();
        let invalid_ptr = Ptr::unchecked_from(100);
        assert_eq!(arena.map_ptr(invalid_ptr), None);
    }

    #[test]
    fn test_arena_indexing() {
        let mut arena = Arena::new();
        let ptr = arena.alloc_circular(42, "hello".to_string());

        unsafe {
            let slot = ptr.as_ref();
            let data = &arena[slot.this];
            assert_eq!(data.key, 42);
            assert_eq!(data.value, "hello");
        }
    }

    #[test]
    fn test_arena_indexing_mut() {
        let mut arena = Arena::new();
        let ptr = arena.alloc_circular(42, "hello".to_string());

        unsafe {
            let slot = ptr.as_ref();
            let this_ptr = slot.this;

            arena[this_ptr].value = "modified".to_string();
            assert_eq!(arena[this_ptr].value, "modified");

            let (key, value) = arena[this_ptr].key_value_mut();
            assert_eq!(*key, 42);
            *value = "mutated".to_string();
            assert_eq!(arena[this_ptr].value, "mutated");
        }
    }

    #[test]
    fn test_arena_free_basic() {
        let mut arena = Arena::new();
        let ptr = arena.alloc_circular(42, "hello".to_string());

        unsafe {
            let slot = ptr.as_ref();
            let this_ptr = slot.this;

            assert!(arena.is_occupied(this_ptr));

            let freed = arena.free_and_unlink(ptr);

            assert!(!arena.is_occupied(this_ptr));
            assert_eq!(arena.map_ptr(this_ptr), None);
            assert_eq!(freed.data.key, 42);
            assert_eq!(freed.data.value, "hello");
        }
    }

    #[test]
    fn test_arena_free_and_reuse() {
        let mut arena = Arena::new();
        let ptr1 = arena.alloc_circular(1, "first".to_string());

        unsafe {
            let slot1 = ptr1.as_ref();
            let first_this = slot1.this;

            arena.free_and_unlink(ptr1);
            assert!(!arena.is_occupied(first_this));

            let ptr2 = arena.alloc_circular(2, "second".to_string());
            let slot2 = ptr2.as_ref();

            assert_eq!(slot2.this, first_this);
            assert!(arena.is_occupied(first_this));
            assert_eq!(arena[first_this].key, 2);
            assert_eq!(arena[first_this].value, "second");
        }
    }

    #[test]
    fn test_arena_multiple_free_and_reuse() {
        let mut arena = Arena::new();
        let ptr1 = arena.alloc_circular(1, "first".to_string());
        let ptr2 = arena.alloc_circular(2, "second".to_string());
        let ptr3 = arena.alloc_circular(3, "third".to_string());

        unsafe {
            let this1 = ptr1.as_ref().this;
            let this2 = ptr2.as_ref().this;
            let this3 = ptr3.as_ref().this;

            arena.free_and_unlink(ptr2);
            arena.free_and_unlink(ptr1);

            assert!(!arena.is_occupied(this1));
            assert!(!arena.is_occupied(this2));
            assert!(arena.is_occupied(this3));

            let ptr4 = arena.alloc_circular(4, "fourth".to_string());
            let ptr5 = arena.alloc_circular(5, "fifth".to_string());

            let this4 = ptr4.as_ref().this;
            let this5 = ptr5.as_ref().this;

            assert_eq!(this5, this2);
            assert_eq!(this4, this1);

            assert_eq!(arena[this4].key, 4);
            assert_eq!(arena[this5].key, 5);
        }
    }

    #[test]
    fn test_arena_with_capacity_allocation() {
        let capacity = 100;
        let mut arena: Arena<i32, String> = Arena::with_capacity(capacity);

        for i in 0..10 {
            let ptr = arena.alloc_circular(i, format!("value_{}", i));
            unsafe {
                let slot = ptr.as_ref();
                assert_eq!(slot.data.assume_init_ref().key, i);
                assert_eq!(slot.data.assume_init_ref().value, format!("value_{}", i));
            }
        }
    }

    #[test]
    fn test_lldata_key_value_mut() {
        let mut data = LLData {
            key: 42,
            value: "hello".to_string(),
        };

        let (key, value) = data.key_value_mut();
        assert_eq!(*key, 42);
        assert_eq!(value, "hello");

        *value = "modified".to_string();
        assert_eq!(data.value, "modified");
    }

    #[test]
    fn test_arena_drop_with_occupied_slots() {
        {
            let mut arena = Arena::new();
            let _ptr1 = arena.alloc_circular(1, Vec::from([1, 2, 3]));
            let _ptr2 = arena.alloc_circular(2, Vec::from([4, 5, 6]));
            let _ptr3 = arena.alloc_circular(3, Vec::from([7, 8, 9]));
        }
    }

    #[test]
    fn test_arena_drop_with_mixed_slots() {
        {
            let mut arena = Arena::new();
            let ptr1 = arena.alloc_circular(1, Vec::from([1, 2, 3]));
            let _ptr2 = arena.alloc_circular(2, Vec::from([4, 5, 6]));

            unsafe {
                arena.free_and_unlink(ptr1);
            }
        }
    }

    #[test]
    fn test_freed_slot_structure() {
        let mut arena = Arena::new();
        let ptr1 = arena.alloc_circular(1, "first".to_string());
        let ptr2 = arena.alloc(2, "second".to_string(), ptr1, ptr1);

        unsafe {
            let freed = arena.free_and_unlink(ptr2);

            assert_eq!(freed.data.key, 2);
            assert_eq!(freed.data.value, "second");
        }
    }

    #[test]
    #[should_panic(expected = "Invalid Ptr")]
    fn test_arena_index_panic_on_freed() {
        let mut arena = Arena::new();
        let ptr = arena.alloc_circular(42, "test".to_string());

        unsafe {
            let slot = ptr.as_ref();
            let this_ptr = slot.this;
            arena.free_and_unlink(ptr);
            let _ = &arena[this_ptr];
        }
    }

    #[test]
    #[should_panic(expected = "Invalid Ptr")]
    fn test_arena_index_mut_panic_on_freed() {
        let mut arena = Arena::new();
        let ptr = arena.alloc_circular(42, "test".to_string());

        unsafe {
            let slot = ptr.as_ref();
            let this_ptr = slot.this;
            arena.free_and_unlink(ptr);
            let _ = &mut arena[this_ptr];
        }
    }

    #[test]
    fn test_arena_mixed_alloc_patterns() {
        let mut arena = Arena::new();
        let ptr1 = arena.alloc_circular(1, "first".to_string());

        unsafe {
            let ptr2 = arena.alloc(2, "second".to_string(), ptr1, ptr1);
            let ptr3 = arena.alloc(3, "third".to_string(), ptr2, ptr1);

            assert_eq!(arena[ptr1.as_ref().this].key, 1);
            assert_eq!(arena[ptr2.as_ref().this].key, 2);
            assert_eq!(arena[ptr3.as_ref().this].key, 3);

            assert_eq!(
                ptr2.as_ref().links,
                Links::Active {
                    prev: ptr1,
                    next: ptr1
                }
            );
            assert_eq!(
                ptr3.as_ref().links,
                Links::Active {
                    prev: ptr2,
                    next: ptr1
                }
            );
        }
    }

    #[test]
    fn test_arena_free_list_order() {
        let mut arena = Arena::new();
        let ptr1 = arena.alloc_circular(1, "first".to_string());
        let ptr2 = arena.alloc_circular(2, "second".to_string());
        let ptr3 = arena.alloc_circular(3, "third".to_string());

        unsafe {
            let this1 = ptr1.as_ref().this;
            let this2 = ptr2.as_ref().this;
            let this3 = ptr3.as_ref().this;

            arena.free_and_unlink(ptr1);
            arena.free_and_unlink(ptr2);
            arena.free_and_unlink(ptr3);

            let new_ptr3 = arena.alloc_circular(30, "new_third".to_string());
            let new_ptr2 = arena.alloc_circular(20, "new_second".to_string());
            let new_ptr1 = arena.alloc_circular(10, "new_first".to_string());

            assert_eq!(new_ptr3.as_ref().this, this3);
            assert_eq!(new_ptr2.as_ref().this, this2);
            assert_eq!(new_ptr1.as_ref().this, this1);

            assert_eq!(arena[this3].key, 30);
            assert_eq!(arena[this2].key, 20);
            assert_eq!(arena[this1].key, 10);
        }
    }

    #[test]
    fn test_arena_zero_capacity() {
        let arena: Arena<i32, String> = Arena::with_capacity(0);
        assert!(!arena.is_occupied(Ptr::unchecked_from(0)));
    }

    #[test]
    fn test_arena_map_ptr_after_free() {
        let mut arena = Arena::new();
        let ptr = arena.alloc_circular(42, "test".to_string());

        unsafe {
            let slot = ptr.as_ref();
            let this_ptr = slot.this;

            assert!(arena.map_ptr(this_ptr).is_some());
            arena.free_and_unlink(ptr);
            assert!(arena.map_ptr(this_ptr).is_none());
        }
    }

    #[test]
    fn test_arena_alloc_with_different_prev_next() {
        let mut arena = Arena::new();
        let ptr1 = arena.alloc_circular(1, "first".to_string());
        let ptr2 = arena.alloc_circular(2, "second".to_string());

        unsafe {
            let ptr3 = arena.alloc(3, "third".to_string(), ptr1, ptr2);

            assert_eq!(
                ptr3.as_ref().links,
                Links::Active {
                    prev: ptr1,
                    next: ptr2
                }
            );
        }
    }

    #[test]
    fn test_arena_unlink_middle_updates_neighbors_and_reuse() {
        let mut arena = Arena::new();
        let mut a = arena.alloc_circular('a', 1);
        let mut b = arena.alloc('b', 2, a, a);
        let mut c = arena.alloc('c', 3, b, a);

        unsafe {
            *a.as_mut().links.next_mut() = b;
            *a.as_mut().links.prev_mut() = b;

            *b.as_mut().links.next_mut() = c;
            *b.as_mut().links.prev_mut() = a;
            *a.as_mut().links.prev_mut() = c;

            assert_eq!(a.as_ref().links.next().as_ptr(), b.as_ptr());
            assert_eq!(b.as_ref().links.prev().as_ptr(), a.as_ptr());
            assert_eq!(b.as_ref().links.next().as_ptr(), c.as_ptr());
            assert_eq!(c.as_ref().links.prev().as_ptr(), b.as_ptr());
            assert_eq!(c.as_ref().links.next().as_ptr(), a.as_ptr());
            assert_eq!(a.as_ref().links.prev().as_ptr(), c.as_ptr());

            let freed_b = arena.free_and_unlink(b);
            assert_eq!(freed_b.data.key, 'b');
            assert!(arena.map_ptr(freed_b.this).is_none());

            assert_eq!(a.as_ref().links.next().as_ptr(), c.as_ptr());
            assert_eq!(c.as_ref().links.prev().as_ptr(), a.as_ptr());

            let d = arena.alloc('d', 4, a, c);
            *a.as_mut().links.next_mut() = d;
            *c.as_mut().links.prev_mut() = d;
            assert_eq!(d.as_ref().this, freed_b.this);
            assert_eq!(a.as_ref().links.next().as_ptr(), d.as_ptr());
            assert_eq!(c.as_ref().links.prev().as_ptr(), d.as_ptr());
        }
    }
}
