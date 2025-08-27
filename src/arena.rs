use alloc::vec::Vec;
use core::{
    clone::Clone,
    ops::{
        Index,
        IndexMut,
    },
    panic,
};

use crate::Ptr;

#[cold]
#[inline(never)]
fn assert_free() -> ! {
    panic!("Attempted to access data of free slot");
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct LLData<K, T> {
    pub(crate) prev: Ptr,
    pub(crate) hash: u64,
    pub(crate) key: K,
    pub(crate) value: T,
}

#[derive(Debug, Clone, Copy)]
enum DataOrFree<K, T> {
    Free,
    Data(LLData<K, T>),
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct LLSlot<K, T> {
    next: Ptr,
    data: DataOrFree<K, T>,
}

impl<K, T> LLSlot<K, T> {
    pub(crate) fn prev(&self) -> Ptr {
        match &self.data {
            DataOrFree::Data(data) => data.prev,
            DataOrFree::Free => assert_free(),
        }
    }

    pub(crate) fn prev_mut(&mut self) -> &mut Ptr {
        match &mut self.data {
            DataOrFree::Data(data) => &mut data.prev,
            DataOrFree::Free => assert_free(),
        }
    }

    pub(crate) fn next(&self) -> Ptr {
        self.next
    }

    pub(crate) fn next_mut(&mut self) -> &mut Ptr {
        &mut self.next
    }

    pub(crate) fn into_data(self) -> LLData<K, T> {
        match self.data {
            DataOrFree::Data(data) => data,
            DataOrFree::Free => assert_free(),
        }
    }

    pub(crate) fn data(&self) -> &LLData<K, T> {
        match &self.data {
            DataOrFree::Data(data) => data,
            DataOrFree::Free => assert_free(),
        }
    }

    pub(crate) fn data_mut(&mut self) -> &mut LLData<K, T> {
        match &mut self.data {
            DataOrFree::Data(data) => data,
            DataOrFree::Free => assert_free(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Arena<K, T> {
    nodes: Vec<LLSlot<K, T>>,
    free_head: Ptr,
}

impl<K, T> Arena<K, T> {
    pub(crate) fn new() -> Self {
        Arena {
            nodes: Vec::new(),
            free_head: Ptr::null(),
        }
    }

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Arena {
            nodes: Vec::with_capacity(capacity),
            free_head: Ptr::null(),
        }
    }

    pub(crate) fn links(&self, ptr: Ptr) -> &LLSlot<K, T> {
        &self.nodes[ptr.unchecked_get()]
    }

    pub(crate) fn links_mut(&mut self, ptr: Ptr) -> &mut LLSlot<K, T> {
        &mut self.nodes[ptr.unchecked_get()]
    }

    pub(crate) fn clear(&mut self) {
        self.nodes.clear();
        self.free_head = Ptr::null();
    }

    pub(crate) fn shrink_to_fit(&mut self) {
        // Note: This may not even shrink anything if the arena has free slots. In
        // general, it's not possible to move around the nodes, since there may be
        // external Ptrs pointing to them. So this is the best we can do.
        // It *might* be possible to compact the arena by moving occupied nodes to
        // fill in free slots, but would require keeping a mapping of all moved Ptrs so
        // they can be remapped when calling free/index/etc. That might even *increase*
        // memory used depending on the exact usage pattern and would both add
        // complexity and likely be slower in the happy path.
        self.nodes.shrink_to_fit();
    }

    pub(crate) fn next_ptr(&self) -> Ptr {
        self.free_head.or(Ptr::unchecked_from(self.nodes.len()))
    }

    pub(crate) fn alloc(&mut self, key: K, value: T, hash: u64, prev: Ptr, next: Ptr) -> Ptr {
        if !self.free_head.is_null() {
            let old = core::mem::replace(
                &mut self.nodes[self.free_head.unchecked_get()],
                LLSlot {
                    next,
                    data: DataOrFree::Data(LLData {
                        prev,
                        key,
                        value,
                        hash,
                    }),
                },
            );
            let ptr = self.free_head;
            self.free_head = old.next;
            ptr
        } else {
            let ptr = Ptr::unchecked_from(self.nodes.len());
            self.nodes.push(LLSlot {
                next,
                data: DataOrFree::Data(LLData {
                    prev,
                    key,
                    value,
                    hash,
                }),
            });
            ptr
        }
    }

    pub(crate) fn is_occupied(&self, ptr: Ptr) -> bool {
        if ptr.is_null() {
            return false;
        }
        matches!(self.nodes[ptr.unchecked_get()].data, DataOrFree::Data(_))
    }

    pub(crate) fn free(&mut self, ptr: Ptr) -> LLSlot<K, T> {
        assert!(self.is_occupied(ptr), "Pointer to free must be occupied");
        let result = core::mem::replace(
            &mut self.nodes[ptr.unchecked_get()],
            LLSlot {
                next: self.free_head,
                data: DataOrFree::Free,
            },
        );
        self.free_head = ptr;

        result
    }

    #[cfg(feature = "iter-mut")]
    pub(crate) fn arena_ptr(&mut self) -> *mut LLSlot<K, T> {
        self.nodes.as_mut_ptr()
    }
}

impl<K, T> Index<Ptr> for Arena<K, T> {
    type Output = LLData<K, T>;

    fn index(&self, index: Ptr) -> &Self::Output {
        self.nodes[index.unchecked_get()].data()
    }
}

impl<K, T> IndexMut<Ptr> for Arena<K, T> {
    fn index_mut(&mut self, index: Ptr) -> &mut Self::Output {
        self.nodes[index.unchecked_get()].data_mut()
    }
}

#[cfg(test)]
mod tests {
    use alloc::{
        format,
        string::ToString,
        vec,
        vec::Vec,
    };
    use core::assert_eq;

    use super::*;

    #[test]
    fn test_ptr_null() {
        let null_ptr = Ptr::null();
        assert!(null_ptr.is_null());
        assert_eq!(null_ptr.optional(), None);
    }

    #[test]
    fn test_ptr_non_null() {
        let ptr = Ptr::unchecked_from(42);
        assert!(!ptr.is_null());
        assert_eq!(ptr.optional(), Some(ptr));
        assert_eq!(ptr.unchecked_get(), 42);
    }

    #[test]
    fn test_ptr_or() {
        let null_ptr = Ptr::null();
        let some_ptr = Ptr::unchecked_from(10);
        let other_ptr = Ptr::unchecked_from(20);

        assert_eq!(null_ptr.or(some_ptr), some_ptr);
        assert_eq!(some_ptr.or(other_ptr), some_ptr);
    }

    #[test]
    fn test_ptr_debug() {
        let null_ptr = Ptr::null();
        let some_ptr = Ptr::unchecked_from(42);

        assert_eq!(format!("{:?}", null_ptr), "Ptr(null)");
        assert_eq!(format!("{:?}", some_ptr), "Ptr(42)");
    }

    #[test]
    fn test_ptr_default() {
        let default_ptr: Ptr = Default::default();
        assert!(default_ptr.is_null());
    }

    #[test]
    fn test_ptr_equality() {
        let ptr1 = Ptr::unchecked_from(42);
        let ptr2 = Ptr::unchecked_from(42);
        let ptr3 = Ptr::unchecked_from(43);

        assert_eq!(ptr1, ptr2);
        assert_ne!(ptr1, ptr3);
    }

    #[test]
    fn test_arena_new() {
        let arena: Arena<i32, Vec<i32>> = Arena::new();
        assert_eq!(arena.nodes.len(), 0);
        assert!(arena.free_head.is_null());
    }

    #[test]
    fn test_arena_with_capacity() {
        let arena: Arena<i32, Vec<i32>> = Arena::with_capacity(10);
        assert_eq!(arena.nodes.capacity(), 10);
    }

    #[test]
    fn test_arena_alloc_single() {
        let mut arena = Arena::new();
        let ptr = arena.alloc(42, vec![1, 2, 3, 4, 5], 12345, Ptr::null(), Ptr::null());

        assert!(!ptr.is_null());
        assert!(arena.is_occupied(ptr));
        assert_eq!(arena.nodes.len(), 1);

        let data = &arena[ptr];
        assert_eq!(data.key, 42);
        assert_eq!(data.value, [1, 2, 3, 4, 5]);
        assert_eq!(data.hash, 12345);
    }

    #[test]
    fn test_arena_alloc_multiple() {
        let mut arena = Arena::new();
        let ptr1 = arena.alloc(1, "one".to_string(), 111, Ptr::null(), Ptr::null());
        let ptr2 = arena.alloc(2, "two".to_string(), 222, Ptr::null(), Ptr::null());
        let ptr3 = arena.alloc(3, "three".to_string(), 333, Ptr::null(), Ptr::null());

        assert_ne!(ptr1, ptr2);
        assert_ne!(ptr2, ptr3);
        assert_ne!(ptr1, ptr3);

        assert!(arena.is_occupied(ptr1));
        assert!(arena.is_occupied(ptr2));
        assert!(arena.is_occupied(ptr3));

        assert_eq!(arena[ptr1].key, 1);
        assert_eq!(arena[ptr2].key, 2);
        assert_eq!(arena[ptr3].key, 3);
    }

    #[test]
    fn test_arena_free_and_reuse() {
        let mut arena = Arena::new();
        let ptr1 = arena.alloc(1, "one".to_string(), 111, Ptr::null(), Ptr::null());
        let ptr2 = arena.alloc(2, "two".to_string(), 222, Ptr::null(), Ptr::null());

        assert!(arena.is_occupied(ptr1));
        assert!(arena.is_occupied(ptr2));

        let data = arena.free(ptr1);
        assert_eq!(data.data().key, 1);
        assert_eq!(data.data().value, "one");
        assert!(!arena.is_occupied(ptr1));
        assert!(arena.is_occupied(ptr2));

        let ptr3 = arena.alloc(3, "three".to_string(), 333, Ptr::null(), Ptr::null());
        assert_eq!(ptr3, ptr1);
        assert!(arena.is_occupied(ptr3));
        assert_eq!(arena[ptr3].key, 3);
    }

    #[test]
    fn test_arena_index_operations() {
        let mut arena = Arena::new();
        let ptr = arena.alloc(42, "hello".to_string(), 12345, Ptr::null(), Ptr::null());

        let data = &arena[ptr];
        assert_eq!(data.key, 42);
        assert_eq!(data.value, "hello");

        arena[ptr].value = "world".to_string();
        assert_eq!(arena[ptr].value, "world");
    }

    #[test]
    fn test_arena_links() {
        let mut arena = Arena::new();
        let ptr = arena.alloc(42, "hello".to_string(), 12345, Ptr::null(), Ptr::null());

        let links = arena.links(ptr);
        assert!(links.prev().is_null());
        assert!(links.next().is_null());

        let links_mut = arena.links_mut(ptr);
        *links_mut.prev_mut() = Ptr::unchecked_from(10);
        *links_mut.next_mut() = Ptr::unchecked_from(20);

        let links = arena.links(ptr);
        assert_eq!(links.prev(), Ptr::unchecked_from(10));
        assert_eq!(links.next(), Ptr::unchecked_from(20));
    }

    #[test]
    fn test_arena_clear() {
        let mut arena = Arena::new();
        arena.alloc(1, "one".to_string(), 111, Ptr::null(), Ptr::null());
        arena.alloc(2, "two".to_string(), 222, Ptr::null(), Ptr::null());

        assert_eq!(arena.nodes.len(), 2);

        arena.clear();

        assert_eq!(arena.nodes.len(), 0);
        assert!(arena.free_head.is_null());
    }

    #[test]
    fn test_arena_clone() {
        let mut arena = Arena::new();
        let ptr1 = arena.alloc(1, "one".to_string(), 111, Ptr::null(), Ptr::null());
        let ptr2 = arena.alloc(2, "two".to_string(), 222, Ptr::null(), Ptr::null());

        *arena.links_mut(ptr1).next_mut() = ptr2;
        *arena.links_mut(ptr2).prev_mut() = ptr1;

        let cloned_arena = arena.clone();

        assert_eq!(cloned_arena.nodes.len(), arena.nodes.len());
        assert_eq!(cloned_arena.free_head, arena.free_head);

        assert_eq!(cloned_arena[ptr1].key, arena[ptr1].key);
        assert_eq!(cloned_arena[ptr1].value, arena[ptr1].value);
        assert_eq!(cloned_arena[ptr2].key, arena[ptr2].key);
        assert_eq!(cloned_arena[ptr2].value, arena[ptr2].value);

        assert_eq!(cloned_arena.links(ptr1).next(), ptr2);
        assert_eq!(cloned_arena.links(ptr2).prev(), ptr1);
    }

    #[test]
    fn test_arena_clone_with_free_slots() {
        let mut arena = Arena::new();
        let ptr1 = arena.alloc(1, "one".to_string(), 111, Ptr::null(), Ptr::null());
        let ptr2 = arena.alloc(2, "two".to_string(), 222, Ptr::null(), Ptr::null());
        let ptr3 = arena.alloc(3, "three".to_string(), 333, Ptr::null(), Ptr::null());

        arena.free(ptr2);

        let cloned_arena = arena.clone();

        assert!(cloned_arena.is_occupied(ptr1));
        assert!(!cloned_arena.is_occupied(ptr2));
        assert!(cloned_arena.is_occupied(ptr3));

        assert_eq!(cloned_arena.free_head, arena.free_head);
    }

    #[test]
    #[should_panic]
    fn test_arena_index_unoccupied_ptr() {
        let mut arena = Arena::new();
        let ptr = arena.alloc(1, "one".to_string(), 111, Ptr::null(), Ptr::null());
        arena.free(ptr);
        let _ = &arena[ptr];
    }

    #[test]
    #[should_panic]
    fn test_arena_index_mut_unoccupied_ptr() {
        let mut arena = Arena::new();
        let ptr = arena.alloc(1, "one".to_string(), 111, Ptr::null(), Ptr::null());
        arena.free(ptr);
        let _ = &mut arena[ptr];
    }

    #[test]
    #[should_panic]
    fn test_arena_free_unoccupied_ptr() {
        let mut arena = Arena::new();
        let ptr = arena.alloc(1, "one".to_string(), 111, Ptr::null(), Ptr::null());
        arena.free(ptr);
        arena.free(ptr);
    }

    #[test]
    #[should_panic]
    fn test_arena_free_null_ptr() {
        let mut arena = Arena::<i32, i32>::new();
        arena.free(Ptr::null());
    }

    #[test]
    fn test_arena_is_occupied_null_ptr() {
        let arena: Arena<i32, Vec<i32>> = Arena::new();
        assert!(!arena.is_occupied(Ptr::null()));
    }

    #[test]
    fn test_niche_optimization() {
        use core::mem::size_of;
        assert_eq!(
            size_of::<DataOrFree<Vec<i32>, Vec<i32>>>(),
            size_of::<LLData<Vec<i32>, Vec<i32>>>()
        );
        assert_eq!(
            size_of::<LLSlot<Vec<i32>, Vec<i32>>>(),
            size_of::<(Ptr, Ptr, LLData<Vec<i32>, Vec<i32>>)>()
        );
    }
}
