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

#[cfg(debug_assertions)]
pub(crate) type ArenaContainer<K, T> = Box<Arena<K, T>>;
#[cfg(not(debug_assertions))]
pub(crate) type ArenaContainer<K, T> = Arena<K, T>;

pub(crate) struct ActiveSlotRef<K, T> {
    slot: NonNull<LLSlot<K, T>>,
    #[cfg(debug_assertions)]
    arena: *const Arena<K, T>,
}

impl<K, T> Debug for ActiveSlotRef<K, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "ActiveSlotRef({:p})", self.slot)
    }
}

impl<K, T> ActiveSlotRef<K, T> {
    #[inline(always)]
    pub(crate) fn data<'a>(&self, _arena: &'a Arena<K, T>) -> &'a NodeData<K, T> {
        #[cfg(debug_assertions)]
        {
            debug_assert_eq!(
                self.arena, _arena as *const Arena<K, T>,
                "ActiveSlotRef used with wrong Arena"
            );
        }
        // SAFETY: Our slot is allocated from the incoming arena, so no UAF. We can't
        // have any mutable borrows out because we always tie them to &mut arena
        // methods. We have no uninitialized data in Active slots.
        unsafe { self.slot.as_ref().data.assume_init_ref() }
    }

    #[inline(always)]
    pub(crate) fn data_mut<'a>(&mut self, _arena: &'a mut Arena<K, T>) -> &'a mut NodeData<K, T> {
        #[cfg(debug_assertions)]
        {
            debug_assert_eq!(
                self.arena, _arena as *const Arena<K, T>,
                "ActiveSlotRef used with wrong Arena"
            );
        }

        // SAFETY: Our slot is allocated from the incoming arena, so no UAF. We can't
        // have any other borrows because we always tie borrows to the lifetime of
        // either &arena or &mut arena. We have no uninitialized data in Active
        // slots.
        unsafe { self.slot.as_mut().data.assume_init_mut() }
    }

    #[inline(always)]
    pub(crate) fn this(&self, _arena: &Arena<K, T>) -> Ptr {
        #[cfg(debug_assertions)]
        {
            debug_assert_eq!(
                self.arena, _arena as *const Arena<K, T>,
                "ActiveSlotRef used with wrong Arena"
            );
        }

        // SAFETY: Our slot is always occupied if we have an ActiveSlotRef.
        unsafe { self.slot.as_ref().this }
    }

    #[inline(always)]
    pub(crate) fn next(&self, _arena: &Arena<K, T>) -> ActiveSlotRef<K, T> {
        #[cfg(debug_assertions)]
        {
            debug_assert_eq!(
                self.arena, _arena as *const Arena<K, T>,
                "ActiveSlotRef used with wrong Arena"
            );
        }
        // SAFETY: Our slot is always occupied if we have an ActiveSlotRef.
        unsafe {
            match &self.slot.as_ref().links {
                Links::Occupied { next, .. } => *next,
                Links::Vacant { .. } => unreachable_unchecked(),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn next_mut<'a>(
        &mut self,
        _arena: &'a mut Arena<K, T>,
    ) -> &'a mut ActiveSlotRef<K, T> {
        #[cfg(debug_assertions)]
        {
            debug_assert_eq!(
                self.arena, _arena as *const Arena<K, T>,
                "ActiveSlotRef used with wrong Arena"
            );
        }
        // SAFETY: Our slot is allocated from the incoming arena, so no UAF. We can't
        // have any other borrows because we always tie borrows to the lifetime of
        // either &arena or &mut arena. We have no uninitialized data in Active
        // slots.
        unsafe {
            match &mut self.slot.as_mut().links {
                Links::Occupied { next, .. } => next,
                Links::Vacant { .. } => unreachable_unchecked(),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn prev(&self, _arena: &Arena<K, T>) -> ActiveSlotRef<K, T> {
        #[cfg(debug_assertions)]
        {
            debug_assert_eq!(
                self.arena, _arena as *const Arena<K, T>,
                "ActiveSlotRef used with wrong Arena"
            );
        }
        // SAFETY: Our slot is always occupied if we have an ActiveSlotRef.
        unsafe {
            match &self.slot.as_ref().links {
                Links::Occupied { prev, .. } => *prev,
                Links::Vacant { .. } => unreachable_unchecked(),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn prev_mut<'a>(
        &mut self,
        _arena: &'a mut Arena<K, T>,
    ) -> &'a mut ActiveSlotRef<K, T> {
        #[cfg(debug_assertions)]
        {
            debug_assert_eq!(
                self.arena, _arena as *const Arena<K, T>,
                "ActiveSlotRef used with wrong Arena"
            );
        }
        // SAFETY: Our slot is allocated from the incoming arena, so no UAF. We can't
        // have any other borrows because we always tie borrows to the lifetime of
        // either &arena or &mut arena. We have no uninitialized data in Active
        // slots.
        unsafe {
            match &mut self.slot.as_mut().links {
                Links::Occupied { prev, .. } => prev,
                Links::Vacant { .. } => unreachable_unchecked(),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn as_ptr(&self) -> NonNull<LLSlot<K, T>> {
        self.slot
    }
}

impl<K, T> Clone for ActiveSlotRef<K, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K, T> Copy for ActiveSlotRef<K, T> {}

impl<K, T> PartialEq for ActiveSlotRef<K, T> {
    fn eq(&self, other: &Self) -> bool {
        self.slot == other.slot
    }
}

impl<K, T> Eq for ActiveSlotRef<K, T> {}

pub(crate) struct NodeData<K, T> {
    pub(crate) key: K,
    pub(crate) value: T,
}

pub(crate) enum Links<K, T> {
    Occupied {
        prev: ActiveSlotRef<K, T>,
        next: ActiveSlotRef<K, T>,
    },
    Vacant {
        next_free: Option<NonNull<LLSlot<K, T>>>,
    },
}

impl<K, T> Debug for Links<K, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Links::Occupied { prev, next } => f
                .debug_struct("Links::Occupied")
                .field("prev", prev)
                .field("next", next)
                .finish(),
            Links::Vacant { next_free } => f
                .debug_struct("Links::Vacant")
                .field("next_free", next_free)
                .finish(),
        }
    }
}

pub(crate) struct LLSlot<K, T> {
    pub(crate) this: Ptr,
    pub(crate) links: Links<K, T>,
    pub(crate) data: MaybeUninit<NodeData<K, T>>,
}

impl<K, T> Debug for LLSlot<K, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LLSlot")
            .field("this", &self.this)
            .field("links", &self.links)
            .finish()
    }
}

pub(crate) struct FreedSlot<K, T> {
    pub(crate) data: NodeData<K, T>,
    pub(crate) this: Ptr,
    pub(crate) prev_next: Option<(ActiveSlotRef<K, T>, ActiveSlotRef<K, T>)>,
}

#[derive(Debug)]
pub(crate) struct Arena<K, T> {
    bump: Bump,
    slots: Vec<NonNull<LLSlot<K, T>>>,
    free_list_head: Option<NonNull<LLSlot<K, T>>>,
}

impl<K, T> Drop for Arena<K, T> {
    fn drop(&mut self) {
        // SAFETY: We have exclusive access to the arena, so we can safely drop
        // all occupied slots.
        unsafe {
            for slot_ptr in &mut self.slots {
                let slot_ref = slot_ptr.as_mut();
                match &slot_ref.links {
                    Links::Occupied { .. } => {
                        slot_ref.data.assume_init_drop();
                    }
                    Links::Vacant { .. } => {}
                }
            }
        }
    }
}

impl<K, T> Arena<K, T> {
    #[inline]
    pub(crate) fn with_capacity(capacity: usize) -> ArenaContainer<K, T> {
        #[cfg(debug_assertions)]
        {
            Box::new(Self {
                bump: Bump::with_capacity(capacity * core::mem::size_of::<LLSlot<K, T>>()),
                slots: Vec::with_capacity(capacity),
                free_list_head: None,
            })
        }
        #[cfg(not(debug_assertions))]
        Self {
            bump: Bump::with_capacity(capacity * core::mem::size_of::<LLSlot<K, T>>()),
            slots: Vec::with_capacity(capacity),
            free_list_head: None,
        }
    }

    #[inline]
    pub(crate) fn map_ptr(&self, ptr: Ptr) -> Option<ActiveSlotRef<K, T>> {
        self.slots
            .get(ptr.unchecked_get())
            .filter(|slot| {
                // SAFETY: Slot was allocated from our own arena, so it's still valid.
                unsafe { matches!(slot.as_ref().links, Links::Occupied { .. }) }
            })
            .map(|&slot| ActiveSlotRef {
                slot,
                #[cfg(debug_assertions)]
                arena: self as *const Arena<K, T>,
            })
    }

    #[inline]
    pub(crate) fn is_occupied(&self, ptr: Ptr) -> bool {
        self.slots.get(ptr.unchecked_get()).is_some_and(|slot| {
            // SAFETY: Slot was allocated from our own arena, so it's still valid.
            unsafe { matches!(slot.as_ref().links, Links::Occupied { .. }) }
        })
    }

    #[inline]
    pub(crate) fn alloc_circular(&mut self, key: K, value: T) -> ActiveSlotRef<K, T> {
        if let Some(mut free_slot) = self.free_list_head {
            // SAFETY: The free list only contains valid, uninitialized slots
            // that are not currently in use. We have exclusive
            // access to the arena, so we can safely take a slot
            // from the free list.
            unsafe {
                let next_free = match &free_slot.as_ref().links {
                    Links::Vacant { next_free } => *next_free,
                    Links::Occupied { .. } => unreachable_unchecked(),
                };

                self.free_list_head = next_free;

                let active_slot = ActiveSlotRef {
                    slot: free_slot,
                    #[cfg(debug_assertions)]
                    arena: self as *const Arena<K, T>,
                };

                free_slot.as_mut().links = Links::Occupied {
                    prev: active_slot,
                    next: active_slot,
                };

                free_slot.as_mut().data = MaybeUninit::new(NodeData { key, value });

                active_slot
            }
        } else {
            let ptr = Ptr::unchecked_from(self.slots.len());
            let slot = self.bump.alloc(LLSlot {
                this: ptr,
                links: Links::Vacant { next_free: None },
                data: MaybeUninit::new(NodeData { key, value }),
            });
            let mut slot_ptr = NonNull::from_mut(slot);
            let active_slot = ActiveSlotRef {
                slot: slot_ptr,
                #[cfg(debug_assertions)]
                arena: self as *const Arena<K, T>,
            };
            // SAFETY: We just allocated this slot.
            unsafe {
                slot_ptr.as_mut().links = Links::Occupied {
                    prev: active_slot,
                    next: active_slot,
                };
            }

            self.slots.push(slot_ptr);
            active_slot
        }
    }

    #[inline]
    pub(crate) fn alloc(
        &mut self,
        key: K,
        value: T,
        prev: ActiveSlotRef<K, T>,
        next: ActiveSlotRef<K, T>,
    ) -> ActiveSlotRef<K, T> {
        if let Some(mut free_slot) = self.free_list_head {
            // SAFETY: The free list only contains valid, uninitialized slots
            // that are not currently in use. We have exclusive
            // access to the arena, so we can safely take a slot
            // from the free list.
            unsafe {
                let next_free = match &free_slot.as_ref().links {
                    Links::Vacant { next_free } => *next_free,
                    Links::Occupied { .. } => unreachable_unchecked(),
                };

                self.free_list_head = next_free;

                let active_slot = ActiveSlotRef {
                    slot: free_slot,
                    #[cfg(debug_assertions)]
                    arena: self as *const Arena<K, T>,
                };

                free_slot.as_mut().links = Links::Occupied { prev, next };
                free_slot.as_mut().data = MaybeUninit::new(NodeData { key, value });

                active_slot
            }
        } else {
            let ptr = Ptr::unchecked_from(self.slots.len());
            let slot = self.bump.alloc(LLSlot {
                this: ptr,
                links: Links::Occupied { prev, next },
                data: MaybeUninit::new(NodeData { key, value }),
            });
            let slot_ptr = NonNull::from_mut(slot);
            let active_slot = ActiveSlotRef {
                slot: slot_ptr,
                #[cfg(debug_assertions)]
                arena: self as *const Arena<K, T>,
            };

            self.slots.push(slot_ptr);
            active_slot
        }
    }

    // # SAFETY
    //
    // The caller **must not** deref any copies of the passed in slot after calling
    // this function.
    #[inline]
    pub(crate) unsafe fn free_and_unlink(
        &mut self,
        mut slot: ActiveSlotRef<K, T>,
    ) -> FreedSlot<K, T> {
        // SAFETY: Active slots are occupied, so we can access their data and links.
        let (prev_next, data, this) = unsafe {
            let slot_ref = slot.slot.as_mut();
            match slot_ref.links {
                Links::Occupied { mut prev, mut next } => {
                    let data = slot_ref.data.assume_init_read();
                    let this = slot_ref.this;

                    slot_ref.links = Links::Vacant {
                        next_free: self.free_list_head,
                    };
                    self.free_list_head = Some(slot.slot);

                    let prev_next = if prev == slot || next == slot {
                        None
                    } else {
                        *prev.next_mut(self) = next;
                        *next.prev_mut(self) = prev;

                        Some((prev, next))
                    };
                    (prev_next, data, this)
                }
                Links::Vacant { .. } => unreachable_unchecked(),
            }
        };

        FreedSlot {
            data,
            this,
            prev_next,
        }
    }
}

impl<K, T> Index<Ptr> for Arena<K, T> {
    type Output = NodeData<K, T>;

    #[inline]
    fn index(&self, index: Ptr) -> &Self::Output {
        self.map_ptr(index)
            .as_ref()
            .expect("Indexing with invalid or freed Ptr")
            .data(self)
    }
}

impl<K, T> IndexMut<Ptr> for Arena<K, T> {
    #[inline]
    fn index_mut(&mut self, index: Ptr) -> &mut Self::Output {
        self.map_ptr(index)
            .as_mut()
            .expect("Indexing with invalid or freed Ptr")
            .data_mut(self)
    }
}
