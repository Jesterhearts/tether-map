# tether-map

A linked hash map with intrusive doubly-linked list ordering and O(1) reordering operations.

## Features

- O(1) insertion, removal, and lookup based on keys
- Maintains relative order of entries
- O(1) reordering operations - Move entries to head/tail or before/after other entries
- Cursor-based navigation and modification
- `no_std` compatible (requires `alloc` and `core`).

## Quick Start

```rust
use tether_map::LinkedHashMap;

let mut map = LinkedHashMap::new();
map.insert_tail("first", 1);
map.insert_tail("second", 2);
map.insert_head("zeroth", 0);

// Iteration follows insertion order
let keys: Vec<_> = map.iter().map(|(k, _)| *k).collect();
assert_eq!(keys, ["zeroth", "first", "second"]);

// O(1) reordering
let ptr = map.get_ptr(&"second").unwrap();
map.move_to_head(ptr);
let keys: Vec<_> = map.iter().map(|(k, _)| *k).collect();
assert_eq!(keys, ["second", "zeroth", "first"]);
```

## Usage of Unsafe
There is a small amount of unsafe code used to implement `IterMut`, which is only enabled when the
`iter-mut` feature is activated. The unsafe code is carefully reviewed to ensure safety and is
checked with miri. If you don't need `IterMut`, you can disable it, and the crate will turn on
`#[forbid(unsafe_code)]` to ensure no unsafe code is present. You can see this behavior in `lib.rs`.