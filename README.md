# tether-map

[![Crates.io](https://img.shields.io/crates/v/tether-map.svg?style=for-the-badge)](https://crates.io/crates/tether-map)
[![Docs.rs](https://img.shields.io/docsrs/tether-map/latest?style=for-the-badge)](https://docs.rs/tether-map)
[![Dependency status](https://deps.rs/repo/github/jesterhearts/tether-map/status.svg?style=for-the-badge)](https://deps.rs/repo/github/jesterhearts/tether-map)

An order-preserving hash map built on an intrusive doubly‑linked list with O(1) reordering.

- O(1) insert, remove, and lookup by key
- Maintains a stable relative order of entries
- O(1) reordering: move entries to head/tail or before/after any other entry
- Cursor API for in-place navigation and mutation
- `no_std` compatible (needs `alloc`)
- Maximum capacity of 2^32 - 1 entries.
- Safe public API; `unsafe` only in well-audited internals

## Install

Add to your Cargo.toml

```toml
[dependencies]
tether-map = "0.1"
```

no_std users:

```toml
[dependencies]
tether-map = { version = "0.1", default-features = false }
```

The crate uses `alloc` internally and does not require `std` unless the `std` feature is enabled
(enabled by default).

Feature flags:

- `std` (default): Enables std-dependent conveniences like the default hasher and common traits.

## Quick start

```rust
use tether_map::LinkedHashMap;

let mut map = LinkedHashMap::new();
map.insert_tail("first", 1);
map.insert_tail("second", 2);
map.insert_head("zeroth", 0);

// Iteration follows insertion order
let keys: Vec<_> = map.iter().map(|(k, _)| *k).collect();
assert_eq!(keys, ["zeroth", "first", "second"]);

// O(1) reordering via Ptr handles
let ptr = map.get_ptr(&"second").unwrap();
map.move_to_head(ptr);
let keys: Vec<_> = map.iter().map(|(k, _)| *k).collect();
assert_eq!(keys, ["second", "zeroth", "first"]);
```

## LRU Example
```rust
use tether_map::LinkedHashMap;
let mut lru = LinkedHashMap::new();
lru.insert_tail("a", 1);
lru.insert_tail("b", 2);
lru.insert_tail("c", 3);
assert_eq!(lru.len(), 3);

// Access "a" and move to most-recent position
if let Some(ptr) = lru.get_ptr(&"a") {
	lru.move_to_tail(ptr);
}		
assert_eq!(lru.iter().map(|(k, _)| *k).collect::<Vec<_>>(), ["b", "c", "a"]);
// Insert "d", evicting the least-recently used entry ("b")
lru.insert_tail("d", 4);
if lru.len() > 3 {
	let (_, removed_entry) = lru.remove_head().unwrap();
	println!("Evicted {}", removed_entry.key);
}
assert_eq!(lru.iter().map(|(k, _)| *k).collect::<Vec<_>>(), ["c", "a", "d"]);
```

## API highlights

- Insertion positions:
	- `insert`, `insert_full`: insert/update without moving existing entries
	- `insert_head(_full)`, `insert_tail(_full)`: insert or update and move to head/tail
- Reordering:
	- `move_to_head`, `move_to_tail`, `move_before`, `move_after` (all O(1))
- Pointers:
	- `get_ptr(&key) -> Option<Ptr>` to obtain an O(1) handle to entries
	- `ptr_get`, `ptr_get_mut`, `ptr_get_entry(_mut)` for direct access
- Cursors:
	- `head_cursor_mut`, `tail_cursor_mut`, `ptr_cursor_mut`, `key_cursor_mut`
	- `CursorMut` supports navigation and in-place edits without extra lookups
- Iteration:
	- `iter`, `iter_mut`, `keys`, `values`, `values_mut` (double-ended where applicable)

## no_std

The crate compiles with `#![no_std]` when the `std` feature is disabled and requires the `alloc` crate at runtime. The default configuration enables `std`.

## Performance and complexity

- Average-case O(1) for insert, remove, lookup, and reordering operations.
- Order is maintained via an intrusive doubly‑linked list; the hash table provides lookups.
- The internal arena reuses freed slots to avoid unnecessary allocations.

Benchmarks: Criterion benchmarks live in `benches/` (with comparisons against `indexmap` and
`hashlink`). Run them locally with `cargo bench`; a report is written under `target/criterion/`.

## Safety notes and pointer semantics

- `Ptr` is a compact, non‑generational handle to an entry. It remains valid until that entry is
  removed from the map. After removal, the pointer should not be used as the same `Ptr` value may be
  reused for a different entry later.
- Cursors and pointer APIs assume pointers originate from this map. Using stale/foreign pointers is
  a logic error and may return unexpected results or panic but will not result in undefined behavior.

## Usage of Unsafe

This crate uses `unsafe` internally where it provides significant, measurable speedups. In several
microbenchmarks this yields over 2x performance in hot paths compared to fully safe alternatives.

- Scope: `unsafe` is largely contained within the internal arena implementation (`src/arena.rs`) and
  a few small pointer‑manipulation helpers. The public API is safe.
- Documentation: Every `unsafe` block is accompanied by a `// SAFETY:` comment explaining the
  invariants and preconditions being relied upon.
- Verification: Tests are regularly exercised under Miri to catch UB‑adjacent mistakes in pointer
  and aliasing logic.
- Extra checks: Debug builds enable additional assertions to validate the source of pointers in
  order to double check lifetime management. These checks impose a 30%+ overhead and are therefore
  disabled in release builds.

## License

Dual-licensed under either of:

- Apache License, Version 2.0
- MIT license

at your option.
