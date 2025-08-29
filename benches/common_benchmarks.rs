use std::hint::black_box;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::criterion_group;
use criterion::criterion_main;
type RandomState = hashbrown::DefaultHashBuilder;
type TetherMap<K, V> = tether_map::linked_hash_map::LinkedHashMap<K, V, RandomState>;

type HashLinkedMap<K, V> = hashlink::LinkedHashMap<K, V, RandomState>;
type IndexMap<K, V> = indexmap::IndexMap<K, V, RandomState>;

const SIZES: &[usize] = &[10000];

fn bench_insertion_at_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("insertion_at_end");

    for &size in SIZES {
        group.throughput(criterion::Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("tether_map", size), &size, |b, &size| {
            b.iter(|| {
                let mut map: TetherMap<usize, usize> = TetherMap::default();
                for i in 0..size {
                    map.insert_tail(black_box(i), black_box(i * 2));
                }
                map
            })
        });

        group.bench_with_input(
            BenchmarkId::new("tether_map_preallocated", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut map: TetherMap<usize, usize> =
                        TetherMap::with_capacity_and_hasher(size, RandomState::default());
                    for i in 0..size {
                        map.insert_tail(black_box(i), black_box(i * 2));
                    }
                    map
                })
            },
        );

        group.bench_with_input(BenchmarkId::new("indexmap", size), &size, |b, &size| {
            b.iter(|| {
                let mut map = IndexMap::default();
                for i in 0..size {
                    map.insert(black_box(i), black_box(i * 2));
                }
                map
            })
        });

        group.bench_with_input(
            BenchmarkId::new("indexmap_preallocated", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut map = IndexMap::with_capacity_and_hasher(size, RandomState::default());
                    for i in 0..size {
                        map.insert(black_box(i), black_box(i * 2));
                    }
                    map
                })
            },
        );

        group.bench_with_input(BenchmarkId::new("hashlinked", size), &size, |b, &size| {
            b.iter(|| {
                let mut map = HashLinkedMap::default();
                for i in 0..size {
                    map.insert(black_box(i), black_box(i * 2));
                }
                map
            })
        });
    }

    group.finish();
}

fn bench_pop_from_end(c: &mut Criterion) {
    let mut group = c.benchmark_group("pop_from_end");

    for &size in SIZES {
        group.throughput(criterion::Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("tether_map", size), &size, |b, &size| {
            b.iter_batched(
                || {
                    let mut map = TetherMap::default();
                    for i in 0..size {
                        map.insert_tail(i, i * 2);
                    }
                    map
                },
                |mut map| {
                    let mut count = 0;
                    while !map.is_empty() {
                        map.remove_tail();
                        count += 1;
                    }
                    count
                },
                criterion::BatchSize::SmallInput,
            )
        });

        group.bench_with_input(
            BenchmarkId::new("tether_map_remove_ptr", size),
            &size,
            |b, &size| {
                b.iter_batched(
                    || {
                        let mut map = TetherMap::default();
                        for i in 0..size {
                            map.insert_tail_full(i, i * 2);
                        }
                        map
                    },
                    |mut map| {
                        let mut count = 0;
                        while !map.is_empty() {
                            map.remove_ptr(map.tail_ptr().unwrap());
                            count += 1;
                        }
                        count
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(BenchmarkId::new("indexmap", size), &size, |b, &size| {
            b.iter_batched(
                || {
                    let mut map = IndexMap::default();
                    for i in 0..size {
                        map.insert(i, i * 2);
                    }
                    map
                },
                |mut map| {
                    let mut count = 0;
                    while !map.is_empty() {
                        map.pop();
                        count += 1;
                    }
                    count
                },
                criterion::BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("hashlinked", size), &size, |b, &size| {
            b.iter_batched(
                || {
                    let mut map = HashLinkedMap::default();
                    for i in 0..size {
                        map.insert(i, i * 2);
                    }
                    map
                },
                |mut map| {
                    let mut count = 0;
                    while !map.is_empty() {
                        map.pop_back();
                        count += 1;
                    }
                    count
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

fn bench_remove_from_middle(c: &mut Criterion) {
    let mut group = c.benchmark_group("remove_from_middle");

    for &size in SIZES {
        let mut next_down = size / 2;
        let mut next_up = size / 2 + 1;
        let mut middle_keys = Vec::with_capacity(size);
        for _ in 0..size / 2 {
            middle_keys.push(next_down);
            middle_keys.push(next_up);
            next_down = next_down.saturating_sub(1);
            if next_up < size - 1 {
                next_up += 1;
            }
        }

        group.throughput(criterion::Throughput::Elements(middle_keys.len() as u64));

        group.bench_with_input(BenchmarkId::new("tether_map", size), &size, |b, &size| {
            b.iter_batched(
                || {
                    let mut map = TetherMap::default();
                    for i in 0..size {
                        map.insert_tail(i, i * 2);
                    }
                    map
                },
                |mut map| {
                    for &key in &middle_keys {
                        map.remove(&black_box(key));
                    }
                    map
                },
                criterion::BatchSize::SmallInput,
            )
        });

        group.bench_with_input(
            BenchmarkId::new("indexmap_swap_remove", size),
            &size,
            |b, &size| {
                b.iter_batched(
                    || {
                        let mut map = IndexMap::default();
                        for i in 0..size {
                            map.insert(i, i * 2);
                        }
                        map
                    },
                    |mut map| {
                        for &key in &middle_keys {
                            map.swap_remove(&black_box(key));
                        }
                        map
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(BenchmarkId::new("hashlinked", size), &size, |b, &size| {
            b.iter_batched(
                || {
                    let mut map = HashLinkedMap::default();
                    for i in 0..size {
                        map.insert(i, i * 2);
                    }
                    map
                },
                |mut map| {
                    for &key in &middle_keys {
                        map.remove(&black_box(key));
                    }
                    map
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

fn bench_insert_in_middle(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_in_middle");

    for &size in SIZES {
        group.throughput(criterion::Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("tether_map", size), &size, |b, &size| {
            b.iter_batched(
                || {
                    let mut ptrs = Vec::with_capacity(size);
                    let mut map = TetherMap::default();
                    for i in 0..size {
                        let (ptr, _) = map.insert_tail_full(i * 2, (i * 2) * 2);
                        ptrs.push(ptr);
                    }
                    (ptrs, map)
                },
                |(ptrs, mut map)| {
                    for (i, ptr) in ptrs.into_iter().enumerate() {
                        let key = i * 2 + 1;
                        let mut cursor = map.ptr_cursor_mut(ptr);
                        cursor.insert_after_move_to(black_box(key), black_box(key * 2));
                    }
                    map
                },
                criterion::BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("indexmap", size), &size, |b, &size| {
            b.iter_batched(
                || {
                    let mut map = IndexMap::default();
                    for i in 0..size {
                        map.insert(i * 2, (i * 2) * 2);
                    }
                    map
                },
                |mut map| {
                    for i in 0..size {
                        let key = i * 2 + 1;
                        let index = if i < map.len() { i + 1 } else { map.len() };
                        map.shift_insert(black_box(index), black_box(key), black_box(key * 2));
                    }
                    map
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

fn bench_random_access_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_access_full");

    for &size in SIZES {
        let access_keys: Vec<usize> = (0..100).map(|_| rand::random_range(0..size)).collect();

        group.throughput(criterion::Throughput::Elements(access_keys.len() as u64));

        group.bench_with_input(BenchmarkId::new("tether_map", size), &size, |b, &size| {
            let mut map = TetherMap::default();
            for i in 0..size {
                map.insert_tail(i, i * 2);
            }

            b.iter(|| {
                let mut sum = 0;
                for &key in &access_keys {
                    if let Some(value) = map.get(&black_box(key)) {
                        sum += *value;
                    }
                }
                sum
            })
        });

        group.bench_with_input(BenchmarkId::new("indexmap", size), &size, |b, &size| {
            let mut map = IndexMap::default();
            for i in 0..size {
                map.insert(i, i * 2);
            }

            b.iter(|| {
                let mut sum = 0;
                for &key in &access_keys {
                    if let Some(value) = map.get(&black_box(key)) {
                        sum += *value;
                    }
                }
                sum
            })
        });

        group.bench_with_input(BenchmarkId::new("hashlinked", size), &size, |b, &size| {
            let mut map = HashLinkedMap::default();
            for i in 0..size {
                map.insert(i, i * 2);
            }

            b.iter(|| {
                let mut sum = 0;
                for &key in &access_keys {
                    if let Some(value) = map.get(&black_box(key)) {
                        sum += *value;
                    }
                }
                sum
            })
        });
    }

    group.finish();
}

fn bench_random_access_sparse(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_access_sparse");

    for &size in SIZES {
        let access_keys: Vec<usize> = (0..100).map(|_| rand::random_range(0..size)).collect();

        group.throughput(criterion::Throughput::Elements(access_keys.len() as u64));

        group.bench_with_input(BenchmarkId::new("tether_map", size), &size, |b, &size| {
            let mut map = TetherMap::default();
            for i in 0..size {
                map.insert_tail(i, i * 2);
            }

            for i in (0..size).step_by(3) {
                map.remove(&i);
            }

            b.iter(|| {
                let mut sum = 0;
                for &key in &access_keys {
                    if let Some(value) = map.get(&black_box(key)) {
                        sum += *value;
                    }
                }
                sum
            })
        });

        group.bench_with_input(BenchmarkId::new("indexmap", size), &size, |b, &size| {
            let mut map = IndexMap::default();
            for i in 0..size {
                map.insert(i, i * 2);
            }
            for i in (0..size).step_by(3) {
                map.swap_remove(&i);
            }

            b.iter(|| {
                let mut sum = 0;
                for &key in &access_keys {
                    if let Some(value) = map.get(&black_box(key)) {
                        sum += *value;
                    }
                }
                sum
            })
        });

        group.bench_with_input(BenchmarkId::new("hashlinked", size), &size, |b, &size| {
            let mut map = HashLinkedMap::default();
            for i in 0..size {
                map.insert(i, i * 2);
            }
            for i in (0..size).step_by(3) {
                map.remove(&i);
            }

            b.iter(|| {
                let mut sum = 0;
                for &key in &access_keys {
                    if let Some(value) = map.get(&black_box(key)) {
                        sum += *value;
                    }
                }
                sum
            })
        });
    }

    group.finish();
}

fn bench_iteration_full(c: &mut Criterion) {
    let mut group = c.benchmark_group("iteration_full");

    for &size in SIZES {
        group.throughput(criterion::Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("tether_map", size), &size, |b, &size| {
            let mut map = TetherMap::default();
            for i in 0..size {
                map.insert_tail(i, i * 2);
            }

            b.iter(|| {
                let mut sum = 0;
                for (key, value) in map.iter() {
                    sum += black_box(*key) + black_box(*value);
                }
                sum
            })
        });

        group.bench_with_input(BenchmarkId::new("indexmap", size), &size, |b, &size| {
            let mut map = IndexMap::default();
            for i in 0..size {
                map.insert(i, i * 2);
            }

            b.iter(|| {
                let mut sum = 0;
                for (key, value) in map.iter() {
                    sum += black_box(*key) + black_box(*value);
                }
                sum
            })
        });

        group.bench_with_input(BenchmarkId::new("hashlinked", size), &size, |b, &size| {
            let mut map = HashLinkedMap::default();
            for i in 0..size {
                map.insert(i, i * 2);
            }

            b.iter(|| {
                let mut sum = 0;
                for (key, value) in map.iter() {
                    sum += black_box(*key) + black_box(*value);
                }
                sum
            })
        });
    }

    group.finish();
}

fn bench_iteration_sparse(c: &mut Criterion) {
    let mut group = c.benchmark_group("iteration_sparse");

    for &size in SIZES {
        group.throughput(criterion::Throughput::Elements(size as u64 / 3));

        group.bench_with_input(BenchmarkId::new("tether_map", size), &size, |b, &size| {
            let mut map = TetherMap::default();
            for i in 0..size {
                map.insert_tail(i, i * 2);
            }

            for i in (0..size).step_by(3) {
                map.remove(&i);
            }

            b.iter(|| {
                let mut sum = 0;
                for (key, value) in map.iter() {
                    sum += black_box(*key) + black_box(*value);
                }
                sum
            })
        });

        group.bench_with_input(BenchmarkId::new("indexmap", size), &size, |b, &size| {
            let mut map = IndexMap::default();
            for i in 0..size {
                map.insert(i, i * 2);
            }
            for i in (0..size).step_by(3) {
                map.swap_remove(&i);
            }

            b.iter(|| {
                let mut sum = 0;
                for (key, value) in map.iter() {
                    sum += black_box(*key) + black_box(*value);
                }
                sum
            })
        });

        group.bench_with_input(BenchmarkId::new("hashlinked", size), &size, |b, &size| {
            let mut map = HashLinkedMap::default();
            for i in 0..size {
                map.insert(i, i * 2);
            }
            for i in (0..size).step_by(3) {
                map.remove(&i);
            }

            b.iter(|| {
                let mut sum = 0;
                for (key, value) in map.iter() {
                    sum += black_box(*key) + black_box(*value);
                }
                sum
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_insertion_at_end,
    bench_pop_from_end,
    bench_remove_from_middle,
    bench_insert_in_middle,
    bench_random_access_full,
    bench_random_access_sparse,
    bench_iteration_full,
    bench_iteration_sparse,
);
criterion_main!(benches);
