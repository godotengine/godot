`eternal.hpp` is a header-only C++ implementation of `constexpr`/compile-time maps and hash maps. It provides an API that is somewhat compatible with `std::map`/`std::unordered_map`, but doesn't support insertion, or other modifications. It's main focus is in **binary size**: it generates minimal code and doesn't incur any static initialization overhead.

**Why is this useful?**

- Lookup tables

**Tested with these compilers/platforms:**
- *Linux GCC 4.9.4* (runtime only, since `constexpr` support is broken in this version)
- Linux GCC 5.5
- Linux GCC 6.5
- Linux GCC 7.3
- Linux GCC 8.1
- Linux Clang 3.9.1
- Linux Clang 4
- Linux Clang 5
- Linux Clang 6
- Linux Clang 7
- macOS Xcode 10.1
- Android NDK r17+

## Usage

```cpp
MAPBOX_ETERNAL_CONSTEXPR const auto colors = mapbox::eternal::map<mapbox::eternal::string, Color>({
    { "red", { 255, 0, 0, 1 } },
    { "green", { 0, 128, 0, 1 } },
    { "yellow", { 255, 255, 0, 1 } },
    { "white", { 255, 255, 255, 1 } },
    { "black", { 0, 0, 0, 1 } }
});
```

- `mapbox::eternal::map<key, value>()` is a factory function that produces a `constexpr` map from the `std::pair<key, value>`s passed to it.
- Alternatively, use `mapbox::eternal::hash_map<key, value>()` to construct a hash map. The `key` needs a specialization of `std::hash`.
- Use `mapbox::eternal::string` for `constexpr` strings.
- If you need to support GCC 4.9, use `MAPBOX_ETERNAL_CONSTEXPR` instead of `constexpr` in the variable definition.
- You can pass the elements in arbitrary order; they will be sorted at compile time (except for GCC 4.9). To speed up compilation, list the elements in sorted order. To determine the sort order for `hash_map`, iterate over the map and print the result. Note that hashes may be architecture-specific.
- Both `map()` and `hash_map()` support multiple values for the same key. To ensure that keys are unique, run `static_assert(map.unique());` (or equivalent for GCC 4.9)
- The preprocessor variable `MAPBOX_ETERNAL_IS_CONSTEXPR` is set to `1` if `constexpr` construction and lookups are supported.

## Performance

Uses a list of 148 unique CSS color names

```
Run on (8 X 2600 MHz CPU s)
CPU Caches:
  L1 Data 32K (x4)
  L1 Instruction 32K (x4)
  L2 Unified 262K (x4)
  L3 Unified 6291K (x1)
-----------------------------------------------------------------------------
Benchmark                                      Time           CPU Iterations
-----------------------------------------------------------------------------
EternalMap_ConstexprLookup                     2 ns          2 ns  424582090
EternalMap_Lookup                             48 ns         48 ns   14494194
EternalMap_LookupMissing                      35 ns         35 ns   20442492
EternalMap_LookupEqualRange                   78 ns         78 ns    8862891

EternalHashMap_ConstexprLookup                 2 ns          2 ns  429076688
EternalHashMap_Lookup                         30 ns         30 ns   22685952
EternalHashMap_LookupMissing                  17 ns         17 ns   39521898
EternalHashMap_LookupEqualRange               43 ns         43 ns   16696442

StdMap_Lookup                                 51 ns         50 ns   13580630
StdMap_LookupMissing                          58 ns         58 ns   11868028
StdMap_LookupEqualRange                       89 ns         89 ns    7766129

StdMultimap_Lookup                            50 ns         50 ns   14262312
StdMultimap_LookupMissing                     60 ns         59 ns   11555922
StdMultimap_LookupEqualRange                 103 ns        103 ns    6783077

StdUnorderedMap_Lookup                        43 ns         43 ns   16175696
StdUnorderedMap_LookupMissing                 50 ns         50 ns   10000000
StdUnorderedMap_LookupEqualRange              57 ns         57 ns   12256189

StdUnorderedMultimap_Lookup                   43 ns         43 ns   16011528
StdUnorderedMultimap_LookupMissing            52 ns         51 ns   10000000
StdUnorderedMultimap_LookupEqualRange         61 ns         60 ns   11519221
```
