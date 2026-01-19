# FlatArray Implementation Plan

## Overview
FlatArray provides contiguous memory layout for arrays of structs, enabling cache-friendly iteration and 10x performance improvements for high-entity-count scenarios.

## Implementation Strategy

### Phase 1: Core FlatArray Class
**File:** `core/templates/flat_array.h`

```cpp
template <typename T>
class FlatArray {
private:
    T* _data;
    uint32_t _size;
    uint32_t _capacity;
    
public:
    // Contiguous memory allocation
    void resize(uint32_t p_size);
    void reserve(uint32_t p_capacity);
    
    // Fast element access
    T& operator[](uint32_t p_index);
    const T& operator[](uint32_t p_index) const;
    
    // Iterator support for range-based loops
    T* begin() { return _data; }
    T* end() { return _data + _size; }
};
```

### Phase 2: GDScript Integration
**File:** `modules/gdscript/gdscript_flat_array.cpp`

- Expose FlatArray to GDScript
- Add `Array.to_flat()` conversion
- Type-safe struct array handling

### Phase 3: Benchmarks
Create performance tests comparing:
- Regular Array vs FlatArray
- Iteration speed
- Memory layout efficiency
- Cache miss rates

## Estimated Impact
- **Iteration:** 10x faster for 10K+ elements
- **Memory:** 500x reduction (16KB class → 32 bytes struct)
- **Cache:** 90%+ cache hit rate vs 30% for scattered objects

## Timeline
- Core FlatArray: 2 weeks
- GDScript integration: 1 week  
- Testing & benchmarks: 1 week
- **Total: 4 weeks**
