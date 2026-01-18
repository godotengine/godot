# GDScript Structs Implementation Roadmap

**Issue Reference:** [godotengine/godot-proposals#7329](https://github.com/godotengine/godot-proposals/issues/7329)  
**Branch:** `feature/gdscript-structs`  
**Problem Statement:** GDScript classes are memory-heavy (~16KB base) and non-contiguous, causing performance issues when managing 10,000+ entities (bullet hell games, simulations). Need lightweight struct types with optional contiguous memory allocation (FlatArray) for cache-friendly processing.

---

## 📋 Executive Summary

This implementation will add:
1. **Struct syntax in GDScript** - Lightweight data containers with type safety
2. **C++ API struct support** - Replace Dictionary exports with typed structs
3. **FlatArray optimization** - Contiguous memory allocation for high-performance scenarios

**Performance Goals:**
- Reduce memory footprint: 16KB class → ~100 bytes struct
- Enable cache-locality for 10K+ entity processing
- Target near-C performance when combined with future JIT/AOT

---

## 🎯 Phase 1: Core Type System & Parser Foundation
**Duration:** 3-4 weeks  
**Risk Level:** High (architectural changes to type system)

### 1.1 Parser & Tokenizer Extensions
**Files:** `modules/gdscript/gdscript_parser.cpp`, `gdscript_tokenizer.cpp`

#### Tasks:
- [ ] Add `TOKEN_STRUCT` keyword to tokenizer
- [ ] Implement struct declaration parsing in `_parse_class_body()`
- [ ] Add `StructNode` to AST representation (`gdscript_parser.h`)
- [ ] Support anonymous inline structs: `var data: struct { var x: int }`
- [ ] Validate struct member declarations (typed/untyped mixing)
- [ ] Add comprehensive error messages for invalid struct syntax

**Key Implementation Points:**
```gdscript
# Syntax to support:
struct Enemy:
    var position: Vector2
    var health: int
    var attacking: bool = false  # Default values

# Anonymous structs
var config: struct:
    var speed: float = 1.0
    var damage: int
```

**Testing:**
- Unit tests for all syntax variations
- Edge cases: nested structs, circular references
- Invalid syntax error handling

---

### 1.2 Type System Integration
**Files:** `core/variant/variant.h`, `variant.cpp`, `type_info.h`

#### Tasks:
- [ ] Add `Variant::STRUCT` type to enum
- [ ] Implement `StructInfo` metadata structure:
  ```cpp
  struct StructInfo {
      StringName struct_name;
      uint32_t member_count;
      StringName* member_names;
      ContainerTypeValidate* member_types;
      uint32_t memory_layout_hash;
  };
  ```
- [ ] Extend `ContainerTypeValidate` with struct support:
  ```cpp
  struct ContainerTypeValidate {
      Variant::Type type;
      StringName class_name;
      Ref<Script> script;
      LocalVector<ContainerTypeValidate> struct_members; // NEW
  };
  ```
- [ ] Implement struct comparison (structural equality vs nominal)
- [ ] Add struct variant construction/destruction paths

**Design Decisions:**
- **Structural vs Nominal Typing:** Use structural equality for layout compatibility
- **Memory Overhead:** Add ~8 bytes per struct for type metadata pointer
- **Null Handling:** Decide if structs can be null or always have default values

---

### 1.3 GDScript Analyzer Integration
**Files:** `gdscript_analyzer.cpp`, `gdscript_analyzer.h`

#### Tasks:
- [ ] Add struct resolution in `_resolve_datatype()`
- [ ] Implement member access validation
- [ ] Type checking for struct assignment
- [ ] Default value validation and initialization
- [ ] Detect cyclic struct dependencies
- [ ] Add struct-to-dict compatibility warnings

**Validation Rules:**
- All typed members must have compatible assignments
- Untyped members accept any value
- Struct layout must be deterministic (same order)
- No inheritance or methods allowed in structs

---

## 🔧 Phase 2: Runtime & Memory Management
**Duration:** 3-4 weeks  
**Risk Level:** Medium (performance-critical code)

### 2.1 Struct Storage Implementation
**Files:** `core/variant/array.cpp`, `array.h`

#### Tasks:
- [ ] Modify `ArrayPrivate` to support struct arrays:
  ```cpp
  class ArrayPrivate {
      SafeRefCount refcount;
      Vector<Variant> array;
      Variant *read_only;
      ContainerTypeValidate typed;
      
      // NEW: Struct support
      uint32_t struct_size = 0;
      StringName* struct_member_names = nullptr;
      bool is_struct_array = false;
      
      int32_t find_member_index(const StringName& p_member) const;
      bool validate_member(uint32_t p_index, const Variant& p_value);
  };
  ```
- [ ] Implement named indexing for structs:
  ```cpp
  Variant Array::get_named(const StringName& p_member);
  void Array::set_named(const StringName& p_member, const Variant& p_value);
  ```
- [ ] Add struct validation in `Array::set()` / `Array::get()`
- [ ] Optimize member lookup (consider hash map for large structs)

**Performance Targets:**
- Member access: < 50ns overhead vs direct array access
- Struct construction: < 1μs for 10-member struct
- Memory overhead: < 5% vs raw Variant array

---

### 2.2 Bytecode Generation & VM
**Files:** `gdscript_compiler.cpp`, `gdscript_vm.cpp`, `gdscript_byte_codegen.cpp`

#### Tasks:
- [ ] Add bytecode opcodes:
  - `OPCODE_CONSTRUCT_STRUCT`
  - `OPCODE_GET_STRUCT_MEMBER`
  - `OPCODE_SET_STRUCT_MEMBER`
- [ ] Compile struct member access to direct indexing
- [ ] Optimize struct copying (use copy-on-write where possible)
- [ ] Implement struct literal syntax compilation
- [ ] Cache member offsets in compiled code

**Optimization Strategies:**
- Use constant folding for known struct member names
- Inline member access for frequently accessed fields
- Avoid repeated type checking in loops

---

### 2.3 Editor & Debugger Support
**Files:** `gdscript_editor.cpp`, `editor/` modules

#### Tasks:
- [ ] Implement code completion for struct members
- [ ] Add struct inspection in debugger
- [ ] Show struct layout in tooltips
- [ ] Syntax highlighting for struct keyword
- [ ] Add struct templates/snippets
- [ ] Document hover shows member types

---

## 🚀 Phase 3: FlatArray High-Performance Implementation
**Duration:** 4-5 weeks  
**Risk Level:** High (complex memory management)

### 3.1 FlatArray Core Implementation
**Files:** `core/variant/array.cpp`, `core/templates/flat_array.h`

#### Tasks:
- [ ] Create `FlatArrayPrivate` structure:
  ```cpp
  class FlatArrayPrivate {
      SafeRefCount refcount;
      uint8_t* flat_memory;  // Contiguous memory block
      uint32_t element_count;
      uint32_t element_stride;
      StructInfo* struct_layout;
      Variant* materialized_view;  // For indexing operations
  };
  ```
- [ ] Implement contiguous memory allocation strategy
- [ ] Add resize with proper memory reallocation
- [ ] Implement efficient element access (return proxy objects)
- [ ] Handle copy-on-write semantics carefully
- [ ] Memory alignment for SIMD operations (16-byte alignment)

**Key Design Challenge: Variant Limitation**
- Variants are 24 bytes, larger types (Transform3D) allocated separately
- FlatArray cannot be truly flat for all types
- Solution: Pool allocator for oversized types, maintain index-based locality

#### Implementation Approach:
```cpp
// Hybrid storage model
struct FlatArrayElement {
    union {
        uint8_t inline_data[24];  // For small types
        void* heap_ptr;            // For large types (pool-allocated)
    };
    uint8_t type_tag;
};
```

---

### 3.2 Memory Pool Management
**Files:** `core/os/memory.cpp`, `core/templates/paged_allocator.h`

#### Tasks:
- [ ] Create `StructMemoryPool` for oversized types
- [ ] Implement block allocation with 64KB pages
- [ ] Add defragmentation support
- [ ] Pool recycling for common struct layouts
- [ ] Memory profiler integration

**Pool Strategy:**
- Pre-allocate pools for common struct patterns
- Use `PagedAllocator` for consistent performance
- Maintain free lists for each struct layout
- Monitor fragmentation, trigger compaction at 25%

---

### 3.3 Access Proxy & Iterator
**Files:** `core/variant/array.cpp`

#### Tasks:
- [ ] Implement `FlatArrayElement` proxy class:
  ```cpp
  class FlatArrayElement {
      FlatArrayPrivate* _array;
      uint32_t _index;
  public:
      Variant get_named(const StringName& p_member);
      void set_named(const StringName& p_member, const Variant& p_value);
      operator Variant() const;  // Materialization
  };
  ```
- [ ] Create efficient iterator for `for...in` loops
- [ ] Minimize materialization overhead
- [ ] Cache proxy objects per-thread

**Performance Goal:**
- FlatArray iteration: 10x faster than Array[Class] for 10K+ elements
- Member access: < 20ns overhead vs raw pointer dereference

---

### 3.4 JIT/AOT Preparation
**Files:** TBD (future JIT module)

#### Tasks:
- [ ] Design metadata format for JIT struct access
- [ ] Document struct memory layout guarantees
- [ ] Add intrinsic hints for JIT compiler
- [ ] Benchmark baseline for future comparison
- [ ] Create test suite for JIT validation

**Future Optimization Hooks:**
```cpp
// Metadata for JIT
struct StructLayoutInfo {
    uint32_t member_count;
    uint32_t* member_offsets;    // For direct memory access
    Variant::Type* member_types;  // For type specialization
};
```

---

## 🔌 Phase 4: C++ API Integration
**Duration:** 2-3 weeks  
**Risk Level:** Low (API sugar layer)

### 4.1 C++ Struct Macros
**Files:** `core/object/class_db.h`, new `core/struct_macros.h`

#### Tasks:
- [ ] Create `STRUCT_LAYOUT` macro:
  ```cpp
  STRUCT_LAYOUT(Object, PropertyInfoLayout,
      STRUCT_MEMBER("name", Variant::STRING, String()),
      STRUCT_MEMBER("type", Variant::INT),
      STRUCT_MEMBER("hint", Variant::INT, 0),
      STRUCT_MEMBER("hint_string", Variant::STRING),
      STRUCT_MEMBER("class_name", Variant::STRING, String())
  );
  ```
- [ ] Implement `TypedArray<Struct<T>>` template
- [ ] Add struct registration to `ClassDB`
- [ ] Generate documentation from struct definitions
- [ ] Create validation helpers

---

### 4.2 Core API Migration
**Files:** `core/object/object.h`, various API endpoints

#### Tasks:
- [ ] Replace `Array` with `TypedArray<Struct<PropertyInfo>>` in:
  - `Object::_get_property_list()`
  - `Script::get_script_property_list()`
- [ ] Update documentation generator
- [ ] Migrate example code
- [ ] Add deprecation warnings for Dictionary returns
- [ ] Ensure backward compatibility

**Migration Strategy:**
- Phase 1: Add new struct-based methods alongside old ones
- Phase 2: Mark old methods as deprecated (Godot 4.x)
- Phase 3: Remove old methods (Godot 5.0)

---

## 📚 Phase 5: Documentation & Migration Tools
**Duration:** 2 weeks  
**Risk Level:** Low

### 5.1 Documentation
**Files:** `doc/classes/`, `modules/gdscript/doc_classes/`

#### Tasks:
- [ ] Write comprehensive GDScript struct guide
- [ ] Document FlatArray usage and performance characteristics
- [ ] Create C++ struct binding tutorial
- [ ] Add performance comparison benchmarks
- [ ] Write migration guide (Dictionary → Struct)
- [ ] Update API reference for all changes

**Documentation Sections:**
1. **Introduction to Structs** - Why and when to use them
2. **Syntax Guide** - All struct declaration forms
3. **Performance Guide** - FlatArray optimization cases
4. **C++ Integration** - Binding structs from C++
5. **Migration Guide** - Converting existing code
6. **Best Practices** - Common patterns and anti-patterns

---

### 5.2 Examples & Demo Projects
**Files:** `doc/demos/structs/`

#### Tasks:
- [ ] Create bullet hell demo (10,000 bullets using FlatArray)
- [ ] Particle system example
- [ ] Game data organization example
- [ ] C++ API integration demo
- [ ] Performance benchmark project

---

### 5.3 Migration & Linting Tools
**Files:** `editor/plugins/`, `modules/gdscript/`

#### Tasks:
- [ ] Create script analyzer warnings:
  - "Consider using struct instead of Dictionary"
  - "Large class count detected, consider structs"
- [ ] Add auto-converter for simple Dictionary → Struct
- [ ] Create performance profiler annotations
- [ ] Add struct refactoring tools to editor

---

## ⚠️ Phase 6: Testing & Validation
**Duration:** 3 weeks (ongoing throughout)  
**Risk Level:** Critical (quality gate)

### 6.1 Unit Tests
**Files:** `modules/gdscript/tests/`, `tests/core/`

#### Test Coverage:
- [ ] Parser: All syntax variations
- [ ] Analyzer: Type checking edge cases
- [ ] Runtime: Struct operations (get/set/compare)
- [ ] Array: Struct array operations
- [ ] FlatArray: Memory management
- [ ] C++ API: Struct binding
- [ ] Performance: Regression tests

**Minimum Coverage Target:** 90% for struct-related code

---

### 6.2 Integration Tests
**Files:** `tests/scene/`, `tests/integration/`

#### Test Scenarios:
- [ ] Struct serialization/deserialization
- [ ] Network replication (multiplayer)
- [ ] Save/load with struct data
- [ ] Editor integration (scene files with structs)
- [ ] Hot reload with struct changes
- [ ] Cross-platform compatibility

---

### 6.3 Performance Benchmarks
**Files:** `tests/performance/benchmarks/`

#### Benchmark Suite:
- [ ] Memory usage: Class vs Struct vs Dictionary
- [ ] Access time: Class vs Struct vs FlatArray
- [ ] Iteration: Array[Class] vs FlatArray[Struct] (1K, 10K, 100K elements)
- [ ] Construction overhead
- [ ] Copy/assignment performance
- [ ] Cache miss analysis (perf/cachegrind)

**Performance Targets:**
| Operation | Class | Struct | FlatArray | Target |
|-----------|-------|--------|-----------|--------|
| Construction | 100ns | 20ns | 5ns | 20x faster |
| Memory/element | 16KB | 100B | 30B | 500x smaller |
| Iteration (10K) | 100ms | 80ms | 10ms | 10x faster |
| Member access | 50ns | 40ns | 20ns | 2x faster |

---

### 6.4 Stress Testing
**Files:** `tests/stress/`

#### Stress Test Scenarios:
- [ ] 100,000 struct allocations
- [ ] FlatArray with 1,000,000 elements
- [ ] Deeply nested struct hierarchies (10 levels)
- [ ] Struct arrays in multithreaded scenarios
- [ ] Memory leak detection (Valgrind/AddressSanitizer)
- [ ] Fuzzing struct parser

---

## 🚧 Known Risks & Mitigation

### Risk 1: Breaking Changes to Core Type System
**Impact:** High | **Probability:** Medium

**Mitigation:**
- Extensive backward compatibility testing
- Feature flag for gradual rollout
- Document all breaking changes in CHANGELOG
- Beta testing period with community

### Risk 2: Performance Regression in Non-Struct Code
**Impact:** High | **Probability:** Low

**Mitigation:**
- Comprehensive benchmarks before/after
- Profiling-guided optimization
- Keep struct code path separate from class path
- Performance regression CI checks

### Risk 3: FlatArray Memory Management Complexity
**Impact:** High | **Probability:** High

**Mitigation:**
- Start with simple implementation, optimize iteratively
- Extensive memory leak testing
- Use existing allocators (PagedAllocator)
- Consider using existing Vector<> infrastructure

### Risk 4: JIT/AOT Integration Delays
**Impact:** Medium | **Probability:** Low

**Mitigation:**
- Design for future JIT compatibility from start
- Document assumptions and requirements
- Provide metadata hooks for JIT developers
- FlatArray useful even without JIT

---

## 📅 Timeline Summary

| Phase | Duration | Dependencies | Completion Criteria |
|-------|----------|--------------|-------------------|
| Phase 1: Parser & Type System | 3-4 weeks | None | Struct syntax parses, analyzer validates |
| Phase 2: Runtime & VM | 3-4 weeks | Phase 1 | Structs work in scripts, tests pass |
| Phase 3: FlatArray | 4-5 weeks | Phase 2 | FlatArray benchmarks meet targets |
| Phase 4: C++ API | 2-3 weeks | Phase 2 | API endpoints migrated, documented |
| Phase 5: Documentation | 2 weeks | Phase 2, 3 | All docs complete, demos working |
| Phase 6: Testing | Ongoing | All | 90% coverage, benchmarks pass |

**Total Estimated Time:** 18-22 weeks (~5 months)

---

## 🎬 Getting Started

### Prerequisites
1. Set up Godot development environment
2. Understand GDScript parser architecture
3. Familiarize with Variant system
4. Review memory allocator implementations

### First Steps
1. Implement struct tokenizer keywords
2. Create basic StructNode AST representation
3. Add simple struct parsing test cases
4. Implement struct validation in analyzer

### Development Workflow
```bash
# Build with struct support
scons platform=linuxbsd target=editor dev_build=yes

# Run tests
scons platform=linuxbsd target=editor tests=yes
./bin/godot.linuxbsd.editor.dev.x86_64 --test --test-filter="*struct*"

# Benchmark
./bin/godot.linuxbsd.editor.dev.x86_64 --benchmark=struct_perf.tscn
```

---

## 🤝 Community Engagement

### Feedback Loops
- [ ] Post RFC in godotengine/godot-proposals
- [ ] Weekly progress updates on GitHub Discussions
- [ ] Demo preview at Godot contributor meeting
- [ ] Beta testing with community projects

### Support Channels
- GitHub Discussions: Design decisions
- Discord #gdscript-dev: Implementation questions
- GitHub Issues: Bug reports
- Forum: Documentation feedback

---

## 📊 Success Metrics

### Feature Completeness
- [ ] All struct syntax variants working
- [ ] 100% API migration (target endpoints)
- [ ] FlatArray performance targets met
- [ ] Documentation complete

### Performance
- [ ] 10x faster iteration for 10K+ FlatArray elements
- [ ] <5% memory overhead vs ideal C struct
- [ ] Zero performance regression in existing code

### Adoption
- [ ] 5+ demo projects using structs
- [ ] Positive community feedback (>80% approval)
- [ ] Core contributors comfortable with codebase changes

---

## 📝 Notes

### Architectural Decisions
1. **Structural Typing:** Enables flexible struct layout compatibility
2. **No Inheritance:** Keeps structs simple, avoids vtable overhead
3. **Hybrid FlatArray:** Balances performance with Variant flexibility
4. **Gradual Migration:** Minimizes disruption to existing projects

### Future Enhancements (Post-MVP)
- Struct methods (lightweight, no inheritance)
- Generic structs: `struct Pair<T, U>`
- Struct packing attributes: `@align(16)`
- SIMD-optimized operations for numeric structs
- Struct serialization format optimization

---

## 📖 References

- **Issue:** https://github.com/godotengine/godot-proposals/issues/7329
- **GDScript Architecture:** `modules/gdscript/README.md`
- **Variant System:** `core/variant/variant.h`
- **Memory Allocators:** `core/templates/paged_allocator.h`

---

**Document Version:** 1.0  
**Last Updated:** 2024  
**Author:** Generated for Godot GDScript Structs Implementation  
**Status:** 🔴 Planning Phase
