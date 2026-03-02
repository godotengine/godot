# Godot Engine Coding Conventions

Severity levels: **[MUST]** = enforced by CI/tooling, **[SHOULD]** = strong convention, **[PREFER]** = stylistic preference.

---

## 1. File Format

**[MUST]** UTF-8 encoding, no BOM (except `.csproj`/`.sln` which require BOM).
**[MUST]** LF line endings (except `.csproj`/`.sln`/`.bat` which use CRLF).
**[MUST]** No trailing whitespace.
**[MUST]** Files end with a single newline.
**[MUST]** Tabs for C++ indentation (4-space width). Spaces for Python/YAML.

Enforced by: `misc/scripts/file_format.py`, `.editorconfig`.

## 2. Copyright Header

**[MUST]** Every `.h` and `.cpp` file starts with the Godot copyright block:

```cpp
/**************************************************************************/
/*  my_file.h                                                             */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* ... (MIT license text) ...                                             */
/**************************************************************************/
```

Enforced by: `misc/scripts/copyright_headers.py`.

## 3. Header Guards

**[MUST]** Use `#pragma once` immediately after the copyright block (line ~31).

```cpp
/**************************************************************************/
/* ... copyright block ... */
/**************************************************************************/

#pragma once

#include "core/object/object.h"
```

Legacy `#ifndef`/`#define` guards are automatically converted by `misc/scripts/header_guards.py`.

## 4. Include Ordering

**[SHOULD]** Follow this exact order in `.cpp` files:

```cpp
#include "own_header.h"                       // 1. Own header (always first)
#include "own_header.compat.inc"              // 2. Compat include (if exists)

STATIC_ASSERT_INCOMPLETE_TYPE(class, Mesh);   // 3. Forward decl assertions (if any)

#include "core/config/project_settings.h"     // 4. core/ includes (alphabetical)
#include "core/io/resource_loader.h"

#include "scene/main/viewport.h"              // 5. scene/server includes (alphabetical)
#include "servers/rendering/rendering_server.h"

#ifdef TOOLS_ENABLED                          // 6. Conditional includes (last)
#include "editor/editor_node.h"
#endif
```

**[SHOULD]** No blank lines between include groups (continuous block, alphabetized).
**[SHOULD]** Use forward declarations in headers when only pointers/references are needed. Include the full header only in the `.cpp` file.

## 5. Naming Conventions

### Classes **[MUST]**
- PascalCase: `Node`, `MeshInstance3D`, `PhysicsServer3D`
- Abbreviations stay uppercase: `RID`, `AABB`, `RPC`
- 2D/3D suffix on spatial classes: `Node3D`, `Camera2D`, `PhysicsBody3D`

### Files **[MUST]**
- `snake_case.h` / `snake_case.cpp`
- Match class name: `MeshInstance3D` -> `mesh_instance_3d.h`

### Methods **[MUST]**
- `snake_case`: `get_transform()`, `set_position()`, `add_child()`
- Getters: `get_x()`, boolean getters: `is_x()` or `has_x()`
- Setters: `set_x()`
- Private/internal: `_` prefix: `_update_process()`, `_validate_child_name()`

### Parameters **[MUST]**
- `p_` prefix for all parameters: `p_name`, `p_child`, `p_enabled`
- `r_` prefix for return-by-reference: `r_error`, `r_vertex_array`
- No other prefixes (no `m_`, `g_`, `s_`)

### Member Variables **[SHOULD]**
- No prefix (unlike many C++ codebases). Plain `snake_case`.
- Complex classes use a `Data data;` struct to group members:
  ```cpp
  struct Data {
      Transform3D global_transform;
      bool top_level = false;
  } data;
  ```

### Constants and Enums **[MUST]**
- `SCREAMING_SNAKE_CASE`: `NOTIFICATION_READY`, `PROCESS_MODE_INHERIT`
- Enum type names: PascalCase: `ProcessMode`, `RotationEditMode`

## 6. Const Correctness

**[MUST]** All getter methods are `const`:
```cpp
String get_text() const;
bool is_visible() const;
Transform3D get_global_transform() const;
```

**[SHOULD]** Pass objects by `const` reference:
```cpp
void set_text(const String &p_text);
void set_mesh(const Ref<Mesh> &p_mesh);
```

**[SHOULD]** Use `mutable` for cached values that are logically const:
```cpp
mutable Transform3D global_transform;
mutable Vector3 euler_rotation;
```

## 7. Error Handling

### Decision Tree

```
Need to validate an index?
  -> ERR_FAIL_INDEX[_V][_MSG](idx, size[, retval][, "msg"])

Need to validate a pointer is not null?
  -> ERR_FAIL_NULL[_V][_MSG](ptr[, retval][, "msg"])

Need to validate a general condition?
  -> ERR_FAIL_COND[_V][_MSG](cond[, retval][, "msg"])

Inside a loop and want to skip/break?
  -> ERR_CONTINUE[_MSG](cond[, "msg"])
  -> ERR_BREAK[_MSG](cond[, "msg"])

Just want to print without returning?
  -> ERR_PRINT("msg")    (error)
  -> WARN_PRINT("msg")   (warning)

Unrecoverable crash?
  -> CRASH_NOW_MSG("msg")     (unconditional)
  -> CRASH_COND_MSG(cond, "msg")

Debug-only assertion?
  -> DEV_ASSERT(cond)         (stripped in release)
```

### Macro Suffix Reference

| Suffix | Meaning |
|--------|---------|
| (none) | Returns `void` |
| `_V` | Returns a value: `ERR_FAIL_COND_V(cond, retval)` |
| `_MSG` | Includes message: `ERR_FAIL_COND_MSG(cond, "msg")` |
| `_V_MSG` | Both: `ERR_FAIL_COND_V_MSG(cond, retval, "msg")` |
| `_EDMSG` | Message shown in editor output panel |
| `_ONCE` | Only prints once per lifetime (for `ERR_PRINT_ONCE`, `WARN_PRINT_ONCE`) |

**[SHOULD]** Prefer `_MSG` variants — they provide context in error logs.
**[SHOULD]** Use `vformat()` for dynamic error messages:
```cpp
ERR_FAIL_COND_MSG(p_index >= size,
    vformat("Index %d out of range (size: %d).", p_index, size));
```

## 8. Memory Management

### Ref<T> for RefCounted Objects **[MUST]**
```cpp
Ref<Mesh> mesh;                           // Declaration
mesh.instantiate();                       // Create new instance
Ref<FileAccess> f = FileAccess::open(...); // Factory pattern

// Passing:
void set_mesh(const Ref<Mesh> &p_mesh);   // const ref for setters
Ref<Mesh> get_mesh() const;               // by value for getters
```

### memnew/memdelete for Manual Objects **[MUST]**
```cpp
Node *child = memnew(Node);               // Allocate
memdelete(child);                          // Free (only if not in tree)
child->queue_free();                       // Deferred free (for nodes in tree)
```

### Node Ownership
- `add_child(node)` transfers ownership to parent
- `remove_child(node)` releases ownership (does NOT free)
- Parent destructor frees all children automatically
- Use `queue_free()` for scene tree nodes (thread-safe, deferred)
- Direct `memdelete()` only for nodes NOT in the scene tree

### Containers **[MUST]**
```cpp
Vector<int> arr;                // Dynamic array (COW)
LocalVector<int> local;         // Non-COW, stack-friendly
List<String> linked;            // Doubly-linked list
HashMap<String, int> map;       // Hash map
HashSet<StringName> set;        // Hash set
```

**[MUST]** Never use STL containers (`std::vector`, `std::map`, `std::string`, etc.).

## 9. Threading

**[MUST]** Use RAII locks, never manual `lock()`/`unlock()`:
```cpp
MutexLock lock(my_mutex);                 // Recursive mutex (general use)
RWLockRead read_lock(my_rwlock);          // Read lock
RWLockWrite write_lock(my_rwlock);        // Write lock
```

**[MUST]** Node scene tree operations from main thread only:
```cpp
// From another thread, defer to main:
callable_mp(this, &MyNode::do_work).call_deferred();
```

**[SHOULD]** Add thread guards to node methods that require main thread:
```cpp
void MyNode::set_property(int p_value) {
    ERR_THREAD_GUARD
    // ... implementation
}
```

## 10. Property Binding

### PropertyInfo Construction
```cpp
PropertyInfo(Variant::TYPE, "name")                                    // Minimal
PropertyInfo(Variant::TYPE, "name", PROPERTY_HINT_X, "hint_string")    // With hint
PropertyInfo(Variant::TYPE, "name", PROPERTY_HINT_X, "hint_string",
    PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_INTERNAL)                  // With usage flags
```

### Common Hint Strings
```
PROPERTY_HINT_RANGE:    "0,100,1"  "0,1,0.01,or_greater,or_less"  "-360,360,0.1,radians_as_degrees"
PROPERTY_HINT_ENUM:     "Option A,Option B,Option C"
PROPERTY_HINT_FLAGS:    "Flag A:1,Flag B:2,Flag C:4"
PROPERTY_HINT_FILE:     "*.png,*.jpg"
PROPERTY_HINT_RESOURCE_TYPE:  "Mesh"  "Texture2D"  "Material"
PROPERTY_HINT_NODE_PATH_VALID_TYPES:  "Skeleton3D"
```

### Property Organization
```cpp
ADD_GROUP("Group Name", "prefix_");           // Inspector group
ADD_SUBGROUP("Subgroup", "sub_prefix_");      // Nested subgroup
ADD_PROPERTY(PropertyInfo(...), "set_x", "get_x");
ADD_PROPERTYI(PropertyInfo(...), "set_x", "get_x", INDEX);  // Indexed
ADD_PROPERTY_DEFAULT("property_name", default_value);
```

## 11. Signal Declaration

```cpp
// No arguments:
ADD_SIGNAL(MethodInfo("changed"));

// With arguments:
ADD_SIGNAL(MethodInfo("value_changed",
    PropertyInfo(Variant::FLOAT, "new_value")));

// With typed Object argument:
ADD_SIGNAL(MethodInfo("child_entered_tree",
    PropertyInfo(Variant::OBJECT, "node", PROPERTY_HINT_NONE, "",
        PROPERTY_USAGE_DEFAULT, "Node")));
```

## 12. Virtual Methods (GDVIRTUAL)

### Declaration (in header)
```
GDVIRTUAL[N]          -> void, N args
GDVIRTUAL[N]R         -> with return value
GDVIRTUAL[N]C         -> const
GDVIRTUAL[N]RC        -> const with return
GDVIRTUAL[N]_REQUIRED -> must be overridden
```

```cpp
GDVIRTUAL0(_ready)                                    // void, no args
GDVIRTUAL1(_process, double)                          // void, 1 arg
GDVIRTUAL0RC(Vector<String>, _get_warnings)           // const, returns Vector<String>
GDVIRTUAL1RC_REQUIRED(Ref<Material>, _get_mat, int)   // const, return, required
```

### Binding (in _bind_methods)
```cpp
GDVIRTUAL_BIND(_ready);
GDVIRTUAL_BIND(_process, "delta");
GDVIRTUAL_BIND(_get_warnings);
```

### Calling
```cpp
GDVIRTUAL_CALL(_process, delta);                      // Fire and forget

Vector<String> result;
if (GDVIRTUAL_CALL(_get_warnings, result)) {          // With return value
    // result is populated
}
```

## 13. Compatibility Methods

When changing a method signature, preserve the old one:

**Create `my_class.compat.inc`:**
```cpp
#ifndef DISABLE_DEPRECATED

void MyClass::_old_method_bind_compat_12345(int p_arg) {
    new_method(p_arg, default_value);  // Adapt to new signature
}

void MyClass::_bind_compatibility_methods() {
    ClassDB::bind_compatibility_method(D_METHOD("old_method", "arg"),
        &MyClass::_old_method_bind_compat_12345);
}

#endif // DISABLE_DEPRECATED
```

**Include in `.cpp`:**
```cpp
#include "my_class.h"
#include "my_class.compat.inc"
```

The number suffix (e.g., `_12345`) is the PR/commit hash that introduced the change.

## 14. Commit Messages

**[MUST]** Imperative form, capitalize first letter, under 72 characters:
```
Add C# iOS support
Fix GLES3 instanced rendering color defaults
Core: Fix Object::has_method() for script static methods
```

**[MUST]** Include unit tests for bugs and new features in the same commit.
**[MUST]** Update `doc/classes/` XML when adding/changing API.
**[SHOULD]** One topic per PR. Multiple small PRs over one large PR.
