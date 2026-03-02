# Godot Engine Anti-Patterns

Things to **never** do when writing code for Godot Engine.

---

## Memory Management

### Never use `new` / `delete`
```cpp
// WRONG
Node *n = new Node();
delete n;

// RIGHT
Node *n = memnew(Node);
memdelete(n);
// Or for scene tree nodes:
n->queue_free();
```
Godot's allocator tracks memory, reports leaks, and hooks into the object system. Raw `new`/`delete` bypasses all of this.

### Never use STL containers
```cpp
// WRONG
#include <vector>
#include <map>
#include <string>
std::vector<int> items;
std::string name;

// RIGHT
#include "core/templates/vector.h"
#include "core/templates/hash_map.h"
#include "core/string/ustring.h"
Vector<int> items;
String name;
```
Godot containers have COW semantics, integrate with Variant, and use the engine allocator.

### Never mix Ref<T> with raw pointers for RefCounted objects
```cpp
// WRONG — will cause double-free or memory leak
RefCounted *rc = memnew(MyRefCounted);
Ref<MyRefCounted> ref = rc;  // Ambiguous ownership

// RIGHT
Ref<MyRefCounted> ref;
ref.instantiate();
// Or:
Ref<MyRefCounted> ref = memnew(MyRefCounted);
```

### Never `memdelete` a node that's in the scene tree
```cpp
// WRONG — can crash, children not properly cleaned up
memdelete(node_in_tree);

// RIGHT
node_in_tree->queue_free();  // Deferred, thread-safe
```

---

## Error Handling

### Never use bare `if/return` for validation
```cpp
// WRONG
if (p_index < 0 || p_index >= size) {
    return;
}

// RIGHT
ERR_FAIL_INDEX(p_index, size);
// Or with message:
ERR_FAIL_INDEX_MSG(p_index, size, "Index out of bounds.");
```
The `ERR_FAIL_*` macros log errors with file/line/function info automatically.

### Never use `assert()` or `static_assert` for runtime checks
```cpp
// WRONG
assert(ptr != nullptr);

// RIGHT
ERR_FAIL_NULL(ptr);
// Or for debug-only:
DEV_ASSERT(ptr != nullptr);
```

### Never swallow errors silently
```cpp
// WRONG
if (result != OK) {
    // ignore
}

// RIGHT
ERR_FAIL_COND_V_MSG(result != OK, ERR_CANT_CREATE,
    vformat("Failed to create resource: %s", error_string));
```

---

## Class Registration

### Never forget `_bind_methods` for a GDCLASS
Every class with `GDCLASS(MyClass, Parent)` **must** have a `static void _bind_methods()` even if empty. The macro references it.

### Never forget `VARIANT_ENUM_CAST` for enums used in bindings
```cpp
// WRONG — enum won't work in Variant/GDScript
class MyClass : public Node {
    GDCLASS(MyClass, Node);
public:
    enum MyEnum { A, B, C };
    // ... BIND_ENUM_CONSTANT in _bind_methods ...
};
// Missing VARIANT_ENUM_CAST!

// RIGHT
class MyClass : public Node { /* same */ };
VARIANT_ENUM_CAST(MyClass::MyEnum);  // Outside the class
```

### Never mismatch D_METHOD argument names with actual parameters
```cpp
// WRONG — argument name doesn't match parameter
ClassDB::bind_method(D_METHOD("set_value", "val"), &MyClass::set_value);
// If the actual parameter is p_value, the D_METHOD name should match docs/script usage

// RIGHT — name matches what scripts see
ClassDB::bind_method(D_METHOD("set_value", "value"), &MyClass::set_value);
```

### Never skip property binding for get/set pairs
```cpp
// WRONG — methods bound but not accessible as property in inspector
ClassDB::bind_method(D_METHOD("set_speed", "speed"), &MyClass::set_speed);
ClassDB::bind_method(D_METHOD("get_speed"), &MyClass::get_speed);
// Missing ADD_PROPERTY!

// RIGHT
ClassDB::bind_method(D_METHOD("set_speed", "speed"), &MyClass::set_speed);
ClassDB::bind_method(D_METHOD("get_speed"), &MyClass::get_speed);
ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "speed"), "set_speed", "get_speed");
```

---

## Formatting and Style

### Never use spaces for C++ indentation
Godot uses **tabs**. Enforced by `.editorconfig` and `clang-format`.

### Never use `#ifndef`/`#define` header guards
```cpp
// WRONG (legacy)
#ifndef MY_CLASS_H
#define MY_CLASS_H
// ...
#endif

// RIGHT
#pragma once
```
The `misc/scripts/header_guards.py` script enforces this.

### Never use `printf` / `std::cout`
```cpp
// WRONG
printf("Debug: %d\n", value);
std::cout << "Hello" << std::endl;

// RIGHT
print_line(vformat("Debug: %d", value));
// Or for errors:
ERR_PRINT(vformat("Something went wrong: %s", msg));
```

### Never add includes in headers when forward declarations work
```cpp
// WRONG — in header
#include "scene/3d/mesh_instance_3d.h"  // Only using pointer
class MyNode : public Node3D {
    MeshInstance3D *mesh_inst = nullptr;
};

// RIGHT — in header
class MeshInstance3D;  // Forward declaration
class MyNode : public Node3D {
    MeshInstance3D *mesh_inst = nullptr;
};
// Full include goes in the .cpp file
```

---

## Threading

### Never use manual lock/unlock
```cpp
// WRONG — prone to deadlocks if exception or early return
mutex.lock();
// ... code that might return early ...
mutex.unlock();

// RIGHT — RAII, automatically unlocks
MutexLock lock(mutex);
// ... code ...
```

### Never modify scene tree from non-main thread
```cpp
// WRONG
void MyWorkerThread::run() {
    node->add_child(new_node);  // Crash or undefined behavior
}

// RIGHT
void MyWorkerThread::run() {
    callable_mp(node, &Node::add_child).call_deferred(new_node, false, 0);
}
```

---

## Code Generation

### Never hand-edit `.gen.h` or `.gen.cpp` files
These are regenerated on every build. Edit the Python generator scripts instead:
- `gdvirtual.gen.h` -> edit `core/object/make_virtuals.py`
- `*.glsl.gen.h` -> edit the `.glsl` source file
- `register_module_types.gen.cpp` -> auto-generated from module `config.py`

---

## Design Patterns

### Never call virtual methods in constructors/destructors
```cpp
// WRONG — undefined behavior in C++, will call base version
MyNode::MyNode() {
    set_process(true);  // This is OK (not virtual)
    _ready();           // WRONG — virtual, may not resolve to override
}
```

### Never store raw pointers to RefCounted objects long-term
```cpp
// WRONG — object may be freed when all Refs drop
Resource *res = my_ref.ptr();
// ... later use of res may be use-after-free

// RIGHT
Ref<Resource> res = my_ref;  // Holds a reference
```

### Never use `Object::cast_to` without null check
```cpp
// WRONG
Node3D *n3d = Object::cast_to<Node3D>(obj);
n3d->set_position(Vector3());  // Crash if cast fails

// RIGHT
Node3D *n3d = Object::cast_to<Node3D>(obj);
ERR_FAIL_NULL(n3d);
n3d->set_position(Vector3());
```

---

## Compatibility

### Never change a public method signature without a `.compat.inc`
If you change parameters of a method that's already bound to scripting, old saved scenes and scripts will break. Always add backward-compatible wrapper in `.compat.inc`.

### Never remove a public API without deprecation
Mark deprecated with `WARN_DEPRECATED` or `WARN_DEPRECATED_MSG`, provide the `.compat.inc` wrapper, and reference the PR number in the compat function name suffix.
