# Godot Engine Glossary

Quick reference for Godot-specific types, macros, and terminology.

---

## Core Types

| Type | Header | Purpose |
|------|--------|---------|
| `Variant` | `core/variant/variant.h` | Universal value container. Holds any type (int, float, String, Object*, Array, etc.). Used for scripting bridge. |
| `String` | `core/string/ustring.h` | UTF-32 string. Use for text data. Heap allocated. |
| `StringName` | `core/string/string_name.h` | Interned string. O(1) comparison. Use for identifiers (method names, property names, signals). |
| `NodePath` | `core/string/node_path.h` | Path to a node in the scene tree (`"../Player/Sprite"`). Optimized for repeated lookups. |
| `RID` | `core/templates/rid.h` | 64-bit opaque handle to server-side resources. Lightweight (8 bytes). |
| `ObjectID` | `core/object/object_id.h` | 64-bit unique object identifier. Every Object gets one. Used for safe references. |
| `Ref<T>` | `core/object/ref_counted.h` | Smart pointer for RefCounted objects. Automatic reference counting. |
| `real_t` | `core/math/math_defs.h` | Typedef: `double` (default) or `float` (if `REAL_T_IS_DOUBLE` is off). Platform-dependent precision. |
| `Vector<T>` | `core/templates/vector.h` | Dynamic array with copy-on-write. Main container type. |
| `LocalVector<T>` | `core/templates/local_vector.h` | Dynamic array without COW. Faster for local/temporary use. |
| `List<T>` | `core/templates/list.h` | Doubly-linked list. |
| `HashMap<K,V>` | `core/templates/hash_map.h` | Hash table. Use instead of `std::unordered_map`. |
| `HashSet<T>` | `core/templates/hash_set.h` | Hash set. Use instead of `std::unordered_set`. |
| `Callable` | `core/variant/callable.h` | Generic callable wrapper. Can hold method pointers, lambdas, GDScript functions. |
| `Mutex` | `core/os/mutex.h` | Recursive mutex. Use with `MutexLock` (RAII). |
| `BinaryMutex` | `core/os/mutex.h` | Non-recursive mutex. Faster but can't re-lock from same thread. |
| `RWLock` | `core/os/rw_lock.h` | Read-write lock. Use `RWLockRead`/`RWLockWrite` for RAII. |
| `SafeRefCount` | `core/templates/safe_refcount.h` | Atomic reference counter using acquire-release semantics. |

## Key Macros

### Class System
| Macro | Purpose |
|-------|---------|
| `GDCLASS(MyClass, Parent)` | Register class with ClassDB. Required in every exposed class. |
| `GDREGISTER_CLASS(T)` | Register instantiable class at startup (in `register_types.cpp`). |
| `GDREGISTER_ABSTRACT_CLASS(T)` | Register non-instantiable base class. |
| `GDREGISTER_VIRTUAL_CLASS(T)` | Register class with virtual methods. |
| `GDREGISTER_INTERNAL_CLASS(T)` | Register class not exposed to scripts. |

### Method Binding
| Macro | Purpose |
|-------|---------|
| `D_METHOD("name", "arg1", "arg2")` | Create method definition with argument names (debug builds only). |
| `DEFVAL(value)` | Default value for optional parameter. |
| `ClassDB::bind_method(D_METHOD(...), &Class::method)` | Bind instance method. |
| `ClassDB::bind_static_method("Class", D_METHOD(...), &Class::method)` | Bind static method. |
| `ClassDB::bind_vararg_method(flags, "name", &Class::method, mi)` | Bind variadic method. |

### Property Binding
| Macro | Purpose |
|-------|---------|
| `ADD_PROPERTY(PropertyInfo(...), "setter", "getter")` | Register property with inspector. |
| `ADD_PROPERTYI(PropertyInfo(...), "setter", "getter", INDEX)` | Register indexed property. |
| `ADD_PROPERTY_DEFAULT("name", value)` | Set default value for property. |
| `ADD_GROUP("Name", "prefix_")` | Start a property group in inspector. |
| `ADD_SUBGROUP("Name", "prefix_")` | Start a property subgroup. |
| `ADD_SIGNAL(MethodInfo("name"))` | Register a signal. |
| `BIND_CONSTANT(CONST_NAME)` | Bind a simple constant. |
| `BIND_ENUM_CONSTANT(ENUM_VALUE)` | Bind an enum value. |
| `BIND_BITFIELD_FLAG(FLAG_NAME)` | Bind a bitfield flag. |
| `VARIANT_ENUM_CAST(Class::Enum)` | Enable enum use in Variant system. Must be outside class. |

### Virtual Methods
| Macro | Purpose |
|-------|---------|
| `GDVIRTUAL0(_method)` | Declare virtual, 0 args, void return. |
| `GDVIRTUAL1(_method, Type)` | Declare virtual, 1 arg, void return. |
| `GDVIRTUAL0R(RetType, _method)` | Declare virtual, 0 args, with return. |
| `GDVIRTUAL0RC(RetType, _method)` | Declare virtual, 0 args, const, with return. |
| `GDVIRTUAL*_REQUIRED` | Must be overridden by subclasses. |
| `GDVIRTUAL_BIND(_method, "arg1")` | Bind virtual in `_bind_methods()`. |
| `GDVIRTUAL_CALL(_method, args...)` | Call virtual method. Returns `bool` (true if overridden). |
| `GDVIRTUAL_IS_OVERRIDDEN(_method)` | Check if virtual is overridden without calling. |

### Error Handling
| Macro | Purpose |
|-------|---------|
| `ERR_FAIL_COND(cond)` | Return void if condition is true. |
| `ERR_FAIL_COND_V(cond, ret)` | Return value if condition is true. |
| `ERR_FAIL_COND_MSG(cond, msg)` | Return void with message. |
| `ERR_FAIL_COND_V_MSG(cond, ret, msg)` | Return value with message. |
| `ERR_FAIL_NULL(ptr)` | Return void if null. |
| `ERR_FAIL_NULL_V(ptr, ret)` | Return value if null. |
| `ERR_FAIL_INDEX(idx, size)` | Return void if out of bounds. |
| `ERR_FAIL_INDEX_V(idx, size, ret)` | Return value if out of bounds. |
| `ERR_CONTINUE(cond)` | `continue` in loop if condition true. |
| `ERR_BREAK(cond)` | `break` in loop if condition true. |
| `ERR_PRINT(msg)` | Print error, don't return. |
| `WARN_PRINT(msg)` | Print warning, don't return. |
| `CRASH_NOW_MSG(msg)` | Fatal crash with message. |
| `DEV_ASSERT(cond)` | Assert in dev builds only. |

### String Helpers
| Macro | Purpose |
|-------|---------|
| `SNAME("string")` | Create cached `StringName`. Use for repeated lookups. |
| `SceneStringName(name)` | Access pre-cached scene StringName (e.g., `SceneStringName(toggled)`). |
| `vformat("fmt %s %d", str, num)` | Printf-style String formatting. |

### Inlining
| Macro | Purpose |
|-------|---------|
| `_ALWAYS_INLINE_` | Force inline on all builds. |
| `_FORCE_INLINE_` | Force inline except in DEV_ENABLED/SIZE_EXTRA builds. |
| `_NO_INLINE_` | Prevent inlining. |

### Thread Safety
| Macro | Purpose |
|-------|---------|
| `ERR_THREAD_GUARD` | Fail if caller thread can't access this node. |
| `ERR_THREAD_GUARD_V(ret)` | Same, with return value. |
| `ERR_MAIN_THREAD_GUARD` | Fail if not on main thread. |
| `ERR_READ_THREAD_GUARD` | Fail if not on main or group thread. |

### Build Conditionals
| Macro | Purpose |
|-------|---------|
| `TOOLS_ENABLED` | Editor build (not export templates). |
| `DEBUG_ENABLED` | Debug build. |
| `DEV_ENABLED` | Developer build (extra checks). |
| `DISABLE_DEPRECATED` | Strip deprecated compatibility code. |

## Notification Constants

Most commonly used (from `Node`):
```
NOTIFICATION_ENTER_TREE           = 10
NOTIFICATION_EXIT_TREE            = 11
NOTIFICATION_READY                = 13
NOTIFICATION_PAUSED               = 14
NOTIFICATION_UNPAUSED             = 15
NOTIFICATION_PHYSICS_PROCESS      = 16
NOTIFICATION_PROCESS              = 17
NOTIFICATION_PARENTED             = 18
NOTIFICATION_UNPARENTED           = 19
NOTIFICATION_INTERNAL_PROCESS     = 25
NOTIFICATION_INTERNAL_PHYSICS_PROCESS = 26
NOTIFICATION_POST_ENTER_TREE      = 27
```

From `Node3D`:
```
NOTIFICATION_TRANSFORM_CHANGED    = 2000
NOTIFICATION_ENTER_WORLD          = 41
NOTIFICATION_EXIT_WORLD           = 42
NOTIFICATION_VISIBILITY_CHANGED   = 43
```

From `CanvasItem`:
```
NOTIFICATION_DRAW                 = 30
NOTIFICATION_VISIBILITY_CHANGED   = 31
NOTIFICATION_ENTER_CANVAS         = 32
NOTIFICATION_EXIT_CANVAS          = 33
```

## PropertyHint Values

```
PROPERTY_HINT_NONE                  No hint
PROPERTY_HINT_RANGE                 "min,max[,step][,or_greater][,or_less][,suffix:unit]"
PROPERTY_HINT_ENUM                  "Option1,Option2,Option3"
PROPERTY_HINT_FLAGS                 "Flag1:1,Flag2:2,Flag3:4"
PROPERTY_HINT_FILE                  "*.ext1,*.ext2"
PROPERTY_HINT_DIR                   Directory picker
PROPERTY_HINT_RESOURCE_TYPE         "ClassName"
PROPERTY_HINT_MULTILINE_TEXT        Multi-line text editor
PROPERTY_HINT_PLACEHOLDER_TEXT      Placeholder in text fields
PROPERTY_HINT_NODE_PATH_VALID_TYPES "NodeType1,NodeType2"
PROPERTY_HINT_NODE_TYPE             "NodeType" (C# [Export] Node-typed fields)
PROPERTY_HINT_INTERFACE_TYPE        "InterfaceName" (TECH DEBT — see mono-module.md §6)
PROPERTY_HINT_LINK                  Linked values (e.g., scale x/y/z)
PROPERTY_HINT_HIDE_QUATERNION_EDIT  Hide quaternion editor
PROPERTY_HINT_PASSWORD              Password field
PROPERTY_HINT_DICTIONARY_TYPE       Key/value type hints for typed dictionaries
PROPERTY_HINT_TOOL_BUTTON           Inspector button
```

## Variant Types

The `Variant::Type` enum (used in PropertyInfo):
```
NIL, BOOL, INT, FLOAT,
STRING, VECTOR2, VECTOR2I, RECT2, RECT2I,
VECTOR3, VECTOR3I, TRANSFORM2D, VECTOR4, VECTOR4I,
PLANE, QUATERNION, AABB, BASIS, TRANSFORM3D, PROJECTION,
COLOR, STRING_NAME, NODE_PATH, RID,
OBJECT, CALLABLE, SIGNAL, DICTIONARY, ARRAY,
PACKED_BYTE_ARRAY, PACKED_INT32_ARRAY, PACKED_INT64_ARRAY,
PACKED_FLOAT32_ARRAY, PACKED_FLOAT64_ARRAY,
PACKED_STRING_ARRAY, PACKED_VECTOR2_ARRAY, PACKED_VECTOR3_ARRAY,
PACKED_COLOR_ARRAY, PACKED_VECTOR4_ARRAY
```

## Server Shorthands

Common aliases used in code:
```cpp
RS   = RenderingServer
DS   = DisplayServer
TS   = TextServer
PS3D = PhysicsServer3D  (less common)
```

Usage: `RS::get_singleton()->method(...)` instead of `RenderingServer::get_singleton()->method(...)`.
