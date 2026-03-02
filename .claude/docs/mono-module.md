# Mono Module (C# Integration)

Architecture of `modules/mono/` — how C# scripts integrate with the Godot engine.

---

## 1. Module Structure

```
modules/mono/
  csharp_script.h/cpp          CSharpScript + CSharpInstance (Script/ScriptInstance impl)
  mono_gd/
    gd_mono_cache.h/cpp        Managed callback registry, interop structs
  glue/
    GodotSharp/GodotSharp/
      Core/
        GodotObject.base.cs    Base class for all C# Godot objects
        Attributes/             [Export], [Tool], [GlobalClass], etc.
        NativeInterop/
          VariantUtils.cs       Variant ↔ C# type conversion
          Marshaling.cs         Type mapping (C# Type → Variant.Type)
          InteropStructs.cs     godot_variant, godot_string, etc.
          InteropUtils.cs       Managed ↔ unmanaged object binding
      Bridge/
        ScriptManagerBridge.cs  Property list/default value gathering
        CSharpInstanceBridge.cs Property get/set forwarding
        ManagedCallbacks.cs     Function pointer table for C++ → C# calls
  editor/
    bindings_generator.h/cpp   Generates GodotSharp API from ClassDB
    Godot.NET.Sdk/
      Godot.SourceGenerators/   Roslyn source generators
        ScriptPropertiesGenerator.cs    [Export] → PropertyInfo code gen
        ScriptPropertyDefValGenerator.cs Default value code gen
        MarshalUtils.cs                 C# type → MarshalType mapping
        GodotEnums.cs                   C# mirror of PropertyHint enum
```

---

## 2. Three Pipelines

The mono module has three distinct code generation/bridging pipelines:

### Pipeline A: Binding Generator (build-time, ClassDB → C# API)

**Files:** `editor/bindings_generator.h/cpp`

Runs at build time. Reads the entire ClassDB and generates the GodotSharp C# API (the `Node3D`, `Resource`, etc. classes that C# users import).

```
ClassDB (all registered classes)
  → _populate_object_type_interfaces()    Build TypeInterface for each class
  → _populate_builtin_type_interfaces()   Build TypeInterface for Variant types
  → _generate_cs_type_member()            Emit C# source for each class member
  → Output: GodotSharp API .cs files
```

**TypeInterface** (`bindings_generator.h:239-621`) — central data structure mapping a Godot type to its C# representation. Contains:
- `c_type` / `c_type_in` / `c_type_out` — native interop types
- `cs_type` / `cs_in_expr` / `cs_out` — C# marshalling expressions
- `cs_variant_to_managed` / `cs_managed_to_variant` — Variant conversion

**Key type mappings:**

| Godot Type | C Type (ptrcall) | C# Type |
|-----------|-----------------|---------|
| `int` (Variant::INT) | `long` (int64_t) | `long` (default), or `int`/`byte`/etc. via metadata |
| `float` (Variant::FLOAT) | `double` | `double` (default) or `float` via metadata |
| `bool` | `godot_bool` | `bool` (via `.ToGodotBool()`) |
| `String` | `godot_string` | `string` (via marshalling, disposable) |
| `StringName` | `godot_string_name` | `StringName` (via `.NativeValue`) |
| Vector2/3/4, Rect2, etc. | passed by ref (`&%0`) | Same struct name (unsafe context) |
| `Object`-derived | `IntPtr` | Class name (via `GodotObject.GetPtr()`) |
| `Array` | `godot_array` | `Godot.Collections.Array` or `Array<T>` |
| `Dictionary` | `godot_dictionary` | `Godot.Collections.Dictionary` or `Dictionary<K,V>` |
| Packed arrays | `godot_packed_*_array` | `T[]` (Span-compatible) |

### Pipeline B: Source Generator (compile-time, C# code → property registration)

**Files:** `editor/Godot.NET.Sdk/Godot.SourceGenerators/`

Roslyn source generator that runs at C# compile time. Processes `[Export]` attributes on fields/properties and generates the engine-facing property registration code.

```
C# user code with [Export] fields
  → ScriptPropertiesGenerator discovers exported members
  → Determines PropertyInfo (Variant type, hint, hint_string)
  → Generates:
      - GetGodotPropertyList()              Static method returning List<PropertyInfo>
      - SetGodotClassPropertyValue()        Override for property setting
      - GetGodotClassPropertyValue()        Override for property getting
      - PropertyName nested class           Cached StringName constants
  → ScriptPropertyDefValGenerator generates:
      - GetGodotPropertyDefaultValues()     Static method returning defaults
```

**[Export] type → PropertyInfo mapping** (`ScriptPropertiesGenerator.cs:712-742`):

| C# Type | Variant.Type | PropertyHint | HintString |
|---------|-------------|--------------|------------|
| `Resource` subclass | OBJECT (24) | `ResourceType` (17) | Class name |
| `Node` subclass | OBJECT (24) | `NodeType` (34) | Class name |
| C# enum | INT (2) | `Enum` (2) | `"A,B,C"` or `"A:0,B:1"` |
| C# `[Flags]` enum | INT (2) | `Flags` (6) | `"A:1,B:2,C:4"` |
| `Array<T>` | ARRAY (28) | `TypeString` (23) | `"elemType/hint:hintStr"` |
| `Dictionary<K,V>` | DICTIONARY (27) | `TypeString` (23) | `"kType/kHint:kStr;vType/vHint:vStr"` |

**Validation errors emitted by source generator:**

| Code | Condition |
|------|-----------|
| GD0101 | Exported member is static |
| GD0102 | Exported type not supported (marshalType == null) |
| GD0103 | Exported member is read-only |
| GD0104 | Exported property is write-only |
| GD0105 | Exported property is an indexer |
| GD0106 | Exported property is explicit interface implementation |
| GD0107 | Non-Node class exports a Node-typed member |

### Pipeline C: Runtime Bridge (CSharpScript ↔ Engine)

**Files:** `csharp_script.h/cpp`, `glue/GodotSharp/.../Bridge/`

At runtime, CSharpScript implements the `Script` interface. CSharpInstance implements `ScriptInstance`. These bridge C# object state to the engine's property/method system.

```
Engine requests property list
  → CSharpScript::_update_exports()
  → Calls managed callback → ScriptManagerBridge.GetPropertyInfoList()
  → C# reflects on GetGodotPropertyList() static method
  → Returns List<PropertyInfo> marshalled to godotsharp_property_info[]
  → CSharpScript caches in exported_members_cache + member_info

Inspector sets a property value
  → CSharpInstance::set()
  → managed callback → CSharpInstanceBridge.Set()
  → Calls generated SetGodotClassPropertyValue() on the C# object
  → Falls back to _Set() if not found
```

**Interop structures** (`gd_mono_cache.h:56-68`):
```cpp
struct godotsharp_property_info {
    godot_string_name name;
    godot_string hint_string;
    Variant::Type type;
    PropertyHint hint;
    PropertyUsageFlags usage;
    bool exported;
};
```

---

## 3. GodotObject Base Class (C#)

**File:** `glue/GodotSharp/GodotSharp/Core/GodotObject.base.cs`

Every C# Godot class inherits from `GodotObject`:

```csharp
public partial class GodotObject : IDisposable {
    internal IntPtr NativePtr;           // Native C++ Object*
    private bool _memoryOwn;             // True for RefCounted (C# owns lifecycle)
}
```

**Memory ownership model:**
- **Node-derived** (`_memoryOwn = false`): Strong GC handle. Engine manages lifetime.
- **RefCounted-derived** (`_memoryOwn = true`): Weak GC handle. C# ref-counting drives disposal.

**Binding strategies** (`InteropUtils.cs`):
1. **Native classes** (e.g., `Node3D`): `TieManagedToUnmanaged()` with native name
2. **User scripts** (e.g., `MyPlayer : Node3D`): `TieManagedToUnmanaged()` with CSharpScript

---

## 4. Marshalling: C# ↔ Variant

**File:** `glue/GodotSharp/GodotSharp/Core/NativeInterop/`

### Variant structure in C#

```csharp
// InteropStructs.cs — mirrors native Variant layout
[StructLayout(LayoutKind.Sequential, Pack = 8)]
public ref struct godot_variant {
    private int _typeField;        // Variant::Type
    private godot_variant_data _data;  // 32-byte union
}
```

### Type resolution chain

`Marshaling.ConvertManagedTypeToVariantType(Type)` — maps C# types:

```
typeof(T) → Variant.Type
├─ Primitives (bool/int/float/etc.) → direct mapping
├─ Godot structs (Vector2, etc.) → direct mapping
├─ Enums → Variant.Type.Int
├─ GodotObject subclass → Variant.Type.Object
├─ Godot.Collections.Array/Dictionary → corresponding type
├─ T[] → PackedArray type based on element type
└─ Unknown → null (export not supported)
```

### Object marshalling

```
C# GodotObject  →  IntPtr (GodotObject.GetPtr())  →  godot_variant (Type.Object)
                                                            ↓
                                                    { ulong id; IntPtr obj; }
```

Reverse: `VariantUtils.ConvertToGodotObject()` → extracts pointer → `InteropUtils.UnmanagedGetManaged()` → walks GC handle chain to find/create C# wrapper.

---

## 5. CSharpScript Property System

**File:** `csharp_script.h/cpp`

### Key caches (TOOLS_ENABLED)

```cpp
class CSharpScript : public Script {
    List<PropertyInfo> exported_members_cache;           // Cached property list
    HashMap<StringName, Variant> exported_members_defval_cache;  // Default values
    HashMap<StringName, PropertyInfo> member_info;       // All members by name
};
```

### TypeInfo for script classes

```cpp
struct TypeInfo {
    String class_name;
    StringName native_base_name;    // The native Godot class this extends
    bool is_tool;                   // [Tool]
    bool is_global_class;           // [GlobalClass]
    bool is_abstract;
};
```

### Property list flow

1. `_update_exports()` (csharp_script.cpp:2121) — entry point
2. Calls managed `GetPropertyInfoList` callback
3. C# invokes source-generated `GetGodotPropertyList()` via reflection
4. Results marshalled through `godotsharp_property_info` array
5. Stored in `exported_members_cache` and `member_info`
6. `get_script_property_list()` (line 2740) returns cached list

### Instance get/set

`CSharpInstance::set()` → managed callback → `CSharpInstanceBridge.Set()`:
1. Tries source-generated `SetGodotClassPropertyValue()` (name → VariantUtils conversion)
2. Falls back to user's `_Set()` override
3. Falls back to base class chain

---

## 6. Existing Interface Support — TECHNICAL DEBT

> **WARNING:** The existing interface-as-export implementation is considered a bad implementation
> from a previous developer. Any work in this area needs a proper redesign, not incremental fixes.

### What exists and where

| Location | What it does | Problem |
|----------|-------------|---------|
| `core/object/object.h:86` | `PROPERTY_HINT_INTERFACE_TYPE = 35` in core enum | C#-only concept leaks into language-agnostic core engine |
| `ScriptPropertiesGenerator.cs:736-741` | Source gen detects `TypeKind.Interface`, emits hint 35 | Only generation side; no compile-time safety |
| `editor/inspector/editor_properties.cpp:4199-4206` | Creates `EditorPropertyNodePath` with `set_interface_mode(true)` | Tied to NodePath picker, assumes interfaces = nodes only |
| `editor/scene/scene_tree_editor.cpp:368-382` | Filters scene tree by `implements_interface()` | `#ifdef MODULE_MONO_ENABLED` — dead code in non-Mono builds |
| `editor/inspector/editor_properties.cpp:3144-3160` | Validates dropped nodes by interface | Same `#ifdef` guard problem |
| `scene/resources/packed_scene.cpp:884+` | Converts interface-typed props to NodePaths for serialization | Treats interfaces identically to node types |
| `csharp_script.cpp:2728-2738` | `CSharpScript::implements_interface()` delegates to managed callback | String-based matching only |
| `ScriptManagerBridge.cs:408-434` | `ScriptImplementsInterface()` — matches by `iface.Name` or `iface.FullName` | Fragile string comparison, no namespace resolution |
| `VariantUtils.generic.cs:199-210` | Runtime `typeof(T).IsInterface` check, casts to `GodotObject` | Assumes all interface implementors are GodotObjects |

### Architectural issues

1. **Core engine pollution**: `PROPERTY_HINT_INTERFACE_TYPE` in `core/object/object.h` makes the entire engine aware of a C#-specific concept. GDScript has no interfaces. GDExtension languages may or may not.

2. **`#ifdef MODULE_MONO_ENABLED` scattered through editor code**: The scene tree editor and inspector have mono-specific interface checks guarded by `#ifdef`. This means:
   - Dead code paths in non-Mono builds
   - No generalization for other scripting languages
   - Maintenance burden across unrelated files

3. **String-based interface matching**: `ScriptManagerBridge.ScriptImplementsInterface()` compares `iface.Name == interfaceNameStr`. This is:
   - Fragile to namespace changes
   - No compile-time verification
   - Ambiguous if two interfaces share a name in different namespaces

4. **Assumes interfaces = node references**: The entire inspector/serialization path treats interface-typed exports as NodePaths. This means:
   - Cannot export a Resource that implements an interface
   - Cannot export a non-tree object implementing an interface
   - Serialization converts to NodePath and back, losing type info

5. **No deserialization validation**: When a scene loads and resolves a NodePath back to a node, there's no check that the resolved node's script actually implements the required interface.

6. **No GDScript interop**: If a GDScript-based node conceptually satisfies an interface contract (has the right methods), it can't be assigned to an interface-typed export.

---

## 7. Inspector Property Assignment Flow

### EditorPropertyNodePath — 4 Validation Checkpoints

`EditorPropertyNodePath` (`editor/inspector/editor_properties.cpp`) is the inspector widget for Node-typed and interface-typed properties. It has 4 distinct validation points:

1. **Scene tree filtering** (`scene_tree_editor.cpp:368-416`): When the node picker opens, `SceneTreeEditor` filters which nodes are shown. In `interface_mode`, only nodes whose script `implements_interface()` are visible. Guarded by `#ifdef MODULE_MONO_ENABLED`.

2. **Drop validation** (`editor_properties.cpp:3121-3181`): `is_drop_valid()` checks drag data before accepting a drop. For interface mode, validates the dropped node's script implements the required interface via `SceneTreeEditor::_is_script_type_valid()`.

3. **Node selection callback** (`editor_properties.cpp:3082-3119`): `_node_selected()` is called when user picks a node from the tree dialog. Converts the selected node to a NodePath relative to the edited node's root.

4. **Emit changed** → `EditorInspector::_edit_set()`: The final property assignment. Goes through `EditorInspector` which wraps the change in an undo/redo action.

### Drag-and-Drop Flow (SceneTreeDock → Inspector)

```
SceneTreeEditor::get_drag_data_fw()
  → Creates drag Dictionary: { "type": "nodes", "nodes": [NodePath...] }

EditorPropertyNodePath::is_drop_valid()
  → Checks drag type == "nodes"
  → Validates node type or interface compliance
  → Returns true/false

EditorPropertyNodePath::drop_data_fw()
  → Calls _node_selected(dropped_node_path)
  → Converts to relative NodePath
  → emit_changed(property_name, new_value)

EditorInspector::_edit_set()
  → Creates UndoRedo action
  → object->set(property, value)       // forward
  → object->set(property, old_value)   // undo
```

---

## 8. Packed Scene Serialization Cycle

### Packing: Node* → NodePath

During `SceneState::pack()` (`packed_scene.cpp:884-1001`):

```
For each property on each node:
  if hint == PROPERTY_HINT_NODE_TYPE || hint == PROPERTY_HINT_INTERFACE_TYPE:
    Convert Node* → NodePath via get_path_to(n)
    Set FLAG_PATH_PROPERTY_IS_NODE bit (bit 30) on property name index
    Store NodePath in variants array
```

Handles three value shapes:
- **Single node** (lines 884-893): Direct `Node* → NodePath`
- **Array of nodes** (lines 912-927): Each element converted
- **Dictionary with node keys/values** (lines 941-975): Keys and/or values converted

### Binary format

```cpp
// packed_scene.h:120-127
FLAG_PATH_PROPERTY_IS_NODE = (1 << 30);      // Bit 30 = deferred
FLAG_PROP_NAME_MASK = FLAG_PATH_PROPERTY_IS_NODE - 1;  // Lower 30 bits = name index
```

### Unpacking: NodePath → Node* (two-phase)

**Phase 1 — Detection** (`packed_scene.cpp:374-394`): During node instantiation loop, properties with `FLAG_PATH_PROPERTY_IS_NODE` are deferred (not set immediately). Stored in `DeferredNodePathProperties` vector.

**Phase 2 — Resolution** (`packed_scene.cpp:595-639`): After ALL nodes exist:

```
for each deferred property:
  base = ObjectDB::get_instance<Node>(dnp.base)
  if value is Array:   resolve each element via get_node_or_null()
  if value is Dict:    resolve keys/values via get_node_or_null()
  if value is single:  base->set(prop, base->get_node_or_null(path))
```

### Critical gaps in unpacking

| Issue | Detail |
|-------|--------|
| **No type validation** | `get_node_or_null()` returns any Node or nullptr — no check against expected type/interface |
| **Silent nullptr** | If referenced node doesn't exist, silently sets nullptr (no warning) |
| **No interface check** | Resolved node's script is never validated against interface requirement |
| **Recovery not available** | `_recover_node_path_index()` only works for parent/owner refs, NOT deferred properties |

### Data structures

```cpp
struct DeferredNodePathProperties {
    ObjectID base;          // Owning node
    StringName property;    // Property name
    Variant value;          // NodePath, Array, or Dictionary of NodePaths
};
```

---

## 9. GDScript Property System (Comparison)

Understanding how GDScript handles the same property reporting helps scope what a redesigned interface system must account for.

### Property pipeline

```
@export annotation (gdscript_parser.cpp:4642-4922)
  → Parser sets variable->exported + variable->export_info
  → Compiler (gdscript_compiler.cpp:2867-2902) builds PropertyInfo via to_property_info()
  → Stored in MemberInfo.property_info
  → get_script_property_list() (gdscript.cpp:311-352) returns cached list
```

### @export → PropertyHint mapping

| DataType Kind | PropertyHint | hint_string |
|--------------|-------------|-------------|
| NATIVE (Resource subclass) | `RESOURCE_TYPE` (17) | class_name |
| NATIVE (Node subclass) | `NODE_TYPE` (34) | class_name |
| SCRIPT (Resource) | `RESOURCE_TYPE` (17) | global_name |
| SCRIPT (Node) | `NODE_TYPE` (34) | global_name |
| ENUM | `ENUM` (2) | `"Name:value,..."` |

### GDScript has NO interface/protocol concept

- Type system is **inheritance-only** (`ClassDB::is_parent_class()`)
- `GDScriptDataType::is_type()` checks VARIANT, BUILTIN, NATIVE, SCRIPT, GDSCRIPT kinds
- No duck-typing / protocol conformance mechanism
- `resolve_class_interface()` in `gdscript_analyzer.cpp` resolves class member types — "interface" here means "class surface", NOT a protocol/interface type

### Dynamic properties

GDScript supports `_get_property_list()` returning Array of Dictionaries with `name`, `type`, `hint`, `hint_string`, `usage`, `class_name` fields. This is the same mechanism available to any scripting language for custom properties.

---

## 10. GDExtension Limitations

### No custom PropertyHint extension mechanism

The `PropertyHint` enum in `core/object/object.h` is a **fixed set**. GDExtension languages cannot add new hint values.

```cpp
// core/extension/gdextension_interface.gen.h:206-215
typedef struct {
    GDExtensionVariantType type;
    GDExtensionStringNamePtr name;
    GDExtensionStringNamePtr class_name;
    uint32_t hint;              // Must use existing PropertyHint values
    GDExtensionStringPtr hint_string;
    uint32_t usage;
} GDExtensionPropertyInfo;
```

**Implication for interface redesign**: Any new property hint (or repurposed hint) for interfaces affects ALL language bindings. GDExtension languages would need to handle it or explicitly ignore it. This is why `PROPERTY_HINT_INTERFACE_TYPE = 35` being in the core enum is problematic — it forces all languages to be aware of a concept that may not apply to them.

**Alternative approaches** that avoid core enum pollution:
- Encode interface info in `hint_string` under an existing hint (e.g., `PROPERTY_HINT_NODE_TYPE` with extended format)
- Use `PROPERTY_USAGE_*` flags to signal interface constraints
- Keep interface resolution entirely in the scripting layer, not in the property system

---

## 11. C# Test Infrastructure

### Location and framework

```
modules/mono/editor/Godot.NET.Sdk/Godot.SourceGenerators.Tests/
  ├── Godot.SourceGenerators.Tests.csproj    xUnit 2.9.3
  ├── CSharpSourceGeneratorVerifier.cs       Snapshot test harness
  ├── CSharpAnalyzerVerifier.cs              Diagnostic test harness
  ├── CSharpCodeFixVerifier.cs               Code fix test harness
  ├── Constants.cs                           Net8.0 refs, GodotSharp assembly
  ├── TestData/
  │   ├── Sources/                           37 test source files
  │   └── GeneratedSources/                  100+ expected output snapshots
  └── 17 test classes (ScriptProperties, ExportDiagnostics, etc.)
```

### Snapshot testing pattern

Tests use Microsoft's Roslyn testing framework. Source generator tests follow this pattern:

```csharp
[Fact]
public async Task ExportedFields() {
    await CSharpSourceGeneratorVerifier<ScriptPropertiesGenerator>.Verify(
        new string[] { "ExportedFields.cs" },           // Input sources
        new string[] { "ExportedFields_ScriptProperties.generated.cs" }  // Expected output
    );
}
```

The test harness:
1. Creates in-memory Roslyn compilation (no Godot runtime needed)
2. Adds GodotSharp.dll as metadata reference
3. Runs source generator on test inputs
4. Compares generated code against snapshot files

### Export diagnostic coverage

| Code | Test File | What's Validated |
|------|-----------|-----------------|
| GD0101-GD0106 | Various | Static, unsupported type, read-only, write-only, indexer, explicit interface |
| GD0107 | `ExportDiagnostics_GD0107.cs` | Node exports only in Node-derived classes |
| GD0108-GD0111 | Various | ExportToolButton constraints |

### What's NOT tested

- **No runtime integration tests**: No tests instantiate scenes with C# scripts
- **No `PROPERTY_HINT_INTERFACE_TYPE` tests**: Interface hint value never appears in test data
- **No cross-language tests**: No testing of C# script properties consumed by GDScript or GDExtension
- **No packed scene round-trip tests**: No testing of serialize → deserialize for node-typed properties

### Running tests

```bash
dotnet test modules/mono/editor/Godot.NET.Sdk/Godot.SourceGenerators.Tests/Godot.SourceGenerators.Tests.csproj
```

### Adding new snapshot tests

1. Add test source to `TestData/Sources/MyTest.cs`
2. Add expected generated output to `TestData/GeneratedSources/MyTest_ScriptProperties.generated.cs`
3. Add test method in appropriate test class:
   ```csharp
   [Fact]
   public async Task MyTest() {
       await CSharpSourceGeneratorVerifier<ScriptPropertiesGenerator>.Verify(
           new string[] { "MyTest.cs" },
           new string[] { "MyTest_ScriptProperties.generated.cs" }
       );
   }
   ```

---

## 12. Key File Reference

| File | Lines | What to find |
|------|-------|-------------|
| `csharp_script.h` | 58-307 | CSharpScript class, TypeInfo struct, exported_members_cache |
| `csharp_script.cpp` | 1491-1626 | CSharpInstance get/set/property_list/validate_property |
| `csharp_script.cpp` | 2075-2189 | Property info callbacks, _update_exports |
| `csharp_script.cpp` | 2728-2738 | `implements_interface()` — string-based delegation |
| `gd_mono_cache.h` | 56-68 | `godotsharp_property_info` interop struct |
| `gd_mono_cache.h` | 71-136 | All managed callback function pointer types |
| `bindings_generator.h` | 239-621 | TypeInterface — complete Godot↔C# type mapping |
| `bindings_generator.cpp` | 3876-4469 | ClassDB → TypeInterface population |
| `bindings_generator.cpp` | 4737-5070 | Builtin type interface setup (int, float, string, etc.) |
| `ScriptPropertiesGenerator.cs` | 243-296 | GetGodotPropertyList code generation |
| `ScriptPropertiesGenerator.cs` | 630-954 | TryGetMemberExportHint — type→hint mapping |
| `ScriptPropertiesGenerator.cs` | 712-742 | Resource/Node/Interface hint determination |
| `ScriptPropertyDefValGenerator.cs` | 186-206 | Export validation (GD0101-GD0107) |
| `MarshalUtils.cs` | 26-85 | C# type → MarshalType mapping |
| `Marshaling.cs` | 19-200 | ConvertManagedTypeToVariantType |
| `VariantUtils.cs` | 334-517 | Object ↔ Variant conversion |
| `InteropUtils.cs` | 9-96 | TieManagedToUnmanaged, UnmanagedGetManaged |
| `ScriptManagerBridge.cs` | 408-434 | ScriptImplementsInterface — string matching |
| `ScriptManagerBridge.cs` | 1015-1110 | GetPropertyInfoListForType |
| `GodotObject.base.cs` | 13-354 | NativePtr, _memoryOwn, Dispose, property virtuals |
| `core/object/object.h` | 51-97 | PropertyHint enum (INTERFACE_TYPE = 35) |
| `editor/inspector/editor_properties.cpp` | 3082-3181 | EditorPropertyNodePath: selection, drop validation |
| `editor/inspector/editor_properties.cpp` | 4199-4206 | Interface hint → EditorPropertyNodePath |
| `editor/scene/scene_tree_editor.cpp` | 364-416 | Interface-mode node filtering + `#ifdef` guards |
| `scene/resources/packed_scene.h` | 64-77 | NodeData::Property, DeferredNodePathProperties structs |
| `scene/resources/packed_scene.h` | 120-127 | FLAG_PATH_PROPERTY_IS_NODE (bit 30) |
| `scene/resources/packed_scene.cpp` | 374-394 | Deferred property detection during instantiation |
| `scene/resources/packed_scene.cpp` | 595-639 | Deferred NodePath → Node* resolution |
| `scene/resources/packed_scene.cpp` | 884-1001 | Packing: Node* → NodePath conversion |
| `gdscript.h` | 67-73 | GDScript::MemberInfo (property_info + data_type) |
| `gdscript.cpp` | 311-352 | GDScript::get_script_property_list() |
| `gdscript.cpp` | 1522-1593 | GDScriptInstance::set() with type validation |
| `gdscript_parser.cpp` | 4642-4922 | @export annotation → PropertyInfo |
| `gdscript_compiler.cpp` | 2867-2902 | Compiler: MemberInfo + PropertyInfo assembly |
| `gdscript_function.h` | 46-180 | GDScriptDataType::is_type() — runtime validation |
| `core/extension/gdextension_interface.gen.h` | 206-215 | GDExtensionPropertyInfo — fixed hint field |
| Source generator tests | `Godot.SourceGenerators.Tests/` | xUnit snapshot tests for [Export] code gen |
