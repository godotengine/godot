# CrossRuntime Module for Godot C# Web Support

## Implementation Documentation

## 1. Purpose and architecture

The **CrossRuntime Module** provides a bidirectional bridge between two independent WebAssembly runtimes:

- **Godot WASM** running on the browser main thread.
- **.NET WASM** running inside a Web Worker.

The design allows C# code to control a live Godot engine instance without requiring direct in-process interop between the two runtimes. Instead, both sides communicate through a shared memory buffer with a fixed layout and a strict command protocol.

The core design goals are:

1. Keep the Godot side authoritative for engine state.
2. Keep the C# side ergonomic and API-like.
3. Avoid handwritten duplication by generating both C++ and C# bindings from the same API dump.
4. Preserve predictable binary compatibility across the two runtimes by using a fixed memory contract.

This system is not merely a generic message bus. It is a deliberately constrained runtime bridge with concrete offsets, explicit type codecs, and generated method dispatchers.

---
## Folder Layout

```
godot/modules/cross_runtime/
  cpp/
    headers/
      bridge_api.h          # command IDs and memory offsets
      bridge_helpers.h      # read/write helpers for data structures
    *cpp files              # Godot API implementation

  cs/
    Interop.cs              # JSImport/JSExport bridge
    Program.cs
    WebGodotApi.csproj
    GodotApi/               # generated C# API files
    Utilities/
      Commands.cs           # C# mirror of bridge_api.h
      CoreTypes.cs
      Helpers.cs            # read/write helpers
      VariantHandling.cs    # Variant codec and dispatch

  api_generator/
    Main.py
    Metadata.py
    Utilities.py
    cpp_generator.py
    cs_generator.py
    FindBindings/dump.gd

  tests/
    run_tests.py
    Tests.md
    C++/
      tests_caller.cpp
      headers/
        bridge_helpers.h
        tests.h
    NET/
      Interop.cs
      Program.cs
      Tests.cs
      Tests.csproj
      Utilities/
        CoreTypes.cs
        Helpers.cs
        VariantBuffer.cs
        VariantHandler.cs

  docs/
    CrossRuntime.md         # explains the whole project
    Tests.md                # explains the tests available
    HowtoReproduce.md       # explains the api reproducibility
    RunningTests.md         # explains how to run the tests



  command_dispatcher.cpp    # command handler table
  command_processor.cpp     # per-frame command execution
  config.py
  register_types.cpp
  register_types.h
  SCsub
```

---

## 2. Runtime model

The project is built around two execution domains:

### 2.1 C# runtime

The C# runtime executes in a Web Worker, where it does the following:

- serializes command arguments into shared memory,
- writes the command ID,
- resets the status flag to pending,
- waits until the Godot side marks the command complete,
- then reads back the result from the shared buffer.

This side behaves like a client API.

### 2.2 Godot runtime

The Godot side runs on the browser main thread and is responsible for:

- inspecting the current command once per frame,
- decoding arguments from memory,
- invoking the proper engine method through `ClassDB::get_method`,
- writing the return value back to the shared region,
- and setting the status flag to done.

This side behaves like the execution backend.

---

## 3. Shared memory contract

The implementation depends on a fixed set of offsets known to both runtimes.

### 3.1 Command region

```text
0x5000  CMD_OFFSET     command ID to execute
0x5004  STATUS_OFFSET  status byte / completion flag
0x5008  CMD_DATA      command payload region
```

### 3.2 Semantics

- `CMD_OFFSET` stores the command identifier.
- `STATUS_OFFSET` stores a completion indicator.
- `CMD_DATA` stores serialized command arguments and return data.

The contract is intentionally simple:

1. C# writes arguments into `CMD_DATA`.
2. C# writes the command ID into `CMD_OFFSET`.
3. C# resets `STATUS_OFFSET` to `STATUS_PENDING`.
4. Godot reads the command during its frame update.
5. Godot executes the handler and writes output back into `CMD_DATA`.
6. Godot sets `STATUS_OFFSET = STATUS_DONE`.
7. C# resumes execution.

### 3.3 Atomicity requirements

`CMD_OFFSET` and `STATUS_OFFSET` are treated as control fields. They are accessed using atomic operations on the C# side and equivalent low-level writes on the Godot side.

The design assumes that these fields are the synchronization boundary; the rest of the payload is treated as data and is transferred through bulk reads and writes.

---

## 4. C# interop layer

The C# side is split into several responsibilities.

### 4.1 `Interop.cs`

This file is the entry point into JS-hosted interop.

It imports the JavaScript bridge functions using `[JSImport]`:

- `BulkRead(int srcOffset, int length)`
- `BulkWrite(byte[] srcArray, int destOffset, int length)`
- `AtomicWriteInt32(int offset, int value)`
- `AtomicReadInt32(int offset)`

It also exports two runtime control functions using `[JSExport]`:

- `InitInterop()`
- `RunGame()`

#### `InitInterop()`

This loads the JS module `interop.js` through `JSHost.ImportAsync("interop", "/cs/interop.js")`.

The important detail is that the .NET runtime does not assume the JS bridge exists at startup. It imports it explicitly and only then proceeds with engine communication.

#### `RunGame()`

This resets the command buffer and then starts the C# test or game entry point through `Tests.Run()`.

That reset step matters because command memory must begin from a known state. Without it, stale command IDs or stale status flags could cause false command execution.

---

## 5. C# helper layer

### 5.1 `Helpers.cs`

This file defines the primitive and structured marshaling functions used by the generated API.

The helpers are intentionally layered:

- low-level scalar operations (`ReadInt32`, `WriteFloat`, etc.),
- string and node-path handling,
- vector/matrix and transform types,
- packed arrays,
- variant encoding and decoding.

The helpers are the canonical C# representation of the shared memory format.

### 5.2 Scalar accessors

The scalar helpers are implemented using `Interop.BulkRead` and `Interop.BulkWrite` plus `BitConverter`:

- `WriteByte` / `ReadByte`
- `WriteInt32` / `ReadInt32`
- `WriteInt64` / `ReadInt64`
- `WriteUInt64` / `ReadUInt64`
- `WriteFloat` / `ReadFloat`
- `WriteDouble` / `ReadDouble`

These helpers are used both directly and indirectly by higher-level encoders.

### 5.3 String handling

Strings are encoded as:

- a 32-bit length prefix,
- followed by UTF-8 bytes.

The implementation writes the byte length first, then writes the raw UTF-8 payload. Reading uses the prefix to reconstruct the string.

There is also a conservative length guard in `ReadString` to avoid reading arbitrarily large or malformed data.

### 5.4 Structured value helpers

The file includes explicit read/write functions for Godot-style value types:

- `Vector2`, `Vector2i`
- `Vector3`, `Vector3i`
- `Vector4`, `Vector4i`
- `Color`
- `Rect2`, `Rect2i`
- `Transform2D`
- `Basis`
- `Transform3D`
- `Quaternion`
- `AABB`
- `Plane`
- `Projection`

Each of these is written in a field-by-field layout that mirrors the C++ side.

This matters because the implementation is not abstractly serializing “a vector”; it is explicitly relying on a known memory layout and field order.

### 5.5 Packed arrays

The packed array helpers serialize a count prefix followed by the array body.

Implemented types include:

- `PackedByteArray`
- `PackedInt32Array`
- `PackedInt64Array`
- `PackedFloat32Array`
- `PackedFloat64Array`
- `PackedStringArray`
- `PackedVector2Array`
- `PackedVector3Array`
- `PackedVector4Array`
- `PackedColorArray`

The array helpers are not all implemented identically. Some use scalar loops, while others use `Buffer.BlockCopy` or bulk byte transfers where possible.

### 5.6 Dictionary and Array helpers

`ReadDictionary`, `WriteDictionary`, `ReadArray`, and `WriteArray` delegate to `VariantHandling`.

This is important because dictionaries and arrays are not encoded as plain scalar arrays. They are serialized recursively as variants.

### 5.7 Signal and callable helpers

`CustomSignal` and `CustomCallable` are project-specific structures that represent Godot `Signal` and `Callable` values in a simplified form.

They are encoded as:

- a `ulong` target ID,
- followed by a string name or method field.

This is a deliberate simplification relative to the full engine type system.

---

## 6. Variant encoding and decoding

### 6.1 `VariantHandling.cs`

This is the most important serialization component in the C# side.

It defines a custom `VariantType` enum matching Godot’s variant IDs.

The encoder and decoder implement a project-specific binary layout for each supported type.

### 6.2 Header format

Variants are prefixed with a 32-bit header containing:

- the variant type ID in the low bits,
- a 64-bit flag where applicable.

The implementation uses:

```csharp
private const int ENCODE_FLAG_64 = 1 << 16;
```

The flag is used to mark integer and float variants that are encoded as 64-bit values in this transport format.

### 6.3 `Encode(int offset, object value)`

This method writes a value starting at `offset` and returns the number of bytes written.

The encoding rules are explicit per case:

- `null` -> `Nil`
- `bool` -> 32-bit integer payload
- `int` and `long` -> integer variant with 64-bit flag
- `float` and `double` -> float variant with 64-bit flag
- `string` -> length prefix plus UTF-8 bytes, with padding logic
- Godot math types -> direct field encoding
- arrays and dictionaries -> recursive variant encoding
- `CustomCallable` and `CustomSignal` -> specialized object-like encodings

The encoder is purposefully case-driven rather than reflection-driven. That keeps the mapping explicit and deterministic.

### 6.4 `DecodeInternal(int offset, out int bytesRead)`

Decoding is the inverse operation.

The method reads the header, inspects the low 16 bits for the type ID, checks the 64-bit flag when needed, and then dispatches to the matching reader.

For strings, packed strings, dictionaries, arrays, callables, and signals, decoding uses recursive or multi-step length traversal.

The implementation also returns `bytesRead`, which is crucial for nested decoding, especially inside dictionaries and arrays.

### 6.5 Important nuance: object-like return values

For `Callable`, `Signal`, `Dictionary`, and `Array`, the decoder returns the corresponding runtime object representation rather than a raw byte structure.

That makes the API convenient on the C# side, but it also means the type system is a transport representation rather than a strict 1:1 engine mirror.

---

## 7. Variant cursor buffer

### 7.1 `VariantBuffer.cs`

`VariantBuffer` is a small cursor-based helper around the variant codec.

It maintains a current position and supports sequential reading and writing.

This is useful for nested structures that are encoded sequentially, especially where offsets need to advance without recomputing lengths manually.

The class exposes:

- `Write(object value)`
- `Read()`
- `WriteCallable(CustomCallable callable)`
- `WriteSignal(CustomSignal signal)`

This class exists to keep cursor progression localized and reduce duplicated offset arithmetic.

---

## 8. Godot-side bridge helpers

### 8.1 `bridge_helpers.h`

This file is the C++ mirror of the serialization logic.

It provides template-based primitive readers and writers plus type-specific adapters for all supported Godot values.

The key distinction is that this side directly manipulates the engine’s native C++ types.

### 8.2 Primitive access

`reader<T>` and `writer<T>` perform byte-wise copy operations between the shared buffer and a local value.

This keeps the implementation simple and avoids assumptions about alignment-sensitive direct casts.

### 8.3 String helpers

`read_string_from_data()` and `write_string_to_data()` implement the same UTF-8 length-prefixed format used by the C# side.

This is one of the most sensitive parts of the bridge, because any divergence in string length handling or padding strategy can corrupt higher-level data structures.

### 8.4 Scalar, vector, matrix, and resource-like helpers

The file defines direct readers and writers for:

- `ObjectID`
- `int32_t`, `int64_t`, `uint64_t`
- `float`, `double`
- `bool`
- `Vector2`, `Vector2i`
- `Rect2`, `Rect2i`
- `Vector3`, `Vector3i`
- `Transform2D`
- `Vector4`, `Vector4i`
- `Plane`
- `Quaternion`
- `AABB`
- `Basis`
- `Transform3D`
- `Projection`
- `Color`
- `StringName`
- `NodePath`
- `RID`

The functions are meant to align with the in-memory layout of the corresponding engine types.

### 8.5 Packed arrays

C++ supports direct packed array readers and writers for the same family as C#.

The implementation writes a length prefix and then iterates element-by-element into the shared memory region.

### 8.6 Variant helpers

At the bottom of the file, the implementation falls back to Godot’s internal `encode_variant` and `decode_variant` for complex types:

- `Dictionary`
- `Array`
- `Callable`
- `Signal`
- arbitrary `Variant` values

This is a deliberate divergence from the hand-written scalar codecs. It means the transport format for these complex values is delegated to Godot’s own variant serializer, which reduces the amount of custom binary logic that must be maintained.

---

## 9. Command processing in Godot

### 9.1 `command_processor.cpp`

This file wires the bridge into the engine frame loop.

It defines a `CommandProcessor` class that owns the frame callback and exposes a static singleton pointer.

### 9.2 Frame callback lifecycle

The processor’s `_frame_update()` method calls `process_api_commands()` once per frame.

This means command execution is intentionally frame-synchronized rather than interrupt-driven.

### 9.3 Registration behavior

`register_command_processing()` waits until the `SceneTree` is available, deferring registration through `MessageQueue` if necessary.

This is an important implementation detail: the processor is not assumed to be attachable at module initialization time. It waits for the engine to be ready.

### 9.4 Signal connection

The code connects the processor to the scene tree’s `process_frame` signal using a callable created from the member method.

This is how the command loop becomes part of Godot’s normal processing rhythm.

### 9.5 Cleanup

`unregister_command_processing()` disconnects the signal and deletes the processor singleton.

That makes the bridge lifecycle explicit and manageable.

---

## 10. Command dispatch generation

### 10.1 Generator architecture

The Python generator emits all runtime bindings from the API dump:

- `bridge_api.h`
- C++ per-class handler files
- `command_dispatcher.cpp`
- `Commands.cs`
- C# API classes
- `GodotObject.cs`

The generator is designed to keep the C++ and C# sides aligned from a single source of truth.

### 10.2 `Main.py`

The main generator performs these steps:

1. load entries from the API JSON,
2. normalize and filter arguments,
3. deduplicate methods by signature,
4. assign command IDs,
5. sort classes by inheritance depth,
6. emit the output files.

The deduplication logic is important because Godot API dumps may contain multiple forms of a method that look similar but differ in arguments or signatures.

### 10.3 Skipped classes

The generator intentionally excludes several class families:

- `Editor*`
- `VisualShader*`
- `GLTF*`
- `FBX*`
- `ResourceImporter*`

This keeps the generated output focused on the subset needed by the runtime bridge.

### 10.4 Command naming

Commands are named by class, method, argument type list, and return type.

This produces names such as:

- `CMD_Class_method__5__9__r24`

The signature-based naming avoids ambiguity and helps keep the binding stable.

### 10.5 Command ID allocation

Command IDs are assigned sequentially starting at `2`.

The IDs `0` and `1` are reserved for `CMD_NONE` and the status protocol.

The `Engine.get_singleton()` command is added as a custom synthetic command at the end of the generated range.

---

## 11. Generated C++ handlers

### 11.1 `generate_cpp_class_file()`

For each engine class, the generator emits a `handle_<Class>()` function with a switch over command IDs.

The handler logic follows a consistent pattern:

1. read the target object ID,
2. resolve it through `ObjectDB::get_instance`,
3. read each method argument from the payload,
4. look up the `MethodBind`,
5. call the method,
6. write the return value, if any,
7. mark the command complete.

### 11.2 Target resolution

Every method call begins by reading an `ObjectID` from the first eight bytes of the payload.

That ID is resolved to an object pointer using `ObjectDB::get_instance(target_id)`.

If the target is invalid, the handler immediately marks the command done and clears the command ID.

### 11.3 Argument decoding

Arguments are decoded using the type-specific `cpp_read` templates from metadata.

Offset progression is handled manually and varies by type. This matters because not all types occupy a fixed or simple size in transport.

The generator includes special handling for certain packed arrays, especially where array lengths make the serialized size variable.

### 11.4 Method invocation

The method is looked up through `ClassDB::get_method(class, method)` and then invoked with `MethodBind::call`.

This means the bridge does not require hand-written call wrappers for each engine method. It uses the engine’s native reflection/binding layer at runtime.

### 11.5 Return encoding

For supported return types, the handler writes the returned value back into `CMD_DATA`.

The generator currently emits explicit writers for some scalar and simple return kinds such as:

- `bool`
- `int64_t`
- `double`
- `Vector2`
- `ObjectID`

The supported return-handling logic is intentionally narrow and should be kept in sync with the metadata map.

---

## 12. Command dispatcher generation

### 12.1 `generate_command_dispatcher()`

This function emits `command_dispatcher.cpp`, which holds the top-level dispatch table used each frame.

It creates an array of function pointers indexed by command ID.

### 12.2 Dispatch flow

At runtime, `process_api_commands()` performs the following:

1. reads the command ID and status flag directly from the shared memory region,
2. exits early if the status is not pending or the command is `CMD_NONE`,
3. validates the command ID against the dispatch table,
4. invokes the corresponding handler,
5. clears the command ID and marks status done.

This is a compact and direct dispatch mechanism, optimized for predictable frame-time behavior.

### 12.3 Out-of-range safety

If the command ID does not map to a valid handler, the dispatcher marks completion and clears the command.

That prevents the bridge from getting stuck waiting forever on a malformed command.

---

## 13. C# API generation

### 13.1 `generate_commands_cs()`

This emits the `Commands.cs` mirror of the C++ command constants.

It also includes the synthetic `CMD_Engine_get_singleton__r0` constant.

This file is the C# side’s authoritative command registry.

### 13.2 `generate_godot_object_cs()`

This creates the minimal `GodotObject` base class.

It exists to store the native object ID and provide a shared base for generated classes.

### 13.3 `generate_cs_class_file()`

For each class, the generator emits a managed wrapper type with methods that:

1. write the target object ID into `CMD_DATA`,
2. serialize arguments in order,
3. send the command,
4. wait for completion,
5. read back the return value, if any.

This makes the generated API feel like direct method invocation even though the actual work happens through shared memory and frame-based dispatch.

### 13.4 Return handling

The return type table determines which managed type is used for each engine return kind.

For example:

- engine `String` becomes C# `string`,
- engine `RID` becomes `ulong`,
- packed arrays become managed arrays.

This mapping is controlled by metadata rather than manually written in every wrapper.

---

## 14. Metadata layer

### 14.1 `metadata.py`

This file is the type authority for the generator.

It defines:

- argument type mappings,
- return type mappings,
- forced return type overrides,
- keyword sets,
- local reserved name sets.

### 14.2 Argument type table

`argument_types` maps Godot type IDs to:

- C++ declaration type,
- C++ read expression,
- transport size,
- C# type,
- C# write expression,
- C# read expression.

This table is the backbone of the generator.

The important thing is that the size values are not cosmetic. They drive offset progression in the generated bindings.

### 14.3 Return type table

`return_types` maps type IDs to:

- C++ type,
- C# type,
- C# read expression.

It also encodes a few special cases where the C# representation differs from the engine’s canonical type.

### 14.4 Forced return types

`forced_return_type_by_method` overrides the raw dump return type when necessary.

This is used for cases like `get_singleton` and `get_instance_id`, where the dump metadata may not be sufficient or where a consistent bridge-level return type is required.

### 14.5 Keyword and reserved name handling

The generator uses language keyword sets and local reserved identifier sets to avoid emitting invalid or colliding variable names.

This is especially important because the input API dump may contain names that are valid at the engine level but unsafe or ambiguous in generated C++ and C# code.

---

## 15. Utility layer

### 15.1 `utilities.py`

This file contains the shared helper logic used by both generators.

Its responsibilities include:

- identifier sanitization,
- uniqueness enforcement,
- API JSON loading,
- argument normalization,
- signature deduplication,
- command-name generation,
- class ordering,
- forced return-type override lookup,
- text file emission.

### 15.2 Identifier sanitization

`sanitize_identifier()` removes invalid characters, prevents leading digits, and appends underscores for language keywords.

This ensures that generated code remains syntactically valid in both languages.

### 15.3 Unique identifier generation

`unique_identifier()` resolves collisions against a set of used identifiers.

It also checks language-specific reserved names so that local variable generation does not shadow important symbols.

### 15.4 Class file naming

`cpp_class_file_name()` and `cs_class_file_name()` generate unique file-safe names from engine class names.

This prevents filename collisions after sanitization.

### 15.5 `normalize_args()`

This function enforces that all arguments have a recognized type and a non-empty name.

It also rejects unsupported argument signatures early, which prevents the generator from producing partially broken output.

### 15.6 `method_signature()` and deduplication

Methods are deduplicated by a tuple of:

- class name,
- method name,
- argument type sequence,
- return type.

This avoids generating multiple wrappers for the same semantic binding.

---

## 16. Important implementation nuances

### 16.1 The bridge is contract-driven, not reflection-driven

Although the engine’s `ClassDB` is used for method resolution, the transport format is not inferred dynamically at runtime. It is defined by explicit generated code and a fixed metadata table.

### 16.2 Offset arithmetic is a correctness boundary

The project depends on correct byte offsets.

The generator therefore treats offset progression as part of the protocol, not as an implementation detail.

This is why packed arrays and variable-length strings are handled with special care.

### 16.3 The type system is intentionally asymmetric in places

Some Godot types are represented in C# by simplified managed equivalents.

For example:

- `ObjectID` is treated as `ulong`,
- `StringName` and `NodePath` are treated as `string`,
- `Callable`, `Signal`, `Dictionary`, and `Array` are surfaced as managed objects.

This is a practical interoperability choice rather than a pure type mirror.

### 16.4 `Quaternion` ordering differs between layers

The C# and C++ helper code must be watched carefully for quaternion component order.

The C++ helper explicitly accounts for Godot’s constructor conventions, while the C# struct stores fields in the order declared there.

This is exactly the kind of subtle binary mismatch that can produce apparently valid but mathematically wrong values.

### 16.5 Packed string handling is a special case

Packed strings are not simply counted by a fixed stride. They are variable-length items, and the cursor must advance by the encoded byte length of each string.

Any mismatch here causes downstream corruption in arrays and nested variants.

### 16.6 `Callable` and `Signal` are simplified transport types

The bridge does not attempt to reproduce every internal nuance of engine callables or signals.

Instead, it uses a compact custom representation that is sufficient for the intended interop flow.

### 16.7 Status handling is a synchronization contract

The system depends on the status flag being set correctly and promptly.

If the status flag is not reset before a new command, or not marked done after a command completes, the worker can stall or misread stale state.

---

## 17. Known design constraints

This implementation intentionally assumes:

- one command in flight at a time,
- one-frame latency on the Godot side,
- shared memory availability in the browser environment,
- a stable ABI-style layout for the bridged types,
- generated bindings that remain synchronized with the dump format.

It is therefore a deterministic bridge, not a concurrent job queue.

---

## 18. Practical reading of the build pipeline

The build pipeline is:

1. generate `api.json` from Godot’s ClassDB,
2. run the Python generator,
3. emit both the C++ and C# binding layers,
4. compile the Godot module and the .NET WASM side,
5. let the worker and engine communicate through the shared buffer.

The key property of the pipeline is that the same metadata source generates both sides of the bridge, reducing drift.

---

## 19. Summary

This project is a tightly engineered cross-runtime bridge between Godot and C# in the browser. Its correctness depends on a few very specific pillars:

- a fixed shared-memory contract,
- explicit binary codecs for supported types,
- generated bindings from a shared API dump,
- frame-based command dispatch on the Godot side,
- and atomic command/status coordination.

The implementation is highly practical and intentionally explicit. It avoids magical abstractions in favor of predictable serialization, known offsets, and generated code that stays close to the engine’s native API surface.

That is the right shape for this kind of system.
