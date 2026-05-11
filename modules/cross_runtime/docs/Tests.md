# CrossRuntime Variant Bridge Tests

These tests verify that the C# and C++ sides of the bridge agree on the exact same memory layout, type encoding, and value interpretation. The goal is not just to prove that values can be written and read, but that both runtimes are using the same wire format for every supported type.

## What the test covers

The test writes a fixed set of values into shared memory from the C# side, then asks the Godot side to read them back and validate them. Each value is placed at a specific offset, and those offsets are intentionally mirrored in both runtimes. That means the test is checking three things at once:

1. the offset map is correct,
2. the read/write helpers are symmetric,
3. the complex types decode into the expected engine values.

The coverage includes:

* scalars such as `int32`, `int64`, `float`, `double`, and `bool`,
* string-like values such as `String`, `StringName`, and `NodePath`,
* object-style numeric values such as `RID`,
* math and transform types such as `Vector2`, `Rect2`, `Basis`, `Transform3D`, `Projection`, and `Quaternion`,
* packed arrays such as `PackedByteArray`, `PackedInt32Array`, `PackedFloat64Array`, and others,
* recursive variant containers such as `Dictionary` and `Array`.

## C# test side

The C# `Tests` class prepares the memory region by writing every value to a known offset.

The key implementation detail is that it uses the same `Helpers` methods that the generated API will use in real runtime calls. That means the test is validating the actual transport layer, not a separate mock serializer.

The test then sends a dedicated command, waits for completion, and reads back a result byte from shared memory. That byte acts as the final pass/fail indicator.

This is important because it exercises the same command pipeline used in normal operation:

* reset the command buffer,
* write test data,
* issue the command,
* wait for Godot to process it,
* inspect the result.

## Godot test side

The C++ side reads the memory using the low-level helpers in `bridge_helpers.h`. It does not reconstruct values manually. Instead, it uses the same helper functions that the command bridge uses for real method calls.

Each test is checked in sequence. If one value does not match, the function stops immediately and returns an error string describing the failure. That makes the failure mode precise and easy to diagnose.

The checks are intentionally strict. For example:

* the string tests verify exact string equality,
* the quaternion test verifies the expected field ordering,
* the packed array tests verify both length and element values,
* dictionary and array tests verify nested content, not just container size.

## Why the offsets matter

The offsets are hard-coded on both sides and must remain identical. They are not decorative constants; they are part of the ABI between the two WASM runtimes.

This test therefore acts as a layout lock. If any helper changes its binary format, the test will fail immediately. That is exactly what you want in a bridge like this, because silent mismatch would be far worse than a loud failure.

## Special cases being validated

A few parts of the test are especially important:

### Quaternion layout

The quaternion write and read logic is sensitive to component ordering. The test makes sure the wire order and the engine-side constructor order are compatible.

### Packed string arrays

These are variable-length and are the easiest place for offset drift to happen. The test verifies that multiple strings can be serialized and deserialized in sequence without corruption.

### Dictionaries and arrays

These are not treated as fixed-size blobs. They are validated through recursive variant decoding, which confirms that nested values survive the transport format correctly.

## What a pass means

A passing test means the bridge is internally consistent for every type under test. In practical terms, it means:

* the C# memory writers and C++ memory readers agree,
* the variant codec is stable enough for nested containers,
* the command round-trip works end to end,
* and the shared memory protocol is trustworthy enough to use for real API calls.

That is a strong signal that the core interop layer is working as intended.
