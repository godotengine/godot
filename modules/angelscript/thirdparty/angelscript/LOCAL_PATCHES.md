# Local AngelScript 2.38 patches

## Native virtual-property validation (2026-10-14)

`source/as_builder.cpp`, `asCBuilder::ValidateVirtualProperty`: when registering an application-object system method (not a script object), search the owner's already-registered method IDs for its getter/setter counterpart instead of scanning every function in the engine. Retain all name, property flag, module, namespace and object-type predicates and existing validation/error codes. Script objects and globals retain the original whole-engine search because script declarations may not yet be present in the method list.

Reason: full Godot ClassDB registration replicates inherited interfaces across types. The original per-property whole-engine search becomes quadratic: a syntax-error fallback exceeded 180 seconds on MSVC x64; with this change the same existing test plus recovery completes in about 3.7 seconds, and the full 192-case AngelScript suite in about 11.7 seconds. These wall times include startup and are diagnostic, not fixed performance guarantees.

Regression coverage: `modules/angelscript/tests/test_angelscript_property_registration.{h,cpp}` checks getter/setter registration order, mismatch rejection, same name on different types, indexed properties, globals, script properties and registration scaling; the existing syntax-error test now also verifies subsequent valid compilation.

This changes registration-time lookup only, not bytecode execution, object lifetime, native calling conventions or public API. Re-evaluate and rebase the patch on any upstream AngelScript upgrade.
