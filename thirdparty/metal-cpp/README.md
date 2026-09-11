# metal-cpp

Drop-in replacement for `metal-cpp` that dispatches via linker-synthesized selector stubs (`_objc_msgSend$<sel>`) instead of the `sel_registerName` + `objc_msgSend` indirection used by Apple's upstream metal-cpp. Same public API surface so Godot's `drivers/metal/` can switch without source-level changes.

## Benefits

### API Availability warnings

* The generated C++ classes and methods are annotated with `availability` attribtues, so you get compile-time warnings if you call an API that may not be present on the target OS version.

### objc_msgSend stubs

See: https://developer.apple.com/videos/play/wwdc2022/110363

* metal-cpp loses an Objective-C (m/mm) optimisation introduced in Xcode 14 where `objc_msgSend` call sites were transposed into a single `bl` to a linker-generated stub. This reduces code at the call site from 12 → 4–8 bytes on ARM64.
* This version reintroduces that, which now results in a tail call for better i-cache/branch behavior.
* No static initializer fan-out. Apple's metal-cpp emits `SEL s_k<sel> = sel_registerName("<sel>")`; per selector; dyld runs all of them at image load. Stubs are pre-resolved by dyld's existing ObjC fixup pass.
* Pre-dedup'd through the linker. All call sites for label across all classes funnel through one stub address — better i-cache, single relocation entry.

### Binary footprint

* No SEL globals. Each removed s_k<sel> is 8 B of .data + a relocation + an initializer call. We had hundreds.
* No per-class inline `Object::sendMessage<Ret>(...)` instantiations. The bridge collapses (ret, args, sel) tuples to a single extern "C" decl: e.g. the 30+ classes with `label() -> NS::String*` share one declaration. Shrinks .o files, speeds link, dedups debug info, gives cleaner stack frames (`_objc_msgSend$label` instead of `NS::Object::sendMessage<NS::String*, ...>`).


## Requirements

The Python environment must have `libclang` and `pyyaml` installed. The script is tested with Python 3.14.

## Running

```sh
python3 thirdparty/metal-cpp/tools/generate.py --metal-cpp <path-to-apple-metal-cpp> --sdk macos
```
