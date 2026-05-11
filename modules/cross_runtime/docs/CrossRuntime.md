# CrossRuntime Module

An optional Godot module that enables C# game logic to run in a Web Worker alongside Godot on the web. The two runtimes communicate through a fixed shared memory contract over `SharedArrayBuffer`.

> This module contains two WASM runtimes: Godot WASM on the main thread and .NET WASM in a worker.

---

## How it works

C# writes a command ID and arguments into shared memory, then waits. Godot reads the command once per frame, executes the corresponding engine method, writes the result back, and signals completion. C# resumes and reads the result.

The three control offsets are:

```text
0x5000  CMD_OFFSET     command ID
0x5004  STATUS_OFFSET  completion flag
0x5008  CMD_DATA       payload
```

Both runtimes use the same constants. `STATUS_OFFSET` is always written atomically.

---

## Folder layout

```
modules/cross_runtime/
  cpp/
    headers/
      bridge_api.h        # command IDs and offsets
      bridge_helpers.h    # read/write helpers
    *.cpp                 # generated Godot API handlers

  cs/
    Interop.cs            # JSImport/JSExport bridge
    Program.cs
    WebGodotApi.csproj
    GodotApi/             # generated C# API
    Utilities/
      Commands.cs         # C# mirror of bridge_api.h
      CoreTypes.cs
      Helpers.cs
      VariantHandling.cs  # Variant codec

  api_generator/
    Main.py
    Metadata.py
    Utilities.py
    cpp_generator.py
    cs_generator.py
    FindBindings/
      dump.gd

  tests/
    run_tests.py
    C++/
      tests_caller.cpp
      headers/
        bridge_helpers.h
        tests.h
    NET/
      Tests.cs
      Tests.csproj
      Utilities/

  docs/
    CrossRuntime.md
    Tests.md
    HowToReproduce.md
    RunningTests.md

  command_dispatcher.cpp
  command_processor.cpp
  register_types.cpp
  register_types.h
  config.py
  SCsub
```

---

## Generating the bindings

The C++ and C# bindings are generated from Godot's ClassDB. You need the Godot Linux editor for .NET to run this step.

**1. Create a minimal dump project**

```
godot_api_dump/project.godot:

[application]
config/name="API Dump"
```

**2. Generate api.json**

```bash
path/to/Godot_v4.x-stable_mono_linux_x86_64 \
  --headless \
  --path path/to/godot_api_dump \
  --script modules/cross_runtime/api_generator/FindBindings/dump.gd > api.json
```

**3. Run the generator**

```bash
cd modules/cross_runtime/api_generator
python3 Main.py path/to/api.json <destination>
```

**4. Output**

```
<destination>/
  command_dispatcher.cpp
  cpp/headers/bridge_api.h
  cpp/headers/bridge_helpers.h
  cpp/*.cpp
  cs/Commands.cs
  cs/GodotApi/*.cs
```

**5. Update SCsub after placing generated files**

```python
Import("env")
cpp_sources = Glob("cpp/*.cpp")
env.add_source_files(env.modules_sources, [
    "register_types.cpp",
    "command_dispatcher.cpp",
    "command_processor.cpp",
] + cpp_sources)
```

---

## Building

```bash
scons platform=web
```

The module is optional and can be excluded at build time through the standard SCons module flag.

---

## Tests

The test suite validates that the C# and C++ sides agree on the binary layout for all supported types. See `docs/Tests.md` for coverage details and `docs/RunningTests.md` for instructions.

---

## Full demo and web hosting layer

The browser hosting layer, worker script, and a working demo are maintained separately at:

[https://github.com/tommygrammar/godot-cross-runtime-demo](https://github.com/tommygrammar/godot-cross-runtime-demo)

---

## Known constraints

- One command in flight at a time.
- One frame of latency per command on the Godot side.
- Requires `SharedArrayBuffer` support in the browser environment.
