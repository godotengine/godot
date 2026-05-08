# CrossRuntime Module - Godot C# Web Support

A cross-runtime architecture that lets C# code control a Godot engine instance running in the browser. Godot runs on the main thread as a WASM module; C# runs in a Web Worker as a separate .NET WASM module. They communicate through a shared memory region.

> **Note:** Two WASM runtimes are involved, that is Godot WASM and .NET WASM.

---

## How It Works

```
C# runtime    writes command ID + payload to shared memory
              busy-waits on STATUS

Godot runtime reads command ID each frame
              executes the corresponding handler
              writes result to shared memory
              sets STATUS = DONE, C# resumes
```

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

  command_dispatcher.cpp    # command handler table
  command_processor.cpp     # per-frame command execution
  config.py
  register_types.cpp
  register_types.h
  SCsub
```

---

## Memory Layout

Both runtimes share a fixed memory contract at known offsets.

```
0x5000  CMD_OFFSET     command ID to execute
0x5004  STATUS_OFFSET  STATUS_PENDING (0) or STATUS_DONE (1)
0x5008  CMD_DATA       payload (arguments in, result out)
```

C++ constants:
```cpp
const uint32_t CMD_OFFSET    = 0x5000u;
const uint32_t STATUS_OFFSET = 0x5004u;
const uint32_t CMD_DATA      = 0x5008u;

const uint32_t CMD_NONE       = 0;
const uint8_t  STATUS_PENDING = 0;
const uint8_t  STATUS_DONE    = 1;
```

C# constants:
```csharp
public const int CMD_OFFSET    = 0x5000;
public const int STATUS_OFFSET = 0x5004;
public const int CMD_DATA      = 0x5008;

public const int  CMD_NONE       = 0;
public const byte STATUS_PENDING = 0;
public const byte STATUS_DONE    = 1;
```

### Field Semantics

- **CMD_OFFSET** - C# writes the integer ID of the command it wants Godot to run. Godot reads this once per frame.
- **STATUS_OFFSET** - Signals completion. Always read/written atomically. Godot sets this to `STATUS_DONE` when it finishes; C#'s `SendCommand` resets it to `STATUS_PENDING` before each new command.
- **CMD_DATA** - Payload region. Arguments are written here before the command is issued; results are written back here by Godot. For the `Variant` type, encoding and decoding is handled by `VariantHandling.cs`.

### Command Lifecycle

```
C# writes payload to CMD_DATA
C# writes command ID to CMD_OFFSET
C# sets STATUS = PENDING
Godot reads command ID (next frame)
Godot executes the handler
Godot writes result to CMD_DATA (if any)
Godot sets STATUS = DONE
C# resumes
```

---

## Synchronization

C# busy-waits until Godot marks the command done:

```csharp
while (Interop.AtomicReadInt32(Commands.STATUS_OFFSET) != Commands.STATUS_DONE)
{
    Thread.SpinWait(1);
}
```

Control fields (`CMD_OFFSET`, `STATUS_OFFSET`) use atomic reads and writes to keep both runtimes consistent. The command processor runs once per frame, so every command carries exactly one frame of latency.

---

## Web Setup

### Startup Sequence

```
Main thread
  starts Godot WASM
  obtains HEAPU8 (Godot heap)
  creates dotnet_worker
  posts heap buffer to worker

Worker
  receives heap buffer
  installs bulk read/write + atomic helpers on self
  loads .NET runtime
  calls Interop.InitInterop()
  sets ready flag
  calls Interop.RunGame()
```

### Heap Attachment

The worker stores the Godot heap so the interop layer can reach it later:

```javascript
self.__heapBuffer = heapBuffer;
self.__heapU8     = new Uint8Array(heapBuffer);
```

### JS Bridge Helpers

Installed on `self` inside the worker:

**Bulk read** - copies bytes out of the Godot heap and returns a typed array slice. Used by C# `ReadInt32`, `ReadVector3`, `ReadString`, and packed array readers.

**Bulk write** - copies a source byte array into the Godot heap. Clamps length to `min(src.length, heap.size)`. Used by C# `WriteInt32`, `WriteVector2`, `WritePackedByteArray`, and variant writes.

**Atomic read/write** - used for STATUS and CMD fields only:

```javascript
self.__atomicWriteInt32 = (offset, value) => {
    Atomics.store(new Int32Array(self.__heapBuffer, offset, 1), 0, value);
};

self.__atomicReadInt32 = (offset) => {
    return Atomics.load(new Int32Array(self.__heapBuffer, offset, 1), 0);
};
```

### interop.js

Imported by C# via `JSHost.ImportAsync`. Exports:

- `bulkRead(srcOffset, length)`
- `bulkWrite(srcArray, destOffset, length)`
- `atomicWriteInt32(offset, value)`
- `atomicReadInt32(offset)`

### cs/Interop.cs

- Imports the JS bridge functions via `[JSImport]`
- Exports `InitInterop()` and `RunGame()` via `[JSExport]`
- `RunGame()` resets the command buffer and starts the C# entry point

---

## API Generation

All C++ and C# bindings are generated from Godot's ClassDB to avoid manual duplication.

### Steps

**1. Get Godot for Linux (.NET edition)**

**2. Create a minimal dump project**

```
godot_api_dump/project.godot:
[application]
config/name="API Dump"
```

**3. Generate api.json**

```bash
path/to/Godot_v4.x-stable_mono_linux_x86_64 \
  --headless \
  --path path/to/godot_api_dump \
  --script godot/modules/cross_runtime/api_generator/FindBindings/dump.gd > api.json
```

**4. Run the generator**

```bash
cd godot/modules/cross_runtime/api_generator
python3 Main.py path/to/api.json <destination_folder>
```

**5. Output**

```
<destination>/
  command_dispatcher.cpp
  cpp/headers/bridge_api.h
  cpp/headers/bridge_helpers.h
  cpp/*               # generated C++ API files
  cs/Commands.cs
  cs/GodotApi/*       # generated C# API files
```

### How dump.gd Works

It iterates `ClassDB.get_class_list()`, collects method signatures (name, return type, argument types), and emits them as JSON. Singleton classes also get a synthetic `get_singleton` entry. This `api.json` is the sole input to the Python generator.