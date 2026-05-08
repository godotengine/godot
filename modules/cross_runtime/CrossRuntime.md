# CrossRuntime Module with C# Web Support for Godot
This project implements a cross-runtime architecture for Godot C# web exports using:

1. Godot WASM
2. a minimal JavaScript bridge
3. a C# runtime running in a separate WASM environment

The goal is to let C# code interact with Godot through a generated API that feels familiar to C# game developers, while the actual communication happens through a fixed shared memory contract.

The project also has a browser hosting layer and a .NET worker layer. The main thread runs Godot Engine while the worker owns the C# runtime and the shared heap connects them.

The two runtimes exchange commands and data through a shared region of memory and Godot executes those commands at frame time.

> Note: this setup contains 2 WASM runtimes, Godot WASM and .NET WASM.

## Overview
At a high level, the flow is:
```text
C# runtime
  -> writes command ids and data when needed into shared memory
  -> waits for completion

Godot runtime
  -> reads command id from shared memory
  -> processes command corresponding to that id
  -> writes result back into shared memory if needed
  -> marks command as done which goes on to signal C# that it is done
```

This gives the C# side a desktop-style API surface while the internals remain explicit and predictable.

## Folder Layout
The module lives inside Godot as an optional module at `godot/modules/cross_runtime`.

```text
godot/modules/cross_runtime/
  cpp/
    headers/
      bridge_api.h            - command ids and memory offsets for reads and writes
      bridge_helpers.h        - helpers for reading and writing data structures
    *cpp files                - Godot API implementation files

  cs/
    build.py                  - builds the cs project
    CoreTypes.cs              - core data structures used by Godot
    Interop.cs                - exports core read/write functions to JS and imports some from JS
    Program.cs
    WebGodotApi.csproj
    GodotApi/                 - generated API files, C# equivalents of the cpp folder
      *cs files
    Utilities/
      Commands.cs              - C# equivalent of bridge_api.h
      CoreTypes.cs             - core data structures used by Godot
      Helpers.cs               - general C# read/write helpers and bridge utilities
      VariantHandling.cs       - variant codec and dispatch layer for write variant and read variant

  api_generator/
    Metadata.py
    Utilities.py
    cpp_generator.py
    cs_generator.py
    Main.py
    FindBindings/
      dump.gd

  command_dispatcher.cpp      - processes commands
  command_processor.cpp       - runs command processing each frame
  config.py                   - build configuration
  register_types.cpp          - registers command processing
  register_types.h
  SCsub                       - build instructions

  web/
    interop.js                - JS module imported by C#
    dotnet_worker.js          - worker that hosts .NET and heap access
    main.js                   - browser bootstrap that starts Godot and the worker
```

## Execution Flow
1. Godot Engine initializes on the main thread.
2. The C# worker initializes.
3. Through C#, you are able to control and manage what the engine does.
4. C# sends commands and data through the bridge.
5. The STATUS flag controls when C# waits and when Godot can proceed.
6. Godot processes the commands(it specifically processes the command associated with the command ID it recieved), the data and does the rendering.

That is the basic interplay in the project.

# Memory Layout
The bridge uses a fixed memory contract. Both runtimes agree on exact offsets and meaning.

## Core Bridge Layout

```text
Offset 0x00005000 -> Command ID
Offset 0x00005004 -> Status
Offset 0x00005008 -> Payload begin
```

In C++:
```cpp
const std::uint32_t CMD_OFFSET    = 0x5000u;
const std::uint32_t STATUS_OFFSET = 0x5004u;
const std::uint32_t CMD_DATA      = 0x5008u;

const std::uint32_t CMD_NONE       = 0;
const std::uint8_t  STATUS_PENDING = 0;
const std::uint8_t  STATUS_DONE    = 1;
const std::uint32_t CMD_AESContext_start__2__29__29__r2 = 2;
```

In C#:
```csharp
public const int CMD_OFFSET    = 0x5000;
public const int STATUS_OFFSET = 0x5004;
public const int CMD_DATA      = 0x5008;

public const int  CMD_NONE       = 0;
public const byte STATUS_PENDING = 0;
public const byte STATUS_DONE    = 1;
public const int CMD_AESContext_start__2__29__29__r2 = 2;
```

## Meaning of the Offsets
1. `CMD_OFFSET` 
- This stores the command identifiers. When C# wants Godot to do something, it writes the command ID there of a function, like for example, for this: `const std::uint32_t CMD_AESContext_start__2__29__29__r2 = 2;`, it will specifically write command 2. Then in a frame, godot will read command two together with the data that has been sent from C# side via the shared memory. It will then use that and access the method and process it. If something is to be returned bback to C#, it is written back to the cmd_data offset which is then read in C#.

2. `STATUS_OFFSET` (This one is always updated atomically to avoid memory races)
- This stores the signal that tells whether a process is finished or in progress/ Has a command been fully processed or not? There is `STATUS_PENDING or STATUS_DONE.`
- Once Godot completes processing  something, what it does is it updates the status offset setting it to done, this process makes the busy/wait poll in C# stop and C# proceeds to the next step in its execution.
- The next time it wants to communicate with the engine, through its `SendCommand` function, it will once again reset the STATUS to PENDING.

3. `CMD_DATA` 
- This is the payload offset. Its where any data that needs to be exchanged is stored. The type of payload for each function is typically known at runtime except for thr Variant type. For the Variant type, since C# specifically had limitations and gaps in its capacity to handle it, `cs/Utilities/VariantHandlers.cs` was setup which specifically helps decode/encode variant types and read and write them correctly into the memory.

4. `CMD IDs`
- These are integer IDs representing all possible commands that were able to be extracted. They help with identification of what needs to be processed in both runtimes.

## Variant Handling
For our system, specifically the C# side, to be able to handle Variants, I built  the Variant Handler which follows the conventions of Godot Variants handling'. It simple decodes or encodes variants to send. This is found in `cs/Utilities/VariantHandlers.cs`. 

For the rest of the serialization/deserialization, you can find them in `bridge_helpers.h` for C++ and `Helpers.cs` for C#. They contain the needed read and writes that will be writing to and reading from the offsets.

## Command Lifecycle Summary
```text
C# writes payload
C# writes command ID
C# marks status as pending
Godot sees pending status
Godot reads command
Godot executes the handler
Godot writes result
Godot marks status as done
C# resumes
```

> Note: Godot processes a command in each frame. See in `cross_runtime/command_processor.cpp`

## Generation and Reproducibility
The project uses generated bindings so the full GodotSarp API can be presented in C++ and C# without manual duplication. 
Belw are the steps to generate the full api

### 1. Download the Godot Linux editor for .NET

Use a Godot build that includes the Mono and .NET support needed for the project.

### 2. Create a dump project

Create a folder named `godot_api_dump` and put a basic `project.godot` file inside it:

```godot
[application]
config/name="API Dump"
```

### 3. Generate `api.json`
- This will run the dump_api.gd and generate all the classdb information that becomes crucial in api generation.

Run:

```bash
path/to/Godot_v4.6.2-stable_mono_linux_x86_64 --headless --path path/to/godot_api_dump --script godot/modules/cross_runtime/api_generator/dump_api.gd > api.json
```
- User the version you have. This step has not been tested on Windows or Mac.

### 4. Run the generator

```bash
cd godot/modules/cross_runtime/api_generator
python3 main.py path/to/api.json <destination_folder>
```

### 5. Generated output

The destination folder will contain:

1. `command_dispatcher.cpp`
2. the `cpp` folder with `headers/bridge_api.h` and `headers/bridge_helpers.h`
3. the generated C++ API files
4. the `cs` folder with `Commands.cs` and the `GodotApi` folder

# Synchronization Model

The synchronization model is based on a simple shared memory state machine.

### States

- `STATUS_PENDING`
- `STATUS_DONE`

### Behavior

- C# writes a command and then waits.
- Godot processes commands during the frame step.
- The bridge uses atomic read and write helpers for control fields.
- C# performs a busy-wait loop until Godot marks completion.

### Busy-wait behavior

The wait loop is intentionally simple:

```csharp
while (Interop.AtomicReadInt32(Commands.STATUS_OFFSET) != Commands.STATUS_DONE)
{
    Thread.SpinWait(1);
}
```

The moment C++ is done processing a command, it sets the status to 1. This in turn signals the end of the wait so C# knows it can proceed.

### Atomic usage
Atomic reads and writes are used for control fields so the bridge state is visible across both runtimes in a consistent way.

### Frame Timing
The command processor runs each frame. This means that in every frame a command will be be processed.

## Processing Loop

```text
C# produces work
Godot processes work during frame
Result becomes visible
```

The timing model should remain explicit, because this is where most subtle bridge bugs appear.

# Web Setup and Runtime Housing
This is the browser side of the project.
The main points are:
- the main thread owns the Godot WASM runtime and the Heap
- the worker owns the .NET WASM runtime
- the shared heap is handed from the main thread into the worker
- the worker installs the memory access shims used by C# interop

```text
main thread
  -> starts Godot
  -> acquires Godot heap
  -> creates worker
  -> passes heap buffer into worker

Worker
  -> receives heap buffer
  -> installs bulk read and bulk write helpers
  -> installs atomic read and atomic write helpers
  -> loads .NET runtime
  -> initializes interop
  -> starts the C# game loop
```

## Main thread script
The browser entry point does the following:
1. Creates the Godot engine configuration
2. Checks required features such as threads
3. Starts the Godot engine
4. Obtains `HEAPU8`
5. Creates the worker
6. Passes the heap buffer to the worker
7. Hides the loading overlay once everything is live

## Worker Responsibilities
The worker owns the .NET runtime lifecycle.
It receives:
- the shared heap buffer
- the `dotnet.js` URL
- the ready flag offset

Then it does the following:
1. Verifies that the buffer is an `ArrayBuffer` or `SharedArrayBuffer`
2. Builds a `Uint8Array` view over the heap
3. Exposes heap helpers on `self`
4. Loads the .NET runtime
5. Reads the assembly exports
6. Calls `Interop.InitInterop()`
7. Sets the ready flag
8. Calls `Interop.RunGame()`
The worker is the place where C# begins executing.

## Heap Attachment
The worker stores the Godot heap on `self` so the interop layer can access it later.
```javascript
self.__heapBuffer = heapBuffer;
self.__heapU8 = new Uint8Array(heapBuffer);
```
This is the shared memory handle used by the bridge.

## Bulk Read
`bulkRead` copies bytes out of the Godot heap and returns a new typed array slice.
- input: source offset and length
- output: byte array view containing the copied data
- used by C# reads through `Interop.BulkRead` found in `cs/Interop.cs`
- used by helper functions such as `ReadInt32`, `ReadVector3`, `ReadString`, and the packed array readers

## Bulk Write
this copies bytes from a source array into the Godot heap.
- input: source bytes, destination offset, requested length
- output: no direct return value
- used by C# writes through `Interop.BulkWrite`
- used by helper functions such as `WriteInt32`, `WriteVector2`, `WritePackedByteArray`, and the variant one

The worker implementation clamps the write length to the source length and the heap size, then writes into the heap with `heapU8.set(...)`.

## Atomic Reads and Writes
The bridge uses atomics for shared control fields, especially status reads and writes.
### Atomic write
```javascript
self.__atomicWriteInt32 = function (offset, value) {
    const view = new Int32Array(self.__heapBuffer, offset, 1);
    Atomics.store(view, 0, value);
};
```

### Atomic read
```javascript
self.__atomicReadInt32 = function (offset) {
    const view = new Int32Array(self.__heapBuffer, offset, 1);
    return Atomics.load(view, 0);
};
```

## interop.js

`interop.js` is the module imported by C# through `JSHost.ImportAsync`.

It exposes the JavaScript bridge functions that C# calls indirectly through `Interop`.

### Exports

- `bulkRead(srcOffset, length)`
- `bulkWrite(srcArray, destOffset, length)`
- `atomicWriteInt32(offset, value)`
- `atomicReadInt32(offset)`

## cs/Interop.cs

### What it does
- Imports JavaScript functions through `JSImport`
- Exports methods that JavaScript calls through `JSExport`
- `InitInterop()` loads the bridge module
- `RunGame()` resets the command buffer and starts the C# runtime entry point

# Generation and Reproduction

This document explains how the API metadata is extracted, how the generator works, and how to reproduce the project output.

---

## API Generation

The project uses generated bindings so the full Godot API can be represented in C# and C++ without manual duplication.

---

## How the metadata is extracted

The `dump.gd` script reads the Godot class database and emits metadata as JSON.

```gdscript
extends SceneTree

func _init():
    var classes = ClassDB.get_class_list()
    classes.sort()
    var output = []
    var singletons = Engine.get_singleton_list()

    for cls in classes:
        var cls_name := String(cls)
        var methods = ClassDB.class_get_method_list(cls_name, false)
        var is_singleton = singletons.has(cls_name)

        for method in methods:
            var ret = method.get("return", {})
            var entry = {
                "class": cls_name,
                "name": String(method.get("name", "")),
                "return_type": int(ret.get("type", 0)),
                "args": [],
                "static": false,
                "singleton": is_singleton
            }

            for arg in method.get("args", []):
                entry["args"].append({
                    "name": String(arg.get("name", "")),
                    "type": int(arg.get("type", 0))
                })

            output.append(entry)

        if is_singleton:
            output.append({
                "class": cls_name,
                "name": "get_singleton",
                "return_type": 24,
                "args": [],
                "static": true,
                "singleton": true
            })

    print(JSON.stringify(output, "\t"))
    quit()
```

This produces `api.json`, which becomes the source for code generation.

---

## Generator Output

The Python generator produces:

- `command_dispatcher.cpp`
- the `cpp/headers` folder
- generated C++ API files
- generated C# API files
- command mappings
- handler tables

---

## Reproduction Steps

### 1. Download the Godot Linux editor for .NET

Use a Godot build that includes the Mono and .NET support needed for the project.

### 2. Create a dump project

Create a folder named `godot_api_dump` and put a basic `project.godot` file inside it:

```godot
[application]
config/name="API Dump"
```

### 3. Generate `api.json`

Run:

```bash
path/to/Godot_v4.6.2-stable_mono_linux_x86_64 --headless --path path/to/godot_api_dump --script godot/modules/cross_runtime/dump_api.gd > api.json
```

### 4. Run the generator

```bash
cd godot/modules/cross_runtime/api_generator
python3 main.py path/to/api.json <destination_folder>
```

### 5. Generated output

The destination folder will contain:

1. `command_dispatcher.cpp`
2. the `cpp` folder with `headers/bridge_api.h` and `headers/bridge_helpers.h`
3. the generated C++ API files
4. the `cs` folder with `Commands.cs` and the `GodotApi` folder

---

## Workflow Summary

The general workflow is:

```text
extract metadata
generate code
build module
run bridge
validate command flow
```

This keeps the API surface synchronized between the C++ and C# sides.