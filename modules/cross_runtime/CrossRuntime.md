# CrossRuntime documentation

## Purpose

This project is a cross‑runtime simulation and rendering pipeline built around a shared WASM memory contract.

The goal is to let three different layers operate on the same entity state without translating it through heavy object‑based APIs:

- C++ inside the Godot module initializes the shared memory and forwards simulation ticks.
- C# compiled to browser‑wasm performs the actual per‑frame entity update logic.
- JavaScript acts as the glue layer that copies bulk memory between the two runtimes.
- Godot reads the same WASM heap and renders the entities as dots (or any visual you choose).

The core idea is that entity state is stored as raw bytes at fixed offsets. Every runtime agrees on the same layout. That is the contract.

This setup exists to work around the long‑standing lack of supported C# web export templates in Godot 4.x and to keep the whole system web‑native through browser‑wasm.

## What the whole system does

At runtime the system behaves like this:

1. A schema defines the memory layout for one struct and the global offsets used by the simulation.
2. A generator writes C++, C# and test files from that schema.
3. Godot starts in the browser and exposes its WASM heap.
4. A browser worker loads .NET browser‑wasm, initializes JS interop, and fills entity memory.
5. The worker sets a ready flag at a fixed memory offset.
6. The C++ bridge forwards tick calls into the C# logic.
7. The C# logic bulk‑reads the entity block, updates positions and velocities, then bulk‑writes the result back.
8. The Godot renderer reads the same memory and draws the entity instances.

The simulation moves as contiguous memory.

## Project goal

The practical goal is to provide a simple workflow for a web‑exported Godot project with C# simulation logic, even though the standard Godot C# web export path is missing upstream support.

The architecture is built to be:

- deterministic,
- easy to regenerate from one schema file,
- easy to test for memory correctness,
- efficient enough to handle many entities,
- stable across C++, C#, JS and Godot rendering.

## Problem solved

The original blocker was the absence of Godot C# web export templates. The upstream issue documented that support was unavailable at the time, so the usual export pipeline could not produce the expected C# web artifacts.

This setup bypasses that limitation by:

- building the C# side directly with `dotnet publish` for `browser‑wasm`,
- copying the resulting `_framework` output into the Godot web bundle,
- using a custom `host.html` and `interop.js` bridge,
- running the simulation in a browser worker,
- using the Godot WASM heap as the shared memory space.

The result is a working end‑to‑end browser build even though the default C# web export route is not available.

## Folder layout used in this setup

The module lives inside the Godot source tree at `godot/modules/cross_runtime`. The important locations are:

```
godot/modules/cross_runtime/
  register_types.cpp          -- registers the module and custom node types
  register_types.h
  SCsub                       -- tells SCons how to build the module
  memory_layout.h             -- generated; the memory contract
  tools/                      -- the schema generator
  tests/                      -- standalone memory contract tests
  cs/                         -- C# side (interop, contract, logic template)
  DotGame/                    -- ready‑made Godot project that uses the module
```

After you build the C# worker and export the Godot web build, the browser bundle lives in:

```
~/godot/bin/.web_zip/
  godot.js
  godot.wasm
  dot.pck         (compiled from the DotGame project)
  host.html
  server.py
  cs/
    interop.js
    _framework/
```

## The DotGame example

Inside the module you will find a folder called `DotGame`. This is a minimal Godot project that demonstrates how to use the cross‑runtime pipeline. Its main scene (`main.tscn`) contains a single node of type `DotRenderer`. `DotRenderer` is a custom node registered by the module; it reads entity positions directly from the shared WASM heap and renders them with a single MultiMesh draw call.

You can open `DotGame` in the Godot editor after building the engine with the module enabled. Use it as a reference or as a starting point for your own game. The project includes all necessary assets and a simple screen layout.

Because the DotGame is a normal Godot project, you will compile it into a `.pck` file when you export the web build. The exported `.pck` is placed in `~/godot/bin/.web_zip/godot.pck` and loaded by the engine at runtime.

## Schema first workflow

The schema is the source of truth. Everything else is generated from it.

A schema defines:

- the struct name,
- how many entities exist,
- the base offset of the entity block inside the WASM heap,
- the worker‑ready flag offset,
- the field list and field defaults.

Example schema:

```json
{
  "struct_name": "Entity",
  "entity_count": 1000,
  "base_offset": 4096,
  "worker_ready_offset": 32768,
  "fields": [
    { "name": "x",  "type": "f32", "default": 100.0 },
    { "name": "y",  "type": "f32", "default": 100.0 },
    { "name": "vx", "type": "f32", "default": 100.0 },
    { "name": "vy", "type": "f32", "default": 0.0 }
  ]
}
```

### Meaning of the schema fields

- `struct_name`: the generated C++ struct name and the conceptual entity type.
- `entity_count`: number of entities allocated in the memory block.
- `base_offset`: byte offset where the entity array starts inside the WASM heap.
- `worker_ready_offset`: byte offset used for a readiness flag.
- `fields`: ordered list of float fields in the struct.

### Why the worker‑ready offset matters

The renderer must not read the entity block before the worker has fully initialized it. The ready flag is the synchronization point.

A typical flow is:

- ready flag = 0 during startup,
- worker fills the entity buffer,
- worker writes 1 to the ready offset,
- renderer begins reading and drawing.

## Generated memory contract

The generator writes `memory_layout.h` from the schema. That file defines the contract in C++ terms.

The essential output includes:

- `ENTITY_COUNT`
- `ENTITIES_OFFSET`
- `ENTITY_STRIDE`
- `WORKER_READY_OFFSET`
- the struct definition
- per‑field byte offsets

The entity layout is 16 bytes for four `f32` values:

- `x` at offset 0
- `y` at offset 4
- `vx` at offset 8
- `vy` at offset 12

This is what keeps all runtimes aligned.

## Generated code files

The generator writes these files:

- `memory_layout.h`
- `bridge.cpp`
- `Bridge.cs`
- `tests/<StructName>_memory_contract_tests.cpp`

The tests are generated from the same schema so that every defined struct has a matching correctness test.

## What each generated file does

### `memory_layout.h`

Defines the byte‑level memory contract.

### `bridge.cpp`

Initializes entity memory, defines the C++ tick entry point, and forwards each frame into the JS bridge.

### `Bridge.cs`

Defines the C# host interop and the simulation logic.

### `tests/<StructName>_memory_contract_tests.cpp`

Checks that the schema contract is correct, the offsets match the struct layout, the stride is correct, and the worker‑ready flag does not overlap the entity storage.

## How generation works

The generator program reads `schema.json` and produces the output files.

The current C++ generator logic contains these core pieces:

- `load_schema()` reads and validates the JSON.
- `compute_offsets()` assigns byte offsets to fields.
- `gen_memory_layout_h()` writes the contract header.
- `gen_bridge_cpp()` writes the C++ bridge.
- `gen_cs_template()` writes the C# template.
- `gen_memory_contract_tests_cpp()` writes the correctness tests.

The important constraint is that the generator does not invent layout. It derives layout from the schema only. The generated files should not be edited manually; they are derived artifacts.

## Full setup guide

Here is the exact order of operations to go from a schema to a running browser build. This guide assumes you have already built Godot with the cross_runtime module enabled (the module is present in `modules/cross_runtime` and compiled into the engine). If you are using a pre‑built Godot with the module, you can skip the engine build step.

### 1. Define your schema

Create or edit the schema file located at:

```
godot/modules/cross_runtime/tools/schema.json
```

This is the single source of truth. Every other file will be generated from it.

### 2. Run the generator

The generator lives in `tools/generate_code`. Execute it however you normally would (for example, compile and run the generator program). It will read the schema and write the following generated files into the appropriate locations:

- `memory_layout.h` (module root)
- `bridge.cpp` (module root, or wherever you place your bridge)
- `Bridge.cs` (into `cs/`)
- a test file in `tests/`

Make sure the generator runs successfully. If you change the schema later, re‑run the generator to keep everything in sync.

### 3. Write your C++ game logic using the memory contract

Now you have `memory_layout.h`. This header gives you the entity struct, the offsets, the stride and the entity count. You can write your own renderer and bridge (or adapt the ones provided in the DotGame example).

The DotGame folder contains reference implementations of `dot_renderer.h`, `dot_renderer.cpp` and `bridge.cpp`. You can study them to understand how to read the entity buffer and how to draw entities with MultiMesh. If you want to use the DotGame as‑is, you will need to compile those source files into the engine by uncommenting the appropriate lines in the module's `SCsub`. However, the recommended approach is to write your own renderer and bridge that fit your game's style, using the DotGame files as a pattern.

### 4. Build the Godot web export (including your game project)

You need to produce a complete web build that contains the Godot engine, your game logic, and all assets. Here is how you do that.

First, make sure you have a Godot project that uses your custom renderer node. The DotGame project inside the module is already set up for this. Put your assets, scenes, and any additional Godot scripts into that project (or create a new one that references your module's node types).

Then, build the Godot engine for the web target. For example:

```bash
cd ~/godot
scons platform=web target=template_release -j4
```

This will produce the engine files (godot.js, godot.wasm) and place them in the output directory.

Next, export your Godot project as a `.pck` file. You can do this by opening the project in the Godot editor (built with the module) and using the Export menu, or by running the engine with the `--export` flag. The resulting `godot.pck` should be placed in the web bundle folder:

```
~/godot/bin/.web_zip/
```

At this point the `.web_zip` directory should contain `godot.js`, `godot.wasm`, `godot.pck`, and the `host.html` file that you provide (the module comes with a ready‑to‑use host.html). Copy the module's `host.html` and `server.py` into this folder as well.

### 5. Write your C# simulation logic

Now turn to the C# side. The generator created a `Bridge.cs` file inside `cs/`. This file contains the bulk read and write methods, the contract constants, and the skeleton of the simulation loop. Open the file and fill in the `UpdateEntities` method (and `InitEntities` if you want custom initial data). You can also rename the file and adjust the namespace as you wish.

The DotGame example includes a complete C# simulation (gravity, wind, bounce, etc.) that you can borrow or modify. The important point is that your C# code reads and writes the entity state using the same offsets and strides defined in the generated `Contract` class.

### 6. Build the C# browser‑wasm package

The module provides a cross‑platform Python script `build.py` inside the `cs/` folder. This script replaces the old shell script and works on Linux, macOS and Windows without any changes. It:

- It cleans old build output.
- It runs `dotnet publish` with the correct settings for browser‑wasm.
- It copies the resulting `_framework` directory into the web bundle folder (`~/godot/bin/.web_zip/cs/_framework`).

To use it, simply run:

```bash
cd ~/godot/modules/cross_runtime/cs
python build.py
```

If your web bundle is in a non‑standard location, you can set an environment variable:

```bash
# On Linux / macOS
export GODOT_WEB_DIR=/path/to/my/web_bundle
python build.py

# On Windows (Command Prompt)
set GODOT_WEB_DIR=D:\my\web_bundle
python build.py
```

The script also copies `interop.js` into the bundle if it finds it in the same directory.

### 7. Serve the web bundle correctly

The provided `server.py` is important because it sets the browser security headers needed for threaded WebAssembly.

```python
#!/usr/bin/env python3
import http.server
import socketserver

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def guess_type(self, path):
        if path.endswith(".wasm"):
            return "application/wasm"
        if path.endswith(".pck"):
            return "application/octet-stream"
        return super().guess_type(path)

    def send_response_only(self, code, message=None):
        super().send_response_only(code, message)
        if self.path.endswith(('.wasm', '.js', '.pck')):
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')

with socketserver.TCPServer(("", 8000), Handler) as httpd:
    print("Serving at http://localhost:8000")
    httpd.serve_forever()
```

Run it from the web bundle directory:

```bash
cd ~/godot/bin/.web_zip
python server.py
```

The `Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy` headers are required for cross‑origin isolation, which is needed for the threaded WASM path. Without them the engine may fail to start the worker.

### 8. Open the exported host page

Now open your browser and navigate to `http://localhost:8000/host.html`. The host page will load Godot, acquire the WASM heap, start the .NET worker, and coordinate the ready flag. Once the worker has initialized the entity buffer, the renderer will begin drawing the dots and you should see the simulation running at a smooth 60 fps.

## Runtime startup sequence

The browser host page performs the following:

1. loads `godot.js`,
2. starts the Godot engine,
3. acquires the heap as `window.godotHeapF32`,
4. creates a web worker,
5. loads .NET inside the worker,
6. calls `Host.InitInterop()`,
7. calls `GameLogic.InitEntities()`,
8. sets the worker‑ready flag,
9. starts the worker simulation loop,
10. hides the overlay and allows rendering to proceed.

This is the main runtime handshake.

## Interop layer

The JavaScript file `cs/interop.js` is the low‑level memory copier.

It exposes:

- `bulkRead(srcOffset, destArray, length)`
- `bulkWrite(srcArray, destOffset, length)`

It uses typed array views on the Godot heap and copies bytes directly. This means the C# side can move full entity blocks efficiently instead of calling across the bridge per field.

## C# runtime behavior

The C# side uses the same contract as the C++ side.

Its job is to:

- bulk read entity data from the heap,
- interpret bytes as float fields,
- update positions and velocities,
- apply motion, gravity, damping, wind, turbulence, and boundary reflection,
- bulk write the updated entity data back.

The C# runtime also includes `InitEntities()` to seed the entity block with deterministic values.

## C++ bridge behavior

The C++ bridge exists for two reasons:

- it owns the initial shared memory shape,
- it provides the exported entry point that calls into the JS bridge each frame.

The bridge can seed entities with default or random data, and then forward each tick into the JS/C# side.

## Godot renderer behavior

The renderer node reads the shared memory directly and uses `MultiMesh` to draw many dots efficiently.

The rendering path is:

- wait for the ready flag,
- read the entity block,
- update instance transforms,
- issue a MultiMesh draw call.

This keeps the rendering cost low.

## Memory contract correctness tests

Every defined struct should have a correctness test.

The generated test file lives in:

```
godot/modules/cross_runtime/tests/
```

and is self‑contained. The tests are compiled as a standalone native binary (not as part of the engine) and run as part of the build. If they fail, the build stops immediately.

The tests verify:

- `sizeof(float) == 4`
- `sizeof(Entity) == ENTITY_STRIDE`
- each `offsetof(Entity, field)` matches the generated offset constant
- the entity base offset is correctly aligned
- the worker‑ready offset is aligned
- the worker‑ready offset does not overlap the entity block
- raw memory round‑trips through the struct layout
- entity index math matches `base + index * stride`

If you ever change the schema and regenerate the sources, run the build again. A failing test means a layout mismatch that would silently corrupt your simulation. This gate keeps the contract honest.

## Current performance notes

My demo fps measurements from a Lenovo 11e (Intel i3 7th gen, integrated HD 620 graphics):

- 1000 entities: 60 fps
- 500 entities: 60 fps
- 300 entities: 60 fps

On more modern hardware the numbers are probably higher. The system stays within the target frame budget at these entity counts because of:

- bulk memory copies instead of per‑entity interop,
- direct byte layout instead of object serialization,
- MultiMesh rendering instead of many separate draw calls,
- a worker‑based simulation loop instead of blocking the main page.

## Why this bypasses the historical web export limitation

This project does not depend on the missing C# web export template that would normally embed the .NET runtime inside the engine. Instead it:

- publishes the C# code to browser‑wasm independently,
- places the runtime artifacts into the web bundle manually,
- uses a custom HTML host and a worker‑based boot path,
- shares memory through the Godot WASM heap.

That is the core workaround.

## Typical usage order

After you have built Godot with the module included, the practical order for working on the project is:

1. Edit `schema.json`.
2. Run the generator.
3. Write your C++ renderer and bridge (following the DotGame example) using `memory_layout.h`.
4. Build the Godot web export (engine + your game’s `.pck`).
5. Write your C# simulation logic inside the generated `Bridge.cs`.
6. Run `python build.py` from the `cs/` directory.
7. Start the server with `python server.py` from the web bundle.
8. Open `http://localhost:8000/host.html` in a browser.

Run the generated memory contract tests whenever the schema changes.

## Troubleshooting notes

### Heap not available

If the interop file throws `Godot heap not available for interop`, then the engine heap was not acquired before the worker or bridge tried to use it. Make sure the host.html script runs `acquireHeap()` at the right time.

### Worker not ready

If the renderer stays blank, check the ready flag and the worker startup path. The worker must set the flag after calling `InitEntities()`.

### Wrong offsets or corrupted motion

If entities behave incorrectly, the first suspect is a schema mismatch between generated C++, generated C#, and the JavaScript constants used in the host page. Re‑generate everything and ensure the `_framework` folder was copied fresh.

### Web features missing

If the page complains about missing features, your browser may not support the required threading or isolation features. Make sure you are using a modern browser and that the server is sending the correct headers.

### Stale framework files

If C# changes do not appear in the browser, delete the old `_framework` folder inside `~/godot/bin/.web_zip/cs/` and run `build.py` again.

## Final mental model

The whole system is one shared memory contract with multiple runtimes attached to it.

- The schema defines the bytes.
- The generator makes all code agree on those bytes.
- The browser host starts Godot and the worker.
- The worker loads C# and updates the entities.
- The renderer visualizes the same memory.
- The tests prove the memory layout is still correct.

That is the complete design.