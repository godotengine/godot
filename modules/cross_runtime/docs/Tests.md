# Tests and Build Guide

## Tests

The test suite verifies that the C# and C++ sides agree on the binary layout, type encoding, and value interpretation for every supported type.

The C# side writes fixed values into shared memory at known offsets. The C++ side reads them back using the same helpers used in real bridge calls and validates each value in sequence. A mismatch stops the run immediately and reports which type failed.

The offsets are hard-coded on both sides and must remain identical. They are part of the ABI between the two runtimes. If any helper changes its binary format, the test fails immediately rather than silently corrupting data downstream.

**Coverage:**
- scalars: `int32`, `int64`, `float`, `double`, `bool`
- string types: `String`, `StringName`, `NodePath`
- `RID`
- math and transform types: `Vector2`, `Vector3`, `Rect2`, `Basis`, `Quaternion`, `Transform3D`, `Projection`, and their integer variants
- packed arrays: all nine types including `PackedStringArray`
- recursive containers: `Dictionary` and `Array` with mixed-type values

**Special cases:**
- `Quaternion` component ordering must be consistent between the C++ constructor convention and the C# field order
- `PackedStringArray` entries are variable-length and the cursor must advance correctly between entries
- `Dictionary` and `Array` are validated recursively, not just by container size

A passing run means the command round-trip works end to end, the memory writers and readers are symmetric, and the variant codec handles nested containers correctly.

---

## Build Guide

### Requirements

- Godot 4.x (.NET edition)
- .NET 8 SDK
- Emscripten (EMSDK)
- Python 3 and SCons
- Chrome or Chromium for headless testing

```bash
dotnet --version
emcc --version
scons --version
```

---

### 1. Build Godot for web

From the repository root:

```bash
scons platform=web target=template_release
```

Output goes to `bin/`.

---

### 2. Build the .NET WASM project

Linux / macOS:
```bash
dotnet publish "Tests.csproj" -c Release -r browser-wasm \
  -p:SelfContained=true -o "./publish_temp"
```

Windows:
```bash
dotnet publish "Tests.csproj" -c Release -r browser-wasm ^
  -p:SelfContained=true -o ".\publish_temp"
```

---

### 3. Assemble the build folder

Target: `bin/.web_zip/`

Create a `cs/` subfolder inside it, then copy the following :

```
bin/.web_zip/
  host.html
  main.js
  server.py
  dotnet_worker.js
  cs/
    interop.js
    _framework
```

- Copy `modules/cross_runtime/tests/Web_Assets/interop.js` into `bin/.web_zip/cs/`
- Copy the `_framework` folder from `publish_temp/wwwroot/_framework` into `bin/.web_zip/cs/`

---

### 4. Start the server

`SharedArrayBuffer` requires COOP and COEP headers. Open a terminal in `bin/.web_zip/` and run:

```bash
python3 server.py
```

---

### 5. Run the headless test

Linux:
```bash
google-chrome --headless=new --no-sandbox \
  --enable-logging=stderr --v=1 \
  http://localhost:8000/host.html 2>&1 \
  | grep -E "\[C\+\+\]|\[Worker\]|bridge_test|\[Main\]"
```

macOS:
```bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --headless=new --enable-logging=stderr --v=1 \
  http://localhost:8000/host.html 2>&1 \
  | grep -E "\[C\+\+\]|\[Worker\]|bridge_test|\[Main\]"
```

Windows (PowerShell):
```powershell
& "C:\Program Files\Google\Chrome\Application\chrome.exe" `
  --headless=new --enable-logging=stderr --v=1 `
  http://localhost:8000/host.html 2>&1 `
  | Select-String "\[C\+\+\]","\[Worker\]","bridge_test","\[Main\]"
```

A successful run prints `Bridge Test Passed`. That confirms both runtimes are reading and writing shared memory correctly.
