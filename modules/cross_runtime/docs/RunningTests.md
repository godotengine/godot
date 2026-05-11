## CrossRuntime Build Guide
This guide explains how to manually build and bridge Godot WASM and .NET WASM. By the end, you will have a web-based environment where C# and Godot communicate via shared memory.

---

## 1. Environment Setup
Ensure these tools are installed and available in your terminal's PATH:

* Godot 4.x (.NET Edition): To compile the engine bridge.
* .NET 8 SDK: To build the C# WASM logic.
* EMSDK (Emscripten): To compile Godot for the web.
* Python 3 & SCons: For build orchestration and hosting.
* Google Chrome / Chromium: Required for the headless test command.

Check your versions:
```bash
dotnet --version
emcc --version
scons --version
```

------------------------------
## 2. Compile Godot for Web
Build the WebAssembly version of the Godot engine. Run this from the root of your repository:

```bash
scons platform=web target=template_release
```

* Note: This produces the essential .wasm and .js engine files in your bin/ folder.

------------------------------
## 3. Build the .NET WASM Project
Compile your C# code into a browser-compatible format.
Linux / macOS:
```bash
dotnet publish "Tests.csproj" -c Release -r browser-wasm -p:SelfContained=true -o "./publish_temp"
```
Windows:
```bash
dotnet publish "Tests.csproj" -c Release -r browser-wasm -p:SelfContained=true -o ".\publish_temp"
```

------------------------------
## 4. Manual Asset Assembly
Because file paths vary significantly between Windows, macOS, and Linux, follow these steps carefully to organize your build folder.
Target Folder: bin/.web_zip/
   1. Create Folders: Ensure bin/.web_zip/ exists, and create a subfolder inside it named cs.
   2. Move Interop Logic: Copy modules/cross_runtime/tests/Web_Assets/interop.cs into bin/.web_zip/cs/.
   3. Move Web Boilerplate: Copy host.html, main.js, server.py, and dotnet_worker.js from the source into bin/.web_zip/.
   4. Move .NET Runtime: Locate the folder publish_temp/wwwroot/_framework. Copy every file inside it into bin/.web_zip/cs/.

Your final structure should look like this:
```
bin/.web_zip/
├── host.html
├── main.js
├── server.py
├── dotnet_worker.js
└── cs/
    ├── interop.cs
    └── (All files from _framework, e.g., dotnet.native.wasm, etc.)
```
------------------------------
## 5. Launch the Server
WebAssembly shared memory requires COOP/COEP headers for security. Standard file opening won't work.

   1. Open your terminal in bin/.web_zip/.
   2. Run: python3 server.py

------------------------------

## 6. Run the Headless Test
To verify the bridge, use Chrome's headless mode to pipe logs directly to your terminal. Open a new terminal window:
Linux:
```bash
google-chrome --headless=new --no-sandbox --enable-logging=stderr --v=1 http://localhost:8000/host.html 2>&1 | grep -E "\[C\+\+\]|\[Worker\]|bridge_test|\[Main\]"
```
macOS:
```bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --headless=new --enable-logging=stderr --v=1 http://localhost:8000/host.html 2>&1 | grep -E "\[C\+\+\]|\[Worker\]|bridge_test|\[Main\]"
```

Windows (PowerShell):
```Bash
& "C:\Program Files\Google\Chrome\Application\chrome.exe" --headless=new --enable-logging=stderr --v=1 http://localhost:8000/host.html 2>&1 | Select-String "\[C\+\+\]","\[Worker\]","bridge_test","\[Main\]"
```
------------------------------

## How it works:
Once loaded, the C# Runtime writes commands into a shared memory buffer. On every frame, the Godot Runtime checks that buffer, executes the requested logic, and writes a result back. If you see "Bridge Test Passed" in your console, the runtimes are communicating successfully.

------------------------------
