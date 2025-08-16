# Examples

## SDL D00M

A sandboxed playable D00M example.

## Embed

An example that shows how to fetch libriscv from github and embed it in your CMake project.

## MSVC example

An example that works in Visual Studio, and allows you to run the [example binaries](/tests/unit/elf) from PowerShell.

## Noexcept example

An example that shows how to embed _libriscv_ in a project that disables C++ exceptions. _libriscv_ still requires them for itself, however it can be cordoned off, and all public headers are designed to support builds with disabled exceptions.

## Package example

An example that uses _libriscv_ when installed on the system as a package.

## WASM example

An example that builds a RISC-V programs with JIT-enabled LuaJIT inside, and then adds that to a WebAssembly program. The program is pre-built so that users can test immediately with `emrun`.

## WebAPI example

An example that uses a WebServer and Varnish Cache to implement a RISC-V playground.
