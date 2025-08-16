## MSVC Clang-cl example

This example can run simple Linux programs on Windows, compiled using a CMake project in MSVC.

In order to test, build example.exe using x86-64 Release, and then run it on the various test programs that are bundled with libriscv. You will need a new Clang-cl in order to be able to build.

Golang example:
```powershell
PS E:\libriscv\examples\msvc\out\build\x64-release> .\example.exe E:\libriscv\tests\unit\elf\golang-riscv64-hello-world
Unhandled system call: 123
Unhandled system call: 130
Unhandled system call: 130
hello world
Runtime: 1.451ms  MI/s: 143.29
```

Rust example:
```powershell
PS E:\libriscv\examples\msvc\out\build\x64-release> .\example.exe E:\libriscv\tests\unit\elf\rust-riscv64-hello-world
Unhandled system call: 73
Hello World!

Runtime: 0.274ms  MI/s: 41.38
```

Zig example:
```powershell
PS E:\libriscv\examples\msvc\out\build\x64-release> .\example.exe E:\libriscv\tests\unit\elf\zig-riscv64-hello-world
Hello, world!

Runtime: 2.757ms  MI/s: 2.52
```

Default: fib(256000000):
```powershell
PS E:\libriscv\examples\msvc\out\build\x64-release> .\example.exe

Runtime: 844.296ms  MI/s: 1516.06
```
The fibonacci example is something I use to measure millions of instructions per second.

