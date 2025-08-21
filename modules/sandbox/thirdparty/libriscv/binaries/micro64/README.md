## RISC-V Micro Binary

Does the absolute minimum libc initiaization and then enters main. Will initialize GP, clear BSS (zero-initialized memory), call constructors and pass (argc, argv) to main.

The minimal binary is TODO bytes and uses TODO kB memory (in pages).

## Expected output

```
Hello, Global Constructor!
Hello World from hello_world!
```
