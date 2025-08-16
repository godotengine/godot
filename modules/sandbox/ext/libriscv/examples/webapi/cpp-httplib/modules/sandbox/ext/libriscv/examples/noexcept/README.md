## Adding _libriscv_ to your project

This example project shows how you can build libriscv without C++ exceptions.

For now, only the outer project is without exceptions. This is on-going work.

## fib.rv64.elf

The example program calculates fibonacci numbers. Run the program with the number as argument:

```
./.build/example fib.rv64.elf 64
```
Will calculate the 64th fibonacci number.
