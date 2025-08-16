## Adding _libriscv_ to your project

This example project shows how you can embed libriscv in your own CMake projects.

We use a locally installed package of _libriscv_:

```CMake
find_package(PkgConfig)
pkg_check_modules(libriscv REQUIRED)
```

This will give us access to the `riscv` library in CMake. The library will automatically add includes and libraries to your own projects:

```CMake
add_executable(example example.cpp)
target_link_libraries(example riscv)
```

## fib.rv64.elf

The example program calculates fibonacci numbers. Run the program with the number as argument:

```
./.build/example fib.rv64.elf 64
```
Will calculate the 64th fibonacci number.
