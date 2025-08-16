## Adding _libriscv_ to your project

This example project shows how you can embed libriscv in your own CMake projects.

We use FetchContent in order to retrive the latest version of _libriscv_:

```CMake
include(FetchContent)
FetchContent_Declare(libriscv
  GIT_REPOSITORY https://github.com/fwsGonzo/libriscv
  GIT_TAG        master
  )

FetchContent_MakeAvailable(libriscv)
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
