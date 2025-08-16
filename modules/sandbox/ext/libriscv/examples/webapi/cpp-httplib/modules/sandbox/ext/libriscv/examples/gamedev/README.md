## Using _libriscv_ in a game engine

This example project shows the basics for how to use _libriscv_ in a game engine.

The [build script](build_and_run.sh) will first build the RISC-V guest program, and then the host program. If it succeeds, the program will run.


## Script_program folder

An [example program](cpp_program/program.cpp) that modifies some libc functions to use host-side functions with native performance.

The program showcases low-latency and full C++ support.


## Guides

- [An Introduction to Low-Latency Scripting With libriscv](https://fwsgonzo.medium.com/an-introduction-to-low-latency-scripting-with-libriscv-ad0619edab40)

- [An Introduction to Low-Latency Scripting With libriscv, Part 2](https://fwsgonzo.medium.com/an-introduction-to-low-latency-scripting-with-libriscv-part-2-4fce605dfa24)


## Other languages

### Nelua

Nelua has been [implemented here](nelua_program/)

[Guide to Scripting with Nelua](https://medium.com/@fwsgonzo/an-introduction-to-low-latency-scripting-with-libriscv-part-3-5947b06bc00c)

### Nim

There is a [Nim example](nim_program/)

[Guide to Scripting with Nim](https://medium.com/@fwsgonzo/an-introduction-to-low-latency-scripting-with-libriscv-part-4-103ff7e67c24)

### Rust

There is a [Rust example](rust_program/)

[Guide to Scripting with Rust](https://fwsgonzo.medium.com/an-introduction-to-low-latency-scripting-for-game-engines-part-5-d7f12630e6bf)
