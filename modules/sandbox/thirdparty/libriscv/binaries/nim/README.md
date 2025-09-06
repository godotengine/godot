## Nim RISC-V program example

This program contains a build system that compiles Nim for RISC-V.
The program prints a few things:

```
[{"name":"John","age":30},{"name":"Susan","age":31}]
{"name":"Isaac","books":["Robot Dreams"],"details":{"age":35,"pi":3.1415}}
woof
70
meow
10

```

### Requirements

You will need to have Nim in your PATH.

Install jq and gcc-10-riscv64-linux-gnu or gcc-11-riscv64-linux-gnu to be able build Nim programs for RISC-V.

```
sudo apt install jq gcc-11-riscv64-linux-gnu
```

### Building

```
./build.sh
```

### Running it

```
./rvlinux ../binaries/nim/riscv64-linux-gnu/hello
```

### Debugging

Run `bash debug.sh` to start remotely debugging with GDB. You will need to install the multiarch GDB package for your distro.
