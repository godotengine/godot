libriscv emulator CLI
============================================

This folder contains a small subproject that enables using libriscv to run RISC-V programs directly from the terminal. A command-line interface using libriscv, if you will.

* [Usage](#usage)
* [Debugging](#debugging)
* [Binary translation](#binary-translation)

## Usage

```sh
bash build.sh
$ ./rvlinux ../tests/unit/elf/zig-riscv64-hello-world 
Hello, world!
>>> Program exited, exit code = 0 (0x0)
Instructions executed: 6980  Runtime: 0.033ms  Insn/s: 214mi/s
Pages in use: 87 (348 kB virtual memory, total 1714 kB)
```

The CLI comes with a few commandline options:

```sh
$ ./rvlinux --help
Usage: ./rvlinux [options] <program> [args]
Options:
  -h, --help         Print this help message
  -v, --verbose      Enable verbose loader output
  -Q, --quit         Quit after loading the program (to produce eg. binary translations)
  -a, --accurate     Accurate instruction counting
  -d, --debug        Enable CLI debugger
  -1, --single-step  One instruction at a time, enabling exact exceptions
  -f, --fuel amt     Set max instructions until program halts
  -m, --memory amt   Set max memory size in MiB (default: 4096 MiB)
  -g, --gdb          Start GDB server on port 2159
  -s, --silent       Suppress program completion information
  -t, --timing       Enable timing information in binary translator
  -T, --trace        Enable tracing in binary translator
  -n, --no-translate Disable binary translation
  -N, --no-translate-future Disable binary translation of non-initial segments
  -R, --translate-regcache Enable register caching in binary translator
  -J, --jump-hints file  Load jump location hints from file, unless empty then record instead
  -B  --background   Run binary translation in background thread
  -M, --mingw        Cross-compile for Windows (MinGW)
  -o, --output file  Output embeddable binary translated code (C99)
  -F, --from-start   Start debugger from the beginning (_start)
  -S  --sandbox      Enable strict sandbox
  -P, --proxy        Enable proxy mode, allowing access to all files (disabling the sandbox)
  -A, --allow file   Allow file to be opened by the guest
  -X, --execute-only Enforce execute-only segments (no read/write)
  -I, --ignore-text  Ignore .text section, and use segments only
  -c, --call func    Call a function after loading the program
```

In order to use the CLI you will need some RISC-V programs. There are a few ready-to-run programs in the [tests/unit/elf](/tests/unit/elf) folder. These are part of the automated tests for the emulator.

If you are looking for the internals of the RISC-V emulator, that is in the [library folder](/lib/libriscv/). If you want to use it as a library, then that is shown in the [examples folder](/examples).

## Build options

The CLI can be built with a variety of options.

```
$ ./build.sh --help

   Usage: build.sh [options] [--embed file] [--embed file] ...

   optional arguments:
     -h, --help           show this help message and exit
     --debug              enable debugging
     --debinfo            enable debugging information
     --defaults           build with default settings
     --perf               build settings for performance profiling
     --no-perf            disable performance profiling
     --static             build static CLI executable
     --no-static          build dynamic CLI executable
     --native             build with -march=native
     --no-native          disable -march=native
     --lto                build with link-time optimization
     --no-lto             disable link-time optimization
     --A                  enable atomic extension
     --no-A               disable atomic extension
     --C                  enable compressed extension
     --no-C               disable compressed extension
     --V                  enable vector extension
     --no-V               disable vector extension
     --32                 enable RV32
     --no-32              disable RV32
     --64                 enable RV64
     --no-64              disable RV64
     --128                enable RV128
     --no-128             disable RV128
     --assembly-dispatch  enable assembly dispatch (experimental feature)
     --no-assembly-dispatch disable assembly dispatch
     -b, --bintr          enable binary translation using system compiler
     --no-bintr           disable binary translation
     -t, --jit            jit-compile using tcc
     --no-jit             disable jit-compile using tcc
     -x, --expr           enable experimental features (eg. unbounded 32-bit addressing)
     -N bits              enable N-bits of masked address space (experimental feature)
     --no-expr            disable experimental features
     --embed FILE         embed binary translated sources into the emulator, produced by CLI -o option
     -v, --verbose        increase the verbosity of the bash script

```

## Debugging

For debugging instructions one by one, use `--debug`:

```
$ ./rvlinux --debug ../tests/unit/elf/newlib-rv32gb-hello-world 

*
* Entered main() @ 0x10670
*

>>> Breakpoint 	[00010670] E9010113 ADDI SP, SP-368 (0x11A1E80)

[RA	000113D8] [SP	011A1E80] [GP	000A09F0] [TP	00000000] 
[LR	7F7F7FFF] [TMP1	00000002] [TMP2	FFFFFFFF] [SR0	00000000] [SR1	00000000] 
[A0	00000001] [A1	011A1E84] [A2	00000000] [A3	0000003E] [A4	FF0A0000] 
[A5	00011360] [A6	00000020] [A7	000A1E4C] [SR2	00000000] [SR3	00000000] 
[SR4	00000000] [SR5	00000000] [SR6	00000000] [SR7	00000000] [SR8	00000000] 
[SR9	00000000] [SR10	00000000] [SR11	00000000] [TMP3	00000000] [TMP4	00000000] 
[TMP5	00000000] [TMP6	00000000] [MEM PAGES          148]
Enter = cont, help, quit: 
```

Will let you step through the program instruction by instruction. By default it stops just after entering `main()`, if the function is found. If you want to prevent that, there is the `--from-start` option.

For remote debugging with GDB, use `--gdb`:

```
$ ./rvlinux --gdb ../tests/unit/elf/newlib-rv32gb-hello-world 
GDB server is listening on localhost:2159

...

$ gdb-multiarch newlib-rv32gb-hello-world
```

Will let you use GDB to remotely debug the program. Use `target remote :2159` in GDB to connect:
```
(gdb) target remote :2159
Remote debugging using :2159
```

Once disconnected the emulator will continue as before, completing the program:

```
GDB is connected
Text contains the phrase 'regular expressions'
Found 20 words
Words longer than 6 characters:
  confronted
  problem
  regular
  expressions
  problems
Some people, when [confronted] with a [problem], think "I know, I'll use [regular] [expressions]." Now they have two [problems].
Testing exception
Caught exception: Hello Exceptions!
It took 1303152 instructions to throw, catch and print the exception
>>> Program exited, exit code = 666 (0x29A)
```

## Instruction counting vs. run forever

With the `-a` option, the CLI will run programs with accurate instruction counting. It will also cancel execution after a certain amount of instructions has been executed. This is a safe and reliable way to execute sandboxed code.

The default in the CLI is to not count instructions, and instead rely on the user to cancel by pressing eg. Ctrl+C. When using binary translation, the inaccurate emulation is faster.


## Binary translation

Binary translation is enabled with a CMake option. However, there are options to `build.sh` to simplify enabling it:

```sh
bash build.sh -x --bintr
```

In doing so, there are now more options available.

- `--timing` will display timing information from the binary translation process. The compilation process dominates heavily.
- `--trace` will make the translator embed tracing information, and print information to terminal for each translated instruction executed. It's a lot of logging.
- `--no-translate` will disable translation, and interpret the program instead.
- `--mingw ` is an experimental option that cross-compiles a Windows .dll for use on end-user systems. The .dll will appear in the current directory.

With binary translation enabled, we can now run a program. Let's try the 25600000th fibonacci number:

```sh
$ ./rvlinux -v ../binaries/measure_mips/fib
* Loading program of size 204 from 0x70424ef08000 to virtual 0x10000 -> 0x100cc
* Program segment readable: 1 writable: 0  executable: 1
libriscv: No embedded translation found for hash EAC3CF71
libriscv: Activated full binary translation with 1/2 mappings
* Entry is at 0x10074
>>> Program exited, exit code = 3819729467 (0xE3AC723B)
Runtime: 45.466ms   (Use --accurate for instruction counting)
Pages in use: 4 (16 kB virtual memory, total 22 kB)
```

Quite fast! Without binary translation, it's a bit slower:

```sh
$ ./rvlinux -na ../binaries/measure_mips/fib
>>> Program exited, exit code = 3819729467 (0xE3AC723B)
Instructions executed: 1280000008  Runtime: 964.903ms  Insn/s: 1327mi/s
Pages in use: 4 (16 kB virtual memory, total 22 kB)
```

So it was ~21x faster with binary translation!


## Using TCC as JIT compiler

Binary translation with TCC is enabled with a CMake option. However, there are options to `build.sh` to simplify enabling it:

```sh
bash build.sh -x --tcc
```

With this option enabled, rvlinux will run programs almost instantly, but execute them faster than when in interpreter mode:

```sh
$ ./rvlinux ../binaries/measure_mips/fib
>>> Program exited, exit code = 3819729467 (0xE3AC723B)
Runtime: 189.872ms   (Use --accurate for instruction counting)
Pages in use: 4 (16 kB virtual memory, total 22 kB)
```

Exactly 5x faster than interpreted!
