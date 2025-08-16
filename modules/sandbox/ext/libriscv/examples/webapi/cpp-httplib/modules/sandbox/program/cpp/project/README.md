# Example C++ CMake project

This folder contains an example CMake project that uses Zig to build RISC-V C and C++ programs with the Godot Sandbox API included.

## Building

The project is built using [build.sh](build.sh):

```sh
Usage: ./build.sh [options]

Options:
  -h, --help      Display this help and exit
  --runtime-api   Download a run-time generated Godot API header
  --no-runtime-api Do not download a run-time generated Godot API header
  --debug         Build with debug symbols
  --debinfo       Build with debug info
  --strip         Strip the binary
  --no-strip      Do not strip the binary
  --single        Build with single precision vectors
  --double        Build with double precision vectors
  --C             Enable RISC-V C extension (default)
  --no-C          Disable RISC-V C extension
  --toolchain     Specify a custom toolchain file
  --verbose       Enable verbose build
```

Example:

```sh
$ ./build.sh --strip
~/github/godot-riscv/program/cpp/project/.build ~/github/godot-riscv/program/cpp/project
-- The C compiler identification is Clang 19.1.0
-- The CXX compiler identification is Clang 19.1.0
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: /home/gonzo/zig/zig - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /home/gonzo/zig/zig - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Building using zig c++
-- Configuring done
-- Generating done
-- Build files have been written to: /home/gonzo/github/godot-riscv/program/cpp/project/.build
[  4%] Building C object CMakeFiles/atomic.dir/src/atomic.c.o
[  9%] Building CXX object cmake/CMakeFiles/sandbox_api.dir/api/array.cpp.o
[ 18%] Building CXX object cmake/CMakeFiles/sandbox_api.dir/api/api.cpp.o
[ 18%] Building CXX object cmake/CMakeFiles/sandbox_api.dir/api/native.cpp.o
[ 22%] Building CXX object cmake/CMakeFiles/sandbox_api.dir/api/basis.cpp.o
[ 27%] Building CXX object cmake/CMakeFiles/sandbox_api.dir/api/node2d.cpp.o
[ 36%] Building CXX object cmake/CMakeFiles/sandbox_api.dir/api/node.cpp.o
[ 36%] Building CXX object cmake/CMakeFiles/sandbox_api.dir/api/dictionary.cpp.o
[ 40%] Linking C static library libatomic.a
[ 40%] Built target atomic
[ 45%] Building CXX object cmake/CMakeFiles/sandbox_api.dir/api/node3d.cpp.o
[ 50%] Building CXX object cmake/CMakeFiles/sandbox_api.dir/api/object.cpp.o
[ 54%] Building CXX object cmake/CMakeFiles/sandbox_api.dir/api/packed_array.cpp.o
[ 59%] Building CXX object cmake/CMakeFiles/sandbox_api.dir/api/quaternion.cpp.o
[ 63%] Building CXX object cmake/CMakeFiles/sandbox_api.dir/api/string.cpp.o
[ 68%] Building CXX object cmake/CMakeFiles/sandbox_api.dir/api/timer.cpp.o
[ 72%] Building CXX object cmake/CMakeFiles/sandbox_api.dir/api/transform2d.cpp.o
[ 77%] Building CXX object cmake/CMakeFiles/sandbox_api.dir/api/transform3d.cpp.o
[ 81%] Building CXX object cmake/CMakeFiles/sandbox_api.dir/api/variant.cpp.o
[ 86%] Building CXX object cmake/CMakeFiles/sandbox_api.dir/api/vector.cpp.o
[ 90%] Linking CXX static library libsandbox_api.a
[ 90%] Built target sandbox_api
[ 95%] Building CXX object CMakeFiles/sandbox_program.dir/src/program.cpp.o
[100%] Linking CXX executable sandbox_program
[100%] Built target sandbox_program
~/github/godot-riscv/program/cpp/project
```

The name of the program files built depends on the name used in CMakeLists.txt:

```cmake
cmake_minimum_required(VERSION 3.18)
project (godot-sandbox-programs)

# Fetch godot-sandbox for add_sandbox_program and other functions
include(FetchContent)
FetchContent_Declare(
	godot-sandbox
	GIT_REPOSITORY https://github.com/libriscv/godot-sandbox.git
	GIT_TAG        main
	SOURCE_SUBDIR  "program/cpp/cmake"
)
FetchContent_MakeAvailable(godot-sandbox)

# Build the godot-sandbox program `sandbox_program.elf`:
add_sandbox_program(sandbox_program.elf
	"src/program.cpp"
)
```

So, based on `add_sandbox_program(name file1 ... fileN)` it should produce an ELF binary named `sandbox_program.elf` inside the build folder.

The build process happens inside a hidden folder called `.build`:

```sh
$ ls -lah .build/
total 268K
drwxrwxr-x 4 gonzo gonzo 4,0K des.  10 11:57 .
drwxrwxr-x 4 gonzo gonzo 4,0K des.  10 11:57 ..
drwxrwxr-x 3 gonzo gonzo 4,0K des.  10 11:57 cmake
-rw-rw-r-- 1 gonzo gonzo  14K des.  10 11:57 CMakeCache.txt
drwxrwxr-x 6 gonzo gonzo 4,0K des.  10 11:57 CMakeFiles
-rw-rw-r-- 1 gonzo gonzo 1,9K des.  10 11:57 cmake_install.cmake
-rw-rw-r-- 1 gonzo gonzo 1,1K des.  10 11:57 libatomic.a
-rw-rw-r-- 1 gonzo gonzo 7,0K des.  10 11:57 Makefile
-rwxrwxr-x 1 gonzo gonzo 219K des.  10 11:57 sandbox_program.elf
```

We can see a 219kb `sandbox_program`. This program is ready to use.

```sh
$ file .build/sandbox_program.elf
.build/sandbox_program.elf: ELF 64-bit LSB executable, UCB RISC-V, RVC, double-float ABI, version 1 (SYSV), statically linked, stripped
```

It's a statically built RISC-V program, so it can be loaded by Godot Sandbox. In order for Godot Sandbox to see the ELF file as a resource, it should have the `.elf` file extension.
