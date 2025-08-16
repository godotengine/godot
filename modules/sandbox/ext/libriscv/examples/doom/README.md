## RISC-V D00M Emulation

Copy shareware doom1.wad or retail doom.wad into this directory and then build and run:

```sh
./build.sh
```

All binary translation modes and related experimental options are supported for this program. On my machine the demo uses 6% CPU with binary translation enabled, and 12% without it.

```sh
cd build
cmake .. -DRISCV_BINARY_TRANSLATION=1
make -j4
```


Requires CMake, SDL2:

```sh
sudo apt install cmake libsdl2-dev
```
