# gdverify

A tiny tool to parse GDScript files using Godot's built-in parser.

## Build

```sh
mkdir build && cd build
cmake ../gdverify -DGODOT_SRC_DIR=../godot -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

## Run

```sh
./gdverify path/to/script.gd
```

Exit code `0` means success, `65` indicates a parse error.
