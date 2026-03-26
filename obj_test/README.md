# Invalid OBJ models

Each file in this folder is malformed in a different way for parser testing.

| File | Malformation |
|------|--------------|
| _invalid_alligator.obj | Vertex with only 2 coordinates (`v 1 2`) |
| _invalid_armadillo.obj | Vertex with non-numeric values (`v x y z`) |
| _invalid_beast.obj | Face with vertex index 0 (invalid in 1-based OBJ) |
| _invalid_beetle-alt.obj | Face with only 2 vertices |
| _invalid_beetle.obj | Unknown directive `vx 1 2 3` inserted |
| _invalid_bimba.obj | File truncated by 50 bytes at end |
| _invalid_cheburashka.obj | Null byte inserted in middle of first vertex line |
| _invalid_cow.obj | Vertex with 6 components instead of 3 or 4 |
| _invalid_fandisk.obj | Normal with 2 components instead of 3 |
| _invalid_happy.obj | Malformed face with slashes `f 1/2 3/4 5/6` |
| _invalid_homer.obj | Empty object name `o ` at start |
| _invalid_horse.obj | Vertex with overflow-style numbers `1e999` |
| _invalid_igea.obj | Vertex with trailing garbage `#@!$` |
| _invalid_lucy.obj | Single-component `vt` or comma-separated vertex |
| _invalid_max-planck.obj | Two vertices on one line |
| _invalid_nefertiti.obj | First vertex line replaced by whitespace only |
| _invalid_ogre.obj | Face with malformed slashes `f 1// 2// 3//` |
| _invalid_rocker-arm.obj | Vertex line with `\r\r` line ending |
| _invalid_spot.obj | Empty `mtllib ` at start |
| _invalid_stanford-bunny.obj | Null byte before newline in first vertex |
| _invalid_suzanne.obj | Face `f 0 0 0` (invalid indices) |
| _invalid_teapot.obj | Empty `usemtl ` at start |
| _invalid_woody.obj | Vertex with 5 components |
| _invalid_xyzrgb_dragon.obj | Last line truncated by 10 characters |


HOW TO USE:
build with tests=yes (e.g. scons tests=yes)
GODOT_OBJ_IMPORT_TEST_DIR=/obj_test ./path-to-built-executable --test --test-case="*ResourceImporterOBJ*"