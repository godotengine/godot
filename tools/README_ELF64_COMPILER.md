# GDScript ELF64 Compiler Tool

A command-line tool to compile GDScript functions to ELF64 binaries.

## Usage

```bash
godot --headless --script tools/gdscript_elf64_compiler.gd <input.gd> [output_dir]
```

### Arguments

- `<input.gd>` - Path to the GDScript file to compile (required)
- `[output_dir]` - Directory where ELF64 binaries will be saved (optional, defaults to current directory)

### Examples

```bash
# Compile script.gd and save ELF64 files to current directory
godot --headless --script tools/gdscript_elf64_compiler.gd script.gd

# Compile script.gd and save ELF64 files to output/ directory
godot --headless --script tools/gdscript_elf64_compiler.gd script.gd output/

# Compile with absolute paths
godot --headless --script tools/gdscript_elf64_compiler.gd /path/to/script.gd /path/to/output/
```

## Output

For each function in the GDScript that can be compiled to ELF64, the tool generates a separate `.elf` file:

- Format: `<script_name>_<function_name>.elf`
- Example: If `math.gd` has a function `add`, the output will be `math_add.elf`

## Requirements

- The GDScript must have at least one function that can be compiled to ELF64
- Functions must have bytecode (not empty functions)
- The output directory must be writable (will be created if it doesn't exist)

## Error Handling

The tool will exit with:
- Exit code 0: Success
- Exit code 1: Error (invalid arguments, compilation failure, I/O errors)

## Example GDScript

```gdscript
# test.gd
func add(a: int, b: int) -> int:
    return a + b

func multiply(x: int, y: int) -> int:
    return x * y
```

Compiling this script will produce:
- `test_add.elf`
- `test_multiply.elf`
