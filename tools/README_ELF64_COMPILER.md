# GDScript ELF64 Compiler Tool

A command-line tool to compile GDScript functions to ELF64 binaries.

## Usage

```bash
godot --headless --script tools/gdscript_elf64_compiler.gd <input.gd> [output_dir] [--mode mode]
```

### Arguments

- `<input.gd>` - Path to the GDScript file to compile (required)
- `[output_dir]` - Directory where ELF64 binaries will be saved (optional, defaults to current directory)
- `[--mode mode]` - Compilation mode (optional, defaults to `hybrid`)
  - `godot` or `0`: Pure Godot sandbox syscalls (ECALL 500+)
  - `linux` or `1`: Pure Linux syscalls (1-400+)
  - `hybrid` or `2`: **Recommended** - Hybrid mode uses Godot syscalls by default, Linux syscalls when needed

### Examples

```bash
# Compile script.gd with hybrid mode (default - recommended)
godot --headless --script tools/gdscript_elf64_compiler.gd script.gd

# Compile with specific mode
godot --headless --script tools/gdscript_elf64_compiler.gd script.gd --mode hybrid
godot --headless --script tools/gdscript_elf64_compiler.gd script.gd --mode godot
godot --headless --script tools/gdscript_elf64_compiler.gd script.gd --mode linux

# Compile to specific output directory
godot --headless --script tools/gdscript_elf64_compiler.gd script.gd output/

# Compile with absolute paths
godot --headless --script tools/gdscript_elf64_compiler.gd /path/to/script.gd /path/to/output/
```

## Compilation Modes

### Hybrid Mode (Default - Recommended)

**Mode:** `hybrid` or `2`

The hybrid mode creates a single ELF64 binary that uses both syscall types:
- **Default**: Uses Godot sandbox syscalls (ECALL 500+) for Godot-specific operations
- **When needed**: Uses Linux syscalls (1-400+) for standard operations that don't require Godot runtime

**Benefits:**
- Single binary works in both Godot sandbox and standalone libriscv
- Optimal syscall selection based on operation type
- Maximum compatibility

**Use when:** You want a binary that works in both environments (recommended for most cases)

### Godot Mode

**Mode:** `godot` or `0`

Pure Godot sandbox mode - all operations use Godot ECALLs (500+).

**Benefits:**
- Full access to Godot API via ECALLs
- Works only in Godot sandbox environment

**Use when:** Binary will only run in Godot sandbox

### Linux Mode

**Mode:** `linux` or `1`

Pure Linux syscall mode - uses standard Linux syscalls (1-400+).

**Limitations:**
- Many GDScript features require Godot runtime and are unsupported
- Only simple operations (arithmetic, basic control flow) are supported

**Use when:** You need a standalone binary for simple operations (limited support)

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
