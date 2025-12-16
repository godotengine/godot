# Dual-Mode ELF64 Compilation Design

## Overview

The GDScript ELF64 compiler supports three execution modes:

1. **Godot Syscall Mode** (Mode 0): Pure Godot sandbox execution
   - Uses custom ECALL numbers (500+) defined in `modules/sandbox/src/syscalls.h`
   - Allows calling back into Godot's API (Variant operations, Object methods, etc.)
   - Requires Godot sandbox runtime to execute

2. **Linux Syscall Mode** (Mode 1): Pure standalone RISC-V Linux execution
   - Uses standard Linux syscall numbers (1-400+)
   - Can run on real RISC-V hardware or emulators (qemu, spike, etc.)
   - Requires standard RISC-V Linux runtime environment
   - Limited support (many GDScript features require Godot runtime)

3. **Hybrid Mode** (Mode 2 - Default): **Recommended** - Single ELF with both syscall types
   - Uses Godot syscalls (ECALL 500+) by default for Godot-specific operations
   - Uses Linux syscalls (1-400+) when appropriate (e.g., standard I/O)
   - Single binary works in both Godot sandbox and standalone libriscv
   - Optimal syscall selection based on operation type

## Architecture Changes

### 1. Mode Enumeration

```cpp
enum class ELF64CompilationMode {
    GODOT_SYSCALL,  // Mode 0: Pure Godot sandbox syscalls
    LINUX_SYSCALL,  // Mode 1: Pure Linux syscalls
    HYBRID          // Mode 2: Hybrid - both syscall types in same ELF (default)
};
```

### 2. API Changes

#### GDScriptFunction
```cpp
// Add mode parameter
PackedByteArray compile_to_elf64(ELF64CompilationMode p_mode = ELF64CompilationMode::GODOT_SYSCALL) const;
bool can_compile_to_elf64(ELF64CompilationMode p_mode = ELF64CompilationMode::GODOT_SYSCALL) const;
```

#### GDScript
```cpp
Dictionary compile_all_functions_to_elf64(ELF64CompilationMode p_mode = ELF64CompilationMode::GODOT_SYSCALL) const;
bool can_compile_to_elf64(ELF64CompilationMode p_mode = ELF64CompilationMode::GODOT_SYSCALL) const;
```

#### GDScriptELF64Writer
```cpp
static PackedByteArray write_elf64(GDScriptFunction *p_function, ELF64CompilationMode p_mode);
static bool can_write_elf64(GDScriptFunction *p_function, ELF64CompilationMode p_mode);
```

#### GDScriptRISCVEncoder
```cpp
static PackedByteArray encode_function(GDScriptFunction *p_function, ELF64CompilationMode p_mode);
static PackedByteArray encode_opcode(int p_opcode, const int *p_code_ptr, int &p_ip, int p_code_size, ELF64CompilationMode p_mode);
static PackedByteArray encode_syscall(int p_syscall_num, ELF64CompilationMode p_mode);
```

### 3. Syscall Encoding Differences

#### Mode 0: Pure Godot Syscalls
- Uses ECALL numbers from `modules/sandbox/src/syscalls.h` (500+)
- Example: `ECALL_VCALL = 501` for Variant calls
- Encoding: `li a7, <ecall_number>; ecall`
- Requires sandbox runtime with Godot syscall handlers

#### Mode 1: Pure Linux Syscalls
- Uses standard Linux syscall numbers
- Example: `write = 64`, `exit = 93`, `read = 63`
- Encoding: `li a7, <syscall_number>; ecall`
- Requires Linux kernel or compatible runtime
- Limited support (many GDScript features unsupported)

#### Mode 2: Hybrid (Default - Recommended)
- **Default**: Uses Godot ECALLs (500+) for Godot-specific operations
  - Member access: `ECALL_OBJ_PROP_GET` (545)
  - Variant operations: `ECALL_VCALL` (501)
  - Object methods: `ECALL_OBJ_CALLP` (506)
- **When needed**: Uses Linux syscalls (1-400+) for standard operations
  - I/O operations: `write` (64) for stdout
  - System operations: `exit` (93) for program termination
- **Single ELF**: Contains both syscall types, works in both environments

### 4. Opcode Translation Strategy

#### Mode 1 (Godot Syscalls)
- Complex operations → Godot ECALLs
  - `OPCODE_GET_MEMBER` → `ECALL_OBJ_PROP_GET`
  - `OPCODE_CALL` → `ECALL_VCALL`
  - `OPCODE_OPERATOR` → `ECALL_MATH_OP32/64`
- Simple operations → Direct RISC-V instructions where possible
  - `OPCODE_ASSIGN` → Direct register/memory operations
  - `OPCODE_JUMP` → Direct `jal` instruction

#### Mode 2 (Linux Syscalls)
- Complex operations → Linux syscalls or libc calls
  - `OPCODE_GET_MEMBER` → Not directly translatable (requires Godot runtime)
  - `OPCODE_CALL` → Function call (if target is also compiled)
  - `OPCODE_OPERATOR` → Direct RISC-V arithmetic instructions
- Simple operations → Direct RISC-V instructions
  - Same as Mode 1
- Limitations: Many GDScript features require Godot runtime
  - Solution: Emit warnings/errors for unsupported opcodes in Linux mode

### 5. ELF64 Structure Differences

#### Mode 1: Godot Syscall Mode
- Entry point: Standard (0x10000)
- Sections: `.text`, `.data` (if needed)
- Interpreter: None (direct execution in sandbox)
- Dependencies: None (syscalls handled by sandbox)

#### Mode 2: Linux Syscall Mode
- Entry point: Standard (0x10000)
- Sections: `.text`, `.data`, `.rodata`, `.bss` (if needed)
- Interpreter: `/lib/ld-linux-riscv64-lp64d.so.1` (for dynamic linking)
- Dependencies: May require libc for complex operations
- Note: Could also be statically linked for standalone execution

### 6. Implementation Plan

#### Phase 1: Add Mode Parameter
1. Add `ELF64CompilationMode` enum
2. Update function signatures to accept mode parameter
3. Pass mode through call chain: `GDScript` → `GDScriptFunction` → `GDScriptELF64Writer` → `GDScriptRISCVEncoder`

#### Phase 2: Mode-Specific Syscall Encoding
1. Create `encode_godot_syscall()` for Mode 1
2. Create `encode_linux_syscall()` for Mode 2
3. Update `encode_vm_call()` to route based on mode

#### Phase 3: Opcode Translation
1. Identify which opcodes can be translated in each mode
2. Implement mode-specific opcode handlers
3. Add validation: warn/error for unsupported opcodes in Linux mode

#### Phase 4: ELF64 Structure
1. Mode 1: Keep current structure (minimal)
2. Mode 2: Add proper ELF sections for Linux compatibility
3. Add `.interp` section for dynamic linking (optional)

#### Phase 5: Testing
1. Mode 1: Test with sandbox (existing E2E tests)
2. Mode 2: Test with qemu-riscv64 or real hardware
3. Create test fixtures for both modes

### 7. GDScript API Usage

```gdscript
# Mode 2: Hybrid (default - recommended)
var elf_hybrid = script.compile_all_functions_to_elf64()  # Defaults to HYBRID
var elf_hybrid_explicit = script.compile_all_functions_to_elf64(2)  # Explicit hybrid

# Mode 0: Pure Godot syscalls
var elf_godot = script.compile_all_functions_to_elf64(0)

# Mode 1: Pure Linux syscalls
var elf_linux = script.compile_all_functions_to_elf64(1)
```

### 8. CLI Tool Updates

The `gdscript_elf64_compiler.gd` tool supports mode selection:

```bash
# Mode 2: Hybrid (default - recommended)
godot --headless --script tools/gdscript_elf64_compiler.gd script.gd
godot --headless --script tools/gdscript_elf64_compiler.gd script.gd --mode hybrid

# Mode 0: Pure Godot
godot --headless --script tools/gdscript_elf64_compiler.gd script.gd --mode godot

# Mode 1: Pure Linux
godot --headless --script tools/gdscript_elf64_compiler.gd script.gd --mode linux
```

## Benefits

1. **Flexibility**: Choose execution environment based on use case
2. **Portability**: Linux mode enables running on real RISC-V hardware
3. **Integration**: Godot mode enables tight integration with engine
4. **Hybrid Mode**: Single binary works in both environments (recommended)
5. **Optimal Syscall Selection**: Hybrid mode chooses best syscall type per operation
6. **Testing**: Can test same code in both environments

## Limitations

1. **Linux Mode Limitations**: Many GDScript features require Godot runtime
   - Solution: Emit clear errors for unsupported features
   - Future: Could compile more opcodes to native RISC-V

2. **Mode Selection**: Must be chosen at compile time
   - Cannot switch modes at runtime
   - Different binaries for different modes

## Future Enhancements

1. **Hybrid Mode**: Some functions in Godot mode, others in Linux mode
2. **Automatic Mode Selection**: Detect which mode is possible based on opcodes used
3. **More Linux Syscalls**: Expand support for more GDScript features in Linux mode
