# Dual-Mode ELF64 Compilation Design

## Overview

The GDScript ELF64 compiler should support two execution modes:

1. **Godot Syscall Mode** (Mode 1): For execution within Godot's sandbox
   - Uses custom ECALL numbers (500+) defined in `modules/sandbox/src/syscalls.h`
   - Allows calling back into Godot's API (Variant operations, Object methods, etc.)
   - Requires Godot sandbox runtime to execute

2. **Linux Syscall Mode** (Mode 2): For standalone RISC-V Linux execution
   - Uses standard Linux syscall numbers (1-400+)
   - Can run on real RISC-V hardware or emulators (qemu, spike, etc.)
   - Requires standard RISC-V Linux runtime environment

## Architecture Changes

### 1. Mode Enumeration

```cpp
enum class ELF64CompilationMode {
    GODOT_SYSCALL,  // Mode 1: Godot sandbox syscalls
    LINUX_SYSCALL   // Mode 2: Standard Linux syscalls
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

#### Mode 1: Godot Syscalls
- Uses ECALL numbers from `modules/sandbox/src/syscalls.h` (500+)
- Example: `ECALL_VCALL = 501` for Variant calls
- Encoding: `li a7, <ecall_number>; ecall`
- Requires sandbox runtime with Godot syscall handlers

#### Mode 2: Linux Syscalls
- Uses standard Linux syscall numbers
- Example: `write = 64`, `exit = 93`, `read = 63`
- Encoding: `li a7, <syscall_number>; ecall`
- Requires Linux kernel or compatible runtime

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
# Mode 1: Godot syscalls (default)
var elf_godot = script.compile_all_functions_to_elf64(ELF64CompilationMode.GODOT_SYSCALL)

# Mode 2: Linux syscalls
var elf_linux = script.compile_all_functions_to_elf64(ELF64CompilationMode.LINUX_SYSCALL)
```

### 8. CLI Tool Updates

The `gdscript_elf64_compiler.gd` tool should support mode selection:

```bash
# Mode 1 (default)
godot --headless --script tools/gdscript_elf64_compiler.gd script.gd --mode godot

# Mode 2
godot --headless --script tools/gdscript_elf64_compiler.gd script.gd --mode linux
```

## Benefits

1. **Flexibility**: Choose execution environment based on use case
2. **Portability**: Linux mode enables running on real RISC-V hardware
3. **Integration**: Godot mode enables tight integration with engine
4. **Testing**: Can test same code in both environments

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
