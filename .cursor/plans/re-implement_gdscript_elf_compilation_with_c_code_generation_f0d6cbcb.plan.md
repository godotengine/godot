---
name: Re-implement GDScript ELF Compilation with C Code Generation
overview: Re-implement ELF compilation infrastructure using C code generation and RISC-V cross-compiler instead of raw instruction encoding. Generate readable C code from GDScript bytecode, then compile to ELF using riscv64-unknown-elf-gcc (or similar) via shell invocation.
todos:
  - id: c_code_generator
    content: Implement GDScriptBytecodeCCodeGenerator to generate C code from bytecode
    status: in_progress
  - id: c_compiler_invocation
    content: Implement GDScriptCCompiler to invoke RISC-V cross-compiler via shell
    status: pending
  - id: elf_compiler
    content: Update GDScriptBytecodeELFCompiler to orchestrate C generation + compilation
    status: pending
  - id: fallback_mechanism
    content: Implement GDScriptELFFallback for VM fallback support
    status: pending
  - id: function_wrapper
    content: Implement GDScriptFunctionWrapper to intercept function execution
    status: pending
  - id: wrapper_integration
    content: Update GDScriptWrapper::reload() to generate and compile C code
    status: pending
  - id: build_system
    content: Update SCsub to include new files and remove riscv_instruction_encoder.cpp
    status: pending
---

# Re-implement GDScript ELF Compilation with C Code Generation

## Current State

### Completed

- Module structure and configuration
- `GDScriptLanguageWrapper` (100% pass-through wrapper)
- `GDScriptWrapper` (100% pass-through wrapper)
- Module registration (replaces original GDScriptLanguage)
- `RISCVInstructionEncoder` implementation exists but is no longer needed (we'll use C compiler instead)
- Agent documentation ([`AGENTS.md`](modules/gdscript_elf/AGENTS.md))

### New Approach

Instead of generating raw RISC-V instructions and writing ELF directly, we will:

1. Generate C code from GDScript bytecode
2. Invoke RISC-V cross-compiler (riscv64-unknown-elf-gcc) via shell command
3. Load the resulting ELF binary

This simplifies implementation significantly - no need for instruction encoding or ELF format writing.

## Implementation Plan

### Phase 1: C Code Generation Infrastructure

#### 1.1 Bytecode to C Code Generator

- **Files**: [`modules/gdscript_elf/src/gdscript_bytecode_c_codegen.h`](modules/gdscript_elf/src/gdscript_bytecode_c_codegen.h) and `.cpp`
- **Purpose**: Generate C code from GDScript bytecode
- **Key Features**:
  - Function signature generation (matching GDScript function signature)
  - Stack frame management (local variables as C variables)
  - Opcode-to-C translation (Phase 1: all opcodes use fallback function calls)
  - Syscall generation using inline assembly (see [`modules/sandbox/src/syscalls.h`](modules/sandbox/src/syscalls.h) for patterns)
  - Constants and globals as C variables/initializers
- **C Code Structure**:
  ```c
  // Generated function
  void gdscript_function_name(void* instance, Variant* args, int argcount, Variant* result) {
      // Local variables (from stack slots)
      Variant var_0, var_1, ...;
      
      // Function body (opcodes translated to C)
      // Phase 1: All opcodes call fallback function
      gdscript_vm_fallback(OPCODE_OPERATOR, ...);
      
      // Phase 2+: Direct C code
      // Arithmetic: var_0 = var_1 + var_2;
      // Syscalls: inline assembly with ecall
      __asm__ volatile ("li a7, %0\n ecall\n" : : "i" (ECALL_VCALL) : "a7");
  }
  ```


#### 1.2 C Compiler Invocation

- **Files**: [`modules/gdscript_elf/src/gdscript_c_compiler.h`](modules/gdscript_elf/src/gdscript_c_compiler.h) and `.cpp`
- **Purpose**: Invoke RISC-V cross-compiler to compile C to ELF
- **Key Features**:
  - Auto-detect cross-compiler from PATH (riscv64-unknown-elf-gcc, riscv64-linux-gnu-gcc, etc.)
  - Write C code to temporary file
  - Invoke compiler via shell: `riscv64-unknown-elf-gcc -o output.elf -nostdlib -static input.c`
  - Read resulting ELF binary
  - Clean up temporary files
  - Error handling and compilation error reporting
- **Compiler Flags**:
  - `-nostdlib`: No standard library (sandbox provides syscalls)
  - `-static`: Static linking
  - `-fPIC` or `-fno-pic`: Position independent code (as needed)
  - `-O0` or `-O1`: Minimal optimization (BeamAsm philosophy)

#### 1.3 ELF Compiler (Orchestration)

- **Files**: [`modules/gdscript_elf/src/gdscript_bytecode_elf_compiler.h`](modules/gdscript_elf/src/gdscript_bytecode_elf_compiler.h) and `.cpp`
- **Purpose**: Orchestrate bytecode-to-ELF compilation via C generation
- **Key Features**:
  - `compile_function_to_elf()` - Main entry point
    - Calls C code generator to generate C code
    - Invokes C compiler to compile C to ELF
    - Returns ELF binary as `PackedByteArray`
  - `can_compile_function()` - Check if function can be compiled
  - `get_unsupported_opcodes()` - Get list of unsupported opcodes
  - Cross-compiler detection and validation

### Phase 2: Fallback and Function Wrapper

#### 2.1 Fallback Mechanism

- **Files**: [`modules/gdscript_elf/src/gdscript_elf_fallback.h`](modules/gdscript_elf/src/gdscript_elf_fallback.h) and `.cpp`
- **Purpose**: Bridge between ELF-compiled code and original GDScript VM
- **Key Features**:
  - `call_original_function()` - Call back to GDScriptFunction::call()
  - `record_fallback_opcode()` - Track which opcodes use fallback
  - `get_fallback_statistics()` - Migration progress tracking
  - `is_opcode_supported()` - Check opcode support (Phase 1: all return false)
- **C Interface**: Generate C function that calls back to C++ fallback:
  ```c
  extern void gdscript_vm_fallback(int opcode, void* instance, Variant* stack, int ip);
  ```


#### 2.2 Function Wrapper

- **Files**: [`modules/gdscript_elf/src/gdscript_function_wrapper.h`](modules/gdscript_elf/src/gdscript_function_wrapper.h) and `.cpp`
- **Purpose**: Wrap GDScriptFunction to intercept execution
- **Key Features**:
  - Store pointer to original GDScriptFunction
  - Store compiled ELF binary
  - `call()` method - Phase 0: delegate to original, Phase 1+: load and execute ELF
  - Integration with fallback mechanism

### Phase 3: Integration

#### 3.1 Update Wrapper Integration

- **File**: [`modules/gdscript_elf/src/gdscript_wrapper.cpp`](modules/gdscript_elf/src/gdscript_wrapper.cpp)
- **Changes**:
  - Add include for `gdscript_bytecode_elf_compiler.h`
  - In `reload()`, add hook to generate C code and compile to ELF (Phase 1: validation only)
  - Store compiled ELF for future use (Phase 2+)

#### 3.2 Update Build System

- **File**: [`modules/gdscript_elf/SCsub`](modules/gdscript_elf/SCsub)
- **Changes**: Add new source files (remove riscv_instruction_encoder.cpp, no longer needed):
  ```python
  sources = [
      "src/gdscript_language_wrapper.cpp",
      "src/gdscript_wrapper.cpp",
      "src/gdscript_function_wrapper.cpp",
      "src/gdscript_bytecode_elf_compiler.cpp",
      "src/gdscript_bytecode_c_codegen.cpp",
      "src/gdscript_c_compiler.cpp",
      "src/gdscript_elf_fallback.cpp",
  ]
  ```


### Phase 4: Testing (Optional)

#### 4.1 Test Suite

- **File**: [`modules/gdscript_elf/tests/test_gdscript_bytecode_elf.h`](modules/gdscript_elf/tests/test_gdscript_bytecode_elf.h)
- **Framework**: doctest via `tests/test_macros.h`
- **Test Cases**:
  - C code generation from bytecode
  - Cross-compiler invocation
  - ELF compilation end-to-end
  - Fallback mechanism
  - Opcode support tracking

## Implementation Order

1. **C Code Generator** - Generate C code from bytecode
2. **C Compiler Invocation** - Shell call to cross-compiler
3. **ELF Compiler** - Orchestrate C generation + compilation
4. **Fallback Mechanism** - VM fallback support
5. **Function Wrapper** - Intercept execution
6. **Integration** - Wire everything together
7. **Build System** - Update SCsub (remove riscv_instruction_encoder.cpp)
8. **Tests** (optional)

## Key Design Decisions

- **C Code Generation**: Much easier than raw instruction encoding
- **Cross-Compiler**: Use standard RISC-V toolchain (riscv64-unknown-elf-gcc)
- **Syscalls**: Use inline assembly in generated C code (pattern from [`modules/sandbox/src/syscalls.h`](modules/sandbox/src/syscalls.h))
- **Temporary Files**: Write C code to temp file, compile, read ELF, cleanup
- **Auto-Detection**: Try common cross-compiler names from PATH
- **No Instruction Encoder**: Not needed - compiler handles it
- **No ELF Writer**: Not needed - compiler generates ELF

## GDScript Bytecode to C Mapping

### Function Structure

```c
void gdscript_function_name(void* instance, Variant* args, int argcount, Variant* result) {
    // Local variables (from stack)
    Variant stack[STACK_SIZE];
    
    // Function body
    // Opcodes translated to C operations
}
```

### Opcode Mappings (Phase 1: All use fallback, Phase 2+: Implement gradually)

- **Arithmetic**: `OPCODE_OPERATOR` → C arithmetic operators (`+`, `-`, `*`, `/`)
- **Method Calls**: `OPCODE_CALL_METHOD` → Inline assembly syscall:
  ```c
  __asm__ volatile ("li a7, %0\n ecall\n" : : "i" (ECALL_VCALL) : "a7");
  ```

- **Property Access**: `OPCODE_GET_MEMBER`/`SET_MEMBER` → Syscalls with inline assembly
- **Control Flow**: `OPCODE_JUMP`, `OPCODE_JUMP_IF` → C `goto` or `if` statements
- **Constants**: GDScript constants → C variable initializers or literals

### Syscall Pattern

Based on [`modules/sandbox/src/syscalls.h`](modules/sandbox/src/syscalls.h), syscalls use:

```c
__asm__ volatile (
    "li a7, %0\n"
    "ecall\n"
    : : "i" (ECALL_NUMBER) : "a7"
);
```

## Files to Create

1. [`src/gdscript_bytecode_c_codegen.h`](modules/gdscript_elf/src/gdscript_bytecode_c_codegen.h) and `.cpp` - Generate C code from bytecode
2. [`src/gdscript_c_compiler.h`](modules/gdscript_elf/src/gdscript_c_compiler.h) and `.cpp` - Invoke cross-compiler
3. [`src/gdscript_bytecode_elf_compiler.h`](modules/gdscript_elf/src/gdscript_bytecode_elf_compiler.h) and `.cpp` - Orchestrate compilation
4. [`src/gdscript_elf_fallback.h`](modules/gdscript_elf/src/gdscript_elf_fallback.h) and `.cpp` - Fallback mechanism
5. [`src/gdscript_function_wrapper.h`](modules/gdscript_elf/src/gdscript_function_wrapper.h) and `.cpp` - Function wrapper
6. [`tests/test_gdscript_bytecode_elf.h`](modules/gdscript_elf/tests/test_gdscript_bytecode_elf.h) (optional)

## Files to Modify

1. [`src/gdscript_wrapper.cpp`](modules/gdscript_elf/src/gdscript_wrapper.cpp) - Add ELF compilation hook in `reload()`
2. [`SCsub`](modules/gdscript_elf/SCsub) - Update source files list (remove riscv_instruction_encoder.cpp)

## Files to Remove (No Longer Needed)

1. [`src/riscv_instruction_encoder.cpp`](modules/gdscript_elf/src/riscv_instruction_encoder.cpp) - Not needed with C code generation approach

## Dependencies

- **RISC-V Cross-Compiler**: Must be available in PATH (riscv64-unknown-elf-gcc or similar)
- **modules/gdscript**: Read-only dependency (to wrap)
- **modules/sandbox**: Read-only dependency (for syscall numbers, ELF execution)
- **Temporary file system**: For C code and ELF output during compilation

## Compilation Flow

```
GDScriptFunction bytecode
    ↓
GDScriptBytecodeCCodeGenerator
    ↓
C source code (temporary file)
    ↓
GDScriptCCompiler (shell: riscv64-unknown-elf-gcc)
    ↓
ELF binary (temporary file)
    ↓
PackedByteArray (loaded into memory)
    ↓
Sandbox execution
```

## Error Handling

- C code generation errors: Return empty result, log via `ERR_PRINT`
- Compiler not found: Return empty result, log error
- Compilation errors: Capture compiler stderr, return empty result, log errors
- Temporary file errors: Clean up on error, return empty result
- All errors logged via `ERR_PRINT` with context