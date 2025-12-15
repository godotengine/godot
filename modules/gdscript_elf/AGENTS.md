# GDScript ELF Module - Agent Documentation

## Project Overview

The `gdscript_elf` module implements a GDScript bytecode-to-ELF compiler that converts GDScript functions to RISC-V ELF executables. This follows the BeamAsm philosophy: load-time conversion (not runtime JIT), eliminating dispatch overhead while maintaining the same register model as the interpreter.

**Key Technologies:**

-   C++ (Godot engine module)
-   C code generation (from GDScript bytecode)
-   RISC-V cross-compiler (riscv64-unknown-elf-gcc)
-   ELF64 binary format
-   GDScript bytecode

**Architecture Pattern:** Strangler Vine - gradual migration from VM to ELF compilation

**Compilation Approach:** Generate C++ code from bytecode, then compile to ELF using RISC-V cross-compiler (simpler than raw instruction encoding)

## Development Environment Setup

### Prerequisites

-   Godot engine source code
-   SCons build system
-   C++ compiler with C++17 support
-   RISC-V cross-compiler (riscv64-unknown-elf-gcc or similar) in PATH
-   Access to `modules/gdscript` (read-only)
-   Access to `modules/sandbox` (read-only)

### Module Structure

```
modules/gdscript_elf/
├── AGENTS.md                          # This file
├── config.py                          # Module configuration
├── register_types.h/cpp               # Module registration
├── SCsub                              # Build configuration
├── src/                               # Source files
└── tests/                             # Test files (header-only)
```

### Build Configuration

The module is configured in `config.py` and built via `SCsub`. Module initialization occurs at `MODULE_INITIALIZATION_LEVEL_SERVERS` (after gdscript module initializes).

## Build and Test Commands

### Building the Module

```bash
# Build Godot with gdscript_elf module
scons target=template_debug

# Module is automatically included when building Godot
```

### Running Tests

```bash
# Run Godot test suite (includes module tests)
godot --test

# Tests are header-only files in tests/ directory
# Automatically registered via doctest framework
```

### Module-Specific Test Cases

Test cases follow the pattern: `TEST_CASE("[GDScript][ELF] Description")` in namespace `TestGDScriptELF`.

## Code Style and Conventions

### Naming Conventions

-   **Classes**: PascalCase (e.g., `GDScriptELFWriter`)
-   **Files**: snake_case matching class name (e.g., `gdscript_elf_writer.h`)
-   **Methods**: snake_case (e.g., `compile_function_to_elf()`)
-   **Constants**: UPPER_SNAKE_CASE (e.g., `OPCODE_LOAD`)

### Code Organization

-   Header files in `src/` directory
-   Implementation files in `src/` directory
-   Test files in `tests/` directory (header-only)
-   Editor files in `editor/` directory (if needed)

### Godot Conventions

-   Use `ERR_FAIL_*` macros for error handling
-   Use `memnew()` / `memdelete()` for memory management
-   Follow Godot's coding style (see `doc/contributing/development/code_style_guidelines.md`)
-   Use `GDCLASS()` macro for Godot classes

## Testing Guidelines

### Test Framework

-   **Framework**: doctest via `tests/test_macros.h`
-   **Structure**: Header-only test files
-   **Namespace**: `namespace TestGDScriptELF { ... }`
-   **Pattern**: `TEST_CASE("[Category][Subcategory] Description")`

### Test Categories

-   `[GDScript][ELF]` - ELF compilation tests (C code generation, cross-compiler detection, ELF compiler)
-   `[GDScript][Fallback]` - Fallback mechanism tests (opcode support detection, statistics tracking)
-   `[RISC-V][Encoder]` - Instruction encoding tests (not applicable with C code generation approach)

**Test File**: `tests/test_gdscript_bytecode_elf.h` - Test suite covering:
- C code generation functionality
- Fallback mechanism (opcode support, statistics)
- C compiler detection and availability
- ELF compiler basic functionality
- **Phase 3 Testing**: ELF execution integration tests (see Testing Status below)

### Writing Tests

```cpp
#include "tests/test_macros.h"

namespace TestGDScriptELF {
    TEST_CASE("[GDScript][ELF] Bytecode to ELF compilation") {
        // Test implementation
    }
}
```

## Implementation Status

### Completed Components

-   Module structure and configuration
-   `GDScriptLanguageWrapper` (100% pass-through)
-   `GDScriptWrapper` (100% pass-through)
-   Module registration (replaces original GDScriptLanguage)
-   `GDScriptBytecodeCCodeGenerator` - Generates C++ code from bytecode
    -   Phase 2: Direct C code generation for all common opcodes
    -   Address resolution for stack, constants, members
    -   Label pre-generation for forward jumps
    -   Syscall generation for property access and method calls
-   `GDScriptCCompiler` - Invokes RISC-V cross-compiler via shell
-   `GDScriptBytecodeELFCompiler` - Orchestrates C generation + compilation
-   `GDScriptELFFallback` - VM fallback mechanism (C interface)
    -   Phase 2: `is_opcode_supported()` tracks all implemented opcodes
    -   Fallback statistics tracking for migration progress
-   `GDScriptFunctionWrapper` - Function execution wrapper
-   Integration hooks in `GDScriptWrapper::reload()` (Phase 1: validation)
-   Build system updates (`SCsub`)
-   Test suite (`tests/test_gdscript_bytecode_elf.h`) - Comprehensive tests for C code generation and compilation

### Phase 1 Status - ✅ COMPLETE

-   All core components implemented and integrated
-   C code generation works for all functions
-   Cross-compiler invocation implemented
-   ELF compilation end-to-end pipeline complete
-   Fallback mechanism ready for integration

### Phase 2 Status - ✅ COMPLETE

**Completed:**
-   Direct C code generation for simple opcodes:
    -   Assignments: `OPCODE_ASSIGN`, `OPCODE_ASSIGN_NULL`, `OPCODE_ASSIGN_TRUE`, `OPCODE_ASSIGN_FALSE`
    -   Control flow: `OPCODE_JUMP`, `OPCODE_JUMP_IF`, `OPCODE_JUMP_IF_NOT`
    -   Arithmetic: `OPCODE_OPERATOR_VALIDATED` (uses validated operator evaluator)
    -   Function control: `OPCODE_RETURN`, `OPCODE_END`, `OPCODE_LINE`
    -   Property access: `OPCODE_GET_MEMBER`, `OPCODE_SET_MEMBER` (via syscall inline assembly)
    -   Method calls: `OPCODE_CALL`, `OPCODE_CALL_RETURN` (via syscall with argument marshaling)
-   Address resolution for stack, constants, and members
-   Label pre-generation for forward jumps
-   Function signature updated to include `constants` and `operator_funcs` parameters
-   Helper function declarations for global name access (`get_global_name_cstr()`)
-   Comprehensive test suite for C code generation and compilation

### Phase 3 Status - ✅ COMPLETE

**Completed:**
-   **Sandbox instance management**: Per-instance sandbox creation and caching via static map
    -   `get_or_create_sandbox()` creates sandbox on-demand for each `GDScriptInstance`
    -   `cleanup_sandbox()` removes sandbox when instance is destroyed
    -   Sandbox instances are cached to avoid recreation on every function call
-   **ELF binary loading**: ELF binaries loaded into sandbox via `load_buffer()`
    -   Loading state checked to avoid reloading on every call
    -   Graceful fallback to VM if loading fails
-   **Function address resolution**: Function addresses resolved from ELF symbol table
    -   Function names match C code generation convention (`gdscript_<function_name>`)
    -   Addresses cached per function wrapper to avoid repeated lookups
    -   Uses `sandbox->address_of()` to resolve symbols
-   **Argument marshaling**: Basic argument passing via `vmcall_address()`
    -   Arguments passed directly as `Variant**` array
    -   Return values extracted from sandbox execution
-   **Error handling**: Comprehensive error handling with VM fallback at each step
    -   Sandbox creation failure → VM fallback
    -   ELF loading failure → VM fallback
    -   Function address resolution failure → VM fallback
    -   `vmcall_address()` errors → VM fallback
    -   All errors logged for debugging
-   **Fallback mechanism**: Function-level fallback to VM when ELF execution fails
    -   Existing opcode-level fallback statistics tracking maintained
    -   Seamless fallback ensures execution always succeeds

**Completed (Phase 3+):**
-   **Constants/operator functions parameter passing**: Implemented Option B - store constants and operator_funcs arrays in sandbox memory, pass addresses as Variant integers through extended args array. Functions requiring constants/operator_funcs now execute via ELF.

### Pending Implementation

-   Phase 2+: Additional method call opcodes (`OPCODE_CALL_METHOD_BIND`, `OPCODE_CALL_METHOD_BIND_RET`, etc.)
-   Phase 3 Testing: Comprehensive test suite for ELF execution integration (see Testing Status below)
-   Editor integration (optional)
-   Migration progress tracking UI (optional)

## Architecture Details

### Strangler Vine Pattern

**Phase 0**: 100% pass-through wrappers - all calls delegate to original GDScript ✅ COMPLETE

**Phase 1**: Generate ELF in parallel, but still use original for execution ✅ COMPLETE

**Phase 2**: Direct C code generation for supported opcodes ✅ COMPLETE

**Phase 3**: ELF execution integration with sandbox ✅ COMPLETE
-   ELF-compiled functions execute via sandbox when available
-   Fallback to VM for unsupported opcodes or execution errors
-   Constants/operator_funcs parameter passing implemented (Option B)

**Phase 2+**: Additional opcode support (ongoing)

### BeamAsm Philosophy

1. **Load-time conversion**: Convert at script load time (not runtime JIT)
2. **Eliminate dispatch overhead**: Direct native code execution
3. **Minimal optimizations**: Keep register model same, just eliminate dispatch
4. **Register arrays work same**: Stack/register model works same as interpreter
5. **Runtime system unchanged**: Only code loading changes
6. **Specialize on types**: Each instruction can be specialized based on argument types

### Key Components

**Wrappers:**

-   `GDScriptLanguageWrapper` - Wraps `GDScriptLanguage`
-   `GDScriptWrapper` - Wraps `GDScript` class
-   `GDScriptFunctionWrapper` - Wraps `GDScriptFunction` (intercepts execution)
    -   Phase 3: ELF execution integration with sandbox
    -   Sandbox instance management (per-instance)
    -   ELF binary loading and function address resolution
    -   Argument marshaling and return value extraction

**ELF Compilation (C Code Generation Approach):**

-   `GDScriptBytecodeCCodeGenerator` - Generates C++ code from bytecode
-   `GDScriptCCompiler` - Invokes RISC-V cross-compiler (riscv64-unknown-elf-gcc)
-   `GDScriptBytecodeELFCompiler` - Orchestrates C generation + compilation
-   `GDScriptELFFallback` - Fallback to VM for unsupported opcodes (C interface)

**Compilation Flow:**

```
GDScriptFunction bytecode
    ↓
GDScriptBytecodeCCodeGenerator
    ↓
C++ source code (temporary file)
    ↓
GDScriptCCompiler (shell: riscv64-unknown-elf-gcc)
    ↓
ELF binary (temporary file)
    ↓
PackedByteArray (loaded into memory)
    ↓
GDScriptFunctionWrapper::call()
    ↓
Sandbox::load_buffer() (if not already loaded)
    ↓
Sandbox::address_of() (resolve function symbol)
    ↓
Sandbox::vmcall_address() (execute ELF function)
    ↓
Return Variant result
```

## GDScript Bytecode to C Code Mapping

### Function Structure

Generated C++ function signature:
```c
void gdscript_function_name(void* instance, Variant* args, int argcount, Variant* result, Variant* constants, Variant::ValidatedOperatorEvaluator* operator_funcs) {
    Variant stack[STACK_SIZE];
    int ip = 0;

    // Function body (opcodes translated to C)
    // Phase 2: Direct C code for supported opcodes
    label_0:
    stack[3] = stack[1];  // OPCODE_ASSIGN
    if (stack[4].booleanize()) goto label_10;  // OPCODE_JUMP_IF
    {
        Variant::ValidatedOperatorEvaluator op_func = operator_funcs[0];
        op_func(&stack[1], &stack[2], &stack[0]);  // OPCODE_OPERATOR_VALIDATED
    }
    *result = stack[0];  // OPCODE_RETURN
    return;
}
```

### Opcode Mappings

**Phase 1**: All opcodes use fallback function calls ✅ COMPLETE
```c
gdscript_vm_fallback(OPCODE_OPERATOR, instance, stack, ip);
```

**Phase 2 (Current)**: Direct C code generation for simple opcodes ✅ COMPLETE

**Implemented:**
-   **Assignments**:
    -   `OPCODE_ASSIGN` → `stack[dst] = stack[src];`
    -   `OPCODE_ASSIGN_NULL` → `stack[dst] = Variant();`
    -   `OPCODE_ASSIGN_TRUE` → `stack[dst] = true;`
    -   `OPCODE_ASSIGN_FALSE` → `stack[dst] = false;`
-   **Control Flow**:
    -   `OPCODE_JUMP` → `goto label_X;`
    -   `OPCODE_JUMP_IF` → `if (condition.booleanize()) goto label_X;`
    -   `OPCODE_JUMP_IF_NOT` → `if (!condition.booleanize()) goto label_X;`
-   **Arithmetic**:
    -   `OPCODE_OPERATOR_VALIDATED` → Uses validated operator evaluator:
    ```c
    Variant::ValidatedOperatorEvaluator op_func = operator_funcs[idx];
    op_func(&left, &right, &dst);
    ```
-   **Function Control**:
    -   `OPCODE_RETURN` → `*result = return_value; return;`
    -   `OPCODE_END` → `return;`
    -   `OPCODE_LINE` → Metadata (skipped)
-   **Property Access**:
    -   `OPCODE_GET_MEMBER` → Syscall with inline assembly using `ECALL_OBJ_PROP_GET`:
    ```c
    register uint64_t object asm("a0") = (uint64_t)instance;
    register const char* property asm("a1") = name_cstr;
    register size_t property_size asm("a2") = name_len;
    register Variant* var_ptr asm("a3") = &stack[dst];
    register int syscall_number asm("a7") = ECALL_OBJ_PROP_GET;
    __asm__ volatile("ecall" : "=m"(*var_ptr) : ...);
    ```
    -   `OPCODE_SET_MEMBER` → Syscall with inline assembly using `ECALL_OBJ_PROP_SET`
-   **Method Calls**:
    -   `OPCODE_CALL` / `OPCODE_CALL_RETURN` → Syscall with inline assembly using `ECALL_VCALL`:
    ```c
    Variant call_args[argc];
    // ... populate call_args from instruction arguments ...
    register const Variant* object asm("a0") = &base;
    register const char* method_ptr asm("a1") = method_cstr;
    register size_t method_size asm("a2") = method_len;
    register const Variant* args_ptr asm("a3") = call_args;
    register size_t argcount_reg asm("a4") = argc;
    register Variant* ret_ptr asm("a5") = &call_result; // if RETURN
    register int syscall_number asm("a7") = ECALL_VCALL;
    __asm__ volatile("ecall" : ...);
    ```
-   **Constants**: GDScript constants → C variable access via `constants` array parameter (handled via address resolution)

**Phase 2+ (Pending):**
-   Additional method call opcodes: `OPCODE_CALL_METHOD_BIND`, `OPCODE_CALL_METHOD_BIND_RET`, etc.

### Syscall Pattern

Based on `modules/sandbox/src/syscalls.h`, syscalls use inline assembly:
```c
__asm__ volatile (
    "li a7, %0\n"
    "ecall\n"
    : : "i" (ECALL_NUMBER) : "a7"
);
```

## Dependencies

-   **modules/gdscript**: Read-only dependency (to wrap)
-   **modules/sandbox**: Read-only dependency (for ELF execution, syscall numbers)
-   **RISC-V Cross-Compiler**: Must be available in PATH
    -   Preferred: `riscv64-unknown-elf-gcc`
    -   Alternatives: `riscv64-linux-gnu-gcc`, `riscv64-elf-gcc`
    -   Auto-detected by `GDScriptCCompiler::detect_cross_compiler()`
-   **Temporary file system**: For C code and ELF output during compilation

## Security Considerations

-   **No code execution from untrusted sources**: ELF compilation is internal to Godot
-   **Sandbox isolation**: ELF executables run in sandbox (modules/sandbox)
-   **Input validation**: Validate GDScript bytecode before compilation
-   **Error handling**: All compilation errors return empty results, logged via `ERR_PRINT`

## Pull Request and Commit Guidelines

### Commit Messages

Follow Godot's commit message style:

-   First line: Brief summary (50 chars or less)
-   Blank line
-   Detailed explanation if needed
-   Reference issue numbers if applicable

Example:

```
Add RISC-V instruction encoder for ELF compilation

Implements RISCVInstructionEncoder class with support for all
RISC-V instruction formats (R/I/S/B/U/J-type). This is the
foundation for GDScript bytecode to ELF compilation.

Fixes #12345
```

### Code Review Checklist

-   [ ] Follows Godot coding style
-   [ ] No modifications to `modules/sandbox` or `modules/gdscript` (read-only)
-   [ ] Tests added/updated for new functionality
-   [ ] Documentation updated if needed
-   [ ] Build system (`SCsub`) updated if adding new files
-   [ ] Error handling implemented
-   [ ] Memory management correct (`memnew`/`memdelete`)

## Development Workflow

### Adding New Opcode Support

1. Add opcode-to-C translation in `GDScriptBytecodeCCodeGenerator::generate_opcode()`
2. Generate appropriate C code (or inline assembly for syscalls)
3. Update `GDScriptELFFallback::is_opcode_supported()` to return `true`
4. Add test case in `test_gdscript_bytecode_elf.h`
5. Update migration tracking
6. Verify C code compiles correctly with cross-compiler

### ELF Format Specifications

-   **Format**: ELF64 (64-bit)
-   **Architecture**: RISC-V (EM_RISCV = 243)
-   **Endianness**: Little-endian
-   **Entry Point**: Function address in `.text` section
-   **Sections**: `.text` (code), `.rodata` (constants), `.strtab` (strings), `.symtab` (symbols)
-   **Compiler Flags**: `-nostdlib -static -O0` (no standard library, static linking, minimal optimization)

**Note**: ELF format is generated by the RISC-V cross-compiler, not written directly. The compiler handles all ELF format details.

### Error Handling

-   C code generation errors return empty `String`, logged via `ERR_PRINT`
-   Compiler not found: Returns empty result, logs error
-   Compilation errors: Captures compiler stderr, returns empty result, logs errors
-   Temporary file errors: Clean up on error, return empty result
-   Unsupported opcodes use fallback mechanism
-   Fallback statistics tracked for migration progress
-   All errors logged via `ERR_PRINT` with context

## Execution Flow (Phase 3)

When `GDScriptFunctionWrapper::call()` is invoked:

1. **Check for ELF binary**: If `has_elf_code()` returns true, proceed with ELF execution
2. **Get/Create sandbox**: Retrieve or create sandbox instance for the `GDScriptInstance`
3. **Load ELF binary**: Load ELF into sandbox via `load_buffer()` if not already loaded
4. **Resolve function address**: Use `sandbox->address_of()` to get function address from symbol table
5. **Call function**: Execute via `sandbox->vmcall_address()` with marshaled arguments
6. **Extract result**: Return `Variant` result from sandbox execution
7. **Fallback on error**: If any step fails, fallback to original `GDScriptFunction::call()`

**Error Handling:**
-   All error paths log via `ERR_PRINT` and fallback to VM
-   Ensures execution always succeeds (graceful degradation)
-   No exceptions thrown - all errors handled gracefully

## Testing Status

### Current Test Coverage

**Phase 1 & 2 Tests**: ✅ Complete
- C code generation functionality
- Fallback mechanism (opcode support, statistics)
- C compiler detection and availability
- ELF compiler basic functionality

**Phase 3 Tests**: ⏳ Pending
- Sandbox instance management
- ELF binary loading
- Function address resolution
- Argument marshaling
- Return value extraction
- Constants/operator_funcs parameter passing
- Error handling and fallback
- End-to-end ELF execution

**Testing Readiness**: Not ready - Phase 3 tests need to be implemented before comprehensive testing can begin.

### Test Requirements

- RISC-V cross-compiler must be available for ELF execution tests
- Tests should gracefully skip when compiler is not found
- Integration tests require sandbox module dependency
- Some tests may require creating minimal GDScriptFunction instances

## References

-   RISC-V Specification: https://riscv.org/technical/specifications/
-   ELF64 Format: System V ABI specification
-   RISC-V Cross-Compiler: https://github.com/riscv/riscv-gnu-toolchain
-   Syscall patterns: `modules/sandbox/src/syscalls.h`
-   Godot Contributing Guide: `doc/contributing/development/`
