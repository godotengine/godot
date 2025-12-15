# GDScript ELF Module - Agent Documentation

## Project Overview

The `gdscript_elf` module implements a GDScript bytecode-to-ELF compiler that converts GDScript functions to RISC-V ELF executables. This follows the BeamAsm philosophy: load-time conversion (not runtime JIT), eliminating dispatch overhead while maintaining the same register model as the interpreter.

**Key Technologies:**

-   C++ (Godot engine module)
-   RISC-V assembly
-   ELF64 binary format
-   GDScript bytecode

**Architecture Pattern:** Strangler Vine - gradual migration from VM to ELF compilation

## Development Environment Setup

### Prerequisites

-   Godot engine source code
-   SCons build system
-   C++ compiler with C++17 support
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

-   `[GDScript][ELF]` - ELF compilation tests
-   `[GDScript][Fallback]` - Fallback mechanism tests
-   `[RISC-V][Encoder]` - Instruction encoding tests

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
-   `RISCVInstructionEncoder` (cpp file exists, header needed)

### Pending Implementation

-   `RISCVInstructionEncoder` header file
-   `GDScriptELFWriter` - ELF64 format writer
-   `GDScriptBytecodeELFCodeGenerator` - Bytecode to RISC-V translation
-   `GDScriptBytecodeELFCompiler` - Compilation orchestration
-   `GDScriptELFFallback` - VM fallback mechanism
-   `GDScriptFunctionWrapper` - Function execution wrapper
-   Integration hooks in `GDScriptWrapper::reload()`
-   Build system updates (`SCsub`)
-   Test suite
-   Editor integration (optional)

## Architecture Details

### Strangler Vine Pattern

**Phase 0 (Current)**: 100% pass-through wrappers - all calls delegate to original GDScript

**Phase 1**: Generate ELF in parallel, but still use original for execution

**Phase 2+**: Gradually replace opcodes with ELF-compiled versions, fallback to original for unsupported opcodes

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
-   `GDScriptFunctionWrapper` - Wraps `GDScriptFunction` (to be implemented)

**ELF Compilation:**

-   `RISCVInstructionEncoder` - Encodes RISC-V instructions
-   `GDScriptELFWriter` - Writes ELF64 format
-   `GDScriptBytecodeELFCodeGenerator` - Generates RISC-V from bytecode
-   `GDScriptBytecodeELFCompiler` - Orchestrates compilation
-   `GDScriptELFFallback` - Fallback to VM for unsupported opcodes

## GDScript Bytecode to RISC-V Mapping

### Function Structure

-   **Prologue**: `addi sp, sp, -frame_size` (allocate stack frame)
-   **Epilogue**: `addi sp, sp, frame_size; ret` (restore stack, return)

### Opcode Mappings

**Phase 1**: All opcodes use fallback

**Phase 2+**: Implement gradually

-   Arithmetic: `OPCODE_OPERATOR` → RISC-V arithmetic (`add`, `sub`, `mul`, `div`)
-   Method Calls: `OPCODE_CALL_METHOD` → `li a7, ECALL_VCALL; ecall`
-   Property Access: `OPCODE_GET_MEMBER`/`SET_MEMBER` → syscalls
-   Control Flow: `OPCODE_JUMP`, `OPCODE_JUMP_IF` → RISC-V branches/jumps

### Constants and Globals

-   Small constants: Use immediate operands (`addi`, `ori`)
-   Large constants: Load from `.rodata` section
-   Global names: ELF symbol table entries

## Dependencies

-   **modules/gdscript**: Read-only dependency (to wrap)
-   **modules/sandbox**: Read-only dependency (for ELF execution, syscall numbers)
-   **No external toolchains**: Direct ELF writing, no external assemblers/compilers

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

1. Add opcode mapping in `GDScriptBytecodeELFCodeGenerator::generate_opcode()`
2. Generate appropriate RISC-V instructions
3. Update `GDScriptELFFallback::is_opcode_supported()` to return `true`
4. Add test case in `test_gdscript_bytecode_elf.h`
5. Update migration tracking

### ELF Format Specifications

-   **Format**: ELF64 (64-bit)
-   **Architecture**: RISC-V (EM_RISCV = 243)
-   **Endianness**: Little-endian
-   **Entry Point**: Function address in `.text` section
-   **Sections**: `.text` (code), `.rodata` (constants), `.strtab` (strings), `.symtab` (symbols)

### Error Handling

-   Compilation errors return empty `PackedByteArray`
-   Unsupported opcodes use fallback mechanism
-   Fallback statistics tracked for migration progress
-   All errors logged via `ERR_PRINT`

## References

-   Original plan: `.cursor/plans/gdscript_to_elf_with_nostradamus_distributor_f0d6cbcb.plan.md`
-   Re-implementation plan: `.cursor/plans/re-implement_gdscript_elf_compilation_infrastructure_f0d6cbcb.plan.md`
-   RISC-V Specification: https://riscv.org/technical/specifications/
-   ELF64 Format: System V ABI specification
-   Godot Contributing Guide: `doc/contributing/development/`
