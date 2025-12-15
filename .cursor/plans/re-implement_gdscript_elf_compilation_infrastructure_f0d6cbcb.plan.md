---
name: Re-implement GDScript ELF Compilation Infrastructure
overview: Re-implement the ELF compilation infrastructure that was removed, following the BeamAsm philosophy and strangler vine pattern. Start from the current state (basic wrappers exist) and build out the full bytecode-to-ELF compilation system.
todos:
  - id: riscv_encoder_header
    content: Recreate riscv_instruction_encoder.h header (cpp file exists)
    status: pending
  - id: elf_writer
    content: Implement GDScriptELFWriter for ELF64 format generation
    status: pending
  - id: code_generator
    content: Implement GDScriptBytecodeELFCodeGenerator for bytecode-to-RISC-V translation
    status: pending
  - id: elf_compiler
    content: Implement GDScriptBytecodeELFCompiler to orchestrate compilation
    status: pending
  - id: fallback_mechanism
    content: Implement GDScriptELFFallback for VM fallback support
    status: pending
  - id: function_wrapper
    content: Implement GDScriptFunctionWrapper to intercept function execution
    status: pending
  - id: wrapper_integration
    content: Update GDScriptWrapper::reload() to generate ELF in parallel
    status: pending
  - id: build_system
    content: Update SCsub to include all new source files
    status: pending
---

# Re-implement GDScript ELF Compilation Infrastructure

## Current State

- **Existing**: `GDScriptLanguageWrapper` and `GDScriptWrapper` (100% pass-through wrappers)
- **Missing**: All ELF compilation infrastructure (compiler, codegen, encoder, writer, fallback, tests, editor integration)

## Implementation Plan

### Phase 1: Core ELF Compilation Infrastructure

#### 1.1 RISC-V Instruction Encoder

- **Files**: `modules/gdscript_elf/src/riscv_instruction_encoder.h` and `.cpp`
- **Purpose**: Encode RISC-V assembly instructions as 32-bit binary
- **Key Features**:
  - Support all instruction formats (R/I/S/B/U/J-type)
  - Helper methods: `encode_add()`, `encode_addi()`, `encode_ld()`, `encode_sd()`, `encode_beq()`, `encode_jal()`, `encode_ecall()`, `encode_ret()`, `encode_nop()`
  - Little-endian byte encoding
- **Note**: `riscv_instruction_encoder.cpp` exists but header was deleted - recreate header

#### 1.2 ELF Writer

- **Files**: `modules/gdscript_elf/src/gdscript_elf_writer.h` and `.cpp`
- **Purpose**: Write ELF64 format directly (no external toolchain)
- **Key Features**:
  - ELF header generation (64-bit, RISC-V, executable)
  - Program headers (loadable segments)
  - Section headers (`.text`, `.rodata`, `.strtab`, `.symtab`)
  - String table and symbol table generation
  - Helper methods: `write_u64()`, `write_u32()`, `write_u16()`, `write_u8()`

#### 1.3 Bytecode to RISC-V Code Generator

- **Files**: `modules/gdscript_elf/src/gdscript_bytecode_elf_codegen.h` and `.cpp`
- **Purpose**: Generate RISC-V assembly from GDScript bytecode
- **Key Features**:
  - Function prologue/epilogue generation
  - Opcode-to-instruction mapping (Phase 1: all opcodes use fallback)
  - Fallback call generation for unsupported opcodes
  - Stack frame management
- **BeamAsm Philosophy**: Minimal optimizations, same register model as interpreter

#### 1.4 ELF Compiler

- **Files**: `modules/gdscript_elf/src/gdscript_bytecode_elf_compiler.h` and `.cpp`
- **Purpose**: Orchestrate bytecode-to-ELF compilation
- **Key Features**:
  - `compile_function_to_elf()` - Main compilation entry point
  - `can_compile_function()` - Check if function can be compiled
  - `get_unsupported_opcodes()` - Get list of unsupported opcodes
  - Integration with code generator and ELF writer

### Phase 2: Fallback and Function Wrapper

#### 2.1 Fallback Mechanism

- **Files**: `modules/gdscript_elf/src/gdscript_elf_fallback.h` and `.cpp`
- **Purpose**: Bridge between ELF-compiled code and original GDScript VM
- **Key Features**:
  - `call_original_function()` - Call back to GDScriptFunction::call()
  - `record_fallback_opcode()` - Track which opcodes use fallback
  - `get_fallback_statistics()` - Migration progress tracking
  - `is_opcode_supported()` - Check opcode support (Phase 1: all return false)

#### 2.2 Function Wrapper

- **Files**: `modules/gdscript_elf/src/gdscript_function_wrapper.h` and `.cpp`
- **Purpose**: Wrap GDScriptFunction to intercept execution
- **Key Features**:
  - Store pointer to original GDScriptFunction
  - `call()` method - Phase 0: delegate to original, Phase 1+: use ELF if available
  - ELF code storage (placeholder for now)
  - Integration with fallback mechanism

### Phase 3: Integration

#### 3.1 Update Wrapper Integration

- **File**: `modules/gdscript_elf/src/gdscript_wrapper.cpp`
- **Changes**: 
  - Add include for `gdscript_bytecode_elf_compiler.h`
  - In `reload()`, add hook to generate ELF in parallel (Phase 1: validation only)
  - Store compiled ELF for future use (Phase 2+)

#### 3.2 Update Build System

- **File**: `modules/gdscript_elf/SCsub`
- **Changes**: Add all new source files to build:
  ```python
  sources = [
      "src/gdscript_language_wrapper.cpp",
      "src/gdscript_wrapper.cpp",
      "src/gdscript_function_wrapper.cpp",
      "src/gdscript_bytecode_elf_compiler.cpp",
      "src/gdscript_bytecode_elf_codegen.cpp",
      "src/riscv_instruction_encoder.cpp",
      "src/gdscript_elf_writer.cpp",
      "src/gdscript_elf_fallback.cpp",
  ]
  ```


### Phase 4: Testing (Optional)

#### 4.1 Test Suite

- **File**: `modules/gdscript_elf/tests/test_gdscript_bytecode_elf.h`
- **Framework**: doctest via `tests/test_macros.h`
- **Test Cases**:
  - Bytecode to ELF compilation
  - Fallback mechanism
  - Opcode support tracking
  - RISC-V instruction encoding

### Phase 5: Editor Integration (Optional)

#### 5.1 Editor Plugin

- **Files**: `modules/gdscript_elf/editor/gdscript_elf_export_plugin.h` and `.cpp`
- **Purpose**: UI for tracking ELF compilation progress
- **Features**: Compilation status, error display, migration tracking

## Implementation Order

1. **RISC-V Instruction Encoder** (foundation)
2. **ELF Writer** (depends on encoder)
3. **Code Generator** (depends on encoder)
4. **ELF Compiler** (orchestrates codegen + writer)
5. **Fallback Mechanism** (standalone)
6. **Function Wrapper** (depends on fallback)
7. **Integration** (wire everything together)
8. **Build System** (add files to SCsub)
9. **Tests** (optional)
10. **Editor Integration** (optional)

## Key Design Decisions

- **Strangler Vine Pattern**: Phase 0 (current) = 100% pass-through, Phase 1 = generate ELF but don't use it, Phase 2+ = gradually use ELF
- **BeamAsm Philosophy**: Load-time conversion, eliminate dispatch overhead, minimal optimizations
- **Phase 1 Opcode Support**: All opcodes use fallback - infrastructure is ready for gradual migration
- **No Sandbox Modifications**: All code in `modules/gdscript_elf`, sandbox remains read-only

## Files to Create

1. `src/riscv_instruction_encoder.h` (recreate - .cpp exists)
2. `src/gdscript_elf_writer.h` and `.cpp`
3. `src/gdscript_bytecode_elf_codegen.h` and `.cpp`
4. `src/gdscript_bytecode_elf_compiler.h` and `.cpp`
5. `src/gdscript_elf_fallback.h` and `.cpp`
6. `src/gdscript_function_wrapper.h` and `.cpp`
7. `tests/test_gdscript_bytecode_elf.h` (optional)
8. `editor/gdscript_elf_export_plugin.h` and `.cpp` (optional)

## Files to Modify

1. `src/gdscript_wrapper.cpp` - Add ELF compilation hook in `reload()`
2. `SCsub` - Add all new source files to build