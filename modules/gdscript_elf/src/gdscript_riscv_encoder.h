/**************************************************************************/
/*  gdscript_riscv_encoder.h                                            */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "core/variant/variant.h"
#include "gdscript_elf64_mode.h"

// Forward declaration
class GDScriptFunction;

// Encode GDScript bytecode to RISC-V machine instructions
class GDScriptRISCVEncoder {
public:
	// Encode GDScript bytecode opcodes to RISC-V instructions
	static PackedByteArray encode_function(GDScriptFunction *p_function, ELF64CompilationMode p_mode = ELF64CompilationMode::HYBRID);

	// Encode single opcode to RISC-V instruction sequence
	static PackedByteArray encode_opcode(int p_opcode, const int *p_code_ptr, int &p_ip, int p_code_size, ELF64CompilationMode p_mode = ELF64CompilationMode::HYBRID);

private:
	// RISC-V instruction encoding helpers (little-endian)
	static uint32_t encode_r_type(uint8_t opcode, uint8_t rd, uint8_t funct3, uint8_t rs1, uint8_t rs2, uint8_t funct7);
	static uint32_t encode_i_type(uint8_t opcode, uint8_t rd, uint8_t funct3, uint8_t rs1, int16_t imm);
	static uint32_t encode_s_type(uint8_t opcode, uint8_t funct3, uint8_t rs1, uint8_t rs2, int16_t imm);
	static uint32_t encode_u_type(uint8_t opcode, uint8_t rd, int32_t imm);
	static uint32_t encode_j_type(uint8_t opcode, uint8_t rd, int32_t imm);

	// Generate call to VM fallback (ecall or function call)
	static PackedByteArray encode_vm_call(int p_opcode, int p_ip, ELF64CompilationMode p_mode = ELF64CompilationMode::HYBRID);

	// Mode-specific syscall encoding
	static PackedByteArray encode_godot_syscall(int p_ecall_number);
	static PackedByteArray encode_linux_syscall(int p_syscall_number);

	// Function prologue: set up stack frame
	static PackedByteArray encode_prologue(int p_stack_size);

	// Function epilogue: restore stack and return
	static PackedByteArray encode_epilogue(int p_stack_size);
};
