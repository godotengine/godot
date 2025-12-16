/**************************************************************************/
/*  gdscript_riscv_encoder.cpp                                           */
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

#include "gdscript_riscv_encoder.h"
#include "gdscript_elf64_mode.h"

#include "modules/gdscript/gdscript_function.h"

// Include sandbox syscalls for ECALL numbers (Mode 1)
#ifdef MODULE_SANDBOX_ENABLED
#include "modules/sandbox/src/syscalls.h"
#endif

PackedByteArray GDScriptRISCVEncoder::encode_function(GDScriptFunction *p_function, ELF64CompilationMode p_mode) {
	PackedByteArray instructions;

	if (!p_function || p_function->_code_ptr == nullptr || p_function->_code_size == 0) {
		return instructions;
	}

	const int *code_ptr = p_function->_code_ptr;
	int code_size = p_function->_code_size;
	int ip = 0;

	// Function prologue: set up stack frame
	int stack_size = p_function->get_max_stack_size() * sizeof(Variant);
	PackedByteArray prologue = encode_prologue(stack_size);
	int old_size = instructions.size();
	instructions.resize(old_size + prologue.size());
	memcpy(instructions.ptrw() + old_size, prologue.ptr(), prologue.size());

	// Encode each bytecode opcode
	while (ip < code_size) {
		PackedByteArray opcode_instructions = encode_opcode(code_ptr[ip], code_ptr, ip, code_size, p_mode);

		// Check if opcode is unsupported in current mode
		if (opcode_instructions.is_empty() && p_mode == ELF64CompilationMode::LINUX_SYSCALL) {
			// Unsupported opcode in pure Linux mode - return empty to indicate failure
			// Hybrid mode should always have a fallback (Godot syscall)
			return PackedByteArray();
		}

		old_size = instructions.size();
		instructions.resize(old_size + opcode_instructions.size());
		memcpy(instructions.ptrw() + old_size, opcode_instructions.ptr(), opcode_instructions.size());
	}

	// Function epilogue: restore stack and return
	PackedByteArray epilogue = encode_epilogue(stack_size);
	old_size = instructions.size();
	instructions.resize(old_size + epilogue.size());
	memcpy(instructions.ptrw() + old_size, epilogue.ptr(), epilogue.size());

	return instructions;
}

PackedByteArray GDScriptRISCVEncoder::encode_opcode(int p_opcode, const int *p_code_ptr, int &p_ip, int p_code_size, ELF64CompilationMode p_mode) {
	PackedByteArray result;

	switch (p_opcode) {
		case GDScriptFunction::OPCODE_RETURN: {
			// Simple return - will be handled by epilogue
			// Just advance IP
			p_ip += 2; // OPCODE + return value count
			break;
		}
		case GDScriptFunction::OPCODE_JUMP: {
			// Unconditional jump - encode as RISC-V jal (both modes)
			if (p_ip + 1 >= p_code_size) {
				break;
			}
			int target_ip = p_code_ptr[p_ip + 1];
			int offset = (target_ip - p_ip) * 4; // Assume 4 bytes per instruction
			uint32_t jal = encode_j_type(0x6F, 0, offset); // jal x0, offset
			result.resize(4);
			*reinterpret_cast<uint32_t *>(result.ptrw()) = jal;
			p_ip += 2;
			break;
		}
		case GDScriptFunction::OPCODE_JUMP_IF:
		case GDScriptFunction::OPCODE_JUMP_IF_NOT: {
			// Conditional jumps - encode as RISC-V branch instructions
			// For now, fall through to VM call - proper implementation would need
			// to handle stack values and condition evaluation
			result = encode_vm_call(p_opcode, p_ip, p_mode);
			// Hybrid mode always has a fallback (Godot syscall), so result should never be empty
			if (result.is_empty() && p_mode == ELF64CompilationMode::LINUX_SYSCALL) {
				// Unsupported in pure Linux mode
				p_ip += 1;
				return result;
			}
			p_ip += 1;
			break;
		}
		case GDScriptFunction::OPCODE_ASSIGN:
		case GDScriptFunction::OPCODE_ASSIGN_NULL:
		case GDScriptFunction::OPCODE_ASSIGN_TRUE:
		case GDScriptFunction::OPCODE_ASSIGN_FALSE: {
			// Simple assignments - could be optimized to direct RISC-V
			// For now, use VM call (Godot syscall in hybrid/default mode)
			result = encode_vm_call(p_opcode, p_ip, p_mode);
			if (result.is_empty() && p_mode == ELF64CompilationMode::LINUX_SYSCALL) {
				// Unsupported in pure Linux mode
				p_ip += 1;
				return result;
			}
			p_ip += 1;
			break;
		}
		case GDScriptFunction::OPCODE_OPERATOR:
		case GDScriptFunction::OPCODE_OPERATOR_VALIDATED: {
			// Arithmetic operations - could be optimized to direct RISC-V
			// For now, use VM call (Godot syscall in hybrid/default mode)
			result = encode_vm_call(p_opcode, p_ip, p_mode);
			if (result.is_empty() && p_mode == ELF64CompilationMode::LINUX_SYSCALL) {
				// Unsupported in pure Linux mode
				p_ip += 1;
				return result;
			}
			p_ip += 1;
			break;
		}
		default: {
			// Fallback: encode syscall to VM (mode-specific)
			// Hybrid mode defaults to Godot syscalls, can use Linux syscalls when appropriate
			result = encode_vm_call(p_opcode, p_ip, p_mode);
			if (result.is_empty() && p_mode == ELF64CompilationMode::LINUX_SYSCALL) {
				// Unsupported opcode in pure Linux mode
				p_ip += 1;
				return result;
			}
			p_ip += 1; // Advance by 1 for unknown opcodes
			break;
		}
	}

	return result;
}

uint32_t GDScriptRISCVEncoder::encode_r_type(uint8_t opcode, uint8_t rd, uint8_t funct3, uint8_t rs1, uint8_t rs2, uint8_t funct7) {
	uint32_t instruction = 0;
	instruction |= (opcode & 0x7F);
	instruction |= ((rd & 0x1F) << 7);
	instruction |= ((funct3 & 0x7) << 12);
	instruction |= ((rs1 & 0x1F) << 15);
	instruction |= ((rs2 & 0x1F) << 20);
	instruction |= ((funct7 & 0x7F) << 25);
	return instruction;
}

uint32_t GDScriptRISCVEncoder::encode_i_type(uint8_t opcode, uint8_t rd, uint8_t funct3, uint8_t rs1, int16_t imm) {
	uint32_t instruction = 0;
	instruction |= (opcode & 0x7F);
	instruction |= ((rd & 0x1F) << 7);
	instruction |= ((funct3 & 0x7) << 12);
	instruction |= ((rs1 & 0x1F) << 15);
	// Sign-extend immediate to 12 bits
	uint32_t imm_unsigned = static_cast<uint32_t>(imm) & 0xFFF;
	instruction |= (imm_unsigned << 20);
	return instruction;
}

uint32_t GDScriptRISCVEncoder::encode_s_type(uint8_t opcode, uint8_t funct3, uint8_t rs1, uint8_t rs2, int16_t imm) {
	uint32_t instruction = 0;
	instruction |= (opcode & 0x7F);
	instruction |= ((imm & 0x1F) << 7); // imm[4:0]
	instruction |= ((funct3 & 0x7) << 12);
	instruction |= ((rs1 & 0x1F) << 15);
	instruction |= ((rs2 & 0x1F) << 20);
	instruction |= (((imm >> 5) & 0x7F) << 25); // imm[11:5]
	return instruction;
}

uint32_t GDScriptRISCVEncoder::encode_u_type(uint8_t opcode, uint8_t rd, int32_t imm) {
	uint32_t instruction = 0;
	instruction |= (opcode & 0x7F);
	instruction |= ((rd & 0x1F) << 7);
	// imm[31:12] goes to bits [31:12]
	instruction |= (static_cast<uint32_t>(imm) & 0xFFFFF000);
	return instruction;
}

uint32_t GDScriptRISCVEncoder::encode_j_type(uint8_t opcode, uint8_t rd, int32_t imm) {
	uint32_t instruction = 0;
	instruction |= (opcode & 0x7F);
	instruction |= ((rd & 0x1F) << 7);
	// J-type immediate encoding: [20|10:1|11|19:12]
	uint32_t imm_unsigned = static_cast<uint32_t>(imm);
	instruction |= ((imm_unsigned & 0xFF000) << 12); // imm[19:12]
	instruction |= ((imm_unsigned & 0x800) << 20); // imm[11]
	instruction |= ((imm_unsigned & 0x7FE) << 20); // imm[10:1]
	instruction |= ((imm_unsigned & 0x100000) << 11); // imm[20]
	return instruction;
}

PackedByteArray GDScriptRISCVEncoder::encode_godot_syscall(int p_ecall_number) {
	// Encode Godot ECALL: li a7, <ecall_number>; ecall
	PackedByteArray result;

	// li a7, <ecall_number>
	// For values > 2047 or < -2048, use lui + addi
	if (p_ecall_number > 2047 || p_ecall_number < -2048) {
		result.resize(12); // 3 instructions: lui + addi + ecall
		
		// lui a7, upper 20 bits (sign-extended)
		int32_t value = static_cast<int32_t>(p_ecall_number);
		// lui loads imm[31:12] into rd, sign-extends to 32 bits
		uint32_t upper = (static_cast<uint32_t>(value) >> 12) & 0xFFFFF;
		// Adjust for sign extension: if bit 11 of lower is set, increment upper
		if ((value & 0x800) != 0) {
			upper = (upper + 1) & 0xFFFFF;
		}
		uint32_t lui = encode_u_type(0x37, 17, static_cast<int32_t>(upper) << 12); // lui a7, upper
		*reinterpret_cast<uint32_t *>(result.ptrw()) = lui;

		// addi a7, a7, lower 12 bits
		int16_t lower = static_cast<int16_t>(value & 0xFFF);
		uint32_t addi = encode_i_type(0x13, 17, 0, 17, lower); // addi a7, a7, lower
		*reinterpret_cast<uint32_t *>(result.ptrw() + 4) = addi;
		
		// ecall
		uint32_t ecall = 0x00000073; // ecall
		*reinterpret_cast<uint32_t *>(result.ptrw() + 8) = ecall;
	} else {
		result.resize(8); // 2 instructions: addi + ecall
		// addi a7, x0, <ecall_number>
		uint32_t addi = encode_i_type(0x13, 17, 0, 0, static_cast<int16_t>(p_ecall_number)); // addi a7, x0, imm
		*reinterpret_cast<uint32_t *>(result.ptrw()) = addi;
		// ecall
		uint32_t ecall = 0x00000073; // ecall
		*reinterpret_cast<uint32_t *>(result.ptrw() + 4) = ecall;
	}

	return result;
}

PackedByteArray GDScriptRISCVEncoder::encode_linux_syscall(int p_syscall_number) {
	// Encode Linux syscall: li a7, <syscall_number>; ecall
	// Same encoding as Godot syscall
	return encode_godot_syscall(p_syscall_number);
}

PackedByteArray GDScriptRISCVEncoder::encode_vm_call(int p_opcode, int p_ip, ELF64CompilationMode p_mode) {
	// Map GDScript opcodes to syscalls based on mode
	PackedByteArray result;

	// Linux syscall numbers (RISC-V 64-bit) - used in hybrid and Linux modes
	const int LINUX_WRITE = 64;   // write syscall
	const int LINUX_EXIT = 93;    // exit syscall
	const int LINUX_READ = 63;    // read syscall

	if (p_mode == ELF64CompilationMode::HYBRID) {
		// Hybrid mode: Choose syscall type based on operation
		// Default to Godot syscalls, use Linux syscalls for standard I/O
#ifdef MODULE_SANDBOX_ENABLED
		// Use ECALL numbers from sandbox syscalls.h
		int ecall_num = ECALL_VCALL; // Default to variant call
		
		// Map specific opcodes
		switch (p_opcode) {
			case GDScriptFunction::OPCODE_GET_MEMBER:
				ecall_num = ECALL_OBJ_PROP_GET; // 545 - Godot-specific
				result = encode_godot_syscall(ecall_num);
				break;
			case GDScriptFunction::OPCODE_SET_MEMBER:
				ecall_num = ECALL_OBJ_PROP_SET; // 546 - Godot-specific
				result = encode_godot_syscall(ecall_num);
				break;
			case GDScriptFunction::OPCODE_CALL:
			case GDScriptFunction::OPCODE_CALL_RETURN:
			case GDScriptFunction::OPCODE_CALL_UTILITY:
			case GDScriptFunction::OPCODE_CALL_UTILITY_VALIDATED:
				// Check if this is a print/IO utility call
				// For now, use Godot ECALL_PRINT for print operations
				// In the future, we could detect print() and use Linux write syscall
				ecall_num = ECALL_VCALL; // 501 - Use Godot variant call
				result = encode_godot_syscall(ecall_num);
				break;
			default:
				// Default: Use Godot syscalls
				ecall_num = ECALL_VCALL; // 501
				result = encode_godot_syscall(ecall_num);
				break;
		}
#else
		// Fallback if sandbox module not available
		const int ECALL_VCALL_FALLBACK = 501;
		const int ECALL_OBJ_PROP_GET_FALLBACK = 545;
		const int ECALL_OBJ_PROP_SET_FALLBACK = 546;
		
		int ecall_num = ECALL_VCALL_FALLBACK;
		switch (p_opcode) {
			case GDScriptFunction::OPCODE_GET_MEMBER:
				ecall_num = ECALL_OBJ_PROP_GET_FALLBACK;
				break;
			case GDScriptFunction::OPCODE_SET_MEMBER:
				ecall_num = ECALL_OBJ_PROP_SET_FALLBACK;
				break;
			default:
				ecall_num = ECALL_VCALL_FALLBACK;
				break;
		}
		result = encode_godot_syscall(ecall_num);
#endif
	} else if (p_mode == ELF64CompilationMode::GODOT_SYSCALL) {
		// Mode 1: Pure Godot syscalls
#ifdef MODULE_SANDBOX_ENABLED
		int ecall_num = ECALL_VCALL;
		switch (p_opcode) {
			case GDScriptFunction::OPCODE_GET_MEMBER:
				ecall_num = ECALL_OBJ_PROP_GET;
				break;
			case GDScriptFunction::OPCODE_SET_MEMBER:
				ecall_num = ECALL_OBJ_PROP_SET;
				break;
			default:
				ecall_num = ECALL_VCALL;
				break;
		}
		result = encode_godot_syscall(ecall_num);
#else
		const int ECALL_VCALL_FALLBACK = 501;
		result = encode_godot_syscall(ECALL_VCALL_FALLBACK);
#endif
	} else {
		// Mode 2: Pure Linux syscalls
		// Most GDScript opcodes require Godot runtime
		// Return empty to indicate unsupported
		result = PackedByteArray();
	}

	return result;
}

PackedByteArray GDScriptRISCVEncoder::encode_prologue(int p_stack_size) {
	PackedByteArray result;
	// Function prologue: addi sp, sp, -stack_size
	// Save return address: sd ra, stack_size-8(sp)
	// For now: minimal prologue
	if (p_stack_size > 0) {
		// addi sp, sp, -stack_size
		uint32_t addi = encode_i_type(0x13, 2, 0, 2, -p_stack_size); // addi sp, sp, -stack_size
		result.resize(4);
		*reinterpret_cast<uint32_t *>(result.ptrw()) = addi;
	}
	return result;
}

PackedByteArray GDScriptRISCVEncoder::encode_epilogue(int p_stack_size) {
	PackedByteArray result;
	// Function epilogue: restore stack and return
	// ld ra, stack_size-8(sp)
	// addi sp, sp, stack_size
	// ret (jalr x0, 0(x1))
	if (p_stack_size > 0) {
		// addi sp, sp, stack_size
		uint32_t addi = encode_i_type(0x13, 2, 0, 2, p_stack_size); // addi sp, sp, stack_size
		result.resize(4);
		*reinterpret_cast<uint32_t *>(result.ptrw()) = addi;
	}
	// ret = jalr x0, 0(x1)
	uint32_t ret = encode_i_type(0x67, 0, 0, 1, 0); // jalr x0, 0(x1)
	int old_size = result.size();
	result.resize(old_size + 4);
	*reinterpret_cast<uint32_t *>(result.ptrw() + old_size) = ret;
	return result;
}
