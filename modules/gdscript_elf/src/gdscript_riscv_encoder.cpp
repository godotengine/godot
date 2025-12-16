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

#include "modules/gdscript/gdscript_function.h"

PackedByteArray GDScriptRISCVEncoder::encode_function(GDScriptFunction *p_function) {
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
		PackedByteArray opcode_instructions = encode_opcode(code_ptr[ip], code_ptr, ip, code_size);

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

PackedByteArray GDScriptRISCVEncoder::encode_opcode(int p_opcode, const int *p_code_ptr, int &p_ip, int p_code_size) {
	PackedByteArray result;

	switch (p_opcode) {
		case GDScriptFunction::OPCODE_RETURN: {
			// Simple return - will be handled by epilogue
			// Just advance IP
			p_ip += 2; // OPCODE + return value count
			break;
		}
		case GDScriptFunction::OPCODE_JUMP: {
			// Unconditional jump - encode as RISC-V jal
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
		default: {
			// Fallback: encode syscall to VM
			result = encode_vm_call(p_opcode, p_ip);
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

PackedByteArray GDScriptRISCVEncoder::encode_vm_call(int p_opcode, int p_ip) {
	// Encode syscall to VM fallback function
	// For RISC-V: use ecall instruction or function call
	// Simplified: encode ecall with opcode in a0 register
	PackedByteArray result;
	result.resize(4);

	// li a0, opcode  (lui + addi for large values, or just addi for small)
	// For now: placeholder - encode ecall
	uint32_t ecall = 0x00000073; // ecall instruction
	*reinterpret_cast<uint32_t *>(result.ptrw()) = ecall;

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
