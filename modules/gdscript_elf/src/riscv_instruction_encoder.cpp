/**************************************************************************/
/*  riscv_instruction_encoder.cpp                                         */
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

#include "riscv_instruction_encoder.h"

#include "core/error/error_macros.h"

// RISC-V opcodes (from RISC-V specification)
#define OPCODE_LOAD 0x03
#define OPCODE_STORE 0x23
#define OPCODE_OP_IMM 0x13
#define OPCODE_OP 0x33
#define OPCODE_BRANCH 0x63
#define OPCODE_JAL 0x6F
#define OPCODE_JALR 0x67
#define OPCODE_LUI 0x37
#define OPCODE_SYSTEM 0x73

// RISC-V funct3 values
#define FUNCT3_ADD 0x0
#define FUNCT3_SUB 0x0
#define FUNCT3_OR 0x6
#define FUNCT3_AND 0x7
#define FUNCT3_LD 0x3
#define FUNCT3_LW 0x2
#define FUNCT3_SD 0x3
#define FUNCT3_SW 0x2
#define FUNCT3_BEQ 0x0
#define FUNCT3_BNE 0x1
#define FUNCT3_BLT 0x4
#define FUNCT3_BGE 0x5

// RISC-V funct7 values
#define FUNCT7_ADD 0x00
#define FUNCT7_SUB 0x20
#define FUNCT7_MUL 0x01

RISCVInstructionEncoder::RISCVInstructionEncoder() {
	last_instruction.resize(4);
}

RISCVInstructionEncoder::~RISCVInstructionEncoder() {
	last_instruction.clear();
}

void RISCVInstructionEncoder::encode_32bit(uint32_t instruction) {
	// Store as little-endian bytes
	last_instruction.write[0] = (instruction >> 0) & 0xFF;
	last_instruction.write[1] = (instruction >> 8) & 0xFF;
	last_instruction.write[2] = (instruction >> 16) & 0xFF;
	last_instruction.write[3] = (instruction >> 24) & 0xFF;
}

uint32_t RISCVInstructionEncoder::encode_r_type(uint32_t opcode, int rd, int funct3, int rs1, int rs2, int funct7) {
	return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode;
}

uint32_t RISCVInstructionEncoder::encode_i_type(uint32_t opcode, int rd, int funct3, int rs1, int imm12) {
	int32_t imm = sign_extend_12(imm12);
	return ((imm & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode;
}

uint32_t RISCVInstructionEncoder::encode_s_type(uint32_t opcode, int funct3, int rs1, int rs2, int imm12) {
	int32_t imm = sign_extend_12(imm12);
	return ((imm & 0xFE0) << 20) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | ((imm & 0x1F) << 7) | opcode;
}

uint32_t RISCVInstructionEncoder::encode_b_type(uint32_t opcode, int funct3, int rs1, int rs2, int imm13) {
	int32_t imm = sign_extend_13(imm13);
	return (((imm >> 12) & 1) << 31) | (((imm >> 5) & 0x3F) << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (((imm >> 1) & 0xF) << 8) | (((imm >> 11) & 1) << 7) | opcode;
}

uint32_t RISCVInstructionEncoder::encode_u_type(uint32_t opcode, int rd, int imm20) {
	int32_t imm = sign_extend_20(imm20);
	return ((imm & 0xFFFFF) << 12) | (rd << 7) | opcode;
}

uint32_t RISCVInstructionEncoder::encode_j_type(uint32_t opcode, int rd, int imm21) {
	int32_t imm = sign_extend_21(imm21);
	return (((imm >> 20) & 1) << 31) | (((imm >> 1) & 0x3FF) << 21) | (((imm >> 11) & 1) << 20) | (((imm >> 12) & 0xFF) << 12) | (rd << 7) | opcode;
}

int32_t RISCVInstructionEncoder::sign_extend_12(int imm) {
	return (int32_t)((int16_t)(imm << 4) >> 4);
}

int32_t RISCVInstructionEncoder::sign_extend_13(int imm) {
	return (int32_t)((int16_t)(imm << 3) >> 3);
}

int32_t RISCVInstructionEncoder::sign_extend_20(int imm) {
	return (int32_t)((int32_t)(imm << 12) >> 12);
}

int32_t RISCVInstructionEncoder::sign_extend_21(int imm) {
	return (int32_t)((int32_t)(imm << 11) >> 11);
}

// R-type instructions
void RISCVInstructionEncoder::encode_add(int rd, int rs1, int rs2) {
	uint32_t inst = encode_r_type(OPCODE_OP, rd, FUNCT3_ADD, rs1, rs2, FUNCT7_ADD);
	encode_32bit(inst);
}

void RISCVInstructionEncoder::encode_sub(int rd, int rs1, int rs2) {
	uint32_t inst = encode_r_type(OPCODE_OP, rd, FUNCT3_SUB, rs1, rs2, FUNCT7_SUB);
	encode_32bit(inst);
}

void RISCVInstructionEncoder::encode_mul(int rd, int rs1, int rs2) {
	uint32_t inst = encode_r_type(OPCODE_OP, rd, FUNCT3_ADD, rs1, rs2, FUNCT7_MUL);
	encode_32bit(inst);
}

void RISCVInstructionEncoder::encode_div(int rd, int rs1, int rs2) {
	// DIV uses funct7 = 0x01, funct3 = 0x4
	uint32_t inst = encode_r_type(OPCODE_OP, rd, 0x4, rs1, rs2, FUNCT7_MUL);
	encode_32bit(inst);
}

// I-type instructions
void RISCVInstructionEncoder::encode_addi(int rd, int rs1, int imm12) {
	uint32_t inst = encode_i_type(OPCODE_OP_IMM, rd, FUNCT3_ADD, rs1, imm12);
	encode_32bit(inst);
}

void RISCVInstructionEncoder::encode_ld(int rd, int rs1, int imm12) {
	uint32_t inst = encode_i_type(OPCODE_LOAD, rd, FUNCT3_LD, rs1, imm12);
	encode_32bit(inst);
}

void RISCVInstructionEncoder::encode_lw(int rd, int rs1, int imm12) {
	uint32_t inst = encode_i_type(OPCODE_LOAD, rd, FUNCT3_LW, rs1, imm12);
	encode_32bit(inst);
}

void RISCVInstructionEncoder::encode_ori(int rd, int rs1, int imm12) {
	uint32_t inst = encode_i_type(OPCODE_OP_IMM, rd, FUNCT3_OR, rs1, imm12);
	encode_32bit(inst);
}

void RISCVInstructionEncoder::encode_andi(int rd, int rs1, int imm12) {
	uint32_t inst = encode_i_type(OPCODE_OP_IMM, rd, FUNCT3_AND, rs1, imm12);
	encode_32bit(inst);
}

// S-type instructions
void RISCVInstructionEncoder::encode_sd(int rs1, int rs2, int imm12) {
	uint32_t inst = encode_s_type(OPCODE_STORE, FUNCT3_SD, rs1, rs2, imm12);
	encode_32bit(inst);
}

void RISCVInstructionEncoder::encode_sw(int rs1, int rs2, int imm12) {
	uint32_t inst = encode_s_type(OPCODE_STORE, FUNCT3_SW, rs1, rs2, imm12);
	encode_32bit(inst);
}

// B-type instructions
void RISCVInstructionEncoder::encode_beq(int rs1, int rs2, int imm13) {
	uint32_t inst = encode_b_type(OPCODE_BRANCH, FUNCT3_BEQ, rs1, rs2, imm13);
	encode_32bit(inst);
}

void RISCVInstructionEncoder::encode_bne(int rs1, int rs2, int imm13) {
	uint32_t inst = encode_b_type(OPCODE_BRANCH, FUNCT3_BNE, rs1, rs2, imm13);
	encode_32bit(inst);
}

void RISCVInstructionEncoder::encode_blt(int rs1, int rs2, int imm13) {
	uint32_t inst = encode_b_type(OPCODE_BRANCH, FUNCT3_BLT, rs1, rs2, imm13);
	encode_32bit(inst);
}

void RISCVInstructionEncoder::encode_bge(int rs1, int rs2, int imm13) {
	uint32_t inst = encode_b_type(OPCODE_BRANCH, FUNCT3_BGE, rs1, rs2, imm13);
	encode_32bit(inst);
}

// U-type instructions
void RISCVInstructionEncoder::encode_lui(int rd, int imm20) {
	uint32_t inst = encode_u_type(OPCODE_LUI, rd, imm20);
	encode_32bit(inst);
}

// J-type instructions
void RISCVInstructionEncoder::encode_jal(int rd, int imm21) {
	uint32_t inst = encode_j_type(OPCODE_JAL, rd, imm21);
	encode_32bit(inst);
}

void RISCVInstructionEncoder::encode_jalr(int rd, int rs1, int imm12) {
	uint32_t inst = encode_i_type(OPCODE_JALR, rd, 0x0, rs1, imm12);
	encode_32bit(inst);
}

// Special instructions
void RISCVInstructionEncoder::encode_ecall() {
	// ECALL: imm12=0, rs1=0, funct3=0, rd=0, opcode=SYSTEM
	uint32_t inst = encode_i_type(OPCODE_SYSTEM, 0, 0, 0, 0);
	encode_32bit(inst);
}

void RISCVInstructionEncoder::encode_ret() {
	// ret = jalr x0, x1, 0
	encode_jalr(0, 1, 0);
}

void RISCVInstructionEncoder::encode_nop() {
	// nop = addi x0, x0, 0
	encode_addi(0, 0, 0);
}
