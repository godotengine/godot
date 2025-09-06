#pragma once
#include "rv32i_instr.hpp"

namespace riscv
{
	union rv32c_instruction
	{
		// register format
		struct {
			uint16_t opcode : 2;
			uint16_t rs2    : 5;
			uint16_t rd     : 5;
			uint16_t funct4 : 4;
		} CR;
		// immediate format
		struct {
			uint16_t opcode : 2;
			uint16_t imm1   : 5;
			uint16_t rd     : 5;
			uint16_t imm2   : 1;
			uint16_t funct3 : 3;

			bool sign() const noexcept {
				return imm2;
			}
			int32_t signed_imm() const noexcept {
				const uint32_t ext = 0xFFFFFFE0;
				return imm1 | (sign() ? ext : 0);
			}
			int32_t upper_imm() const noexcept {
				const uint32_t ext = 0xFFFE0000;
				return (imm1 << 12) | (sign() ? ext : 0);
			}
			uint32_t shift_imm() const noexcept {
				return imm1;
			}
			uint32_t shift64_imm() const noexcept {
				return imm1 | (imm2 << 5);
			}
		} CI;
		struct {
			uint16_t opcode : 2;
			uint16_t imm67  : 2;
			uint16_t imm234 : 3;
			uint16_t rd     : 5;
			uint16_t imm5   : 1;
			uint16_t funct3 : 3;

			uint32_t offset() const noexcept {
				uint32_t val = imm234 | (imm5 << 3) | (imm67 << 4);
				return (val << 2); // scaled by 4
			}
		} CI2;
		struct {
			uint16_t opcode : 2;
			uint16_t imm678 : 3;
			uint16_t imm34  : 2;
			uint16_t rd     : 5;
			uint16_t imm5   : 1;
			uint16_t funct3 : 3;

			uint32_t offset() const noexcept {
				uint32_t val = imm34 | (imm5 << 2) | (imm678 << 3);
				return (val << 3); // scaled by 8
			}
		} CIFLD;
		struct {
			uint16_t opcode : 2;
			uint16_t imm5   : 1;
			uint16_t imm78  : 2;
			uint16_t imm6   : 1;
			uint16_t imm4   : 1;
			uint16_t rd     : 5;
			uint16_t imm9   : 1;
			uint16_t funct3 : 3;

			bool sign() const noexcept {
				return imm9;
			}
			int32_t signed_imm() const noexcept {
				const uint32_t ext = 0xFFFFFE00;
				int32_t val = imm4 | (imm5 << 1) | (imm6 << 2) | (imm78 << 3);
				return (val << 4) | (sign() ? ext : 0); // scaled by 16
			}
		} CI16;
		// stack-relative store
		struct {
			uint16_t opcode : 2;
			uint16_t rs2    : 5;
			uint16_t imm67  : 2;
			uint16_t imm25  : 4;
			uint16_t funct3 : 3;

			int32_t offset(int factor) const noexcept {
				int32_t val = imm25 | (imm67 << 4);
				return val * factor;
			}
		} CSS;
		// stack-relative store
		struct {
			uint16_t opcode : 2;
			uint16_t rs2    : 5;
			uint16_t imm678 : 3;
			uint16_t imm345 : 3;
			uint16_t funct3 : 3;

			int32_t offset() const noexcept {
				int32_t val = imm345 | (imm678 << 3);
				return val * 8; // 64-bit instruction
			}
		} CSFSD;
		// wide immediate format
		struct {
			uint16_t opcode : 2;
			uint16_t srd    : 3;
			uint16_t imm3   : 1;
			uint16_t imm2   : 1;
			uint16_t imm6789: 4;
			uint16_t imm45  : 2;
			uint16_t funct3 : 3;

			uint32_t offset() const noexcept {
				uint32_t val = imm2 | (imm3 << 1) | (imm45 << 2) | (imm6789 << 4);
				return (val << 2); // scaled by 4 (ADDI4SPN)
			}
		} CIW;
		// load format
		struct {
			uint16_t opcode : 2;
			uint16_t srd    : 3;
			uint16_t imm6   : 1;
			uint16_t imm2   : 1;
			uint16_t srs1   : 3;
			uint16_t imm345 : 3;
			uint16_t funct3 : 3;

			uint32_t offset() const noexcept {
				const uint32_t val = imm2 | (imm345 << 1) | (imm6 << 4);
				return val << 2;
			}
		} CL;
		// store format
		struct {
			uint16_t opcode : 2;
			uint16_t srs2   : 3;
			uint16_t imm6   : 1;
			uint16_t imm2   : 1;
			uint16_t srs1   : 3;
			uint16_t imm345 : 3;
			uint16_t funct3 : 3;

			uint32_t offset4() const noexcept {
				return (imm2 << 2) | (imm345 << 3) | (imm6 << 6);
			}
		} CS;
		// D store format
		struct {
			uint16_t opcode : 2;
			uint16_t srs2   : 3;
			uint16_t imm67  : 2;
			uint16_t srs1   : 3;
			uint16_t imm345 : 3;
			uint16_t funct3 : 3;

			uint32_t offset8() const noexcept {
				return (imm345 << 3) | (imm67 << 6);
			}
		} CSD;
		// arithmetic format
		struct {
			uint16_t opcode : 2;
			uint16_t srs2   : 3;
			uint16_t funct2 : 2;
			uint16_t srd    : 3;
			uint16_t funct6 : 6;
		} CA;
		struct {
			uint16_t opcode : 2;
			uint16_t imm04  : 5;
			uint16_t srd    : 3;
			uint16_t funct2 : 2;
			uint16_t imm5   : 1;
			uint16_t funct3 : 3;

			uint32_t shift_imm() const noexcept {
				return imm04;
			}
			uint32_t shift64_imm() const noexcept {
				return imm04 | (imm5 << 5);
			}
			int32_t signed_imm() const noexcept {
				const uint32_t ext = 0xFFFFFFE0;
				return imm04 | (imm5 ? ext : 0);
			}
		} CAB;
		// branch format
		struct {
			uint16_t opcode : 2;
			uint16_t off5   : 1;
			uint16_t off12  : 2;
			uint16_t off67  : 2;
			uint16_t srs1   : 3;
			uint16_t off34  : 2;
			uint16_t off8   : 1;
			uint16_t funct3 : 3;

			bool sign() const noexcept {
				return off8;
			}
			int32_t signed_imm() const noexcept {
				int32_t val = (off12 | (off34 << 2) | (off5 << 4) | (off67 << 5));
				const uint32_t ext = 0xFFFFFF00; // 7 immediate bits + 1 sign
				return (val << 1) | (sign() ? ext : 0);
			}
		} CB;
		// jump format
		struct {
			uint16_t opcode : 2;
			//uint16_t jump   : 11;
			uint16_t off5   : 1;
			uint16_t off13  : 3;
			uint16_t off7   : 1;
			uint16_t off6   : 1;
			uint16_t off10  : 1;
			uint16_t off89  : 2;
			uint16_t off4   : 1;
			uint16_t off11  : 1;
			uint16_t funct3 : 3;

			bool sign() const noexcept {
				return off11;
			}
			int32_t signed_imm() const noexcept {
				int32_t val = (off13 | (off4 << 3) | (off5 << 4) | (off6 << 5)
							| (off7 << 6) | (off89 << 7) | (off10 << 9));
				const uint32_t ext = 0xFFFFF800; // 12 immediate bits
				return (val << 1) | (sign() ? ext : 0);
			}
		} CJ;

		uint16_t whole;
		uint8_t  bytes[2];

		rv32c_instruction() : whole(0) {}
		rv32c_instruction(uint16_t another) : whole(another) {}
		rv32c_instruction(rv32i_instruction i) : whole(i.half[0]) {}

		#define RISCV_CI_CODE(x, y) ((x << 13) | (y))
		uint16_t opcode() const noexcept
		{
			return whole & 0b1110000000000011;
		}
		uint16_t funct3() const noexcept {
			return whole >> 13;
		}
	};
	static_assert(sizeof(rv32c_instruction) == 2, "Instruction is 2 bytes");
}
