#include "rv32i_instr.hpp"

namespace riscv
{
	struct UnalignedLoad32 {
		uint16_t data[2];
		operator uint32_t() const {
			return data[0] | uint32_t(data[1]) << 16;
		}
	};
	struct AlignedLoad16 {
		uint16_t data;
		operator uint32_t() { return data; }
		unsigned length() const {
			return (data & 3) != 3 ? 2 : 4;
		}
		unsigned opcode() const {
			return data & 0x7F;
		}
		uint16_t half() const {
			return data;
		}
	};
	inline rv32i_instruction read_instruction(
		const uint8_t* exec_segment, uint64_t pc, uint64_t end_pc)
	{
		if (pc + 4 <= end_pc)
			return rv32i_instruction{*(UnalignedLoad32 *)&exec_segment[pc]};
		else if (pc + 2 <= end_pc)
			return rv32i_instruction{*(AlignedLoad16 *)&exec_segment[pc]};
		else
			return rv32i_instruction{0};
	}
}
