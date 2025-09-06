#include "machine.hpp"

#include "decoder_cache.hpp"
#include "rv32i_instr.hpp"
#undef RISCV_EXT_COMPRESSED
#define RISCV_128BIT_ISA_INSTRUCTIONS

#define INSTRUCTION(x, ...) \
	static const CPU<16>::instruction_t instr128i_##x { __VA_ARGS__ }
#define DECODED_INSTR(x) instr128i_##x
#include "rvi_instr.cpp"
#include "rv128i_instr.cpp"
#include "rvf_instr.cpp"
#ifdef RISCV_EXT_ATOMICS
#include "rva_instr.cpp"
#include "rva128_instr.cpp"
#endif
#ifdef RISCV_EXT_VECTOR
#include "rvv_instr.cpp"
#endif
#include "instruction_list.hpp"

namespace riscv
{
	template <> RISCV_INTERNAL
	const CPU<16>::instruction_t &CPU<16>::decode(const format_t instruction)
	{
#define DECODER(x) return(x)
#include "instr_decoding.inc"
#undef DECODER
	}

	template <> RISCV_INTERNAL
	void CPU<16>::execute(const format_t instruction)
	{
#define DECODER(x) { x.handler(*this, instruction); return; }
#include "instr_decoding.inc"
#undef DECODER
	}

	template <> RISCV_INTERNAL
	void CPU<16>::execute(uint8_t& handler_idx, uint32_t instr)
	{
		if (handler_idx == 0 && instr != 0) {
			[[unlikely]];
			handler_idx = DecoderData<16>::handler_index_for(decode(instr).handler);
		}
		DecoderData<16>::get_handlers()[handler_idx](*this, instr);
	}

	template <>
	const Instruction<16>& CPU<16>::get_unimplemented_instruction() noexcept
	{
		return DECODED_INSTR(UNIMPLEMENTED);
	}

	static size_t to_hex(char *buffer, size_t len, __uint128_t value)
	{
		if (len < 32)
			return 0;
		len = 8; /* At least print 8 hex digits */
		static constexpr char lut[] = "0123456789ABCDEF";
		for (unsigned i = 0; i < 16 - len / 2; i++)
		{
			if ((value >> ((15 - i) * 8)) & 0xFF)
			{
				len = 32 - i * 2;
				break;
			}
		}
		const size_t max = len / 2;
		for (unsigned i = 0; i < max; i++)
		{
			buffer[i * 2 + 0] = lut[(value >> ((max - 1 - i) * 8 + 4)) & 0xF];
			buffer[i * 2 + 1] = lut[(value >> ((max - 1 - i) * 8 + 0)) & 0xF];
		}
		return len;
	}

	template <> RISCV_COLD_PATH()
	std::string Registers<16>::to_string() const
	{
		char buffer[1800];
		int  len = 0;
		char regbuffer[32];

		for (int i = 1; i < 32; i++) {
			const int reglen =
				to_hex(regbuffer, sizeof(regbuffer), this->get(i));
			len += snprintf(buffer+len, sizeof(buffer) - len,
					"[%s\t%.*s] ", RISCV::regname(i), reglen, regbuffer);
			if (i % 5 == 4) {
				len += snprintf(buffer+len, sizeof(buffer)-len, "\n");
			}
		}
		return std::string(buffer, len);
	}

	template <> RISCV_COLD_PATH()
	std::string CPU<16>::to_string(instruction_format format, const Instruction<16>& instr) const
	{
		char buffer[512];
		char ibuffer[256];
		int  ibuflen = instr.printer(ibuffer, sizeof(ibuffer), *this, format);
		int  len = 0;
		char pcbuffer[32];
		int pclen = to_hex(pcbuffer, sizeof(pcbuffer), this->pc());
		if (format.length() == 4) {
			len = snprintf(buffer, sizeof(buffer),
					"[0x%.*s] %08X %.*s",
					pclen, pcbuffer, format.whole, ibuflen, ibuffer);
		}
		else if (format.length() == 2) {
			len = snprintf(buffer, sizeof(buffer),
					"[0x%.*s]     %04hX %.*s",
					pclen, pcbuffer, (uint16_t) format.whole, ibuflen, ibuffer);
		}
		else {
			throw MachineException(UNIMPLEMENTED_INSTRUCTION_LENGTH,
				"Unimplemented instruction format length", format.length());
		}
		return std::string(buffer, len);
	}
}
