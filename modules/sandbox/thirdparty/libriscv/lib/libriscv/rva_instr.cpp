#include "cpu.hpp"

#include "instr_helpers.hpp"
#if __has_include(<atomic>)
#define USE_ATOMIC_OPS __cpp_lib_atomic_ref
#include <atomic>
#else
#define USE_ATOMIC_OPS 0
#endif
#include <cstdint>
#include <inttypes.h>
static const char atomic_type[] { '?', '?', 'W', 'D', 'Q', '?', '?', '?' };
static const char* atomic_name2[] {
	"AMOADD", "AMOXOR", "AMOOR", "AMOAND", "AMOMIN", "AMOMAX", "AMOMINU", "AMOMAXU"
};
#define AMOSIZE_W   0x2
#define AMOSIZE_D   0x3
#define AMOSIZE_Q   0x4

namespace riscv
{
	template <int W>
	template <typename Type>
	inline void CPU<W>::amo(format_t instr,
		Type(*op)(CPU&, Type&, uint32_t))
	{
		// 1. load address from rs1
		const auto addr = this->reg(instr.Atype.rs1);
		// 2. verify address alignment vs Type
		if (UNLIKELY(addr % sizeof(Type) != 0)) {
			trigger_exception(INVALID_ALIGNMENT, addr);
		}
		// 3. read value from writable memory location
		// TODO: Make Type unsigned to match other templates, avoiding spam
		Type& mem = machine().memory.template writable_read<Type> (addr);
		// 4. apply <op>, writing the value to mem and returning old value
		const Type old_value = op(*this, mem, instr.Atype.rs2);
		// 5. place value into rd
		// NOTE: we have to do it in this order, because we can
		// clobber rs2 when writing to rd, if they are the same!
		if (instr.Atype.rd != 0) {
			// For RV64, 32-bit AMOs always sign-extend the value
			// placed in rd, and ignore the upper 32 bits of the original
			// value of rs2.
			using signed_t = std::make_signed_t<Type>;
			this->reg(instr.Atype.rd) = (RVSIGNTYPE(*this))signed_t(old_value);
		}
	}

	ATOMIC_INSTR(AMOADD_W,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_COLDATTR
	{
		cpu.template amo<int32_t>(instr,
		[] (auto& proc, auto& value, auto rs2) {
#if USE_ATOMIC_OPS
			return std::atomic_ref(value).fetch_add(proc.reg(rs2));
#else
			auto old_value = value;
			value += proc.reg(rs2);
			return old_value;
#endif
		});
	},
	[] (char* buffer, size_t len, auto&, rv32i_instruction instr) RVPRINTR_ATTR {
		return snprintf(buffer, len, "%s.%c [%s] %s, %s",
						atomic_name2[instr.Atype.funct5 >> 2],
						atomic_type[instr.Atype.funct3 & 7],
                        RISCV::regname(instr.Atype.rs1),
                        RISCV::regname(instr.Atype.rs2),
                        RISCV::regname(instr.Atype.rd));
	});

	ATOMIC_INSTR(AMOXOR_W,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_COLDATTR
	{
		cpu.template amo<int32_t>(instr,
		[] (auto& proc, auto& value, auto rs2) {
#if USE_ATOMIC_OPS
			return std::atomic_ref(value).fetch_xor(proc.reg(rs2));
#else
			auto old_value = value;
			value ^= proc.reg(rs2);
			return old_value;
#endif
		});
	}, DECODED_ATOMIC(AMOADD_W).printer);

	ATOMIC_INSTR(AMOOR_W,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_COLDATTR
	{
		cpu.template amo<int32_t>(instr,
		[] (auto& proc, auto& value, auto rs2) {
#if USE_ATOMIC_OPS
			return std::atomic_ref(value).fetch_or(proc.reg(rs2));
#else
			auto old_value = value;
			value |= proc.reg(rs2);
			return old_value;
#endif
		});
	}, DECODED_ATOMIC(AMOADD_W).printer);

	ATOMIC_INSTR(AMOAND_W,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_COLDATTR
	{
		cpu.template amo<int32_t>(instr,
		[] (auto& proc, auto& value, auto rs2) {
#if USE_ATOMIC_OPS
			return std::atomic_ref(value).fetch_and(proc.reg(rs2));
#else
			auto old_value = value;
			value &= proc.reg(rs2);
			return old_value;
#endif
		});
	}, DECODED_ATOMIC(AMOADD_W).printer);

	ATOMIC_INSTR(AMOMAX_W,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_COLDATTR
	{
		cpu.template amo<int32_t>(instr,
		[] (auto& proc, auto& value, auto rs2) {
			auto old_val = value;
			value = std::max(value, (int32_t)proc.reg(rs2));
			return old_val;
		});
	}, DECODED_ATOMIC(AMOADD_W).printer);

	ATOMIC_INSTR(AMOMIN_W,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_COLDATTR
	{
		cpu.template amo<int32_t>(instr,
		[] (auto& proc, auto& value, auto rs2) {
			auto old_val = value;
			value = std::min(value, (int32_t)proc.reg(rs2));
			return old_val;
		});
	}, DECODED_ATOMIC(AMOADD_W).printer);

	ATOMIC_INSTR(AMOMAXU_W,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_COLDATTR
	{
		cpu.template amo<uint32_t>(instr,
		[] (auto& proc, auto& value, auto rs2) {
			auto old_val = value;
			value = std::max(value, (uint32_t)proc.reg(rs2));
			return old_val;
		});
	}, DECODED_ATOMIC(AMOADD_W).printer);

	ATOMIC_INSTR(AMOMINU_W,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_COLDATTR
	{
		cpu.template amo<uint32_t>(instr,
		[] (auto& proc, auto& value, auto rs2) {
			auto old_val = value;
			value = std::min(value, (uint32_t)proc.reg(rs2));
			return old_val;
		});
	}, DECODED_ATOMIC(AMOADD_W).printer);

	ATOMIC_INSTR(AMOADD_D,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_COLDATTR
	{
		cpu.template amo<int64_t>(instr,
		[] (auto& proc, auto& value, auto rs2) {
#if USE_ATOMIC_OPS
			return std::atomic_ref(value).fetch_add(proc.reg(rs2));
#else
			auto old_value = value;
			value += proc.reg(rs2);
			return old_value;
#endif
		});
	}, DECODED_ATOMIC(AMOADD_W).printer);

	ATOMIC_INSTR(AMOXOR_D,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_COLDATTR
	{
		cpu.template amo<int64_t>(instr,
		[] (auto& proc, auto& value, auto rs2) {
#if USE_ATOMIC_OPS
			return std::atomic_ref(value).fetch_xor(proc.reg(rs2));
#else
			auto old_value = value;
			value ^= proc.reg(rs2);
			return old_value;
#endif
		});
	}, DECODED_ATOMIC(AMOADD_W).printer);

	ATOMIC_INSTR(AMOOR_D,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_COLDATTR
	{
		cpu.template amo<int64_t>(instr,
		[] (auto& proc, auto& value, auto rs2) {
#if USE_ATOMIC_OPS
			return std::atomic_ref(value).fetch_or(proc.reg(rs2));
#else
			auto old_value = value;
			value |= proc.reg(rs2);
			return old_value;
#endif
		});
	}, DECODED_ATOMIC(AMOADD_W).printer);

	ATOMIC_INSTR(AMOAND_D,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_COLDATTR
	{
		cpu.template amo<int64_t>(instr,
		[] (auto& proc, auto& value, auto rs2) {
#if USE_ATOMIC_OPS
			return std::atomic_ref(value).fetch_and(proc.reg(rs2));
#else
			auto old_value = value;
			value &= proc.reg(rs2);
			return old_value;
#endif
		});
	}, DECODED_ATOMIC(AMOADD_W).printer);

	ATOMIC_INSTR(AMOMAX_D,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_COLDATTR
	{
		cpu.template amo<int64_t>(instr,
		[] (auto& proc, auto& value, auto rs2) {
			auto old_val = value;
			value = std::max(value, int64_t(proc.reg(rs2)));
			return old_val;
		});
	}, DECODED_ATOMIC(AMOADD_W).printer);

	ATOMIC_INSTR(AMOMIN_D,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_COLDATTR
	{
		cpu.template amo<int64_t>(instr,
		[] (auto& proc, auto& value, auto rs2) {
			auto old_val = value;
			value = std::min(value, int64_t(proc.reg(rs2)));
			return old_val;
		});
	}, DECODED_ATOMIC(AMOADD_W).printer);

	ATOMIC_INSTR(AMOMAXU_D,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_COLDATTR
	{
		cpu.template amo<uint64_t>(instr,
		[] (auto& proc, auto& value, auto rs2) {
			auto old_val = value;
			value = std::max(value, (uint64_t)proc.reg(rs2));
			return old_val;
		});
	}, DECODED_ATOMIC(AMOADD_W).printer);

	ATOMIC_INSTR(AMOMINU_D,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_COLDATTR
	{
		cpu.template amo<uint64_t>(instr,
		[] (auto& proc, auto& value, auto rs2) {
			auto old_val = value;
			value = std::min(value, (uint64_t)proc.reg(rs2));
			return old_val;
		});
	}, DECODED_ATOMIC(AMOADD_W).printer);

	ATOMIC_INSTR(AMOSWAP_W,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_COLDATTR
	{
		cpu.template amo<int32_t>(instr,
		[] (auto& proc, auto& value, auto rs2) {
#if USE_ATOMIC_OPS
			return std::atomic_ref(value).exchange(proc.reg(rs2));
#else
			auto old_value = value;
			value = proc.reg(rs2);
			return old_value;
#endif
		});
	},
	[] (char* buffer, size_t len, auto&, rv32i_instruction instr) RVPRINTR_ATTR {
		return snprintf(buffer, len, "AMOSWAP.%c [%s] %s, %s",
						atomic_type[instr.Atype.funct3 & 7],
                        RISCV::regname(instr.Atype.rs1),
                        RISCV::regname(instr.Atype.rs2),
                        RISCV::regname(instr.Atype.rd));
	});

	ATOMIC_INSTR(AMOSWAP_D,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_COLDATTR
	{
		cpu.template amo<int64_t>(instr,
		[] (auto& proc, auto& value, auto rs2) {
#if USE_ATOMIC_OPS
			return std::atomic_ref(value).exchange(proc.reg(rs2));
#else
			auto old_value = value;
			value = proc.reg(rs2);
			return old_value;
#endif
		});
	}, DECODED_ATOMIC(AMOSWAP_W).printer);

    ATOMIC_INSTR(LOAD_RESV,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_COLDATTR
	{
		const auto addr = cpu.reg(instr.Atype.rs1);
		RVSIGNTYPE(cpu) value;
		// switch on atomic type
		if (instr.Atype.funct3 == AMOSIZE_W)
		{
			if (!cpu.atomics().load_reserve(4, addr))
				cpu.trigger_exception(DEADLOCK_REACHED);
			value = (int32_t)cpu.machine().memory.template read<uint32_t> (addr);
		}
		else if (instr.Atype.funct3 == AMOSIZE_D)
		{
			if constexpr (RVISGE64BIT(cpu)) {
				if (!cpu.atomics().load_reserve(8, addr))
					cpu.trigger_exception(DEADLOCK_REACHED);
				value = (int64_t)cpu.machine().memory.template read<uint64_t> (addr);
			} else
				cpu.trigger_exception(ILLEGAL_OPCODE);
		}
		else if (instr.Atype.funct3 == AMOSIZE_Q)
		{
			if constexpr (RVIS128BIT(cpu)) {
				if (!cpu.atomics().load_reserve(16, addr))
					cpu.trigger_exception(DEADLOCK_REACHED);
				value = cpu.machine().memory.template read<RVREGTYPE(cpu)> (addr);
			} else
				cpu.trigger_exception(ILLEGAL_OPCODE);
		}
		else {
			cpu.trigger_exception(ILLEGAL_OPCODE);
		}
		if (instr.Atype.rd != 0)
			cpu.reg(instr.Atype.rd) = value;
	},
	[] (char* buffer, size_t len, auto& cpu, rv32i_instruction instr) RVPRINTR_ATTR {
		const uint64_t addr = cpu.reg(instr.Atype.rs1);
		return snprintf(buffer, len, "LR.%c [%s = 0x%" PRIX64 "], %s",
				atomic_type[instr.Atype.funct3 & 7],
				RISCV::regname(instr.Atype.rs1), addr,
				RISCV::regname(instr.Atype.rd));
	});

	ATOMIC_INSTR(STORE_COND,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_COLDATTR
	{
		const auto addr = cpu.reg(instr.Atype.rs1);
		bool resv = false;
		if (instr.Atype.funct3 == AMOSIZE_W)
		{
			resv = cpu.atomics().store_conditional(4, addr);
			if (resv) {
				cpu.machine().memory.template write<uint32_t> (addr, cpu.reg(instr.Atype.rs2));
			}
		}
		else if (instr.Atype.funct3 == AMOSIZE_D)
		{
			if constexpr (RVISGE64BIT(cpu)) {
				resv = cpu.atomics().store_conditional(8, addr);
				if (resv) {
					cpu.machine().memory.template write<uint64_t> (addr, cpu.reg(instr.Atype.rs2));
				}
			} else
				cpu.trigger_exception(ILLEGAL_OPCODE);
		}
		else if (instr.Atype.funct3 == AMOSIZE_Q)
		{
			if constexpr (RVIS128BIT(cpu)) {
				resv = cpu.atomics().store_conditional(16, addr);
				if (resv) {
					cpu.machine().memory.template write<RVREGTYPE(cpu)> (addr, cpu.reg(instr.Atype.rs2));
				}
			} else
				cpu.trigger_exception(ILLEGAL_OPCODE);
		}
		else {
			cpu.trigger_exception(ILLEGAL_OPCODE);
		}
		// Write non-zero value to RD on failure
		if (instr.Atype.rd != 0)
			cpu.reg(instr.Atype.rd) = !resv;
	},
	[] (char* buffer, size_t len, auto&, rv32i_instruction instr) RVPRINTR_ATTR {
		return snprintf(buffer, len, "SC.%c [%s], %s res=%s",
				atomic_type[instr.Atype.funct3 & 7],
				RISCV::regname(instr.Atype.rs1),
				RISCV::regname(instr.Atype.rs2),
				RISCV::regname(instr.Atype.rd));
	});
}
