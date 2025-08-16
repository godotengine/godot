#include "rvfd.hpp"
#include "instr_helpers.hpp"
#include <cmath>

namespace riscv
{
	// RISC-V Canonical NaNs
	static constexpr uint32_t CANONICAL_NAN_F32 = 0x7fc00000;
	static constexpr uint64_t CANONICAL_NAN_F64 = 0x7ff8000000000000;

	template <typename T>
	static bool is_signaling_nan(T t) {
		if constexpr (sizeof(T) == 4)
			return (*(uint32_t*)&t & 0x7fa00000) == 0x7f800000;
		else
			return (*(uint64_t*)&t & 0x7ffe000000000000) == 0x7ff0000000000000;
	}

#ifdef RISCV_FCSR
	template <int W, typename T>
	static void fsflags(CPU<W>& cpu, long double exact, T& inexact) {
		if constexpr (fcsr_emulation) {
			auto& fcsr = cpu.registers().fcsr();
			fcsr.fflags = 0;
			if (std::isnan(exact) || std::isnan(inexact)) {
				fcsr.fflags |= 16;
				// Canonical NaN
				if constexpr (sizeof(T) == 4)
					*(int32_t *)&inexact = CANONICAL_NAN_F32;
				else
					*(int64_t *)&inexact = CANONICAL_NAN_F64;
			} else {
				if (exact != inexact) fcsr.fflags |= 1;
			}
		}
	}
#else
#define fsflags(c, e, i) /**/
#endif
	template <bool Signaling, int W, typename T, typename R>
	static void feqflags(CPU<W>& cpu, T a, T b, R& dst) {
		if constexpr (fcsr_emulation) {
			auto& fcsr = cpu.registers().fcsr();
			fcsr.fflags = 0;
			if (std::isnan(a) || std::isnan(b)) {
				// All operations return 0 when either operand is NaN
				dst = 0;
			}
			if constexpr (Signaling) {
				if (std::isnan(a) || std::isnan(b))
					fcsr.fflags |= 16;
			} else { // Quiet
				if (is_signaling_nan(a) || is_signaling_nan(b))
					fcsr.fflags |= 16;
			}
		}
	}

	FLOAT_INSTR(FLW,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR
	{
		const rv32f_instruction fi { instr };
		auto addr = cpu.reg(fi.Itype.rs1) + fi.Itype.signed_imm();
		auto& dst = cpu.registers().getfl(fi.Itype.rd);
		dst.load_u32(cpu.machine().memory.template read<uint32_t> (addr));
	},
	[] (char* buffer, size_t len, auto&, rv32i_instruction instr) RVPRINTR_ATTR {
		const rv32f_instruction fi { instr };
		static const std::array<const char*, 8> insn {
			"???", "FLH", "FLW", "FLD", "FLQ", "???", "???", "???"
		};
		return snprintf(buffer, len, "%s %s, [%s%+d]",
						insn[fi.Itype.funct3],
						RISCV::flpname(fi.Itype.rd),
						RISCV::regname(fi.Stype.rs1),
						fi.Itype.signed_imm());
	});
	FLOAT_INSTR(FLD,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR
	{
		const rv32f_instruction fi { instr };
		auto addr = cpu.reg(fi.Itype.rs1) + fi.Itype.signed_imm();
		auto& dst = cpu.registers().getfl(fi.Itype.rd);
		dst.load_u64(cpu.machine().memory.template read<uint64_t> (addr));
	}, DECODED_FLOAT(FLW).printer);

	FLOAT_INSTR(FSW,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR
	{
		const rv32f_instruction fi { instr };
		const auto& src = cpu.registers().getfl(fi.Stype.rs2);
		auto addr = cpu.reg(fi.Stype.rs1) + fi.Stype.signed_imm();
		cpu.machine().memory.template write<uint32_t> (addr, src.i32[0]);
	},
	[] (char* buffer, size_t len, auto&, rv32i_instruction instr) RVPRINTR_ATTR {
		const rv32f_instruction fi { instr };
		static const std::array<const char*, 8> insn {
			"???", "FSH", "FSW", "FSD", "FSQ", "???", "???", "???"
		};
		return snprintf(buffer, len, "%s [%s%+d], %s",
						insn[fi.Stype.funct3],
						RISCV::regname(fi.Stype.rs1),
						fi.Stype.signed_imm(),
						RISCV::flpname(fi.Stype.rs2));
	});
	FLOAT_INSTR(FSD,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR
	{
		const rv32f_instruction fi { instr };
		const auto& src = cpu.registers().getfl(fi.Stype.rs2);
		auto addr = cpu.reg(fi.Stype.rs1) + fi.Stype.signed_imm();
		cpu.machine().memory.template write<uint64_t> (addr, src.i64);
	}, DECODED_FLOAT(FSW).printer);

	FLOAT_INSTR(FMADD,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR
	{
		const rv32f_instruction fi { instr };
		auto& dst = cpu.registers().getfl(fi.R4type.rd);
		auto& rs1 = cpu.registers().getfl(fi.R4type.rs1);
		auto& rs2 = cpu.registers().getfl(fi.R4type.rs2);
		auto& rs3 = cpu.registers().getfl(fi.R4type.rs3);
		if (fi.R4type.funct2 == 0x0) { // float32
			dst.set_float(rs1.f32[0] * rs2.f32[0] + rs3.f32[0]);
			fsflags(cpu, (double)rs1.f32[0] * (double)rs2.f32[0] + (double)rs3.f32[0], dst.f32[0]);
		} else if (fi.R4type.funct2 == 0x1) { // float64
			dst.f64 = rs1.f64 * rs2.f64 + rs3.f64;
			fsflags(cpu, (long double)rs1.f64 * (long double)rs2.f64 + (long double)rs3.f64, dst.f64);
		} else {
			cpu.trigger_exception(ILLEGAL_OPERATION);
		}
	},
	[] (char* buffer, size_t len, auto&, rv32i_instruction instr) RVPRINTR_ATTR {
		const rv32f_instruction fi { instr };
		static const std::array<const char*, 4> f2 {
			"FMADD.S", "FMADD.D", "???", "FMADD.Q"
		};
		return snprintf(buffer, len, "%s %s * %s + %s, %s", f2[fi.R4type.funct2],
						RISCV::flpname(fi.R4type.rs1),
						RISCV::flpname(fi.R4type.rs2),
						RISCV::flpname(fi.R4type.rs3),
						RISCV::flpname(fi.R4type.rd));
	});

	FLOAT_INSTR(FMSUB,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR
	{
		const rv32f_instruction fi { instr };
		auto& dst = cpu.registers().getfl(fi.R4type.rd);
		auto& rs1 = cpu.registers().getfl(fi.R4type.rs1);
		auto& rs2 = cpu.registers().getfl(fi.R4type.rs2);
		auto& rs3 = cpu.registers().getfl(fi.R4type.rs3);
		if (fi.R4type.funct2 == 0x0) { // float32
			dst.set_float(rs1.f32[0] * rs2.f32[0] - rs3.f32[0]);
			fsflags(cpu, (double)rs1.f32[0] * (double)rs2.f32[0] - (double)rs3.f32[0], dst.f32[0]);
		} else if (fi.R4type.funct2 == 0x1) { // float64
			dst.f64 = rs1.f64 * rs2.f64 - rs3.f64;
			fsflags(cpu, (long double)rs1.f64 * (long double)rs2.f64 - (long double)rs3.f64, dst.f64);
		} else {
			cpu.trigger_exception(ILLEGAL_OPERATION);
		}
	},
	[] (char* buffer, size_t len, auto&, rv32i_instruction instr) RVPRINTR_ATTR {
		const rv32f_instruction fi { instr };
		static const std::array<const char*, 4> f2 {
			"FMSUB.S", "FMSUB.D", "???", "FMSUB.Q"
		};
		return snprintf(buffer, len, "%s %s * %s - %s, %s", f2[fi.R4type.funct2],
						RISCV::flpname(fi.R4type.rs1),
						RISCV::flpname(fi.R4type.rs2),
						RISCV::flpname(fi.R4type.rs3),
						RISCV::flpname(fi.R4type.rd));
	});

	FLOAT_INSTR(FNMADD,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR
	{
		const rv32f_instruction fi { instr };
		auto& dst = cpu.registers().getfl(fi.R4type.rd);
		auto& rs1 = cpu.registers().getfl(fi.R4type.rs1);
		auto& rs2 = cpu.registers().getfl(fi.R4type.rs2);
		auto& rs3 = cpu.registers().getfl(fi.R4type.rs3);
		if (fi.R4type.funct2 == 0x0) { // float32
			dst.set_float(-(rs1.f32[0] * rs2.f32[0]) - rs3.f32[0]);
			fsflags(cpu, (double)-rs1.f32[0] * (double)rs2.f32[0] - (double)rs3.f32[0], dst.f32[0]);
		} else if (fi.R4type.funct2 == 0x1) { // float64
			dst.f64 = -(rs1.f64 * rs2.f64) - rs3.f64;
			fsflags(cpu, (long double)-rs1.f64 * (long double)rs2.f64 - (long double)rs3.f64, dst.f64);
		} else {
			cpu.trigger_exception(ILLEGAL_OPERATION);
		}
	},
	[] (char* buffer, size_t len, auto&, rv32i_instruction instr) RVPRINTR_ATTR {
		const rv32f_instruction fi { instr };
		static const std::array<const char*, 4> f2 {
			"FMADD.S", "FMADD.D", "???", "FMADD.Q"
		};
		return snprintf(buffer, len, "%s %s %s, %s", f2[fi.R4type.funct2],
						RISCV::flpname(fi.R4type.rs1),
						RISCV::flpname(fi.R4type.rs2),
						RISCV::flpname(fi.R4type.rd));
	});

	FLOAT_INSTR(FNMSUB,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR
	{
		const rv32f_instruction fi { instr };
		auto& dst = cpu.registers().getfl(fi.R4type.rd);
		auto& rs1 = cpu.registers().getfl(fi.R4type.rs1);
		auto& rs2 = cpu.registers().getfl(fi.R4type.rs2);
		auto& rs3 = cpu.registers().getfl(fi.R4type.rs3);
		if (fi.R4type.funct2 == 0x0) { // float32
			dst.set_float(-(rs1.f32[0] * rs2.f32[0]) + rs3.f32[0]);
			fsflags(cpu, (double)-rs1.f32[0] * (double)rs2.f32[0] + (double)rs3.f32[0], dst.f32[0]);
		} else if (fi.R4type.funct2 == 0x1) { // float64
			dst.f64 = -(rs1.f64 * rs2.f64) + rs3.f64;
			fsflags(cpu, (long double)-rs1.f64 * (long double)rs2.f64 + (long double)rs3.f64, dst.f64);
		} else {
			cpu.trigger_exception(ILLEGAL_OPERATION);
		}
	},
	[] (char* buffer, size_t len, auto&, rv32i_instruction instr) RVPRINTR_ATTR {
		const rv32f_instruction fi { instr };
		static const std::array<const char*, 4> f2 {
			"FNMSUB.S", "FNMSUB.D", "???", "FNMSUB.Q"
		};
		return snprintf(buffer, len, "%s -(%s * %s) + %s, %s",
						f2[fi.R4type.funct2],
						RISCV::flpname(fi.R4type.rs1),
						RISCV::flpname(fi.R4type.rs2),
						RISCV::flpname(fi.R4type.rs3),
						RISCV::flpname(fi.R4type.rd));
	});

	FLOAT_INSTR(FADD,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR
	{
		const rv32f_instruction fi { instr };
		auto& dst = cpu.registers().getfl(fi.R4type.rd);
		auto& rs1 = cpu.registers().getfl(fi.R4type.rs1);
		auto& rs2 = cpu.registers().getfl(fi.R4type.rs2);
		if (fi.R4type.funct2 == 0x0) { // float32
			dst.set_float(rs1.f32[0] + rs2.f32[0]);
			fsflags(cpu, (double)(rs1.f32[0]) + (double)(rs2.f32[0]), dst.f32[0]);
		} else if (fi.R4type.funct2 == 0x1) { // float64
			dst.f64 = rs1.f64 + rs2.f64;
			fsflags(cpu, (long double)(rs1.f64) + (long double)(rs2.f64), dst.f64);
		} else {
			cpu.trigger_exception(ILLEGAL_OPERATION);
		}
	},
	[] (char* buffer, size_t len, auto&, rv32i_instruction instr) RVPRINTR_ATTR {
		const rv32f_instruction fi { instr };
		static const std::array<const char*, 4> f2 {
			"FADD.S", "FADD.D", "???", "FADD.Q"
		};
		return snprintf(buffer, len, "%s %s %s, %s", f2[fi.R4type.funct2],
						RISCV::flpname(fi.R4type.rs1),
						RISCV::flpname(fi.R4type.rs2),
						RISCV::flpname(fi.R4type.rd));
	});

	FLOAT_INSTR(FSUB,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR
	{
		const rv32f_instruction fi { instr };
		auto& dst = cpu.registers().getfl(fi.R4type.rd);
		auto& rs1 = cpu.registers().getfl(fi.R4type.rs1);
		auto& rs2 = cpu.registers().getfl(fi.R4type.rs2);
		if (fi.R4type.funct2 == 0x0) { // float32
			dst.set_float(rs1.f32[0] - rs2.f32[0]);
			fsflags(cpu, (double)(rs1.f32[0]) - (double)(rs2.f32[0]), dst.f32[0]);
		} else if (fi.R4type.funct2 == 0x1) { // float64
			dst.f64 = rs1.f64 - rs2.f64;
			fsflags(cpu, (long double)(rs1.f64) - (long double)(rs2.f64), dst.f64);
		} else {
			cpu.trigger_exception(ILLEGAL_OPERATION);
		}
	},
	[] (char* buffer, size_t len, auto&, rv32i_instruction instr) RVPRINTR_ATTR {
		const rv32f_instruction fi { instr };
		static const std::array<const char*, 4> f2 {
			"FSUB.S", "FSUB.D", "???", "FSUB.Q"
		};
		return snprintf(buffer, len, "%s %s %s, %s", f2[fi.R4type.funct2],
						RISCV::flpname(fi.R4type.rs1),
						RISCV::flpname(fi.R4type.rs2),
						RISCV::flpname(fi.R4type.rd));
	});

	FLOAT_INSTR(FMUL,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR
	{
		const rv32f_instruction fi { instr };
		auto& dst = cpu.registers().getfl(fi.R4type.rd);
		auto& rs1 = cpu.registers().getfl(fi.R4type.rs1);
		auto& rs2 = cpu.registers().getfl(fi.R4type.rs2);
		if (fi.R4type.funct2 == 0x0) { // float32
			dst.set_float(rs1.f32[0] * rs2.f32[0]);
			fsflags(cpu, (double)(rs1.f32[0]) * (double)(rs2.f32[0]), dst.f32[0]);
		} else if (fi.R4type.funct2 == 0x1) { // float64
			dst.f64 = rs1.f64 * rs2.f64;
			fsflags(cpu, (long double)(rs1.f64) * (long double)(rs2.f64), dst.f64);
		} else {
			cpu.trigger_exception(ILLEGAL_OPERATION);
		}
	},
	[] (char* buffer, size_t len, auto&, rv32i_instruction instr) RVPRINTR_ATTR {
		const rv32f_instruction fi { instr };
		static const std::array<const char*, 4> f2 {
			"FMUL.S", "FMUL.D", "???", "FMUL.Q"
		};
		return snprintf(buffer, len, "%s %s %s, %s", f2[fi.R4type.funct2],
						RISCV::flpname(fi.R4type.rs1),
						RISCV::flpname(fi.R4type.rs2),
						RISCV::flpname(fi.R4type.rd));
	});

	FLOAT_INSTR(FDIV,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR
	{
		const rv32f_instruction fi { instr };
		auto& dst = cpu.registers().getfl(fi.R4type.rd);
		auto& rs1 = cpu.registers().getfl(fi.R4type.rs1);
		auto& rs2 = cpu.registers().getfl(fi.R4type.rs2);
		if (fi.R4type.funct2 == 0x0) { // fp32
			dst.set_float(rs1.f32[0] / rs2.f32[0]);
			fsflags(cpu, (double)(rs1.f32[0]) / (double)(rs2.f32[0]), dst.f32[0]);
		} else if (fi.R4type.funct2 == 0x1) { // fp64
			dst.f64 = rs1.f64 / rs2.f64;
			fsflags(cpu, (long double)(rs1.f64) / (long double)(rs2.f64), dst.f64);
		} else {
			cpu.trigger_exception(ILLEGAL_OPERATION);
		}
	},
	[] (char* buffer, size_t len, auto&, rv32i_instruction instr) RVPRINTR_ATTR {
		const rv32f_instruction fi { instr };
		static const std::array<const char*, 4> f2 {
			"FDIV.S", "FDIV.D", "???", "FDIV.Q"
		};
		return snprintf(buffer, len, "%s %s %s, %s", f2[fi.R4type.funct2],
						RISCV::flpname(fi.R4type.rs1),
						RISCV::flpname(fi.R4type.rs2),
						RISCV::flpname(fi.R4type.rd));
	});

	FLOAT_INSTR(FSQRT,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR
	{
		const rv32f_instruction fi { instr };
		auto& rs1 = cpu.registers().getfl(fi.R4type.rs1);
		auto& dst = cpu.registers().getfl(fi.R4type.rd);
		switch (fi.R4type.funct2) {
		case 0x0: // FSQRT.S
			dst.set_float(sqrtf(rs1.f32[0]));
			fsflags(cpu, std::sqrt((double)(rs1.f32[0])), dst.f32[0]);
			break;
		case 0x1: // FSQRT.D
			dst.f64 = sqrt(rs1.f64);
			fsflags(cpu, std::sqrt((long double)(rs1.f64)), dst.f64);
			break;
		default:
			cpu.trigger_exception(ILLEGAL_OPERATION);
		}
	},
	[] (char* buffer, size_t len, auto&, rv32i_instruction instr) RVPRINTR_ATTR {
		const rv32f_instruction fi { instr };
		static const std::array<const char*, 4> f2 {
			"FSQRT.S", "FSQRT.D", "???", "FSQRT.Q"
		};
		return snprintf(buffer, len, "%s %s, %s", f2[fi.R4type.funct2],
						RISCV::flpname(fi.R4type.rs1),
						RISCV::flpname(fi.R4type.rd));
	});

	FLOAT_INSTR(FMIN_FMAX,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_COLDATTR
	{
		const rv32f_instruction fi { instr };
		auto& rs1 = cpu.registers().getfl(fi.R4type.rs1);
		auto& rs2 = cpu.registers().getfl(fi.R4type.rs2);
		auto& dst = cpu.registers().getfl(fi.R4type.rd);

		switch (fi.R4type.funct3 | (fi.R4type.funct2 << 4))
		{
		case 0x0: // FMIN.S
			if constexpr (fcsr_emulation) {
				if (std::isnan(rs1.f32[0]) && std::isnan(rs2.f32[0]))
					dst.load_u32(CANONICAL_NAN_F32);
				else
					dst.set_float(std::fmin(rs1.f32[0], rs2.f32[0]));
			} else {
				dst.set_float(std::fmin(rs1.f32[0], rs2.f32[0]));
			}
			break;
		case 0x1: // FMAX.S
			if constexpr (fcsr_emulation) {
				if (std::isnan(rs1.f32[0]) && std::isnan(rs2.f32[0]))
					dst.load_u32(CANONICAL_NAN_F32);
				else
					dst.set_float(std::fmax(rs1.f32[0], rs2.f32[0]));
			} else {
				dst.set_float(std::fmax(rs1.f32[0], rs2.f32[0]));
			}
			break;
		case 0x10: // FMIN.D
			if constexpr (fcsr_emulation) {
				if (std::isnan(rs1.f64) && std::isnan(rs2.f64))
					dst.load_u64(CANONICAL_NAN_F64);
				else
					dst.f64 = std::fmin(rs1.f64, rs2.f64);
			} else {
				dst.f64 = std::fmin(rs1.f64, rs2.f64);
			}
			break;
		case 0x11: // FMAX.D
			if constexpr (fcsr_emulation) {
				if (std::isnan(rs1.f64) && std::isnan(rs2.f64))
					dst.load_u64(CANONICAL_NAN_F64);
				else
					dst.f64 = std::fmax(rs1.f64, rs2.f64);
			} else {
				dst.f64 = std::fmax(rs1.f64, rs2.f64);
			}
			break;
		default:
			cpu.trigger_exception(ILLEGAL_OPERATION);
		}
		if constexpr (fcsr_emulation) {
			if (is_signaling_nan(rs1.f32[0]) || is_signaling_nan(rs2.f32[0]))
				cpu.registers().fcsr().fflags = 16;
			else
				cpu.registers().fcsr().fflags = 0;
		}
	},
	[] (char* buffer, size_t len, auto&, rv32i_instruction instr) RVPRINTR_ATTR {
		const rv32f_instruction fi { instr };
		static const std::array<const char*, 8> insn {
			"FMIN", "FMAX", "???", "???", "???", "???", "???", "???"
		};
		return snprintf(buffer, len, "%s.%c %s %s, %s",
						insn[fi.R4type.funct3],
						RISCV::flpsize(fi.R4type.funct2),
						RISCV::flpname(fi.R4type.rs1),
						RISCV::flpname(fi.R4type.rs2),
						RISCV::regname(fi.R4type.rd));
	});

	FLOAT_INSTR(FEQ_FLT_FLE,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR
	{
		const rv32f_instruction fi { instr };
		auto& rs1 = cpu.registers().getfl(fi.R4type.rs1);
		auto& rs2 = cpu.registers().getfl(fi.R4type.rs2);
		auto& dst = cpu.reg(fi.R4type.rd);

		switch (fi.R4type.funct3 | (fi.R4type.funct2 << 4))
		{
		case 0x0: // FLE.S
			dst = (rs1.f32[0] <= rs2.f32[0]) ? 1 : 0;
			feqflags<true>(cpu, rs1.f32[0], rs2.f32[0], dst);
			break;
		case 0x1: // FLT.S
			dst = (rs1.f32[0] < rs2.f32[0]) ? 1 : 0;
			feqflags<true>(cpu, rs1.f32[0], rs2.f32[0], dst);
			break;
		case 0x2: // FEQ.S
			dst = (rs1.f32[0] == rs2.f32[0]) ? 1 : 0;
			feqflags<false>(cpu, rs1.f32[0], rs2.f32[0], dst);
			break;
		case 0x10: // FLE.D
			dst = (rs1.f64 <= rs2.f64) ? 1 : 0;
			feqflags<true>(cpu, rs1.f64, rs2.f64, dst);
			break;
		case 0x11: // FLT.D
			dst = (rs1.f64 < rs2.f64) ? 1 : 0;
			feqflags<true>(cpu, rs1.f64, rs2.f64, dst);
			break;
		case 0x12: // FEQ.D
			dst = (rs1.f64 == rs2.f64) ? 1 : 0;
			feqflags<false>(cpu, rs1.f64, rs2.f64, dst);
			break;
		default:
			cpu.trigger_exception(ILLEGAL_OPERATION);
		}
	},
	[] (char* buffer, size_t len, auto&, rv32i_instruction instr) RVPRINTR_ATTR {
		const rv32f_instruction fi { instr };
		static const std::array<const char*, 4> insn {
			"FLE", "FLT", "FEQ", "F???"
		};
		return snprintf(buffer, len, "%s.%c %s %s, %s",
						insn[fi.R4type.funct3],
						RISCV::flpsize(fi.R4type.funct2),
						RISCV::flpname(fi.R4type.rs1),
						RISCV::flpname(fi.R4type.rs2),
						RISCV::regname(fi.R4type.rd));
	});

	FLOAT_INSTR(FCVT_SD_DS,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR
	{
		const rv32f_instruction fi { instr };
		auto& rs1 = cpu.registers().getfl(fi.R4type.rs1);
		auto& dst = cpu.registers().getfl(fi.R4type.rd);
		switch (fi.R4type.funct2) {
		case 0x0: // FCVT.S.D (64 -> 32)
			dst.set_float(rs1.f64);
			break;
		case 0x1: // FCVT.D.S (32 -> 64)
			dst.f64 = rs1.f32[0];
			break;
		default:
			cpu.trigger_exception(ILLEGAL_OPERATION);
		}
	},
	[] (char* buffer, size_t len, auto&, rv32i_instruction instr) RVPRINTR_ATTR {
		const rv32f_instruction fi { instr };
		static const std::array<const char*, 4> f2 {
			"FCVT.S.D", "FCVT.D.S", "???", "???"
		};
		return snprintf(buffer, len, "%s %s, %s", f2[fi.R4type.funct2],
						RISCV::flpname(fi.R4type.rs1),
						RISCV::flpname(fi.R4type.rd));
	});

	FLOAT_INSTR(FCVT_W_SD,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR
	{
		const rv32f_instruction fi { instr };
		auto& rs1 = cpu.registers().getfl(fi.R4type.rs1);
		auto& dst = cpu.reg(fi.R4type.rd);
		switch (fi.R4type.funct2) {
		case 0x0: // from float32
			if (fi.R4type.rs2 == 0x0)
				dst = (int32_t) rs1.f32[0];
			else
				dst = (uint32_t) rs1.f32[0];
			return;
		case 0x1: // from float64
			switch (fi.R4type.rs2) {
			case 0x0: // FCVT.W.D
				dst = (int32_t) rs1.f64;
				return;
			case 0x1: // FCVT.WU.D
				dst = (uint32_t) rs1.f64;
				return;
			case 0x2: // FCVT.L.D
				dst = (int64_t) rs1.f64;
				return;
			case 0x3: // FCVT.LU.D
				dst = (uint64_t) rs1.f64;
				return;
			}
		}
		cpu.trigger_exception(ILLEGAL_OPERATION);
	},
	[] (char* buffer, size_t len, auto&, rv32i_instruction instr) RVPRINTR_ATTR {
		const rv32f_instruction fi { instr };
		static const std::array<const char*, 4> f2 {
			"FCVT.W.S", "FCVT.W.D", "???", "FCVT.W.Q"
		};
		return snprintf(buffer, len, "%s %s, %s", f2[fi.R4type.funct2],
						RISCV::flpname(fi.R4type.rs1),
						RISCV::regname(fi.R4type.rd));
	});

	FLOAT_INSTR(FCVT_SD_W,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR
	{
		const rv32f_instruction fi { instr };
		auto& rs1 = cpu.reg(fi.R4type.rs1);
		auto& dst = cpu.registers().getfl(fi.R4type.rd);
		switch (fi.R4type.funct2) {
		case 0x0: // to float32
			if (fi.R4type.rs2 == 0x0) // FCVT.S.W
				dst.set_float((int32_t)rs1);
			else // FCVT.S.WU
				dst.set_float((uint32_t)rs1);
			return;
		case 0x1: // to float64
			switch (fi.R4type.rs2) {
			case 0x0: // FCVT.D.W
				dst.f64 = (int32_t)rs1;
				return;
			case 0x1: // FCVT.D.WU
				dst.f64 = (uint32_t)rs1;
				return;
			case 0x2: // FCVT.D.L
				dst.f64 = (int64_t)rs1;
				return;
			case 0x3: // FCVT.D.LU
				dst.f64 = (uint64_t)rs1;
				return;
			}
		}
		cpu.trigger_exception(ILLEGAL_OPERATION);
	},
	[] (char* buffer, size_t len, auto&, rv32i_instruction instr) RVPRINTR_ATTR {
		const rv32f_instruction fi { instr };
		static const std::array<const char*, 4> f2 {
			"FCVT.S.W", "FCVT.D.W", "???", "FCVT.Q.W"
		};
		return snprintf(buffer, len, "%s %s, %s", f2[fi.R4type.funct2],
						RISCV::regname(fi.R4type.rs1),
						RISCV::flpname(fi.R4type.rd));
	});

	FLOAT_INSTR(FSGNJ_NX,
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR
	{
		const rv32f_instruction fi { instr };
		auto& rs1 = cpu.registers().getfl(fi.R4type.rs1);
		auto& rs2 = cpu.registers().getfl(fi.R4type.rs2);
		auto& dst = cpu.registers().getfl(fi.R4type.rd);
		switch (fi.R4type.funct3) {
		case 0x0: // FSGNJ
			switch (fi.R4type.funct2) {
			case 0x0: // float32
				dst.load_u32((rs2.lsign.sign << 31) | rs1.lsign.bits);
				break;
			case 0x1: // float64
				dst.i64 = ((uint64_t) rs2.usign.sign << 63) | rs1.usign.bits;
				break;
			default:
				cpu.trigger_exception(ILLEGAL_OPERATION);
			}
			break;
		case 0x1: // FSGNJ_N
			switch (fi.R4type.funct2) {
			case 0x0: // float32
				dst.load_u32((~rs2.lsign.sign << 31) | rs1.lsign.bits);
				break;
			case 0x1: // float64
				dst.i64 = (~(uint64_t) rs2.usign.sign << 63) | rs1.usign.bits;
				break;
			default:
				cpu.trigger_exception(ILLEGAL_OPERATION);
			}
			break;
		case 0x2: // FSGNJ_X
			switch (fi.R4type.funct2) {
			case 0x0: // float32
				dst.load_u32(((rs1.lsign.sign ^ rs2.lsign.sign) << 31) | rs1.lsign.bits);
				break;
			case 0x1: // float64
				dst.i64 = ((uint64_t)(rs1.usign.sign ^ rs2.usign.sign) << 63) | rs1.usign.bits;
				break;
			default:
				cpu.trigger_exception(ILLEGAL_OPERATION);
			}
			break;
		default:
			cpu.trigger_exception(ILLEGAL_OPERATION);
		}
	},
	[] (char* buffer, size_t len, auto&, rv32i_instruction instr) RVPRINTR_ATTR {
		const rv32f_instruction fi { instr };

		if (fi.R4type.rs1 == fi.R4type.rs2) {
			static const char* insn[4] = {"FMV", "FNEG", "FABS", "???"};
			return snprintf(buffer, len, "%s.%c %s, %s",
							insn[fi.R4type.funct3],
							RISCV::flpsize(fi.R4type.funct2),
							RISCV::flpname(fi.R4type.rs1),
							RISCV::flpname(fi.R4type.rd));
		}
		static const char* insn[4] = {"FSGNJ", "FSGNJN", "FSGNJX", "???"};
		return snprintf(buffer, len, "%s.%c %s %s, %s",
						insn[fi.R4type.funct3],
						RISCV::flpsize(fi.R4type.funct2),
						RISCV::flpname(fi.R4type.rs1),
						RISCV::flpname(fi.R4type.rs2),
						RISCV::flpname(fi.R4type.rd));
	});

	FLOAT_INSTR(FCLASS, // 1110 f3 = 0x1
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR
	{
		const rv32f_instruction fi { instr };
		auto& dst = cpu.reg(fi.R4type.rd);
		auto& rs1 = cpu.registers().getfl(fi.R4type.rs1);
		switch (fi.R4type.funct2) {
		case 0x0: // FCLASS.S
			dst = 0;
			if (rs1.f32[0] == -std::numeric_limits<float>::infinity())
				dst |= 1U << 0;
			if (rs1.f32[0] < 0)
				dst |= 1U << 1;
			if (rs1.f32[0] == -std::numeric_limits<float>::denorm_min())
				dst |= 1U << 2;
			if (rs1.f32[0] == -0.0)
				dst |= 1U << 3;
			if (rs1.f32[0] == +0.0)
				dst |= 1U << 4;
			if (rs1.f32[0] == std::numeric_limits<float>::denorm_min())
				dst |= 1U << 5;
			if (rs1.f32[0] >= std::numeric_limits<float>::epsilon())
				dst |= 1U << 6;
			if (rs1.f32[0] == std::numeric_limits<float>::infinity())
				dst |= 1U << 7;
			if (std::isnan(rs1.f32[0]))
				dst |= 3U << 8;
			return;
		case 0x1: // FCLASS.D
			dst = 0;
			if (rs1.f64 == -std::numeric_limits<double>::infinity())
				dst |= 1U << 0;
			if (rs1.f64 < 0)
				dst |= 1U << 1;
			if (rs1.f64 == -std::numeric_limits<double>::denorm_min())
				dst |= 1U << 2;
			if (rs1.f64 == -0.0)
				dst |= 1U << 3;
			if (rs1.f64 == +0.0)
				dst |= 1U << 4;
			if (rs1.f64 == std::numeric_limits<double>::denorm_min())
				dst |= 1U << 5;
			if (rs1.f64 >= std::numeric_limits<double>::epsilon())
				dst |= 1U << 6;
			if (rs1.f64 == std::numeric_limits<double>::infinity())
				dst |= 1U << 7;
			if (std::isnan(rs1.f64))
				dst |= 3U << 8;
			return;
		}
		cpu.trigger_exception(ILLEGAL_OPERATION);
	},
	[] (char* buffer, size_t len, auto&, rv32i_instruction instr) RVPRINTR_ATTR {
		const rv32f_instruction fi { instr };
		static const std::array<const char*, 4> f2 {
			"FCLASS.S", "FCLASS.D", "???", "FCLASS.Q"
		};
		return snprintf(buffer, len, "%s %s, %s", f2[fi.R4type.funct2],
						RISCV::flpname(fi.R4type.rs1),
						RISCV::regname(fi.R4type.rd));
	});

	FLOAT_INSTR(FMV_X_W, // 1110 f3 = 0x0
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR
	{
		const rv32f_instruction fi { instr };
		auto& dst = cpu.reg(fi.R4type.rd);
		auto& rs1 = cpu.registers().getfl(fi.R4type.rs1);
		switch (fi.R4type.funct2) {
		case 0x0: // FMV.X.W
// FMV.X.W moves the single-precision value in floating-point register rs1 represented in IEEE 754-
// 2008 encoding to the lower 32 bits of integer register rd. The bits are not modified in the transfer,
// and in particular, the payloads of non-canonical NaNs are preserved. For RV64, the higher 32 bits
// of the destination register are filled with copies of the floating-point number’s sign bit.
			dst = RVSIGNTYPE(cpu)(rs1.i32[0]);
			return;
		case 0x1: // FMV.X.D
			if constexpr (RVISGE64BIT(cpu)) {
				dst = RVSIGNTYPE(cpu)(rs1.i64);
				return;
			}
			break;
		}
		cpu.trigger_exception(ILLEGAL_OPERATION);
	},
	[] (char* buffer, size_t len, auto&, rv32i_instruction instr) RVPRINTR_ATTR {
		const rv32f_instruction fi { instr };
		static const std::array<const char*, 4> f2 {
			"FMV.X.W", "FMV.X.D", "???", "FMV.X.Q"
		};
		return snprintf(buffer, len, "%s %s, %s", f2[fi.R4type.funct2],
						RISCV::flpname(fi.R4type.rs1),
						RISCV::regname(fi.R4type.rd));
	});

	FLOAT_INSTR(FMV_W_X, // 1111
	[] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR
	{
		const rv32f_instruction fi { instr };
		auto& rs1 = cpu.reg(fi.R4type.rs1);
		auto& dst = cpu.registers().getfl(fi.R4type.rd);
		switch (fi.R4type.funct2) {
		case 0x0: // FMV.W.X
			dst.load_u32(rs1);
			return;
		case 0x1: // FMV.D.X
			if constexpr (RVISGE64BIT(cpu)) {
				dst.load_u64(rs1);
				return;
			}
			break;
		}
		cpu.trigger_exception(ILLEGAL_OPERATION);
	},
	[] (char* buffer, size_t len, auto&, rv32i_instruction instr) RVPRINTR_ATTR {
		const rv32f_instruction fi { instr };
		static const std::array<const char*, 4> f2 {
			"FMV.W.X", "FMV.D.X", "???", "FMV.Q.X"
		};
		return snprintf(buffer, len, "%s %s, %s", f2[fi.R4type.funct2],
						RISCV::regname(fi.R4type.rs1),
						RISCV::flpname(fi.R4type.rd));
	});
}
