template <typename T>
T& view_as(rv32i_instruction& i) {
	static_assert(sizeof(T) == sizeof(i), "Must be same size as instruction!");
	return *(T*) &i;
}

template <int W>
static void fused_li_ecall(
	typename CPU<W>::instr_pair& i1, typename CPU<W>::instr_pair& i2, int sysno)
{
	union FusedSyscall {
		struct {
			uint8_t lower;
			uint8_t ilen;
			uint16_t sysno;
		};
		uint32_t whole;
	};
	const FusedSyscall fsys = { {
		.lower = (uint8_t)  i2.second.half[0],  // Trick emulator to step
		.ilen  = (uint8_t)  i1.second.length(), // over second instruction.
		.sysno = (uint16_t) sysno,
	} };
	i1.second.whole = fsys.whole;
	i1.first = [] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		auto& fop = view_as<FusedSyscall> (instr);
		if constexpr (compressed_enabled)
			cpu.increment_pc(fop.ilen);
		else
			cpu.increment_pc(4);
		cpu.reg(REG_ECALL) = fop.sysno;
		cpu.machine().unchecked_system_call(fop.sysno);
	};
}

template <int W, typename T>
static void fused_store(
	typename CPU<W>::instr_pair& i1, typename CPU<W>::instr_pair& i2)
{
	union FusedStores {
		struct {
			uint32_t imm    : 12;
			uint32_t src1   : 5;
			uint32_t src2   : 5;
			uint32_t dst    : 5;
			uint32_t opcode : 5;
		};
		static bool sign(uint32_t imm) {
			return imm & 0x800;
		}
		static int64_t signed_imm(uint32_t imm) {
			const uint64_t ext = 0xFFFFFFFFFFFFF000;
			return imm | (sign(imm) ? ext : 0);
		}
		uint32_t whole;
	};
	i1.second.whole = FusedStores {{
		.imm = (uint32_t) (i1.second.Stype.imm1 | (i1.second.Stype.imm2 << 5)),
		.src1 = i1.second.Stype.rs2,
		.src2 = i2.second.Stype.rs2,
		.dst  = i1.second.Stype.rs1,
		.opcode = i1.second.Stype.opcode,
	}}.whole;
	i1.first = [] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
		auto& fop = view_as<FusedStores> (instr);

		const auto& value1 = cpu.reg(fop.src1);
		const auto addr1  = cpu.reg(fop.dst) + fop.signed_imm(fop.imm);
		cpu.machine().memory.template write<T>(addr1, value1);

		const auto& value2 = cpu.reg(fop.src2);
		const auto addr2  = cpu.reg(fop.dst) + fop.signed_imm(fop.imm) - sizeof(T);
		cpu.machine().memory.template write<T>(addr2, value2);

		cpu.increment_pc(4);
	};
}

template <int W>
bool CPU<W>::try_fuse(instr_pair i1, instr_pair i2) const
{
	// LI + ECALL fused
	if (i1.first == DECODED_INSTR(OP_IMM_LI).handler &&
		i2.first == DECODED_INSTR(SYSCALL).handler)
	{
		// fastest possible system calls
		const uint16_t sysno = i1.second.Itype.signed_imm();
		if (i1.second.Itype.rd == REG_ECALL && sysno < RISCV_SYSCALLS_MAX)
		{
			fused_li_ecall<W>(i1, i2, sysno);
			return true;
		}
	}
	// ADDI x, x + ADDI y, y fused
	if (i1.first == DECODED_INSTR(OP_IMM_ADDI).handler &&
		i2.first == DECODED_INSTR(OP_IMM_ADDI).handler)
	{
		if (i1.second.Itype.rd == i1.second.Itype.rs1 && i1.second.Itype.rd < 16)
		if (i2.second.Itype.rd == i2.second.Itype.rs1 && i2.second.Itype.rd < 16)
		if constexpr (!compressed_enabled)
		{
			union FusedAddi {
				struct {
					uint32_t addi1 : 12;
					uint32_t reg1  : 4;
					uint32_t addi2 : 12;
					uint32_t reg2  : 4;
				};
				static bool sign(uint32_t imm) {
					return imm & 0x800;
				}
				static int64_t signed_imm(uint32_t imm) {
					const uint64_t ext = 0xFFFFFFFFFFFFF000;
					return imm | (sign(imm) ? ext : 0);
				}
				uint32_t whole;
			};
			FusedAddi fop;
			fop.addi1 = i1.second.Itype.imm;
			fop.reg1  = i1.second.Itype.rd;
			fop.addi2 = i2.second.Itype.imm;
			fop.reg2  = i2.second.Itype.rd;
			i1.second.whole = fop.whole;
			i1.first = [] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
				auto& fop = view_as<FusedAddi> (instr);
				cpu.reg(fop.reg1) += FusedAddi::signed_imm(fop.addi1);
				cpu.reg(fop.reg2) += FusedAddi::signed_imm(fop.addi2);
				cpu.increment_pc(4);
			};
			return true;
		}
	}
	// LI x, n + LI y, m fused
	if (i1.first == DECODED_INSTR(OP_IMM_LI).handler &&
		i2.first == DECODED_INSTR(OP_IMM_LI).handler)
	{
		if (i1.second.Itype.rd < 16 && i2.second.Itype.rd < 16)
		if constexpr (!compressed_enabled)
		{
			union FusedLili {
				struct {
					uint32_t li1   : 12;
					uint32_t reg1  : 4;
					uint32_t li2   : 12;
					uint32_t reg2  : 4;
				};
				static bool sign(uint32_t imm) {
					return imm & 0x800;
				}
				static int64_t signed_imm(uint32_t imm) {
					const uint64_t ext = 0xFFFFFFFFFFFFF000;
					return imm | (sign(imm) ? ext : 0);
				}
				uint32_t whole;
			};
			const FusedLili lili = { {
				.li1  = i1.second.Itype.imm,
				.reg1 = i1.second.Itype.rd,
				.li2  = i2.second.Itype.imm,
				.reg2 = i2.second.Itype.rd
			} };
			i1.second.whole = lili.whole;
			i1.first = [] (auto& cpu, rv32i_instruction instr) RVINSTR_ATTR {
				auto& fop = view_as<FusedLili> (instr);
				cpu.reg(fop.reg1) = FusedLili::signed_imm(fop.li1);
				cpu.reg(fop.reg2) = FusedLili::signed_imm(fop.li2);
				cpu.increment_pc(4);
			};
			return true;
		}
	}
	// ST x, n-0*W + ST y, n-1*W fused
	if (i1.first == DECODED_INSTR(STORE_I32_IMM).handler &&
		i2.first == DECODED_INSTR(STORE_I32_IMM).handler &&
		i1.second.Stype.signed_imm()-4 == i2.second.Stype.signed_imm() &&
		i1.second.Stype.rs1 == i2.second.Stype.rs1)
	{
		fused_store<W, uint32_t> (i1, i2);
		return true;
	}
	if (i1.first == DECODED_INSTR(STORE_I64_IMM).handler &&
		i2.first == DECODED_INSTR(STORE_I64_IMM).handler &&
		i1.second.Stype.signed_imm()-8 == i2.second.Stype.signed_imm() &&
		i1.second.Stype.rs1 == i2.second.Stype.rs1 &&
		!compressed_enabled)
	{
		fused_store<W, uint64_t> (i1, i2);
		return true;
	}
# ifdef RISCV_EXT_COMPRESSED
	// C.LI + ECALL fused
	else if (i1.first == DECODED_INSTR(C1_LI).handler &&
		i2.first == DECODED_INSTR(SYSCALL).handler)
	{
		const rv32c_instruction ci { i1.second };
		const uint16_t sysno = ci.CI.signed_imm();
		if (ci.CI.rd == REG_ECALL && sysno < RISCV_SYSCALLS_MAX)
		{
			fused_li_ecall<W>(i1, i2, sysno);
			return true;
		}
	}
# endif
	return false;
}
