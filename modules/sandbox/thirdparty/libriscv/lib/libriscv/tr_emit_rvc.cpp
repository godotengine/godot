
template <int W>
rv32i_instruction Emitter<W>::emit_rvc()
{
	#define CI_CODE(x, y) ((x << 13) | (y))
	const rv32c_instruction ci { instr };

	switch (ci.opcode())
	{
		case CI_CODE(0b000, 0b00): // C.ADDI4SPN
			if (ci.whole != 0) {
				instr.Itype.opcode = RV32I_OP_IMM;
				instr.Itype.funct3 = 0b000; // ADDI
				instr.Itype.rd = ci.CIW.srd + 8;
				instr.Itype.rs1 = 2; // sp
				instr.Itype.imm = ci.CIW.offset();
			}
			break;
		case CI_CODE(0b001, 0b00):
		case CI_CODE(0b010, 0b00):
		case CI_CODE(0b011, 0b00): {
			if (ci.CL.funct3 == 0x1) {
				// C.FLD
				instr.Itype.opcode = RV32F_LOAD;
				instr.Itype.funct3 = 0b011; // FLD
				instr.Itype.rd  = ci.CL.srd  + 8;
				instr.Itype.rs1 = ci.CL.srs1 + 8;
				instr.Itype.imm = ci.CSD.offset8();

				if (instr.Itype.signed_imm() != int32_t(ci.CSD.offset8()))
					throw MachineException(INVALID_PROGRAM, "Failed to sign-extend C.FLD immediate");
			}
			else if (ci.CL.funct3 == 0x2) {
				// C.LW
				instr.Itype.opcode = RV32I_LOAD;
				instr.Itype.funct3 = 0b010; // LW
				instr.Itype.rd  = ci.CL.srd  + 8;
				instr.Itype.rs1 = ci.CL.srs1 + 8;
				instr.Itype.imm = ci.CL.offset();
			}
			else if (ci.CL.funct3 == 0x3) {
				if constexpr (sizeof(address_t) >= 8) {
					// C.LD
					instr.Itype.opcode = RV32I_LOAD;
					instr.Itype.funct3 = 0b011; // LD
					instr.Itype.rd  = ci.CSD.srs2 + 8;
					instr.Itype.rs1 = ci.CSD.srs1 + 8;
					instr.Itype.imm = ci.CSD.offset8();
				} else {
					// C.FLW
					instr.Itype.opcode = RV32F_LOAD;
					instr.Itype.funct3 = 0b010; // FLW
					instr.Itype.rd  = ci.CL.srd  + 8;
					instr.Itype.rs1 = ci.CL.srs1 + 8;
					instr.Itype.imm = ci.CL.offset();
				}
			}
			// C.UNIMP
			break;
		}
		case CI_CODE(0b101, 0b00):
		case CI_CODE(0b110, 0b00):
		case CI_CODE(0b111, 0b00):
			switch (ci.CS.funct3) {
			case 4: // C.ILLEGAL
				break;
			case 5: { // C.FSD
				rv32f_instruction fi {instr};
				fi.Stype.opcode = RV32F_STORE;
				fi.Stype.funct3 = 0b011; // FSD
				fi.Stype.rs1 = ci.CSD.srs1 + 8;
				fi.Stype.rs2 = ci.CSD.srs2 + 8;
				const auto imm = ci.CSD.offset8();
				fi.Stype.imm04  = (imm >> 0) & 0x1F;
				fi.Stype.imm510 = (imm >> 5) & 0x3F;
				fi.Stype.imm11  = (imm >> 11) & 0x1;
				instr.whole = fi.whole;
				}
				break;
			case 6: { // C.SW
				instr.Stype.opcode = RV32I_STORE;
				instr.Stype.funct3 = 0b010; // SW
				instr.Stype.rs1 = ci.CS.srs1 + 8;
				instr.Stype.rs2 = ci.CS.srs2 + 8;
				const auto imm = ci.CS.offset4();
				instr.Stype.imm1 = imm & 0x1F;
				instr.Stype.imm2 = imm >> 5;
				}
				break;
			case 7: // C.SD / C.FSW
				if constexpr (W >= 8) {
					// C.SD
					instr.Stype.opcode = RV32I_STORE;
					instr.Stype.funct3 = 0b011; // SD
					instr.Stype.rs1 = ci.CSD.srs1 + 8;
					instr.Stype.rs2 = ci.CSD.srs2 + 8;
					const auto imm = ci.CSD.offset8();
					instr.Stype.imm1 = imm & 0x1F;
					instr.Stype.imm2 = imm >> 5;
				} else {
					// C.FSW
					rv32f_instruction fi {instr};
					fi.Stype.opcode = RV32F_STORE;
					fi.Stype.funct3 = 0b010; // FSW
					fi.Stype.rs1 = ci.CS.srs1 + 8;
					fi.Stype.rs2 = ci.CS.srs2 + 8;
					const auto imm = ci.CS.offset4();
					fi.Stype.imm04  = (imm >> 0) & 0x1F;
					fi.Stype.imm510 = (imm >> 5) & 0x3F;
					fi.Stype.imm11  = (imm >> 11) & 0x1;
					instr.whole = fi.whole;
				}
				break;
			}
			break;
		case CI_CODE(0b000, 0b01): { // C.ADDI
				instr.Itype.opcode = RV32I_OP_IMM;
				instr.Itype.funct3 = 0b000; // ADDI
				instr.Itype.rd = ci.CI.rd;
				instr.Itype.rs1 = ci.CI.rd;
				instr.Itype.imm = ci.CI.signed_imm();
			}
			break;
		case CI_CODE(0b010, 0b01): { // C.LI
			instr.Itype.opcode = RV32I_OP_IMM;
			instr.Itype.funct3 = 0b000; // ADDI
			instr.Itype.rd = ci.CI.rd;
			instr.Itype.rs1 = 0; // x0
			instr.Itype.imm = ci.CI.signed_imm();
			}
			break;
		case CI_CODE(0b011, 0b01): // C.ADDI16SP & C.LUI
			if (ci.CI.rd == 2) { // C.ADDI16SP
				instr.Itype.opcode = RV32I_OP_IMM;
				instr.Itype.funct3 = 0b000; // ADDI
				instr.Itype.rd = 2; // sp
				instr.Itype.rs1 = 2; // sp
				instr.Itype.imm = ci.CI16.signed_imm();
			}
			else if (ci.CI.rd != 0) { // C.LUI
				instr.Utype.opcode = RV32I_LUI;
				instr.Utype.rd = ci.CI.rd;
				instr.Utype.imm = ci.CI.signed_imm();
			}
			break; // C.ILLEGAL?
		case CI_CODE(0b001, 0b01):
			if constexpr (W == 4) {
				instr.Jtype.opcode = RV32I_JAL;
				instr.Jtype.rd = 1; // ra
				const auto imm = ci.CJ.signed_imm();
				instr.Jtype.imm3 = (imm >> 1) & 0x3FF;
				instr.Jtype.imm2 = (imm >> 11) & 0x1;
				instr.Jtype.imm1 = (imm >> 12) & 0xFF;
				instr.Jtype.imm4 = (imm < 0);

				if (instr.Jtype.jump_offset() != imm)
					throw MachineException(INVALID_PROGRAM, "Failed to sign-extend C.JAL immediate");
			} else { // C.ADDIW
				instr.Itype.opcode = RV64I_OP_IMM32;
				instr.Itype.funct3 = 0b000; // ADDIW
				instr.Itype.rd = ci.CI.rd;
				instr.Itype.rs1 = ci.CI.rd;
				instr.Itype.imm = ci.CI.signed_imm();
			}
			break;
		case CI_CODE(0b100, 0b01): { // Compressed ALU OPS
			switch (ci.CA.funct6 & 0x3) {
			case 0: // C.SRLI
				instr.Itype.opcode = RV32I_OP_IMM;
				instr.Itype.funct3 = 0b101; // SRLI
				instr.Itype.rd  = ci.CA.srd + 8;
				instr.Itype.rs1 = ci.CA.srd + 8;
				if constexpr (W >= 8) {
					instr.Itype.imm = ci.CAB.shift64_imm();
				} else {
					instr.Itype.imm = ci.CAB.shift_imm();
				}
				break;
			case 1: // C.SRAI (preserve sign)
				instr.Itype.opcode = RV32I_OP_IMM;
				instr.Itype.funct3 = 0b101; // SRAI
				instr.Itype.rd  = ci.CA.srd + 8;
				instr.Itype.rs1 = ci.CA.srd + 8;
				if constexpr (W >= 8) {
					instr.Itype.imm = ci.CAB.shift64_imm();
				} else {
					instr.Itype.imm = ci.CAB.shift_imm();
				}
				instr.Itype.imm |= 0x400; // SRAI
				break;
			case 2: // C.ANDI
				instr.Itype.opcode = RV32I_OP_IMM;
				instr.Itype.funct3 = 0b111; // ANDI
				instr.Itype.rd  = ci.CA.srd + 8;
				instr.Itype.rs1 = ci.CA.srd + 8;
				instr.Itype.imm = ci.CAB.signed_imm();
				break;
			case 3: // more ops
				switch (ci.CA.funct2 | (ci.CA.funct6 & 0x4))
				{
					case 0: // C.SUB
						instr.Rtype.opcode = RV32I_OP;
						instr.Rtype.funct3 = 0b000; // ADD
						instr.Rtype.funct7 = 0x20; // SUB
						instr.Rtype.rd  = ci.CA.srd + 8;
						instr.Rtype.rs1 = ci.CA.srd + 8;
						instr.Rtype.rs2 = ci.CA.srs2 + 8;
						break;
					case 1: // C.XOR
						instr.Rtype.opcode = RV32I_OP;
						instr.Rtype.funct3 = 0b100; // XOR
						instr.Rtype.funct7 = 0;
						instr.Rtype.rd  = ci.CA.srd + 8;
						instr.Rtype.rs1 = ci.CA.srd + 8;
						instr.Rtype.rs2 = ci.CA.srs2 + 8;
						break;
					case 2: // C.OR
						instr.Rtype.opcode = RV32I_OP;
						instr.Rtype.funct3 = 0b110; // OR
						instr.Rtype.funct7 = 0;
						instr.Rtype.rd  = ci.CA.srd + 8;
						instr.Rtype.rs1 = ci.CA.srd + 8;
						instr.Rtype.rs2 = ci.CA.srs2 + 8;
						break;
					case 3: // C.AND
						instr.Rtype.opcode = RV32I_OP;
						instr.Rtype.funct3 = 0b111; // AND
						instr.Rtype.funct7 = 0;
						instr.Rtype.rd  = ci.CA.srd + 8;
						instr.Rtype.rs1 = ci.CA.srd + 8;
						instr.Rtype.rs2 = ci.CA.srs2 + 8;
						break;
					case 0x4: // C.SUBW
					if constexpr (W >= 8) {
						instr.Rtype.opcode = RV64I_OP32;
						instr.Rtype.funct3 = 0b000; // ADD.W
						instr.Rtype.funct7 = 0x20; // SUB.W
						instr.Rtype.rd  = ci.CA.srd + 8;
						instr.Rtype.rs1 = ci.CA.srd + 8;
						instr.Rtype.rs2 = ci.CA.srs2 + 8;
						break;
					}
					case 0x5: // C.ADDW
					if constexpr (W >= 8) {
						instr.Rtype.opcode = RV64I_OP32;
						instr.Rtype.funct3 = 0b000; // ADD.W
						instr.Rtype.funct7 = 0; // ADD.W
						instr.Rtype.rd  = ci.CA.srd + 8;
						instr.Rtype.rs1 = ci.CA.srd + 8;
						instr.Rtype.rs2 = ci.CA.srs2 + 8;
						break;
					}
					case 0x6: // RESERVED
					case 0x7: // RESERVED
						break;
				}
			}
			} // Compressed ALU OPS
			break;
		case CI_CODE(0b101, 0b01): { // C.JMP
			instr.Jtype.opcode = RV32I_JAL;
			instr.Jtype.rd = 0;
			const auto imm = ci.CJ.signed_imm();
			instr.Jtype.imm3 = (imm >> 1) & 0x3FF;
			instr.Jtype.imm2 = (imm >> 11) & 0x1;
			instr.Jtype.imm1 = (imm >> 12) & 0xFF;
			instr.Jtype.imm4 = (imm < 0);

			if (instr.Jtype.jump_offset() != imm)
				throw MachineException(INVALID_PROGRAM, "Failed to sign-extend C.JMP immediate");
			}
			break;
		case CI_CODE(0b110, 0b01): { // C.BEQZ
			instr.Btype.opcode = RV32I_BRANCH;
			instr.Btype.funct3 = 0; // BEQ
			instr.Btype.rs1 = ci.CB.srs1 + 8;
			instr.Btype.rs2 = 0;
			const auto imm = ci.CB.signed_imm();
			instr.Btype.imm2 = (imm >> 1) & 0xF;
			instr.Btype.imm3 = (imm >> 5) & 0x3F;
			instr.Btype.imm1 = (imm >> 11) & 0x1;
			instr.Btype.imm4 = (imm < 0);

			if (instr.Btype.signed_imm() != imm)
				throw MachineException(INVALID_PROGRAM, "Failed to sign-extend C.BEQZ immediate");
			}
			break;
		case CI_CODE(0b111, 0b01): { // C.BNEZ
			instr.Btype.opcode = RV32I_BRANCH;
			instr.Btype.funct3 = 1; // BNE
			instr.Btype.rs1 = ci.CB.srs1 + 8;
			instr.Btype.rs2 = 0;
			const auto imm = ci.CB.signed_imm();
			instr.Btype.imm2 = (imm >> 1) & 0xF;
			instr.Btype.imm3 = (imm >> 5) & 0x3F;
			instr.Btype.imm1 = (imm >> 11) & 0x1;
			instr.Btype.imm4 = (imm < 0);

			if (instr.Btype.signed_imm() != imm)
				throw MachineException(INVALID_PROGRAM, "Failed to sign-extend C.BNEZ immediate");
			break;
			}
		// Quadrant 2
		case CI_CODE(0b001, 0b10): {
			// C.FLDSP
			instr.Itype.opcode = RV32F_LOAD;
			instr.Itype.funct3 = 0b011; // FLD
			instr.Itype.rd  = ci.CIFLD.rd;
			instr.Itype.rs1 = 2; // sp
			instr.Itype.imm = ci.CIFLD.offset();
			break;
		}
		case CI_CODE(0b000, 0b10):
		case CI_CODE(0b010, 0b10):
		case CI_CODE(0b011, 0b10): {
			if (ci.CI.funct3 == 0x0 && ci.CI.rd != 0) {
				// C.SLLI
				instr.Itype.opcode = RV32I_OP_IMM;
				instr.Itype.funct3 = 0b001; // SLLI
				instr.Itype.rd  = ci.CI.rd;
				instr.Itype.rs1 = ci.CI.rd;
				if constexpr (W >= 8) {
					instr.Itype.imm = ci.CI.shift64_imm();
				} else {
					instr.Itype.imm = ci.CI.shift_imm();
				}
			}
			else if (ci.CI2.funct3 == 0x2 && ci.CI2.rd != 0) {
				// C.LWSP
				instr.Itype.opcode = RV32I_LOAD;
				instr.Itype.funct3 = 0b010; // LW
				instr.Itype.rd  = ci.CI2.rd;
				instr.Itype.rs1 = 2; // sp
				instr.Itype.imm = ci.CI2.offset();
			}
			else if (ci.CI2.funct3 == 0x3) {
				if constexpr (sizeof(address_t) == 8) {
					if (ci.CI2.rd != 0) {
						// C.LDSP
						instr.Itype.opcode = RV32I_LOAD;
						instr.Itype.funct3 = 0b011; // LD
						instr.Itype.rd  = ci.CIFLD.rd;
						instr.Itype.rs1 = 2; // sp
						instr.Itype.imm = ci.CIFLD.offset();
					}
				} else {
					// C.FLWSP
					instr.Itype.opcode = RV32F_LOAD;
					instr.Itype.funct3 = 0b010; // FLW
					instr.Itype.rd  = ci.CI2.rd;
					instr.Itype.rs1 = 2; // sp
					instr.Itype.imm = ci.CI2.offset();
				}
			}
			else if (ci.CI.rd == 0) {
				// C.HINT
				instr.Itype.opcode = RV32I_OP_IMM;
				instr.Itype.funct3 = 0b000; // ADDI
				instr.Itype.rd  = 0;
				instr.Itype.rs1 = 0;
				instr.Itype.imm = 0;
			}
			// C.UNIMP?
			break;
		}
		case CI_CODE(0b100, 0b10): { // C.VARIOUS
			const bool topbit = ci.whole & (1 << 12);
			if (ci.CR.rd != 0 && ci.CR.rs2 == 0) {
				if (topbit) {
					// C.JALR ra, rd+0 (aka. RET)
					instr.Itype.opcode = RV32I_JALR;
					instr.Itype.rd = 1; // ra
					instr.Itype.rs1 = ci.CR.rd;
					instr.Itype.imm = 0;
				} else {
					// C.JR rd (Jump to register rd)
					instr.Itype.opcode = RV32I_JALR;
					instr.Itype.rd = 0; // x0
					instr.Itype.rs1 = ci.CR.rd;
					instr.Itype.imm = 0;
				}
			}
			else if (!topbit && ci.CR.rd != 0 && ci.CR.rs2 != 0)
			{	// C.MV rd, rs2
				instr.Itype.opcode = RV32I_OP_IMM;
				instr.Itype.funct3 = 0b000; // ADDI
				instr.Itype.rd  = ci.CR.rd;
				instr.Itype.rs1 = ci.CR.rs2;
				instr.Itype.imm = 0;
			}
			else if (ci.CR.rd != 0)
			{	// C.ADD rd, rd + rs2
				instr.Rtype.opcode = RV32I_OP;
				instr.Rtype.funct3 = 0b000; // ADD
				instr.Rtype.funct7 = 0;
				instr.Rtype.rd  = ci.CR.rd;
				instr.Rtype.rs1 = ci.CR.rd;
				instr.Rtype.rs2 = ci.CR.rs2;
			}
			else if (topbit && ci.CR.rd == 0 && ci.CR.rs2 == 0)
			{	// C.EBREAK
				instr.Itype.opcode = RV32I_SYSTEM;
				instr.Itype.funct3 = 0b000; // SYSTEM
				instr.Itype.rd = 0;
				instr.Itype.rs1 = 0;
				instr.Itype.imm = 0x001; // EBREAK
			}
		} break;
		case CI_CODE(0b101, 0b10):
		case CI_CODE(0b110, 0b10):
		case CI_CODE(0b111, 0b10): {
			if (ci.CSS.funct3 == 5) {
				// FSDSP
				rv32f_instruction fi {instr};
				fi.Stype.opcode = RV32F_STORE;
				fi.Stype.funct3 = 0b011; // FSD
				fi.Stype.rs1 = 2; // sp
				fi.Stype.rs2 = ci.CSFSD.rs2;
				const auto imm = ci.CSFSD.offset();
				fi.Stype.imm04  = (imm >> 0) & 0x1F;
				fi.Stype.imm510 = (imm >> 5) & 0x3F;
				fi.Stype.imm11  = (imm >> 11) & 0x1;
				instr.whole = fi.whole;
			}
			else if (ci.CSS.funct3 == 6) {
				// SWSP
				instr.Stype.opcode = RV32I_STORE;
				instr.Stype.funct3 = 0b010; // SW
				instr.Stype.rs1 = 2; // sp
				instr.Stype.rs2 = ci.CSS.rs2;
				const auto imm = ci.CSS.offset(4);
				instr.Stype.imm1 = (imm >> 0) & 0x1F;
				instr.Stype.imm2 = (imm >> 5);
			}
			else if (ci.CSS.funct3 == 7) {
				if constexpr (W == 8) {
					// SDSP
					instr.Stype.opcode = RV32I_STORE;
					instr.Stype.funct3 = 0b011; // SD
					instr.Stype.rs1 = 2; // sp
					instr.Stype.rs2 = ci.CSFSD.rs2;
					const auto imm = ci.CSFSD.offset();
					instr.Stype.imm1 = (imm >> 0) & 0x1F;
					instr.Stype.imm2 = (imm >> 5);
				} else {
					// FSWSP
					rv32f_instruction fi {instr};
					fi.Stype.opcode = RV32F_STORE;
					fi.Stype.funct3 = 0b010; // FSW
					fi.Stype.rs1 = 2; // sp
					fi.Stype.rs2 = ci.CSS.rs2;
					const auto imm = ci.CSS.offset(4);
					fi.Stype.imm04  = (imm >> 0) & 0x1F;
					fi.Stype.imm510 = (imm >> 5) & 0x3F;
					fi.Stype.imm11  = (imm >> 11) & 0x1;
					instr.whole = fi.whole;
				}
			}
			// C.UNIMP?
			break;
		}
	} // switch
	#undef CI_CODE
	return instr;
}
