/*
 *    Stack-less Just-In-Time compiler
 *
 *    Copyright Zoltan Herczeg (hzmester@freemail.hu). All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are
 * permitted provided that the following conditions are met:
 *
 *   1. Redistributions of source code must retain the above copyright notice, this list of
 *      conditions and the following disclaimer.
 *
 *   2. Redistributions in binary form must reproduce the above copyright notice, this list
 *      of conditions and the following disclaimer in the documentation and/or other materials
 *      provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER(S) AND CONTRIBUTORS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDER(S) OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <sys/auxv.h>

#ifdef __ARCH__
#define ENABLE_STATIC_FACILITY_DETECTION 1
#else
#define ENABLE_STATIC_FACILITY_DETECTION 0
#endif
#define ENABLE_DYNAMIC_FACILITY_DETECTION 1

SLJIT_API_FUNC_ATTRIBUTE const char* sljit_get_platform_name(void)
{
	return "s390x" SLJIT_CPUINFO;
}

/* Instructions. */
typedef sljit_uw sljit_ins;

/* Instruction tags (most significant halfword). */
static const sljit_ins sljit_ins_const = (sljit_ins)1 << 48;

#define TMP_REG1	(SLJIT_NUMBER_OF_REGISTERS + 2)
#define TMP_REG2	(SLJIT_NUMBER_OF_REGISTERS + 3)

static const sljit_u8 reg_map[SLJIT_NUMBER_OF_REGISTERS + 4] = {
	0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 0, 1
};

/* there are also a[2-15] available, but they are slower to access and
 * their use is limited as mundaym explained:
 *   https://github.com/zherczeg/sljit/pull/91#discussion_r486895689
 */

/* General Purpose Registers [0-15]. */
typedef sljit_uw sljit_gpr;

/*
 * WARNING
 * the following code is non standard and should be improved for
 * consistency, but doesn't use SLJIT_NUMBER_OF_REGISTERS based
 * registers because r0 and r1 are the ABI recommended volatiles.
 * there is a gpr() function that maps sljit to physical register numbers
 * that should be used instead of the usual index into reg_map[] and
 * will be retired ASAP (TODO: carenas)
 */

static const sljit_gpr r0 = 0;		/* reg_map[SLJIT_NUMBER_OF_REGISTERS + 2]: 0 in address calculations; reserved */
static const sljit_gpr r1 = 1;		/* reg_map[SLJIT_NUMBER_OF_REGISTERS + 3]: reserved */
static const sljit_gpr r2 = 2;		/* reg_map[1]: 1st argument */
static const sljit_gpr r3 = 3;		/* reg_map[2]: 2nd argument */
static const sljit_gpr r4 = 4;		/* reg_map[3]: 3rd argument */
static const sljit_gpr r5 = 5;		/* reg_map[4]: 4th argument */
static const sljit_gpr r6 = 6;		/* reg_map[5]: 5th argument; 1st saved register */
static const sljit_gpr r7 = 7;		/* reg_map[6] */
static const sljit_gpr r8 = 8;		/* reg_map[7] */
static const sljit_gpr r9 = 9;		/* reg_map[8] */
static const sljit_gpr r10 = 10;	/* reg_map[9] */
static const sljit_gpr r11 = 11;	/* reg_map[10] */
static const sljit_gpr r12 = 12;	/* reg_map[11]: GOT */
static const sljit_gpr r13 = 13;	/* reg_map[12]: Literal Pool pointer */
static const sljit_gpr r14 = 14;	/* reg_map[0]: return address and flag register */
static const sljit_gpr r15 = 15;	/* reg_map[SLJIT_NUMBER_OF_REGISTERS + 1]: stack pointer */

/* WARNING: r12 and r13 shouldn't be used as per ABI recommendation */
/* TODO(carenas): r12 might conflict in PIC code, reserve? */
/* TODO(carenas): r13 is usually pointed to "pool" per ABI, using a tmp
 *                like we do know might be faster though, reserve?
 */

/* TODO(carenas): should be named TMP_REG[1-2] for consistency */
#define tmp0	r0
#define tmp1	r1

/* TODO(carenas): flags should move to a different register so that
 *                link register doesn't need to change
 */

/* When reg cannot be unused. */
#define IS_GPR_REG(reg)		((reg > 0) && (reg) <= SLJIT_SP)

/* Link registers. The normal link register is r14, but since
   we use that for flags we need to use r0 instead to do fast
   calls so that flags are preserved. */
static const sljit_gpr link_r = 14;     /* r14 */
static const sljit_gpr fast_link_r = 0; /* r0 */

#define TMP_FREG1	(0)

static const sljit_u8 freg_map[SLJIT_NUMBER_OF_FLOAT_REGISTERS + 1] = {
	1, 0, 2, 4, 6, 3, 5, 7, 15, 14, 13, 12, 11, 10, 9, 8,
};

#define R0A(r) (r)
#define R4A(r) ((r) << 4)
#define R8A(r) ((r) << 8)
#define R12A(r) ((r) << 12)
#define R16A(r) ((r) << 16)
#define R20A(r) ((r) << 20)
#define R28A(r) ((r) << 28)
#define R32A(r) ((r) << 32)
#define R36A(r) ((r) << 36)

#define R0(r) ((sljit_ins)reg_map[r])

#define F0(r) ((sljit_ins)freg_map[r])
#define F4(r) (R4A((sljit_ins)freg_map[r]))
#define F20(r) (R20A((sljit_ins)freg_map[r]))
#define F36(r) (R36A((sljit_ins)freg_map[r]))

struct sljit_s390x_const {
	struct sljit_const const_; /* must be first */
	sljit_sw init_value;       /* required to build literal pool */
};

/* Convert SLJIT register to hardware register. */
static SLJIT_INLINE sljit_gpr gpr(sljit_s32 r)
{
	SLJIT_ASSERT(r >= 0 && r < (sljit_s32)(sizeof(reg_map) / sizeof(reg_map[0])));
	return reg_map[r];
}

static SLJIT_INLINE sljit_gpr fgpr(sljit_s32 r)
{
	SLJIT_ASSERT(r >= 0 && r < (sljit_s32)(sizeof(freg_map) / sizeof(freg_map[0])));
	return freg_map[r];
}

/* Size of instruction in bytes. Tags must already be cleared. */
static SLJIT_INLINE sljit_uw sizeof_ins(sljit_ins ins)
{
	/* keep faulting instructions */
	if (ins == 0)
		return 2;

	if ((ins & 0x00000000ffffL) == ins)
		return 2;
	if ((ins & 0x0000ffffffffL) == ins)
		return 4;
	if ((ins & 0xffffffffffffL) == ins)
		return 6;

	SLJIT_UNREACHABLE();
	return (sljit_uw)-1;
}

static sljit_s32 push_inst(struct sljit_compiler *compiler, sljit_ins ins)
{
	sljit_ins *ibuf = (sljit_ins *)ensure_buf(compiler, sizeof(sljit_ins));
	FAIL_IF(!ibuf);
	*ibuf = ins;
	compiler->size++;
	return SLJIT_SUCCESS;
}

static sljit_s32 encode_inst(void **ptr, sljit_ins ins)
{
	sljit_u16 *ibuf = (sljit_u16 *)*ptr;
	sljit_uw size = sizeof_ins(ins);

	SLJIT_ASSERT((size & 6) == size);
	switch (size) {
	case 6:
		*ibuf++ = (sljit_u16)(ins >> 32);
		/* fallthrough */
	case 4:
		*ibuf++ = (sljit_u16)(ins >> 16);
		/* fallthrough */
	case 2:
		*ibuf++ = (sljit_u16)(ins);
	}
	*ptr = (void*)ibuf;
	return SLJIT_SUCCESS;
}

#define SLJIT_ADD_SUB_NO_COMPARE(status_flags_state) \
	(((status_flags_state) & (SLJIT_CURRENT_FLAGS_ADD | SLJIT_CURRENT_FLAGS_SUB)) \
		&& !((status_flags_state) & SLJIT_CURRENT_FLAGS_COMPARE))

/* Map the given type to a 4-bit condition code mask. */
static SLJIT_INLINE sljit_u8 get_cc(struct sljit_compiler *compiler, sljit_s32 type) {
	const sljit_u8 cc0 = 1 << 3; /* equal {,to zero} */
	const sljit_u8 cc1 = 1 << 2; /* less than {,zero} */
	const sljit_u8 cc2 = 1 << 1; /* greater than {,zero} */
	const sljit_u8 cc3 = 1 << 0; /* {overflow,NaN} */

	switch (type) {
	case SLJIT_EQUAL:
		if (SLJIT_ADD_SUB_NO_COMPARE(compiler->status_flags_state)) {
			sljit_s32 type = GET_FLAG_TYPE(compiler->status_flags_state);
			if (type >= SLJIT_SIG_LESS && type <= SLJIT_SIG_LESS_EQUAL)
				return cc0;
			if (type == SLJIT_OVERFLOW)
				return (cc0 | cc3);
			return (cc0 | cc2);
		}
		/* fallthrough */

	case SLJIT_EQUAL_F64:
		return cc0;

	case SLJIT_NOT_EQUAL:
		if (SLJIT_ADD_SUB_NO_COMPARE(compiler->status_flags_state)) {
			sljit_s32 type = GET_FLAG_TYPE(compiler->status_flags_state);
			if (type >= SLJIT_SIG_LESS && type <= SLJIT_SIG_LESS_EQUAL)
				return (cc1 | cc2 | cc3);
			if (type == SLJIT_OVERFLOW)
				return (cc1 | cc2);
			return (cc1 | cc3);
		}
		/* fallthrough */

	case SLJIT_NOT_EQUAL_F64:
		return (cc1 | cc2 | cc3);

	case SLJIT_LESS:
		return cc1;

	case SLJIT_GREATER_EQUAL:
		return (cc0 | cc2 | cc3);

	case SLJIT_GREATER:
		if (compiler->status_flags_state & SLJIT_CURRENT_FLAGS_COMPARE)
			return cc2;
		return cc3;

	case SLJIT_LESS_EQUAL:
		if (compiler->status_flags_state & SLJIT_CURRENT_FLAGS_COMPARE)
			return (cc0 | cc1);
		return (cc0 | cc1 | cc2);

	case SLJIT_SIG_LESS:
	case SLJIT_LESS_F64:
		return cc1;

	case SLJIT_NOT_CARRY:
		if (compiler->status_flags_state & SLJIT_CURRENT_FLAGS_SUB)
			return (cc2 | cc3);
		/* fallthrough */

	case SLJIT_SIG_LESS_EQUAL:
	case SLJIT_LESS_EQUAL_F64:
		return (cc0 | cc1);

	case SLJIT_CARRY:
		if (compiler->status_flags_state & SLJIT_CURRENT_FLAGS_SUB)
			return (cc0 | cc1);
		/* fallthrough */

	case SLJIT_SIG_GREATER:
		/* Overflow is considered greater, see SLJIT_SUB. */
		return cc2 | cc3;

	case SLJIT_SIG_GREATER_EQUAL:
		return (cc0 | cc2 | cc3);

	case SLJIT_OVERFLOW:
		if (compiler->status_flags_state & SLJIT_SET_Z)
			return (cc2 | cc3);
		/* fallthrough */

	case SLJIT_UNORDERED_F64:
		return cc3;

	case SLJIT_NOT_OVERFLOW:
		if (compiler->status_flags_state & SLJIT_SET_Z)
			return (cc0 | cc1);
		/* fallthrough */

	case SLJIT_ORDERED_F64:
		return (cc0 | cc1 | cc2);

	case SLJIT_GREATER_F64:
		return cc2;

	case SLJIT_GREATER_EQUAL_F64:
		return (cc0 | cc2);
	}

	SLJIT_UNREACHABLE();
	return (sljit_u8)-1;
}

/* Facility to bit index mappings.
   Note: some facilities share the same bit index. */
typedef sljit_uw facility_bit;
#define STORE_FACILITY_LIST_EXTENDED_FACILITY 7
#define FAST_LONG_DISPLACEMENT_FACILITY 19
#define EXTENDED_IMMEDIATE_FACILITY 21
#define GENERAL_INSTRUCTION_EXTENSION_FACILITY 34
#define DISTINCT_OPERAND_FACILITY 45
#define HIGH_WORD_FACILITY 45
#define POPULATION_COUNT_FACILITY 45
#define LOAD_STORE_ON_CONDITION_1_FACILITY 45
#define MISCELLANEOUS_INSTRUCTION_EXTENSIONS_1_FACILITY 49
#define LOAD_STORE_ON_CONDITION_2_FACILITY 53
#define MISCELLANEOUS_INSTRUCTION_EXTENSIONS_2_FACILITY 58
#define VECTOR_FACILITY 129
#define VECTOR_ENHANCEMENTS_1_FACILITY 135

/* Report whether a facility is known to be present due to the compiler
   settings. This function should always be compiled to a constant
   value given a constant argument. */
static SLJIT_INLINE int have_facility_static(facility_bit x)
{
#if ENABLE_STATIC_FACILITY_DETECTION
	switch (x) {
	case FAST_LONG_DISPLACEMENT_FACILITY:
		return (__ARCH__ >=  6 /* z990 */);
	case EXTENDED_IMMEDIATE_FACILITY:
	case STORE_FACILITY_LIST_EXTENDED_FACILITY:
		return (__ARCH__ >=  7 /* z9-109 */);
	case GENERAL_INSTRUCTION_EXTENSION_FACILITY:
		return (__ARCH__ >=  8 /* z10 */);
	case DISTINCT_OPERAND_FACILITY:
		return (__ARCH__ >=  9 /* z196 */);
	case MISCELLANEOUS_INSTRUCTION_EXTENSIONS_1_FACILITY:
		return (__ARCH__ >= 10 /* zEC12 */);
	case LOAD_STORE_ON_CONDITION_2_FACILITY:
	case VECTOR_FACILITY:
		return (__ARCH__ >= 11 /* z13 */);
	case MISCELLANEOUS_INSTRUCTION_EXTENSIONS_2_FACILITY:
	case VECTOR_ENHANCEMENTS_1_FACILITY:
		return (__ARCH__ >= 12 /* z14 */);
	default:
		SLJIT_UNREACHABLE();
	}
#endif
	return 0;
}

static SLJIT_INLINE unsigned long get_hwcap()
{
	static unsigned long hwcap = 0;
	if (SLJIT_UNLIKELY(!hwcap)) {
		hwcap = getauxval(AT_HWCAP);
		SLJIT_ASSERT(hwcap != 0);
	}
	return hwcap;
}

static SLJIT_INLINE int have_stfle()
{
	if (have_facility_static(STORE_FACILITY_LIST_EXTENDED_FACILITY))
		return 1;

	return (get_hwcap() & HWCAP_S390_STFLE);
}

/* Report whether the given facility is available. This function always
   performs a runtime check. */
static int have_facility_dynamic(facility_bit x)
{
#if ENABLE_DYNAMIC_FACILITY_DETECTION
	static struct {
		sljit_uw bits[4];
	} cpu_features;
	size_t size = sizeof(cpu_features);
	const sljit_uw word_index = x >> 6;
	const sljit_uw bit_index = ((1UL << 63) >> (x & 63));

	SLJIT_ASSERT(x < size * 8);
	if (SLJIT_UNLIKELY(!have_stfle()))
		return 0;

	if (SLJIT_UNLIKELY(cpu_features.bits[0] == 0)) {
		__asm__ __volatile__ (
			"lgr   %%r0, %0;"
			"stfle 0(%1);"
			/* outputs  */:
			/* inputs   */: "d" ((size / 8) - 1), "a" (&cpu_features)
			/* clobbers */: "r0", "cc", "memory"
		);
		SLJIT_ASSERT(cpu_features.bits[0] != 0);
	}
	return (cpu_features.bits[word_index] & bit_index) != 0;
#else
	return 0;
#endif
}

#define HAVE_FACILITY(name, bit) \
static SLJIT_INLINE int name() \
{ \
	static int have = -1; \
	/* Static check first. May allow the function to be optimized away. */ \
	if (have_facility_static(bit)) \
		have = 1; \
	else if (SLJIT_UNLIKELY(have < 0)) \
		have = have_facility_dynamic(bit) ? 1 : 0; \
\
	return have; \
}

HAVE_FACILITY(have_eimm,    EXTENDED_IMMEDIATE_FACILITY)
HAVE_FACILITY(have_ldisp,   FAST_LONG_DISPLACEMENT_FACILITY)
HAVE_FACILITY(have_genext,  GENERAL_INSTRUCTION_EXTENSION_FACILITY)
HAVE_FACILITY(have_lscond1, LOAD_STORE_ON_CONDITION_1_FACILITY)
HAVE_FACILITY(have_lscond2, LOAD_STORE_ON_CONDITION_2_FACILITY)
HAVE_FACILITY(have_misc2,   MISCELLANEOUS_INSTRUCTION_EXTENSIONS_2_FACILITY)
#undef HAVE_FACILITY

#define is_u12(d)	(0 <= (d) && (d) <= 0x00000fffL)
#define is_u32(d)	(0 <= (d) && (d) <= 0xffffffffL)

#define CHECK_SIGNED(v, bitlen) \
	((v) >= -(1 << ((bitlen) - 1)) && (v) < (1 << ((bitlen) - 1)))

#define is_s8(d)	CHECK_SIGNED((d), 8)
#define is_s16(d)	CHECK_SIGNED((d), 16)
#define is_s20(d)	CHECK_SIGNED((d), 20)
#define is_s32(d)	((d) == (sljit_s32)(d))

static SLJIT_INLINE sljit_ins disp_s20(sljit_s32 d)
{
	SLJIT_ASSERT(is_s20(d));

	sljit_uw dh = (d >> 12) & 0xff;
	sljit_uw dl = (d << 8) & 0xfff00;
	return (dh | dl) << 8;
}

/* TODO(carenas): variadic macro is not strictly needed */
#define SLJIT_S390X_INSTRUCTION(op, ...) \
static SLJIT_INLINE sljit_ins op(__VA_ARGS__)

/* RR form instructions. */
#define SLJIT_S390X_RR(name, pattern) \
SLJIT_S390X_INSTRUCTION(name, sljit_gpr dst, sljit_gpr src) \
{ \
	return (pattern) | ((dst & 0xf) << 4) | (src & 0xf); \
}

/* AND */
SLJIT_S390X_RR(nr,   0x1400)

/* BRANCH AND SAVE */
SLJIT_S390X_RR(basr, 0x0d00)

/* BRANCH ON CONDITION */
SLJIT_S390X_RR(bcr,  0x0700) /* TODO(mundaym): type for mask? */

/* DIVIDE */
SLJIT_S390X_RR(dr,   0x1d00)

/* EXCLUSIVE OR */
SLJIT_S390X_RR(xr,   0x1700)

/* LOAD */
SLJIT_S390X_RR(lr,   0x1800)

/* LOAD COMPLEMENT */
SLJIT_S390X_RR(lcr,  0x1300)

/* OR */
SLJIT_S390X_RR(or,   0x1600)

#undef SLJIT_S390X_RR

/* RRE form instructions */
#define SLJIT_S390X_RRE(name, pattern) \
SLJIT_S390X_INSTRUCTION(name, sljit_gpr dst, sljit_gpr src) \
{ \
	return (pattern) | R4A(dst) | R0A(src); \
}

/* AND */
SLJIT_S390X_RRE(ngr,   0xb9800000)

/* DIVIDE LOGICAL */
SLJIT_S390X_RRE(dlr,   0xb9970000)
SLJIT_S390X_RRE(dlgr,  0xb9870000)

/* DIVIDE SINGLE */
SLJIT_S390X_RRE(dsgr,  0xb90d0000)

/* EXCLUSIVE OR */
SLJIT_S390X_RRE(xgr,   0xb9820000)

/* LOAD */
SLJIT_S390X_RRE(lgr,   0xb9040000)
SLJIT_S390X_RRE(lgfr,  0xb9140000)

/* LOAD BYTE */
SLJIT_S390X_RRE(lbr,   0xb9260000)
SLJIT_S390X_RRE(lgbr,  0xb9060000)

/* LOAD COMPLEMENT */
SLJIT_S390X_RRE(lcgr,  0xb9030000)

/* LOAD HALFWORD */
SLJIT_S390X_RRE(lhr,   0xb9270000)
SLJIT_S390X_RRE(lghr,  0xb9070000)

/* LOAD LOGICAL */
SLJIT_S390X_RRE(llgfr, 0xb9160000)

/* LOAD LOGICAL CHARACTER */
SLJIT_S390X_RRE(llcr,  0xb9940000)
SLJIT_S390X_RRE(llgcr, 0xb9840000)

/* LOAD LOGICAL HALFWORD */
SLJIT_S390X_RRE(llhr,  0xb9950000)
SLJIT_S390X_RRE(llghr, 0xb9850000)

/* MULTIPLY LOGICAL */
SLJIT_S390X_RRE(mlgr,  0xb9860000)

/* MULTIPLY SINGLE */
SLJIT_S390X_RRE(msgfr, 0xb91c0000)

/* OR */
SLJIT_S390X_RRE(ogr,   0xb9810000)

/* SUBTRACT */
SLJIT_S390X_RRE(sgr,   0xb9090000)

#undef SLJIT_S390X_RRE

/* RI-a form instructions */
#define SLJIT_S390X_RIA(name, pattern, imm_type) \
SLJIT_S390X_INSTRUCTION(name, sljit_gpr reg, imm_type imm) \
{ \
	return (pattern) | R20A(reg) | (imm & 0xffff); \
}

/* ADD HALFWORD IMMEDIATE */
SLJIT_S390X_RIA(aghi,  0xa70b0000, sljit_s16)

/* LOAD HALFWORD IMMEDIATE */
SLJIT_S390X_RIA(lhi,   0xa7080000, sljit_s16)
SLJIT_S390X_RIA(lghi,  0xa7090000, sljit_s16)

/* LOAD LOGICAL IMMEDIATE */
SLJIT_S390X_RIA(llihh, 0xa50c0000, sljit_u16)
SLJIT_S390X_RIA(llihl, 0xa50d0000, sljit_u16)
SLJIT_S390X_RIA(llilh, 0xa50e0000, sljit_u16)
SLJIT_S390X_RIA(llill, 0xa50f0000, sljit_u16)

/* MULTIPLY HALFWORD IMMEDIATE */
SLJIT_S390X_RIA(mhi,   0xa70c0000, sljit_s16)
SLJIT_S390X_RIA(mghi,  0xa70d0000, sljit_s16)

/* OR IMMEDIATE */
SLJIT_S390X_RIA(oilh,  0xa50a0000, sljit_u16)

#undef SLJIT_S390X_RIA

/* RIL-a form instructions (requires extended immediate facility) */
#define SLJIT_S390X_RILA(name, pattern, imm_type) \
SLJIT_S390X_INSTRUCTION(name, sljit_gpr reg, imm_type imm) \
{ \
	SLJIT_ASSERT(have_eimm()); \
	return (pattern) | R36A(reg) | ((sljit_ins)imm & 0xffffffffu); \
}

/* ADD IMMEDIATE */
SLJIT_S390X_RILA(agfi,  0xc20800000000, sljit_s32)

/* ADD IMMEDIATE HIGH */
SLJIT_S390X_RILA(aih,   0xcc0800000000, sljit_s32) /* TODO(mundaym): high-word facility? */

/* AND IMMEDIATE */
SLJIT_S390X_RILA(nihf,  0xc00a00000000, sljit_u32)

/* EXCLUSIVE OR IMMEDIATE */
SLJIT_S390X_RILA(xilf,  0xc00700000000, sljit_u32)

/* INSERT IMMEDIATE */
SLJIT_S390X_RILA(iihf,  0xc00800000000, sljit_u32)
SLJIT_S390X_RILA(iilf,  0xc00900000000, sljit_u32)

/* LOAD IMMEDIATE */
SLJIT_S390X_RILA(lgfi,  0xc00100000000, sljit_s32)

/* LOAD LOGICAL IMMEDIATE */
SLJIT_S390X_RILA(llihf, 0xc00e00000000, sljit_u32)
SLJIT_S390X_RILA(llilf, 0xc00f00000000, sljit_u32)

/* SUBTRACT LOGICAL IMMEDIATE */
SLJIT_S390X_RILA(slfi,  0xc20500000000, sljit_u32)

#undef SLJIT_S390X_RILA

/* RX-a form instructions */
#define SLJIT_S390X_RXA(name, pattern) \
SLJIT_S390X_INSTRUCTION(name, sljit_gpr r, sljit_s32 d, sljit_gpr x, sljit_gpr b) \
{ \
	SLJIT_ASSERT((d & 0xfff) == d); \
\
	return (pattern) | R20A(r) | R16A(x) | R12A(b) | (sljit_ins)(d & 0xfff); \
}

/* LOAD */
SLJIT_S390X_RXA(l,   0x58000000)

/* LOAD ADDRESS */
SLJIT_S390X_RXA(la,  0x41000000)

/* LOAD HALFWORD */
SLJIT_S390X_RXA(lh,  0x48000000)

/* MULTIPLY SINGLE */
SLJIT_S390X_RXA(ms,  0x71000000)

/* STORE */
SLJIT_S390X_RXA(st,  0x50000000)

/* STORE CHARACTER */
SLJIT_S390X_RXA(stc, 0x42000000)

/* STORE HALFWORD */
SLJIT_S390X_RXA(sth, 0x40000000)

#undef SLJIT_S390X_RXA

/* RXY-a instructions */
#define SLJIT_S390X_RXYA(name, pattern, cond) \
SLJIT_S390X_INSTRUCTION(name, sljit_gpr r, sljit_s32 d, sljit_gpr x, sljit_gpr b) \
{ \
	SLJIT_ASSERT(cond); \
\
	return (pattern) | R36A(r) | R32A(x) | R28A(b) | disp_s20(d); \
}

/* LOAD */
SLJIT_S390X_RXYA(ly,    0xe30000000058, have_ldisp())
SLJIT_S390X_RXYA(lg,    0xe30000000004, 1)
SLJIT_S390X_RXYA(lgf,   0xe30000000014, 1)

/* LOAD BYTE */
SLJIT_S390X_RXYA(lb,    0xe30000000076, have_ldisp())
SLJIT_S390X_RXYA(lgb,   0xe30000000077, have_ldisp())

/* LOAD HALFWORD */
SLJIT_S390X_RXYA(lhy,   0xe30000000078, have_ldisp())
SLJIT_S390X_RXYA(lgh,   0xe30000000015, 1)

/* LOAD LOGICAL */
SLJIT_S390X_RXYA(llgf,  0xe30000000016, 1)

/* LOAD LOGICAL CHARACTER */
SLJIT_S390X_RXYA(llc,   0xe30000000094, have_eimm())
SLJIT_S390X_RXYA(llgc,  0xe30000000090, 1)

/* LOAD LOGICAL HALFWORD */
SLJIT_S390X_RXYA(llh,   0xe30000000095, have_eimm())
SLJIT_S390X_RXYA(llgh,  0xe30000000091, 1)

/* MULTIPLY SINGLE */
SLJIT_S390X_RXYA(msy,   0xe30000000051, have_ldisp())
SLJIT_S390X_RXYA(msg,   0xe3000000000c, 1)

/* STORE */
SLJIT_S390X_RXYA(sty,   0xe30000000050, have_ldisp())
SLJIT_S390X_RXYA(stg,   0xe30000000024, 1)

/* STORE CHARACTER */
SLJIT_S390X_RXYA(stcy,  0xe30000000072, have_ldisp())

/* STORE HALFWORD */
SLJIT_S390X_RXYA(sthy,  0xe30000000070, have_ldisp())

#undef SLJIT_S390X_RXYA

/* RSY-a instructions */
#define SLJIT_S390X_RSYA(name, pattern, cond) \
SLJIT_S390X_INSTRUCTION(name, sljit_gpr dst, sljit_gpr src, sljit_s32 d, sljit_gpr b) \
{ \
	SLJIT_ASSERT(cond); \
\
	return (pattern) | R36A(dst) | R32A(src) | R28A(b) | disp_s20(d); \
}

/* LOAD MULTIPLE */
SLJIT_S390X_RSYA(lmg,   0xeb0000000004, 1)

/* SHIFT LEFT LOGICAL */
SLJIT_S390X_RSYA(sllg,  0xeb000000000d, 1)

/* SHIFT RIGHT SINGLE */
SLJIT_S390X_RSYA(srag,  0xeb000000000a, 1)

/* STORE MULTIPLE */
SLJIT_S390X_RSYA(stmg,  0xeb0000000024, 1)

#undef SLJIT_S390X_RSYA

/* RIE-f instructions (require general-instructions-extension facility) */
#define SLJIT_S390X_RIEF(name, pattern) \
SLJIT_S390X_INSTRUCTION(name, sljit_gpr dst, sljit_gpr src, sljit_u8 start, sljit_u8 end, sljit_u8 rot) \
{ \
	sljit_ins i3, i4, i5; \
\
	SLJIT_ASSERT(have_genext()); \
	i3 = (sljit_ins)start << 24; \
	i4 = (sljit_ins)end << 16; \
	i5 = (sljit_ins)rot << 8; \
\
	return (pattern) | R36A(dst & 0xf) | R32A(src & 0xf) | i3 | i4 | i5; \
}

/* ROTATE THEN AND SELECTED BITS */
/* SLJIT_S390X_RIEF(rnsbg,  0xec0000000054) */

/* ROTATE THEN EXCLUSIVE OR SELECTED BITS */
/* SLJIT_S390X_RIEF(rxsbg,  0xec0000000057) */

/* ROTATE THEN OR SELECTED BITS */
SLJIT_S390X_RIEF(rosbg,  0xec0000000056)

/* ROTATE THEN INSERT SELECTED BITS */
/* SLJIT_S390X_RIEF(risbg,  0xec0000000055) */
/* SLJIT_S390X_RIEF(risbgn, 0xec0000000059) */

/* ROTATE THEN INSERT SELECTED BITS HIGH */
SLJIT_S390X_RIEF(risbhg, 0xec000000005d)

/* ROTATE THEN INSERT SELECTED BITS LOW */
/* SLJIT_S390X_RIEF(risblg, 0xec0000000051) */

#undef SLJIT_S390X_RIEF

/* RRF-c instructions (require load/store-on-condition 1 facility) */
#define SLJIT_S390X_RRFC(name, pattern) \
SLJIT_S390X_INSTRUCTION(name, sljit_gpr dst, sljit_gpr src, sljit_uw mask) \
{ \
	sljit_ins m3; \
\
	SLJIT_ASSERT(have_lscond1()); \
	m3 = (sljit_ins)(mask & 0xf) << 12; \
\
	return (pattern) | m3 | R4A(dst) | R0A(src); \
}

/* LOAD HALFWORD IMMEDIATE ON CONDITION */
SLJIT_S390X_RRFC(locr,  0xb9f20000)
SLJIT_S390X_RRFC(locgr, 0xb9e20000)

#undef SLJIT_S390X_RRFC

/* RIE-g instructions (require load/store-on-condition 2 facility) */
#define SLJIT_S390X_RIEG(name, pattern) \
SLJIT_S390X_INSTRUCTION(name, sljit_gpr reg, sljit_sw imm, sljit_uw mask) \
{ \
	sljit_ins m3, i2; \
\
	SLJIT_ASSERT(have_lscond2()); \
	m3 = (sljit_ins)(mask & 0xf) << 32; \
	i2 = (sljit_ins)(imm & 0xffffL) << 16; \
\
	return (pattern) | R36A(reg) | m3 | i2; \
}

/* LOAD HALFWORD IMMEDIATE ON CONDITION */
SLJIT_S390X_RIEG(lochi,  0xec0000000042)
SLJIT_S390X_RIEG(locghi, 0xec0000000046)

#undef SLJIT_S390X_RIEG

#define SLJIT_S390X_RILB(name, pattern, cond) \
SLJIT_S390X_INSTRUCTION(name, sljit_gpr reg, sljit_sw ri) \
{ \
	SLJIT_ASSERT(cond); \
\
	return (pattern) | R36A(reg) | (sljit_ins)(ri & 0xffffffff); \
}

/* BRANCH RELATIVE AND SAVE LONG */
SLJIT_S390X_RILB(brasl, 0xc00500000000, 1)

/* LOAD ADDRESS RELATIVE LONG */
SLJIT_S390X_RILB(larl,  0xc00000000000, 1)

/* LOAD RELATIVE LONG */
SLJIT_S390X_RILB(lgrl,  0xc40800000000, have_genext())

#undef SLJIT_S390X_RILB

SLJIT_S390X_INSTRUCTION(br, sljit_gpr target)
{
	return 0x07f0 | target;
}

SLJIT_S390X_INSTRUCTION(brc, sljit_uw mask, sljit_sw target)
{
	sljit_ins m1 = (sljit_ins)(mask & 0xf) << 20;
	sljit_ins ri2 = (sljit_ins)target & 0xffff;
	return 0xa7040000L | m1 | ri2;
}

SLJIT_S390X_INSTRUCTION(brcl, sljit_uw mask, sljit_sw target)
{
	sljit_ins m1 = (sljit_ins)(mask & 0xf) << 36;
	sljit_ins ri2 = (sljit_ins)target & 0xffffffff;
	return 0xc00400000000L | m1 | ri2;
}

SLJIT_S390X_INSTRUCTION(flogr, sljit_gpr dst, sljit_gpr src)
{
	SLJIT_ASSERT(have_eimm());
	return 0xb9830000 | R8A(dst) | R0A(src);
}

/* INSERT PROGRAM MASK */
SLJIT_S390X_INSTRUCTION(ipm, sljit_gpr dst)
{
	return 0xb2220000 | R4A(dst);
}

/* SET PROGRAM MASK */
SLJIT_S390X_INSTRUCTION(spm, sljit_gpr dst)
{
	return 0x0400 | R4A(dst);
}

/* ROTATE THEN INSERT SELECTED BITS HIGH (ZERO) */
SLJIT_S390X_INSTRUCTION(risbhgz, sljit_gpr dst, sljit_gpr src, sljit_u8 start, sljit_u8 end, sljit_u8 rot)
{
	return risbhg(dst, src, start, 0x8 | end, rot);
}

#undef SLJIT_S390X_INSTRUCTION

static sljit_s32 update_zero_overflow(struct sljit_compiler *compiler, sljit_s32 op, sljit_gpr dst_r)
{
	/* Condition codes: bits 18 and 19.
	   Transformation:
	     0 (zero and no overflow) : unchanged
	     1 (non-zero and no overflow) : unchanged
	     2 (zero and overflow) : decreased by 1
	     3 (non-zero and overflow) : decreased by 1 if non-zero */
	FAIL_IF(push_inst(compiler, brc(0xc, 2 + 2 + ((op & SLJIT_32) ? 1 : 2) + 2 + 3 + 1)));
	FAIL_IF(push_inst(compiler, ipm(tmp1)));
	FAIL_IF(push_inst(compiler, (op & SLJIT_32) ? or(dst_r, dst_r) : ogr(dst_r, dst_r)));
	FAIL_IF(push_inst(compiler, brc(0x8, 2 + 3)));
	FAIL_IF(push_inst(compiler, slfi(tmp1, 0x10000000)));
	FAIL_IF(push_inst(compiler, spm(tmp1)));
	return SLJIT_SUCCESS;
}

/* load 64-bit immediate into register without clobbering flags */
static sljit_s32 push_load_imm_inst(struct sljit_compiler *compiler, sljit_gpr target, sljit_sw v)
{
	/* 4 byte instructions */
	if (is_s16(v))
		return push_inst(compiler, lghi(target, (sljit_s16)v));

	if (((sljit_uw)v & ~(sljit_uw)0x000000000000ffff) == 0)
		return push_inst(compiler, llill(target, (sljit_u16)v));

	if (((sljit_uw)v & ~(sljit_uw)0x00000000ffff0000) == 0)
		return push_inst(compiler, llilh(target, (sljit_u16)(v >> 16)));

	if (((sljit_uw)v & ~(sljit_uw)0x0000ffff00000000) == 0)
		return push_inst(compiler, llihl(target, (sljit_u16)(v >> 32)));

	if (((sljit_uw)v & ~(sljit_uw)0xffff000000000000) == 0)
		return push_inst(compiler, llihh(target, (sljit_u16)(v >> 48)));

	/* 6 byte instructions (requires extended immediate facility) */
	if (have_eimm()) {
		if (is_s32(v))
			return push_inst(compiler, lgfi(target, (sljit_s32)v));

		if (((sljit_uw)v >> 32) == 0)
			return push_inst(compiler, llilf(target, (sljit_u32)v));

		if (((sljit_uw)v << 32) == 0)
			return push_inst(compiler, llihf(target, (sljit_u32)((sljit_uw)v >> 32)));

		FAIL_IF(push_inst(compiler, llilf(target, (sljit_u32)v)));
		return push_inst(compiler, iihf(target, (sljit_u32)(v >> 32)));
	}

	/* TODO(mundaym): instruction sequences that don't use extended immediates */
	abort();
}

struct addr {
	sljit_gpr base;
	sljit_gpr index;
	sljit_s32 offset;
};

/* transform memory operand into D(X,B) form with a signed 20-bit offset */
static sljit_s32 make_addr_bxy(struct sljit_compiler *compiler,
	struct addr *addr, sljit_s32 mem, sljit_sw off,
	sljit_gpr tmp /* clobbered, must not be r0 */)
{
	sljit_gpr base = r0;
	sljit_gpr index = r0;

	SLJIT_ASSERT(tmp != r0);
	if (mem & REG_MASK)
		base = gpr(mem & REG_MASK);

	if (mem & OFFS_REG_MASK) {
		index = gpr(OFFS_REG(mem));
		if (off != 0) {
			/* shift and put the result into tmp */
			SLJIT_ASSERT(0 <= off && off < 64);
			FAIL_IF(push_inst(compiler, sllg(tmp, index, (sljit_s32)off, 0)));
			index = tmp;
			off = 0; /* clear offset */
		}
	}
	else if (!is_s20(off)) {
		FAIL_IF(push_load_imm_inst(compiler, tmp, off));
		index = tmp;
		off = 0; /* clear offset */
	}
	addr->base = base;
	addr->index = index;
	addr->offset = (sljit_s32)off;
	return SLJIT_SUCCESS;
}

/* transform memory operand into D(X,B) form with an unsigned 12-bit offset */
static sljit_s32 make_addr_bx(struct sljit_compiler *compiler,
	struct addr *addr, sljit_s32 mem, sljit_sw off,
	sljit_gpr tmp /* clobbered, must not be r0 */)
{
	sljit_gpr base = r0;
	sljit_gpr index = r0;

	SLJIT_ASSERT(tmp != r0);
	if (mem & REG_MASK)
		base = gpr(mem & REG_MASK);

	if (mem & OFFS_REG_MASK) {
		index = gpr(OFFS_REG(mem));
		if (off != 0) {
			/* shift and put the result into tmp */
			SLJIT_ASSERT(0 <= off && off < 64);
			FAIL_IF(push_inst(compiler, sllg(tmp, index, (sljit_s32)off, 0)));
			index = tmp;
			off = 0; /* clear offset */
		}
	}
	else if (!is_u12(off)) {
		FAIL_IF(push_load_imm_inst(compiler, tmp, off));
		index = tmp;
		off = 0; /* clear offset */
	}
	addr->base = base;
	addr->index = index;
	addr->offset = (sljit_s32)off;
	return SLJIT_SUCCESS;
}

#define EVAL(op, r, addr) op(r, addr.offset, addr.index, addr.base)
#define WHEN(cond, r, i1, i2, addr) \
	(cond) ? EVAL(i1, r, addr) : EVAL(i2, r, addr)

/* May clobber tmp1. */
static sljit_s32 load_word(struct sljit_compiler *compiler, sljit_gpr dst,
		sljit_s32 src, sljit_sw srcw,
		sljit_s32 is_32bit)
{
	struct addr addr;
	sljit_ins ins;

	SLJIT_ASSERT(src & SLJIT_MEM);
	if (have_ldisp() || !is_32bit)
		FAIL_IF(make_addr_bxy(compiler, &addr, src, srcw, tmp1));
	else
		FAIL_IF(make_addr_bx(compiler, &addr, src, srcw, tmp1));

	if (is_32bit)
		ins = WHEN(is_u12(addr.offset), dst, l, ly, addr);
	else
		ins = lg(dst, addr.offset, addr.index, addr.base);

	return push_inst(compiler, ins);
}

/* May clobber tmp1. */
static sljit_s32 store_word(struct sljit_compiler *compiler, sljit_gpr src,
		sljit_s32 dst, sljit_sw dstw,
		sljit_s32 is_32bit)
{
	struct addr addr;
	sljit_ins ins;

	SLJIT_ASSERT(dst & SLJIT_MEM);
	if (have_ldisp() || !is_32bit)
		FAIL_IF(make_addr_bxy(compiler, &addr, dst, dstw, tmp1));
	else
		FAIL_IF(make_addr_bx(compiler, &addr, dst, dstw, tmp1));

	if (is_32bit)
		ins = WHEN(is_u12(addr.offset), src, st, sty, addr);
	else
		ins = stg(src, addr.offset, addr.index, addr.base);

	return push_inst(compiler, ins);
}

#undef WHEN

static sljit_s32 emit_move(struct sljit_compiler *compiler,
	sljit_gpr dst_r,
	sljit_s32 src, sljit_sw srcw)
{
	SLJIT_ASSERT(!IS_GPR_REG(src) || dst_r != gpr(src & REG_MASK));

	if (src & SLJIT_IMM)
		return push_load_imm_inst(compiler, dst_r, srcw);

	if (src & SLJIT_MEM)
		return load_word(compiler, dst_r, src, srcw, (compiler->mode & SLJIT_32) != 0);

	sljit_gpr src_r = gpr(src & REG_MASK);
	return push_inst(compiler, (compiler->mode & SLJIT_32) ? lr(dst_r, src_r) : lgr(dst_r, src_r));
}

static sljit_s32 emit_rr(struct sljit_compiler *compiler, sljit_ins ins,
	sljit_s32 dst,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_gpr dst_r = tmp0;
	sljit_gpr src_r = tmp1;
	sljit_s32 needs_move = 1;

	if (FAST_IS_REG(dst)) {
		dst_r = gpr(dst);

		if (dst == src1)
			needs_move = 0;
		else if (dst == src2) {
			dst_r = tmp0;
			needs_move = 2;
		}
	}

	if (needs_move)
		FAIL_IF(emit_move(compiler, dst_r, src1, src1w));

	if (FAST_IS_REG(src2))
		src_r = gpr(src2);
	else
		FAIL_IF(emit_move(compiler, tmp1, src2, src2w));

	FAIL_IF(push_inst(compiler, ins | R4A(dst_r) | R0A(src_r)));

	if (needs_move != 2)
		return SLJIT_SUCCESS;

	dst_r = gpr(dst & REG_MASK);
	return push_inst(compiler, (compiler->mode & SLJIT_32) ? lr(dst_r, tmp0) : lgr(dst_r, tmp0));
}

static sljit_s32 emit_rr1(struct sljit_compiler *compiler, sljit_ins ins,
	sljit_s32 dst,
	sljit_s32 src1, sljit_sw src1w)
{
	sljit_gpr dst_r = FAST_IS_REG(dst) ? gpr(dst) : tmp0;
	sljit_gpr src_r = tmp1;

	if (FAST_IS_REG(src1))
		src_r = gpr(src1);
	else
		FAIL_IF(emit_move(compiler, tmp1, src1, src1w));

	return push_inst(compiler, ins | R4A(dst_r) | R0A(src_r));
}

static sljit_s32 emit_rrf(struct sljit_compiler *compiler, sljit_ins ins,
	sljit_s32 dst,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_gpr dst_r = FAST_IS_REG(dst) ? gpr(dst & REG_MASK) : tmp0;
	sljit_gpr src1_r = tmp0;
	sljit_gpr src2_r = tmp1;

	if (FAST_IS_REG(src1))
		src1_r = gpr(src1);
	else
		FAIL_IF(emit_move(compiler, tmp0, src1, src1w));

	if (FAST_IS_REG(src2))
		src2_r = gpr(src2);
	else
		FAIL_IF(emit_move(compiler, tmp1, src2, src2w));

	return push_inst(compiler, ins | R4A(dst_r) | R0A(src1_r) | R12A(src2_r));
}

typedef enum {
	RI_A,
	RIL_A,
} emit_ril_type;

static sljit_s32 emit_ri(struct sljit_compiler *compiler, sljit_ins ins,
	sljit_s32 dst,
	sljit_s32 src1, sljit_sw src1w,
	sljit_sw src2w,
	emit_ril_type type)
{
	sljit_gpr dst_r = tmp0;
	sljit_s32 needs_move = 1;

	if (FAST_IS_REG(dst)) {
		dst_r = gpr(dst);

		if (dst == src1)
			needs_move = 0;
	}

	if (needs_move)
		FAIL_IF(emit_move(compiler, dst_r, src1, src1w));

	if (type == RIL_A)
		return push_inst(compiler, ins | R36A(dst_r) | (src2w & 0xffffffff));
	return push_inst(compiler, ins | R20A(dst_r) | (src2w & 0xffff));
}

static sljit_s32 emit_rie_d(struct sljit_compiler *compiler, sljit_ins ins,
	sljit_s32 dst,
	sljit_s32 src1, sljit_sw src1w,
	sljit_sw src2w)
{
	sljit_gpr dst_r = FAST_IS_REG(dst) ? gpr(dst) : tmp0;
	sljit_gpr src_r = tmp0;

	if (!FAST_IS_REG(src1))
		FAIL_IF(emit_move(compiler, tmp0, src1, src1w));
	else
		src_r = gpr(src1 & REG_MASK);

	return push_inst(compiler, ins | R36A(dst_r) | R32A(src_r) | (sljit_ins)(src2w & 0xffff) << 16);
}

typedef enum {
	RX_A,
	RXY_A,
} emit_rx_type;

static sljit_s32 emit_rx(struct sljit_compiler *compiler, sljit_ins ins,
	sljit_s32 dst,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w,
	emit_rx_type type)
{
	sljit_gpr dst_r = tmp0;
	sljit_s32 needs_move = 1;
	sljit_gpr base, index;

	SLJIT_ASSERT(src2 & SLJIT_MEM);

	if (FAST_IS_REG(dst)) {
		dst_r = gpr(dst);

		if (dst == src1)
			needs_move = 0;
		else if (dst == (src2 & REG_MASK) || (dst == OFFS_REG(src2))) {
			dst_r = tmp0;
			needs_move = 2;
		}
	}

	if (needs_move)
		FAIL_IF(emit_move(compiler, dst_r, src1, src1w));

	base = gpr(src2 & REG_MASK);
	index = tmp0;

	if (src2 & OFFS_REG_MASK) {
		index = gpr(OFFS_REG(src2));

		if (src2w != 0) {
			FAIL_IF(push_inst(compiler, sllg(tmp1, index, src2w & 0x3, 0)));
			src2w = 0;
			index = tmp1;
		}
	} else if ((type == RX_A && !is_u12(src2w)) || (type == RXY_A && !is_s20(src2w))) {
		FAIL_IF(push_load_imm_inst(compiler, tmp1, src2w));

		if (src2 & REG_MASK)
			index = tmp1;
		else
			base = tmp1;
		src2w = 0;
	}

	if (type == RX_A)
		ins |= R20A(dst_r) | R16A(index) | R12A(base) | (sljit_ins)src2w;
	else
		ins |= R36A(dst_r) | R32A(index) | R28A(base) | disp_s20((sljit_s32)src2w);

	FAIL_IF(push_inst(compiler, ins));

	if (needs_move != 2)
		return SLJIT_SUCCESS;

	dst_r = gpr(dst);
	return push_inst(compiler, (compiler->mode & SLJIT_32) ? lr(dst_r, tmp0) : lgr(dst_r, tmp0));
}

static sljit_s32 emit_siy(struct sljit_compiler *compiler, sljit_ins ins,
	sljit_s32 dst, sljit_sw dstw,
	sljit_sw srcw)
{
	SLJIT_ASSERT(dst & SLJIT_MEM);

	sljit_gpr dst_r = tmp1;

	if (dst & OFFS_REG_MASK) {
		sljit_gpr index = tmp1;

		if ((dstw & 0x3) == 0)
			index = gpr(OFFS_REG(dst));
		else
			FAIL_IF(push_inst(compiler, sllg(tmp1, index, dstw & 0x3, 0)));

		FAIL_IF(push_inst(compiler, la(tmp1, 0, dst_r, index)));
		dstw = 0;
	}
	else if (!is_s20(dstw)) {
		FAIL_IF(push_load_imm_inst(compiler, tmp1, dstw));

		if (dst & REG_MASK)
			FAIL_IF(push_inst(compiler, la(tmp1, 0, dst_r, tmp1)));

		dstw = 0;
	}
	else
		dst_r = gpr(dst & REG_MASK);

	return push_inst(compiler, ins | ((sljit_ins)(srcw & 0xff) << 32) | R28A(dst_r) | disp_s20((sljit_s32)dstw));
}

struct ins_forms {
	sljit_ins op_r;
	sljit_ins op_gr;
	sljit_ins op_rk;
	sljit_ins op_grk;
	sljit_ins op;
	sljit_ins op_y;
	sljit_ins op_g;
};

static sljit_s32 emit_commutative(struct sljit_compiler *compiler, const struct ins_forms *forms,
	sljit_s32 dst,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_s32 mode = compiler->mode;
	sljit_ins ins, ins_k;

	if ((src1 | src2) & SLJIT_MEM) {
		sljit_ins ins12, ins20;

		if (mode & SLJIT_32) {
			ins12 = forms->op;
			ins20 = forms->op_y;
		}
		else {
			ins12 = 0;
			ins20 = forms->op_g;
		}

		if (ins12 && ins20) {
			/* Extra instructions needed for address computation can be executed independently. */
			if ((src2 & SLJIT_MEM) && (!(src1 & SLJIT_MEM)
					|| ((src1 & OFFS_REG_MASK) ? (src1w & 0x3) == 0 : is_s20(src1w)))) {
				if ((src2 & OFFS_REG_MASK) || is_u12(src2w) || !is_s20(src2w))
					return emit_rx(compiler, ins12, dst, src1, src1w, src2, src2w, RX_A);

				return emit_rx(compiler, ins20, dst, src1, src1w, src2, src2w, RXY_A);
			}

			if (src1 & SLJIT_MEM) {
				if ((src1 & OFFS_REG_MASK) || is_u12(src1w) || !is_s20(src1w))
					return emit_rx(compiler, ins12, dst, src2, src2w, src1, src1w, RX_A);

				return emit_rx(compiler, ins20, dst, src2, src2w, src1, src1w, RXY_A);
			}
		}
		else if (ins12 || ins20) {
			emit_rx_type rx_type;

			if (ins12) {
				rx_type = RX_A;
				ins = ins12;
			}
			else {
				rx_type = RXY_A;
				ins = ins20;
			}

			if ((src2 & SLJIT_MEM) && (!(src1 & SLJIT_MEM)
					|| ((src1 & OFFS_REG_MASK) ? (src1w & 0x3) == 0 : (rx_type == RX_A ? is_u12(src1w) : is_s20(src1w)))))
				return emit_rx(compiler, ins, dst, src1, src1w, src2, src2w, rx_type);

			if (src1 & SLJIT_MEM)
				return emit_rx(compiler, ins, dst, src2, src2w, src1, src1w, rx_type);
		}
	}

	if (mode & SLJIT_32) {
		ins = forms->op_r;
		ins_k = forms->op_rk;
	}
	else {
		ins = forms->op_gr;
		ins_k = forms->op_grk;
	}

	SLJIT_ASSERT(ins != 0 || ins_k != 0);

	if (ins && FAST_IS_REG(dst)) {
		if (dst == src1)
			return emit_rr(compiler, ins, dst, src1, src1w, src2, src2w);

		if (dst == src2)
			return emit_rr(compiler, ins, dst, src2, src2w, src1, src1w);
	}

	if (ins_k == 0)
		return emit_rr(compiler, ins, dst, src1, src1w, src2, src2w);

	return emit_rrf(compiler, ins_k, dst, src1, src1w, src2, src2w);
}

static sljit_s32 emit_non_commutative(struct sljit_compiler *compiler, const struct ins_forms *forms,
	sljit_s32 dst,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_s32 mode = compiler->mode;
	sljit_ins ins;

	if (src2 & SLJIT_MEM) {
		sljit_ins ins12, ins20;

		if (mode & SLJIT_32) {
			ins12 = forms->op;
			ins20 = forms->op_y;
		}
		else {
			ins12 = 0;
			ins20 = forms->op_g;
		}

		if (ins12 && ins20) {
			if ((src2 & OFFS_REG_MASK) || is_u12(src2w) || !is_s20(src2w))
				return emit_rx(compiler, ins12, dst, src1, src1w, src2, src2w, RX_A);

			return emit_rx(compiler, ins20, dst, src1, src1w, src2, src2w, RXY_A);
		}
		else if (ins12)
			return emit_rx(compiler, ins12, dst, src1, src1w, src2, src2w, RX_A);
		else if (ins20)
			return emit_rx(compiler, ins20, dst, src1, src1w, src2, src2w, RXY_A);
	}

	ins = (mode & SLJIT_32) ? forms->op_rk : forms->op_grk;

	if (ins == 0 || (FAST_IS_REG(dst) && dst == src1))
		return emit_rr(compiler, (mode & SLJIT_32) ? forms->op_r : forms->op_gr, dst, src1, src1w, src2, src2w);

	return emit_rrf(compiler, ins, dst, src1, src1w, src2, src2w);
}

SLJIT_API_FUNC_ATTRIBUTE void* sljit_generate_code(struct sljit_compiler *compiler)
{
	struct sljit_label *label;
	struct sljit_jump *jump;
	struct sljit_s390x_const *const_;
	struct sljit_put_label *put_label;
	sljit_sw executable_offset;
	sljit_uw ins_size = 0; /* instructions */
	sljit_uw pool_size = 0; /* literal pool */
	sljit_uw pad_size;
	sljit_uw i, j = 0;
	struct sljit_memory_fragment *buf;
	void *code, *code_ptr;
	sljit_uw *pool, *pool_ptr;
	sljit_sw source, offset; /* TODO(carenas): only need 32 bit */

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_generate_code(compiler));
	reverse_buf(compiler);

	/* branch handling */
	label = compiler->labels;
	jump = compiler->jumps;
	put_label = compiler->put_labels;

	/* TODO(carenas): compiler->executable_size could be calculated
         *                before to avoid the following loop (except for
         *                pool_size)
         */
	/* calculate the size of the code */
	for (buf = compiler->buf; buf != NULL; buf = buf->next) {
		sljit_uw len = buf->used_size / sizeof(sljit_ins);
		sljit_ins *ibuf = (sljit_ins *)buf->memory;
		for (i = 0; i < len; ++i, ++j) {
			sljit_ins ins = ibuf[i];

			/* TODO(carenas): instruction tag vs size/addr == j
			 * using instruction tags for const is creative
			 * but unlike all other architectures, and is not
			 * done consistently for all other objects.
			 * This might need reviewing later.
			 */
			if (ins & sljit_ins_const) {
				pool_size += sizeof(*pool);
				ins &= ~sljit_ins_const;
			}
			if (label && label->size == j) {
				label->size = ins_size;
				label = label->next;
			}
			if (jump && jump->addr == j) {
				if ((jump->flags & SLJIT_REWRITABLE_JUMP) || (jump->flags & JUMP_ADDR)) {
					/* encoded: */
					/*   brasl %r14, <rel_addr> (or brcl <mask>, <rel_addr>) */
					/* replace with: */
					/*   lgrl %r1, <pool_addr> */
					/*   bras %r14, %r1 (or bcr <mask>, %r1) */
					pool_size += sizeof(*pool);
					ins_size += 2;
				}
				jump = jump->next;
			}
			if (put_label && put_label->addr == j) {
				pool_size += sizeof(*pool);
				put_label = put_label->next;
			}
			ins_size += sizeof_ins(ins);
		}
	}

	/* emit trailing label */
	if (label && label->size == j) {
		label->size = ins_size;
		label = label->next;
	}

	SLJIT_ASSERT(!label);
	SLJIT_ASSERT(!jump);
	SLJIT_ASSERT(!put_label);

	/* pad code size to 8 bytes so is accessible with half word offsets */
	/* the literal pool needs to be doubleword aligned */
	pad_size = ((ins_size + 7UL) & ~7UL) - ins_size;
	SLJIT_ASSERT(pad_size < 8UL);

	/* allocate target buffer */
	code = SLJIT_MALLOC_EXEC(ins_size + pad_size + pool_size,
					compiler->exec_allocator_data);
	PTR_FAIL_WITH_EXEC_IF(code);
	code_ptr = code;
	executable_offset = SLJIT_EXEC_OFFSET(code);

	/* TODO(carenas): pool is optional, and the ABI recommends it to
         *                be created before the function code, instead of
         *                globally; if generated code is too big could
         *                need offsets bigger than 32bit words and asser()
         */
	pool = (sljit_uw *)((sljit_uw)code + ins_size + pad_size);
	pool_ptr = pool;
	const_ = (struct sljit_s390x_const *)compiler->consts;

	/* update label addresses */
	label = compiler->labels;
	while (label) {
		label->addr = (sljit_uw)SLJIT_ADD_EXEC_OFFSET(
			(sljit_uw)code_ptr + label->size, executable_offset);
		label = label->next;
	}

	/* reset jumps */
	jump = compiler->jumps;
	put_label = compiler->put_labels;

	/* emit the code */
	j = 0;
	for (buf = compiler->buf; buf != NULL; buf = buf->next) {
		sljit_uw len = buf->used_size / sizeof(sljit_ins);
		sljit_ins *ibuf = (sljit_ins *)buf->memory;
		for (i = 0; i < len; ++i, ++j) {
			sljit_ins ins = ibuf[i];
			if (ins & sljit_ins_const) {
				/* clear the const tag */
				ins &= ~sljit_ins_const;

				/* update instruction with relative address of constant */
				source = (sljit_sw)code_ptr;
				offset = (sljit_sw)pool_ptr - source;

				SLJIT_ASSERT(!(offset & 1));
				offset >>= 1; /* halfword (not byte) offset */
				SLJIT_ASSERT(is_s32(offset));

				ins |= (sljit_ins)offset & 0xffffffff;

				/* update address */
				const_->const_.addr = (sljit_uw)pool_ptr;

				/* store initial value into pool and update pool address */
				*(pool_ptr++) = (sljit_uw)const_->init_value;

				/* move to next constant */
				const_ = (struct sljit_s390x_const *)const_->const_.next;
			}
			if (jump && jump->addr == j) {
				sljit_sw target = (sljit_sw)((jump->flags & JUMP_LABEL) ? jump->u.label->addr : jump->u.target);
				if ((jump->flags & SLJIT_REWRITABLE_JUMP) || (jump->flags & JUMP_ADDR)) {
					jump->addr = (sljit_uw)pool_ptr;

					/* load address into tmp1 */
					source = (sljit_sw)SLJIT_ADD_EXEC_OFFSET(code_ptr, executable_offset);
					offset = (sljit_sw)SLJIT_ADD_EXEC_OFFSET(pool_ptr, executable_offset) - source;

					SLJIT_ASSERT(!(offset & 1));
					offset >>= 1;
					SLJIT_ASSERT(is_s32(offset));

					encode_inst(&code_ptr, lgrl(tmp1, offset & 0xffffffff));

					/* store jump target into pool and update pool address */
					*(pool_ptr++) = (sljit_uw)target;

					/* branch to tmp1 */
					sljit_ins op = (ins >> 32) & 0xf;
					sljit_ins arg = (ins >> 36) & 0xf;
					switch (op) {
					case 4: /* brcl -> bcr */
						ins = bcr(arg, tmp1);
						break;
					case 5: /* brasl -> basr */
						ins = basr(arg, tmp1);
						break;
					default:
						abort();
					}
				}
				else {
					jump->addr = (sljit_uw)code_ptr + 2;
					source = (sljit_sw)SLJIT_ADD_EXEC_OFFSET(code_ptr, executable_offset);
					offset = target - source;

					/* offset must be halfword aligned */
					SLJIT_ASSERT(!(offset & 1));
					offset >>= 1;
					SLJIT_ASSERT(is_s32(offset)); /* TODO(mundaym): handle arbitrary offsets */

					/* patch jump target */
					ins |= (sljit_ins)offset & 0xffffffff;
				}
				jump = jump->next;
			}
			if (put_label && put_label->addr == j) {
				source = (sljit_sw)SLJIT_ADD_EXEC_OFFSET(code_ptr, executable_offset);

				SLJIT_ASSERT(put_label->label);
				put_label->addr = (sljit_uw)code_ptr;

				/* store target into pool */
				*pool_ptr = put_label->label->addr;
				offset = (sljit_sw)SLJIT_ADD_EXEC_OFFSET(pool_ptr, executable_offset) - source;
				pool_ptr++;

				SLJIT_ASSERT(!(offset & 1));
				offset >>= 1;
				SLJIT_ASSERT(is_s32(offset));
				ins |= (sljit_ins)offset & 0xffffffff;

				put_label = put_label->next;
			}
			encode_inst(&code_ptr, ins);
		}
	}
	SLJIT_ASSERT((sljit_u8 *)code + ins_size == code_ptr);
	SLJIT_ASSERT((sljit_u8 *)pool + pool_size == (sljit_u8 *)pool_ptr);

	compiler->error = SLJIT_ERR_COMPILED;
	compiler->executable_offset = executable_offset;
	compiler->executable_size = ins_size;
	code = SLJIT_ADD_EXEC_OFFSET(code, executable_offset);
	code_ptr = SLJIT_ADD_EXEC_OFFSET(code_ptr, executable_offset);
	SLJIT_CACHE_FLUSH(code, code_ptr);
	SLJIT_UPDATE_WX_FLAGS(code, code_ptr, 1);
	return code;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_has_cpu_feature(sljit_s32 feature_type)
{
	/* TODO(mundaym): implement all */
	switch (feature_type) {
	case SLJIT_HAS_CLZ:
		return have_eimm() ? 1 : 0; /* FLOGR instruction */
	case SLJIT_HAS_CMOV:
		return have_lscond1() ? 1 : 0;
	case SLJIT_HAS_FPU:
		return 1;
	}
	return 0;
}

/* --------------------------------------------------------------------- */
/*  Entry, exit                                                          */
/* --------------------------------------------------------------------- */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_enter(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 arg_types, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size)
{
	sljit_s32 word_arg_count = 0;
	sljit_s32 offset, i, tmp;

	CHECK_ERROR();
	CHECK(check_sljit_emit_enter(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size));
	set_emit_enter(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size);

	/* Saved registers are stored in callee allocated save area. */
	SLJIT_ASSERT(gpr(SLJIT_FIRST_SAVED_REG) == r6 && gpr(SLJIT_S0) == r13);

	offset = 2 * SSIZE_OF(sw);
	if (saveds + scratches >= SLJIT_NUMBER_OF_REGISTERS) {
		FAIL_IF(push_inst(compiler, stmg(r6, r14, offset, r15))); /* save registers TODO(MGM): optimize */
		offset += 9 * SSIZE_OF(sw);
	} else {
		if (scratches == SLJIT_FIRST_SAVED_REG) {
			FAIL_IF(push_inst(compiler, stg(r6, offset, 0, r15)));
			offset += SSIZE_OF(sw);
		} else if (scratches > SLJIT_FIRST_SAVED_REG) {
			FAIL_IF(push_inst(compiler, stmg(r6, r6 + (sljit_gpr)(scratches - SLJIT_FIRST_SAVED_REG), offset, r15)));
			offset += (scratches - (SLJIT_FIRST_SAVED_REG - 1)) * SSIZE_OF(sw);
		}

		if (saveds == 0) {
			FAIL_IF(push_inst(compiler, stg(r14, offset, 0, r15)));
			offset += SSIZE_OF(sw);
		} else {
			FAIL_IF(push_inst(compiler, stmg(r14 - (sljit_gpr)saveds, r14, offset, r15)));
			offset += (saveds + 1) * SSIZE_OF(sw);
		}
	}

	tmp = SLJIT_FS0 - fsaveds;
	for (i = SLJIT_FS0; i > tmp; i--) {
		FAIL_IF(push_inst(compiler, 0x60000000 /* std */ | F20(i) | R12A(r15) | (sljit_ins)offset));
		offset += SSIZE_OF(sw);
	}

	for (i = fscratches; i >= SLJIT_FIRST_SAVED_FLOAT_REG; i--) {
		FAIL_IF(push_inst(compiler, 0x60000000 /* std */ | F20(i) | R12A(r15) | (sljit_ins)offset));
		offset += SSIZE_OF(sw);
	}

	local_size = (local_size + SLJIT_S390X_DEFAULT_STACK_FRAME_SIZE + 0xf) & ~0xf;
	compiler->local_size = local_size;

	FAIL_IF(push_inst(compiler, 0xe30000000071 /* lay */ | R36A(r15) | R28A(r15) | disp_s20(-local_size)));

	arg_types >>= SLJIT_ARG_SHIFT;
	tmp = 0;
	while (arg_types > 0) {
		if ((arg_types & SLJIT_ARG_MASK) < SLJIT_ARG_TYPE_F64) {
			if (!(arg_types & SLJIT_ARG_TYPE_SCRATCH_REG)) {
				FAIL_IF(push_inst(compiler, lgr(gpr(SLJIT_S0 - tmp), gpr(SLJIT_R0 + word_arg_count))));
				tmp++;
			}
			word_arg_count++;
		}

		arg_types >>= SLJIT_ARG_SHIFT;
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_set_context(struct sljit_compiler *compiler,
	sljit_s32 options, sljit_s32 arg_types, sljit_s32 scratches, sljit_s32 saveds,
	sljit_s32 fscratches, sljit_s32 fsaveds, sljit_s32 local_size)
{
	CHECK_ERROR();
	CHECK(check_sljit_set_context(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size));
	set_set_context(compiler, options, arg_types, scratches, saveds, fscratches, fsaveds, local_size);

	compiler->local_size = (local_size + SLJIT_S390X_DEFAULT_STACK_FRAME_SIZE + 0xf) & ~0xf;
	return SLJIT_SUCCESS;
}

static sljit_s32 emit_stack_frame_release(struct sljit_compiler *compiler)
{
	sljit_s32 offset, i, tmp;
	sljit_s32 local_size = compiler->local_size;
	sljit_s32 saveds = compiler->saveds;
	sljit_s32 scratches = compiler->scratches;

	if (is_u12(local_size))
		FAIL_IF(push_inst(compiler, 0x41000000 /* ly */ | R20A(r15) | R12A(r15) | (sljit_ins)local_size));
	else
		FAIL_IF(push_inst(compiler, 0xe30000000071 /* lay */ | R36A(r15) | R28A(r15) | disp_s20(local_size)));

	offset = 2 * SSIZE_OF(sw);
	if (saveds + scratches >= SLJIT_NUMBER_OF_REGISTERS) {
		FAIL_IF(push_inst(compiler, lmg(r6, r14, offset, r15))); /* save registers TODO(MGM): optimize */
		offset += 9 * SSIZE_OF(sw);
	} else {
		if (scratches == SLJIT_FIRST_SAVED_REG) {
			FAIL_IF(push_inst(compiler, lg(r6, offset, 0, r15)));
			offset += SSIZE_OF(sw);
		} else if (scratches > SLJIT_FIRST_SAVED_REG) {
			FAIL_IF(push_inst(compiler, lmg(r6, r6 + (sljit_gpr)(scratches - SLJIT_FIRST_SAVED_REG), offset, r15)));
			offset += (scratches - (SLJIT_FIRST_SAVED_REG - 1)) * SSIZE_OF(sw);
		}

		if (saveds == 0) {
			FAIL_IF(push_inst(compiler, lg(r14, offset, 0, r15)));
			offset += SSIZE_OF(sw);
		} else {
			FAIL_IF(push_inst(compiler, lmg(r14 - (sljit_gpr)saveds, r14, offset, r15)));
			offset += (saveds + 1) * SSIZE_OF(sw);
		}
	}

	tmp = SLJIT_FS0 - compiler->fsaveds;
	for (i = SLJIT_FS0; i > tmp; i--) {
		FAIL_IF(push_inst(compiler, 0x68000000 /* ld */ | F20(i) | R12A(r15) | (sljit_ins)offset));
		offset += SSIZE_OF(sw);
	}

	for (i = compiler->fscratches; i >= SLJIT_FIRST_SAVED_FLOAT_REG; i--) {
		FAIL_IF(push_inst(compiler, 0x68000000 /* ld */ | F20(i) | R12A(r15) | (sljit_ins)offset));
		offset += SSIZE_OF(sw);
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_return_void(struct sljit_compiler *compiler)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_return_void(compiler));

	FAIL_IF(emit_stack_frame_release(compiler));
	return push_inst(compiler, br(r14)); /* return */
}

/* --------------------------------------------------------------------- */
/*  Operators                                                            */
/* --------------------------------------------------------------------- */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op0(struct sljit_compiler *compiler, sljit_s32 op)
{
	sljit_gpr arg0 = gpr(SLJIT_R0);
	sljit_gpr arg1 = gpr(SLJIT_R1);

	CHECK_ERROR();
	CHECK(check_sljit_emit_op0(compiler, op));

	op = GET_OPCODE(op) | (op & SLJIT_32);
	switch (op) {
	case SLJIT_BREAKPOINT:
		/* The following invalid instruction is emitted by gdb. */
		return push_inst(compiler, 0x0001 /* 2-byte trap */);
	case SLJIT_NOP:
		return push_inst(compiler, 0x0700 /* 2-byte nop */);
	case SLJIT_LMUL_UW:
		FAIL_IF(push_inst(compiler, mlgr(arg0, arg0)));
		break;
	case SLJIT_LMUL_SW:
		/* signed multiplication from: */
		/* Hacker's Delight, Second Edition: Chapter 8-3. */
		FAIL_IF(push_inst(compiler, srag(tmp0, arg0, 63, 0)));
		FAIL_IF(push_inst(compiler, srag(tmp1, arg1, 63, 0)));
		FAIL_IF(push_inst(compiler, ngr(tmp0, arg1)));
		FAIL_IF(push_inst(compiler, ngr(tmp1, arg0)));

		/* unsigned multiplication */
		FAIL_IF(push_inst(compiler, mlgr(arg0, arg0)));

		FAIL_IF(push_inst(compiler, sgr(arg0, tmp0)));
		FAIL_IF(push_inst(compiler, sgr(arg0, tmp1)));
		break;
	case SLJIT_DIV_U32:
	case SLJIT_DIVMOD_U32:
		FAIL_IF(push_inst(compiler, lhi(tmp0, 0)));
		FAIL_IF(push_inst(compiler, lr(tmp1, arg0)));
		FAIL_IF(push_inst(compiler, dlr(tmp0, arg1)));
		FAIL_IF(push_inst(compiler, lr(arg0, tmp1))); /* quotient */
		if (op == SLJIT_DIVMOD_U32)
			return push_inst(compiler, lr(arg1, tmp0)); /* remainder */

		return SLJIT_SUCCESS;
	case SLJIT_DIV_S32:
	case SLJIT_DIVMOD_S32:
		FAIL_IF(push_inst(compiler, lhi(tmp0, 0)));
		FAIL_IF(push_inst(compiler, lr(tmp1, arg0)));
		FAIL_IF(push_inst(compiler, dr(tmp0, arg1)));
		FAIL_IF(push_inst(compiler, lr(arg0, tmp1))); /* quotient */
		if (op == SLJIT_DIVMOD_S32)
			return push_inst(compiler, lr(arg1, tmp0)); /* remainder */

		return SLJIT_SUCCESS;
	case SLJIT_DIV_UW:
	case SLJIT_DIVMOD_UW:
		FAIL_IF(push_inst(compiler, lghi(tmp0, 0)));
		FAIL_IF(push_inst(compiler, lgr(tmp1, arg0)));
		FAIL_IF(push_inst(compiler, dlgr(tmp0, arg1)));
		FAIL_IF(push_inst(compiler, lgr(arg0, tmp1))); /* quotient */
		if (op == SLJIT_DIVMOD_UW)
			return push_inst(compiler, lgr(arg1, tmp0)); /* remainder */

		return SLJIT_SUCCESS;
	case SLJIT_DIV_SW:
	case SLJIT_DIVMOD_SW:
		FAIL_IF(push_inst(compiler, lgr(tmp1, arg0)));
		FAIL_IF(push_inst(compiler, dsgr(tmp0, arg1)));
		FAIL_IF(push_inst(compiler, lgr(arg0, tmp1))); /* quotient */
		if (op == SLJIT_DIVMOD_SW)
			return push_inst(compiler, lgr(arg1, tmp0)); /* remainder */

		return SLJIT_SUCCESS;
	case SLJIT_ENDBR:
		return SLJIT_SUCCESS;
	case SLJIT_SKIP_FRAMES_BEFORE_RETURN:
		return SLJIT_SUCCESS;
	default:
		SLJIT_UNREACHABLE();
	}
	/* swap result registers */
	FAIL_IF(push_inst(compiler, lgr(tmp0, arg0)));
	FAIL_IF(push_inst(compiler, lgr(arg0, arg1)));
	return push_inst(compiler, lgr(arg1, tmp0));
}

/* LEVAL will be defined later with different parameters as needed */
#define WHEN2(cond, i1, i2) (cond) ? LEVAL(i1) : LEVAL(i2)

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op1(struct sljit_compiler *compiler, sljit_s32 op,
        sljit_s32 dst, sljit_sw dstw,
        sljit_s32 src, sljit_sw srcw)
{
	sljit_ins ins;
	struct addr mem;
	sljit_gpr dst_r;
	sljit_gpr src_r;
	sljit_s32 opcode = GET_OPCODE(op);

	CHECK_ERROR();
	CHECK(check_sljit_emit_op1(compiler, op, dst, dstw, src, srcw));
	ADJUST_LOCAL_OFFSET(dst, dstw);
	ADJUST_LOCAL_OFFSET(src, srcw);

	if (opcode >= SLJIT_MOV && opcode <= SLJIT_MOV_P) {
		/* LOAD REGISTER */
		if (FAST_IS_REG(dst) && FAST_IS_REG(src)) {
			dst_r = gpr(dst);
			src_r = gpr(src);
			switch (opcode | (op & SLJIT_32)) {
			/* 32-bit */
			case SLJIT_MOV32_U8:
				ins = llcr(dst_r, src_r);
				break;
			case SLJIT_MOV32_S8:
				ins = lbr(dst_r, src_r);
				break;
			case SLJIT_MOV32_U16:
				ins = llhr(dst_r, src_r);
				break;
			case SLJIT_MOV32_S16:
				ins = lhr(dst_r, src_r);
				break;
			case SLJIT_MOV32:
				if (dst_r == src_r)
					return SLJIT_SUCCESS;
				ins = lr(dst_r, src_r);
				break;
			/* 64-bit */
			case SLJIT_MOV_U8:
				ins = llgcr(dst_r, src_r);
				break;
			case SLJIT_MOV_S8:
				ins = lgbr(dst_r, src_r);
				break;
			case SLJIT_MOV_U16:
				ins = llghr(dst_r, src_r);
				break;
			case SLJIT_MOV_S16:
				ins = lghr(dst_r, src_r);
				break;
			case SLJIT_MOV_U32:
				ins = llgfr(dst_r, src_r);
				break;
			case SLJIT_MOV_S32:
				ins = lgfr(dst_r, src_r);
				break;
			case SLJIT_MOV:
			case SLJIT_MOV_P:
				if (dst_r == src_r)
					return SLJIT_SUCCESS;
				ins = lgr(dst_r, src_r);
				break;
			default:
				ins = 0;
				SLJIT_UNREACHABLE();
				break;
			}
			FAIL_IF(push_inst(compiler, ins));
			return SLJIT_SUCCESS;
		}
		/* LOAD IMMEDIATE */
		if (FAST_IS_REG(dst) && (src & SLJIT_IMM)) {
			switch (opcode) {
			case SLJIT_MOV_U8:
				srcw = (sljit_sw)((sljit_u8)(srcw));
				break;
			case SLJIT_MOV_S8:
				srcw = (sljit_sw)((sljit_s8)(srcw));
				break;
			case SLJIT_MOV_U16:
				srcw = (sljit_sw)((sljit_u16)(srcw));
				break;
			case SLJIT_MOV_S16:
				srcw = (sljit_sw)((sljit_s16)(srcw));
				break;
			case SLJIT_MOV_U32:
				srcw = (sljit_sw)((sljit_u32)(srcw));
				break;
			case SLJIT_MOV_S32:
			case SLJIT_MOV32:
				srcw = (sljit_sw)((sljit_s32)(srcw));
				break;
			}
			return push_load_imm_inst(compiler, gpr(dst), srcw);
		}
		/* LOAD */
		/* TODO(carenas): avoid reg being defined later */
		#define LEVAL(i) EVAL(i, reg, mem)
		if (FAST_IS_REG(dst) && (src & SLJIT_MEM)) {
			sljit_gpr reg = gpr(dst);

			FAIL_IF(make_addr_bxy(compiler, &mem, src, srcw, tmp1));
			/* TODO(carenas): convert all calls below to LEVAL */
			switch (opcode | (op & SLJIT_32)) {
			case SLJIT_MOV32_U8:
				ins = llc(reg, mem.offset, mem.index, mem.base);
				break;
			case SLJIT_MOV32_S8:
				ins = lb(reg, mem.offset, mem.index, mem.base);
				break;
			case SLJIT_MOV32_U16:
				ins = llh(reg, mem.offset, mem.index, mem.base);
				break;
			case SLJIT_MOV32_S16:
				ins = WHEN2(is_u12(mem.offset), lh, lhy);
				break;
			case SLJIT_MOV32:
				ins = WHEN2(is_u12(mem.offset), l, ly);
				break;
			case SLJIT_MOV_U8:
				ins = LEVAL(llgc);
				break;
			case SLJIT_MOV_S8:
				ins = lgb(reg, mem.offset, mem.index, mem.base);
				break;
			case SLJIT_MOV_U16:
				ins = LEVAL(llgh);
				break;
			case SLJIT_MOV_S16:
				ins = lgh(reg, mem.offset, mem.index, mem.base);
				break;
			case SLJIT_MOV_U32:
				ins = LEVAL(llgf);
				break;
			case SLJIT_MOV_S32:
				ins = lgf(reg, mem.offset, mem.index, mem.base);
				break;
			case SLJIT_MOV_P:
			case SLJIT_MOV:
				ins = lg(reg, mem.offset, mem.index, mem.base);
				break;
			default:
				ins = 0;
				SLJIT_UNREACHABLE();
				break;
			}
			FAIL_IF(push_inst(compiler, ins));
			return SLJIT_SUCCESS;
		}
		/* STORE and STORE IMMEDIATE */
		if ((dst & SLJIT_MEM)
			&& (FAST_IS_REG(src) || (src & SLJIT_IMM))) {
			sljit_gpr reg = FAST_IS_REG(src) ? gpr(src) : tmp0;
			if (src & SLJIT_IMM) {
				/* TODO(mundaym): MOVE IMMEDIATE? */
				FAIL_IF(push_load_imm_inst(compiler, reg, srcw));
			}
			struct addr mem;
			FAIL_IF(make_addr_bxy(compiler, &mem, dst, dstw, tmp1));
			switch (opcode) {
			case SLJIT_MOV_U8:
			case SLJIT_MOV_S8:
				return push_inst(compiler,
					WHEN2(is_u12(mem.offset), stc, stcy));
			case SLJIT_MOV_U16:
			case SLJIT_MOV_S16:
				return push_inst(compiler,
					WHEN2(is_u12(mem.offset), sth, sthy));
			case SLJIT_MOV_U32:
			case SLJIT_MOV_S32:
			case SLJIT_MOV32:
				return push_inst(compiler,
					WHEN2(is_u12(mem.offset), st, sty));
			case SLJIT_MOV_P:
			case SLJIT_MOV:
				FAIL_IF(push_inst(compiler, LEVAL(stg)));
				return SLJIT_SUCCESS;
			default:
				SLJIT_UNREACHABLE();
			}
		}
		#undef LEVAL
		/* MOVE CHARACTERS */
		if ((dst & SLJIT_MEM) && (src & SLJIT_MEM)) {
			struct addr mem;
			FAIL_IF(make_addr_bxy(compiler, &mem, src, srcw, tmp1));
			switch (opcode) {
			case SLJIT_MOV_U8:
			case SLJIT_MOV_S8:
				FAIL_IF(push_inst(compiler,
					EVAL(llgc, tmp0, mem)));
				FAIL_IF(make_addr_bxy(compiler, &mem, dst, dstw, tmp1));
				return push_inst(compiler,
					EVAL(stcy, tmp0, mem));
			case SLJIT_MOV_U16:
			case SLJIT_MOV_S16:
				FAIL_IF(push_inst(compiler,
					EVAL(llgh, tmp0, mem)));
				FAIL_IF(make_addr_bxy(compiler, &mem, dst, dstw, tmp1));
				return push_inst(compiler,
					EVAL(sthy, tmp0, mem));
			case SLJIT_MOV_U32:
			case SLJIT_MOV_S32:
			case SLJIT_MOV32:
				FAIL_IF(push_inst(compiler,
					EVAL(ly, tmp0, mem)));
				FAIL_IF(make_addr_bxy(compiler, &mem, dst, dstw, tmp1));
				return push_inst(compiler,
					EVAL(sty, tmp0, mem));
			case SLJIT_MOV_P:
			case SLJIT_MOV:
				FAIL_IF(push_inst(compiler,
					EVAL(lg, tmp0, mem)));
				FAIL_IF(make_addr_bxy(compiler, &mem, dst, dstw, tmp1));
				FAIL_IF(push_inst(compiler,
					EVAL(stg, tmp0, mem)));
				return SLJIT_SUCCESS;
			default:
				SLJIT_UNREACHABLE();
			}
		}
		SLJIT_UNREACHABLE();
	}

	SLJIT_ASSERT((src & SLJIT_IMM) == 0); /* no immediates */

	dst_r = FAST_IS_REG(dst) ? gpr(REG_MASK & dst) : tmp0;
	src_r = FAST_IS_REG(src) ? gpr(REG_MASK & src) : tmp0;
	if (src & SLJIT_MEM)
		FAIL_IF(load_word(compiler, src_r, src, srcw, src & SLJIT_32));

	compiler->status_flags_state = op & (VARIABLE_FLAG_MASK | SLJIT_SET_Z);

	/* TODO(mundaym): optimize loads and stores */
	switch (opcode | (op & SLJIT_32)) {
	case SLJIT_NOT:
		/* emulate ~x with x^-1 */
		FAIL_IF(push_load_imm_inst(compiler, tmp1, -1));
		if (src_r != dst_r)
			FAIL_IF(push_inst(compiler, lgr(dst_r, src_r)));

		FAIL_IF(push_inst(compiler, xgr(dst_r, tmp1)));
		break;
	case SLJIT_NOT32:
		/* emulate ~x with x^-1 */
		if (have_eimm())
			FAIL_IF(push_inst(compiler, xilf(dst_r, 0xffffffff)));
		else {
			FAIL_IF(push_load_imm_inst(compiler, tmp1, -1));
			if (src_r != dst_r)
				FAIL_IF(push_inst(compiler, lr(dst_r, src_r)));

			FAIL_IF(push_inst(compiler, xr(dst_r, tmp1)));
		}
		break;
	case SLJIT_CLZ:
		if (have_eimm()) {
			FAIL_IF(push_inst(compiler, flogr(tmp0, src_r))); /* clobbers tmp1 */
			if (dst_r != tmp0)
				FAIL_IF(push_inst(compiler, lgr(dst_r, tmp0)));
		} else {
			abort(); /* TODO(mundaym): no eimm (?) */
		}
		break;
	case SLJIT_CLZ32:
		if (have_eimm()) {
			FAIL_IF(push_inst(compiler, sllg(tmp1, src_r, 32, 0)));
			FAIL_IF(push_inst(compiler, iilf(tmp1, 0xffffffff)));
			FAIL_IF(push_inst(compiler, flogr(tmp0, tmp1))); /* clobbers tmp1 */
			if (dst_r != tmp0)
				FAIL_IF(push_inst(compiler, lr(dst_r, tmp0)));
		} else {
			abort(); /* TODO(mundaym): no eimm (?) */
		}
		break;
	default:
		SLJIT_UNREACHABLE();
	}

	if ((op & (SLJIT_SET_Z | VARIABLE_FLAG_MASK)) == (SLJIT_SET_Z | SLJIT_SET_OVERFLOW))
		FAIL_IF(update_zero_overflow(compiler, op, dst_r));

	/* TODO(carenas): doesn't need FAIL_IF */
	if (dst & SLJIT_MEM)
		FAIL_IF(store_word(compiler, dst_r, dst, dstw, op & SLJIT_32));

	return SLJIT_SUCCESS;
}

static SLJIT_INLINE int is_commutative(sljit_s32 op)
{
	switch (GET_OPCODE(op)) {
	case SLJIT_ADD:
	case SLJIT_ADDC:
	case SLJIT_MUL:
	case SLJIT_AND:
	case SLJIT_OR:
	case SLJIT_XOR:
		return 1;
	}
	return 0;
}

static SLJIT_INLINE int is_shift(sljit_s32 op) {
	sljit_s32 v = GET_OPCODE(op);
	return (v == SLJIT_SHL || v == SLJIT_ASHR || v == SLJIT_LSHR) ? 1 : 0;
}

static const struct ins_forms add_forms = {
	0x1a00, /* ar */
	0xb9080000, /* agr */
	0xb9f80000, /* ark */
	0xb9e80000, /* agrk */
	0x5a000000, /* a */
	0xe3000000005a, /* ay */
	0xe30000000008, /* ag */
};

static const struct ins_forms logical_add_forms = {
	0x1e00, /* alr */
	0xb90a0000, /* algr */
	0xb9fa0000, /* alrk */
	0xb9ea0000, /* algrk */
	0x5e000000, /* al */
	0xe3000000005e, /* aly */
	0xe3000000000a, /* alg */
};

static sljit_s32 sljit_emit_add(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	int sets_overflow = (op & VARIABLE_FLAG_MASK) == SLJIT_SET_OVERFLOW;
	int sets_zero_overflow = (op & (SLJIT_SET_Z | VARIABLE_FLAG_MASK)) == (SLJIT_SET_Z | SLJIT_SET_OVERFLOW);
	const struct ins_forms *forms;
	sljit_ins ins;

	if (src2 & SLJIT_IMM) {
		if (!sets_zero_overflow && is_s8(src2w) && (src1 & SLJIT_MEM) && (dst == src1 && dstw == src1w)) {
			if (sets_overflow)
				ins = (op & SLJIT_32) ? 0xeb000000006a /* asi */ : 0xeb000000007a /* agsi */;
			else
				ins = (op & SLJIT_32) ? 0xeb000000006e /* alsi */ : 0xeb000000007e /* algsi */;
			return emit_siy(compiler, ins, dst, dstw, src2w);
		}

		if (is_s16(src2w)) {
			if (sets_overflow)
				ins = (op & SLJIT_32) ? 0xec00000000d8 /* ahik */ : 0xec00000000d9 /* aghik */;
			else
				ins = (op & SLJIT_32) ? 0xec00000000da /* alhsik */ : 0xec00000000db /* alghsik */;
			FAIL_IF(emit_rie_d(compiler, ins, dst, src1, src1w, src2w));
			goto done;
		}

		if (!sets_overflow) {
			if ((op & SLJIT_32) || is_u32(src2w)) {
				ins = (op & SLJIT_32) ? 0xc20b00000000 /* alfi */ : 0xc20a00000000 /* algfi */;
				FAIL_IF(emit_ri(compiler, ins, dst, src1, src1w, src2w, RIL_A));
				goto done;
			}
			if (is_u32(-src2w)) {
				FAIL_IF(emit_ri(compiler, 0xc20400000000 /* slgfi */, dst, src1, src1w, -src2w, RIL_A));
				goto done;
			}
		}
		else if ((op & SLJIT_32) || is_s32(src2w)) {
			ins = (op & SLJIT_32) ? 0xc20900000000 /* afi */ : 0xc20800000000 /* agfi */;
			FAIL_IF(emit_ri(compiler, ins, dst, src1, src1w, src2w, RIL_A));
			goto done;
		}
	}

	forms = sets_overflow ? &add_forms : &logical_add_forms;
	FAIL_IF(emit_commutative(compiler, forms, dst, src1, src1w, src2, src2w));

done:
	if (sets_zero_overflow)
		FAIL_IF(update_zero_overflow(compiler, op, FAST_IS_REG(dst) ? gpr(dst & REG_MASK) : tmp0));

	if (dst & SLJIT_MEM)
		return store_word(compiler, tmp0, dst, dstw, op & SLJIT_32);

	return SLJIT_SUCCESS;
}

static const struct ins_forms sub_forms = {
	0x1b00, /* sr */
	0xb9090000, /* sgr */
	0xb9f90000, /* srk */
	0xb9e90000, /* sgrk */
	0x5b000000, /* s */
	0xe3000000005b, /* sy */
	0xe30000000009, /* sg */
};

static const struct ins_forms logical_sub_forms = {
	0x1f00, /* slr */
	0xb90b0000, /* slgr */
	0xb9fb0000, /* slrk */
	0xb9eb0000, /* slgrk */
	0x5f000000, /* sl */
	0xe3000000005f, /* sly */
	0xe3000000000b, /* slg */
};

static sljit_s32 sljit_emit_sub(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_s32 flag_type = GET_FLAG_TYPE(op);
	int sets_signed = (flag_type >= SLJIT_SIG_LESS && flag_type <= SLJIT_NOT_OVERFLOW);
	int sets_zero_overflow = (op & (SLJIT_SET_Z | VARIABLE_FLAG_MASK)) == (SLJIT_SET_Z | SLJIT_SET_OVERFLOW);
	const struct ins_forms *forms;
	sljit_ins ins;

	if (dst == (sljit_s32)tmp0 && flag_type <= SLJIT_SIG_LESS_EQUAL) {
		int compare_signed = flag_type >= SLJIT_SIG_LESS;

		compiler->status_flags_state |= SLJIT_CURRENT_FLAGS_COMPARE;

		if (src2 & SLJIT_IMM) {
			if (compare_signed || ((op & VARIABLE_FLAG_MASK) == 0 && is_s32(src2w)))
			{
				if ((op & SLJIT_32) || is_s32(src2w)) {
					ins = (op & SLJIT_32) ? 0xc20d00000000 /* cfi */ : 0xc20c00000000 /* cgfi */;
					return emit_ri(compiler, ins, src1, src1, src1w, src2w, RIL_A);
				}
			}
			else {
				if ((op & SLJIT_32) || is_u32(src2w)) {
					ins = (op & SLJIT_32) ? 0xc20f00000000 /* clfi */ : 0xc20e00000000 /* clgfi */;
					return emit_ri(compiler, ins, src1, src1, src1w, src2w, RIL_A);
				}
				if (is_s16(src2w))
					return emit_rie_d(compiler, 0xec00000000db /* alghsik */, (sljit_s32)tmp0, src1, src1w, src2w);
			}
		}
		else if (src2 & SLJIT_MEM) {
			if ((op & SLJIT_32) && ((src2 & OFFS_REG_MASK) || is_u12(src2w))) {
				ins = compare_signed ? 0x59000000 /* c */ : 0x55000000 /* cl */;
				return emit_rx(compiler, ins, src1, src1, src1w, src2, src2w, RX_A);
			}

			if (compare_signed)
				ins = (op & SLJIT_32) ? 0xe30000000059 /* cy */ : 0xe30000000020 /* cg */;
			else
				ins = (op & SLJIT_32) ? 0xe30000000055 /* cly */ : 0xe30000000021 /* clg */;
			return emit_rx(compiler, ins, src1, src1, src1w, src2, src2w, RXY_A);
		}

		if (compare_signed)
			ins = (op & SLJIT_32) ? 0x1900 /* cr */ : 0xb9200000 /* cgr */;
		else
			ins = (op & SLJIT_32) ? 0x1500 /* clr */ : 0xb9210000 /* clgr */;
		return emit_rr(compiler, ins, src1, src1, src1w, src2, src2w);
	}

	if (src1 == SLJIT_IMM && src1w == 0 && (flag_type == 0 || sets_signed)) {
		ins = (op & SLJIT_32) ? 0x1300 /* lcr */ : 0xb9030000 /* lcgr */;
		FAIL_IF(emit_rr1(compiler, ins, dst, src2, src2w));
		goto done;
	}

	if (src2 & SLJIT_IMM) {
		sljit_sw neg_src2w = -src2w;

		if (sets_signed || neg_src2w != 0 || (op & (SLJIT_SET_Z | VARIABLE_FLAG_MASK)) == 0) {
			if (!sets_zero_overflow && is_s8(neg_src2w) && (src1 & SLJIT_MEM) && (dst == src1 && dstw == src1w)) {
				if (sets_signed)
					ins = (op & SLJIT_32) ? 0xeb000000006a /* asi */ : 0xeb000000007a /* agsi */;
				else
					ins = (op & SLJIT_32) ? 0xeb000000006e /* alsi */ : 0xeb000000007e /* algsi */;
				return emit_siy(compiler, ins, dst, dstw, neg_src2w);
			}

			if (is_s16(neg_src2w)) {
				if (sets_signed)
					ins = (op & SLJIT_32) ? 0xec00000000d8 /* ahik */ : 0xec00000000d9 /* aghik */;
				else
					ins = (op & SLJIT_32) ? 0xec00000000da /* alhsik */ : 0xec00000000db /* alghsik */;
				FAIL_IF(emit_rie_d(compiler, ins, dst, src1, src1w, neg_src2w));
				goto done;
			}
		}

		if (!sets_signed) {
			if ((op & SLJIT_32) || is_u32(src2w)) {
				ins = (op & SLJIT_32) ? 0xc20500000000 /* slfi */ : 0xc20400000000 /* slgfi */;
				FAIL_IF(emit_ri(compiler, ins, dst, src1, src1w, src2w, RIL_A));
				goto done;
			}
			if (is_u32(neg_src2w)) {
				FAIL_IF(emit_ri(compiler, 0xc20a00000000 /* algfi */, dst, src1, src1w, neg_src2w, RIL_A));
				goto done;
			}
		}
		else if ((op & SLJIT_32) || is_s32(neg_src2w)) {
			ins = (op & SLJIT_32) ? 0xc20900000000 /* afi */ : 0xc20800000000 /* agfi */;
			FAIL_IF(emit_ri(compiler, ins, dst, src1, src1w, neg_src2w, RIL_A));
			goto done;
		}
	}

	forms = sets_signed ? &sub_forms : &logical_sub_forms;
	FAIL_IF(emit_non_commutative(compiler, forms, dst, src1, src1w, src2, src2w));

done:
	if (sets_signed) {
		sljit_gpr dst_r = FAST_IS_REG(dst) ? gpr(dst & REG_MASK) : tmp0;

		if ((op & VARIABLE_FLAG_MASK) != SLJIT_SET_OVERFLOW) {
			/* In case of overflow, the sign bit of the two source operands must be different, and
			     - the first operand is greater if the sign bit of the result is set
			     - the first operand is less if the sign bit of the result is not set
			   The -result operation sets the corrent sign, because the result cannot be zero.
			   The overflow is considered greater, since the result must be equal to INT_MIN so its sign bit is set. */
			FAIL_IF(push_inst(compiler, brc(0xe, 2 + 2)));
			FAIL_IF(push_inst(compiler, (op & SLJIT_32) ? lcr(tmp1, dst_r) : lcgr(tmp1, dst_r)));
		}
		else if (op & SLJIT_SET_Z)
			FAIL_IF(update_zero_overflow(compiler, op, dst_r));
	}

	if (dst & SLJIT_MEM)
		return store_word(compiler, tmp0, dst, dstw, op & SLJIT_32);

	return SLJIT_SUCCESS;
}

static const struct ins_forms multiply_forms = {
	0xb2520000, /* msr */
	0xb90c0000, /* msgr */
	0xb9fd0000, /* msrkc */
	0xb9ed0000, /* msgrkc */
	0x71000000, /* ms */
	0xe30000000051, /* msy */
	0xe3000000000c, /* msg */
};

static const struct ins_forms multiply_overflow_forms = {
	0,
	0,
	0xb9fd0000, /* msrkc */
	0xb9ed0000, /* msgrkc */
	0,
	0xe30000000053, /* msc */
	0xe30000000083, /* msgc */
};

static sljit_s32 sljit_emit_multiply(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_ins ins;

	if (HAS_FLAGS(op)) {
		/* if have_misc2 fails, this operation should be emulated. 32 bit emulation:
		FAIL_IF(push_inst(compiler, lgfr(tmp0, src1_r)));
		FAIL_IF(push_inst(compiler, msgfr(tmp0, src2_r)));
		if (dst_r != tmp0) {
			FAIL_IF(push_inst(compiler, lr(dst_r, tmp0)));
		}
		FAIL_IF(push_inst(compiler, aih(tmp0, 1)));
		FAIL_IF(push_inst(compiler, nihf(tmp0, ~1U)));
		FAIL_IF(push_inst(compiler, ipm(tmp1)));
		FAIL_IF(push_inst(compiler, oilh(tmp1, 0x2000))); */

		return emit_commutative(compiler, &multiply_overflow_forms, dst, src1, src1w, src2, src2w);
	}

	if (src2 & SLJIT_IMM) {
		if (is_s16(src2w)) {
			ins = (op & SLJIT_32) ? 0xa70c0000 /* mhi */ : 0xa70d0000 /* mghi */;
			return emit_ri(compiler, ins, dst, src1, src1w, src2w, RI_A);
		}

		if (is_s32(src2w)) {
			ins = (op & SLJIT_32) ? 0xc20100000000 /* msfi */ : 0xc20000000000 /* msgfi */;
			return emit_ri(compiler, ins, dst, src1, src1w, src2w, RIL_A);
		}
	}

	return emit_commutative(compiler, &multiply_forms, dst, src1, src1w, src2, src2w);
}

static sljit_s32 sljit_emit_bitwise_imm(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst,
	sljit_s32 src1, sljit_sw src1w,
	sljit_uw imm, sljit_s32 count16)
{
	sljit_s32 mode = compiler->mode;
	sljit_gpr dst_r = tmp0;
	sljit_s32 needs_move = 1;

	if (IS_GPR_REG(dst)) {
		dst_r = gpr(dst & REG_MASK);
		if (dst == src1)
			needs_move = 0;
	}

	if (needs_move)
		FAIL_IF(emit_move(compiler, dst_r, src1, src1w));

	if (type == SLJIT_AND) {
		if (!(mode & SLJIT_32))
			FAIL_IF(push_inst(compiler, 0xc00a00000000 /* nihf */ | R36A(dst_r) | (imm >> 32)));
		return push_inst(compiler, 0xc00b00000000 /* nilf */ | R36A(dst_r) | (imm & 0xffffffff));
	}
	else if (type == SLJIT_OR) {
		if (count16 >= 3) {
			FAIL_IF(push_inst(compiler, 0xc00c00000000 /* oihf */ | R36A(dst_r) | (imm >> 32)));
			return push_inst(compiler, 0xc00d00000000 /* oilf */ | R36A(dst_r) | (imm & 0xffffffff));
		}

		if (count16 >= 2) {
			if ((imm & 0x00000000ffffffffull) == 0)
				return push_inst(compiler, 0xc00c00000000 /* oihf */ | R36A(dst_r) | (imm >> 32));
			if ((imm & 0xffffffff00000000ull) == 0)
				return push_inst(compiler, 0xc00d00000000 /* oilf */ | R36A(dst_r) | (imm & 0xffffffff));
		}

		if ((imm & 0xffff000000000000ull) != 0)
			FAIL_IF(push_inst(compiler, 0xa5080000 /* oihh */ | R20A(dst_r) | (imm >> 48)));
		if ((imm & 0x0000ffff00000000ull) != 0)
			FAIL_IF(push_inst(compiler, 0xa5090000 /* oihl */ | R20A(dst_r) | ((imm >> 32) & 0xffff)));
		if ((imm & 0x00000000ffff0000ull) != 0)
			FAIL_IF(push_inst(compiler, 0xa50a0000 /* oilh */ | R20A(dst_r) | ((imm >> 16) & 0xffff)));
		if ((imm & 0x000000000000ffffull) != 0 || imm == 0)
			return push_inst(compiler, 0xa50b0000 /* oill */ | R20A(dst_r) | (imm & 0xffff));
		return SLJIT_SUCCESS;
	}

	if ((imm & 0xffffffff00000000ull) != 0)
		FAIL_IF(push_inst(compiler, 0xc00600000000 /* xihf */ | R36A(dst_r) | (imm >> 32)));
	if ((imm & 0x00000000ffffffffull) != 0 || imm == 0)
		return push_inst(compiler, 0xc00700000000 /* xilf */ | R36A(dst_r) | (imm & 0xffffffff));
	return SLJIT_SUCCESS;
}

static const struct ins_forms bitwise_and_forms = {
	0x1400, /* nr */
	0xb9800000, /* ngr */
	0xb9f40000, /* nrk */
	0xb9e40000, /* ngrk */
	0x54000000, /* n */
	0xe30000000054, /* ny */
	0xe30000000080, /* ng */
};

static const struct ins_forms bitwise_or_forms = {
	0x1600, /* or */
	0xb9810000, /* ogr */
	0xb9f60000, /* ork */
	0xb9e60000, /* ogrk */
	0x56000000, /* o */
	0xe30000000056, /* oy */
	0xe30000000081, /* og */
};

static const struct ins_forms bitwise_xor_forms = {
	0x1700, /* xr */
	0xb9820000, /* xgr */
	0xb9f70000, /* xrk */
	0xb9e70000, /* xgrk */
	0x57000000, /* x */
	0xe30000000057, /* xy */
	0xe30000000082, /* xg */
};

static sljit_s32 sljit_emit_bitwise(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_s32 type = GET_OPCODE(op);
	const struct ins_forms *forms;

	if ((src2 & SLJIT_IMM) && (!(op & SLJIT_SET_Z) || (type == SLJIT_AND && dst == (sljit_s32)tmp0))) {
		sljit_s32 count16 = 0;
		sljit_uw imm = (sljit_uw)src2w;

		if (op & SLJIT_32)
			imm &= 0xffffffffull;

		if ((imm & 0x000000000000ffffull) != 0 || imm == 0)
			count16++;
		if ((imm & 0x00000000ffff0000ull) != 0)
			count16++;
		if ((imm & 0x0000ffff00000000ull) != 0)
			count16++;
		if ((imm & 0xffff000000000000ull) != 0)
			count16++;

		if (type == SLJIT_AND && dst == (sljit_s32)tmp0 && count16 == 1) {
			sljit_gpr src_r = tmp0;

			if (FAST_IS_REG(src1))
				src_r = gpr(src1 & REG_MASK);
			else
				FAIL_IF(emit_move(compiler, tmp0, src1, src1w));

			if ((imm & 0x000000000000ffffull) != 0 || imm == 0)
				return push_inst(compiler, 0xa7010000 | R20A(src_r) | imm);
			if ((imm & 0x00000000ffff0000ull) != 0)
				return push_inst(compiler, 0xa7000000 | R20A(src_r) | (imm >> 16));
			if ((imm & 0x0000ffff00000000ull) != 0)
				return push_inst(compiler, 0xa7030000 | R20A(src_r) | (imm >> 32));
			return push_inst(compiler, 0xa7020000 | R20A(src_r) | (imm >> 48));
		}

		if (!(op & SLJIT_SET_Z))
			return sljit_emit_bitwise_imm(compiler, type, dst, src1, src1w, imm, count16);
	}

	if (type == SLJIT_AND)
		forms = &bitwise_and_forms;
	else if (type == SLJIT_OR)
		forms = &bitwise_or_forms;
	else
		forms = &bitwise_xor_forms;

	return emit_commutative(compiler, forms, dst, src1, src1w, src2, src2w);
}

static sljit_s32 sljit_emit_shift(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_s32 type = GET_OPCODE(op);
	sljit_gpr dst_r = FAST_IS_REG(dst) ? gpr(dst & REG_MASK) : tmp0;
	sljit_gpr src_r = tmp0;
	sljit_gpr base_r = tmp0;
	sljit_ins imm = 0;
	sljit_ins ins;

	if (FAST_IS_REG(src1))
		src_r = gpr(src1 & REG_MASK);
	else
		FAIL_IF(emit_move(compiler, tmp0, src1, src1w));

	if (src2 & SLJIT_IMM)
		imm = (sljit_ins)(src2w & ((op & SLJIT_32) ? 0x1f : 0x3f));
	else if (FAST_IS_REG(src2))
		base_r = gpr(src2 & REG_MASK);
	else {
		FAIL_IF(emit_move(compiler, tmp1, src2, src2w));
		base_r = tmp1;
	}

	if ((op & SLJIT_32) && dst_r == src_r) {
		if (type == SLJIT_SHL)
			ins = 0x89000000 /* sll */;
		else if (type == SLJIT_LSHR)
			ins = 0x88000000 /* srl */;
		else
			ins = 0x8a000000 /* sra */;

		FAIL_IF(push_inst(compiler, ins | R20A(dst_r) | R12A(base_r) | imm));
	}
	else {
		if (type == SLJIT_SHL)
			ins = (op & SLJIT_32) ? 0xeb00000000df /* sllk */ : 0xeb000000000d /* sllg */;
		else if (type == SLJIT_LSHR)
			ins = (op & SLJIT_32) ? 0xeb00000000de /* srlk */ : 0xeb000000000c /* srlg */;
		else
			ins = (op & SLJIT_32) ? 0xeb00000000dc /* srak */ : 0xeb000000000a /* srag */;

		FAIL_IF(push_inst(compiler, ins | R36A(dst_r) | R32A(src_r) | R28A(base_r) | (imm << 16)));
	}

	if ((op & SLJIT_SET_Z) && type != SLJIT_ASHR)
		return push_inst(compiler, (op & SLJIT_32) ? or(dst_r, dst_r) : ogr(dst_r, dst_r));

	return SLJIT_SUCCESS;
}

static const struct ins_forms addc_forms = {
	0xb9980000, /* alcr */
	0xb9880000, /* alcgr */
	0,
	0,
	0,
	0xe30000000098, /* alc */
	0xe30000000088, /* alcg */
};

static const struct ins_forms subc_forms = {
	0xb9990000, /* slbr */
	0xb9890000, /* slbgr */
	0,
	0,
	0,
	0xe30000000099, /* slb */
	0xe30000000089, /* slbg */
};

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op2(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_op2(compiler, op, 0, dst, dstw, src1, src1w, src2, src2w));
	ADJUST_LOCAL_OFFSET(dst, dstw);
	ADJUST_LOCAL_OFFSET(src1, src1w);
	ADJUST_LOCAL_OFFSET(src2, src2w);

	compiler->mode = op & SLJIT_32;
	compiler->status_flags_state = op & (VARIABLE_FLAG_MASK | SLJIT_SET_Z);

	if (is_commutative(op) && (src1 & SLJIT_IMM) && !(src2 & SLJIT_IMM)) {
		src1 ^= src2;
		src2 ^= src1;
		src1 ^= src2;

		src1w ^= src2w;
		src2w ^= src1w;
		src1w ^= src2w;
	}

	switch (GET_OPCODE(op)) {
	case SLJIT_ADD:
		compiler->status_flags_state |= SLJIT_CURRENT_FLAGS_ADD;
		return sljit_emit_add(compiler, op, dst, dstw, src1, src1w, src2, src2w);
	case SLJIT_ADDC:
		compiler->status_flags_state |= SLJIT_CURRENT_FLAGS_ADD;
		FAIL_IF(emit_commutative(compiler, &addc_forms, dst, src1, src1w, src2, src2w));
		if (dst & SLJIT_MEM)
			return store_word(compiler, tmp0, dst, dstw, op & SLJIT_32);
		return SLJIT_SUCCESS;
	case SLJIT_SUB:
		compiler->status_flags_state |= SLJIT_CURRENT_FLAGS_SUB;
		return sljit_emit_sub(compiler, op, dst, dstw, src1, src1w, src2, src2w);
	case SLJIT_SUBC:
		compiler->status_flags_state |= SLJIT_CURRENT_FLAGS_SUB;
		FAIL_IF(emit_non_commutative(compiler, &subc_forms, dst, src1, src1w, src2, src2w));
		if (dst & SLJIT_MEM)
			return store_word(compiler, tmp0, dst, dstw, op & SLJIT_32);
		return SLJIT_SUCCESS;
	case SLJIT_MUL:
		FAIL_IF(sljit_emit_multiply(compiler, op, dst, src1, src1w, src2, src2w));
		break;
	case SLJIT_AND:
	case SLJIT_OR:
	case SLJIT_XOR:
		FAIL_IF(sljit_emit_bitwise(compiler, op, dst, src1, src1w, src2, src2w));
		break;
	case SLJIT_SHL:
	case SLJIT_LSHR:
	case SLJIT_ASHR:
		FAIL_IF(sljit_emit_shift(compiler, op, dst, src1, src1w, src2, src2w));
		break;
	}

	if (dst & SLJIT_MEM)
		return store_word(compiler, tmp0, dst, dstw, op & SLJIT_32);
	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op2u(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_op2(compiler, op, 1, 0, 0, src1, src1w, src2, src2w));

#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE) \
		|| (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	compiler->skip_checks = 1;
#endif
	return sljit_emit_op2(compiler, op, (sljit_s32)tmp0, 0, src1, src1w, src2, src2w);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_src(
	struct sljit_compiler *compiler,
	sljit_s32 op, sljit_s32 src, sljit_sw srcw)
{
	sljit_gpr src_r;

	CHECK_ERROR();
	CHECK(check_sljit_emit_op_src(compiler, op, src, srcw));
	ADJUST_LOCAL_OFFSET(src, srcw);

	switch (op) {
	case SLJIT_FAST_RETURN:
		src_r = FAST_IS_REG(src) ? gpr(src) : tmp1;
		if (src & SLJIT_MEM)
			FAIL_IF(load_word(compiler, tmp1, src, srcw, 0));

		return push_inst(compiler, br(src_r));
	case SLJIT_SKIP_FRAMES_BEFORE_FAST_RETURN:
		/* TODO(carenas): implement? */
		return SLJIT_SUCCESS;
	case SLJIT_PREFETCH_L1:
	case SLJIT_PREFETCH_L2:
	case SLJIT_PREFETCH_L3:
	case SLJIT_PREFETCH_ONCE:
		/* TODO(carenas): implement */
		return SLJIT_SUCCESS;
	default:
                /* TODO(carenas): probably should not success by default */
		return SLJIT_SUCCESS;
	}

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_get_register_index(sljit_s32 reg)
{
	CHECK_REG_INDEX(check_sljit_get_register_index(reg));
	return (sljit_s32)gpr(reg);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_get_float_register_index(sljit_s32 reg)
{
	CHECK_REG_INDEX(check_sljit_get_float_register_index(reg));
	return (sljit_s32)fgpr(reg);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_custom(struct sljit_compiler *compiler,
	void *instruction, sljit_u32 size)
{
	sljit_ins ins = 0;

	CHECK_ERROR();
	CHECK(check_sljit_emit_op_custom(compiler, instruction, size));

	memcpy((sljit_u8 *)&ins + sizeof(ins) - size, instruction, size);
	return push_inst(compiler, ins);
}

/* --------------------------------------------------------------------- */
/*  Floating point operators                                             */
/* --------------------------------------------------------------------- */

#define FLOAT_LOAD 0
#define FLOAT_STORE 1

static sljit_s32 float_mem(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 reg,
	sljit_s32 mem, sljit_sw memw)
{
	struct addr addr;
	sljit_ins ins;

	SLJIT_ASSERT(mem & SLJIT_MEM);

	if ((mem & OFFS_REG_MASK) || is_u12(memw) || !is_s20(memw)) {
		FAIL_IF(make_addr_bx(compiler, &addr, mem, memw, tmp1));

		if (op & FLOAT_STORE)
			ins = (op & SLJIT_32) ? 0x70000000 /* ste */ : 0x60000000 /* std */;
		else
			ins = (op & SLJIT_32) ? 0x78000000 /* le */ : 0x68000000 /* ld */;

		return push_inst(compiler, ins | F20(reg) | R16A(addr.index) | R12A(addr.base) | (sljit_ins)addr.offset);
	}

	FAIL_IF(make_addr_bxy(compiler, &addr, mem, memw, tmp1));

	if (op & FLOAT_STORE)
		ins = (op & SLJIT_32) ? 0xed0000000066 /* stey */ : 0xed0000000067 /* stdy */;
	else
		ins = (op & SLJIT_32) ? 0xed0000000064 /* ley */ : 0xed0000000065 /* ldy */;

	return push_inst(compiler, ins | F36(reg) | R32A(addr.index) | R28A(addr.base) | disp_s20(addr.offset));
}

static sljit_s32 emit_float(struct sljit_compiler *compiler, sljit_ins ins_r, sljit_ins ins,
	sljit_s32 reg,
	sljit_s32 src, sljit_sw srcw)
{
	struct addr addr;

	if (!(src & SLJIT_MEM))
		return push_inst(compiler, ins_r | F4(reg) | F0(src));

	FAIL_IF(make_addr_bx(compiler, &addr, src, srcw, tmp1));
	return push_inst(compiler, ins | F36(reg) | R32A(addr.index) | R28A(addr.base) | ((sljit_ins)addr.offset << 16));
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_sw_from_f64(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_ins dst_r = FAST_IS_REG(dst) ? gpr(dst) : tmp0;
	sljit_ins ins;

	if (src & SLJIT_MEM) {
		FAIL_IF(float_mem(compiler, FLOAT_LOAD | (op & SLJIT_32), TMP_FREG1, src, srcw));
		src = TMP_FREG1;
	}

	/* M3 is set to 5 */
	if (GET_OPCODE(op) == SLJIT_CONV_SW_FROM_F64)
		ins = (op & SLJIT_32) ? 0xb3a85000 /* cgebr */ : 0xb3a95000 /* cgdbr */;
	else
		ins = (op & SLJIT_32) ? 0xb3985000 /* cfebr */ : 0xb3995000 /* cfdbr */;

	FAIL_IF(push_inst(compiler, ins | R4A(dst_r) | F0(src)));

	if (dst & SLJIT_MEM)
		return store_word(compiler, dst_r, dst, dstw, GET_OPCODE(op) >= SLJIT_CONV_S32_FROM_F64);

	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_conv_f64_from_sw(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG1;
	sljit_ins ins;

	if (src & SLJIT_IMM) {
		FAIL_IF(push_load_imm_inst(compiler, tmp0, srcw));
		src = (sljit_s32)tmp0;
	}
	else if (src & SLJIT_MEM) {
		FAIL_IF(load_word(compiler, tmp0, src, srcw, GET_OPCODE(op) >= SLJIT_CONV_F64_FROM_S32));
		src = (sljit_s32)tmp0;
	}

	if (GET_OPCODE(op) == SLJIT_CONV_F64_FROM_SW)
		ins = (op & SLJIT_32) ? 0xb3a40000 /* cegbr */ : 0xb3a50000 /* cdgbr */;
	else
		ins = (op & SLJIT_32) ? 0xb3940000 /* cefbr */ : 0xb3950000 /* cdfbr */;

	FAIL_IF(push_inst(compiler, ins | F4(dst_r) | R0(src)));

	if (dst & SLJIT_MEM)
		return float_mem(compiler, FLOAT_STORE | (op & SLJIT_32), TMP_FREG1, dst, dstw);

	return SLJIT_SUCCESS;
}

static SLJIT_INLINE sljit_s32 sljit_emit_fop1_cmp(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_ins ins_r, ins;

	if (src1 & SLJIT_MEM) {
		FAIL_IF(float_mem(compiler, FLOAT_LOAD | (op & SLJIT_32), TMP_FREG1, src1, src1w));
		src1 = TMP_FREG1;
	}

	if (op & SLJIT_32) {
		ins_r = 0xb3090000 /* cebr */;
		ins = 0xed0000000009 /* ceb */;
	} else {
		ins_r = 0xb3190000 /* cdbr */;
		ins = 0xed0000000019 /* cdb */;
	}

	return emit_float(compiler, ins_r, ins, src1, src2, src2w);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fop1(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_s32 dst_r;
	sljit_ins ins;

	CHECK_ERROR();

	SELECT_FOP1_OPERATION_WITH_CHECKS(compiler, op, dst, dstw, src, srcw);

	dst_r = FAST_IS_REG(dst) ? dst : TMP_FREG1;

	if (op == SLJIT_CONV_F64_FROM_F32)
		FAIL_IF(emit_float(compiler, 0xb3040000 /* ldebr */, 0xed0000000004 /* ldeb */, dst_r, src, srcw));
	else {
		if (src & SLJIT_MEM) {
			FAIL_IF(float_mem(compiler, FLOAT_LOAD | (op == SLJIT_CONV_F32_FROM_F64 ? 0 : (op & SLJIT_32)), dst_r, src, srcw));
			src = dst_r;
		}

		switch (GET_OPCODE(op)) {
		case SLJIT_MOV_F64:
			if (FAST_IS_REG(dst)) {
				if (dst == src)
					return SLJIT_SUCCESS;

				ins = (op & SLJIT_32) ? 0x3800 /* ler */ : 0x2800 /* ldr */;
				break;
			}
			return float_mem(compiler, FLOAT_STORE | (op & SLJIT_32), src, dst, dstw);
		case SLJIT_CONV_F64_FROM_F32:
			/* Only SLJIT_CONV_F32_FROM_F64. */
			ins = 0xb3440000 /* ledbr */;
			break;
		case SLJIT_NEG_F64:
			ins = (op & SLJIT_32) ? 0xb3030000 /* lcebr */ : 0xb3130000 /* lcdbr */;
			break;
		default:
			SLJIT_ASSERT(GET_OPCODE(op) == SLJIT_ABS_F64);
			ins = (op & SLJIT_32) ? 0xb3000000 /* lpebr */ : 0xb3100000 /* lpdbr */;
			break;
		}

		FAIL_IF(push_inst(compiler, ins | F4(dst_r) | F0(src)));
	}

	if (!(dst & SLJIT_MEM))
		return SLJIT_SUCCESS;

	SLJIT_ASSERT(dst_r == TMP_FREG1);

	return float_mem(compiler, FLOAT_STORE | (op & SLJIT_32), TMP_FREG1, dst, dstw);
}

#define FLOAT_MOV(op, dst_r, src_r) \
	(((op & SLJIT_32) ? 0x3800 /* ler */ : 0x2800 /* ldr */) | F4(dst_r) | F0(src_r))

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fop2(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 src1, sljit_sw src1w,
	sljit_s32 src2, sljit_sw src2w)
{
	sljit_s32 dst_r = TMP_FREG1;
	sljit_ins ins_r, ins;

	CHECK_ERROR();
	CHECK(check_sljit_emit_fop2(compiler, op, dst, dstw, src1, src1w, src2, src2w));
	ADJUST_LOCAL_OFFSET(dst, dstw);
	ADJUST_LOCAL_OFFSET(src1, src1w);
	ADJUST_LOCAL_OFFSET(src2, src2w);

	do {
		if (FAST_IS_REG(dst)) {
			dst_r = dst;

			if (dst == src1)
				break;

			if (dst == src2) {
				if (GET_OPCODE(op) == SLJIT_ADD_F64 || GET_OPCODE(op) == SLJIT_MUL_F64) {
					src2 = src1;
					src2w = src1w;
					src1 = dst;
					break;
				}

				FAIL_IF(push_inst(compiler, FLOAT_MOV(op, TMP_FREG1, src2)));
				src2 = TMP_FREG1;
			}
		}

		if (src1 & SLJIT_MEM)
			FAIL_IF(float_mem(compiler, FLOAT_LOAD | (op & SLJIT_32), dst_r, src1, src1w));
		else
			FAIL_IF(push_inst(compiler, FLOAT_MOV(op, dst_r, src1)));
	} while (0);

	switch (GET_OPCODE(op)) {
	case SLJIT_ADD_F64:
		ins_r = (op & SLJIT_32) ? 0xb30a0000 /* aebr */ : 0xb31a0000 /* adbr */;
		ins = (op & SLJIT_32) ? 0xed000000000a /* aeb */ : 0xed000000001a /* adb */;
		break;
	case SLJIT_SUB_F64:
		ins_r = (op & SLJIT_32) ? 0xb30b0000 /* sebr */ : 0xb31b0000 /* sdbr */;
		ins = (op & SLJIT_32) ? 0xed000000000b /* seb */ : 0xed000000001b /* sdb */;
		break;
	case SLJIT_MUL_F64:
		ins_r = (op & SLJIT_32) ? 0xb3170000 /* meebr */ : 0xb31c0000 /* mdbr */;
		ins = (op & SLJIT_32) ? 0xed0000000017 /* meeb */ : 0xed000000001c /* mdb */;
		break;
	default:
		SLJIT_ASSERT(GET_OPCODE(op) == SLJIT_DIV_F64);
		ins_r = (op & SLJIT_32) ? 0xb30d0000 /* debr */ : 0xb31d0000 /* ddbr */;
		ins = (op & SLJIT_32) ? 0xed000000000d /* deb */ : 0xed000000001d /* ddb */;
		break;
	}

	FAIL_IF(emit_float(compiler, ins_r, ins, dst_r, src2, src2w));

	if (dst & SLJIT_MEM)
		return float_mem(compiler, FLOAT_STORE | (op & SLJIT_32), TMP_FREG1, dst, dstw);

	SLJIT_ASSERT(dst_r != TMP_FREG1);
	return SLJIT_SUCCESS;
}

/* --------------------------------------------------------------------- */
/*  Other instructions                                                   */
/* --------------------------------------------------------------------- */

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_fast_enter(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_fast_enter(compiler, dst, dstw));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	if (FAST_IS_REG(dst))
		return push_inst(compiler, lgr(gpr(dst), fast_link_r));

	/* memory */
	return store_word(compiler, fast_link_r, dst, dstw, 0);
}

/* --------------------------------------------------------------------- */
/*  Conditional instructions                                             */
/* --------------------------------------------------------------------- */

SLJIT_API_FUNC_ATTRIBUTE struct sljit_label* sljit_emit_label(struct sljit_compiler *compiler)
{
	struct sljit_label *label;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_label(compiler));

	if (compiler->last_label && compiler->last_label->size == compiler->size)
		return compiler->last_label;

	label = (struct sljit_label*)ensure_abuf(compiler, sizeof(struct sljit_label));
	PTR_FAIL_IF(!label);
	set_label(label, compiler);
	return label;
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_jump(struct sljit_compiler *compiler, sljit_s32 type)
{
	sljit_u8 mask = ((type & 0xff) < SLJIT_JUMP) ? get_cc(compiler, type & 0xff) : 0xf;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_jump(compiler, type));

	/* record jump */
	struct sljit_jump *jump = (struct sljit_jump *)
		ensure_abuf(compiler, sizeof(struct sljit_jump));
	PTR_FAIL_IF(!jump);
	set_jump(jump, compiler, type & SLJIT_REWRITABLE_JUMP);
	jump->addr = compiler->size;

	/* emit jump instruction */
	type &= 0xff;
	if (type >= SLJIT_FAST_CALL)
		PTR_FAIL_IF(push_inst(compiler, brasl(type == SLJIT_FAST_CALL ? fast_link_r : link_r, 0)));
	else
		PTR_FAIL_IF(push_inst(compiler, brcl(mask, 0)));

	return jump;
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_jump* sljit_emit_call(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 arg_types)
{
	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_call(compiler, type, arg_types));

	if (type & SLJIT_CALL_RETURN) {
		PTR_FAIL_IF(emit_stack_frame_release(compiler));
		type = SLJIT_JUMP | (type & SLJIT_REWRITABLE_JUMP);
	}

#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE) \
		|| (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	compiler->skip_checks = 1;
#endif

	return sljit_emit_jump(compiler, type);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_ijump(struct sljit_compiler *compiler, sljit_s32 type, sljit_s32 src, sljit_sw srcw)
{
	sljit_gpr src_r = FAST_IS_REG(src) ? gpr(src) : tmp1;

	CHECK_ERROR();
	CHECK(check_sljit_emit_ijump(compiler, type, src, srcw));

	if (src & SLJIT_IMM) {
		SLJIT_ASSERT(!(srcw & 1)); /* target address must be even */
		FAIL_IF(push_load_imm_inst(compiler, src_r, srcw));
	}
	else if (src & SLJIT_MEM) {
		ADJUST_LOCAL_OFFSET(src, srcw);
		FAIL_IF(load_word(compiler, src_r, src, srcw, 0 /* 64-bit */));
	}

	/* emit jump instruction */
	if (type >= SLJIT_FAST_CALL)
		return push_inst(compiler, basr(type == SLJIT_FAST_CALL ? fast_link_r : link_r, src_r));

	return push_inst(compiler, br(src_r));
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_icall(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 arg_types,
	sljit_s32 src, sljit_sw srcw)
{
	CHECK_ERROR();
	CHECK(check_sljit_emit_icall(compiler, type, arg_types, src, srcw));

	SLJIT_ASSERT(gpr(TMP_REG2) == tmp1);

	if (src & SLJIT_MEM) {
		ADJUST_LOCAL_OFFSET(src, srcw);
		FAIL_IF(load_word(compiler, tmp1, src, srcw, 0 /* 64-bit */));
		src = TMP_REG2;
	}

	if (type & SLJIT_CALL_RETURN) {
		if (src >= SLJIT_FIRST_SAVED_REG && src <= SLJIT_S0) {
			FAIL_IF(push_inst(compiler, lgr(tmp1, gpr(src))));
			src = TMP_REG2;
		}

		FAIL_IF(emit_stack_frame_release(compiler));
		type = SLJIT_JUMP;
	}

#if (defined SLJIT_VERBOSE && SLJIT_VERBOSE) \
		|| (defined SLJIT_ARGUMENT_CHECKS && SLJIT_ARGUMENT_CHECKS)
	compiler->skip_checks = 1;
#endif

	return sljit_emit_ijump(compiler, type, src, srcw);
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_op_flags(struct sljit_compiler *compiler, sljit_s32 op,
	sljit_s32 dst, sljit_sw dstw,
	sljit_s32 type)
{
	sljit_u8 mask = get_cc(compiler, type & 0xff);

	CHECK_ERROR();
	CHECK(check_sljit_emit_op_flags(compiler, op, dst, dstw, type));

	sljit_gpr dst_r = FAST_IS_REG(dst) ? gpr(dst & REG_MASK) : tmp0;
	sljit_gpr loc_r = tmp1;
	switch (GET_OPCODE(op)) {
	case SLJIT_AND:
	case SLJIT_OR:
	case SLJIT_XOR:
		compiler->status_flags_state = op & SLJIT_SET_Z;

		/* dst is also source operand */
		if (dst & SLJIT_MEM)
			FAIL_IF(load_word(compiler, dst_r, dst, dstw, op & SLJIT_32));

		break;
	case SLJIT_MOV32:
		op |= SLJIT_32;
		/* fallthrough */
	case SLJIT_MOV:
		/* can write straight into destination */
		loc_r = dst_r;
		break;
	default:
		SLJIT_UNREACHABLE();
	}

	/* TODO(mundaym): fold into cmov helper function? */
	#define LEVAL(i) i(loc_r, 1, mask)
	if (have_lscond2()) {
		FAIL_IF(push_load_imm_inst(compiler, loc_r, 0));
		FAIL_IF(push_inst(compiler,
			WHEN2(op & SLJIT_32, lochi, locghi)));
	} else {
		/* TODO(mundaym): no load/store-on-condition 2 facility (ipm? branch-and-set?) */
		abort();
	}
	#undef LEVAL

	/* apply bitwise op and set condition codes */
	switch (GET_OPCODE(op)) {
	#define LEVAL(i) i(dst_r, loc_r)
	case SLJIT_AND:
		FAIL_IF(push_inst(compiler,
			WHEN2(op & SLJIT_32, nr, ngr)));
		break;
	case SLJIT_OR:
		FAIL_IF(push_inst(compiler,
			WHEN2(op & SLJIT_32, or, ogr)));
		break;
	case SLJIT_XOR:
		FAIL_IF(push_inst(compiler,
			WHEN2(op & SLJIT_32, xr, xgr)));
		break;
	#undef LEVAL
	}

	/* store result to memory if required */
	if (dst & SLJIT_MEM)
		return store_word(compiler, dst_r, dst, dstw, (op & SLJIT_32));

	return SLJIT_SUCCESS;
}

SLJIT_API_FUNC_ATTRIBUTE sljit_s32 sljit_emit_cmov(struct sljit_compiler *compiler, sljit_s32 type,
	sljit_s32 dst_reg,
	sljit_s32 src, sljit_sw srcw)
{
	sljit_u8 mask = get_cc(compiler, type & 0xff);
	sljit_gpr dst_r = gpr(dst_reg & ~SLJIT_32);
	sljit_gpr src_r = FAST_IS_REG(src) ? gpr(src) : tmp0;

	CHECK_ERROR();
	CHECK(check_sljit_emit_cmov(compiler, type, dst_reg, src, srcw));

	if (src & SLJIT_IMM) {
		/* TODO(mundaym): fast path with lscond2 */
		FAIL_IF(push_load_imm_inst(compiler, src_r, srcw));
	}

	#define LEVAL(i) i(dst_r, src_r, mask)
	if (have_lscond1())
		return push_inst(compiler,
			WHEN2(dst_reg & SLJIT_32, locr, locgr));

	#undef LEVAL

	/* TODO(mundaym): implement */
	return SLJIT_ERR_UNSUPPORTED;
}

/* --------------------------------------------------------------------- */
/*  Other instructions                                                   */
/* --------------------------------------------------------------------- */

/* On s390x we build a literal pool to hold constants. This has two main
   advantages:

     1. we only need one instruction in the instruction stream (LGRL)
     2. we can store 64 bit addresses and use 32 bit offsets

   To retrofit the extra information needed to build the literal pool we
   add a new sljit_s390x_const struct that contains the initial value but
   can still be cast to a sljit_const. */

SLJIT_API_FUNC_ATTRIBUTE struct sljit_const* sljit_emit_const(struct sljit_compiler *compiler, sljit_s32 dst, sljit_sw dstw, sljit_sw init_value)
{
	struct sljit_s390x_const *const_;
	sljit_gpr dst_r;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_const(compiler, dst, dstw, init_value));

	const_ = (struct sljit_s390x_const*)ensure_abuf(compiler,
					sizeof(struct sljit_s390x_const));
	PTR_FAIL_IF(!const_);
	set_const((struct sljit_const*)const_, compiler);
	const_->init_value = init_value;

	dst_r = FAST_IS_REG(dst) ? gpr(dst & REG_MASK) : tmp0;
	if (have_genext())
		PTR_FAIL_IF(push_inst(compiler, sljit_ins_const | lgrl(dst_r, 0)));
	else {
		PTR_FAIL_IF(push_inst(compiler, sljit_ins_const | larl(tmp1, 0)));
		PTR_FAIL_IF(push_inst(compiler, lg(dst_r, 0, r0, tmp1)));
	}

	if (dst & SLJIT_MEM)
		PTR_FAIL_IF(store_word(compiler, dst_r, dst, dstw, 0 /* always 64-bit */));

	return (struct sljit_const*)const_;
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_jump_addr(sljit_uw addr, sljit_uw new_target, sljit_sw executable_offset)
{
	/* Update the constant pool. */
	sljit_uw *ptr = (sljit_uw *)addr;
	SLJIT_UNUSED_ARG(executable_offset);

	SLJIT_UPDATE_WX_FLAGS(ptr, ptr + 1, 0);
	*ptr = new_target;
	SLJIT_UPDATE_WX_FLAGS(ptr, ptr + 1, 1);
	SLJIT_CACHE_FLUSH(ptr, ptr + 1);
}

SLJIT_API_FUNC_ATTRIBUTE void sljit_set_const(sljit_uw addr, sljit_sw new_constant, sljit_sw executable_offset)
{
	sljit_set_jump_addr(addr, (sljit_uw)new_constant, executable_offset);
}

SLJIT_API_FUNC_ATTRIBUTE struct sljit_put_label *sljit_emit_put_label(
	struct sljit_compiler *compiler,
	sljit_s32 dst, sljit_sw dstw)
{
	struct sljit_put_label *put_label;
	sljit_gpr dst_r;

	CHECK_ERROR_PTR();
	CHECK_PTR(check_sljit_emit_put_label(compiler, dst, dstw));
	ADJUST_LOCAL_OFFSET(dst, dstw);

	put_label = (struct sljit_put_label*)ensure_abuf(compiler, sizeof(struct sljit_put_label));
	PTR_FAIL_IF(!put_label);
	set_put_label(put_label, compiler, 0);

	dst_r = FAST_IS_REG(dst) ? gpr(dst & REG_MASK) : tmp0;

	if (have_genext())
		PTR_FAIL_IF(push_inst(compiler, lgrl(dst_r, 0)));
	else {
		PTR_FAIL_IF(push_inst(compiler, larl(tmp1, 0)));
		PTR_FAIL_IF(push_inst(compiler, lg(dst_r, 0, r0, tmp1)));
	}

	if (dst & SLJIT_MEM)
		PTR_FAIL_IF(store_word(compiler, dst_r, dst, dstw, 0));

	return put_label;
}

/* TODO(carenas): EVAL probably should move up or be refactored */
#undef WHEN2
#undef EVAL

#undef tmp1
#undef tmp0

/* TODO(carenas): undef other macros that spill like is_u12? */
