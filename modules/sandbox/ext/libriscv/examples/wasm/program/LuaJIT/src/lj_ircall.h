/*
** IR CALL* instruction definitions.
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
*/

#ifndef _LJ_IRCALL_H
#define _LJ_IRCALL_H

#include "lj_obj.h"
#include "lj_ir.h"
#include "lj_jit.h"

/* C call info for CALL* instructions. */
typedef struct CCallInfo {
  ASMFunction func;		/* Function pointer. */
  uint32_t flags;		/* Number of arguments and flags. */
} CCallInfo;

#define CCI_NARGS(ci)		((ci)->flags & 0xff)	/* # of args. */
#define CCI_NARGS_MAX		32			/* Max. # of args. */

#define CCI_OTSHIFT		16
#define CCI_OPTYPE(ci)		((ci)->flags >> CCI_OTSHIFT)  /* Get op/type. */
#define CCI_TYPE(ci)		(((ci)->flags>>CCI_OTSHIFT) & IRT_TYPE)
#define CCI_OPSHIFT		24
#define CCI_OP(ci)		((ci)->flags >> CCI_OPSHIFT)  /* Get op. */

#define CCI_CALL_N		(IR_CALLN << CCI_OPSHIFT)
#define CCI_CALL_A		(IR_CALLA << CCI_OPSHIFT)
#define CCI_CALL_L		(IR_CALLL << CCI_OPSHIFT)
#define CCI_CALL_S		(IR_CALLS << CCI_OPSHIFT)
#define CCI_CALL_FN		(CCI_CALL_N|CCI_CC_FASTCALL)
#define CCI_CALL_FA		(CCI_CALL_A|CCI_CC_FASTCALL)
#define CCI_CALL_FL		(CCI_CALL_L|CCI_CC_FASTCALL)
#define CCI_CALL_FS		(CCI_CALL_S|CCI_CC_FASTCALL)

/* C call info flags. */
#define CCI_T			(IRT_GUARD << CCI_OTSHIFT)  /* May throw. */
#define CCI_L			0x0100	/* Implicit L arg. */
#define CCI_CASTU64		0x0200	/* Cast u64 result to number. */
#define CCI_NOFPRCLOBBER	0x0400	/* Does not clobber any FPRs. */
#define CCI_VARARG		0x0800	/* Vararg function. */

#define CCI_CC_MASK		0x3000	/* Calling convention mask. */
#define CCI_CC_SHIFT		12
/* ORDER CC */
#define CCI_CC_CDECL		0x0000	/* Default cdecl calling convention. */
#define CCI_CC_THISCALL		0x1000	/* Thiscall calling convention. */
#define CCI_CC_FASTCALL		0x2000	/* Fastcall calling convention. */
#define CCI_CC_STDCALL		0x3000	/* Stdcall calling convention. */

/* Extra args for SOFTFP, SPLIT 64 bit. */
#define CCI_XARGS_SHIFT		14
#define CCI_XARGS(ci)		(((ci)->flags >> CCI_XARGS_SHIFT) & 3)
#define CCI_XA			(1u << CCI_XARGS_SHIFT)

#if LJ_SOFTFP32 || (LJ_32 && LJ_HASFFI)
#define CCI_XNARGS(ci)		(CCI_NARGS((ci)) + CCI_XARGS((ci)))
#else
#define CCI_XNARGS(ci)		CCI_NARGS((ci))
#endif

/* Helpers for conditional function definitions. */
#define IRCALLCOND_ANY(x)		x

#if LJ_TARGET_X86ORX64 || LJ_TARGET_ARM64
#define IRCALLCOND_FPMATH(x)		NULL
#else
#define IRCALLCOND_FPMATH(x)		x
#endif

#if LJ_SOFTFP
#define IRCALLCOND_SOFTFP(x)		x
#if LJ_HASFFI
#define IRCALLCOND_SOFTFP_FFI(x)	x
#else
#define IRCALLCOND_SOFTFP_FFI(x)	NULL
#endif
#else
#define IRCALLCOND_SOFTFP(x)		NULL
#define IRCALLCOND_SOFTFP_FFI(x)	NULL
#endif

#if LJ_SOFTFP && LJ_TARGET_MIPS
#define IRCALLCOND_SOFTFP_MIPS(x)	x
#else
#define IRCALLCOND_SOFTFP_MIPS(x)	NULL
#endif

#if LJ_SOFTFP && LJ_TARGET_MIPS64
#define IRCALLCOND_SOFTFP_MIPS64(x)	x
#else
#define IRCALLCOND_SOFTFP_MIPS64(x)	NULL
#endif

#define LJ_NEED_FP64	(LJ_TARGET_ARM || LJ_TARGET_PPC || LJ_TARGET_MIPS)

#if LJ_HASFFI && (LJ_SOFTFP || LJ_NEED_FP64)
#define IRCALLCOND_FP64_FFI(x)		x
#else
#define IRCALLCOND_FP64_FFI(x)		NULL
#endif

#if LJ_HASFFI
#define IRCALLCOND_FFI(x)		x
#if LJ_32
#define IRCALLCOND_FFI32(x)		x
#else
#define IRCALLCOND_FFI32(x)		NULL
#endif
#else
#define IRCALLCOND_FFI(x)		NULL
#define IRCALLCOND_FFI32(x)		NULL
#endif

#if LJ_HASBUFFER
#define IRCALLCOND_BUFFER(x)		x
#else
#define IRCALLCOND_BUFFER(x)		NULL
#endif

#if LJ_HASBUFFER && LJ_HASFFI
#define IRCALLCOND_BUFFFI(x)		x
#else
#define IRCALLCOND_BUFFFI(x)		NULL
#endif

#if LJ_SOFTFP
#define XA_FP		CCI_XA
#define XA2_FP		(CCI_XA+CCI_XA)
#else
#define XA_FP		0
#define XA2_FP		0
#endif

#if LJ_SOFTFP32
#define XA_FP32		CCI_XA
#define XA2_FP32	(CCI_XA+CCI_XA)
#else
#define XA_FP32		0
#define XA2_FP32	0
#endif

#if LJ_32
#define XA_64		CCI_XA
#define XA2_64		(CCI_XA+CCI_XA)
#else
#define XA_64		0
#define XA2_64		0
#endif

/* Function definitions for CALL* instructions. */
#define IRCALLDEF(_) \
  _(ANY,	lj_str_cmp,		2,  FN, INT, CCI_NOFPRCLOBBER) \
  _(ANY,	lj_str_find,		4,   N, PGC, 0) \
  _(ANY,	lj_str_new,		3,   S, STR, CCI_L|CCI_T) \
  _(ANY,	lj_strscan_num,		2,  FN, INT, 0) \
  _(ANY,	lj_strfmt_int,		2,  FN, STR, CCI_L|CCI_T) \
  _(ANY,	lj_strfmt_num,		2,  FN, STR, CCI_L|CCI_T) \
  _(ANY,	lj_strfmt_char,		2,  FN, STR, CCI_L|CCI_T) \
  _(ANY,	lj_strfmt_putint,	2,  FL, PGC, CCI_T) \
  _(ANY,	lj_strfmt_putnum,	2,  FL, PGC, CCI_T) \
  _(ANY,	lj_strfmt_putquoted,	2,  FL, PGC, CCI_T) \
  _(ANY,	lj_strfmt_putfxint,	3,   L, PGC, XA_64|CCI_T) \
  _(ANY,	lj_strfmt_putfnum_int,	3,   L, PGC, XA_FP|CCI_T) \
  _(ANY,	lj_strfmt_putfnum_uint,	3,   L, PGC, XA_FP|CCI_T) \
  _(ANY,	lj_strfmt_putfnum,	3,   L, PGC, XA_FP|CCI_T) \
  _(ANY,	lj_strfmt_putfstr,	3,   L, PGC, CCI_T) \
  _(ANY,	lj_strfmt_putfchar,	3,   L, PGC, CCI_T) \
  _(ANY,	lj_buf_putmem,		3,   S, PGC, CCI_T) \
  _(ANY,	lj_buf_putstr,		2,  FL, PGC, CCI_T) \
  _(ANY,	lj_buf_putchar,		2,  FL, PGC, CCI_T) \
  _(ANY,	lj_buf_putstr_reverse,	2,  FL, PGC, CCI_T) \
  _(ANY,	lj_buf_putstr_lower,	2,  FL, PGC, CCI_T) \
  _(ANY,	lj_buf_putstr_upper,	2,  FL, PGC, CCI_T) \
  _(ANY,	lj_buf_putstr_rep,	3,   L, PGC, CCI_T) \
  _(ANY,	lj_buf_puttab,		5,   L, PGC, CCI_T) \
  _(BUFFER,	lj_bufx_set,		4,   S, NIL, 0) \
  _(BUFFFI,	lj_bufx_more,		2,  FS, INT, CCI_T) \
  _(BUFFER,	lj_serialize_put,	2,  FS, PGC, CCI_T) \
  _(BUFFER,	lj_serialize_get,	2,  FS, PTR, CCI_T) \
  _(BUFFER,	lj_serialize_encode,	2,  FA, STR, CCI_L|CCI_T) \
  _(BUFFER,	lj_serialize_decode,	3,   A, INT, CCI_L|CCI_T) \
  _(ANY,	lj_buf_tostr,		1,  FL, STR, CCI_T) \
  _(ANY,	lj_tab_new_ah,		3,   A, TAB, CCI_L|CCI_T) \
  _(ANY,	lj_tab_new1,		2,  FA, TAB, CCI_L|CCI_T) \
  _(ANY,	lj_tab_dup,		2,  FA, TAB, CCI_L|CCI_T) \
  _(ANY,	lj_tab_clear,		1,  FS, NIL, 0) \
  _(ANY,	lj_tab_newkey,		3,   S, PGC, CCI_L|CCI_T) \
  _(ANY,	lj_tab_keyindex,	2,  FL, INT, 0) \
  _(ANY,	lj_vm_next,		2,  FL, PTR, 0) \
  _(ANY,	lj_tab_len,		1,  FL, INT, 0) \
  _(ANY,	lj_tab_len_hint,	2,  FL, INT, 0) \
  _(ANY,	lj_gc_step_jit,		2,  FS, NIL, CCI_L) \
  _(ANY,	lj_gc_barrieruv,	2,  FS, NIL, 0) \
  _(ANY,	lj_mem_newgco,		2,  FA, PGC, CCI_L|CCI_T) \
  _(ANY,	lj_prng_u64d,		1,  FS, NUM, CCI_CASTU64) \
  _(ANY,	lj_vm_modi,		2,  FN, INT, 0) \
  _(ANY,	log10,			1,   N, NUM, XA_FP) \
  _(ANY,	exp,			1,   N, NUM, XA_FP) \
  _(ANY,	sin,			1,   N, NUM, XA_FP) \
  _(ANY,	cos,			1,   N, NUM, XA_FP) \
  _(ANY,	tan,			1,   N, NUM, XA_FP) \
  _(ANY,	asin,			1,   N, NUM, XA_FP) \
  _(ANY,	acos,			1,   N, NUM, XA_FP) \
  _(ANY,	atan,			1,   N, NUM, XA_FP) \
  _(ANY,	sinh,			1,   N, NUM, XA_FP) \
  _(ANY,	cosh,			1,   N, NUM, XA_FP) \
  _(ANY,	tanh,			1,   N, NUM, XA_FP) \
  _(ANY,	fputc,			2,   S, INT, 0) \
  _(ANY,	fwrite,			4,   S, INT, 0) \
  _(ANY,	fflush,			1,   S, INT, 0) \
  /* ORDER FPM */ \
  _(FPMATH,	lj_vm_floor,		1,   N, NUM, XA_FP) \
  _(FPMATH,	lj_vm_ceil,		1,   N, NUM, XA_FP) \
  _(FPMATH,	lj_vm_trunc,		1,   N, NUM, XA_FP) \
  _(FPMATH,	sqrt,			1,   N, NUM, XA_FP) \
  _(ANY,	log,			1,   N, NUM, XA_FP) \
  _(ANY,	lj_vm_log2,		1,   N, NUM, XA_FP) \
  _(ANY,	pow,			2,   N, NUM, XA2_FP) \
  _(ANY,	atan2,			2,   N, NUM, XA2_FP) \
  _(ANY,	ldexp,			2,   N, NUM, XA_FP) \
  _(SOFTFP,	lj_vm_tobit,		1,   N, INT, XA_FP32) \
  _(SOFTFP,	softfp_add,		2,   N, NUM, XA2_FP32) \
  _(SOFTFP,	softfp_sub,		2,   N, NUM, XA2_FP32) \
  _(SOFTFP,	softfp_mul,		2,   N, NUM, XA2_FP32) \
  _(SOFTFP,	softfp_div,		2,   N, NUM, XA2_FP32) \
  _(SOFTFP,	softfp_cmp,		2,   N, NIL, XA2_FP32) \
  _(SOFTFP,	softfp_i2d,		1,   N, NUM, 0) \
  _(SOFTFP,	softfp_d2i,		1,   N, INT, XA_FP32) \
  _(SOFTFP_MIPS, lj_vm_sfmin,		2,   N, NUM, XA2_FP32) \
  _(SOFTFP_MIPS, lj_vm_sfmax,		2,   N, NUM, XA2_FP32) \
  _(SOFTFP_MIPS64, lj_vm_tointg,	1,   N, INT, 0) \
  _(SOFTFP_FFI,	softfp_ui2d,		1,   N, NUM, 0) \
  _(SOFTFP_FFI,	softfp_f2d,		1,   N, NUM, 0) \
  _(SOFTFP_FFI,	softfp_d2ui,		1,   N, INT, XA_FP32) \
  _(SOFTFP_FFI,	softfp_d2f,		1,   N, FLOAT, XA_FP32) \
  _(SOFTFP_FFI,	softfp_i2f,		1,   N, FLOAT, 0) \
  _(SOFTFP_FFI,	softfp_ui2f,		1,   N, FLOAT, 0) \
  _(SOFTFP_FFI,	softfp_f2i,		1,   N, INT, 0) \
  _(SOFTFP_FFI,	softfp_f2ui,		1,   N, INT, 0) \
  _(FP64_FFI,	fp64_l2d,		1,   N, NUM, XA_64) \
  _(FP64_FFI,	fp64_ul2d,		1,   N, NUM, XA_64) \
  _(FP64_FFI,	fp64_l2f,		1,   N, FLOAT, XA_64) \
  _(FP64_FFI,	fp64_ul2f,		1,   N, FLOAT, XA_64) \
  _(FP64_FFI,	fp64_d2l,		1,   N, I64, XA_FP) \
  _(FP64_FFI,	fp64_d2ul,		1,   N, U64, XA_FP) \
  _(FP64_FFI,	fp64_f2l,		1,   N, I64, 0) \
  _(FP64_FFI,	fp64_f2ul,		1,   N, U64, 0) \
  _(FFI,	lj_carith_divi64,	2,   N, I64, XA2_64|CCI_NOFPRCLOBBER) \
  _(FFI,	lj_carith_divu64,	2,   N, U64, XA2_64|CCI_NOFPRCLOBBER) \
  _(FFI,	lj_carith_modi64,	2,   N, I64, XA2_64|CCI_NOFPRCLOBBER) \
  _(FFI,	lj_carith_modu64,	2,   N, U64, XA2_64|CCI_NOFPRCLOBBER) \
  _(FFI,	lj_carith_powi64,	2,   N, I64, XA2_64|CCI_NOFPRCLOBBER) \
  _(FFI,	lj_carith_powu64,	2,   N, U64, XA2_64|CCI_NOFPRCLOBBER) \
  _(FFI,	lj_cdata_newv,		4,   S, CDATA, CCI_L) \
  _(FFI,	lj_cdata_setfin,	4,   S, NIL, CCI_L) \
  _(FFI,	strlen,			1,   L, INTP, 0) \
  _(FFI,	memcpy,			3,   S, PTR, 0) \
  _(FFI,	memset,			3,   S, PTR, 0) \
  _(FFI,	lj_vm_errno,		0,   S, INT, CCI_NOFPRCLOBBER) \
  _(FFI32,	lj_carith_mul64,	2,   N, I64, XA2_64|CCI_NOFPRCLOBBER) \
  _(FFI32,	lj_carith_shl64,	2,   N, U64, XA_64|CCI_NOFPRCLOBBER) \
  _(FFI32,	lj_carith_shr64,	2,   N, U64, XA_64|CCI_NOFPRCLOBBER) \
  _(FFI32,	lj_carith_sar64,	2,   N, U64, XA_64|CCI_NOFPRCLOBBER) \
  _(FFI32,	lj_carith_rol64,	2,   N, U64, XA_64|CCI_NOFPRCLOBBER) \
  _(FFI32,	lj_carith_ror64,	2,   N, U64, XA_64|CCI_NOFPRCLOBBER) \
  \
  /* End of list. */

typedef enum {
#define IRCALLENUM(cond, name, nargs, kind, type, flags)	IRCALL_##name,
IRCALLDEF(IRCALLENUM)
#undef IRCALLENUM
  IRCALL__MAX
} IRCallID;

LJ_FUNC TRef lj_ir_call(jit_State *J, IRCallID id, ...);

LJ_DATA const CCallInfo lj_ir_callinfo[IRCALL__MAX+1];

/* Soft-float declarations. */
#if LJ_SOFTFP
#if LJ_TARGET_ARM
#define softfp_add __aeabi_dadd
#define softfp_sub __aeabi_dsub
#define softfp_mul __aeabi_dmul
#define softfp_div __aeabi_ddiv
#define softfp_cmp __aeabi_cdcmple
#define softfp_i2d __aeabi_i2d
#define softfp_d2i __aeabi_d2iz
#define softfp_ui2d __aeabi_ui2d
#define softfp_f2d __aeabi_f2d
#define softfp_d2ui __aeabi_d2uiz
#define softfp_d2f __aeabi_d2f
#define softfp_i2f __aeabi_i2f
#define softfp_ui2f __aeabi_ui2f
#define softfp_f2i __aeabi_f2iz
#define softfp_f2ui __aeabi_f2uiz
#define fp64_l2d __aeabi_l2d
#define fp64_ul2d __aeabi_ul2d
#define fp64_l2f __aeabi_l2f
#define fp64_ul2f __aeabi_ul2f
#if LJ_TARGET_IOS
#define fp64_d2l __fixdfdi
#define fp64_d2ul __fixunsdfdi
#define fp64_f2l __fixsfdi
#define fp64_f2ul __fixunssfdi
#else
#define fp64_d2l __aeabi_d2lz
#define fp64_d2ul __aeabi_d2ulz
#define fp64_f2l __aeabi_f2lz
#define fp64_f2ul __aeabi_f2ulz
#endif
#elif LJ_TARGET_MIPS || LJ_TARGET_PPC
#define softfp_add __adddf3
#define softfp_sub __subdf3
#define softfp_mul __muldf3
#define softfp_div __divdf3
#define softfp_cmp __ledf2
#define softfp_i2d __floatsidf
#define softfp_d2i __fixdfsi
#define softfp_ui2d __floatunsidf
#define softfp_f2d __extendsfdf2
#define softfp_d2ui __fixunsdfsi
#define softfp_d2f __truncdfsf2
#define softfp_i2f __floatsisf
#define softfp_ui2f __floatunsisf
#define softfp_f2i __fixsfsi
#define softfp_f2ui __fixunssfsi
#else
#error "Missing soft-float definitions for target architecture"
#endif
extern double softfp_add(double a, double b);
extern double softfp_sub(double a, double b);
extern double softfp_mul(double a, double b);
extern double softfp_div(double a, double b);
extern void softfp_cmp(double a, double b);
extern double softfp_i2d(int32_t a);
extern int32_t softfp_d2i(double a);
#if LJ_HASFFI
extern double softfp_ui2d(uint32_t a);
extern double softfp_f2d(float a);
extern uint32_t softfp_d2ui(double a);
extern float softfp_d2f(double a);
extern float softfp_i2f(int32_t a);
extern float softfp_ui2f(uint32_t a);
extern int32_t softfp_f2i(float a);
extern uint32_t softfp_f2ui(float a);
#endif
#if LJ_TARGET_MIPS
extern double lj_vm_sfmin(double a, double b);
extern double lj_vm_sfmax(double a, double b);
#endif
#endif

#if LJ_HASFFI && LJ_NEED_FP64 && !(LJ_TARGET_ARM && LJ_SOFTFP)
#if defined(__GNUC__) || defined(__clang__)
#define fp64_l2d __floatdidf
#define fp64_ul2d __floatundidf
#define fp64_l2f __floatdisf
#define fp64_ul2f __floatundisf
#define fp64_d2l __fixdfdi
#define fp64_d2ul __fixunsdfdi
#define fp64_f2l __fixsfdi
#define fp64_f2ul __fixunssfdi
#else
#error "Missing fp64 helper definitions for this compiler"
#endif
#endif

#if LJ_HASFFI && (LJ_SOFTFP || LJ_NEED_FP64)
extern double fp64_l2d(int64_t a);
extern double fp64_ul2d(uint64_t a);
extern float fp64_l2f(int64_t a);
extern float fp64_ul2f(uint64_t a);
extern int64_t fp64_d2l(double a);
extern uint64_t fp64_d2ul(double a);
extern int64_t fp64_f2l(float a);
extern uint64_t fp64_f2ul(float a);
#endif

#endif
