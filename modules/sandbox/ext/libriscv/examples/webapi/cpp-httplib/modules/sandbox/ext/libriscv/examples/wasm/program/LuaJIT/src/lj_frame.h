/*
** Stack frames.
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
*/

#ifndef _LJ_FRAME_H
#define _LJ_FRAME_H

#include "lj_obj.h"
#include "lj_bc.h"

/* -- Lua stack frame ----------------------------------------------------- */

/* Frame type markers in LSB of PC (4-byte aligned) or delta (8-byte aligned:
**
**    PC  00  Lua frame
** delta 001  C frame
** delta 010  Continuation frame
** delta 011  Lua vararg frame
** delta 101  cpcall() frame
** delta 110  ff pcall() frame
** delta 111  ff pcall() frame with active hook
*/
enum {
  FRAME_LUA, FRAME_C, FRAME_CONT, FRAME_VARG,
  FRAME_LUAP, FRAME_CP, FRAME_PCALL, FRAME_PCALLH
};
#define FRAME_TYPE		3
#define FRAME_P			4
#define FRAME_TYPEP		(FRAME_TYPE|FRAME_P)

/* Macros to access and modify Lua frames. */
#if LJ_FR2
/* Two-slot frame info, required for 64 bit PC/GCRef:
**
**                   base-2  base-1      |  base  base+1 ...
**                  [func   PC/delta/ft] | [slots ...]
**                  ^-- frame            | ^-- base   ^-- top
**
** Continuation frames:
**
**   base-4  base-3  base-2  base-1      |  base  base+1 ...
**  [cont      PC ] [func   PC/delta/ft] | [slots ...]
**                  ^-- frame            | ^-- base   ^-- top
*/
#define frame_gc(f)		(gcval((f)-1))
#define frame_ftsz(f)		((ptrdiff_t)(f)->ftsz)
#define frame_pc(f)		((const BCIns *)frame_ftsz(f))
#define setframe_gc(f, p, tp)	(setgcVraw((f), (p), (tp)))
#define setframe_ftsz(f, sz)	((f)->ftsz = (sz))
#define setframe_pc(f, pc)	((f)->ftsz = (int64_t)(intptr_t)(pc))
#else
/* One-slot frame info, sufficient for 32 bit PC/GCRef:
**
**              base-1              |  base  base+1 ...
**              lo     hi           |
**             [func | PC/delta/ft] | [slots ...]
**             ^-- frame            | ^-- base   ^-- top
**
** Continuation frames:
**
**  base-2      base-1              |  base  base+1 ...
**  lo     hi   lo     hi           |
** [cont | PC] [func | PC/delta/ft] | [slots ...]
**             ^-- frame            | ^-- base   ^-- top
*/
#define frame_gc(f)		(gcref((f)->fr.func))
#define frame_ftsz(f)		((ptrdiff_t)(f)->fr.tp.ftsz)
#define frame_pc(f)		(mref((f)->fr.tp.pcr, const BCIns))
#define setframe_gc(f, p, tp)	(setgcref((f)->fr.func, (p)), UNUSED(tp))
#define setframe_ftsz(f, sz)	((f)->fr.tp.ftsz = (int32_t)(sz))
#define setframe_pc(f, pc)	(setmref((f)->fr.tp.pcr, (pc)))
#endif

#define frame_type(f)		(frame_ftsz(f) & FRAME_TYPE)
#define frame_typep(f)		(frame_ftsz(f) & FRAME_TYPEP)
#define frame_islua(f)		(frame_type(f) == FRAME_LUA)
#define frame_isc(f)		(frame_type(f) == FRAME_C)
#define frame_iscont(f)		(frame_typep(f) == FRAME_CONT)
#define frame_isvarg(f)		(frame_typep(f) == FRAME_VARG)
#define frame_ispcall(f)	((frame_ftsz(f) & 6) == FRAME_PCALL)

#define frame_func(f)		(&frame_gc(f)->fn)
#define frame_delta(f)		(frame_ftsz(f) >> 3)
#define frame_sized(f)		(frame_ftsz(f) & ~FRAME_TYPEP)

enum { LJ_CONT_TAILCALL, LJ_CONT_FFI_CALLBACK };  /* Special continuations. */

#if LJ_FR2
#define frame_contpc(f)		(frame_pc((f)-2))
#define frame_contv(f)		(((f)-3)->u64)
#else
#define frame_contpc(f)		(frame_pc((f)-1))
#define frame_contv(f)		(((f)-1)->u32.lo)
#endif
#if LJ_FR2
#define frame_contf(f)		((ASMFunction)(uintptr_t)((f)-3)->u64)
#elif LJ_64
#define frame_contf(f) \
  ((ASMFunction)(void *)((intptr_t)lj_vm_asm_begin + \
			 (intptr_t)(int32_t)((f)-1)->u32.lo))
#else
#define frame_contf(f)		((ASMFunction)gcrefp(((f)-1)->gcr, void))
#endif
#define frame_iscont_fficb(f) \
  (LJ_HASFFI && frame_contv(f) == LJ_CONT_FFI_CALLBACK)

#define frame_prevl(f)		((f) - (1+LJ_FR2+bc_a(frame_pc(f)[-1])))
#define frame_prevd(f)		((TValue *)((char *)(f) - frame_sized(f)))
#define frame_prev(f)		(frame_islua(f)?frame_prevl(f):frame_prevd(f))
/* Note: this macro does not skip over FRAME_VARG. */

/* -- C stack frame ------------------------------------------------------- */

/* Macros to access and modify the C stack frame chain. */

/* These definitions must match with the arch-specific *.dasc files. */
#if LJ_TARGET_X86
#if LJ_ABI_WIN
#define CFRAME_OFS_ERRF		(19*4)
#define CFRAME_OFS_NRES		(18*4)
#define CFRAME_OFS_PREV		(17*4)
#define CFRAME_OFS_L		(16*4)
#define CFRAME_OFS_SEH		(9*4)
#define CFRAME_OFS_PC		(6*4)
#define CFRAME_OFS_MULTRES	(5*4)
#define CFRAME_SIZE		(16*4)
#define CFRAME_SHIFT_MULTRES	0
#else
#define CFRAME_OFS_ERRF		(15*4)
#define CFRAME_OFS_NRES		(14*4)
#define CFRAME_OFS_PREV		(13*4)
#define CFRAME_OFS_L		(12*4)
#define CFRAME_OFS_PC		(6*4)
#define CFRAME_OFS_MULTRES	(5*4)
#define CFRAME_SIZE		(12*4)
#define CFRAME_SHIFT_MULTRES	0
#endif
#elif LJ_TARGET_X64
#if LJ_ABI_WIN
#define CFRAME_OFS_PREV		(13*8)
#if LJ_GC64
#define CFRAME_OFS_PC		(12*8)
#define CFRAME_OFS_L		(11*8)
#define CFRAME_OFS_ERRF		(21*4)
#define CFRAME_OFS_NRES		(20*4)
#define CFRAME_OFS_MULTRES	(8*4)
#else
#define CFRAME_OFS_PC		(25*4)
#define CFRAME_OFS_L		(24*4)
#define CFRAME_OFS_ERRF		(23*4)
#define CFRAME_OFS_NRES		(22*4)
#define CFRAME_OFS_MULTRES	(21*4)
#endif
#define CFRAME_SIZE		(10*8)
#define CFRAME_SIZE_JIT		(CFRAME_SIZE + 9*16 + 4*8)
#define CFRAME_SHIFT_MULTRES	0
#else
#define CFRAME_OFS_PREV		(4*8)
#if LJ_GC64
#define CFRAME_OFS_PC		(3*8)
#define CFRAME_OFS_L		(2*8)
#define CFRAME_OFS_ERRF		(3*4)
#define CFRAME_OFS_NRES		(2*4)
#define CFRAME_OFS_MULTRES	(0*4)
#else
#define CFRAME_OFS_PC		(7*4)
#define CFRAME_OFS_L		(6*4)
#define CFRAME_OFS_ERRF		(5*4)
#define CFRAME_OFS_NRES		(4*4)
#define CFRAME_OFS_MULTRES	(1*4)
#endif
#if LJ_NO_UNWIND
#define CFRAME_SIZE		(12*8)
#else
#define CFRAME_SIZE		(10*8)
#endif
#define CFRAME_SIZE_JIT		(CFRAME_SIZE + 16)
#define CFRAME_SHIFT_MULTRES	0
#endif
#elif LJ_TARGET_ARM
#define CFRAME_OFS_ERRF		24
#define CFRAME_OFS_NRES		20
#define CFRAME_OFS_PREV		16
#define CFRAME_OFS_L		12
#define CFRAME_OFS_PC		8
#define CFRAME_OFS_MULTRES	4
#if LJ_ARCH_HASFPU
#define CFRAME_SIZE		128
#else
#define CFRAME_SIZE		64
#endif
#define CFRAME_SHIFT_MULTRES	3
#elif LJ_TARGET_ARM64
#define CFRAME_OFS_ERRF		36
#define CFRAME_OFS_NRES		40
#define CFRAME_OFS_PREV		0
#define CFRAME_OFS_L		16
#define CFRAME_OFS_PC		8
#define CFRAME_OFS_MULTRES	32
#define CFRAME_SIZE		208
#define CFRAME_SHIFT_MULTRES	3
#elif LJ_TARGET_PPC
#if LJ_TARGET_XBOX360
#define CFRAME_OFS_ERRF		424
#define CFRAME_OFS_NRES		420
#define CFRAME_OFS_PREV		400
#define CFRAME_OFS_L		416
#define CFRAME_OFS_PC		412
#define CFRAME_OFS_MULTRES	408
#define CFRAME_SIZE		384
#define CFRAME_SHIFT_MULTRES	3
#elif LJ_ARCH_PPC32ON64
#define CFRAME_OFS_ERRF		472
#define CFRAME_OFS_NRES		468
#define CFRAME_OFS_PREV		448
#define CFRAME_OFS_L		464
#define CFRAME_OFS_PC		460
#define CFRAME_OFS_MULTRES	456
#define CFRAME_SIZE		400
#define CFRAME_SHIFT_MULTRES	3
#else
#define CFRAME_OFS_ERRF		48
#define CFRAME_OFS_NRES		44
#define CFRAME_OFS_PREV		40
#define CFRAME_OFS_L		36
#define CFRAME_OFS_PC		32
#define CFRAME_OFS_MULTRES	28
#define CFRAME_SIZE		(LJ_ARCH_HASFPU ? 272 : 128)
#define CFRAME_SHIFT_MULTRES	3
#endif
#elif LJ_TARGET_MIPS32
#if LJ_ARCH_HASFPU
#define CFRAME_OFS_ERRF		124
#define CFRAME_OFS_NRES		120
#define CFRAME_OFS_PREV		116
#define CFRAME_OFS_L		112
#define CFRAME_SIZE		112
#else
#define CFRAME_OFS_ERRF		76
#define CFRAME_OFS_NRES		72
#define CFRAME_OFS_PREV		68
#define CFRAME_OFS_L		64
#define CFRAME_SIZE		64
#endif
#define CFRAME_OFS_PC		20
#define CFRAME_OFS_MULTRES	16
#define CFRAME_SHIFT_MULTRES	3
#elif LJ_TARGET_MIPS64
#if LJ_ARCH_HASFPU
#define CFRAME_OFS_ERRF		188
#define CFRAME_OFS_NRES		184
#define CFRAME_OFS_PREV		176
#define CFRAME_OFS_L		168
#define CFRAME_OFS_PC		160
#define CFRAME_SIZE		192
#else
#define CFRAME_OFS_ERRF		124
#define CFRAME_OFS_NRES		120
#define CFRAME_OFS_PREV		112
#define CFRAME_OFS_L		104
#define CFRAME_OFS_PC		96
#define CFRAME_SIZE		128
#endif
#define CFRAME_OFS_MULTRES	0
#define CFRAME_SHIFT_MULTRES	3
#elif LJ_TARGET_RISCV64
#define CFRAME_OFS_ERRF		252
#define CFRAME_OFS_NRES		248
#define CFRAME_OFS_PREV		240
#define CFRAME_OFS_L		232
#define CFRAME_OFS_PC		224
#define CFRAME_OFS_MULTRES	0
#define CFRAME_SIZE		256
#define CFRAME_SHIFT_MULTRES	3
#else
#error "Missing CFRAME_* definitions for this architecture"
#endif

#ifndef CFRAME_SIZE_JIT
#define CFRAME_SIZE_JIT		CFRAME_SIZE
#endif

#define CFRAME_RESUME		1
#define CFRAME_UNWIND_FF	2  /* Only used in unwinder. */
#define CFRAME_RAWMASK		(~(intptr_t)(CFRAME_RESUME|CFRAME_UNWIND_FF))

#define cframe_errfunc(cf)	(*(int32_t *)(((char *)(cf))+CFRAME_OFS_ERRF))
#define cframe_nres(cf)		(*(int32_t *)(((char *)(cf))+CFRAME_OFS_NRES))
#define cframe_prev(cf)		(*(void **)(((char *)(cf))+CFRAME_OFS_PREV))
#define cframe_multres(cf)  (*(uint32_t *)(((char *)(cf))+CFRAME_OFS_MULTRES))
#define cframe_multres_n(cf)	(cframe_multres((cf)) >> CFRAME_SHIFT_MULTRES)
#define cframe_L(cf) \
  (&gcref(*(GCRef *)(((char *)(cf))+CFRAME_OFS_L))->th)
#define cframe_pc(cf) \
  (mref(*(MRef *)(((char *)(cf))+CFRAME_OFS_PC), const BCIns))
#define setcframe_L(cf, L) \
  (setmref(*(MRef *)(((char *)(cf))+CFRAME_OFS_L), (L)))
#define setcframe_pc(cf, pc) \
  (setmref(*(MRef *)(((char *)(cf))+CFRAME_OFS_PC), (pc)))
#define cframe_canyield(cf)	((intptr_t)(cf) & CFRAME_RESUME)
#define cframe_unwind_ff(cf)	((intptr_t)(cf) & CFRAME_UNWIND_FF)
#define cframe_raw(cf)		((void *)((intptr_t)(cf) & CFRAME_RAWMASK))
#define cframe_Lpc(L)		cframe_pc(cframe_raw(L->cframe))

#endif
