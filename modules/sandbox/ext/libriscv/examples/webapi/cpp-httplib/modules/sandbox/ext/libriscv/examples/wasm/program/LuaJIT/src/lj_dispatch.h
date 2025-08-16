/*
** Instruction dispatch handling.
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
*/

#ifndef _LJ_DISPATCH_H
#define _LJ_DISPATCH_H

#include "lj_obj.h"
#include "lj_bc.h"
#if LJ_HASJIT
#include "lj_jit.h"
#endif

#if LJ_TARGET_MIPS
/* Need our own global offset table for the dreaded MIPS calling conventions. */

#ifndef _LJ_VM_H
LJ_ASMF int32_t LJ_FASTCALL lj_vm_modi(int32_t a, int32_t b);
#endif

#if LJ_SOFTFP
#ifndef _LJ_IRCALL_H
extern double __adddf3(double a, double b);
extern double __subdf3(double a, double b);
extern double __muldf3(double a, double b);
extern double __divdf3(double a, double b);
#endif
#define SFGOTDEF(_)	_(sqrt) _(__adddf3) _(__subdf3) _(__muldf3) _(__divdf3)
#else
#define SFGOTDEF(_)
#endif
#if LJ_HASJIT
#define JITGOTDEF(_)	_(lj_err_trace) _(lj_trace_exit) _(lj_trace_hot)
#else
#define JITGOTDEF(_)
#endif
#if LJ_HASFFI
#define FFIGOTDEF(_) \
  _(lj_meta_equal_cd) _(lj_ccallback_enter) _(lj_ccallback_leave)
#else
#define FFIGOTDEF(_)
#endif
#define GOTDEF(_) \
  _(floor) _(ceil) _(trunc) _(log) _(log10) _(exp) _(sin) _(cos) _(tan) \
  _(asin) _(acos) _(atan) _(sinh) _(cosh) _(tanh) _(frexp) _(modf) _(atan2) \
  _(pow) _(fmod) _(ldexp) _(lj_vm_modi) \
  _(lj_dispatch_call) _(lj_dispatch_ins) _(lj_dispatch_stitch) \
  _(lj_dispatch_profile) _(lj_err_throw) \
  _(lj_ffh_coroutine_wrap_err) _(lj_func_closeuv) _(lj_func_newL_gc) \
  _(lj_gc_barrieruv) _(lj_gc_step) _(lj_gc_step_fixtop) _(lj_meta_arith) \
  _(lj_meta_call) _(lj_meta_cat) _(lj_meta_comp) _(lj_meta_equal) \
  _(lj_meta_for) _(lj_meta_istype) _(lj_meta_len) _(lj_meta_tget) \
  _(lj_meta_tset) _(lj_state_growstack) _(lj_strfmt_number) \
  _(lj_str_new) _(lj_tab_dup) _(lj_tab_get) _(lj_tab_getinth) _(lj_tab_len) \
  _(lj_tab_new) _(lj_tab_newkey) _(lj_tab_next) _(lj_tab_reasize) \
  _(lj_tab_setinth) _(lj_buf_putstr_reverse) _(lj_buf_putstr_lower) \
  _(lj_buf_putstr_upper) _(lj_buf_tostr) \
  JITGOTDEF(_) FFIGOTDEF(_) SFGOTDEF(_)

enum {
#define GOTENUM(name) LJ_GOT_##name,
GOTDEF(GOTENUM)
#undef GOTENUM
  LJ_GOT__MAX
};
#endif

/* Type of hot counter. Must match the code in the assembler VM. */
/* 16 bits are sufficient. Only 0.0015% overhead with maximum slot penalty. */
typedef uint16_t HotCount;

/* Number of hot counter hash table entries (must be a power of two). */
#define HOTCOUNT_SIZE		64
#define HOTCOUNT_PCMASK		((HOTCOUNT_SIZE-1)*sizeof(HotCount))

/* Hotcount decrements. */
#define HOTCOUNT_LOOP		2
#define HOTCOUNT_CALL		1

/* This solves a circular dependency problem -- bump as needed. Sigh. */
#define GG_NUM_ASMFF	57

#define GG_LEN_DDISP	(BC__MAX + GG_NUM_ASMFF)
#define GG_LEN_SDISP	BC_FUNCF
#define GG_LEN_DISP	(GG_LEN_DDISP + GG_LEN_SDISP)

/* Global state, main thread and extra fields are allocated together. */
typedef struct GG_State {
  lua_State L;				/* Main thread. */
  global_State g;			/* Global state. */
#if LJ_TARGET_ARM && !LJ_TARGET_NX
  /* Make g reachable via K12 encoded DISPATCH-relative addressing. */
  uint8_t align1[(16-sizeof(global_State))&15];
#endif
#if LJ_TARGET_MIPS
  ASMFunction got[LJ_GOT__MAX];		/* Global offset table. */
#endif
#if LJ_HASJIT
  jit_State J;				/* JIT state. */
  HotCount hotcount[HOTCOUNT_SIZE];	/* Hot counters. */
#if LJ_TARGET_ARM && !LJ_TARGET_NX
  /* Ditto for J. */
  uint8_t align2[(16-sizeof(jit_State)-sizeof(HotCount)*HOTCOUNT_SIZE)&15];
#endif
#endif
  ASMFunction dispatch[GG_LEN_DISP];	/* Instruction dispatch tables. */
  BCIns bcff[GG_NUM_ASMFF];		/* Bytecode for ASM fast functions. */
} GG_State;

#define GG_OFS(field)	((int)offsetof(GG_State, field))
#define G2GG(gl)	((GG_State *)((char *)(gl) - GG_OFS(g)))
#define J2GG(j)		((GG_State *)((char *)(j) - GG_OFS(J)))
#define L2GG(L)		(G2GG(G(L)))
#define J2G(J)		(&J2GG(J)->g)
#define G2J(gl)		(&G2GG(gl)->J)
#define L2J(L)		(&L2GG(L)->J)
#define GG_G2J		(GG_OFS(J) - GG_OFS(g))
#define GG_G2DISP	(GG_OFS(dispatch) - GG_OFS(g))
#define GG_DISP2G	(GG_OFS(g) - GG_OFS(dispatch))
#define GG_DISP2J	(GG_OFS(J) - GG_OFS(dispatch))
#define GG_DISP2HOT	(GG_OFS(hotcount) - GG_OFS(dispatch))
#define GG_DISP2STATIC	(GG_LEN_DDISP*(int)sizeof(ASMFunction))

#define hotcount_get(gg, pc) \
  (gg)->hotcount[(u32ptr(pc)>>2) & (HOTCOUNT_SIZE-1)]
#define hotcount_set(gg, pc, val) \
  (hotcount_get((gg), (pc)) = (HotCount)(val))

/* Dispatch table management. */
LJ_FUNC void lj_dispatch_init(GG_State *GG);
#if LJ_HASJIT
LJ_FUNC void lj_dispatch_init_hotcount(global_State *g);
#endif
LJ_FUNC void lj_dispatch_update(global_State *g);

/* Instruction dispatch callback for hooks or when recording. */
LJ_FUNCA void LJ_FASTCALL lj_dispatch_ins(lua_State *L, const BCIns *pc);
LJ_FUNCA ASMFunction LJ_FASTCALL lj_dispatch_call(lua_State *L, const BCIns*pc);
#if LJ_HASJIT
LJ_FUNCA void LJ_FASTCALL lj_dispatch_stitch(jit_State *J, const BCIns *pc);
#endif
#if LJ_HASPROFILE
LJ_FUNCA void LJ_FASTCALL lj_dispatch_profile(lua_State *L, const BCIns *pc);
#endif

#if LJ_HASFFI && !defined(_BUILDVM_H)
/* Save/restore errno and GetLastError() around hooks, exits and recording. */
#include <errno.h>
#if LJ_TARGET_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define ERRNO_SAVE	int olderr = errno; DWORD oldwerr = GetLastError();
#define ERRNO_RESTORE	errno = olderr; SetLastError(oldwerr);
#else
#define ERRNO_SAVE	int olderr = errno;
#define ERRNO_RESTORE	errno = olderr;
#endif
#else
#define ERRNO_SAVE
#define ERRNO_RESTORE
#endif

#endif
