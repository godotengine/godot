/*
** Math library.
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
*/

#include <math.h>

#define lib_math_c
#define LUA_LIB

#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"

#include "lj_obj.h"
#include "lj_err.h"
#include "lj_lib.h"
#include "lj_vm.h"
#include "lj_prng.h"

/* ------------------------------------------------------------------------ */

#define LJLIB_MODULE_math

LJLIB_ASM(math_abs)		LJLIB_REC(.)
{
  lj_lib_checknumber(L, 1);
  return FFH_RETRY;
}
LJLIB_ASM_(math_floor)		LJLIB_REC(math_round IRFPM_FLOOR)
LJLIB_ASM_(math_ceil)		LJLIB_REC(math_round IRFPM_CEIL)

LJLIB_ASM(math_sqrt)		LJLIB_REC(math_unary IRFPM_SQRT)
{
  lj_lib_checknum(L, 1);
  return FFH_RETRY;
}
LJLIB_ASM_(math_log10)		LJLIB_REC(math_call IRCALL_log10)
LJLIB_ASM_(math_exp)		LJLIB_REC(math_call IRCALL_exp)
LJLIB_ASM_(math_sin)		LJLIB_REC(math_call IRCALL_sin)
LJLIB_ASM_(math_cos)		LJLIB_REC(math_call IRCALL_cos)
LJLIB_ASM_(math_tan)		LJLIB_REC(math_call IRCALL_tan)
LJLIB_ASM_(math_asin)		LJLIB_REC(math_call IRCALL_asin)
LJLIB_ASM_(math_acos)		LJLIB_REC(math_call IRCALL_acos)
LJLIB_ASM_(math_atan)		LJLIB_REC(math_call IRCALL_atan)
LJLIB_ASM_(math_sinh)		LJLIB_REC(math_call IRCALL_sinh)
LJLIB_ASM_(math_cosh)		LJLIB_REC(math_call IRCALL_cosh)
LJLIB_ASM_(math_tanh)		LJLIB_REC(math_call IRCALL_tanh)
LJLIB_ASM_(math_frexp)
LJLIB_ASM_(math_modf)

LJLIB_ASM(math_log)		LJLIB_REC(math_log)
{
  double x = lj_lib_checknum(L, 1);
  if (L->base+1 < L->top) {
    double y = lj_lib_checknum(L, 2);
#ifdef LUAJIT_NO_LOG2
    x = log(x); y = 1.0 / log(y);
#else
    x = lj_vm_log2(x); y = 1.0 / lj_vm_log2(y);
#endif
    setnumV(L->base-1-LJ_FR2, x*y);  /* Do NOT join the expression to x / y. */
    return FFH_RES(1);
  }
  return FFH_RETRY;
}

LJLIB_LUA(math_deg) /* function(x) return x * 57.29577951308232 end */
LJLIB_LUA(math_rad) /* function(x) return x * 0.017453292519943295 end */

LJLIB_ASM(math_atan2)		LJLIB_REC(.)
{
  lj_lib_checknum(L, 1);
  lj_lib_checknum(L, 2);
  return FFH_RETRY;
}
LJLIB_ASM_(math_pow)		LJLIB_REC(.)
LJLIB_ASM_(math_fmod)

LJLIB_ASM(math_ldexp)		LJLIB_REC(.)
{
  lj_lib_checknum(L, 1);
#if LJ_DUALNUM && !LJ_TARGET_X86ORX64
  lj_lib_checkint(L, 2);
#else
  lj_lib_checknum(L, 2);
#endif
  return FFH_RETRY;
}

LJLIB_ASM(math_min)		LJLIB_REC(math_minmax IR_MIN)
{
  int i = 0;
  do { lj_lib_checknumber(L, ++i); } while (L->base+i < L->top);
  return FFH_RETRY;
}
LJLIB_ASM_(math_max)		LJLIB_REC(math_minmax IR_MAX)

LJLIB_PUSH(3.14159265358979323846) LJLIB_SET(pi)
LJLIB_PUSH(1e310) LJLIB_SET(huge)

/* ------------------------------------------------------------------------ */

/* This implements a Tausworthe PRNG with period 2^223. Based on:
**   Tables of maximally-equidistributed combined LFSR generators,
**   Pierre L'Ecuyer, 1991, table 3, 1st entry.
** Full-period ME-CF generator with L=64, J=4, k=223, N1=49.
*/

/* Union needed for bit-pattern conversion between uint64_t and double. */
typedef union { uint64_t u64; double d; } U64double;

/* PRNG seeding function. */
static void random_seed(PRNGState *rs, double d)
{
  uint32_t r = 0x11090601;  /* 64-k[i] as four 8 bit constants. */
  int i;
  for (i = 0; i < 4; i++) {
    U64double u;
    uint32_t m = 1u << (r&255);
    r >>= 8;
    u.d = d = d * 3.14159265358979323846 + 2.7182818284590452354;
    if (u.u64 < m) u.u64 += m;  /* Ensure k[i] MSB of u[i] are non-zero. */
    rs->u[i] = u.u64;
  }
  for (i = 0; i < 10; i++)
    (void)lj_prng_u64(rs);
}

/* PRNG extract function. */
LJLIB_PUSH(top-2)  /* Upvalue holds userdata with PRNGState. */
LJLIB_CF(math_random)		LJLIB_REC(.)
{
  int n = (int)(L->top - L->base);
  PRNGState *rs = (PRNGState *)(uddata(udataV(lj_lib_upvalue(L, 1))));
  U64double u;
  double d;
  u.u64 = lj_prng_u64d(rs);
  d = u.d - 1.0;
  if (n > 0) {
#if LJ_DUALNUM
    int isint = 1;
    double r1;
    lj_lib_checknumber(L, 1);
    if (tvisint(L->base)) {
      r1 = (lua_Number)intV(L->base);
    } else {
      isint = 0;
      r1 = numV(L->base);
    }
#else
    double r1 = lj_lib_checknum(L, 1);
#endif
    if (n == 1) {
      d = lj_vm_floor(d*r1) + 1.0;  /* d is an int in range [1, r1] */
    } else {
#if LJ_DUALNUM
      double r2;
      lj_lib_checknumber(L, 2);
      if (tvisint(L->base+1)) {
	r2 = (lua_Number)intV(L->base+1);
      } else {
	isint = 0;
	r2 = numV(L->base+1);
      }
#else
      double r2 = lj_lib_checknum(L, 2);
#endif
      d = lj_vm_floor(d*(r2-r1+1.0)) + r1;  /* d is an int in range [r1, r2] */
    }
#if LJ_DUALNUM
    if (isint) {
      setintV(L->top-1, lj_num2int(d));
      return 1;
    }
#endif
  }  /* else: d is a double in range [0, 1] */
  setnumV(L->top++, d);
  return 1;
}

/* PRNG seed function. */
LJLIB_PUSH(top-2)  /* Upvalue holds userdata with PRNGState. */
LJLIB_CF(math_randomseed)
{
  PRNGState *rs = (PRNGState *)(uddata(udataV(lj_lib_upvalue(L, 1))));
  if (L->base != L->top)
    random_seed(rs, lj_lib_checknum(L, 1));
  else if (!lj_prng_seed_secure(rs))
    lj_err_caller(L, LJ_ERR_PRNGSD);
  return 0;
}

/* ------------------------------------------------------------------------ */

#include "lj_libdef.h"

LUALIB_API int luaopen_math(lua_State *L)
{
  PRNGState *rs = (PRNGState *)lua_newuserdata(L, sizeof(PRNGState));
  lj_prng_seed_fixed(rs);
  LJ_LIB_REG(L, LUA_MATHLIBNAME, math);
  return 1;
}

