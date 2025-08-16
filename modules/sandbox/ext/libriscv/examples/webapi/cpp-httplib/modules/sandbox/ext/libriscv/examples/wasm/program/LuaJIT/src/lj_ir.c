/*
** SSA IR (Intermediate Representation) emitter.
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
*/

#define lj_ir_c
#define LUA_CORE

/* For pointers to libc/libm functions. */
#include <stdio.h>
#include <math.h>

#include "lj_obj.h"

#if LJ_HASJIT

#include "lj_gc.h"
#include "lj_buf.h"
#include "lj_str.h"
#include "lj_tab.h"
#include "lj_ir.h"
#include "lj_jit.h"
#include "lj_ircall.h"
#include "lj_iropt.h"
#include "lj_trace.h"
#if LJ_HASFFI
#include "lj_ctype.h"
#include "lj_cdata.h"
#include "lj_carith.h"
#endif
#include "lj_vm.h"
#include "lj_strscan.h"
#include "lj_serialize.h"
#include "lj_strfmt.h"
#include "lj_prng.h"

/* Some local macros to save typing. Undef'd at the end. */
#define IR(ref)			(&J->cur.ir[(ref)])
#define fins			(&J->fold.ins)

/* Pass IR on to next optimization in chain (FOLD). */
#define emitir(ot, a, b)	(lj_ir_set(J, (ot), (a), (b)), lj_opt_fold(J))

/* -- IR tables ----------------------------------------------------------- */

/* IR instruction modes. */
LJ_DATADEF const uint8_t lj_ir_mode[IR__MAX+1] = {
IRDEF(IRMODE)
  0
};

/* IR type sizes. */
LJ_DATADEF const uint8_t lj_ir_type_size[IRT__MAX+1] = {
#define IRTSIZE(name, size)	size,
IRTDEF(IRTSIZE)
#undef IRTSIZE
  0
};

/* C call info for CALL* instructions. */
LJ_DATADEF const CCallInfo lj_ir_callinfo[] = {
#define IRCALLCI(cond, name, nargs, kind, type, flags) \
  { (ASMFunction)IRCALLCOND_##cond(name), \
    (nargs)|(CCI_CALL_##kind)|(IRT_##type<<CCI_OTSHIFT)|(flags) },
IRCALLDEF(IRCALLCI)
#undef IRCALLCI
  { NULL, 0 }
};

/* -- IR emitter ---------------------------------------------------------- */

/* Grow IR buffer at the top. */
void LJ_FASTCALL lj_ir_growtop(jit_State *J)
{
  IRIns *baseir = J->irbuf + J->irbotlim;
  MSize szins = J->irtoplim - J->irbotlim;
  if (szins) {
    baseir = (IRIns *)lj_mem_realloc(J->L, baseir, szins*sizeof(IRIns),
				     2*szins*sizeof(IRIns));
    J->irtoplim = J->irbotlim + 2*szins;
  } else {
    baseir = (IRIns *)lj_mem_realloc(J->L, NULL, 0, LJ_MIN_IRSZ*sizeof(IRIns));
    J->irbotlim = REF_BASE - LJ_MIN_IRSZ/4;
    J->irtoplim = J->irbotlim + LJ_MIN_IRSZ;
  }
  J->cur.ir = J->irbuf = baseir - J->irbotlim;
}

/* Grow IR buffer at the bottom or shift it up. */
static void lj_ir_growbot(jit_State *J)
{
  IRIns *baseir = J->irbuf + J->irbotlim;
  MSize szins = J->irtoplim - J->irbotlim;
  lj_assertJ(szins != 0, "zero IR size");
  lj_assertJ(J->cur.nk == J->irbotlim || J->cur.nk-1 == J->irbotlim,
	     "unexpected IR growth");
  if (J->cur.nins + (szins >> 1) < J->irtoplim) {
    /* More than half of the buffer is free on top: shift up by a quarter. */
    MSize ofs = szins >> 2;
    memmove(baseir + ofs, baseir, (J->cur.nins - J->irbotlim)*sizeof(IRIns));
    J->irbotlim -= ofs;
    J->irtoplim -= ofs;
    J->cur.ir = J->irbuf = baseir - J->irbotlim;
  } else {
    /* Double the buffer size, but split the growth amongst top/bottom. */
    IRIns *newbase = lj_mem_newt(J->L, 2*szins*sizeof(IRIns), IRIns);
    MSize ofs = szins >= 256 ? 128 : (szins >> 1);  /* Limit bottom growth. */
    memcpy(newbase + ofs, baseir, (J->cur.nins - J->irbotlim)*sizeof(IRIns));
    lj_mem_free(G(J->L), baseir, szins*sizeof(IRIns));
    J->irbotlim -= ofs;
    J->irtoplim = J->irbotlim + 2*szins;
    J->cur.ir = J->irbuf = newbase - J->irbotlim;
  }
}

/* Emit IR without any optimizations. */
TRef LJ_FASTCALL lj_ir_emit(jit_State *J)
{
  IRRef ref = lj_ir_nextins(J);
  IRIns *ir = IR(ref);
  IROp op = fins->o;
  ir->prev = J->chain[op];
  J->chain[op] = (IRRef1)ref;
  ir->o = op;
  ir->op1 = fins->op1;
  ir->op2 = fins->op2;
  J->guardemit.irt |= fins->t.irt;
  return TREF(ref, irt_t((ir->t = fins->t)));
}

/* Emit call to a C function. */
TRef lj_ir_call(jit_State *J, IRCallID id, ...)
{
  const CCallInfo *ci = &lj_ir_callinfo[id];
  uint32_t n = CCI_NARGS(ci);
  TRef tr = TREF_NIL;
  va_list argp;
  va_start(argp, id);
  if ((ci->flags & CCI_L)) n--;
  if (n > 0)
    tr = va_arg(argp, IRRef);
  while (n-- > 1)
    tr = emitir(IRT(IR_CARG, IRT_NIL), tr, va_arg(argp, IRRef));
  va_end(argp);
  if (CCI_OP(ci) == IR_CALLS)
    J->needsnap = 1;  /* Need snapshot after call with side effect. */
  return emitir(CCI_OPTYPE(ci), tr, id);
}

/* Load field of type t from GG_State + offset. Must be 32 bit aligned. */
TRef lj_ir_ggfload(jit_State *J, IRType t, uintptr_t ofs)
{
  lj_assertJ((ofs & 3) == 0, "unaligned GG_State field offset");
  ofs >>= 2;
  lj_assertJ(ofs >= IRFL__MAX && ofs <= 0x3ff,
	     "GG_State field offset breaks 10 bit FOLD key limit");
  lj_ir_set(J, IRT(IR_FLOAD, t), REF_NIL, ofs);
  return lj_opt_fold(J);
}

/* -- Interning of constants ---------------------------------------------- */

/*
** IR instructions for constants are kept between J->cur.nk >= ref < REF_BIAS.
** They are chained like all other instructions, but grow downwards.
** The are interned (like strings in the VM) to facilitate reference
** comparisons. The same constant must get the same reference.
*/

/* Get ref of next IR constant and optionally grow IR.
** Note: this may invalidate all IRIns *!
*/
static LJ_AINLINE IRRef ir_nextk(jit_State *J)
{
  IRRef ref = J->cur.nk;
  if (LJ_UNLIKELY(ref <= J->irbotlim)) lj_ir_growbot(J);
  J->cur.nk = --ref;
  return ref;
}

/* Get ref of next 64 bit IR constant and optionally grow IR.
** Note: this may invalidate all IRIns *!
*/
static LJ_AINLINE IRRef ir_nextk64(jit_State *J)
{
  IRRef ref = J->cur.nk - 2;
  lj_assertJ(J->state != LJ_TRACE_ASM, "bad JIT state");
  if (LJ_UNLIKELY(ref < J->irbotlim)) lj_ir_growbot(J);
  J->cur.nk = ref;
  return ref;
}

#if LJ_GC64
#define ir_nextkgc ir_nextk64
#else
#define ir_nextkgc ir_nextk
#endif

/* Intern int32_t constant. */
TRef LJ_FASTCALL lj_ir_kint(jit_State *J, int32_t k)
{
  IRIns *ir, *cir = J->cur.ir;
  IRRef ref;
  for (ref = J->chain[IR_KINT]; ref; ref = cir[ref].prev)
    if (cir[ref].i == k)
      goto found;
  ref = ir_nextk(J);
  ir = IR(ref);
  ir->i = k;
  ir->t.irt = IRT_INT;
  ir->o = IR_KINT;
  ir->prev = J->chain[IR_KINT];
  J->chain[IR_KINT] = (IRRef1)ref;
found:
  return TREF(ref, IRT_INT);
}

/* Intern 64 bit constant, given by its 64 bit pattern. */
TRef lj_ir_k64(jit_State *J, IROp op, uint64_t u64)
{
  IRIns *ir, *cir = J->cur.ir;
  IRRef ref;
  IRType t = op == IR_KNUM ? IRT_NUM : IRT_I64;
  for (ref = J->chain[op]; ref; ref = cir[ref].prev)
    if (ir_k64(&cir[ref])->u64 == u64)
      goto found;
  ref = ir_nextk64(J);
  ir = IR(ref);
  ir[1].tv.u64 = u64;
  ir->t.irt = t;
  ir->o = op;
  ir->op12 = 0;
  ir->prev = J->chain[op];
  J->chain[op] = (IRRef1)ref;
found:
  return TREF(ref, t);
}

/* Intern FP constant, given by its 64 bit pattern. */
TRef lj_ir_knum_u64(jit_State *J, uint64_t u64)
{
  return lj_ir_k64(J, IR_KNUM, u64);
}

/* Intern 64 bit integer constant. */
TRef lj_ir_kint64(jit_State *J, uint64_t u64)
{
  return lj_ir_k64(J, IR_KINT64, u64);
}

/* Check whether a number is int and return it. -0 is NOT considered an int. */
static int numistrueint(lua_Number n, int32_t *kp)
{
  int32_t k = lj_num2int(n);
  if (n == (lua_Number)k) {
    if (kp) *kp = k;
    if (k == 0) {  /* Special check for -0. */
      TValue tv;
      setnumV(&tv, n);
      if (tv.u32.hi != 0)
	return 0;
    }
    return 1;
  }
  return 0;
}

/* Intern number as int32_t constant if possible, otherwise as FP constant. */
TRef lj_ir_knumint(jit_State *J, lua_Number n)
{
  int32_t k;
  if (numistrueint(n, &k))
    return lj_ir_kint(J, k);
  else
    return lj_ir_knum(J, n);
}

/* Intern GC object "constant". */
TRef lj_ir_kgc(jit_State *J, GCobj *o, IRType t)
{
  IRIns *ir, *cir = J->cur.ir;
  IRRef ref;
  lj_assertJ(!isdead(J2G(J), o), "interning of dead GC object");
  for (ref = J->chain[IR_KGC]; ref; ref = cir[ref].prev)
    if (ir_kgc(&cir[ref]) == o)
      goto found;
  ref = ir_nextkgc(J);
  ir = IR(ref);
  /* NOBARRIER: Current trace is a GC root. */
  ir->op12 = 0;
  setgcref(ir[LJ_GC64].gcr, o);
  ir->t.irt = (uint8_t)t;
  ir->o = IR_KGC;
  ir->prev = J->chain[IR_KGC];
  J->chain[IR_KGC] = (IRRef1)ref;
found:
  return TREF(ref, t);
}

/* Allocate GCtrace constant placeholder (no interning). */
TRef lj_ir_ktrace(jit_State *J)
{
  IRRef ref = ir_nextkgc(J);
  IRIns *ir = IR(ref);
  lj_assertJ(irt_toitype_(IRT_P64) == LJ_TTRACE, "mismatched type mapping");
  ir->t.irt = IRT_P64;
  ir->o = LJ_GC64 ? IR_KNUM : IR_KNULL;  /* Not IR_KGC yet, but same size. */
  ir->op12 = 0;
  ir->prev = 0;
  return TREF(ref, IRT_P64);
}

/* Intern pointer constant. */
TRef lj_ir_kptr_(jit_State *J, IROp op, void *ptr)
{
  IRIns *ir, *cir = J->cur.ir;
  IRRef ref;
#if LJ_64 && !LJ_GC64
  lj_assertJ((void *)(uintptr_t)u32ptr(ptr) == ptr, "out-of-range GC pointer");
#endif
  for (ref = J->chain[op]; ref; ref = cir[ref].prev)
    if (ir_kptr(&cir[ref]) == ptr)
      goto found;
#if LJ_GC64
  ref = ir_nextk64(J);
#else
  ref = ir_nextk(J);
#endif
  ir = IR(ref);
  ir->op12 = 0;
  setmref(ir[LJ_GC64].ptr, ptr);
  ir->t.irt = IRT_PGC;
  ir->o = op;
  ir->prev = J->chain[op];
  J->chain[op] = (IRRef1)ref;
found:
  return TREF(ref, IRT_PGC);
}

/* Intern typed NULL constant. */
TRef lj_ir_knull(jit_State *J, IRType t)
{
  IRIns *ir, *cir = J->cur.ir;
  IRRef ref;
  for (ref = J->chain[IR_KNULL]; ref; ref = cir[ref].prev)
    if (irt_t(cir[ref].t) == t)
      goto found;
  ref = ir_nextk(J);
  ir = IR(ref);
  ir->i = 0;
  ir->t.irt = (uint8_t)t;
  ir->o = IR_KNULL;
  ir->prev = J->chain[IR_KNULL];
  J->chain[IR_KNULL] = (IRRef1)ref;
found:
  return TREF(ref, t);
}

/* Intern key slot. */
TRef lj_ir_kslot(jit_State *J, TRef key, IRRef slot)
{
  IRIns *ir, *cir = J->cur.ir;
  IRRef2 op12 = IRREF2((IRRef1)key, (IRRef1)slot);
  IRRef ref;
  /* Const part is not touched by CSE/DCE, so 0-65535 is ok for IRMlit here. */
  lj_assertJ(tref_isk(key) && slot == (IRRef)(IRRef1)slot,
	     "out-of-range key/slot");
  for (ref = J->chain[IR_KSLOT]; ref; ref = cir[ref].prev)
    if (cir[ref].op12 == op12)
      goto found;
  ref = ir_nextk(J);
  ir = IR(ref);
  ir->op12 = op12;
  ir->t.irt = IRT_P32;
  ir->o = IR_KSLOT;
  ir->prev = J->chain[IR_KSLOT];
  J->chain[IR_KSLOT] = (IRRef1)ref;
found:
  return TREF(ref, IRT_P32);
}

/* -- Access to IR constants ---------------------------------------------- */

/* Copy value of IR constant. */
void lj_ir_kvalue(lua_State *L, TValue *tv, const IRIns *ir)
{
  UNUSED(L);
  lj_assertL(ir->o != IR_KSLOT, "unexpected KSLOT");  /* Common mistake. */
  switch (ir->o) {
  case IR_KPRI: setpriV(tv, irt_toitype(ir->t)); break;
  case IR_KINT: setintV(tv, ir->i); break;
  case IR_KGC: setgcV(L, tv, ir_kgc(ir), irt_toitype(ir->t)); break;
  case IR_KPTR: case IR_KKPTR:
    setnumV(tv, (lua_Number)(uintptr_t)ir_kptr(ir));
    break;
  case IR_KNULL: setintV(tv, 0); break;
  case IR_KNUM: setnumV(tv, ir_knum(ir)->n); break;
#if LJ_HASFFI
  case IR_KINT64: {
    GCcdata *cd = lj_cdata_new_(L, CTID_INT64, 8);
    *(uint64_t *)cdataptr(cd) = ir_kint64(ir)->u64;
    setcdataV(L, tv, cd);
    break;
    }
#endif
  default: lj_assertL(0, "bad IR constant op %d", ir->o); break;
  }
}

/* -- Convert IR operand types -------------------------------------------- */

/* Convert from string to number. */
TRef LJ_FASTCALL lj_ir_tonumber(jit_State *J, TRef tr)
{
  if (!tref_isnumber(tr)) {
    if (tref_isstr(tr))
      tr = emitir(IRTG(IR_STRTO, IRT_NUM), tr, 0);
    else
      lj_trace_err(J, LJ_TRERR_BADTYPE);
  }
  return tr;
}

/* Convert from integer or string to number. */
TRef LJ_FASTCALL lj_ir_tonum(jit_State *J, TRef tr)
{
  if (!tref_isnum(tr)) {
    if (tref_isinteger(tr))
      tr = emitir(IRTN(IR_CONV), tr, IRCONV_NUM_INT);
    else if (tref_isstr(tr))
      tr = emitir(IRTG(IR_STRTO, IRT_NUM), tr, 0);
    else
      lj_trace_err(J, LJ_TRERR_BADTYPE);
  }
  return tr;
}

/* Convert from integer or number to string. */
TRef LJ_FASTCALL lj_ir_tostr(jit_State *J, TRef tr)
{
  if (!tref_isstr(tr)) {
    if (!tref_isnumber(tr))
      lj_trace_err(J, LJ_TRERR_BADTYPE);
    tr = emitir(IRT(IR_TOSTR, IRT_STR), tr,
		tref_isnum(tr) ? IRTOSTR_NUM : IRTOSTR_INT);
  }
  return tr;
}

/* -- Miscellaneous IR ops ------------------------------------------------ */

/* Evaluate numeric comparison. */
int lj_ir_numcmp(lua_Number a, lua_Number b, IROp op)
{
  switch (op) {
  case IR_EQ: return (a == b);
  case IR_NE: return (a != b);
  case IR_LT: return (a < b);
  case IR_GE: return (a >= b);
  case IR_LE: return (a <= b);
  case IR_GT: return (a > b);
  case IR_ULT: return !(a >= b);
  case IR_UGE: return !(a < b);
  case IR_ULE: return !(a > b);
  case IR_UGT: return !(a <= b);
  default: lj_assertX(0, "bad IR op %d", op); return 0;
  }
}

/* Evaluate string comparison. */
int lj_ir_strcmp(GCstr *a, GCstr *b, IROp op)
{
  int res = lj_str_cmp(a, b);
  switch (op) {
  case IR_LT: return (res < 0);
  case IR_GE: return (res >= 0);
  case IR_LE: return (res <= 0);
  case IR_GT: return (res > 0);
  default: lj_assertX(0, "bad IR op %d", op); return 0;
  }
}

/* Rollback IR to previous state. */
void lj_ir_rollback(jit_State *J, IRRef ref)
{
  IRRef nins = J->cur.nins;
  while (nins > ref) {
    IRIns *ir;
    nins--;
    ir = IR(nins);
    J->chain[ir->o] = ir->prev;
  }
  J->cur.nins = nins;
}

#undef IR
#undef fins
#undef emitir

#endif
