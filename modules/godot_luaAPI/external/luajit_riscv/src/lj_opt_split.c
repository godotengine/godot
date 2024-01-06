/*
** SPLIT: Split 64 bit IR instructions into 32 bit IR instructions.
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
*/

#define lj_opt_split_c
#define LUA_CORE

#include "lj_obj.h"

#if LJ_HASJIT && (LJ_SOFTFP32 || (LJ_32 && LJ_HASFFI))

#include "lj_err.h"
#include "lj_buf.h"
#include "lj_ir.h"
#include "lj_jit.h"
#include "lj_ircall.h"
#include "lj_iropt.h"
#include "lj_dispatch.h"
#include "lj_vm.h"

/* SPLIT pass:
**
** This pass splits up 64 bit IR instructions into multiple 32 bit IR
** instructions. It's only active for soft-float targets or for 32 bit CPUs
** which lack native 64 bit integer operations (the FFI is currently the
** only emitter for 64 bit integer instructions).
**
** Splitting the IR in a separate pass keeps each 32 bit IR assembler
** backend simple. Only a small amount of extra functionality needs to be
** implemented. This is much easier than adding support for allocating
** register pairs to each backend (believe me, I tried). A few simple, but
** important optimizations can be performed by the SPLIT pass, which would
** be tedious to do in the backend.
**
** The basic idea is to replace each 64 bit IR instruction with its 32 bit
** equivalent plus an extra HIOP instruction. The splitted IR is not passed
** through FOLD or any other optimizations, so each HIOP is guaranteed to
** immediately follow it's counterpart. The actual functionality of HIOP is
** inferred from the previous instruction.
**
** The operands of HIOP hold the hiword input references. The output of HIOP
** is the hiword output reference, which is also used to hold the hiword
** register or spill slot information. The register allocator treats this
** instruction independently of any other instruction, which improves code
** quality compared to using fixed register pairs.
**
** It's easier to split up some instructions into two regular 32 bit
** instructions. E.g. XLOAD is split up into two XLOADs with two different
** addresses. Obviously 64 bit constants need to be split up into two 32 bit
** constants, too. Some hiword instructions can be entirely omitted, e.g.
** when zero-extending a 32 bit value to 64 bits. 64 bit arguments for calls
** are split up into two 32 bit arguments each.
**
** On soft-float targets, floating-point instructions are directly converted
** to soft-float calls by the SPLIT pass (except for comparisons and MIN/MAX).
** HIOP for number results has the type IRT_SOFTFP ("sfp" in -jdump).
**
** Here's the IR and x64 machine code for 'x.b = x.a + 1' for a struct with
** two int64_t fields:
**
** 0100    p32 ADD    base  +8
** 0101    i64 XLOAD  0100
** 0102    i64 ADD    0101  +1
** 0103    p32 ADD    base  +16
** 0104    i64 XSTORE 0103  0102
**
**         mov rax, [esi+0x8]
**         add rax, +0x01
**         mov [esi+0x10], rax
**
** Here's the transformed IR and the x86 machine code after the SPLIT pass:
**
** 0100    p32 ADD    base  +8
** 0101    int XLOAD  0100
** 0102    p32 ADD    base  +12
** 0103    int XLOAD  0102
** 0104    int ADD    0101  +1
** 0105    int HIOP   0103  +0
** 0106    p32 ADD    base  +16
** 0107    int XSTORE 0106  0104
** 0108    int HIOP   0106  0105
**
**         mov eax, [esi+0x8]
**         mov ecx, [esi+0xc]
**         add eax, +0x01
**         adc ecx, +0x00
**         mov [esi+0x10], eax
**         mov [esi+0x14], ecx
**
** You may notice the reassociated hiword address computation, which is
** later fused into the mov operands by the assembler.
*/

/* Some local macros to save typing. Undef'd at the end. */
#define IR(ref)		(&J->cur.ir[(ref)])

/* Directly emit the transformed IR without updating chains etc. */
static IRRef split_emit(jit_State *J, uint16_t ot, IRRef1 op1, IRRef1 op2)
{
  IRRef nref = lj_ir_nextins(J);
  IRIns *ir = IR(nref);
  ir->ot = ot;
  ir->op1 = op1;
  ir->op2 = op2;
  return nref;
}

#if LJ_SOFTFP
/* Emit a (checked) number to integer conversion. */
static IRRef split_num2int(jit_State *J, IRRef lo, IRRef hi, int check)
{
  IRRef tmp, res;
#if LJ_LE
  tmp = split_emit(J, IRT(IR_CARG, IRT_NIL), lo, hi);
#else
  tmp = split_emit(J, IRT(IR_CARG, IRT_NIL), hi, lo);
#endif
  res = split_emit(J, IRTI(IR_CALLN), tmp, IRCALL_softfp_d2i);
  if (check) {
    tmp = split_emit(J, IRTI(IR_CALLN), res, IRCALL_softfp_i2d);
    split_emit(J, IRT(IR_HIOP, IRT_SOFTFP), tmp, tmp);
    split_emit(J, IRTGI(IR_EQ), tmp, lo);
    split_emit(J, IRTG(IR_HIOP, IRT_SOFTFP), tmp+1, hi);
  }
  return res;
}

/* Emit a CALLN with one split 64 bit argument. */
static IRRef split_call_l(jit_State *J, IRRef1 *hisubst, IRIns *oir,
			  IRIns *ir, IRCallID id)
{
  IRRef tmp, op1 = ir->op1;
  J->cur.nins--;
#if LJ_LE
  tmp = split_emit(J, IRT(IR_CARG, IRT_NIL), oir[op1].prev, hisubst[op1]);
#else
  tmp = split_emit(J, IRT(IR_CARG, IRT_NIL), hisubst[op1], oir[op1].prev);
#endif
  ir->prev = tmp = split_emit(J, IRTI(IR_CALLN), tmp, id);
  return split_emit(J, IRT(IR_HIOP, IRT_SOFTFP), tmp, tmp);
}
#endif

/* Emit a CALLN with one split 64 bit argument and a 32 bit argument. */
static IRRef split_call_li(jit_State *J, IRRef1 *hisubst, IRIns *oir,
			   IRIns *ir, IRCallID id)
{
  IRRef tmp, op1 = ir->op1, op2 = ir->op2;
  J->cur.nins--;
#if LJ_LE
  tmp = split_emit(J, IRT(IR_CARG, IRT_NIL), oir[op1].prev, hisubst[op1]);
#else
  tmp = split_emit(J, IRT(IR_CARG, IRT_NIL), hisubst[op1], oir[op1].prev);
#endif
  tmp = split_emit(J, IRT(IR_CARG, IRT_NIL), tmp, oir[op2].prev);
  ir->prev = tmp = split_emit(J, IRTI(IR_CALLN), tmp, id);
  return split_emit(J, IRT(IR_HIOP, IRT_SOFTFP), tmp, tmp);
}

/* Emit a CALLN with two split 64 bit arguments. */
static IRRef split_call_ll(jit_State *J, IRRef1 *hisubst, IRIns *oir,
			   IRIns *ir, IRCallID id)
{
  IRRef tmp, op1 = ir->op1, op2 = ir->op2;
  J->cur.nins--;
#if LJ_LE
  tmp = split_emit(J, IRT(IR_CARG, IRT_NIL), oir[op1].prev, hisubst[op1]);
  tmp = split_emit(J, IRT(IR_CARG, IRT_NIL), tmp, oir[op2].prev);
  tmp = split_emit(J, IRT(IR_CARG, IRT_NIL), tmp, hisubst[op2]);
#else
  tmp = split_emit(J, IRT(IR_CARG, IRT_NIL), hisubst[op1], oir[op1].prev);
  tmp = split_emit(J, IRT(IR_CARG, IRT_NIL), tmp, hisubst[op2]);
  tmp = split_emit(J, IRT(IR_CARG, IRT_NIL), tmp, oir[op2].prev);
#endif
  ir->prev = tmp = split_emit(J, IRTI(IR_CALLN), tmp, id);
  return split_emit(J,
    IRT(IR_HIOP, (LJ_SOFTFP && irt_isnum(ir->t)) ? IRT_SOFTFP : IRT_INT),
    tmp, tmp);
}

/* Get a pointer to the other 32 bit word (LE: hiword, BE: loword). */
static IRRef split_ptr(jit_State *J, IRIns *oir, IRRef ref)
{
  IRRef nref = oir[ref].prev;
  IRIns *ir = IR(nref);
  int32_t ofs = 4;
  if (ir->o == IR_KPTR)
    return lj_ir_kptr(J, (char *)ir_kptr(ir) + ofs);
  if (ir->o == IR_ADD && irref_isk(ir->op2) && !irt_isphi(oir[ref].t)) {
    /* Reassociate address. */
    ofs += IR(ir->op2)->i;
    nref = ir->op1;
    if (ofs == 0) return nref;
  }
  return split_emit(J, IRT(IR_ADD, IRT_PTR), nref, lj_ir_kint(J, ofs));
}

#if LJ_HASFFI
static IRRef split_bitshift(jit_State *J, IRRef1 *hisubst,
			    IRIns *oir, IRIns *nir, IRIns *ir)
{
  IROp op = ir->o;
  IRRef kref = nir->op2;
  if (irref_isk(kref)) {  /* Optimize constant shifts. */
    int32_t k = (IR(kref)->i & 63);
    IRRef lo = nir->op1, hi = hisubst[ir->op1];
    if (op == IR_BROL || op == IR_BROR) {
      if (op == IR_BROR) k = (-k & 63);
      if (k >= 32) { IRRef t = lo; lo = hi; hi = t; k -= 32; }
      if (k == 0) {
      passthrough:
	J->cur.nins--;
	ir->prev = lo;
	return hi;
      } else {
	TRef k1, k2;
	IRRef t1, t2, t3, t4;
	J->cur.nins--;
	k1 = lj_ir_kint(J, k);
	k2 = lj_ir_kint(J, (-k & 31));
	t1 = split_emit(J, IRTI(IR_BSHL), lo, k1);
	t2 = split_emit(J, IRTI(IR_BSHL), hi, k1);
	t3 = split_emit(J, IRTI(IR_BSHR), lo, k2);
	t4 = split_emit(J, IRTI(IR_BSHR), hi, k2);
	ir->prev = split_emit(J, IRTI(IR_BOR), t1, t4);
	return split_emit(J, IRTI(IR_BOR), t2, t3);
      }
    } else if (k == 0) {
      goto passthrough;
    } else if (k < 32) {
      if (op == IR_BSHL) {
	IRRef t1 = split_emit(J, IRTI(IR_BSHL), hi, kref);
	IRRef t2 = split_emit(J, IRTI(IR_BSHR), lo, lj_ir_kint(J, (-k&31)));
	return split_emit(J, IRTI(IR_BOR), t1, t2);
      } else {
	IRRef t1 = ir->prev, t2;
	lj_assertJ(op == IR_BSHR || op == IR_BSAR, "bad usage");
	nir->o = IR_BSHR;
	t2 = split_emit(J, IRTI(IR_BSHL), hi, lj_ir_kint(J, (-k&31)));
	ir->prev = split_emit(J, IRTI(IR_BOR), t1, t2);
	return split_emit(J, IRTI(op), hi, kref);
      }
    } else {
      if (op == IR_BSHL) {
	if (k == 32)
	  J->cur.nins--;
	else
	  lo = ir->prev;
	ir->prev = lj_ir_kint(J, 0);
	return lo;
      } else {
	lj_assertJ(op == IR_BSHR || op == IR_BSAR, "bad usage");
	if (k == 32) {
	  J->cur.nins--;
	  ir->prev = hi;
	} else {
	  nir->op1 = hi;
	}
	if (op == IR_BSHR)
	  return lj_ir_kint(J, 0);
	else
	  return split_emit(J, IRTI(IR_BSAR), hi, lj_ir_kint(J, 31));
      }
    }
  }
  return split_call_li(J, hisubst, oir, ir,
		       op - IR_BSHL + IRCALL_lj_carith_shl64);
}

static IRRef split_bitop(jit_State *J, IRRef1 *hisubst,
			 IRIns *nir, IRIns *ir)
{
  IROp op = ir->o;
  IRRef hi, kref = nir->op2;
  if (irref_isk(kref)) {  /* Optimize bit operations with lo constant. */
    int32_t k = IR(kref)->i;
    if (k == 0 || k == -1) {
      if (op == IR_BAND) k = ~k;
      if (k == 0) {
	J->cur.nins--;
	ir->prev = nir->op1;
      } else if (op == IR_BXOR) {
	nir->o = IR_BNOT;
	nir->op2 = 0;
      } else {
	J->cur.nins--;
	ir->prev = kref;
      }
    }
  }
  hi = hisubst[ir->op1];
  kref = hisubst[ir->op2];
  if (irref_isk(kref)) {  /* Optimize bit operations with hi constant. */
    int32_t k = IR(kref)->i;
    if (k == 0 || k == -1) {
      if (op == IR_BAND) k = ~k;
      if (k == 0) {
	return hi;
      } else if (op == IR_BXOR) {
	return split_emit(J, IRTI(IR_BNOT), hi, 0);
      } else {
	return kref;
      }
    }
  }
  return split_emit(J, IRTI(op), hi, kref);
}
#endif

/* Substitute references of a snapshot. */
static void split_subst_snap(jit_State *J, SnapShot *snap, IRIns *oir)
{
  SnapEntry *map = &J->cur.snapmap[snap->mapofs];
  MSize n, nent = snap->nent;
  for (n = 0; n < nent; n++) {
    SnapEntry sn = map[n];
    IRIns *ir = &oir[snap_ref(sn)];
    if (!(LJ_SOFTFP && (sn & SNAP_SOFTFPNUM) && irref_isk(snap_ref(sn))))
      map[n] = ((sn & 0xffff0000) | ir->prev);
  }
}

/* Transform the old IR to the new IR. */
static void split_ir(jit_State *J)
{
  IRRef nins = J->cur.nins, nk = J->cur.nk;
  MSize irlen = nins - nk;
  MSize need = (irlen+1)*(sizeof(IRIns) + sizeof(IRRef1));
  IRIns *oir = (IRIns *)lj_buf_tmp(J->L, need);
  IRRef1 *hisubst;
  IRRef ref, snref;
  SnapShot *snap;

  /* Copy old IR to buffer. */
  memcpy(oir, IR(nk), irlen*sizeof(IRIns));
  /* Bias hiword substitution table and old IR. Loword kept in field prev. */
  hisubst = (IRRef1 *)&oir[irlen] - nk;
  oir -= nk;

  /* Remove all IR instructions, but retain IR constants. */
  J->cur.nins = REF_FIRST;
  J->loopref = 0;

  /* Process constants and fixed references. */
  for (ref = nk; ref <= REF_BASE; ref++) {
    IRIns *ir = &oir[ref];
    if ((LJ_SOFTFP && ir->o == IR_KNUM) || ir->o == IR_KINT64) {
      /* Split up 64 bit constant. */
      TValue tv = *ir_k64(ir);
      ir->prev = lj_ir_kint(J, (int32_t)tv.u32.lo);
      hisubst[ref] = lj_ir_kint(J, (int32_t)tv.u32.hi);
    } else {
      ir->prev = ref;  /* Identity substitution for loword. */
      hisubst[ref] = 0;
    }
    if (irt_is64(ir->t) && ir->o != IR_KNULL)
      ref++;
  }

  /* Process old IR instructions. */
  snap = J->cur.snap;
  snref = snap->ref;
  for (ref = REF_FIRST; ref < nins; ref++) {
    IRIns *ir = &oir[ref];
    IRRef nref = lj_ir_nextins(J);
    IRIns *nir = IR(nref);
    IRRef hi = 0;

    if (ref >= snref) {
      snap->ref = nref;
      split_subst_snap(J, snap++, oir);
      snref = snap < &J->cur.snap[J->cur.nsnap] ? snap->ref : ~(IRRef)0;
    }

    /* Copy-substitute old instruction to new instruction. */
    nir->op1 = ir->op1 < nk ? ir->op1 : oir[ir->op1].prev;
    nir->op2 = ir->op2 < nk ? ir->op2 : oir[ir->op2].prev;
    ir->prev = nref;  /* Loword substitution. */
    nir->o = ir->o;
    nir->t.irt = ir->t.irt & ~(IRT_MARK|IRT_ISPHI);
    hisubst[ref] = 0;

    /* Split 64 bit instructions. */
#if LJ_SOFTFP
    if (irt_isnum(ir->t)) {
      nir->t.irt = IRT_INT | (nir->t.irt & IRT_GUARD);  /* Turn into INT op. */
      /* Note: hi ref = lo ref + 1! Required for SNAP_SOFTFPNUM logic. */
      switch (ir->o) {
      case IR_ADD:
	hi = split_call_ll(J, hisubst, oir, ir, IRCALL_softfp_add);
	break;
      case IR_SUB:
	hi = split_call_ll(J, hisubst, oir, ir, IRCALL_softfp_sub);
	break;
      case IR_MUL:
	hi = split_call_ll(J, hisubst, oir, ir, IRCALL_softfp_mul);
	break;
      case IR_DIV:
	hi = split_call_ll(J, hisubst, oir, ir, IRCALL_softfp_div);
	break;
      case IR_POW:
	hi = split_call_ll(J, hisubst, oir, ir, IRCALL_pow);
	break;
      case IR_FPMATH:
	hi = split_call_l(J, hisubst, oir, ir, IRCALL_lj_vm_floor + ir->op2);
	break;
      case IR_LDEXP:
	hi = split_call_li(J, hisubst, oir, ir, IRCALL_ldexp);
	break;
      case IR_NEG: case IR_ABS:
	nir->o = IR_CONV;  /* Pass through loword. */
	nir->op2 = (IRT_INT << 5) | IRT_INT;
	hi = split_emit(J, IRT(ir->o == IR_NEG ? IR_BXOR : IR_BAND, IRT_SOFTFP),
	       hisubst[ir->op1],
	       lj_ir_kint(J, (int32_t)(0x7fffffffu + (ir->o == IR_NEG))));
	break;
      case IR_SLOAD:
	if ((nir->op2 & IRSLOAD_CONVERT)) {  /* Convert from int to number. */
	  nir->op2 &= ~IRSLOAD_CONVERT;
	  ir->prev = nref = split_emit(J, IRTI(IR_CALLN), nref,
				       IRCALL_softfp_i2d);
	  hi = split_emit(J, IRT(IR_HIOP, IRT_SOFTFP), nref, nref);
	  break;
	}
	/* fallthrough */
      case IR_ALOAD: case IR_HLOAD: case IR_ULOAD: case IR_VLOAD:
      case IR_STRTO:
	hi = split_emit(J, IRT(IR_HIOP, IRT_SOFTFP), nref, nref);
	break;
      case IR_FLOAD:
	lj_assertJ(ir->op1 == REF_NIL, "expected FLOAD from GG_State");
	hi = lj_ir_kint(J, *(int32_t*)((char*)J2GG(J) + ir->op2 + LJ_LE*4));
	nir->op2 += LJ_BE*4;
	break;
      case IR_XLOAD: {
	IRIns inslo = *nir;  /* Save/undo the emit of the lo XLOAD. */
	J->cur.nins--;
	hi = split_ptr(J, oir, ir->op1);  /* Insert the hiref ADD. */
#if LJ_BE
	hi = split_emit(J, IRT(IR_XLOAD, IRT_INT), hi, ir->op2);
	inslo.t.irt = IRT_SOFTFP | (inslo.t.irt & IRT_GUARD);
#endif
	nref = lj_ir_nextins(J);
	nir = IR(nref);
	*nir = inslo;  /* Re-emit lo XLOAD. */
#if LJ_LE
	hi = split_emit(J, IRT(IR_XLOAD, IRT_SOFTFP), hi, ir->op2);
	ir->prev = nref;
#else
	ir->prev = hi; hi = nref;
#endif
	break;
	}
      case IR_ASTORE: case IR_HSTORE: case IR_USTORE: case IR_XSTORE:
	split_emit(J, IRT(IR_HIOP, IRT_SOFTFP), nir->op1, hisubst[ir->op2]);
	break;
      case IR_CONV: {  /* Conversion to number. Others handled below. */
	IRType st = (IRType)(ir->op2 & IRCONV_SRCMASK);
	UNUSED(st);
#if LJ_32 && LJ_HASFFI
	if (st == IRT_I64 || st == IRT_U64) {
	  hi = split_call_l(J, hisubst, oir, ir,
		 st == IRT_I64 ? IRCALL_fp64_l2d : IRCALL_fp64_ul2d);
	  break;
	}
#endif
	lj_assertJ(st == IRT_INT ||
		   (LJ_32 && LJ_HASFFI && (st == IRT_U32 || st == IRT_FLOAT)),
		   "bad source type for CONV");
	nir->o = IR_CALLN;
#if LJ_32 && LJ_HASFFI
	nir->op2 = st == IRT_INT ? IRCALL_softfp_i2d :
		   st == IRT_FLOAT ? IRCALL_softfp_f2d :
		   IRCALL_softfp_ui2d;
#else
	nir->op2 = IRCALL_softfp_i2d;
#endif
	hi = split_emit(J, IRT(IR_HIOP, IRT_SOFTFP), nref, nref);
	break;
	}
      case IR_CALLN:
      case IR_CALLL:
      case IR_CALLS:
      case IR_CALLXS:
	goto split_call;
      case IR_PHI:
	if (nir->op1 == nir->op2)
	  J->cur.nins--;  /* Drop useless PHIs. */
	if (hisubst[ir->op1] != hisubst[ir->op2])
	  split_emit(J, IRT(IR_PHI, IRT_SOFTFP),
		     hisubst[ir->op1], hisubst[ir->op2]);
	break;
      case IR_HIOP:
	J->cur.nins--;  /* Drop joining HIOP. */
	ir->prev = nir->op1;
	hi = nir->op2;
	break;
      default:
	lj_assertJ(ir->o <= IR_NE || ir->o == IR_MIN || ir->o == IR_MAX,
		   "bad IR op %d", ir->o);
	hi = split_emit(J, IRTG(IR_HIOP, IRT_SOFTFP),
			hisubst[ir->op1], hisubst[ir->op2]);
	break;
      }
    } else
#endif
#if LJ_32 && LJ_HASFFI
    if (irt_isint64(ir->t)) {
      IRRef hiref = hisubst[ir->op1];
      nir->t.irt = IRT_INT | (nir->t.irt & IRT_GUARD);  /* Turn into INT op. */
      switch (ir->o) {
      case IR_ADD:
      case IR_SUB:
	/* Use plain op for hiword if loword cannot produce a carry/borrow. */
	if (irref_isk(nir->op2) && IR(nir->op2)->i == 0) {
	  ir->prev = nir->op1;  /* Pass through loword. */
	  nir->op1 = hiref; nir->op2 = hisubst[ir->op2];
	  hi = nref;
	  break;
	}
	/* fallthrough */
      case IR_NEG:
	hi = split_emit(J, IRTI(IR_HIOP), hiref, hisubst[ir->op2]);
	break;
      case IR_MUL:
	hi = split_call_ll(J, hisubst, oir, ir, IRCALL_lj_carith_mul64);
	break;
      case IR_DIV:
	hi = split_call_ll(J, hisubst, oir, ir,
			   irt_isi64(ir->t) ? IRCALL_lj_carith_divi64 :
					      IRCALL_lj_carith_divu64);
	break;
      case IR_MOD:
	hi = split_call_ll(J, hisubst, oir, ir,
			   irt_isi64(ir->t) ? IRCALL_lj_carith_modi64 :
					      IRCALL_lj_carith_modu64);
	break;
      case IR_POW:
	hi = split_call_ll(J, hisubst, oir, ir,
			   irt_isi64(ir->t) ? IRCALL_lj_carith_powi64 :
					      IRCALL_lj_carith_powu64);
	break;
      case IR_BNOT:
	hi = split_emit(J, IRTI(IR_BNOT), hiref, 0);
	break;
      case IR_BSWAP:
	ir->prev = split_emit(J, IRTI(IR_BSWAP), hiref, 0);
	hi = nref;
	break;
      case IR_BAND: case IR_BOR: case IR_BXOR:
	hi = split_bitop(J, hisubst, nir, ir);
	break;
      case IR_BSHL: case IR_BSHR: case IR_BSAR: case IR_BROL: case IR_BROR:
	hi = split_bitshift(J, hisubst, oir, nir, ir);
	break;
      case IR_FLOAD:
	lj_assertJ(ir->op2 == IRFL_CDATA_INT64, "only INT64 supported");
	hi = split_emit(J, IRTI(IR_FLOAD), nir->op1, IRFL_CDATA_INT64_4);
#if LJ_BE
	ir->prev = hi; hi = nref;
#endif
	break;
      case IR_XLOAD:
	hi = split_emit(J, IRTI(IR_XLOAD), split_ptr(J, oir, ir->op1), ir->op2);
#if LJ_BE
	ir->prev = hi; hi = nref;
#endif
	break;
      case IR_XSTORE:
	split_emit(J, IRTI(IR_HIOP), nir->op1, hisubst[ir->op2]);
	break;
      case IR_CONV: {  /* Conversion to 64 bit integer. Others handled below. */
	IRType st = (IRType)(ir->op2 & IRCONV_SRCMASK);
#if LJ_SOFTFP
	if (st == IRT_NUM) {  /* NUM to 64 bit int conv. */
	  hi = split_call_l(J, hisubst, oir, ir,
		 irt_isi64(ir->t) ? IRCALL_fp64_d2l : IRCALL_fp64_d2ul);
	} else if (st == IRT_FLOAT) {  /* FLOAT to 64 bit int conv. */
	  nir->o = IR_CALLN;
	  nir->op2 = irt_isi64(ir->t) ? IRCALL_fp64_f2l : IRCALL_fp64_f2ul;
	  hi = split_emit(J, IRTI(IR_HIOP), nref, nref);
	}
#else
	if (st == IRT_NUM || st == IRT_FLOAT) {  /* FP to 64 bit int conv. */
	  hi = split_emit(J, IRTI(IR_HIOP), nir->op1, nref);
	}
#endif
	else if (st == IRT_I64 || st == IRT_U64) {  /* 64/64 bit cast. */
	  /* Drop cast, since assembler doesn't care. But fwd both parts. */
	  hi = hiref;
	  goto fwdlo;
	} else if ((ir->op2 & IRCONV_SEXT)) {  /* Sign-extend to 64 bit. */
	  IRRef k31 = lj_ir_kint(J, 31);
	  nir = IR(nref);  /* May have been reallocated. */
	  ir->prev = nir->op1;  /* Pass through loword. */
	  nir->o = IR_BSAR;  /* hi = bsar(lo, 31). */
	  nir->op2 = k31;
	  hi = nref;
	} else {  /* Zero-extend to 64 bit. */
	  hi = lj_ir_kint(J, 0);
	  goto fwdlo;
	}
	break;
	}
      case IR_CALLXS:
	goto split_call;
      case IR_PHI: {
	IRRef hiref2;
	if ((irref_isk(nir->op1) && irref_isk(nir->op2)) ||
	    nir->op1 == nir->op2)
	  J->cur.nins--;  /* Drop useless PHIs. */
	hiref2 = hisubst[ir->op2];
	if (!((irref_isk(hiref) && irref_isk(hiref2)) || hiref == hiref2))
	  split_emit(J, IRTI(IR_PHI), hiref, hiref2);
	break;
	}
      case IR_HIOP:
	J->cur.nins--;  /* Drop joining HIOP. */
	ir->prev = nir->op1;
	hi = nir->op2;
	break;
      default:
	lj_assertJ(ir->o <= IR_NE, "bad IR op %d", ir->o);  /* Comparisons. */
	split_emit(J, IRTGI(IR_HIOP), hiref, hisubst[ir->op2]);
	break;
      }
    } else
#endif
#if LJ_SOFTFP
    if (ir->o == IR_SLOAD) {
      if ((nir->op2 & IRSLOAD_CONVERT)) {  /* Convert from number to int. */
	nir->op2 &= ~IRSLOAD_CONVERT;
	if (!(nir->op2 & IRSLOAD_TYPECHECK))
	  nir->t.irt = IRT_INT;  /* Drop guard. */
	split_emit(J, IRT(IR_HIOP, IRT_SOFTFP), nref, nref);
	ir->prev = split_num2int(J, nref, nref+1, irt_isguard(ir->t));
      }
    } else if (ir->o == IR_TOBIT) {
      IRRef tmp, op1 = ir->op1;
      J->cur.nins--;
#if LJ_LE
      tmp = split_emit(J, IRT(IR_CARG, IRT_NIL), oir[op1].prev, hisubst[op1]);
#else
      tmp = split_emit(J, IRT(IR_CARG, IRT_NIL), hisubst[op1], oir[op1].prev);
#endif
      ir->prev = split_emit(J, IRTI(IR_CALLN), tmp, IRCALL_lj_vm_tobit);
    } else if (ir->o == IR_TOSTR || ir->o == IR_TMPREF) {
      if (hisubst[ir->op1]) {
	if (irref_isk(ir->op1))
	  nir->op1 = ir->op1;
	else
	  split_emit(J, IRT(IR_HIOP, IRT_NIL), hisubst[ir->op1], nref);
      }
    } else if (ir->o == IR_HREF || ir->o == IR_NEWREF) {
      if (irref_isk(ir->op2) && hisubst[ir->op2])
	nir->op2 = ir->op2;
    } else
#endif
    if (ir->o == IR_CONV) {  /* See above, too. */
      IRType st = (IRType)(ir->op2 & IRCONV_SRCMASK);
#if LJ_32 && LJ_HASFFI
      if (st == IRT_I64 || st == IRT_U64) {  /* Conversion from 64 bit int. */
#if LJ_SOFTFP
	if (irt_isfloat(ir->t)) {
	  split_call_l(J, hisubst, oir, ir,
		       st == IRT_I64 ? IRCALL_fp64_l2f : IRCALL_fp64_ul2f);
	  J->cur.nins--;  /* Drop unused HIOP. */
	}
#else
	if (irt_isfp(ir->t)) {  /* 64 bit integer to FP conversion. */
	  ir->prev = split_emit(J, IRT(IR_HIOP, irt_type(ir->t)),
				hisubst[ir->op1], nref);
	}
#endif
	else {  /* Truncate to lower 32 bits. */
	fwdlo:
	  ir->prev = nir->op1;  /* Forward loword. */
	  /* Replace with NOP to avoid messing up the snapshot logic. */
	  nir->ot = IRT(IR_NOP, IRT_NIL);
	  nir->op1 = nir->op2 = 0;
	}
      }
#endif
#if LJ_SOFTFP && LJ_32 && LJ_HASFFI
      else if (irt_isfloat(ir->t)) {
	if (st == IRT_NUM) {
	  split_call_l(J, hisubst, oir, ir, IRCALL_softfp_d2f);
	  J->cur.nins--;  /* Drop unused HIOP. */
	} else {
	  nir->o = IR_CALLN;
	  nir->op2 = st == IRT_INT ? IRCALL_softfp_i2f : IRCALL_softfp_ui2f;
	}
      } else if (st == IRT_FLOAT) {
	nir->o = IR_CALLN;
	nir->op2 = irt_isint(ir->t) ? IRCALL_softfp_f2i : IRCALL_softfp_f2ui;
      } else
#endif
#if LJ_SOFTFP
      if (st == IRT_NUM || (LJ_32 && LJ_HASFFI && st == IRT_FLOAT)) {
	if (irt_isguard(ir->t)) {
	  lj_assertJ(st == IRT_NUM && irt_isint(ir->t), "bad CONV types");
	  J->cur.nins--;
	  ir->prev = split_num2int(J, nir->op1, hisubst[ir->op1], 1);
	} else {
	  split_call_l(J, hisubst, oir, ir,
#if LJ_32 && LJ_HASFFI
	    st == IRT_NUM ?
	      (irt_isint(ir->t) ? IRCALL_softfp_d2i : IRCALL_softfp_d2ui) :
	      (irt_isint(ir->t) ? IRCALL_softfp_f2i : IRCALL_softfp_f2ui)
#else
	    IRCALL_softfp_d2i
#endif
	  );
	  J->cur.nins--;  /* Drop unused HIOP. */
	}
      }
#endif
    } else if (ir->o == IR_CALLXS) {
      IRRef hiref;
    split_call:
      hiref = hisubst[ir->op1];
      if (hiref) {
	IROpT ot = nir->ot;
	IRRef op2 = nir->op2;
	nir->ot = IRT(IR_CARG, IRT_NIL);
#if LJ_LE
	nir->op2 = hiref;
#else
	nir->op2 = nir->op1; nir->op1 = hiref;
#endif
	ir->prev = nref = split_emit(J, ot, nref, op2);
      }
      if (LJ_SOFTFP ? irt_is64(ir->t) : irt_isint64(ir->t))
	hi = split_emit(J,
	  IRT(IR_HIOP, (LJ_SOFTFP && irt_isnum(ir->t)) ? IRT_SOFTFP : IRT_INT),
	  nref, nref);
    } else if (ir->o == IR_CARG) {
      IRRef hiref = hisubst[ir->op1];
      if (hiref) {
	IRRef op2 = nir->op2;
#if LJ_LE
	nir->op2 = hiref;
#else
	nir->op2 = nir->op1; nir->op1 = hiref;
#endif
	ir->prev = nref = split_emit(J, IRT(IR_CARG, IRT_NIL), nref, op2);
	nir = IR(nref);
      }
      hiref = hisubst[ir->op2];
      if (hiref) {
#if !LJ_TARGET_X86
	int carg = 0;
	IRIns *cir;
	for (cir = IR(nir->op1); cir->o == IR_CARG; cir = IR(cir->op1))
	  carg++;
	if ((carg & 1) == 0) {  /* Align 64 bit arguments. */
	  IRRef op2 = nir->op2;
	  nir->op2 = REF_NIL;
	  nref = split_emit(J, IRT(IR_CARG, IRT_NIL), nref, op2);
	  nir = IR(nref);
	}
#endif
#if LJ_BE
	{ IRRef tmp = nir->op2; nir->op2 = hiref; hiref = tmp; }
#endif
	ir->prev = split_emit(J, IRT(IR_CARG, IRT_NIL), nref, hiref);
      }
    } else if (ir->o == IR_CNEWI) {
      if (hisubst[ir->op2])
	split_emit(J, IRT(IR_HIOP, IRT_NIL), nref, hisubst[ir->op2]);
    } else if (ir->o == IR_LOOP) {
      J->loopref = nref;  /* Needed by assembler. */
    }
    hisubst[ref] = hi;  /* Store hiword substitution. */
  }
  if (snref == nins) {  /* Substitution for last snapshot. */
    snap->ref = J->cur.nins;
    split_subst_snap(J, snap, oir);
  }

  /* Add PHI marks. */
  for (ref = J->cur.nins-1; ref >= REF_FIRST; ref--) {
    IRIns *ir = IR(ref);
    if (ir->o != IR_PHI) break;
    if (!irref_isk(ir->op1)) irt_setphi(IR(ir->op1)->t);
    if (ir->op2 > J->loopref) irt_setphi(IR(ir->op2)->t);
  }
}

/* Protected callback for split pass. */
static TValue *cpsplit(lua_State *L, lua_CFunction dummy, void *ud)
{
  jit_State *J = (jit_State *)ud;
  split_ir(J);
  UNUSED(L); UNUSED(dummy);
  return NULL;
}

#if defined(LUA_USE_ASSERT) || LJ_SOFTFP
/* Slow, but sure way to check whether a SPLIT pass is needed. */
static int split_needsplit(jit_State *J)
{
  IRIns *ir, *irend;
  IRRef ref;
  for (ir = IR(REF_FIRST), irend = IR(J->cur.nins); ir < irend; ir++)
    if (LJ_SOFTFP ? irt_is64orfp(ir->t) : irt_isint64(ir->t))
      return 1;
  if (LJ_SOFTFP) {
    for (ref = J->chain[IR_SLOAD]; ref; ref = IR(ref)->prev)
      if ((IR(ref)->op2 & IRSLOAD_CONVERT))
	return 1;
    if (J->chain[IR_TOBIT])
      return 1;
  }
  for (ref = J->chain[IR_CONV]; ref; ref = IR(ref)->prev) {
    IRType st = (IR(ref)->op2 & IRCONV_SRCMASK);
    if ((LJ_SOFTFP && (st == IRT_NUM || st == IRT_FLOAT)) ||
	st == IRT_I64 || st == IRT_U64)
      return 1;
  }
  return 0;  /* Nope. */
}
#endif

/* SPLIT pass. */
void lj_opt_split(jit_State *J)
{
#if LJ_SOFTFP
  if (!J->needsplit)
    J->needsplit = split_needsplit(J);
#else
  lj_assertJ(J->needsplit >= split_needsplit(J), "bad SPLIT state");
#endif
  if (J->needsplit) {
    int errcode = lj_vm_cpcall(J->L, NULL, J, cpsplit);
    if (errcode) {
      /* Completely reset the trace to avoid inconsistent dump on abort. */
      J->cur.nins = J->cur.nk = REF_BASE;
      J->cur.nsnap = 0;
      lj_err_throw(J->L, errcode);  /* Propagate errors. */
    }
  }
}

#undef IR

#endif
