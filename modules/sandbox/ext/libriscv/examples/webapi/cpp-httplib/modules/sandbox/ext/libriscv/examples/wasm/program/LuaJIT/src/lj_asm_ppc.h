/*
** PPC IR assembler (SSA IR -> machine code).
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
*/

/* -- Register allocator extensions --------------------------------------- */

/* Allocate a register with a hint. */
static Reg ra_hintalloc(ASMState *as, IRRef ref, Reg hint, RegSet allow)
{
  Reg r = IR(ref)->r;
  if (ra_noreg(r)) {
    if (!ra_hashint(r) && !iscrossref(as, ref))
      ra_sethint(IR(ref)->r, hint);  /* Propagate register hint. */
    r = ra_allocref(as, ref, allow);
  }
  ra_noweak(as, r);
  return r;
}

/* Allocate two source registers for three-operand instructions. */
static Reg ra_alloc2(ASMState *as, IRIns *ir, RegSet allow)
{
  IRIns *irl = IR(ir->op1), *irr = IR(ir->op2);
  Reg left = irl->r, right = irr->r;
  if (ra_hasreg(left)) {
    ra_noweak(as, left);
    if (ra_noreg(right))
      right = ra_allocref(as, ir->op2, rset_exclude(allow, left));
    else
      ra_noweak(as, right);
  } else if (ra_hasreg(right)) {
    ra_noweak(as, right);
    left = ra_allocref(as, ir->op1, rset_exclude(allow, right));
  } else if (ra_hashint(right)) {
    right = ra_allocref(as, ir->op2, allow);
    left = ra_alloc1(as, ir->op1, rset_exclude(allow, right));
  } else {
    left = ra_allocref(as, ir->op1, allow);
    right = ra_alloc1(as, ir->op2, rset_exclude(allow, left));
  }
  return left | (right << 8);
}

/* -- Guard handling ------------------------------------------------------ */

/* Setup exit stubs after the end of each trace. */
static void asm_exitstub_setup(ASMState *as, ExitNo nexits)
{
  ExitNo i;
  MCode *mxp = as->mctop;
  if (mxp - (nexits + 3 + MCLIM_REDZONE) < as->mclim)
    asm_mclimit(as);
  /* 1: mflr r0; bl ->vm_exit_handler; li r0, traceno; bl <1; bl <1; ... */
  for (i = nexits-1; (int32_t)i >= 0; i--)
    *--mxp = PPCI_BL|(((-3-i)&0x00ffffffu)<<2);
  *--mxp = PPCI_LI|PPCF_T(RID_TMP)|as->T->traceno;  /* Read by exit handler. */
  mxp--;
  *mxp = PPCI_BL|((((MCode *)(void *)lj_vm_exit_handler-mxp)&0x00ffffffu)<<2);
  *--mxp = PPCI_MFLR|PPCF_T(RID_TMP);
  as->mctop = mxp;
}

static MCode *asm_exitstub_addr(ASMState *as, ExitNo exitno)
{
  /* Keep this in-sync with exitstub_trace_addr(). */
  return as->mctop + exitno + 3;
}

/* Emit conditional branch to exit for guard. */
static void asm_guardcc(ASMState *as, PPCCC cc)
{
  MCode *target = asm_exitstub_addr(as, as->snapno);
  MCode *p = as->mcp;
  if (LJ_UNLIKELY(p == as->invmcp)) {
    as->loopinv = 1;
    *p = PPCI_B | (((target-p) & 0x00ffffffu) << 2);
    emit_condbranch(as, PPCI_BC, cc^4, p);
    return;
  }
  emit_condbranch(as, PPCI_BC, cc, target);
}

/* -- Operand fusion ------------------------------------------------------ */

/* Limit linear search to this distance. Avoids O(n^2) behavior. */
#define CONFLICT_SEARCH_LIM	31

/* Check if there's no conflicting instruction between curins and ref. */
static int noconflict(ASMState *as, IRRef ref, IROp conflict)
{
  IRIns *ir = as->ir;
  IRRef i = as->curins;
  if (i > ref + CONFLICT_SEARCH_LIM)
    return 0;  /* Give up, ref is too far away. */
  while (--i > ref)
    if (ir[i].o == conflict)
      return 0;  /* Conflict found. */
  return 1;  /* Ok, no conflict. */
}

/* Fuse the array base of colocated arrays. */
static int32_t asm_fuseabase(ASMState *as, IRRef ref)
{
  IRIns *ir = IR(ref);
  if (ir->o == IR_TNEW && ir->op1 <= LJ_MAX_COLOSIZE &&
      !neverfuse(as) && noconflict(as, ref, IR_NEWREF))
    return (int32_t)sizeof(GCtab);
  return 0;
}

/* Indicates load/store indexed is ok. */
#define AHUREF_LSX	((int32_t)0x80000000)

/* Fuse array/hash/upvalue reference into register+offset operand. */
static Reg asm_fuseahuref(ASMState *as, IRRef ref, int32_t *ofsp, RegSet allow)
{
  IRIns *ir = IR(ref);
  if (ra_noreg(ir->r)) {
    if (ir->o == IR_AREF) {
      if (mayfuse(as, ref)) {
	if (irref_isk(ir->op2)) {
	  IRRef tab = IR(ir->op1)->op1;
	  int32_t ofs = asm_fuseabase(as, tab);
	  IRRef refa = ofs ? tab : ir->op1;
	  ofs += 8*IR(ir->op2)->i;
	  if (checki16(ofs)) {
	    *ofsp = ofs;
	    return ra_alloc1(as, refa, allow);
	  }
	}
	if (*ofsp == AHUREF_LSX) {
	  Reg base = ra_alloc1(as, ir->op1, allow);
	  Reg idx = ra_alloc1(as, ir->op2, rset_exclude(RSET_GPR, base));
	  return base | (idx << 8);
	}
      }
    } else if (ir->o == IR_HREFK) {
      if (mayfuse(as, ref)) {
	int32_t ofs = (int32_t)(IR(ir->op2)->op2 * sizeof(Node));
	if (checki16(ofs)) {
	  *ofsp = ofs;
	  return ra_alloc1(as, ir->op1, allow);
	}
      }
    } else if (ir->o == IR_UREFC) {
      if (irref_isk(ir->op1)) {
	GCfunc *fn = ir_kfunc(IR(ir->op1));
	int32_t ofs = i32ptr(&gcref(fn->l.uvptr[(ir->op2 >> 8)])->uv.tv);
	int32_t jgl = (intptr_t)J2G(as->J);
	if ((uint32_t)(ofs-jgl) < 65536) {
	  *ofsp = ofs-jgl-32768;
	  return RID_JGL;
	} else {
	  *ofsp = (int16_t)ofs;
	  return ra_allock(as, ofs-(int16_t)ofs, allow);
	}
      }
    } else if (ir->o == IR_TMPREF) {
      *ofsp = (int32_t)(offsetof(global_State, tmptv)-32768);
      return RID_JGL;
    }
  }
  *ofsp = 0;
  return ra_alloc1(as, ref, allow);
}

/* Fuse XLOAD/XSTORE reference into load/store operand. */
static void asm_fusexref(ASMState *as, PPCIns pi, Reg rt, IRRef ref,
			 RegSet allow, int32_t ofs)
{
  IRIns *ir = IR(ref);
  Reg base;
  if (ra_noreg(ir->r) && canfuse(as, ir)) {
    if (ir->o == IR_ADD) {
      int32_t ofs2;
      if (irref_isk(ir->op2) && (ofs2 = ofs + IR(ir->op2)->i, checki16(ofs2))) {
	ofs = ofs2;
	ref = ir->op1;
      } else if (ofs == 0) {
	Reg right, left = ra_alloc2(as, ir, allow);
	right = (left >> 8); left &= 255;
	emit_fab(as, PPCI_LWZX | ((pi >> 20) & 0x780), rt, left, right);
	return;
      }
    } else if (ir->o == IR_STRREF) {
      lj_assertA(ofs == 0, "bad usage");
      ofs = (int32_t)sizeof(GCstr);
      if (irref_isk(ir->op2)) {
	ofs += IR(ir->op2)->i;
	ref = ir->op1;
      } else if (irref_isk(ir->op1)) {
	ofs += IR(ir->op1)->i;
	ref = ir->op2;
      } else {
	/* NYI: Fuse ADD with constant. */
	Reg tmp, right, left = ra_alloc2(as, ir, allow);
	right = (left >> 8); left &= 255;
	tmp = ra_scratch(as, rset_exclude(rset_exclude(allow, left), right));
	emit_fai(as, pi, rt, tmp, ofs);
	emit_tab(as, PPCI_ADD, tmp, left, right);
	return;
      }
      if (!checki16(ofs)) {
	Reg left = ra_alloc1(as, ref, allow);
	Reg right = ra_allock(as, ofs, rset_exclude(allow, left));
	emit_fab(as, PPCI_LWZX | ((pi >> 20) & 0x780), rt, left, right);
	return;
      }
    }
  }
  base = ra_alloc1(as, ref, allow);
  emit_fai(as, pi, rt, base, ofs);
}

/* Fuse XLOAD/XSTORE reference into indexed-only load/store operand. */
static void asm_fusexrefx(ASMState *as, PPCIns pi, Reg rt, IRRef ref,
			  RegSet allow)
{
  IRIns *ira = IR(ref);
  Reg right, left;
  if (canfuse(as, ira) && ira->o == IR_ADD && ra_noreg(ira->r)) {
    left = ra_alloc2(as, ira, allow);
    right = (left >> 8); left &= 255;
  } else {
    right = ra_alloc1(as, ref, allow);
    left = RID_R0;
  }
  emit_tab(as, pi, rt, left, right);
}

#if !LJ_SOFTFP
/* Fuse to multiply-add/sub instruction. */
static int asm_fusemadd(ASMState *as, IRIns *ir, PPCIns pi, PPCIns pir)
{
  IRRef lref = ir->op1, rref = ir->op2;
  IRIns *irm;
  if ((as->flags & JIT_F_OPT_FMA) &&
      lref != rref &&
      ((mayfuse(as, lref) && (irm = IR(lref), irm->o == IR_MUL) &&
	ra_noreg(irm->r)) ||
       (mayfuse(as, rref) && (irm = IR(rref), irm->o == IR_MUL) &&
	(rref = lref, pi = pir, ra_noreg(irm->r))))) {
    Reg dest = ra_dest(as, ir, RSET_FPR);
    Reg add = ra_alloc1(as, rref, RSET_FPR);
    Reg right, left = ra_alloc2(as, irm, rset_exclude(RSET_FPR, add));
    right = (left >> 8); left &= 255;
    emit_facb(as, pi, dest, left, right, add);
    return 1;
  }
  return 0;
}
#endif

/* -- Calls --------------------------------------------------------------- */

/* Generate a call to a C function. */
static void asm_gencall(ASMState *as, const CCallInfo *ci, IRRef *args)
{
  uint32_t n, nargs = CCI_XNARGS(ci);
  int32_t ofs = 8;
  Reg gpr = REGARG_FIRSTGPR;
#if !LJ_SOFTFP
  Reg fpr = REGARG_FIRSTFPR;
#endif
  if ((void *)ci->func)
    emit_call(as, (void *)ci->func);
  for (n = 0; n < nargs; n++) {  /* Setup args. */
    IRRef ref = args[n];
    if (ref) {
      IRIns *ir = IR(ref);
#if !LJ_SOFTFP
      if (irt_isfp(ir->t)) {
	if (fpr <= REGARG_LASTFPR) {
	  lj_assertA(rset_test(as->freeset, fpr),
		     "reg %d not free", fpr);  /* Already evicted. */
	  ra_leftov(as, fpr, ref);
	  fpr++;
	} else {
	  Reg r = ra_alloc1(as, ref, RSET_FPR);
	  if (irt_isnum(ir->t)) ofs = (ofs + 4) & ~4;
	  emit_spstore(as, ir, r, ofs);
	  ofs += irt_isnum(ir->t) ? 8 : 4;
	}
      } else
#endif
      {
	if (gpr <= REGARG_LASTGPR) {
	  lj_assertA(rset_test(as->freeset, gpr),
		     "reg %d not free", gpr);  /* Already evicted. */
	  ra_leftov(as, gpr, ref);
	  gpr++;
	} else {
	  Reg r = ra_alloc1(as, ref, RSET_GPR);
	  emit_spstore(as, ir, r, ofs);
	  ofs += 4;
	}
      }
    } else {
      if (gpr <= REGARG_LASTGPR)
	gpr++;
      else
	ofs += 4;
    }
    checkmclim(as);
  }
#if !LJ_SOFTFP
  if ((ci->flags & CCI_VARARG))  /* Vararg calls need to know about FPR use. */
    emit_tab(as, fpr == REGARG_FIRSTFPR ? PPCI_CRXOR : PPCI_CREQV, 6, 6, 6);
#endif
}

/* Setup result reg/sp for call. Evict scratch regs. */
static void asm_setupresult(ASMState *as, IRIns *ir, const CCallInfo *ci)
{
  RegSet drop = RSET_SCRATCH;
  int hiop = ((ir+1)->o == IR_HIOP && !irt_isnil((ir+1)->t));
#if !LJ_SOFTFP
  if ((ci->flags & CCI_NOFPRCLOBBER))
    drop &= ~RSET_FPR;
#endif
  if (ra_hasreg(ir->r))
    rset_clear(drop, ir->r);  /* Dest reg handled below. */
  if (hiop && ra_hasreg((ir+1)->r))
    rset_clear(drop, (ir+1)->r);  /* Dest reg handled below. */
  ra_evictset(as, drop);  /* Evictions must be performed first. */
  if (ra_used(ir)) {
    lj_assertA(!irt_ispri(ir->t), "PRI dest");
    if (!LJ_SOFTFP && irt_isfp(ir->t)) {
      if ((ci->flags & CCI_CASTU64)) {
	/* Use spill slot or temp slots. */
	int32_t ofs = ir->s ? sps_scale(ir->s) : SPOFS_TMP;
	Reg dest = ir->r;
	if (ra_hasreg(dest)) {
	  ra_free(as, dest);
	  ra_modified(as, dest);
	  emit_fai(as, PPCI_LFD, dest, RID_SP, ofs);
	}
	emit_tai(as, PPCI_STW, RID_RETHI, RID_SP, ofs);
	emit_tai(as, PPCI_STW, RID_RETLO, RID_SP, ofs+4);
      } else {
	ra_destreg(as, ir, RID_FPRET);
      }
    } else if (hiop) {
      ra_destpair(as, ir);
    } else {
      ra_destreg(as, ir, RID_RET);
    }
  }
}

static void asm_callx(ASMState *as, IRIns *ir)
{
  IRRef args[CCI_NARGS_MAX*2];
  CCallInfo ci;
  IRRef func;
  IRIns *irf;
  ci.flags = asm_callx_flags(as, ir);
  asm_collectargs(as, ir, &ci, args);
  asm_setupresult(as, ir, &ci);
  func = ir->op2; irf = IR(func);
  if (irf->o == IR_CARG) { func = irf->op1; irf = IR(func); }
  if (irref_isk(func)) {  /* Call to constant address. */
    ci.func = (ASMFunction)(void *)(intptr_t)(irf->i);
  } else {  /* Need a non-argument register for indirect calls. */
    RegSet allow = RSET_GPR & ~RSET_RANGE(RID_R0, REGARG_LASTGPR+1);
    Reg freg = ra_alloc1(as, func, allow);
    *--as->mcp = PPCI_BCTRL;
    *--as->mcp = PPCI_MTCTR | PPCF_T(freg);
    ci.func = (ASMFunction)(void *)0;
  }
  asm_gencall(as, &ci, args);
}

/* -- Returns ------------------------------------------------------------- */

/* Return to lower frame. Guard that it goes to the right spot. */
static void asm_retf(ASMState *as, IRIns *ir)
{
  Reg base = ra_alloc1(as, REF_BASE, RSET_GPR);
  void *pc = ir_kptr(IR(ir->op2));
  int32_t delta = 1+LJ_FR2+bc_a(*((const BCIns *)pc - 1));
  as->topslot -= (BCReg)delta;
  if ((int32_t)as->topslot < 0) as->topslot = 0;
  irt_setmark(IR(REF_BASE)->t);  /* Children must not coalesce with BASE reg. */
  emit_setgl(as, base, jit_base);
  emit_addptr(as, base, -8*delta);
  asm_guardcc(as, CC_NE);
  emit_ab(as, PPCI_CMPW, RID_TMP,
	  ra_allock(as, i32ptr(pc), rset_exclude(RSET_GPR, base)));
  emit_tai(as, PPCI_LWZ, RID_TMP, base, -8);
}

/* -- Buffer operations --------------------------------------------------- */

#if LJ_HASBUFFER
static void asm_bufhdr_write(ASMState *as, Reg sb)
{
  Reg tmp = ra_scratch(as, rset_exclude(RSET_GPR, sb));
  IRIns irgc;
  irgc.ot = IRT(0, IRT_PGC);  /* GC type. */
  emit_storeofs(as, &irgc, RID_TMP, sb, offsetof(SBuf, L));
  emit_rot(as, PPCI_RLWIMI, RID_TMP, tmp, 0, 31-lj_fls(SBUF_MASK_FLAG), 31);
  emit_getgl(as, RID_TMP, cur_L);
  emit_loadofs(as, &irgc, tmp, sb, offsetof(SBuf, L));
}
#endif

/* -- Type conversions ---------------------------------------------------- */

#if !LJ_SOFTFP
static void asm_tointg(ASMState *as, IRIns *ir, Reg left)
{
  RegSet allow = RSET_FPR;
  Reg tmp = ra_scratch(as, rset_clear(allow, left));
  Reg fbias = ra_scratch(as, rset_clear(allow, tmp));
  Reg dest = ra_dest(as, ir, RSET_GPR);
  Reg hibias = ra_allock(as, 0x43300000, rset_exclude(RSET_GPR, dest));
  asm_guardcc(as, CC_NE);
  emit_fab(as, PPCI_FCMPU, 0, tmp, left);
  emit_fab(as, PPCI_FSUB, tmp, tmp, fbias);
  emit_fai(as, PPCI_LFD, tmp, RID_SP, SPOFS_TMP);
  emit_tai(as, PPCI_STW, RID_TMP, RID_SP, SPOFS_TMPLO);
  emit_tai(as, PPCI_STW, hibias, RID_SP, SPOFS_TMPHI);
  emit_asi(as, PPCI_XORIS, RID_TMP, dest, 0x8000);
  emit_tai(as, PPCI_LWZ, dest, RID_SP, SPOFS_TMPLO);
  emit_lsptr(as, PPCI_LFS, (fbias & 31),
	     (void *)&as->J->k32[LJ_K32_2P52_2P31], RSET_GPR);
  emit_fai(as, PPCI_STFD, tmp, RID_SP, SPOFS_TMP);
  emit_fb(as, PPCI_FCTIWZ, tmp, left);
}

static void asm_tobit(ASMState *as, IRIns *ir)
{
  RegSet allow = RSET_FPR;
  Reg dest = ra_dest(as, ir, RSET_GPR);
  Reg left = ra_alloc1(as, ir->op1, allow);
  Reg right = ra_alloc1(as, ir->op2, rset_clear(allow, left));
  Reg tmp = ra_scratch(as, rset_clear(allow, right));
  emit_tai(as, PPCI_LWZ, dest, RID_SP, SPOFS_TMPLO);
  emit_fai(as, PPCI_STFD, tmp, RID_SP, SPOFS_TMP);
  emit_fab(as, PPCI_FADD, tmp, left, right);
}
#endif

static void asm_conv(ASMState *as, IRIns *ir)
{
  IRType st = (IRType)(ir->op2 & IRCONV_SRCMASK);
#if !LJ_SOFTFP
  int stfp = (st == IRT_NUM || st == IRT_FLOAT);
#endif
  IRRef lref = ir->op1;
  /* 64 bit integer conversions are handled by SPLIT. */
  lj_assertA(!(irt_isint64(ir->t) || (st == IRT_I64 || st == IRT_U64)),
	     "IR %04d has unsplit 64 bit type",
	     (int)(ir - as->ir) - REF_BIAS);
#if LJ_SOFTFP
  /* FP conversions are handled by SPLIT. */
  lj_assertA(!irt_isfp(ir->t) && !(st == IRT_NUM || st == IRT_FLOAT),
	     "IR %04d has FP type",
	     (int)(ir - as->ir) - REF_BIAS);
  /* Can't check for same types: SPLIT uses CONV int.int + BXOR for sfp NEG. */
#else
  lj_assertA(irt_type(ir->t) != st, "inconsistent types for CONV");
  if (irt_isfp(ir->t)) {
    Reg dest = ra_dest(as, ir, RSET_FPR);
    if (stfp) {  /* FP to FP conversion. */
      if (st == IRT_NUM)  /* double -> float conversion. */
	emit_fb(as, PPCI_FRSP, dest, ra_alloc1(as, lref, RSET_FPR));
      else  /* float -> double conversion is a no-op on PPC. */
	ra_leftov(as, dest, lref);  /* Do nothing, but may need to move regs. */
    } else {  /* Integer to FP conversion. */
      /* IRT_INT: Flip hibit, bias with 2^52, subtract 2^52+2^31. */
      /* IRT_U32: Bias with 2^52, subtract 2^52. */
      RegSet allow = RSET_GPR;
      Reg left = ra_alloc1(as, lref, allow);
      Reg hibias = ra_allock(as, 0x43300000, rset_clear(allow, left));
      Reg fbias = ra_scratch(as, rset_exclude(RSET_FPR, dest));
      if (irt_isfloat(ir->t)) emit_fb(as, PPCI_FRSP, dest, dest);
      emit_fab(as, PPCI_FSUB, dest, dest, fbias);
      emit_fai(as, PPCI_LFD, dest, RID_SP, SPOFS_TMP);
      emit_lsptr(as, PPCI_LFS, (fbias & 31),
		 &as->J->k32[st == IRT_U32 ? LJ_K32_2P52 : LJ_K32_2P52_2P31],
		 rset_clear(allow, hibias));
      emit_tai(as, PPCI_STW, st == IRT_U32 ? left : RID_TMP,
	       RID_SP, SPOFS_TMPLO);
      emit_tai(as, PPCI_STW, hibias, RID_SP, SPOFS_TMPHI);
      if (st != IRT_U32) emit_asi(as, PPCI_XORIS, RID_TMP, left, 0x8000);
    }
  } else if (stfp) {  /* FP to integer conversion. */
    if (irt_isguard(ir->t)) {
      /* Checked conversions are only supported from number to int. */
      lj_assertA(irt_isint(ir->t) && st == IRT_NUM,
		 "bad type for checked CONV");
      asm_tointg(as, ir, ra_alloc1(as, lref, RSET_FPR));
    } else {
      Reg dest = ra_dest(as, ir, RSET_GPR);
      Reg left = ra_alloc1(as, lref, RSET_FPR);
      Reg tmp = ra_scratch(as, rset_exclude(RSET_FPR, left));
      if (irt_isu32(ir->t)) {
	/* Convert both x and x-2^31 to int and merge results. */
	Reg tmpi = ra_scratch(as, rset_exclude(RSET_GPR, dest));
	emit_asb(as, PPCI_OR, dest, dest, tmpi);  /* Select with mask idiom. */
	emit_asb(as, PPCI_AND, tmpi, tmpi, RID_TMP);
	emit_asb(as, PPCI_ANDC, dest, dest, RID_TMP);
	emit_tai(as, PPCI_LWZ, tmpi, RID_SP, SPOFS_TMPLO);  /* tmp = (int)(x) */
	emit_tai(as, PPCI_ADDIS, dest, dest, 0x8000);  /* dest += 2^31 */
	emit_asb(as, PPCI_SRAWI, RID_TMP, dest, 31);  /* mask = -(dest < 0) */
	emit_fai(as, PPCI_STFD, tmp, RID_SP, SPOFS_TMP);
	emit_tai(as, PPCI_LWZ, dest,
		 RID_SP, SPOFS_TMPLO);  /* dest = (int)(x-2^31) */
	emit_fb(as, PPCI_FCTIWZ, tmp, left);
	emit_fai(as, PPCI_STFD, tmp, RID_SP, SPOFS_TMP);
	emit_fb(as, PPCI_FCTIWZ, tmp, tmp);
	emit_fab(as, PPCI_FSUB, tmp, left, tmp);
	emit_lsptr(as, PPCI_LFS, (tmp & 31),
		   (void *)&as->J->k32[LJ_K32_2P31], RSET_GPR);
      } else {
	emit_tai(as, PPCI_LWZ, dest, RID_SP, SPOFS_TMPLO);
	emit_fai(as, PPCI_STFD, tmp, RID_SP, SPOFS_TMP);
	emit_fb(as, PPCI_FCTIWZ, tmp, left);
      }
    }
  } else
#endif
  {
    Reg dest = ra_dest(as, ir, RSET_GPR);
    if (st >= IRT_I8 && st <= IRT_U16) {  /* Extend to 32 bit integer. */
      Reg left = ra_alloc1(as, ir->op1, RSET_GPR);
      lj_assertA(irt_isint(ir->t) || irt_isu32(ir->t), "bad type for CONV EXT");
      if ((ir->op2 & IRCONV_SEXT))
	emit_as(as, st == IRT_I8 ? PPCI_EXTSB : PPCI_EXTSH, dest, left);
      else
	emit_rot(as, PPCI_RLWINM, dest, left, 0, st == IRT_U8 ? 24 : 16, 31);
    } else {  /* 32/64 bit integer conversions. */
      /* Only need to handle 32/32 bit no-op (cast) on 32 bit archs. */
      ra_leftov(as, dest, lref);  /* Do nothing, but may need to move regs. */
    }
  }
}

static void asm_strto(ASMState *as, IRIns *ir)
{
  const CCallInfo *ci = &lj_ir_callinfo[IRCALL_lj_strscan_num];
  IRRef args[2];
  int32_t ofs = SPOFS_TMP;
#if LJ_SOFTFP
  ra_evictset(as, RSET_SCRATCH);
  if (ra_used(ir)) {
    if (ra_hasspill(ir->s) && ra_hasspill((ir+1)->s) &&
	(ir->s & 1) == LJ_BE && (ir->s ^ 1) == (ir+1)->s) {
      int i;
      for (i = 0; i < 2; i++) {
	Reg r = (ir+i)->r;
	if (ra_hasreg(r)) {
	  ra_free(as, r);
	  ra_modified(as, r);
	  emit_spload(as, ir+i, r, sps_scale((ir+i)->s));
	}
      }
      ofs = sps_scale(ir->s & ~1);
    } else {
      Reg rhi = ra_dest(as, ir+1, RSET_GPR);
      Reg rlo = ra_dest(as, ir, rset_exclude(RSET_GPR, rhi));
      emit_tai(as, PPCI_LWZ, rhi, RID_SP, ofs);
      emit_tai(as, PPCI_LWZ, rlo, RID_SP, ofs+4);
    }
  }
#else
  RegSet drop = RSET_SCRATCH;
  if (ra_hasreg(ir->r)) rset_set(drop, ir->r);  /* Spill dest reg (if any). */
  ra_evictset(as, drop);
  if (ir->s) ofs = sps_scale(ir->s);
#endif
  asm_guardcc(as, CC_EQ);
  emit_ai(as, PPCI_CMPWI, RID_RET, 0);  /* Test return status. */
  args[0] = ir->op1;      /* GCstr *str */
  args[1] = ASMREF_TMP1;  /* TValue *n  */
  asm_gencall(as, ci, args);
  /* Store the result to the spill slot or temp slots. */
  emit_tai(as, PPCI_ADDI, ra_releasetmp(as, ASMREF_TMP1), RID_SP, ofs);
}

/* -- Memory references --------------------------------------------------- */

/* Get pointer to TValue. */
static void asm_tvptr(ASMState *as, Reg dest, IRRef ref, MSize mode)
{
  int32_t tmpofs = (int32_t)(offsetof(global_State, tmptv)-32768);
  if ((mode & IRTMPREF_IN1)) {
    IRIns *ir = IR(ref);
    if (irt_isnum(ir->t)) {
      if ((mode & IRTMPREF_OUT1)) {
#if LJ_SOFTFP
	lj_assertA(irref_isk(ref), "unsplit FP op");
	emit_tai(as, PPCI_ADDI, dest, RID_JGL, tmpofs);
	emit_setgl(as,
		   ra_allock(as, (int32_t)ir_knum(ir)->u32.lo, RSET_GPR),
		   tmptv.u32.lo);
	emit_setgl(as,
		   ra_allock(as, (int32_t)ir_knum(ir)->u32.hi, RSET_GPR),
		   tmptv.u32.hi);
#else
	Reg src = ra_alloc1(as, ref, RSET_FPR);
	emit_tai(as, PPCI_ADDI, dest, RID_JGL, tmpofs);
	emit_fai(as, PPCI_STFD, src, RID_JGL, tmpofs);
#endif
      } else if (irref_isk(ref)) {
	/* Use the number constant itself as a TValue. */
	ra_allockreg(as, i32ptr(ir_knum(ir)), dest);
      } else {
#if LJ_SOFTFP
	lj_assertA(0, "unsplit FP op");
#else
	/* Otherwise force a spill and use the spill slot. */
	emit_tai(as, PPCI_ADDI, dest, RID_SP, ra_spill(as, ir));
#endif
      }
    } else {
      /* Otherwise use g->tmptv to hold the TValue. */
      Reg type;
      emit_tai(as, PPCI_ADDI, dest, RID_JGL, tmpofs);
      if (!irt_ispri(ir->t)) {
	Reg src = ra_alloc1(as, ref, RSET_GPR);
	emit_setgl(as, src, tmptv.gcr);
      }
      if (LJ_SOFTFP && (ir+1)->o == IR_HIOP && !irt_isnil((ir+1)->t))
	type = ra_alloc1(as, ref+1, RSET_GPR);
      else
	type = ra_allock(as, irt_toitype(ir->t), RSET_GPR);
      emit_setgl(as, type, tmptv.it);
    }
  } else {
    emit_tai(as, PPCI_ADDI, dest, RID_JGL, tmpofs);
  }
}

static void asm_aref(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  Reg idx, base;
  if (irref_isk(ir->op2)) {
    IRRef tab = IR(ir->op1)->op1;
    int32_t ofs = asm_fuseabase(as, tab);
    IRRef refa = ofs ? tab : ir->op1;
    ofs += 8*IR(ir->op2)->i;
    if (checki16(ofs)) {
      base = ra_alloc1(as, refa, RSET_GPR);
      emit_tai(as, PPCI_ADDI, dest, base, ofs);
      return;
    }
  }
  base = ra_alloc1(as, ir->op1, RSET_GPR);
  idx = ra_alloc1(as, ir->op2, rset_exclude(RSET_GPR, base));
  emit_tab(as, PPCI_ADD, dest, RID_TMP, base);
  emit_slwi(as, RID_TMP, idx, 3);
}

/* Inlined hash lookup. Specialized for key type and for const keys.
** The equivalent C code is:
**   Node *n = hashkey(t, key);
**   do {
**     if (lj_obj_equal(&n->key, key)) return &n->val;
**   } while ((n = nextnode(n)));
**   return niltv(L);
*/
static void asm_href(ASMState *as, IRIns *ir, IROp merge)
{
  RegSet allow = RSET_GPR;
  int destused = ra_used(ir);
  Reg dest = ra_dest(as, ir, allow);
  Reg tab = ra_alloc1(as, ir->op1, rset_clear(allow, dest));
  Reg key = RID_NONE, tmp1 = RID_TMP, tmp2;
  Reg tisnum = RID_NONE, tmpnum = RID_NONE;
  IRRef refkey = ir->op2;
  IRIns *irkey = IR(refkey);
  int isk = irref_isk(refkey);
  IRType1 kt = irkey->t;
  uint32_t khash;
  MCLabel l_end, l_loop, l_next;

  rset_clear(allow, tab);
#if LJ_SOFTFP
  if (!isk) {
    key = ra_alloc1(as, refkey, allow);
    rset_clear(allow, key);
    if (irkey[1].o == IR_HIOP) {
      if (ra_hasreg((irkey+1)->r)) {
	tmpnum = (irkey+1)->r;
	ra_noweak(as, tmpnum);
      } else {
	tmpnum = ra_allocref(as, refkey+1, allow);
      }
      rset_clear(allow, tmpnum);
    }
  }
#else
  if (irt_isnum(kt)) {
    key = ra_alloc1(as, refkey, RSET_FPR);
    tmpnum = ra_scratch(as, rset_exclude(RSET_FPR, key));
    tisnum = ra_allock(as, (int32_t)LJ_TISNUM, allow);
    rset_clear(allow, tisnum);
  } else if (!irt_ispri(kt)) {
    key = ra_alloc1(as, refkey, allow);
    rset_clear(allow, key);
  }
#endif
  tmp2 = ra_scratch(as, allow);
  rset_clear(allow, tmp2);

  /* Key not found in chain: jump to exit (if merged) or load niltv. */
  l_end = emit_label(as);
  as->invmcp = NULL;
  if (merge == IR_NE)
    asm_guardcc(as, CC_EQ);
  else if (destused)
    emit_loada(as, dest, niltvg(J2G(as->J)));

  /* Follow hash chain until the end. */
  l_loop = --as->mcp;
  emit_ai(as, PPCI_CMPWI, dest, 0);
  emit_tai(as, PPCI_LWZ, dest, dest, (int32_t)offsetof(Node, next));
  l_next = emit_label(as);

  /* Type and value comparison. */
  if (merge == IR_EQ)
    asm_guardcc(as, CC_EQ);
  else
    emit_condbranch(as, PPCI_BC|PPCF_Y, CC_EQ, l_end);
  if (!LJ_SOFTFP && irt_isnum(kt)) {
    emit_fab(as, PPCI_FCMPU, 0, tmpnum, key);
    emit_condbranch(as, PPCI_BC, CC_GE, l_next);
    emit_ab(as, PPCI_CMPLW, tmp1, tisnum);
    emit_fai(as, PPCI_LFD, tmpnum, dest, (int32_t)offsetof(Node, key.n));
  } else {
    if (!irt_ispri(kt)) {
      emit_ab(as, PPCI_CMPW, tmp2, key);
      emit_condbranch(as, PPCI_BC, CC_NE, l_next);
    }
    if (LJ_SOFTFP && ra_hasreg(tmpnum))
      emit_ab(as, PPCI_CMPW, tmp1, tmpnum);
    else
      emit_ai(as, PPCI_CMPWI, tmp1, irt_toitype(irkey->t));
    if (!irt_ispri(kt))
      emit_tai(as, PPCI_LWZ, tmp2, dest, (int32_t)offsetof(Node, key.gcr));
  }
  emit_tai(as, PPCI_LWZ, tmp1, dest, (int32_t)offsetof(Node, key.it));
  *l_loop = PPCI_BC | PPCF_Y | PPCF_CC(CC_NE) |
	    (((char *)as->mcp-(char *)l_loop) & 0xffffu);

  /* Load main position relative to tab->node into dest. */
  khash = isk ? ir_khash(as, irkey) : 1;
  if (khash == 0) {
    emit_tai(as, PPCI_LWZ, dest, tab, (int32_t)offsetof(GCtab, node));
  } else {
    Reg tmphash = tmp1;
    if (isk)
      tmphash = ra_allock(as, khash, allow);
    emit_tab(as, PPCI_ADD, dest, dest, tmp1);
    emit_tai(as, PPCI_MULLI, tmp1, tmp1, sizeof(Node));
    emit_asb(as, PPCI_AND, tmp1, tmp2, tmphash);
    emit_tai(as, PPCI_LWZ, dest, tab, (int32_t)offsetof(GCtab, node));
    emit_tai(as, PPCI_LWZ, tmp2, tab, (int32_t)offsetof(GCtab, hmask));
    if (isk) {
      /* Nothing to do. */
    } else if (irt_isstr(kt)) {
      emit_tai(as, PPCI_LWZ, tmp1, key, (int32_t)offsetof(GCstr, sid));
    } else {  /* Must match with hash*() in lj_tab.c. */
      emit_tab(as, PPCI_SUBF, tmp1, tmp2, tmp1);
      emit_rotlwi(as, tmp2, tmp2, HASH_ROT3);
      emit_asb(as, PPCI_XOR, tmp1, tmp1, tmp2);
      emit_rotlwi(as, tmp1, tmp1, (HASH_ROT2+HASH_ROT1)&31);
      emit_tab(as, PPCI_SUBF, tmp2, dest, tmp2);
      if (LJ_SOFTFP ? (irkey[1].o == IR_HIOP) : irt_isnum(kt)) {
#if LJ_SOFTFP
	emit_asb(as, PPCI_XOR, tmp2, key, tmp1);
	emit_rotlwi(as, dest, tmp1, HASH_ROT1);
	emit_tab(as, PPCI_ADD, tmp1, tmpnum, tmpnum);
#else
	int32_t ofs = ra_spill(as, irkey);
	emit_asb(as, PPCI_XOR, tmp2, tmp2, tmp1);
	emit_rotlwi(as, dest, tmp1, HASH_ROT1);
	emit_tab(as, PPCI_ADD, tmp1, tmp1, tmp1);
	emit_tai(as, PPCI_LWZ, tmp2, RID_SP, ofs+4);
	emit_tai(as, PPCI_LWZ, tmp1, RID_SP, ofs);
#endif
      } else {
	emit_asb(as, PPCI_XOR, tmp2, key, tmp1);
	emit_rotlwi(as, dest, tmp1, HASH_ROT1);
	emit_tai(as, PPCI_ADDI, tmp1, tmp2, HASH_BIAS);
	emit_tai(as, PPCI_ADDIS, tmp2, key, (HASH_BIAS + 32768)>>16);
      }
    }
  }
}

static void asm_hrefk(ASMState *as, IRIns *ir)
{
  IRIns *kslot = IR(ir->op2);
  IRIns *irkey = IR(kslot->op1);
  int32_t ofs = (int32_t)(kslot->op2 * sizeof(Node));
  int32_t kofs = ofs + (int32_t)offsetof(Node, key);
  Reg dest = (ra_used(ir)||ofs > 32736) ? ra_dest(as, ir, RSET_GPR) : RID_NONE;
  Reg node = ra_alloc1(as, ir->op1, RSET_GPR);
  Reg key = RID_NONE, type = RID_TMP, idx = node;
  RegSet allow = rset_exclude(RSET_GPR, node);
  lj_assertA(ofs % sizeof(Node) == 0, "unaligned HREFK slot");
  if (ofs > 32736) {
    idx = dest;
    rset_clear(allow, dest);
    kofs = (int32_t)offsetof(Node, key);
  } else if (ra_hasreg(dest)) {
    emit_tai(as, PPCI_ADDI, dest, node, ofs);
  }
  asm_guardcc(as, CC_NE);
  if (!irt_ispri(irkey->t)) {
    key = ra_scratch(as, allow);
    rset_clear(allow, key);
  }
  rset_clear(allow, type);
  if (irt_isnum(irkey->t)) {
    emit_cmpi(as, key, (int32_t)ir_knum(irkey)->u32.lo);
    asm_guardcc(as, CC_NE);
    emit_cmpi(as, type, (int32_t)ir_knum(irkey)->u32.hi);
  } else {
    if (ra_hasreg(key)) {
      emit_cmpi(as, key, irkey->i);  /* May use RID_TMP, i.e. type. */
      asm_guardcc(as, CC_NE);
    }
    emit_ai(as, PPCI_CMPWI, type, irt_toitype(irkey->t));
  }
  if (ra_hasreg(key)) emit_tai(as, PPCI_LWZ, key, idx, kofs+4);
  emit_tai(as, PPCI_LWZ, type, idx, kofs);
  if (ofs > 32736) {
    emit_tai(as, PPCI_ADDIS, dest, dest, (ofs + 32768) >> 16);
    emit_tai(as, PPCI_ADDI, dest, node, ofs);
  }
}

static void asm_uref(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  int guarded = (irt_t(ir->t) & (IRT_GUARD|IRT_TYPE)) == (IRT_GUARD|IRT_PGC);
  if (irref_isk(ir->op1) && !guarded) {
    GCfunc *fn = ir_kfunc(IR(ir->op1));
    MRef *v = &gcref(fn->l.uvptr[(ir->op2 >> 8)])->uv.v;
    emit_lsptr(as, PPCI_LWZ, dest, v, RSET_GPR);
  } else {
    if (guarded) {
      asm_guardcc(as, ir->o == IR_UREFC ? CC_NE : CC_EQ);
      emit_ai(as, PPCI_CMPWI, RID_TMP, 1);
    }
    if (ir->o == IR_UREFC)
      emit_tai(as, PPCI_ADDI, dest, dest, (int32_t)offsetof(GCupval, tv));
    else
      emit_tai(as, PPCI_LWZ, dest, dest, (int32_t)offsetof(GCupval, v));
    if (guarded)
      emit_tai(as, PPCI_LBZ, RID_TMP, dest, (int32_t)offsetof(GCupval, closed));
    if (irref_isk(ir->op1)) {
      GCfunc *fn = ir_kfunc(IR(ir->op1));
      int32_t k = (int32_t)gcrefu(fn->l.uvptr[(ir->op2 >> 8)]);
      emit_loadi(as, dest, k);
    } else {
      emit_tai(as, PPCI_LWZ, dest, ra_alloc1(as, ir->op1, RSET_GPR),
	       (int32_t)offsetof(GCfuncL, uvptr) + 4*(int32_t)(ir->op2 >> 8));
    }
  }
}

static void asm_fref(ASMState *as, IRIns *ir)
{
  UNUSED(as); UNUSED(ir);
  lj_assertA(!ra_used(ir), "unfused FREF");
}

static void asm_strref(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  IRRef ref = ir->op2, refk = ir->op1;
  int32_t ofs = (int32_t)sizeof(GCstr);
  Reg r;
  if (irref_isk(ref)) {
    IRRef tmp = refk; refk = ref; ref = tmp;
  } else if (!irref_isk(refk)) {
    Reg right, left = ra_alloc1(as, ir->op1, RSET_GPR);
    IRIns *irr = IR(ir->op2);
    if (ra_hasreg(irr->r)) {
      ra_noweak(as, irr->r);
      right = irr->r;
    } else if (mayfuse(as, irr->op2) &&
	       irr->o == IR_ADD && irref_isk(irr->op2) &&
	       checki16(ofs + IR(irr->op2)->i)) {
      ofs += IR(irr->op2)->i;
      right = ra_alloc1(as, irr->op1, rset_exclude(RSET_GPR, left));
    } else {
      right = ra_allocref(as, ir->op2, rset_exclude(RSET_GPR, left));
    }
    emit_tai(as, PPCI_ADDI, dest, dest, ofs);
    emit_tab(as, PPCI_ADD, dest, left, right);
    return;
  }
  r = ra_alloc1(as, ref, RSET_GPR);
  ofs += IR(refk)->i;
  if (checki16(ofs))
    emit_tai(as, PPCI_ADDI, dest, r, ofs);
  else
    emit_tab(as, PPCI_ADD, dest, r,
	     ra_allock(as, ofs, rset_exclude(RSET_GPR, r)));
}

/* -- Loads and stores ---------------------------------------------------- */

static PPCIns asm_fxloadins(ASMState *as, IRIns *ir)
{
  UNUSED(as);
  switch (irt_type(ir->t)) {
  case IRT_I8: return PPCI_LBZ;  /* Needs sign-extension. */
  case IRT_U8: return PPCI_LBZ;
  case IRT_I16: return PPCI_LHA;
  case IRT_U16: return PPCI_LHZ;
  case IRT_NUM: lj_assertA(!LJ_SOFTFP, "unsplit FP op"); return PPCI_LFD;
  case IRT_FLOAT: if (!LJ_SOFTFP) return PPCI_LFS;
  default: return PPCI_LWZ;
  }
}

static PPCIns asm_fxstoreins(ASMState *as, IRIns *ir)
{
  UNUSED(as);
  switch (irt_type(ir->t)) {
  case IRT_I8: case IRT_U8: return PPCI_STB;
  case IRT_I16: case IRT_U16: return PPCI_STH;
  case IRT_NUM: lj_assertA(!LJ_SOFTFP, "unsplit FP op"); return PPCI_STFD;
  case IRT_FLOAT: if (!LJ_SOFTFP) return PPCI_STFS;
  default: return PPCI_STW;
  }
}

static void asm_fload(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  PPCIns pi = asm_fxloadins(as, ir);
  Reg idx;
  int32_t ofs;
  if (ir->op1 == REF_NIL) {  /* FLOAD from GG_State with offset. */
    idx = RID_JGL;
    ofs = (ir->op2 << 2) - 32768 - GG_OFS(g);
  } else {
    idx = ra_alloc1(as, ir->op1, RSET_GPR);
    if (ir->op2 == IRFL_TAB_ARRAY) {
      ofs = asm_fuseabase(as, ir->op1);
      if (ofs) {  /* Turn the t->array load into an add for colocated arrays. */
	emit_tai(as, PPCI_ADDI, dest, idx, ofs);
	return;
      }
    }
    ofs = field_ofs[ir->op2];
  }
  lj_assertA(!irt_isi8(ir->t), "unsupported FLOAD I8");
  emit_tai(as, pi, dest, idx, ofs);
}

static void asm_fstore(ASMState *as, IRIns *ir)
{
  if (ir->r != RID_SINK) {
    Reg src = ra_alloc1(as, ir->op2, RSET_GPR);
    IRIns *irf = IR(ir->op1);
    Reg idx = ra_alloc1(as, irf->op1, rset_exclude(RSET_GPR, src));
    int32_t ofs = field_ofs[irf->op2];
    PPCIns pi = asm_fxstoreins(as, ir);
    emit_tai(as, pi, src, idx, ofs);
  }
}

static void asm_xload(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir,
    (!LJ_SOFTFP && irt_isfp(ir->t)) ? RSET_FPR : RSET_GPR);
  lj_assertA(!(ir->op2 & IRXLOAD_UNALIGNED), "unaligned XLOAD");
  if (irt_isi8(ir->t))
    emit_as(as, PPCI_EXTSB, dest, dest);
  asm_fusexref(as, asm_fxloadins(as, ir), dest, ir->op1, RSET_GPR, 0);
}

static void asm_xstore_(ASMState *as, IRIns *ir, int32_t ofs)
{
  IRIns *irb;
  if (ir->r == RID_SINK)
    return;
  if (ofs == 0 && mayfuse(as, ir->op2) && (irb = IR(ir->op2))->o == IR_BSWAP &&
      ra_noreg(irb->r) && (irt_isint(ir->t) || irt_isu32(ir->t))) {
    /* Fuse BSWAP with XSTORE to stwbrx. */
    Reg src = ra_alloc1(as, irb->op1, RSET_GPR);
    asm_fusexrefx(as, PPCI_STWBRX, src, ir->op1, rset_exclude(RSET_GPR, src));
  } else {
    Reg src = ra_alloc1(as, ir->op2,
      (!LJ_SOFTFP && irt_isfp(ir->t)) ? RSET_FPR : RSET_GPR);
    asm_fusexref(as, asm_fxstoreins(as, ir), src, ir->op1,
		 rset_exclude(RSET_GPR, src), ofs);
  }
}

#define asm_xstore(as, ir)	asm_xstore_(as, ir, 0)

static void asm_ahuvload(ASMState *as, IRIns *ir)
{
  IRType1 t = ir->t;
  Reg dest = RID_NONE, type = RID_TMP, tmp = RID_TMP, idx;
  RegSet allow = RSET_GPR;
  int32_t ofs = AHUREF_LSX;
  if (LJ_SOFTFP && (ir+1)->o == IR_HIOP) {
    t.irt = IRT_NUM;
    if (ra_used(ir+1)) {
      type = ra_dest(as, ir+1, allow);
      rset_clear(allow, type);
    }
    ofs = 0;
  }
  if (ra_used(ir)) {
    lj_assertA((LJ_SOFTFP ? 0 : irt_isnum(ir->t)) ||
	       irt_isint(ir->t) || irt_isaddr(ir->t),
	       "bad load type %d", irt_type(ir->t));
    if (LJ_SOFTFP || !irt_isnum(t)) ofs = 0;
    dest = ra_dest(as, ir, (!LJ_SOFTFP && irt_isnum(t)) ? RSET_FPR : allow);
    rset_clear(allow, dest);
  }
  idx = asm_fuseahuref(as, ir->op1, &ofs, allow);
  if (ir->o == IR_VLOAD) {
    ofs = ofs != AHUREF_LSX ? ofs + 8 * ir->op2 :
	  ir->op2 ? 8 * ir->op2 : AHUREF_LSX;
  }
  if (irt_isnum(t)) {
    Reg tisnum = ra_allock(as, (int32_t)LJ_TISNUM, rset_exclude(allow, idx));
    asm_guardcc(as, CC_GE);
    emit_ab(as, PPCI_CMPLW, type, tisnum);
    if (ra_hasreg(dest)) {
      if (!LJ_SOFTFP && ofs == AHUREF_LSX) {
	tmp = ra_scratch(as, rset_exclude(rset_exclude(RSET_GPR,
						       (idx&255)), (idx>>8)));
	emit_fab(as, PPCI_LFDX, dest, (idx&255), tmp);
      } else {
	emit_fai(as, LJ_SOFTFP ? PPCI_LWZ : PPCI_LFD, dest, idx,
		 ofs+4*LJ_SOFTFP);
      }
    }
  } else {
    asm_guardcc(as, CC_NE);
    emit_ai(as, PPCI_CMPWI, type, irt_toitype(t));
    if (ra_hasreg(dest)) emit_tai(as, PPCI_LWZ, dest, idx, ofs+4);
  }
  if (ofs == AHUREF_LSX) {
    emit_tab(as, PPCI_LWZX, type, (idx&255), tmp);
    emit_slwi(as, tmp, (idx>>8), 3);
  } else {
    emit_tai(as, PPCI_LWZ, type, idx, ofs);
  }
}

static void asm_ahustore(ASMState *as, IRIns *ir)
{
  RegSet allow = RSET_GPR;
  Reg idx, src = RID_NONE, type = RID_NONE;
  int32_t ofs = AHUREF_LSX;
  if (ir->r == RID_SINK)
    return;
  if (!LJ_SOFTFP && irt_isnum(ir->t)) {
    src = ra_alloc1(as, ir->op2, RSET_FPR);
  } else {
    if (!irt_ispri(ir->t)) {
      src = ra_alloc1(as, ir->op2, allow);
      rset_clear(allow, src);
      ofs = 0;
    }
    if (LJ_SOFTFP && (ir+1)->o == IR_HIOP)
      type = ra_alloc1(as, (ir+1)->op2, allow);
    else
      type = ra_allock(as, (int32_t)irt_toitype(ir->t), allow);
    rset_clear(allow, type);
  }
  idx = asm_fuseahuref(as, ir->op1, &ofs, allow);
  if (!LJ_SOFTFP && irt_isnum(ir->t)) {
    if (ofs == AHUREF_LSX) {
      emit_fab(as, PPCI_STFDX, src, (idx&255), RID_TMP);
      emit_slwi(as, RID_TMP, (idx>>8), 3);
    } else {
      emit_fai(as, PPCI_STFD, src, idx, ofs);
    }
  } else {
    if (ra_hasreg(src))
      emit_tai(as, PPCI_STW, src, idx, ofs+4);
    if (ofs == AHUREF_LSX) {
      emit_tab(as, PPCI_STWX, type, (idx&255), RID_TMP);
      emit_slwi(as, RID_TMP, (idx>>8), 3);
    } else {
      emit_tai(as, PPCI_STW, type, idx, ofs);
    }
  }
}

static void asm_sload(ASMState *as, IRIns *ir)
{
  int32_t ofs = 8*((int32_t)ir->op1-1) + ((ir->op2 & IRSLOAD_FRAME) ? 0 : 4);
  IRType1 t = ir->t;
  Reg dest = RID_NONE, type = RID_NONE, base;
  RegSet allow = RSET_GPR;
  int hiop = (LJ_SOFTFP && (ir+1)->o == IR_HIOP);
  if (hiop)
    t.irt = IRT_NUM;
  lj_assertA(!(ir->op2 & IRSLOAD_PARENT),
	     "bad parent SLOAD");  /* Handled by asm_head_side(). */
  lj_assertA(irt_isguard(ir->t) || !(ir->op2 & IRSLOAD_TYPECHECK),
	     "inconsistent SLOAD variant");
  lj_assertA(LJ_DUALNUM ||
	     !irt_isint(t) ||
	     (ir->op2 & (IRSLOAD_CONVERT|IRSLOAD_FRAME|IRSLOAD_KEYINDEX)),
	     "bad SLOAD type");
#if LJ_SOFTFP
  lj_assertA(!(ir->op2 & IRSLOAD_CONVERT),
	     "unsplit SLOAD convert");  /* Handled by LJ_SOFTFP SPLIT. */
  if (hiop && ra_used(ir+1)) {
    type = ra_dest(as, ir+1, allow);
    rset_clear(allow, type);
  }
#else
  if ((ir->op2 & IRSLOAD_CONVERT) && irt_isguard(t) && irt_isint(t)) {
    dest = ra_scratch(as, RSET_FPR);
    asm_tointg(as, ir, dest);
    t.irt = IRT_NUM;  /* Continue with a regular number type check. */
  } else
#endif
  if (ra_used(ir)) {
    lj_assertA(irt_isnum(t) || irt_isint(t) || irt_isaddr(t),
	       "bad SLOAD type %d", irt_type(ir->t));
    dest = ra_dest(as, ir, (!LJ_SOFTFP && irt_isnum(t)) ? RSET_FPR : allow);
    rset_clear(allow, dest);
    base = ra_alloc1(as, REF_BASE, allow);
    rset_clear(allow, base);
    if (!LJ_SOFTFP && (ir->op2 & IRSLOAD_CONVERT)) {
      if (irt_isint(t)) {
	emit_tai(as, PPCI_LWZ, dest, RID_SP, SPOFS_TMPLO);
	dest = ra_scratch(as, RSET_FPR);
	emit_fai(as, PPCI_STFD, dest, RID_SP, SPOFS_TMP);
	emit_fb(as, PPCI_FCTIWZ, dest, dest);
	t.irt = IRT_NUM;  /* Check for original type. */
      } else {
	Reg tmp = ra_scratch(as, allow);
	Reg hibias = ra_allock(as, 0x43300000, rset_clear(allow, tmp));
	Reg fbias = ra_scratch(as, rset_exclude(RSET_FPR, dest));
	emit_fab(as, PPCI_FSUB, dest, dest, fbias);
	emit_fai(as, PPCI_LFD, dest, RID_SP, SPOFS_TMP);
	emit_lsptr(as, PPCI_LFS, (fbias & 31),
		   (void *)&as->J->k32[LJ_K32_2P52_2P31],
		   rset_clear(allow, hibias));
	emit_tai(as, PPCI_STW, tmp, RID_SP, SPOFS_TMPLO);
	emit_tai(as, PPCI_STW, hibias, RID_SP, SPOFS_TMPHI);
	emit_asi(as, PPCI_XORIS, tmp, tmp, 0x8000);
	dest = tmp;
	t.irt = IRT_INT;  /* Check for original type. */
      }
    }
    goto dotypecheck;
  }
  base = ra_alloc1(as, REF_BASE, allow);
  rset_clear(allow, base);
dotypecheck:
  if (irt_isnum(t)) {
    if ((ir->op2 & IRSLOAD_TYPECHECK)) {
      Reg tisnum = ra_allock(as, (int32_t)LJ_TISNUM, allow);
      asm_guardcc(as, CC_GE);
#if !LJ_SOFTFP
      type = RID_TMP;
#endif
      emit_ab(as, PPCI_CMPLW, type, tisnum);
    }
    if (ra_hasreg(dest)) emit_fai(as, LJ_SOFTFP ? PPCI_LWZ : PPCI_LFD, dest,
				  base, ofs-(LJ_SOFTFP?0:4));
  } else {
    if ((ir->op2 & IRSLOAD_TYPECHECK)) {
      asm_guardcc(as, CC_NE);
      if ((ir->op2 & IRSLOAD_KEYINDEX)) {
	emit_ai(as, PPCI_CMPWI, RID_TMP, (LJ_KEYINDEX & 0xffff));
	emit_asi(as, PPCI_XORIS, RID_TMP, RID_TMP, (LJ_KEYINDEX >> 16));
      } else {
	emit_ai(as, PPCI_CMPWI, RID_TMP, irt_toitype(t));
      }
      type = RID_TMP;
    }
    if (ra_hasreg(dest)) emit_tai(as, PPCI_LWZ, dest, base, ofs);
  }
  if (ra_hasreg(type)) emit_tai(as, PPCI_LWZ, type, base, ofs-4);
}

/* -- Allocations --------------------------------------------------------- */

#if LJ_HASFFI
static void asm_cnew(ASMState *as, IRIns *ir)
{
  CTState *cts = ctype_ctsG(J2G(as->J));
  CTypeID id = (CTypeID)IR(ir->op1)->i;
  CTSize sz;
  CTInfo info = lj_ctype_info(cts, id, &sz);
  const CCallInfo *ci = &lj_ir_callinfo[IRCALL_lj_mem_newgco];
  IRRef args[4];
  RegSet drop = RSET_SCRATCH;
  lj_assertA(sz != CTSIZE_INVALID || (ir->o == IR_CNEW && ir->op2 != REF_NIL),
	     "bad CNEW/CNEWI operands");

  as->gcsteps++;
  if (ra_hasreg(ir->r))
    rset_clear(drop, ir->r);  /* Dest reg handled below. */
  ra_evictset(as, drop);
  if (ra_used(ir))
    ra_destreg(as, ir, RID_RET);  /* GCcdata * */

  /* Initialize immutable cdata object. */
  if (ir->o == IR_CNEWI) {
    RegSet allow = (RSET_GPR & ~RSET_SCRATCH);
    int32_t ofs = sizeof(GCcdata);
    lj_assertA(sz == 4 || sz == 8, "bad CNEWI size %d", sz);
    if (sz == 8) {
      ofs += 4;
      lj_assertA((ir+1)->o == IR_HIOP, "expected HIOP for CNEWI");
    }
    for (;;) {
      Reg r = ra_alloc1(as, ir->op2, allow);
      emit_tai(as, PPCI_STW, r, RID_RET, ofs);
      rset_clear(allow, r);
      if (ofs == sizeof(GCcdata)) break;
      ofs -= 4; ir++;
    }
  } else if (ir->op2 != REF_NIL) {  /* Create VLA/VLS/aligned cdata. */
    ci = &lj_ir_callinfo[IRCALL_lj_cdata_newv];
    args[0] = ASMREF_L;     /* lua_State *L */
    args[1] = ir->op1;      /* CTypeID id   */
    args[2] = ir->op2;      /* CTSize sz    */
    args[3] = ASMREF_TMP1;  /* CTSize align */
    asm_gencall(as, ci, args);
    emit_loadi(as, ra_releasetmp(as, ASMREF_TMP1), (int32_t)ctype_align(info));
    return;
  }

  /* Initialize gct and ctypeid. lj_mem_newgco() already sets marked. */
  emit_tai(as, PPCI_STB, RID_RET+1, RID_RET, offsetof(GCcdata, gct));
  emit_tai(as, PPCI_STH, RID_TMP, RID_RET, offsetof(GCcdata, ctypeid));
  emit_ti(as, PPCI_LI, RID_RET+1, ~LJ_TCDATA);
  emit_ti(as, PPCI_LI, RID_TMP, id);  /* Lower 16 bit used. Sign-ext ok. */
  args[0] = ASMREF_L;     /* lua_State *L */
  args[1] = ASMREF_TMP1;  /* MSize size   */
  asm_gencall(as, ci, args);
  ra_allockreg(as, (int32_t)(sz+sizeof(GCcdata)),
	       ra_releasetmp(as, ASMREF_TMP1));
}
#endif

/* -- Write barriers ------------------------------------------------------ */

static void asm_tbar(ASMState *as, IRIns *ir)
{
  Reg tab = ra_alloc1(as, ir->op1, RSET_GPR);
  Reg mark = ra_scratch(as, rset_exclude(RSET_GPR, tab));
  Reg link = RID_TMP;
  MCLabel l_end = emit_label(as);
  emit_tai(as, PPCI_STW, link, tab, (int32_t)offsetof(GCtab, gclist));
  emit_tai(as, PPCI_STB, mark, tab, (int32_t)offsetof(GCtab, marked));
  emit_setgl(as, tab, gc.grayagain);
  lj_assertA(LJ_GC_BLACK == 0x04, "bad LJ_GC_BLACK");
  emit_rot(as, PPCI_RLWINM, mark, mark, 0, 30, 28);  /* Clear black bit. */
  emit_getgl(as, link, gc.grayagain);
  emit_condbranch(as, PPCI_BC|PPCF_Y, CC_EQ, l_end);
  emit_asi(as, PPCI_ANDIDOT, RID_TMP, mark, LJ_GC_BLACK);
  emit_tai(as, PPCI_LBZ, mark, tab, (int32_t)offsetof(GCtab, marked));
}

static void asm_obar(ASMState *as, IRIns *ir)
{
  const CCallInfo *ci = &lj_ir_callinfo[IRCALL_lj_gc_barrieruv];
  IRRef args[2];
  MCLabel l_end;
  Reg obj, val, tmp;
  /* No need for other object barriers (yet). */
  lj_assertA(IR(ir->op1)->o == IR_UREFC, "bad OBAR type");
  ra_evictset(as, RSET_SCRATCH);
  l_end = emit_label(as);
  args[0] = ASMREF_TMP1;  /* global_State *g */
  args[1] = ir->op1;      /* TValue *tv      */
  asm_gencall(as, ci, args);
  emit_tai(as, PPCI_ADDI, ra_releasetmp(as, ASMREF_TMP1), RID_JGL, -32768);
  obj = IR(ir->op1)->r;
  tmp = ra_scratch(as, rset_exclude(RSET_GPR, obj));
  emit_condbranch(as, PPCI_BC|PPCF_Y, CC_EQ, l_end);
  emit_asi(as, PPCI_ANDIDOT, tmp, tmp, LJ_GC_BLACK);
  emit_condbranch(as, PPCI_BC, CC_EQ, l_end);
  emit_asi(as, PPCI_ANDIDOT, RID_TMP, RID_TMP, LJ_GC_WHITES);
  val = ra_alloc1(as, ir->op2, rset_exclude(RSET_GPR, obj));
  emit_tai(as, PPCI_LBZ, tmp, obj,
	   (int32_t)offsetof(GCupval, marked)-(int32_t)offsetof(GCupval, tv));
  emit_tai(as, PPCI_LBZ, RID_TMP, val, (int32_t)offsetof(GChead, marked));
}

/* -- Arithmetic and logic operations ------------------------------------- */

#if !LJ_SOFTFP
static void asm_fparith(ASMState *as, IRIns *ir, PPCIns pi)
{
  Reg dest = ra_dest(as, ir, RSET_FPR);
  Reg right, left = ra_alloc2(as, ir, RSET_FPR);
  right = (left >> 8); left &= 255;
  if (pi == PPCI_FMUL)
    emit_fac(as, pi, dest, left, right);
  else
    emit_fab(as, pi, dest, left, right);
}

static void asm_fpunary(ASMState *as, IRIns *ir, PPCIns pi)
{
  Reg dest = ra_dest(as, ir, RSET_FPR);
  Reg left = ra_hintalloc(as, ir->op1, dest, RSET_FPR);
  emit_fb(as, pi, dest, left);
}

static void asm_fpmath(ASMState *as, IRIns *ir)
{
  if (ir->op2 == IRFPM_SQRT && (as->flags & JIT_F_SQRT))
    asm_fpunary(as, ir, PPCI_FSQRT);
  else
    asm_callid(as, ir, IRCALL_lj_vm_floor + ir->op2);
}
#endif

static void asm_add(ASMState *as, IRIns *ir)
{
#if !LJ_SOFTFP
  if (irt_isnum(ir->t)) {
    if (!asm_fusemadd(as, ir, PPCI_FMADD, PPCI_FMADD))
      asm_fparith(as, ir, PPCI_FADD);
  } else
#endif
  {
    Reg dest = ra_dest(as, ir, RSET_GPR);
    Reg right, left = ra_hintalloc(as, ir->op1, dest, RSET_GPR);
    PPCIns pi;
    if (irref_isk(ir->op2)) {
      int32_t k = IR(ir->op2)->i;
      if (checki16(k)) {
	pi = PPCI_ADDI;
	/* May fail due to spills/restores above, but simplifies the logic. */
	if (as->flagmcp == as->mcp) {
	  as->flagmcp = NULL;
	  as->mcp++;
	  pi = PPCI_ADDICDOT;
	}
	emit_tai(as, pi, dest, left, k);
	return;
      } else if ((k & 0xffff) == 0) {
	emit_tai(as, PPCI_ADDIS, dest, left, (k >> 16));
	return;
      } else if (!as->sectref) {
	emit_tai(as, PPCI_ADDIS, dest, dest, (k + 32768) >> 16);
	emit_tai(as, PPCI_ADDI, dest, left, k);
	return;
      }
    }
    pi = PPCI_ADD;
    /* May fail due to spills/restores above, but simplifies the logic. */
    if (as->flagmcp == as->mcp) {
      as->flagmcp = NULL;
      as->mcp++;
      pi |= PPCF_DOT;
    }
    right = ra_alloc1(as, ir->op2, rset_exclude(RSET_GPR, left));
    emit_tab(as, pi, dest, left, right);
  }
}

static void asm_sub(ASMState *as, IRIns *ir)
{
#if !LJ_SOFTFP
  if (irt_isnum(ir->t)) {
    if (!asm_fusemadd(as, ir, PPCI_FMSUB, PPCI_FNMSUB))
      asm_fparith(as, ir, PPCI_FSUB);
  } else
#endif
  {
    PPCIns pi = PPCI_SUBF;
    Reg dest = ra_dest(as, ir, RSET_GPR);
    Reg left, right;
    if (irref_isk(ir->op1)) {
      int32_t k = IR(ir->op1)->i;
      if (checki16(k)) {
	right = ra_alloc1(as, ir->op2, RSET_GPR);
	emit_tai(as, PPCI_SUBFIC, dest, right, k);
	return;
      }
    }
    /* May fail due to spills/restores above, but simplifies the logic. */
    if (as->flagmcp == as->mcp) {
      as->flagmcp = NULL;
      as->mcp++;
      pi |= PPCF_DOT;
    }
    left = ra_hintalloc(as, ir->op1, dest, RSET_GPR);
    right = ra_alloc1(as, ir->op2, rset_exclude(RSET_GPR, left));
    emit_tab(as, pi, dest, right, left);  /* Subtract right _from_ left. */
  }
}

static void asm_mul(ASMState *as, IRIns *ir)
{
#if !LJ_SOFTFP
  if (irt_isnum(ir->t)) {
    asm_fparith(as, ir, PPCI_FMUL);
  } else
#endif
  {
    PPCIns pi = PPCI_MULLW;
    Reg dest = ra_dest(as, ir, RSET_GPR);
    Reg right, left = ra_hintalloc(as, ir->op1, dest, RSET_GPR);
    if (irref_isk(ir->op2)) {
      int32_t k = IR(ir->op2)->i;
      if (checki16(k)) {
	emit_tai(as, PPCI_MULLI, dest, left, k);
	return;
      }
    }
    /* May fail due to spills/restores above, but simplifies the logic. */
    if (as->flagmcp == as->mcp) {
      as->flagmcp = NULL;
      as->mcp++;
      pi |= PPCF_DOT;
    }
    right = ra_alloc1(as, ir->op2, rset_exclude(RSET_GPR, left));
    emit_tab(as, pi, dest, left, right);
  }
}

#define asm_fpdiv(as, ir)	asm_fparith(as, ir, PPCI_FDIV)

static void asm_neg(ASMState *as, IRIns *ir)
{
#if !LJ_SOFTFP
  if (irt_isnum(ir->t)) {
    asm_fpunary(as, ir, PPCI_FNEG);
  } else
#endif
  {
    Reg dest, left;
    PPCIns pi = PPCI_NEG;
    if (as->flagmcp == as->mcp) {
      as->flagmcp = NULL;
      as->mcp++;
      pi |= PPCF_DOT;
    }
    dest = ra_dest(as, ir, RSET_GPR);
    left = ra_hintalloc(as, ir->op1, dest, RSET_GPR);
    emit_tab(as, pi, dest, left, 0);
  }
}

#define asm_abs(as, ir)		asm_fpunary(as, ir, PPCI_FABS)

static void asm_arithov(ASMState *as, IRIns *ir, PPCIns pi)
{
  Reg dest, left, right;
  if (as->flagmcp == as->mcp) {
    as->flagmcp = NULL;
    as->mcp++;
  }
  asm_guardcc(as, CC_SO);
  dest = ra_dest(as, ir, RSET_GPR);
  left = ra_alloc2(as, ir, RSET_GPR);
  right = (left >> 8); left &= 255;
  if (pi == PPCI_SUBFO) { Reg tmp = left; left = right; right = tmp; }
  emit_tab(as, pi|PPCF_DOT, dest, left, right);
}

#define asm_addov(as, ir)	asm_arithov(as, ir, PPCI_ADDO)
#define asm_subov(as, ir)	asm_arithov(as, ir, PPCI_SUBFO)
#define asm_mulov(as, ir)	asm_arithov(as, ir, PPCI_MULLWO)

#if LJ_HASFFI
static void asm_add64(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  Reg right, left = ra_alloc1(as, ir->op1, RSET_GPR);
  PPCIns pi = PPCI_ADDE;
  if (irref_isk(ir->op2)) {
    int32_t k = IR(ir->op2)->i;
    if (k == 0)
      pi = PPCI_ADDZE;
    else if (k == -1)
      pi = PPCI_ADDME;
    else
      goto needright;
    right = 0;
  } else {
  needright:
    right = ra_alloc1(as, ir->op2, rset_exclude(RSET_GPR, left));
  }
  emit_tab(as, pi, dest, left, right);
  ir--;
  dest = ra_dest(as, ir, RSET_GPR);
  left = ra_alloc1(as, ir->op1, RSET_GPR);
  if (irref_isk(ir->op2)) {
    int32_t k = IR(ir->op2)->i;
    if (checki16(k)) {
      emit_tai(as, PPCI_ADDIC, dest, left, k);
      return;
    }
  }
  right = ra_alloc1(as, ir->op2, rset_exclude(RSET_GPR, left));
  emit_tab(as, PPCI_ADDC, dest, left, right);
}

static void asm_sub64(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  Reg left, right = ra_alloc1(as, ir->op2, RSET_GPR);
  PPCIns pi = PPCI_SUBFE;
  if (irref_isk(ir->op1)) {
    int32_t k = IR(ir->op1)->i;
    if (k == 0)
      pi = PPCI_SUBFZE;
    else if (k == -1)
      pi = PPCI_SUBFME;
    else
      goto needleft;
    left = 0;
  } else {
  needleft:
    left = ra_alloc1(as, ir->op1, rset_exclude(RSET_GPR, right));
  }
  emit_tab(as, pi, dest, right, left);  /* Subtract right _from_ left. */
  ir--;
  dest = ra_dest(as, ir, RSET_GPR);
  right = ra_alloc1(as, ir->op2, RSET_GPR);
  if (irref_isk(ir->op1)) {
    int32_t k = IR(ir->op1)->i;
    if (checki16(k)) {
      emit_tai(as, PPCI_SUBFIC, dest, right, k);
      return;
    }
  }
  left = ra_alloc1(as, ir->op1, rset_exclude(RSET_GPR, right));
  emit_tab(as, PPCI_SUBFC, dest, right, left);
}

static void asm_neg64(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  Reg left = ra_alloc1(as, ir->op1, RSET_GPR);
  emit_tab(as, PPCI_SUBFZE, dest, left, 0);
  ir--;
  dest = ra_dest(as, ir, RSET_GPR);
  left = ra_alloc1(as, ir->op1, RSET_GPR);
  emit_tai(as, PPCI_SUBFIC, dest, left, 0);
}
#endif

static void asm_bnot(ASMState *as, IRIns *ir)
{
  Reg dest, left, right;
  PPCIns pi = PPCI_NOR;
  if (as->flagmcp == as->mcp) {
    as->flagmcp = NULL;
    as->mcp++;
    pi |= PPCF_DOT;
  }
  dest = ra_dest(as, ir, RSET_GPR);
  if (mayfuse(as, ir->op1)) {
    IRIns *irl = IR(ir->op1);
    if (irl->o == IR_BAND)
      pi ^= (PPCI_NOR ^ PPCI_NAND);
    else if (irl->o == IR_BXOR)
      pi ^= (PPCI_NOR ^ PPCI_EQV);
    else if (irl->o != IR_BOR)
      goto nofuse;
    left = ra_hintalloc(as, irl->op1, dest, RSET_GPR);
    right = ra_alloc1(as, irl->op2, rset_exclude(RSET_GPR, left));
  } else {
nofuse:
    left = right = ra_hintalloc(as, ir->op1, dest, RSET_GPR);
  }
  emit_asb(as, pi, dest, left, right);
}

static void asm_bswap(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  IRIns *irx;
  if (mayfuse(as, ir->op1) && (irx = IR(ir->op1))->o == IR_XLOAD &&
      ra_noreg(irx->r) && (irt_isint(irx->t) || irt_isu32(irx->t))) {
    /* Fuse BSWAP with XLOAD to lwbrx. */
    asm_fusexrefx(as, PPCI_LWBRX, dest, irx->op1, RSET_GPR);
  } else {
    Reg left = ra_alloc1(as, ir->op1, RSET_GPR);
    Reg tmp = dest;
    if (tmp == left) {
      tmp = RID_TMP;
      emit_mr(as, dest, RID_TMP);
    }
    emit_rot(as, PPCI_RLWIMI, tmp, left, 24, 16, 23);
    emit_rot(as, PPCI_RLWIMI, tmp, left, 24, 0, 7);
    emit_rotlwi(as, tmp, left, 8);
  }
}

/* Fuse BAND with contiguous bitmask and a shift to rlwinm. */
static void asm_fuseandsh(ASMState *as, PPCIns pi, int32_t mask, IRRef ref)
{
  IRIns *ir;
  Reg left;
  if (mayfuse(as, ref) && (ir = IR(ref), ra_noreg(ir->r)) &&
      irref_isk(ir->op2) && ir->o >= IR_BSHL && ir->o <= IR_BROR) {
    int32_t sh = (IR(ir->op2)->i & 31);
    switch (ir->o) {
    case IR_BSHL:
      if ((mask & ((1u<<sh)-1))) goto nofuse;
      break;
    case IR_BSHR:
      if ((mask & ~((~0u)>>sh))) goto nofuse;
      sh = ((32-sh)&31);
      break;
    case IR_BROL:
      break;
    default:
      goto nofuse;
    }
    left = ra_alloc1(as, ir->op1, RSET_GPR);
    *--as->mcp = pi | PPCF_T(left) | PPCF_B(sh);
    return;
  }
nofuse:
  left = ra_alloc1(as, ref, RSET_GPR);
  *--as->mcp = pi | PPCF_T(left);
}

static void asm_band(ASMState *as, IRIns *ir)
{
  Reg dest, left, right;
  IRRef lref = ir->op1;
  PPCIns dot = 0;
  IRRef op2;
  if (as->flagmcp == as->mcp) {
    as->flagmcp = NULL;
    as->mcp++;
    dot = PPCF_DOT;
  }
  dest = ra_dest(as, ir, RSET_GPR);
  if (irref_isk(ir->op2)) {
    int32_t k = IR(ir->op2)->i;
    if (k) {
      /* First check for a contiguous bitmask as used by rlwinm. */
      uint32_t s1 = lj_ffs((uint32_t)k);
      uint32_t k1 = ((uint32_t)k >> s1);
      if ((k1 & (k1+1)) == 0) {
	asm_fuseandsh(as, PPCI_RLWINM|dot | PPCF_A(dest) |
			  PPCF_MB(31-lj_fls((uint32_t)k)) | PPCF_ME(31-s1),
			  k, lref);
	return;
      }
      if (~(uint32_t)k) {
	uint32_t s2 = lj_ffs(~(uint32_t)k);
	uint32_t k2 = (~(uint32_t)k >> s2);
	if ((k2 & (k2+1)) == 0) {
	  asm_fuseandsh(as, PPCI_RLWINM|dot | PPCF_A(dest) |
			    PPCF_MB(32-s2) | PPCF_ME(30-lj_fls(~(uint32_t)k)),
			    k, lref);
	  return;
	}
      }
    }
    if (checku16(k)) {
      left = ra_alloc1(as, lref, RSET_GPR);
      emit_asi(as, PPCI_ANDIDOT, dest, left, k);
      return;
    } else if ((k & 0xffff) == 0) {
      left = ra_alloc1(as, lref, RSET_GPR);
      emit_asi(as, PPCI_ANDISDOT, dest, left, (k >> 16));
      return;
    }
  }
  op2 = ir->op2;
  if (mayfuse(as, op2) && IR(op2)->o == IR_BNOT && ra_noreg(IR(op2)->r)) {
    dot ^= (PPCI_AND ^ PPCI_ANDC);
    op2 = IR(op2)->op1;
  }
  left = ra_hintalloc(as, lref, dest, RSET_GPR);
  right = ra_alloc1(as, op2, rset_exclude(RSET_GPR, left));
  emit_asb(as, PPCI_AND ^ dot, dest, left, right);
}

static void asm_bitop(ASMState *as, IRIns *ir, PPCIns pi, PPCIns pik)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  Reg right, left = ra_hintalloc(as, ir->op1, dest, RSET_GPR);
  if (irref_isk(ir->op2)) {
    int32_t k = IR(ir->op2)->i;
    Reg tmp = left;
    if ((checku16(k) || (k & 0xffff) == 0) || (tmp = dest, !as->sectref)) {
      if (!checku16(k)) {
	emit_asi(as, pik ^ (PPCI_ORI ^ PPCI_ORIS), dest, tmp, (k >> 16));
	if ((k & 0xffff) == 0) return;
      }
      emit_asi(as, pik, dest, left, k);
      return;
    }
  }
  /* May fail due to spills/restores above, but simplifies the logic. */
  if (as->flagmcp == as->mcp) {
    as->flagmcp = NULL;
    as->mcp++;
    pi |= PPCF_DOT;
  }
  right = ra_alloc1(as, ir->op2, rset_exclude(RSET_GPR, left));
  emit_asb(as, pi, dest, left, right);
}

#define asm_bor(as, ir)		asm_bitop(as, ir, PPCI_OR, PPCI_ORI)
#define asm_bxor(as, ir)	asm_bitop(as, ir, PPCI_XOR, PPCI_XORI)

static void asm_bitshift(ASMState *as, IRIns *ir, PPCIns pi, PPCIns pik)
{
  Reg dest, left;
  Reg dot = 0;
  if (as->flagmcp == as->mcp) {
    as->flagmcp = NULL;
    as->mcp++;
    dot = PPCF_DOT;
  }
  dest = ra_dest(as, ir, RSET_GPR);
  left = ra_alloc1(as, ir->op1, RSET_GPR);
  if (irref_isk(ir->op2)) {  /* Constant shifts. */
    int32_t shift = (IR(ir->op2)->i & 31);
    if (pik == 0)  /* SLWI */
      emit_rot(as, PPCI_RLWINM|dot, dest, left, shift, 0, 31-shift);
    else if (pik == 1)  /* SRWI */
      emit_rot(as, PPCI_RLWINM|dot, dest, left, (32-shift)&31, shift, 31);
    else
      emit_asb(as, pik|dot, dest, left, shift);
  } else {
    Reg right = ra_alloc1(as, ir->op2, rset_exclude(RSET_GPR, left));
    emit_asb(as, pi|dot, dest, left, right);
  }
}

#define asm_bshl(as, ir)	asm_bitshift(as, ir, PPCI_SLW, 0)
#define asm_bshr(as, ir)	asm_bitshift(as, ir, PPCI_SRW, 1)
#define asm_bsar(as, ir)	asm_bitshift(as, ir, PPCI_SRAW, PPCI_SRAWI)
#define asm_brol(as, ir) \
  asm_bitshift(as, ir, PPCI_RLWNM|PPCF_MB(0)|PPCF_ME(31), \
		       PPCI_RLWINM|PPCF_MB(0)|PPCF_ME(31))
#define asm_bror(as, ir)	lj_assertA(0, "unexpected BROR")

#if LJ_SOFTFP
static void asm_sfpmin_max(ASMState *as, IRIns *ir)
{
  CCallInfo ci = lj_ir_callinfo[IRCALL_softfp_cmp];
  IRRef args[4];
  MCLabel l_right, l_end;
  Reg desthi = ra_dest(as, ir, RSET_GPR), destlo = ra_dest(as, ir+1, RSET_GPR);
  Reg righthi, lefthi = ra_alloc2(as, ir, RSET_GPR);
  Reg rightlo, leftlo = ra_alloc2(as, ir+1, RSET_GPR);
  PPCCC cond = (IROp)ir->o == IR_MIN ? CC_EQ : CC_NE;
  righthi = (lefthi >> 8); lefthi &= 255;
  rightlo = (leftlo >> 8); leftlo &= 255;
  args[0^LJ_BE] = ir->op1; args[1^LJ_BE] = (ir+1)->op1;
  args[2^LJ_BE] = ir->op2; args[3^LJ_BE] = (ir+1)->op2;
  l_end = emit_label(as);
  if (desthi != righthi) emit_mr(as, desthi, righthi);
  if (destlo != rightlo) emit_mr(as, destlo, rightlo);
  l_right = emit_label(as);
  if (l_end != l_right) emit_jmp(as, l_end);
  if (desthi != lefthi) emit_mr(as, desthi, lefthi);
  if (destlo != leftlo) emit_mr(as, destlo, leftlo);
  if (l_right == as->mcp+1) {
    cond ^= 4; l_right = l_end; ++as->mcp;
  }
  emit_condbranch(as, PPCI_BC, cond, l_right);
  ra_evictset(as, RSET_SCRATCH);
  emit_cmpi(as, RID_RET, 1);
  asm_gencall(as, &ci, args);
}
#endif

static void asm_min_max(ASMState *as, IRIns *ir, int ismax)
{
  if (!LJ_SOFTFP && irt_isnum(ir->t)) {
    Reg dest = ra_dest(as, ir, RSET_FPR);
    Reg tmp = dest;
    Reg right, left = ra_alloc2(as, ir, RSET_FPR);
    right = (left >> 8); left &= 255;
    if (tmp == left || tmp == right)
      tmp = ra_scratch(as, rset_exclude(rset_exclude(rset_exclude(RSET_FPR,
					dest), left), right));
    emit_facb(as, PPCI_FSEL, dest, tmp, left, right);
    emit_fab(as, PPCI_FSUB, tmp, ismax ? left : right, ismax ? right : left);
  } else {
    Reg dest = ra_dest(as, ir, RSET_GPR);
    Reg tmp1 = RID_TMP, tmp2 = dest;
    Reg right, left = ra_alloc2(as, ir, RSET_GPR);
    right = (left >> 8); left &= 255;
    if (tmp2 == left || tmp2 == right)
      tmp2 = ra_scratch(as, rset_exclude(rset_exclude(rset_exclude(RSET_GPR,
					 dest), left), right));
    emit_tab(as, PPCI_ADD, dest, tmp2, right);
    emit_asb(as, ismax ? PPCI_ANDC : PPCI_AND, tmp2, tmp2, tmp1);
    emit_tab(as, PPCI_SUBFE, tmp1, tmp1, tmp1);
    emit_tab(as, PPCI_SUBFC, tmp2, tmp2, tmp1);
    emit_asi(as, PPCI_XORIS, tmp2, right, 0x8000);
    emit_asi(as, PPCI_XORIS, tmp1, left, 0x8000);
  }
}

#define asm_min(as, ir)		asm_min_max(as, ir, 0)
#define asm_max(as, ir)		asm_min_max(as, ir, 1)

/* -- Comparisons --------------------------------------------------------- */

#define CC_UNSIGNED	0x08	/* Unsigned integer comparison. */
#define CC_TWO		0x80	/* Check two flags for FP comparison. */

/* Map of comparisons to flags. ORDER IR. */
static const uint8_t asm_compmap[IR_ABC+1] = {
  /* op     int cc                 FP cc */
  /* LT  */ CC_GE               + (CC_GE<<4),
  /* GE  */ CC_LT               + (CC_LE<<4) + CC_TWO,
  /* LE  */ CC_GT               + (CC_GE<<4) + CC_TWO,
  /* GT  */ CC_LE               + (CC_LE<<4),
  /* ULT */ CC_GE + CC_UNSIGNED + (CC_GT<<4) + CC_TWO,
  /* UGE */ CC_LT + CC_UNSIGNED + (CC_LT<<4),
  /* ULE */ CC_GT + CC_UNSIGNED + (CC_GT<<4),
  /* UGT */ CC_LE + CC_UNSIGNED + (CC_LT<<4) + CC_TWO,
  /* EQ  */ CC_NE               + (CC_NE<<4),
  /* NE  */ CC_EQ               + (CC_EQ<<4),
  /* ABC */ CC_LE + CC_UNSIGNED + (CC_LT<<4) + CC_TWO  /* Same as UGT. */
};

static void asm_intcomp_(ASMState *as, IRRef lref, IRRef rref, Reg cr, PPCCC cc)
{
  Reg right, left = ra_alloc1(as, lref, RSET_GPR);
  if (irref_isk(rref)) {
    int32_t k = IR(rref)->i;
    if ((cc & CC_UNSIGNED) == 0) {  /* Signed comparison with constant. */
      if (checki16(k)) {
	emit_tai(as, PPCI_CMPWI, cr, left, k);
	/* Signed comparison with zero and referencing previous ins? */
	if (k == 0 && lref == as->curins-1)
	  as->flagmcp = as->mcp;  /* Allow elimination of the compare. */
	return;
      } else if ((cc & 3) == (CC_EQ & 3)) {  /* Use CMPLWI for EQ or NE. */
	if (checku16(k)) {
	  emit_tai(as, PPCI_CMPLWI, cr, left, k);
	  return;
	} else if (!as->sectref && ra_noreg(IR(rref)->r)) {
	  emit_tai(as, PPCI_CMPLWI, cr, RID_TMP, k);
	  emit_asi(as, PPCI_XORIS, RID_TMP, left, (k >> 16));
	  return;
	}
      }
    } else {  /* Unsigned comparison with constant. */
      if (checku16(k)) {
	emit_tai(as, PPCI_CMPLWI, cr, left, k);
	return;
      }
    }
  }
  right = ra_alloc1(as, rref, rset_exclude(RSET_GPR, left));
  emit_tab(as, (cc & CC_UNSIGNED) ? PPCI_CMPLW : PPCI_CMPW, cr, left, right);
}

static void asm_comp(ASMState *as, IRIns *ir)
{
  PPCCC cc = asm_compmap[ir->o];
  if (!LJ_SOFTFP && irt_isnum(ir->t)) {
    Reg right, left = ra_alloc2(as, ir, RSET_FPR);
    right = (left >> 8); left &= 255;
    asm_guardcc(as, (cc >> 4));
    if ((cc & CC_TWO))
      emit_tab(as, PPCI_CROR, ((cc>>4)&3), ((cc>>4)&3), (CC_EQ&3));
    emit_fab(as, PPCI_FCMPU, 0, left, right);
  } else {
    IRRef lref = ir->op1, rref = ir->op2;
    if (irref_isk(lref) && !irref_isk(rref)) {
      /* Swap constants to the right (only for ABC). */
      IRRef tmp = lref; lref = rref; rref = tmp;
      if ((cc & 2) == 0) cc ^= 1;  /* LT <-> GT, LE <-> GE */
    }
    asm_guardcc(as, cc);
    asm_intcomp_(as, lref, rref, 0, cc);
  }
}

#define asm_equal(as, ir)	asm_comp(as, ir)

#if LJ_SOFTFP
/* SFP comparisons. */
static void asm_sfpcomp(ASMState *as, IRIns *ir)
{
  const CCallInfo *ci = &lj_ir_callinfo[IRCALL_softfp_cmp];
  RegSet drop = RSET_SCRATCH;
  Reg r;
  IRRef args[4];
  args[0^LJ_BE] = ir->op1; args[1^LJ_BE] = (ir+1)->op1;
  args[2^LJ_BE] = ir->op2; args[3^LJ_BE] = (ir+1)->op2;

  for (r = REGARG_FIRSTGPR; r <= REGARG_FIRSTGPR+3; r++) {
    if (!rset_test(as->freeset, r) &&
	regcost_ref(as->cost[r]) == args[r-REGARG_FIRSTGPR])
      rset_clear(drop, r);
  }
  ra_evictset(as, drop);
  asm_setupresult(as, ir, ci);
  switch ((IROp)ir->o) {
  case IR_ULT:
    asm_guardcc(as, CC_EQ);
    emit_ai(as, PPCI_CMPWI, RID_RET, 0);
  case IR_ULE:
    asm_guardcc(as, CC_EQ);
    emit_ai(as, PPCI_CMPWI, RID_RET, 1);
    break;
  case IR_GE: case IR_GT:
    asm_guardcc(as, CC_EQ);
    emit_ai(as, PPCI_CMPWI, RID_RET, 2);
  default:
    asm_guardcc(as, (asm_compmap[ir->o] & 0xf));
    emit_ai(as, PPCI_CMPWI, RID_RET, 0);
    break;
  }
  asm_gencall(as, ci, args);
}
#endif

#if LJ_HASFFI
/* 64 bit integer comparisons. */
static void asm_comp64(ASMState *as, IRIns *ir)
{
  PPCCC cc = asm_compmap[(ir-1)->o];
  if ((cc&3) == (CC_EQ&3)) {
    asm_guardcc(as, cc);
    emit_tab(as, (cc&4) ? PPCI_CRAND : PPCI_CROR,
	     (CC_EQ&3), (CC_EQ&3), 4+(CC_EQ&3));
  } else {
    asm_guardcc(as, CC_EQ);
    emit_tab(as, PPCI_CROR, (CC_EQ&3), (CC_EQ&3), ((cc^~(cc>>2))&1));
    emit_tab(as, (cc&4) ? PPCI_CRAND : PPCI_CRANDC,
	     (CC_EQ&3), (CC_EQ&3), 4+(cc&3));
  }
  /* Loword comparison sets cr1 and is unsigned, except for equality. */
  asm_intcomp_(as, (ir-1)->op1, (ir-1)->op2, 4,
	       cc | ((cc&3) == (CC_EQ&3) ? 0 : CC_UNSIGNED));
  /* Hiword comparison sets cr0. */
  asm_intcomp_(as, ir->op1, ir->op2, 0, cc);
  as->flagmcp = NULL;  /* Doesn't work here. */
}
#endif

/* -- Split register ops -------------------------------------------------- */

/* Hiword op of a split 32/32 bit op. Previous op is be the loword op. */
static void asm_hiop(ASMState *as, IRIns *ir)
{
  /* HIOP is marked as a store because it needs its own DCE logic. */
  int uselo = ra_used(ir-1), usehi = ra_used(ir);  /* Loword/hiword used? */
  if (LJ_UNLIKELY(!(as->flags & JIT_F_OPT_DCE))) uselo = usehi = 1;
#if LJ_HASFFI || LJ_SOFTFP
  if ((ir-1)->o == IR_CONV) {  /* Conversions to/from 64 bit. */
    as->curins--;  /* Always skip the CONV. */
#if LJ_HASFFI && !LJ_SOFTFP
    if (usehi || uselo)
      asm_conv64(as, ir);
    return;
#endif
  } else if ((ir-1)->o <= IR_NE) {  /* 64 bit integer comparisons. ORDER IR. */
    as->curins--;  /* Always skip the loword comparison. */
#if LJ_SOFTFP
    if (!irt_isint(ir->t)) {
      asm_sfpcomp(as, ir-1);
      return;
    }
#endif
#if LJ_HASFFI
    asm_comp64(as, ir);
#endif
    return;
#if LJ_SOFTFP
  } else if ((ir-1)->o == IR_MIN || (ir-1)->o == IR_MAX) {
      as->curins--;  /* Always skip the loword min/max. */
    if (uselo || usehi)
      asm_sfpmin_max(as, ir-1);
    return;
#endif
  } else if ((ir-1)->o == IR_XSTORE) {
    as->curins--;  /* Handle both stores here. */
    if ((ir-1)->r != RID_SINK) {
      asm_xstore_(as, ir, 0);
      asm_xstore_(as, ir-1, 4);
    }
    return;
  }
#endif
  if (!usehi) return;  /* Skip unused hiword op for all remaining ops. */
  switch ((ir-1)->o) {
#if LJ_HASFFI
  case IR_ADD: as->curins--; asm_add64(as, ir); break;
  case IR_SUB: as->curins--; asm_sub64(as, ir); break;
  case IR_NEG: as->curins--; asm_neg64(as, ir); break;
  case IR_CNEWI:
    /* Nothing to do here. Handled by lo op itself. */
    break;
#endif
#if LJ_SOFTFP
  case IR_SLOAD: case IR_ALOAD: case IR_HLOAD: case IR_ULOAD: case IR_VLOAD:
  case IR_STRTO:
    if (!uselo)
      ra_allocref(as, ir->op1, RSET_GPR);  /* Mark lo op as used. */
    break;
  case IR_ASTORE: case IR_HSTORE: case IR_USTORE: case IR_TOSTR: case IR_TMPREF:
    /* Nothing to do here. Handled by lo op itself. */
    break;
#endif
  case IR_CALLN: case IR_CALLL: case IR_CALLS: case IR_CALLXS:
    if (!uselo)
      ra_allocref(as, ir->op1, RID2RSET(RID_RETLO));  /* Mark lo op as used. */
    break;
  default: lj_assertA(0, "bad HIOP for op %d", (ir-1)->o); break;
  }
}

/* -- Profiling ----------------------------------------------------------- */

static void asm_prof(ASMState *as, IRIns *ir)
{
  UNUSED(ir);
  asm_guardcc(as, CC_NE);
  emit_asi(as, PPCI_ANDIDOT, RID_TMP, RID_TMP, HOOK_PROFILE);
  emit_lsglptr(as, PPCI_LBZ, RID_TMP,
	       (int32_t)offsetof(global_State, hookmask));
}

/* -- Stack handling ------------------------------------------------------ */

/* Check Lua stack size for overflow. Use exit handler as fallback. */
static void asm_stack_check(ASMState *as, BCReg topslot,
			    IRIns *irp, RegSet allow, ExitNo exitno)
{
  /* Try to get an unused temp. register, otherwise spill/restore RID_RET*. */
  Reg tmp, pbase = irp ? (ra_hasreg(irp->r) ? irp->r : RID_TMP) : RID_BASE;
  rset_clear(allow, pbase);
  tmp = allow ? rset_pickbot(allow) :
		(pbase == RID_RETHI ? RID_RETLO : RID_RETHI);
  emit_condbranch(as, PPCI_BC, CC_LT, asm_exitstub_addr(as, exitno));
  if (allow == RSET_EMPTY)  /* Restore temp. register. */
    emit_tai(as, PPCI_LWZ, tmp, RID_SP, SPOFS_TMPW);
  else
    ra_modified(as, tmp);
  emit_ai(as, PPCI_CMPLWI, RID_TMP, (int32_t)(8*topslot));
  emit_tab(as, PPCI_SUBF, RID_TMP, pbase, tmp);
  emit_tai(as, PPCI_LWZ, tmp, tmp, offsetof(lua_State, maxstack));
  if (pbase == RID_TMP)
    emit_getgl(as, RID_TMP, jit_base);
  emit_getgl(as, tmp, cur_L);
  if (allow == RSET_EMPTY)  /* Spill temp. register. */
    emit_tai(as, PPCI_STW, tmp, RID_SP, SPOFS_TMPW);
}

/* Restore Lua stack from on-trace state. */
static void asm_stack_restore(ASMState *as, SnapShot *snap)
{
  SnapEntry *map = &as->T->snapmap[snap->mapofs];
  SnapEntry *flinks = &as->T->snapmap[snap_nextofs(as->T, snap)-1];
  MSize n, nent = snap->nent;
  /* Store the value of all modified slots to the Lua stack. */
  for (n = 0; n < nent; n++) {
    SnapEntry sn = map[n];
    BCReg s = snap_slot(sn);
    int32_t ofs = 8*((int32_t)s-1);
    IRRef ref = snap_ref(sn);
    IRIns *ir = IR(ref);
    if ((sn & SNAP_NORESTORE))
      continue;
    if (irt_isnum(ir->t)) {
#if LJ_SOFTFP
      Reg tmp;
      RegSet allow = rset_exclude(RSET_GPR, RID_BASE);
      /* LJ_SOFTFP: must be a number constant. */
      lj_assertA(irref_isk(ref), "unsplit FP op");
      tmp = ra_allock(as, (int32_t)ir_knum(ir)->u32.lo, allow);
      emit_tai(as, PPCI_STW, tmp, RID_BASE, ofs+(LJ_BE?4:0));
      if (rset_test(as->freeset, tmp+1)) allow = RID2RSET(tmp+1);
      tmp = ra_allock(as, (int32_t)ir_knum(ir)->u32.hi, allow);
      emit_tai(as, PPCI_STW, tmp, RID_BASE, ofs+(LJ_BE?0:4));
#else
      Reg src = ra_alloc1(as, ref, RSET_FPR);
      emit_fai(as, PPCI_STFD, src, RID_BASE, ofs);
#endif
    } else {
      Reg type;
      RegSet allow = rset_exclude(RSET_GPR, RID_BASE);
      lj_assertA(irt_ispri(ir->t) || irt_isaddr(ir->t) || irt_isinteger(ir->t),
		 "restore of IR type %d", irt_type(ir->t));
      if (!irt_ispri(ir->t)) {
	Reg src = ra_alloc1(as, ref, allow);
	rset_clear(allow, src);
	emit_tai(as, PPCI_STW, src, RID_BASE, ofs+4);
      }
      if ((sn & (SNAP_CONT|SNAP_FRAME))) {
	if (s == 0) continue;  /* Do not overwrite link to previous frame. */
	type = ra_allock(as, (int32_t)(*flinks--), allow);
#if LJ_SOFTFP
      } else if ((sn & SNAP_SOFTFPNUM)) {
	type = ra_alloc1(as, ref+1, rset_exclude(RSET_GPR, RID_BASE));
#endif
      } else if ((sn & SNAP_KEYINDEX)) {
	type = ra_allock(as, (int32_t)LJ_KEYINDEX, allow);
      } else {
	type = ra_allock(as, (int32_t)irt_toitype(ir->t), allow);
      }
      emit_tai(as, PPCI_STW, type, RID_BASE, ofs);
    }
    checkmclim(as);
  }
  lj_assertA(map + nent == flinks, "inconsistent frames in snapshot");
}

/* -- GC handling --------------------------------------------------------- */

/* Marker to prevent patching the GC check exit. */
#define PPC_NOPATCH_GC_CHECK	PPCI_ORIS

/* Check GC threshold and do one or more GC steps. */
static void asm_gc_check(ASMState *as)
{
  const CCallInfo *ci = &lj_ir_callinfo[IRCALL_lj_gc_step_jit];
  IRRef args[2];
  MCLabel l_end;
  Reg tmp;
  ra_evictset(as, RSET_SCRATCH);
  l_end = emit_label(as);
  /* Exit trace if in GCSatomic or GCSfinalize. Avoids syncing GC objects. */
  asm_guardcc(as, CC_NE);  /* Assumes asm_snap_prep() already done. */
  *--as->mcp = PPC_NOPATCH_GC_CHECK;
  emit_ai(as, PPCI_CMPWI, RID_RET, 0);
  args[0] = ASMREF_TMP1;  /* global_State *g */
  args[1] = ASMREF_TMP2;  /* MSize steps     */
  asm_gencall(as, ci, args);
  emit_tai(as, PPCI_ADDI, ra_releasetmp(as, ASMREF_TMP1), RID_JGL, -32768);
  tmp = ra_releasetmp(as, ASMREF_TMP2);
  emit_loadi(as, tmp, as->gcsteps);
  /* Jump around GC step if GC total < GC threshold. */
  emit_condbranch(as, PPCI_BC|PPCF_Y, CC_LT, l_end);
  emit_ab(as, PPCI_CMPLW, RID_TMP, tmp);
  emit_getgl(as, tmp, gc.threshold);
  emit_getgl(as, RID_TMP, gc.total);
  as->gcsteps = 0;
  checkmclim(as);
}

/* -- Loop handling ------------------------------------------------------- */

/* Fixup the loop branch. */
static void asm_loop_fixup(ASMState *as)
{
  MCode *p = as->mctop;
  MCode *target = as->mcp;
  if (as->loopinv) {  /* Inverted loop branch? */
    /* asm_guardcc already inverted the cond branch and patched the final b. */
    p[-2] = (p[-2] & (0xffff0000u & ~PPCF_Y)) | (((target-p+2) & 0x3fffu) << 2);
  } else {
    p[-1] = PPCI_B|(((target-p+1)&0x00ffffffu)<<2);
  }
}

/* Fixup the tail of the loop. */
static void asm_loop_tail_fixup(ASMState *as)
{
  UNUSED(as);  /* Nothing to do. */
}

/* -- Head of trace ------------------------------------------------------- */

/* Coalesce BASE register for a root trace. */
static void asm_head_root_base(ASMState *as)
{
  IRIns *ir = IR(REF_BASE);
  Reg r = ir->r;
  if (ra_hasreg(r)) {
    ra_free(as, r);
    if (rset_test(as->modset, r) || irt_ismarked(ir->t))
      ir->r = RID_INIT;  /* No inheritance for modified BASE register. */
    if (r != RID_BASE)
      emit_mr(as, r, RID_BASE);
  }
}

/* Coalesce BASE register for a side trace. */
static Reg asm_head_side_base(ASMState *as, IRIns *irp)
{
  IRIns *ir = IR(REF_BASE);
  Reg r = ir->r;
  if (ra_hasreg(r)) {
    ra_free(as, r);
    if (rset_test(as->modset, r) || irt_ismarked(ir->t))
      ir->r = RID_INIT;  /* No inheritance for modified BASE register. */
    if (irp->r == r) {
      return r;  /* Same BASE register already coalesced. */
    } else if (ra_hasreg(irp->r) && rset_test(as->freeset, irp->r)) {
      emit_mr(as, r, irp->r);  /* Move from coalesced parent reg. */
      return irp->r;
    } else {
      emit_getgl(as, r, jit_base);  /* Otherwise reload BASE. */
    }
  }
  return RID_NONE;
}

/* -- Tail of trace ------------------------------------------------------- */

/* Fixup the tail code. */
static void asm_tail_fixup(ASMState *as, TraceNo lnk)
{
  MCode *p = as->mctop;
  MCode *target;
  int32_t spadj = as->T->spadjust;
  if (spadj == 0) {
    *--p = PPCI_NOP;
    *--p = PPCI_NOP;
    as->mctop = p;
  } else {
    /* Patch stack adjustment. */
    lj_assertA(checki16(CFRAME_SIZE+spadj), "stack adjustment out of range");
    p[-3] = PPCI_ADDI | PPCF_T(RID_TMP) | PPCF_A(RID_SP) | (CFRAME_SIZE+spadj);
    p[-2] = PPCI_STWU | PPCF_T(RID_TMP) | PPCF_A(RID_SP) | spadj;
  }
  /* Patch exit branch. */
  target = lnk ? traceref(as->J, lnk)->mcode : (MCode *)lj_vm_exit_interp;
  p[-1] = PPCI_B|(((target-p+1)&0x00ffffffu)<<2);
}

/* Prepare tail of code. */
static void asm_tail_prep(ASMState *as)
{
  MCode *p = as->mctop - 1;  /* Leave room for exit branch. */
  if (as->loopref) {
    as->invmcp = as->mcp = p;
  } else {
    as->mcp = p-2;  /* Leave room for stack pointer adjustment. */
    as->invmcp = NULL;
  }
}

/* -- Trace setup --------------------------------------------------------- */

/* Ensure there are enough stack slots for call arguments. */
static Reg asm_setup_call_slots(ASMState *as, IRIns *ir, const CCallInfo *ci)
{
  IRRef args[CCI_NARGS_MAX*2];
  uint32_t i, nargs = CCI_XNARGS(ci);
  int nslots = 2, ngpr = REGARG_NUMGPR, nfpr = REGARG_NUMFPR;
  asm_collectargs(as, ir, ci, args);
  for (i = 0; i < nargs; i++)
    if (!LJ_SOFTFP && args[i] && irt_isfp(IR(args[i])->t)) {
      if (nfpr > 0) nfpr--; else nslots = (nslots+3) & ~1;
    } else {
      if (ngpr > 0) ngpr--; else nslots++;
    }
  if (nslots > as->evenspill)  /* Leave room for args in stack slots. */
    as->evenspill = nslots;
  return (!LJ_SOFTFP && irt_isfp(ir->t)) ? REGSP_HINT(RID_FPRET) :
					   REGSP_HINT(RID_RET);
}

static void asm_setup_target(ASMState *as)
{
  asm_exitstub_setup(as, as->T->nsnap + (as->parent ? 1 : 0));
}

/* -- Trace patching ------------------------------------------------------ */

/* Patch exit jumps of existing machine code to a new target. */
void lj_asm_patchexit(jit_State *J, GCtrace *T, ExitNo exitno, MCode *target)
{
  MCode *p = T->mcode;
  MCode *pe = (MCode *)((char *)p + T->szmcode);
  MCode *px = exitstub_trace_addr(T, exitno);
  MCode *cstart = NULL;
  MCode *mcarea = lj_mcode_patch(J, p, 0);
  int clearso = 0, patchlong = 1;
  for (; p < pe; p++) {
    /* Look for exitstub branch, try to replace with branch to target. */
    uint32_t ins = *p;
    if ((ins & 0xfc000000u) == 0x40000000u &&
	((ins ^ ((char *)px-(char *)p)) & 0xffffu) == 0) {
      ptrdiff_t delta = (char *)target - (char *)p;
      if (((ins >> 16) & 3) == (CC_SO&3)) {
	clearso = sizeof(MCode);
	delta -= sizeof(MCode);
      }
      /* Many, but not all short-range branches can be patched directly. */
      if (p[-1] == PPC_NOPATCH_GC_CHECK) {
	patchlong = 0;
      } else if (((delta + 0x8000) >> 16) == 0) {
	*p = (ins & 0xffdf0000u) | ((uint32_t)delta & 0xffffu) |
	     ((delta & 0x8000) * (PPCF_Y/0x8000));
	if (!cstart) cstart = p;
      }
    } else if ((ins & 0xfc000000u) == PPCI_B &&
	       ((ins ^ ((char *)px-(char *)p)) & 0x03ffffffu) == 0) {
      ptrdiff_t delta = (char *)target - (char *)p;
      lj_assertJ(((delta + 0x02000000) >> 26) == 0,
		 "branch target out of range");
      *p = PPCI_B | ((uint32_t)delta & 0x03ffffffu);
      if (!cstart) cstart = p;
    }
  }
  /* Always patch long-range branch in exit stub itself. Except, if we can't. */
  if (patchlong) {
    ptrdiff_t delta = (char *)target - (char *)px - clearso;
    lj_assertJ(((delta + 0x02000000) >> 26) == 0,
	       "branch target out of range");
    *px = PPCI_B | ((uint32_t)delta & 0x03ffffffu);
  }
  if (!cstart) cstart = px;
  lj_mcode_sync(cstart, px+1);
  if (clearso) {  /* Extend the current trace. Ugly workaround. */
    MCode *pp = J->cur.mcode;
    J->cur.szmcode += sizeof(MCode);
    *--pp = PPCI_MCRXR;  /* Clear SO flag. */
    J->cur.mcode = pp;
    lj_mcode_sync(pp, pp+1);
  }
  lj_mcode_patch(J, mcarea, 1);
}

