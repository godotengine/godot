/*
** ARM IR assembler (SSA IR -> machine code).
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

/* Allocate a scratch register pair. */
static Reg ra_scratchpair(ASMState *as, RegSet allow)
{
  RegSet pick1 = as->freeset & allow;
  RegSet pick2 = pick1 & (pick1 >> 1) & RSET_GPREVEN;
  Reg r;
  if (pick2) {
    r = rset_picktop(pick2);
  } else {
    RegSet pick = pick1 & (allow >> 1) & RSET_GPREVEN;
    if (pick) {
      r = rset_picktop(pick);
      ra_restore(as, regcost_ref(as->cost[r+1]));
    } else {
      pick = pick1 & (allow << 1) & RSET_GPRODD;
      if (pick) {
	r = ra_restore(as, regcost_ref(as->cost[rset_picktop(pick)-1]));
      } else {
	r = ra_evict(as, allow & (allow >> 1) & RSET_GPREVEN);
	ra_restore(as, regcost_ref(as->cost[r+1]));
      }
    }
  }
  lj_assertA(rset_test(RSET_GPREVEN, r), "odd reg %d", r);
  ra_modified(as, r);
  ra_modified(as, r+1);
  RA_DBGX((as, "scratchpair    $r $r", r, r+1));
  return r;
}

#if !LJ_SOFTFP
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
#endif

/* -- Guard handling ------------------------------------------------------ */

/* Generate an exit stub group at the bottom of the reserved MCode memory. */
static MCode *asm_exitstub_gen(ASMState *as, ExitNo group)
{
  MCode *mxp = as->mcbot;
  int i;
  if (mxp + 4*4+4*EXITSTUBS_PER_GROUP >= as->mctop)
    asm_mclimit(as);
  /* str lr, [sp]; bl ->vm_exit_handler; .long DISPATCH_address, group. */
  *mxp++ = ARMI_STR|ARMI_LS_P|ARMI_LS_U|ARMF_D(RID_LR)|ARMF_N(RID_SP);
  *mxp = ARMI_BL|((((MCode *)(void *)lj_vm_exit_handler-mxp)-2)&0x00ffffffu);
  mxp++;
  *mxp++ = (MCode)i32ptr(J2GG(as->J)->dispatch);  /* DISPATCH address */
  *mxp++ = group*EXITSTUBS_PER_GROUP;
  for (i = 0; i < EXITSTUBS_PER_GROUP; i++)
    *mxp++ = ARMI_B|((-6-i)&0x00ffffffu);
  lj_mcode_sync(as->mcbot, mxp);
  lj_mcode_commitbot(as->J, mxp);
  as->mcbot = mxp;
  as->mclim = as->mcbot + MCLIM_REDZONE;
  return mxp - EXITSTUBS_PER_GROUP;
}

/* Setup all needed exit stubs. */
static void asm_exitstub_setup(ASMState *as, ExitNo nexits)
{
  ExitNo i;
  if (nexits >= EXITSTUBS_PER_GROUP*LJ_MAX_EXITSTUBGR)
    lj_trace_err(as->J, LJ_TRERR_SNAPOV);
  for (i = 0; i < (nexits+EXITSTUBS_PER_GROUP-1)/EXITSTUBS_PER_GROUP; i++)
    if (as->J->exitstubgroup[i] == NULL)
      as->J->exitstubgroup[i] = asm_exitstub_gen(as, i);
}

/* Emit conditional branch to exit for guard. */
static void asm_guardcc(ASMState *as, ARMCC cc)
{
  MCode *target = exitstub_addr(as->J, as->snapno);
  MCode *p = as->mcp;
  if (LJ_UNLIKELY(p == as->invmcp)) {
    as->loopinv = 1;
    *p = ARMI_BL | ((target-p-2) & 0x00ffffffu);
    emit_branch(as, ARMF_CC(ARMI_B, cc^1), p+1);
    return;
  }
  emit_branch(as, ARMF_CC(ARMI_BL, cc), target);
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

/* Fuse array/hash/upvalue reference into register+offset operand. */
static Reg asm_fuseahuref(ASMState *as, IRRef ref, int32_t *ofsp, RegSet allow,
			  int lim)
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
	  if (ofs > -lim && ofs < lim) {
	    *ofsp = ofs;
	    return ra_alloc1(as, refa, allow);
	  }
	}
      }
    } else if (ir->o == IR_HREFK) {
      if (mayfuse(as, ref)) {
	int32_t ofs = (int32_t)(IR(ir->op2)->op2 * sizeof(Node));
	if (ofs < lim) {
	  *ofsp = ofs;
	  return ra_alloc1(as, ir->op1, allow);
	}
      }
    } else if (ir->o == IR_UREFC) {
      if (irref_isk(ir->op1)) {
	GCfunc *fn = ir_kfunc(IR(ir->op1));
	int32_t ofs = i32ptr(&gcref(fn->l.uvptr[(ir->op2 >> 8)])->uv.tv);
	*ofsp = (ofs & 255);  /* Mask out less bits to allow LDRD. */
	return ra_allock(as, (ofs & ~255), allow);
      }
    } else if (ir->o == IR_TMPREF) {
      *ofsp = 0;
      return RID_SP;
    }
  }
  *ofsp = 0;
  return ra_alloc1(as, ref, allow);
}

/* Fuse m operand into arithmetic/logic instructions. */
static uint32_t asm_fuseopm(ASMState *as, ARMIns ai, IRRef ref, RegSet allow)
{
  IRIns *ir = IR(ref);
  if (ra_hasreg(ir->r)) {
    ra_noweak(as, ir->r);
    return ARMF_M(ir->r);
  } else if (irref_isk(ref)) {
    uint32_t k = emit_isk12(ai, ir->i);
    if (k)
      return k;
  } else if (mayfuse(as, ref)) {
    if (ir->o >= IR_BSHL && ir->o <= IR_BROR) {
      Reg m = ra_alloc1(as, ir->op1, allow);
      ARMShift sh = ir->o == IR_BSHL ? ARMSH_LSL :
		    ir->o == IR_BSHR ? ARMSH_LSR :
		    ir->o == IR_BSAR ? ARMSH_ASR : ARMSH_ROR;
      if (irref_isk(ir->op2)) {
	return m | ARMF_SH(sh, (IR(ir->op2)->i & 31));
      } else {
	Reg s = ra_alloc1(as, ir->op2, rset_exclude(allow, m));
	return m | ARMF_RSH(sh, s);
      }
    } else if (ir->o == IR_ADD && ir->op1 == ir->op2) {
      Reg m = ra_alloc1(as, ir->op1, allow);
      return m | ARMF_SH(ARMSH_LSL, 1);
    }
  }
  return ra_allocref(as, ref, allow);
}

/* Fuse shifts into loads/stores. Only bother with BSHL 2 => lsl #2. */
static IRRef asm_fuselsl2(ASMState *as, IRRef ref)
{
  IRIns *ir = IR(ref);
  if (ra_noreg(ir->r) && mayfuse(as, ref) && ir->o == IR_BSHL &&
      irref_isk(ir->op2) && IR(ir->op2)->i == 2)
    return ir->op1;
  return 0;  /* No fusion. */
}

/* Fuse XLOAD/XSTORE reference into load/store operand. */
static void asm_fusexref(ASMState *as, ARMIns ai, Reg rd, IRRef ref,
			 RegSet allow, int32_t ofs)
{
  IRIns *ir = IR(ref);
  Reg base;
  if (ra_noreg(ir->r) && canfuse(as, ir)) {
    int32_t lim = (!LJ_SOFTFP && (ai & 0x08000000)) ? 1024 :
		   (ai & 0x04000000) ? 4096 : 256;
    if (ir->o == IR_ADD) {
      int32_t ofs2;
      if (irref_isk(ir->op2) &&
	  (ofs2 = ofs + IR(ir->op2)->i) > -lim && ofs2 < lim &&
	  (!(!LJ_SOFTFP && (ai & 0x08000000)) || !(ofs2 & 3))) {
	ofs = ofs2;
	ref = ir->op1;
      } else if (ofs == 0 && !(!LJ_SOFTFP && (ai & 0x08000000))) {
	IRRef lref = ir->op1, rref = ir->op2;
	Reg rn, rm;
	if ((ai & 0x04000000)) {
	  IRRef sref = asm_fuselsl2(as, rref);
	  if (sref) {
	    rref = sref;
	    ai |= ARMF_SH(ARMSH_LSL, 2);
	  } else if ((sref = asm_fuselsl2(as, lref)) != 0) {
	    lref = rref;
	    rref = sref;
	    ai |= ARMF_SH(ARMSH_LSL, 2);
	  }
	}
	rn = ra_alloc1(as, lref, allow);
	rm = ra_alloc1(as, rref, rset_exclude(allow, rn));
	if ((ai & 0x04000000)) ai |= ARMI_LS_R;
	emit_dnm(as, ai|ARMI_LS_P|ARMI_LS_U, rd, rn, rm);
	return;
      }
    } else if (ir->o == IR_STRREF && !(!LJ_SOFTFP && (ai & 0x08000000))) {
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
	Reg rn = ra_alloc1(as, ir->op1, allow);
	uint32_t m = asm_fuseopm(as, 0, ir->op2, rset_exclude(allow, rn));
	if ((ai & 0x04000000))
	  emit_lso(as, ai, rd, rd, ofs);
	else
	  emit_lsox(as, ai, rd, rd, ofs);
	emit_dn(as, ARMI_ADD^m, rd, rn);
	return;
      }
      if (ofs <= -lim || ofs >= lim) {
	Reg rn = ra_alloc1(as, ref, allow);
	Reg rm = ra_allock(as, ofs, rset_exclude(allow, rn));
	if ((ai & 0x04000000)) ai |= ARMI_LS_R;
	emit_dnm(as, ai|ARMI_LS_P|ARMI_LS_U, rd, rn, rm);
	return;
      }
    }
  }
  base = ra_alloc1(as, ref, allow);
#if !LJ_SOFTFP
  if ((ai & 0x08000000))
    emit_vlso(as, ai, rd, base, ofs);
  else
#endif
  if ((ai & 0x04000000))
    emit_lso(as, ai, rd, base, ofs);
  else
    emit_lsox(as, ai, rd, base, ofs);
}

#if !LJ_SOFTFP
/*
** Fuse to multiply-add/sub instruction.
** VMLA rounds twice (UMA, not FMA) -- no need to check for JIT_F_OPT_FMA.
** VFMA needs VFPv4, which is uncommon on the remaining ARM32 targets.
*/
static int asm_fusemadd(ASMState *as, IRIns *ir, ARMIns ai, ARMIns air)
{
  IRRef lref = ir->op1, rref = ir->op2;
  IRIns *irm;
  if (lref != rref &&
      ((mayfuse(as, lref) && (irm = IR(lref), irm->o == IR_MUL) &&
	ra_noreg(irm->r)) ||
       (mayfuse(as, rref) && (irm = IR(rref), irm->o == IR_MUL) &&
	(rref = lref, ai = air, ra_noreg(irm->r))))) {
    Reg dest = ra_dest(as, ir, RSET_FPR);
    Reg add = ra_hintalloc(as, rref, dest, RSET_FPR);
    Reg right, left = ra_alloc2(as, irm,
			rset_exclude(rset_exclude(RSET_FPR, dest), add));
    right = (left >> 8); left &= 255;
    emit_dnm(as, ai, (dest & 15), (left & 15), (right & 15));
    if (dest != add) emit_dm(as, ARMI_VMOV_D, (dest & 15), (add & 15));
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
  int32_t ofs = 0;
#if LJ_SOFTFP
  Reg gpr = REGARG_FIRSTGPR;
#else
  Reg gpr, fpr = REGARG_FIRSTFPR, fprodd = 0;
#endif
  if ((void *)ci->func)
    emit_call(as, (void *)ci->func);
#if !LJ_SOFTFP
  for (gpr = REGARG_FIRSTGPR; gpr <= REGARG_LASTGPR; gpr++)
    as->cost[gpr] = REGCOST(~0u, ASMREF_L);
  gpr = REGARG_FIRSTGPR;
#endif
  for (n = 0; n < nargs; n++) {  /* Setup args. */
    IRRef ref = args[n];
    IRIns *ir = IR(ref);
#if !LJ_SOFTFP
    if (ref && irt_isfp(ir->t)) {
      RegSet of = as->freeset;
      Reg src;
      if (!LJ_ABI_SOFTFP && !(ci->flags & CCI_VARARG)) {
	if (irt_isnum(ir->t)) {
	  if (fpr <= REGARG_LASTFPR) {
	    ra_leftov(as, fpr, ref);
	    fpr++;
	    continue;
	  }
	} else if (fprodd) {  /* Ick. */
	  src = ra_alloc1(as, ref, RSET_FPR);
	  emit_dm(as, ARMI_VMOV_S, (fprodd & 15), (src & 15) | 0x00400000);
	  fprodd = 0;
	  continue;
	} else if (fpr <= REGARG_LASTFPR) {
	  ra_leftov(as, fpr, ref);
	  fprodd = fpr++;
	  continue;
	}
	/* Workaround to protect argument GPRs from being used for remat. */
	as->freeset &= ~RSET_RANGE(REGARG_FIRSTGPR, REGARG_LASTGPR+1);
	src = ra_alloc1(as, ref, RSET_FPR);  /* May alloc GPR to remat FPR. */
	as->freeset |= (of & RSET_RANGE(REGARG_FIRSTGPR, REGARG_LASTGPR+1));
	fprodd = 0;
	goto stackfp;
      }
      /* Workaround to protect argument GPRs from being used for remat. */
      as->freeset &= ~RSET_RANGE(REGARG_FIRSTGPR, REGARG_LASTGPR+1);
      src = ra_alloc1(as, ref, RSET_FPR);  /* May alloc GPR to remat FPR. */
      as->freeset |= (of & RSET_RANGE(REGARG_FIRSTGPR, REGARG_LASTGPR+1));
      if (irt_isnum(ir->t)) gpr = (gpr+1) & ~1u;
      if (gpr <= REGARG_LASTGPR) {
	lj_assertA(rset_test(as->freeset, gpr),
		   "reg %d not free", gpr);  /* Must have been evicted. */
	if (irt_isnum(ir->t)) {
	  lj_assertA(rset_test(as->freeset, gpr+1),
		     "reg %d not free", gpr+1);  /* Ditto. */
	  emit_dnm(as, ARMI_VMOV_RR_D, gpr, gpr+1, (src & 15));
	  gpr += 2;
	} else {
	  emit_dn(as, ARMI_VMOV_R_S, gpr, (src & 15));
	  gpr++;
	}
      } else {
      stackfp:
	if (irt_isnum(ir->t)) ofs = (ofs + 4) & ~4;
	emit_spstore(as, ir, src, ofs);
	ofs += irt_isnum(ir->t) ? 8 : 4;
      }
    } else
#endif
    {
      if (gpr <= REGARG_LASTGPR) {
	lj_assertA(rset_test(as->freeset, gpr),
		   "reg %d not free", gpr);  /* Must have been evicted. */
	if (ref) ra_leftov(as, gpr, ref);
	gpr++;
      } else {
	if (ref) {
	  Reg r = ra_alloc1(as, ref, RSET_GPR);
	  emit_spstore(as, ir, r, ofs);
	}
	ofs += 4;
      }
    }
  }
}

/* Setup result reg/sp for call. Evict scratch regs. */
static void asm_setupresult(ASMState *as, IRIns *ir, const CCallInfo *ci)
{
  RegSet drop = RSET_SCRATCH;
  int hiop = ((ir+1)->o == IR_HIOP && !irt_isnil((ir+1)->t));
  if (ra_hasreg(ir->r))
    rset_clear(drop, ir->r);  /* Dest reg handled below. */
  if (hiop && ra_hasreg((ir+1)->r))
    rset_clear(drop, (ir+1)->r);  /* Dest reg handled below. */
  ra_evictset(as, drop);  /* Evictions must be performed first. */
  if (ra_used(ir)) {
    lj_assertA(!irt_ispri(ir->t), "PRI dest");
    if (!LJ_SOFTFP && irt_isfp(ir->t)) {
      if (LJ_ABI_SOFTFP || (ci->flags & (CCI_CASTU64|CCI_VARARG))) {
	Reg dest = (ra_dest(as, ir, RSET_FPR) & 15);
	if (irt_isnum(ir->t))
	  emit_dnm(as, ARMI_VMOV_D_RR, RID_RETLO, RID_RETHI, dest);
	else
	  emit_dn(as, ARMI_VMOV_S_R, RID_RET, dest);
      } else {
	ra_destreg(as, ir, RID_FPRET);
      }
    } else if (hiop) {
      ra_destpair(as, ir);
    } else {
      ra_destreg(as, ir, RID_RET);
    }
  }
  UNUSED(ci);
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
    ci.func = (ASMFunction)(void *)(irf->i);
  } else {  /* Need a non-argument register for indirect calls. */
    Reg freg = ra_alloc1(as, func, RSET_RANGE(RID_R4, RID_R12+1));
    emit_m(as, ARMI_BLXr, freg);
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
  /* Need to force a spill on REF_BASE now to update the stack slot. */
  emit_lso(as, ARMI_STR, base, RID_SP, ra_spill(as, IR(REF_BASE)));
  emit_setgl(as, base, jit_base);
  emit_addptr(as, base, -8*delta);
  asm_guardcc(as, CC_NE);
  emit_nm(as, ARMI_CMP, RID_TMP,
	  ra_allock(as, i32ptr(pc), rset_exclude(RSET_GPR, base)));
  emit_lso(as, ARMI_LDR, RID_TMP, base, -4);
}

/* -- Buffer operations --------------------------------------------------- */

#if LJ_HASBUFFER
static void asm_bufhdr_write(ASMState *as, Reg sb)
{
  Reg tmp = ra_scratch(as, rset_exclude(RSET_GPR, sb));
  IRIns irgc;
  int32_t addr = i32ptr((void *)&J2G(as->J)->cur_L);
  irgc.ot = IRT(0, IRT_PGC);  /* GC type. */
  emit_storeofs(as, &irgc, RID_TMP, sb, offsetof(SBuf, L));
  if ((as->flags & JIT_F_ARMV6T2)) {
    emit_dnm(as, ARMI_BFI, RID_TMP, lj_fls(SBUF_MASK_FLAG), tmp);
  } else {
    emit_dnm(as, ARMI_ORR, RID_TMP, RID_TMP, tmp);
    emit_dn(as, ARMI_AND|ARMI_K12|SBUF_MASK_FLAG, tmp, tmp);
  }
  emit_lso(as, ARMI_LDR, RID_TMP,
	   ra_allock(as, (addr & ~4095),
		     rset_exclude(rset_exclude(RSET_GPR, sb), tmp)),
	   (addr & 4095));
  emit_loadofs(as, &irgc, tmp, sb, offsetof(SBuf, L));
}
#endif

/* -- Type conversions ---------------------------------------------------- */

#if !LJ_SOFTFP
static void asm_tointg(ASMState *as, IRIns *ir, Reg left)
{
  Reg tmp = ra_scratch(as, rset_exclude(RSET_FPR, left));
  Reg dest = ra_dest(as, ir, RSET_GPR);
  asm_guardcc(as, CC_NE);
  emit_d(as, ARMI_VMRS, 0);
  emit_dm(as, ARMI_VCMP_D, (tmp & 15), (left & 15));
  emit_dm(as, ARMI_VCVT_F64_S32, (tmp & 15), (tmp & 15));
  emit_dn(as, ARMI_VMOV_R_S, dest, (tmp & 15));
  emit_dm(as, ARMI_VCVT_S32_F64, (tmp & 15), (left & 15));
}

static void asm_tobit(ASMState *as, IRIns *ir)
{
  RegSet allow = RSET_FPR;
  Reg left = ra_alloc1(as, ir->op1, allow);
  Reg right = ra_alloc1(as, ir->op2, rset_clear(allow, left));
  Reg tmp = ra_scratch(as, rset_clear(allow, right));
  Reg dest = ra_dest(as, ir, RSET_GPR);
  emit_dn(as, ARMI_VMOV_R_S, dest, (tmp & 15));
  emit_dnm(as, ARMI_VADD_D, (tmp & 15), (left & 15), (right & 15));
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
  lj_assertA(!irt_isint64(ir->t) && !(st == IRT_I64 || st == IRT_U64),
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
      emit_dm(as, st == IRT_NUM ? ARMI_VCVT_F32_F64 : ARMI_VCVT_F64_F32,
	      (dest & 15), (ra_alloc1(as, lref, RSET_FPR) & 15));
    } else {  /* Integer to FP conversion. */
      Reg left = ra_alloc1(as, lref, RSET_GPR);
      ARMIns ai = irt_isfloat(ir->t) ?
	(st == IRT_INT ? ARMI_VCVT_F32_S32 : ARMI_VCVT_F32_U32) :
	(st == IRT_INT ? ARMI_VCVT_F64_S32 : ARMI_VCVT_F64_U32);
      emit_dm(as, ai, (dest & 15), (dest & 15));
      emit_dn(as, ARMI_VMOV_S_R, left, (dest & 15));
    }
  } else if (stfp) {  /* FP to integer conversion. */
    if (irt_isguard(ir->t)) {
      /* Checked conversions are only supported from number to int. */
      lj_assertA(irt_isint(ir->t) && st == IRT_NUM,
		 "bad type for checked CONV");
      asm_tointg(as, ir, ra_alloc1(as, lref, RSET_FPR));
    } else {
      Reg left = ra_alloc1(as, lref, RSET_FPR);
      Reg tmp = ra_scratch(as, rset_exclude(RSET_FPR, left));
      Reg dest = ra_dest(as, ir, RSET_GPR);
      ARMIns ai;
      emit_dn(as, ARMI_VMOV_R_S, dest, (tmp & 15));
      ai = irt_isint(ir->t) ?
	(st == IRT_NUM ? ARMI_VCVT_S32_F64 : ARMI_VCVT_S32_F32) :
	(st == IRT_NUM ? ARMI_VCVT_U32_F64 : ARMI_VCVT_U32_F32);
      emit_dm(as, ai, (tmp & 15), (left & 15));
    }
  } else
#endif
  {
    Reg dest = ra_dest(as, ir, RSET_GPR);
    if (st >= IRT_I8 && st <= IRT_U16) {  /* Extend to 32 bit integer. */
      Reg left = ra_alloc1(as, lref, RSET_GPR);
      lj_assertA(irt_isint(ir->t) || irt_isu32(ir->t), "bad type for CONV EXT");
      if ((as->flags & JIT_F_ARMV6)) {
	ARMIns ai = st == IRT_I8 ? ARMI_SXTB :
		    st == IRT_U8 ? ARMI_UXTB :
		    st == IRT_I16 ? ARMI_SXTH : ARMI_UXTH;
	emit_dm(as, ai, dest, left);
      } else if (st == IRT_U8) {
	emit_dn(as, ARMI_AND|ARMI_K12|255, dest, left);
      } else {
	uint32_t shift = st == IRT_I8 ? 24 : 16;
	ARMShift sh = st == IRT_U16 ? ARMSH_LSR : ARMSH_ASR;
	emit_dm(as, ARMI_MOV|ARMF_SH(sh, shift), dest, RID_TMP);
	emit_dm(as, ARMI_MOV|ARMF_SH(ARMSH_LSL, shift), RID_TMP, left);
      }
    } else {  /* Handle 32/32 bit no-op (cast). */
      ra_leftov(as, dest, lref);  /* Do nothing, but may need to move regs. */
    }
  }
}

static void asm_strto(ASMState *as, IRIns *ir)
{
  const CCallInfo *ci = &lj_ir_callinfo[IRCALL_lj_strscan_num];
  IRRef args[2];
  Reg rlo = 0, rhi = 0, tmp;
  int destused = ra_used(ir);
  int32_t ofs = 0;
  ra_evictset(as, RSET_SCRATCH);
#if LJ_SOFTFP
  if (destused) {
    if (ra_hasspill(ir->s) && ra_hasspill((ir+1)->s) &&
	(ir->s & 1) == 0 && ir->s + 1 == (ir+1)->s) {
      int i;
      for (i = 0; i < 2; i++) {
	Reg r = (ir+i)->r;
	if (ra_hasreg(r)) {
	  ra_free(as, r);
	  ra_modified(as, r);
	  emit_spload(as, ir+i, r, sps_scale((ir+i)->s));
	}
      }
      ofs = sps_scale(ir->s);
      destused = 0;
    } else {
      rhi = ra_dest(as, ir+1, RSET_GPR);
      rlo = ra_dest(as, ir, rset_exclude(RSET_GPR, rhi));
    }
  }
  asm_guardcc(as, CC_EQ);
  if (destused) {
    emit_lso(as, ARMI_LDR, rhi, RID_SP, 4);
    emit_lso(as, ARMI_LDR, rlo, RID_SP, 0);
  }
#else
  UNUSED(rhi);
  if (destused) {
    if (ra_hasspill(ir->s)) {
      ofs = sps_scale(ir->s);
      destused = 0;
      if (ra_hasreg(ir->r)) {
	ra_free(as, ir->r);
	ra_modified(as, ir->r);
	emit_spload(as, ir, ir->r, ofs);
      }
    } else {
      rlo = ra_dest(as, ir, RSET_FPR);
    }
  }
  asm_guardcc(as, CC_EQ);
  if (destused)
    emit_vlso(as, ARMI_VLDR_D, rlo, RID_SP, 0);
#endif
  emit_n(as, ARMI_CMP|ARMI_K12|0, RID_RET);  /* Test return status. */
  args[0] = ir->op1;      /* GCstr *str */
  args[1] = ASMREF_TMP1;  /* TValue *n  */
  asm_gencall(as, ci, args);
  tmp = ra_releasetmp(as, ASMREF_TMP1);
  if (ofs == 0)
    emit_dm(as, ARMI_MOV, tmp, RID_SP);
  else
    emit_opk(as, ARMI_ADD, tmp, RID_SP, ofs, RSET_GPR);
}

/* -- Memory references --------------------------------------------------- */

/* Get pointer to TValue. */
static void asm_tvptr(ASMState *as, Reg dest, IRRef ref, MSize mode)
{
  if ((mode & IRTMPREF_IN1)) {
    IRIns *ir = IR(ref);
    if (irt_isnum(ir->t)) {
      if ((mode & IRTMPREF_OUT1)) {
#if LJ_SOFTFP
	lj_assertA(irref_isk(ref), "unsplit FP op");
	emit_dm(as, ARMI_MOV, dest, RID_SP);
	emit_lso(as, ARMI_STR,
		 ra_allock(as, (int32_t)ir_knum(ir)->u32.lo, RSET_GPR),
		 RID_SP, 0);
	emit_lso(as, ARMI_STR,
		 ra_allock(as, (int32_t)ir_knum(ir)->u32.hi, RSET_GPR),
		 RID_SP, 4);
#else
	Reg src = ra_alloc1(as, ref, RSET_FPR);
	emit_dm(as, ARMI_MOV, dest, RID_SP);
	emit_vlso(as, ARMI_VSTR_D, src, RID_SP, 0);
#endif
      } else if (irref_isk(ref)) {
	/* Use the number constant itself as a TValue. */
	ra_allockreg(as, i32ptr(ir_knum(ir)), dest);
      } else {
#if LJ_SOFTFP
	lj_assertA(0, "unsplit FP op");
#else
	/* Otherwise force a spill and use the spill slot. */
	emit_opk(as, ARMI_ADD, dest, RID_SP, ra_spill(as, ir), RSET_GPR);
#endif
      }
    } else {
      /* Otherwise use [sp] and [sp+4] to hold the TValue.
      ** This assumes the following call has max. 4 args.
      */
      Reg type;
      emit_dm(as, ARMI_MOV, dest, RID_SP);
      if (!irt_ispri(ir->t)) {
	Reg src = ra_alloc1(as, ref, RSET_GPR);
	emit_lso(as, ARMI_STR, src, RID_SP, 0);
      }
      if (LJ_SOFTFP && (ir+1)->o == IR_HIOP && !irt_isnil((ir+1)->t))
	type = ra_alloc1(as, ref+1, RSET_GPR);
      else
	type = ra_allock(as, irt_toitype(ir->t), RSET_GPR);
      emit_lso(as, ARMI_STR, type, RID_SP, 4);
    }
  } else {
    emit_dm(as, ARMI_MOV, dest, RID_SP);
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
    uint32_t k = emit_isk12(ARMI_ADD, ofs + 8*IR(ir->op2)->i);
    if (k) {
      base = ra_alloc1(as, refa, RSET_GPR);
      emit_dn(as, ARMI_ADD^k, dest, base);
      return;
    }
  }
  base = ra_alloc1(as, ir->op1, RSET_GPR);
  idx = ra_alloc1(as, ir->op2, rset_exclude(RSET_GPR, base));
  emit_dnm(as, ARMI_ADD|ARMF_SH(ARMSH_LSL, 3), dest, base, idx);
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
  Reg key = 0, keyhi = 0, keynumhi = RID_NONE, tmp = RID_TMP;
  IRRef refkey = ir->op2;
  IRIns *irkey = IR(refkey);
  IRType1 kt = irkey->t;
  int32_t k = 0, khi = emit_isk12(ARMI_CMP, irt_toitype(kt));
  uint32_t khash;
  MCLabel l_end, l_loop;
  rset_clear(allow, tab);
  if (!irref_isk(refkey) || irt_isstr(kt)) {
#if LJ_SOFTFP
    key = ra_alloc1(as, refkey, allow);
    rset_clear(allow, key);
    if (irkey[1].o == IR_HIOP) {
      if (ra_hasreg((irkey+1)->r)) {
	keynumhi = (irkey+1)->r;
	keyhi = RID_TMP;
	ra_noweak(as, keynumhi);
      } else {
	keyhi = keynumhi = ra_allocref(as, refkey+1, allow);
      }
      rset_clear(allow, keynumhi);
      khi = 0;
    }
#else
    if (irt_isnum(kt)) {
      key = ra_scratch(as, allow);
      rset_clear(allow, key);
      keyhi = keynumhi = ra_scratch(as, allow);
      rset_clear(allow, keyhi);
      khi = 0;
    } else {
      key = ra_alloc1(as, refkey, allow);
      rset_clear(allow, key);
    }
#endif
  } else if (irt_isnum(kt)) {
    int32_t val = (int32_t)ir_knum(irkey)->u32.lo;
    k = emit_isk12(ARMI_CMP, val);
    if (!k) {
      key = ra_allock(as, val, allow);
      rset_clear(allow, key);
    }
    val = (int32_t)ir_knum(irkey)->u32.hi;
    khi = emit_isk12(ARMI_CMP, val);
    if (!khi) {
      keyhi = ra_allock(as, val, allow);
      rset_clear(allow, keyhi);
    }
  } else if (!irt_ispri(kt)) {
    k = emit_isk12(ARMI_CMP, irkey->i);
    if (!k) {
      key = ra_alloc1(as, refkey, allow);
      rset_clear(allow, key);
    }
  }
  if (!irt_ispri(kt))
    tmp = ra_scratchpair(as, allow);

  /* Key not found in chain: jump to exit (if merged) or load niltv. */
  l_end = emit_label(as);
  as->invmcp = NULL;
  if (merge == IR_NE)
    asm_guardcc(as, CC_AL);
  else if (destused)
    emit_loada(as, dest, niltvg(J2G(as->J)));

  /* Follow hash chain until the end. */
  l_loop = --as->mcp;
  emit_n(as, ARMI_CMP|ARMI_K12|0, dest);
  emit_lso(as, ARMI_LDR, dest, dest, (int32_t)offsetof(Node, next));

  /* Type and value comparison. */
  if (merge == IR_EQ)
    asm_guardcc(as, CC_EQ);
  else
    emit_branch(as, ARMF_CC(ARMI_B, CC_EQ), l_end);
  if (!irt_ispri(kt)) {
    emit_nm(as, ARMF_CC(ARMI_CMP, CC_EQ)^k, tmp, key);
    emit_nm(as, ARMI_CMP^khi, tmp+1, keyhi);
    emit_lsox(as, ARMI_LDRD, tmp, dest, (int32_t)offsetof(Node, key));
  } else {
    emit_n(as, ARMI_CMP^khi, tmp);
    emit_lso(as, ARMI_LDR, tmp, dest, (int32_t)offsetof(Node, key.it));
  }
  *l_loop = ARMF_CC(ARMI_B, CC_NE) | ((as->mcp-l_loop-2) & 0x00ffffffu);

  /* Load main position relative to tab->node into dest. */
  khash = irref_isk(refkey) ? ir_khash(as, irkey) : 1;
  if (khash == 0) {
    emit_lso(as, ARMI_LDR, dest, tab, (int32_t)offsetof(GCtab, node));
  } else {
    emit_dnm(as, ARMI_ADD|ARMF_SH(ARMSH_LSL, 3), dest, dest, tmp);
    emit_dnm(as, ARMI_ADD|ARMF_SH(ARMSH_LSL, 1), tmp, tmp, tmp);
    if (irt_isstr(kt)) {  /* Fetch of str->sid is cheaper than ra_allock. */
      emit_dnm(as, ARMI_AND, tmp, tmp+1, RID_TMP);
      emit_lso(as, ARMI_LDR, dest, tab, (int32_t)offsetof(GCtab, node));
      emit_lso(as, ARMI_LDR, tmp+1, key, (int32_t)offsetof(GCstr, sid));
      emit_lso(as, ARMI_LDR, RID_TMP, tab, (int32_t)offsetof(GCtab, hmask));
    } else if (irref_isk(refkey)) {
      emit_opk(as, ARMI_AND, tmp, RID_TMP, (int32_t)khash,
	       rset_exclude(rset_exclude(RSET_GPR, tab), dest));
      emit_lso(as, ARMI_LDR, dest, tab, (int32_t)offsetof(GCtab, node));
      emit_lso(as, ARMI_LDR, RID_TMP, tab, (int32_t)offsetof(GCtab, hmask));
    } else {  /* Must match with hash*() in lj_tab.c. */
      if (ra_hasreg(keynumhi)) {  /* Canonicalize +-0.0 to 0.0. */
	if (keyhi == RID_TMP)
	  emit_dm(as, ARMF_CC(ARMI_MOV, CC_NE), keyhi, keynumhi);
	emit_d(as, ARMF_CC(ARMI_MOV, CC_EQ)|ARMI_K12|0, keyhi);
      }
      emit_dnm(as, ARMI_AND, tmp, tmp, RID_TMP);
      emit_dnm(as, ARMI_SUB|ARMF_SH(ARMSH_ROR, 32-HASH_ROT3), tmp, tmp, tmp+1);
      emit_lso(as, ARMI_LDR, dest, tab, (int32_t)offsetof(GCtab, node));
      emit_dnm(as, ARMI_EOR|ARMF_SH(ARMSH_ROR, 32-((HASH_ROT2+HASH_ROT1)&31)),
	       tmp, tmp+1, tmp);
      emit_lso(as, ARMI_LDR, RID_TMP, tab, (int32_t)offsetof(GCtab, hmask));
      emit_dnm(as, ARMI_SUB|ARMF_SH(ARMSH_ROR, 32-HASH_ROT1), tmp+1, tmp+1, tmp);
      if (ra_hasreg(keynumhi)) {
	emit_dnm(as, ARMI_EOR, tmp+1, tmp, key);
	emit_dnm(as, ARMI_ORR|ARMI_S, RID_TMP, tmp, key);  /* Test for +-0.0. */
	emit_dnm(as, ARMI_ADD, tmp, keynumhi, keynumhi);
#if !LJ_SOFTFP
	emit_dnm(as, ARMI_VMOV_RR_D, key, keynumhi,
		 (ra_alloc1(as, refkey, RSET_FPR) & 15));
#endif
      } else {
	emit_dnm(as, ARMI_EOR, tmp+1, tmp, key);
	emit_opk(as, ARMI_ADD, tmp, key, (int32_t)HASH_BIAS,
		 rset_exclude(rset_exclude(RSET_GPR, tab), key));
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
  Reg dest = (ra_used(ir) || ofs > 4095) ? ra_dest(as, ir, RSET_GPR) : RID_NONE;
  Reg node = ra_alloc1(as, ir->op1, RSET_GPR);
  Reg key = RID_NONE, type = RID_TMP, idx = node;
  RegSet allow = rset_exclude(RSET_GPR, node);
  lj_assertA(ofs % sizeof(Node) == 0, "unaligned HREFK slot");
  if (ofs > 4095) {
    idx = dest;
    rset_clear(allow, dest);
    kofs = (int32_t)offsetof(Node, key);
  } else if (ra_hasreg(dest)) {
    emit_opk(as, ARMI_ADD, dest, node, ofs, allow);
  }
  asm_guardcc(as, CC_NE);
  if (!irt_ispri(irkey->t)) {
    RegSet even = (as->freeset & allow);
    even = even & (even >> 1) & RSET_GPREVEN;
    if (even) {
      key = ra_scratch(as, even);
      if (rset_test(as->freeset, key+1)) {
	type = key+1;
	ra_modified(as, type);
      }
    } else {
      key = ra_scratch(as, allow);
    }
    rset_clear(allow, key);
  }
  rset_clear(allow, type);
  if (irt_isnum(irkey->t)) {
    emit_opk(as, ARMF_CC(ARMI_CMP, CC_EQ), 0, type,
	     (int32_t)ir_knum(irkey)->u32.hi, allow);
    emit_opk(as, ARMI_CMP, 0, key,
	     (int32_t)ir_knum(irkey)->u32.lo, allow);
  } else {
    if (ra_hasreg(key))
      emit_opk(as, ARMF_CC(ARMI_CMP, CC_EQ), 0, key, irkey->i, allow);
    emit_n(as, ARMI_CMN|ARMI_K12|-irt_toitype(irkey->t), type);
  }
  emit_lso(as, ARMI_LDR, type, idx, kofs+4);
  if (ra_hasreg(key)) emit_lso(as, ARMI_LDR, key, idx, kofs);
  if (ofs > 4095)
    emit_opk(as, ARMI_ADD, dest, node, ofs, RSET_GPR);
}

static void asm_uref(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  int guarded = (irt_t(ir->t) & (IRT_GUARD|IRT_TYPE)) == (IRT_GUARD|IRT_PGC);
  if (irref_isk(ir->op1) && !guarded) {
    GCfunc *fn = ir_kfunc(IR(ir->op1));
    MRef *v = &gcref(fn->l.uvptr[(ir->op2 >> 8)])->uv.v;
    emit_lsptr(as, ARMI_LDR, dest, v);
  } else {
    if (guarded) {
      asm_guardcc(as, ir->o == IR_UREFC ? CC_NE : CC_EQ);
      emit_n(as, ARMI_CMP|ARMI_K12|1, RID_TMP);
    }
    if (ir->o == IR_UREFC)
      emit_opk(as, ARMI_ADD, dest, dest,
	       (int32_t)offsetof(GCupval, tv), RSET_GPR);
    else
      emit_lso(as, ARMI_LDR, dest, dest, (int32_t)offsetof(GCupval, v));
    if (guarded)
      emit_lso(as, ARMI_LDRB, RID_TMP, dest,
	       (int32_t)offsetof(GCupval, closed));
    if (irref_isk(ir->op1)) {
      GCfunc *fn = ir_kfunc(IR(ir->op1));
      int32_t k = (int32_t)gcrefu(fn->l.uvptr[(ir->op2 >> 8)]);
      emit_loadi(as, dest, k);
    } else {
      emit_lso(as, ARMI_LDR, dest, ra_alloc1(as, ir->op1, RSET_GPR),
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
  Reg r;
  if (irref_isk(ref)) {
    IRRef tmp = refk; refk = ref; ref = tmp;
  } else if (!irref_isk(refk)) {
    uint32_t k, m = ARMI_K12|sizeof(GCstr);
    Reg right, left = ra_alloc1(as, ir->op1, RSET_GPR);
    IRIns *irr = IR(ir->op2);
    if (ra_hasreg(irr->r)) {
      ra_noweak(as, irr->r);
      right = irr->r;
    } else if (mayfuse(as, irr->op2) &&
	       irr->o == IR_ADD && irref_isk(irr->op2) &&
	       (k = emit_isk12(ARMI_ADD,
			       (int32_t)sizeof(GCstr) + IR(irr->op2)->i))) {
      m = k;
      right = ra_alloc1(as, irr->op1, rset_exclude(RSET_GPR, left));
    } else {
      right = ra_allocref(as, ir->op2, rset_exclude(RSET_GPR, left));
    }
    emit_dn(as, ARMI_ADD^m, dest, dest);
    emit_dnm(as, ARMI_ADD, dest, left, right);
    return;
  }
  r = ra_alloc1(as, ref, RSET_GPR);
  emit_opk(as, ARMI_ADD, dest, r,
	   sizeof(GCstr) + IR(refk)->i, rset_exclude(RSET_GPR, r));
}

/* -- Loads and stores ---------------------------------------------------- */

static ARMIns asm_fxloadins(ASMState *as, IRIns *ir)
{
  UNUSED(as);
  switch (irt_type(ir->t)) {
  case IRT_I8: return ARMI_LDRSB;
  case IRT_U8: return ARMI_LDRB;
  case IRT_I16: return ARMI_LDRSH;
  case IRT_U16: return ARMI_LDRH;
  case IRT_NUM: lj_assertA(!LJ_SOFTFP, "unsplit FP op"); return ARMI_VLDR_D;
  case IRT_FLOAT: if (!LJ_SOFTFP) return ARMI_VLDR_S;  /* fallthrough */
  default: return ARMI_LDR;
  }
}

static ARMIns asm_fxstoreins(ASMState *as, IRIns *ir)
{
  UNUSED(as);
  switch (irt_type(ir->t)) {
  case IRT_I8: case IRT_U8: return ARMI_STRB;
  case IRT_I16: case IRT_U16: return ARMI_STRH;
  case IRT_NUM: lj_assertA(!LJ_SOFTFP, "unsplit FP op"); return ARMI_VSTR_D;
  case IRT_FLOAT: if (!LJ_SOFTFP) return ARMI_VSTR_S;  /* fallthrough */
  default: return ARMI_STR;
  }
}

static void asm_fload(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  ARMIns ai = asm_fxloadins(as, ir);
  Reg idx;
  int32_t ofs;
  if (ir->op1 == REF_NIL) {  /* FLOAD from GG_State with offset. */
    idx = ra_allock(as, (int32_t)(ir->op2<<2) + (int32_t)J2GG(as->J), RSET_GPR);
    ofs = 0;
  } else {
    idx = ra_alloc1(as, ir->op1, RSET_GPR);
    if (ir->op2 == IRFL_TAB_ARRAY) {
      ofs = asm_fuseabase(as, ir->op1);
      if (ofs) {  /* Turn the t->array load into an add for colocated arrays. */
	emit_dn(as, ARMI_ADD|ARMI_K12|ofs, dest, idx);
	return;
      }
    }
    ofs = field_ofs[ir->op2];
  }
  if ((ai & 0x04000000))
    emit_lso(as, ai, dest, idx, ofs);
  else
    emit_lsox(as, ai, dest, idx, ofs);
}

static void asm_fstore(ASMState *as, IRIns *ir)
{
  if (ir->r != RID_SINK) {
    Reg src = ra_alloc1(as, ir->op2, RSET_GPR);
    IRIns *irf = IR(ir->op1);
    Reg idx = ra_alloc1(as, irf->op1, rset_exclude(RSET_GPR, src));
    int32_t ofs = field_ofs[irf->op2];
    ARMIns ai = asm_fxstoreins(as, ir);
    if ((ai & 0x04000000))
      emit_lso(as, ai, src, idx, ofs);
    else
      emit_lsox(as, ai, src, idx, ofs);
  }
}

static void asm_xload(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir,
		     (!LJ_SOFTFP && irt_isfp(ir->t)) ? RSET_FPR : RSET_GPR);
  lj_assertA(!(ir->op2 & IRXLOAD_UNALIGNED), "unaligned XLOAD");
  asm_fusexref(as, asm_fxloadins(as, ir), dest, ir->op1, RSET_GPR, 0);
}

static void asm_xstore_(ASMState *as, IRIns *ir, int32_t ofs)
{
  if (ir->r != RID_SINK) {
    Reg src = ra_alloc1(as, ir->op2,
			(!LJ_SOFTFP && irt_isfp(ir->t)) ? RSET_FPR : RSET_GPR);
    asm_fusexref(as, asm_fxstoreins(as, ir), src, ir->op1,
		 rset_exclude(RSET_GPR, src), ofs);
  }
}

#define asm_xstore(as, ir)	asm_xstore_(as, ir, 0)

static void asm_ahuvload(ASMState *as, IRIns *ir)
{
  int hiop = (LJ_SOFTFP && (ir+1)->o == IR_HIOP);
  IRType t = hiop ? IRT_NUM : irt_type(ir->t);
  Reg dest = RID_NONE, type = RID_NONE, idx;
  RegSet allow = RSET_GPR;
  int32_t ofs = 0;
  if (hiop && ra_used(ir+1)) {
    type = ra_dest(as, ir+1, allow);
    rset_clear(allow, type);
  }
  if (ra_used(ir)) {
    lj_assertA((LJ_SOFTFP ? 0 : irt_isnum(ir->t)) ||
	       irt_isint(ir->t) || irt_isaddr(ir->t),
	       "bad load type %d", irt_type(ir->t));
    dest = ra_dest(as, ir, (!LJ_SOFTFP && t == IRT_NUM) ? RSET_FPR : allow);
    rset_clear(allow, dest);
  }
  idx = asm_fuseahuref(as, ir->op1, &ofs, allow,
		       (!LJ_SOFTFP && t == IRT_NUM) ? 1024 : 4096);
  if (ir->o == IR_VLOAD) ofs += 8 * ir->op2;
  if (!hiop || type == RID_NONE) {
    rset_clear(allow, idx);
    if (ofs < 256 && ra_hasreg(dest) && (dest & 1) == 0 &&
	rset_test((as->freeset & allow), dest+1)) {
      type = dest+1;
      ra_modified(as, type);
    } else {
      type = RID_TMP;
    }
  }
  asm_guardcc(as, t == IRT_NUM ? CC_HS : CC_NE);
  emit_n(as, ARMI_CMN|ARMI_K12|-irt_toitype_(t), type);
  if (ra_hasreg(dest)) {
#if !LJ_SOFTFP
    if (t == IRT_NUM)
      emit_vlso(as, ARMI_VLDR_D, dest, idx, ofs);
    else
#endif
      emit_lso(as, ARMI_LDR, dest, idx, ofs);
  }
  emit_lso(as, ARMI_LDR, type, idx, ofs+4);
}

static void asm_ahustore(ASMState *as, IRIns *ir)
{
  if (ir->r != RID_SINK) {
    RegSet allow = RSET_GPR;
    Reg idx, src = RID_NONE, type = RID_NONE;
    int32_t ofs = 0;
#if !LJ_SOFTFP
    if (irt_isnum(ir->t)) {
      src = ra_alloc1(as, ir->op2, RSET_FPR);
      idx = asm_fuseahuref(as, ir->op1, &ofs, allow, 1024);
      emit_vlso(as, ARMI_VSTR_D, src, idx, ofs);
    } else
#endif
    {
      int hiop = (LJ_SOFTFP && (ir+1)->o == IR_HIOP);
      if (!irt_ispri(ir->t)) {
	src = ra_alloc1(as, ir->op2, allow);
	rset_clear(allow, src);
      }
      if (hiop)
	type = ra_alloc1(as, (ir+1)->op2, allow);
      else
	type = ra_allock(as, (int32_t)irt_toitype(ir->t), allow);
      idx = asm_fuseahuref(as, ir->op1, &ofs, rset_exclude(allow, type), 4096);
      if (ra_hasreg(src)) emit_lso(as, ARMI_STR, src, idx, ofs);
      emit_lso(as, ARMI_STR, type, idx, ofs+4);
    }
  }
}

static void asm_sload(ASMState *as, IRIns *ir)
{
  int32_t ofs = 8*((int32_t)ir->op1-1) + ((ir->op2 & IRSLOAD_FRAME) ? 4 : 0);
  int hiop = (LJ_SOFTFP && (ir+1)->o == IR_HIOP);
  IRType t = hiop ? IRT_NUM : irt_type(ir->t);
  Reg dest = RID_NONE, type = RID_NONE, base;
  RegSet allow = RSET_GPR;
  lj_assertA(!(ir->op2 & IRSLOAD_PARENT),
	     "bad parent SLOAD");  /* Handled by asm_head_side(). */
  lj_assertA(irt_isguard(ir->t) || !(ir->op2 & IRSLOAD_TYPECHECK),
	     "inconsistent SLOAD variant");
#if LJ_SOFTFP
  lj_assertA(!(ir->op2 & IRSLOAD_CONVERT),
	     "unsplit SLOAD convert");  /* Handled by LJ_SOFTFP SPLIT. */
  if (hiop && ra_used(ir+1)) {
    type = ra_dest(as, ir+1, allow);
    rset_clear(allow, type);
  }
#else
  if ((ir->op2 & IRSLOAD_CONVERT) && irt_isguard(ir->t) && t == IRT_INT) {
    dest = ra_scratch(as, RSET_FPR);
    asm_tointg(as, ir, dest);
    t = IRT_NUM;  /* Continue with a regular number type check. */
  } else
#endif
  if (ra_used(ir)) {
    Reg tmp = RID_NONE;
    if ((ir->op2 & IRSLOAD_CONVERT))
      tmp = ra_scratch(as, t == IRT_INT ? RSET_FPR : RSET_GPR);
    lj_assertA((LJ_SOFTFP ? 0 : irt_isnum(ir->t)) ||
	       irt_isint(ir->t) || irt_isaddr(ir->t),
	       "bad SLOAD type %d", irt_type(ir->t));
    dest = ra_dest(as, ir, (!LJ_SOFTFP && t == IRT_NUM) ? RSET_FPR : allow);
    rset_clear(allow, dest);
    base = ra_alloc1(as, REF_BASE, allow);
    if ((ir->op2 & IRSLOAD_CONVERT)) {
      if (t == IRT_INT) {
	emit_dn(as, ARMI_VMOV_R_S, dest, (tmp & 15));
	emit_dm(as, ARMI_VCVT_S32_F64, (tmp & 15), (tmp & 15));
	t = IRT_NUM;  /* Check for original type. */
      } else {
	emit_dm(as, ARMI_VCVT_F64_S32, (dest & 15), (dest & 15));
	emit_dn(as, ARMI_VMOV_S_R, tmp, (dest & 15));
	t = IRT_INT;  /* Check for original type. */
      }
      dest = tmp;
    }
    goto dotypecheck;
  }
  base = ra_alloc1(as, REF_BASE, allow);
dotypecheck:
  rset_clear(allow, base);
  if ((ir->op2 & IRSLOAD_TYPECHECK)) {
    if (ra_noreg(type)) {
      if (ofs < 256 && ra_hasreg(dest) && (dest & 1) == 0 &&
	  rset_test((as->freeset & allow), dest+1)) {
	type = dest+1;
	ra_modified(as, type);
      } else {
	type = RID_TMP;
      }
    }
    asm_guardcc(as, t == IRT_NUM ? CC_HS : CC_NE);
    if ((ir->op2 & IRSLOAD_KEYINDEX)) {
      emit_n(as, ARMI_CMN|ARMI_K12|1, type);
      emit_dn(as, ARMI_EOR^emit_isk12(ARMI_EOR, ~LJ_KEYINDEX), type, type);
    } else {
      emit_n(as, ARMI_CMN|ARMI_K12|-irt_toitype_(t), type);
    }
  }
  if (ra_hasreg(dest)) {
#if !LJ_SOFTFP
    if (t == IRT_NUM) {
      if (ofs < 1024) {
	emit_vlso(as, ARMI_VLDR_D, dest, base, ofs);
      } else {
	if (ra_hasreg(type)) emit_lso(as, ARMI_LDR, type, base, ofs+4);
	emit_vlso(as, ARMI_VLDR_D, dest, RID_TMP, 0);
	emit_opk(as, ARMI_ADD, RID_TMP, base, ofs, allow);
	return;
      }
    } else
#endif
      emit_lso(as, ARMI_LDR, dest, base, ofs);
  }
  if (ra_hasreg(type)) emit_lso(as, ARMI_LDR, type, base, ofs+4);
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
  RegSet allow = (RSET_GPR & ~RSET_SCRATCH);
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
    int32_t ofs = sizeof(GCcdata);
    lj_assertA(sz == 4 || sz == 8, "bad CNEWI size %d", sz);
    if (sz == 8) {
      ofs += 4; ir++;
      lj_assertA(ir->o == IR_HIOP, "expected HIOP for CNEWI");
    }
    for (;;) {
      Reg r = ra_alloc1(as, ir->op2, allow);
      emit_lso(as, ARMI_STR, r, RID_RET, ofs);
      rset_clear(allow, r);
      if (ofs == sizeof(GCcdata)) break;
      ofs -= 4; ir--;
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
  {
    uint32_t k = emit_isk12(ARMI_MOV, id);
    Reg r = k ? RID_R1 : ra_allock(as, id, allow);
    emit_lso(as, ARMI_STRB, RID_TMP, RID_RET, offsetof(GCcdata, gct));
    emit_lsox(as, ARMI_STRH, r, RID_RET, offsetof(GCcdata, ctypeid));
    emit_d(as, ARMI_MOV|ARMI_K12|~LJ_TCDATA, RID_TMP);
    if (k) emit_d(as, ARMI_MOV^k, RID_R1);
  }
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
  Reg link = ra_scratch(as, rset_exclude(RSET_GPR, tab));
  Reg gr = ra_allock(as, i32ptr(J2G(as->J)),
		     rset_exclude(rset_exclude(RSET_GPR, tab), link));
  Reg mark = RID_TMP;
  MCLabel l_end = emit_label(as);
  emit_lso(as, ARMI_STR, link, tab, (int32_t)offsetof(GCtab, gclist));
  emit_lso(as, ARMI_STRB, mark, tab, (int32_t)offsetof(GCtab, marked));
  emit_lso(as, ARMI_STR, tab, gr,
	   (int32_t)offsetof(global_State, gc.grayagain));
  emit_dn(as, ARMI_BIC|ARMI_K12|LJ_GC_BLACK, mark, mark);
  emit_lso(as, ARMI_LDR, link, gr,
	   (int32_t)offsetof(global_State, gc.grayagain));
  emit_branch(as, ARMF_CC(ARMI_B, CC_EQ), l_end);
  emit_n(as, ARMI_TST|ARMI_K12|LJ_GC_BLACK, mark);
  emit_lso(as, ARMI_LDRB, mark, tab, (int32_t)offsetof(GCtab, marked));
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
  if ((l_end[-1] >> 28) == CC_AL)
    l_end[-1] = ARMF_CC(l_end[-1], CC_NE);
  else
    emit_branch(as, ARMF_CC(ARMI_B, CC_EQ), l_end);
  ra_allockreg(as, i32ptr(J2G(as->J)), ra_releasetmp(as, ASMREF_TMP1));
  obj = IR(ir->op1)->r;
  tmp = ra_scratch(as, rset_exclude(RSET_GPR, obj));
  emit_n(as, ARMF_CC(ARMI_TST, CC_NE)|ARMI_K12|LJ_GC_BLACK, tmp);
  emit_n(as, ARMI_TST|ARMI_K12|LJ_GC_WHITES, RID_TMP);
  val = ra_alloc1(as, ir->op2, rset_exclude(RSET_GPR, obj));
  emit_lso(as, ARMI_LDRB, tmp, obj,
	   (int32_t)offsetof(GCupval, marked)-(int32_t)offsetof(GCupval, tv));
  emit_lso(as, ARMI_LDRB, RID_TMP, val, (int32_t)offsetof(GChead, marked));
}

/* -- Arithmetic and logic operations ------------------------------------- */

#if !LJ_SOFTFP
static void asm_fparith(ASMState *as, IRIns *ir, ARMIns ai)
{
  Reg dest = ra_dest(as, ir, RSET_FPR);
  Reg right, left = ra_alloc2(as, ir, RSET_FPR);
  right = (left >> 8); left &= 255;
  emit_dnm(as, ai, (dest & 15), (left & 15), (right & 15));
}

static void asm_fpunary(ASMState *as, IRIns *ir, ARMIns ai)
{
  Reg dest = ra_dest(as, ir, RSET_FPR);
  Reg left = ra_hintalloc(as, ir->op1, dest, RSET_FPR);
  emit_dm(as, ai, (dest & 15), (left & 15));
}

static void asm_callround(ASMState *as, IRIns *ir, int id)
{
  /* The modified regs must match with the *.dasc implementation. */
  RegSet drop = RID2RSET(RID_R0)|RID2RSET(RID_R1)|RID2RSET(RID_R2)|
		RID2RSET(RID_R3)|RID2RSET(RID_R12);
  RegSet of;
  Reg dest, src;
  ra_evictset(as, drop);
  dest = ra_dest(as, ir, RSET_FPR);
  emit_dnm(as, ARMI_VMOV_D_RR, RID_RETLO, RID_RETHI, (dest & 15));
  emit_call(as, id == IRFPM_FLOOR ? (void *)lj_vm_floor_sf :
		id == IRFPM_CEIL ? (void *)lj_vm_ceil_sf :
				   (void *)lj_vm_trunc_sf);
  /* Workaround to protect argument GPRs from being used for remat. */
  of = as->freeset;
  as->freeset &= ~RSET_RANGE(RID_R0, RID_R1+1);
  as->cost[RID_R0] = as->cost[RID_R1] = REGCOST(~0u, ASMREF_L);
  src = ra_alloc1(as, ir->op1, RSET_FPR);  /* May alloc GPR to remat FPR. */
  as->freeset |= (of & RSET_RANGE(RID_R0, RID_R1+1));
  emit_dnm(as, ARMI_VMOV_RR_D, RID_R0, RID_R1, (src & 15));
}

static void asm_fpmath(ASMState *as, IRIns *ir)
{
  if (ir->op2 <= IRFPM_TRUNC)
    asm_callround(as, ir, ir->op2);
  else if (ir->op2 == IRFPM_SQRT)
    asm_fpunary(as, ir, ARMI_VSQRT_D);
  else
    asm_callid(as, ir, IRCALL_lj_vm_floor + ir->op2);
}
#endif

static int asm_swapops(ASMState *as, IRRef lref, IRRef rref)
{
  IRIns *ir;
  if (irref_isk(rref))
    return 0;  /* Don't swap constants to the left. */
  if (irref_isk(lref))
    return 1;  /* But swap constants to the right. */
  ir = IR(rref);
  if ((ir->o >= IR_BSHL && ir->o <= IR_BROR) ||
      (ir->o == IR_ADD && ir->op1 == ir->op2))
    return 0;  /* Don't swap fusable operands to the left. */
  ir = IR(lref);
  if ((ir->o >= IR_BSHL && ir->o <= IR_BROR) ||
      (ir->o == IR_ADD && ir->op1 == ir->op2))
    return 1;  /* But swap fusable operands to the right. */
  return 0;  /* Otherwise don't swap. */
}

static void asm_intop(ASMState *as, IRIns *ir, ARMIns ai)
{
  IRRef lref = ir->op1, rref = ir->op2;
  Reg left, dest = ra_dest(as, ir, RSET_GPR);
  uint32_t m;
  if (asm_swapops(as, lref, rref)) {
    IRRef tmp = lref; lref = rref; rref = tmp;
    if ((ai & ~ARMI_S) == ARMI_SUB || (ai & ~ARMI_S) == ARMI_SBC)
      ai ^= (ARMI_SUB^ARMI_RSB);
  }
  left = ra_hintalloc(as, lref, dest, RSET_GPR);
  m = asm_fuseopm(as, ai, rref, rset_exclude(RSET_GPR, left));
  if (irt_isguard(ir->t)) {  /* For IR_ADDOV etc. */
    asm_guardcc(as, CC_VS);
    ai |= ARMI_S;
  }
  emit_dn(as, ai^m, dest, left);
}

/* Try to drop cmp r, #0. */
static ARMIns asm_drop_cmp0(ASMState *as, ARMIns ai)
{
  if (as->flagmcp == as->mcp) {
    uint32_t cc = (as->mcp[1] >> 28);
    as->flagmcp = NULL;
    if (cc <= CC_NE) {
      as->mcp++;
      ai |= ARMI_S;
    } else if (cc == CC_GE) {
      *++as->mcp ^= ((CC_GE^CC_PL) << 28);
      ai |= ARMI_S;
    } else if (cc == CC_LT) {
      *++as->mcp ^= ((CC_LT^CC_MI) << 28);
      ai |= ARMI_S;
    }  /* else: other conds don't work in general. */
  }
  return ai;
}

static void asm_intop_s(ASMState *as, IRIns *ir, ARMIns ai)
{
  asm_intop(as, ir, asm_drop_cmp0(as, ai));
}

static void asm_intneg(ASMState *as, IRIns *ir, ARMIns ai)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  Reg left = ra_hintalloc(as, ir->op1, dest, RSET_GPR);
  emit_dn(as, ai|ARMI_K12|0, dest, left);
}

/* NYI: use add/shift for MUL(OV) with constants. FOLD only does 2^k. */
static void asm_intmul(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  Reg left = ra_alloc1(as, ir->op1, rset_exclude(RSET_GPR, dest));
  Reg right = ra_alloc1(as, ir->op2, rset_exclude(RSET_GPR, left));
  Reg tmp = RID_NONE;
  /* ARMv5 restriction: dest != left and dest_hi != left. */
  if (dest == left && left != right) { left = right; right = dest; }
  if (irt_isguard(ir->t)) {  /* IR_MULOV */
    if (!(as->flags & JIT_F_ARMV6) && dest == left)
      tmp = left = ra_scratch(as, rset_exclude(RSET_GPR, left));
    asm_guardcc(as, CC_NE);
    emit_nm(as, ARMI_TEQ|ARMF_SH(ARMSH_ASR, 31), RID_TMP, dest);
    emit_dnm(as, ARMI_SMULL|ARMF_S(right), dest, RID_TMP, left);
  } else {
    if (!(as->flags & JIT_F_ARMV6) && dest == left) tmp = left = RID_TMP;
    emit_nm(as, ARMI_MUL|ARMF_S(right), dest, left);
  }
  /* Only need this for the dest == left == right case. */
  if (ra_hasreg(tmp)) emit_dm(as, ARMI_MOV, tmp, right);
}

static void asm_add(ASMState *as, IRIns *ir)
{
#if !LJ_SOFTFP
  if (irt_isnum(ir->t)) {
    if (!asm_fusemadd(as, ir, ARMI_VMLA_D, ARMI_VMLA_D))
      asm_fparith(as, ir, ARMI_VADD_D);
    return;
  }
#endif
  asm_intop_s(as, ir, ARMI_ADD);
}

static void asm_sub(ASMState *as, IRIns *ir)
{
#if !LJ_SOFTFP
  if (irt_isnum(ir->t)) {
    if (!asm_fusemadd(as, ir, ARMI_VNMLS_D, ARMI_VMLS_D))
      asm_fparith(as, ir, ARMI_VSUB_D);
    return;
  }
#endif
  asm_intop_s(as, ir, ARMI_SUB);
}

static void asm_mul(ASMState *as, IRIns *ir)
{
#if !LJ_SOFTFP
  if (irt_isnum(ir->t)) {
    asm_fparith(as, ir, ARMI_VMUL_D);
    return;
  }
#endif
  asm_intmul(as, ir);
}

#define asm_addov(as, ir)	asm_add(as, ir)
#define asm_subov(as, ir)	asm_sub(as, ir)
#define asm_mulov(as, ir)	asm_mul(as, ir)

#if !LJ_SOFTFP
#define asm_fpdiv(as, ir)	asm_fparith(as, ir, ARMI_VDIV_D)
#define asm_abs(as, ir)		asm_fpunary(as, ir, ARMI_VABS_D)
#endif

static void asm_neg(ASMState *as, IRIns *ir)
{
#if !LJ_SOFTFP
  if (irt_isnum(ir->t)) {
    asm_fpunary(as, ir, ARMI_VNEG_D);
    return;
  }
#endif
  asm_intneg(as, ir, ARMI_RSB);
}

static void asm_bitop(ASMState *as, IRIns *ir, ARMIns ai)
{
  ai = asm_drop_cmp0(as, ai);
  if (ir->op2 == 0) {
    Reg dest = ra_dest(as, ir, RSET_GPR);
    uint32_t m = asm_fuseopm(as, ai, ir->op1, RSET_GPR);
    emit_d(as, ai^m, dest);
  } else {
    /* NYI: Turn BAND !k12 into uxtb, uxth or bfc or shl+shr. */
    asm_intop(as, ir, ai);
  }
}

#define asm_bnot(as, ir)	asm_bitop(as, ir, ARMI_MVN)

static void asm_bswap(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  Reg left = ra_alloc1(as, ir->op1, RSET_GPR);
  if ((as->flags & JIT_F_ARMV6)) {
    emit_dm(as, ARMI_REV, dest, left);
  } else {
    Reg tmp2 = dest;
    if (tmp2 == left)
      tmp2 = ra_scratch(as, rset_exclude(rset_exclude(RSET_GPR, dest), left));
    emit_dnm(as, ARMI_EOR|ARMF_SH(ARMSH_LSR, 8), dest, tmp2, RID_TMP);
    emit_dm(as, ARMI_MOV|ARMF_SH(ARMSH_ROR, 8), tmp2, left);
    emit_dn(as, ARMI_BIC|ARMI_K12|256*8|255, RID_TMP, RID_TMP);
    emit_dnm(as, ARMI_EOR|ARMF_SH(ARMSH_ROR, 16), RID_TMP, left, left);
  }
}

#define asm_band(as, ir)	asm_bitop(as, ir, ARMI_AND)
#define asm_bor(as, ir)		asm_bitop(as, ir, ARMI_ORR)
#define asm_bxor(as, ir)	asm_bitop(as, ir, ARMI_EOR)

static void asm_bitshift(ASMState *as, IRIns *ir, ARMShift sh)
{
  if (irref_isk(ir->op2)) {  /* Constant shifts. */
    /* NYI: Turn SHL+SHR or BAND+SHR into uxtb, uxth or ubfx. */
    /* NYI: Turn SHL+ASR into sxtb, sxth or sbfx. */
    Reg dest = ra_dest(as, ir, RSET_GPR);
    Reg left = ra_alloc1(as, ir->op1, RSET_GPR);
    int32_t shift = (IR(ir->op2)->i & 31);
    emit_dm(as, ARMI_MOV|ARMF_SH(sh, shift), dest, left);
  } else {
    Reg dest = ra_dest(as, ir, RSET_GPR);
    Reg left = ra_alloc1(as, ir->op1, RSET_GPR);
    Reg right = ra_alloc1(as, ir->op2, rset_exclude(RSET_GPR, left));
    emit_dm(as, ARMI_MOV|ARMF_RSH(sh, right), dest, left);
  }
}

#define asm_bshl(as, ir)	asm_bitshift(as, ir, ARMSH_LSL)
#define asm_bshr(as, ir)	asm_bitshift(as, ir, ARMSH_LSR)
#define asm_bsar(as, ir)	asm_bitshift(as, ir, ARMSH_ASR)
#define asm_bror(as, ir)	asm_bitshift(as, ir, ARMSH_ROR)
#define asm_brol(as, ir)	lj_assertA(0, "unexpected BROL")

static void asm_intmin_max(ASMState *as, IRIns *ir, int cc)
{
  uint32_t kcmp = 0, kmov = 0;
  Reg dest = ra_dest(as, ir, RSET_GPR);
  Reg left = ra_hintalloc(as, ir->op1, dest, RSET_GPR);
  Reg right = 0;
  if (irref_isk(ir->op2)) {
    kcmp = emit_isk12(ARMI_CMP, IR(ir->op2)->i);
    if (kcmp) kmov = emit_isk12(ARMI_MOV, IR(ir->op2)->i);
  }
  if (!kmov) {
    kcmp = 0;
    right = ra_alloc1(as, ir->op2, rset_exclude(RSET_GPR, left));
  }
  if (kmov || dest != right) {
    emit_dm(as, ARMF_CC(ARMI_MOV, cc)^kmov, dest, right);
    cc ^= 1;  /* Must use opposite conditions for paired moves. */
  } else {
    cc ^= (CC_LT^CC_GT);  /* Otherwise may swap CC_LT <-> CC_GT. */
  }
  if (dest != left) emit_dm(as, ARMF_CC(ARMI_MOV, cc), dest, left);
  emit_nm(as, ARMI_CMP^kcmp, left, right);
}

#if LJ_SOFTFP
static void asm_sfpmin_max(ASMState *as, IRIns *ir, int cc)
{
  const CCallInfo *ci = &lj_ir_callinfo[IRCALL_softfp_cmp];
  RegSet drop = RSET_SCRATCH;
  Reg r;
  IRRef args[4];
  args[0] = ir->op1; args[1] = (ir+1)->op1;
  args[2] = ir->op2; args[3] = (ir+1)->op2;
  /* __aeabi_cdcmple preserves r0-r3. */
  if (ra_hasreg(ir->r)) rset_clear(drop, ir->r);
  if (ra_hasreg((ir+1)->r)) rset_clear(drop, (ir+1)->r);
  if (!rset_test(as->freeset, RID_R2) &&
      regcost_ref(as->cost[RID_R2]) == args[2]) rset_clear(drop, RID_R2);
  if (!rset_test(as->freeset, RID_R3) &&
      regcost_ref(as->cost[RID_R3]) == args[3]) rset_clear(drop, RID_R3);
  ra_evictset(as, drop);
  ra_destpair(as, ir);
  emit_dm(as, ARMF_CC(ARMI_MOV, cc), RID_RETHI, RID_R3);
  emit_dm(as, ARMF_CC(ARMI_MOV, cc), RID_RETLO, RID_R2);
  emit_call(as, (void *)ci->func);
  for (r = RID_R0; r <= RID_R3; r++)
    ra_leftov(as, r, args[r-RID_R0]);
}
#else
static void asm_fpmin_max(ASMState *as, IRIns *ir, int cc)
{
  Reg dest = (ra_dest(as, ir, RSET_FPR) & 15);
  Reg right, left = ra_alloc2(as, ir, RSET_FPR);
  right = ((left >> 8) & 15); left &= 15;
  if (dest != left) emit_dm(as, ARMF_CC(ARMI_VMOV_D, cc^1), dest, left);
  if (dest != right) emit_dm(as, ARMF_CC(ARMI_VMOV_D, cc), dest, right);
  emit_d(as, ARMI_VMRS, 0);
  emit_dm(as, ARMI_VCMP_D, left, right);
}
#endif

static void asm_min_max(ASMState *as, IRIns *ir, int cc, int fcc)
{
#if LJ_SOFTFP
  UNUSED(fcc);
#else
  if (irt_isnum(ir->t))
    asm_fpmin_max(as, ir, fcc);
  else
#endif
    asm_intmin_max(as, ir, cc);
}

#define asm_min(as, ir)		asm_min_max(as, ir, CC_GT, CC_PL)
#define asm_max(as, ir)		asm_min_max(as, ir, CC_LT, CC_LE)

/* -- Comparisons --------------------------------------------------------- */

/* Map of comparisons to flags. ORDER IR. */
static const uint8_t asm_compmap[IR_ABC+1] = {
  /* op  FP swp  int cc   FP cc */
  /* LT       */ CC_GE + (CC_HS << 4),
  /* GE    x  */ CC_LT + (CC_HI << 4),
  /* LE       */ CC_GT + (CC_HI << 4),
  /* GT    x  */ CC_LE + (CC_HS << 4),
  /* ULT   x  */ CC_HS + (CC_LS << 4),
  /* UGE      */ CC_LO + (CC_LO << 4),
  /* ULE   x  */ CC_HI + (CC_LO << 4),
  /* UGT      */ CC_LS + (CC_LS << 4),
  /* EQ       */ CC_NE + (CC_NE << 4),
  /* NE       */ CC_EQ + (CC_EQ << 4),
  /* ABC      */ CC_LS + (CC_LS << 4)  /* Same as UGT. */
};

#if LJ_SOFTFP
/* FP comparisons. */
static void asm_sfpcomp(ASMState *as, IRIns *ir)
{
  const CCallInfo *ci = &lj_ir_callinfo[IRCALL_softfp_cmp];
  RegSet drop = RSET_SCRATCH;
  Reg r;
  IRRef args[4];
  int swp = (((ir->o ^ (ir->o >> 2)) & ~(ir->o >> 3) & 1) << 1);
  args[swp^0] = ir->op1; args[swp^1] = (ir+1)->op1;
  args[swp^2] = ir->op2; args[swp^3] = (ir+1)->op2;
  /* __aeabi_cdcmple preserves r0-r3. This helps to reduce spills. */
  for (r = RID_R0; r <= RID_R3; r++)
    if (!rset_test(as->freeset, r) &&
	regcost_ref(as->cost[r]) == args[r-RID_R0]) rset_clear(drop, r);
  ra_evictset(as, drop);
  asm_guardcc(as, (asm_compmap[ir->o] >> 4));
  emit_call(as, (void *)ci->func);
  for (r = RID_R0; r <= RID_R3; r++)
    ra_leftov(as, r, args[r-RID_R0]);
}
#else
/* FP comparisons. */
static void asm_fpcomp(ASMState *as, IRIns *ir)
{
  Reg left, right;
  ARMIns ai;
  int swp = ((ir->o ^ (ir->o >> 2)) & ~(ir->o >> 3) & 1);
  if (!swp && irref_isk(ir->op2) && ir_knum(IR(ir->op2))->u64 == 0) {
    left = (ra_alloc1(as, ir->op1, RSET_FPR) & 15);
    right = 0;
    ai = ARMI_VCMPZ_D;
  } else {
    left = ra_alloc2(as, ir, RSET_FPR);
    if (swp) {
      right = (left & 15); left = ((left >> 8) & 15);
    } else {
      right = ((left >> 8) & 15); left &= 15;
    }
    ai = ARMI_VCMP_D;
  }
  asm_guardcc(as, (asm_compmap[ir->o] >> 4));
  emit_d(as, ARMI_VMRS, 0);
  emit_dm(as, ai, left, right);
}
#endif

/* Integer comparisons. */
static void asm_intcomp(ASMState *as, IRIns *ir)
{
  ARMCC cc = (asm_compmap[ir->o] & 15);
  IRRef lref = ir->op1, rref = ir->op2;
  Reg left;
  uint32_t m;
  int cmpprev0 = 0;
  lj_assertA(irt_isint(ir->t) || irt_isu32(ir->t) || irt_isaddr(ir->t),
	     "bad comparison data type %d", irt_type(ir->t));
  if (asm_swapops(as, lref, rref)) {
    Reg tmp = lref; lref = rref; rref = tmp;
    if (cc >= CC_GE) cc ^= 7;  /* LT <-> GT, LE <-> GE */
    else if (cc > CC_NE) cc ^= 11;  /* LO <-> HI, LS <-> HS */
  }
  if (irref_isk(rref) && IR(rref)->i == 0) {
    IRIns *irl = IR(lref);
    cmpprev0 = (irl+1 == ir);
    /* Combine comp(BAND(left, right), 0) into tst left, right. */
    if (cmpprev0 && irl->o == IR_BAND && !ra_used(irl)) {
      IRRef blref = irl->op1, brref = irl->op2;
      uint32_t m2 = 0;
      Reg bleft;
      if (asm_swapops(as, blref, brref)) {
	Reg tmp = blref; blref = brref; brref = tmp;
      }
      if (irref_isk(brref)) {
	m2 = emit_isk12(ARMI_AND, IR(brref)->i);
	if ((m2 & (ARMI_AND^ARMI_BIC)))
	  goto notst;  /* Not beneficial if we miss a constant operand. */
      }
      if (cc == CC_GE) cc = CC_PL;
      else if (cc == CC_LT) cc = CC_MI;
      else if (cc > CC_NE) goto notst;  /* Other conds don't work with tst. */
      bleft = ra_alloc1(as, blref, RSET_GPR);
      if (!m2) m2 = asm_fuseopm(as, 0, brref, rset_exclude(RSET_GPR, bleft));
      asm_guardcc(as, cc);
      emit_n(as, ARMI_TST^m2, bleft);
      return;
    }
  }
notst:
  left = ra_alloc1(as, lref, RSET_GPR);
  m = asm_fuseopm(as, ARMI_CMP, rref, rset_exclude(RSET_GPR, left));
  asm_guardcc(as, cc);
  emit_n(as, ARMI_CMP^m, left);
  /* Signed comparison with zero and referencing previous ins? */
  if (cmpprev0 && (cc <= CC_NE || cc >= CC_GE))
    as->flagmcp = as->mcp;  /* Allow elimination of the compare. */
}

static void asm_comp(ASMState *as, IRIns *ir)
{
#if !LJ_SOFTFP
  if (irt_isnum(ir->t))
    asm_fpcomp(as, ir);
  else
#endif
    asm_intcomp(as, ir);
}

#define asm_equal(as, ir)	asm_comp(as, ir)

#if LJ_HASFFI
/* 64 bit integer comparisons. */
static void asm_int64comp(ASMState *as, IRIns *ir)
{
  int signedcomp = (ir->o <= IR_GT);
  ARMCC cclo, cchi;
  Reg leftlo, lefthi;
  uint32_t mlo, mhi;
  RegSet allow = RSET_GPR, oldfree;

  /* Always use unsigned comparison for loword. */
  cclo = asm_compmap[ir->o + (signedcomp ? 4 : 0)] & 15;
  leftlo = ra_alloc1(as, ir->op1, allow);
  oldfree = as->freeset;
  mlo = asm_fuseopm(as, ARMI_CMP, ir->op2, rset_clear(allow, leftlo));
  allow &= ~(oldfree & ~as->freeset);  /* Update for allocs of asm_fuseopm. */

  /* Use signed or unsigned comparison for hiword. */
  cchi = asm_compmap[ir->o] & 15;
  lefthi = ra_alloc1(as, (ir+1)->op1, allow);
  mhi = asm_fuseopm(as, ARMI_CMP, (ir+1)->op2, rset_clear(allow, lefthi));

  /* All register allocations must be performed _before_ this point. */
  if (signedcomp) {
    MCLabel l_around = emit_label(as);
    asm_guardcc(as, cclo);
    emit_n(as, ARMI_CMP^mlo, leftlo);
    emit_branch(as, ARMF_CC(ARMI_B, CC_NE), l_around);
    if (cchi == CC_GE || cchi == CC_LE) cchi ^= 6;  /* GE -> GT, LE -> LT */
    asm_guardcc(as, cchi);
  } else {
    asm_guardcc(as, cclo);
    emit_n(as, ARMF_CC(ARMI_CMP, CC_EQ)^mlo, leftlo);
  }
  emit_n(as, ARMI_CMP^mhi, lefthi);
}
#endif

/* -- Split register ops -------------------------------------------------- */

/* Hiword op of a split 32/32 bit op. Previous op is the loword op. */
static void asm_hiop(ASMState *as, IRIns *ir)
{
  /* HIOP is marked as a store because it needs its own DCE logic. */
  int uselo = ra_used(ir-1), usehi = ra_used(ir);  /* Loword/hiword used? */
  if (LJ_UNLIKELY(!(as->flags & JIT_F_OPT_DCE))) uselo = usehi = 1;
#if LJ_HASFFI || LJ_SOFTFP
  if ((ir-1)->o <= IR_NE) {  /* 64 bit integer or FP comparisons. ORDER IR. */
    as->curins--;  /* Always skip the loword comparison. */
#if LJ_SOFTFP
    if (!irt_isint(ir->t)) {
      asm_sfpcomp(as, ir-1);
      return;
    }
#endif
#if LJ_HASFFI
    asm_int64comp(as, ir-1);
#endif
    return;
#if LJ_SOFTFP
  } else if ((ir-1)->o == IR_MIN || (ir-1)->o == IR_MAX) {
    as->curins--;  /* Always skip the loword min/max. */
    if (uselo || usehi)
      asm_sfpmin_max(as, ir-1, (ir-1)->o == IR_MIN ? CC_PL : CC_LE);
    return;
#elif LJ_HASFFI
  } else if ((ir-1)->o == IR_CONV) {
    as->curins--;  /* Always skip the CONV. */
    if (usehi || uselo)
      asm_conv64(as, ir);
    return;
#endif
  } else if ((ir-1)->o == IR_XSTORE) {
    if ((ir-1)->r != RID_SINK)
      asm_xstore_(as, ir, 4);
    return;
  }
#endif
  if (!usehi) return;  /* Skip unused hiword op for all remaining ops. */
  switch ((ir-1)->o) {
#if LJ_HASFFI
  case IR_ADD:
    as->curins--;
    asm_intop(as, ir, ARMI_ADC);
    asm_intop(as, ir-1, ARMI_ADD|ARMI_S);
    break;
  case IR_SUB:
    as->curins--;
    asm_intop(as, ir, ARMI_SBC);
    asm_intop(as, ir-1, ARMI_SUB|ARMI_S);
    break;
  case IR_NEG:
    as->curins--;
    asm_intneg(as, ir, ARMI_RSC);
    asm_intneg(as, ir-1, ARMI_RSB|ARMI_S);
    break;
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
  emit_n(as, ARMI_TST|ARMI_K12|HOOK_PROFILE, RID_TMP);
  emit_lsptr(as, ARMI_LDRB, RID_TMP, (void *)&J2G(as->J)->hookmask);
}

/* -- Stack handling ------------------------------------------------------ */

/* Check Lua stack size for overflow. Use exit handler as fallback. */
static void asm_stack_check(ASMState *as, BCReg topslot,
			    IRIns *irp, RegSet allow, ExitNo exitno)
{
  int savereg = 0;
  Reg pbase;
  uint32_t k;
  if (irp) {
    if (!ra_hasspill(irp->s)) {
      pbase = irp->r;
      lj_assertA(ra_hasreg(pbase), "base reg lost");
    } else if (allow) {
      pbase = rset_pickbot(allow);
    } else {
      pbase = RID_RET;
      savereg = 1;
    }
  } else {
    pbase = RID_BASE;
  }
  emit_branch(as, ARMF_CC(ARMI_BL, CC_LS), exitstub_addr(as->J, exitno));
  if (savereg)
    emit_lso(as, ARMI_LDR, RID_RET, RID_SP, 0);  /* Restore temp. register. */
  k = emit_isk12(0, (int32_t)(8*topslot));
  lj_assertA(k, "slot offset %d does not fit in K12", 8*topslot);
  emit_n(as, ARMI_CMP^k, RID_TMP);
  emit_dnm(as, ARMI_SUB, RID_TMP, RID_TMP, pbase);
  emit_lso(as, ARMI_LDR, RID_TMP, RID_TMP,
	   (int32_t)offsetof(lua_State, maxstack));
  if (irp) {  /* Must not spill arbitrary registers in head of side trace. */
    int32_t i = i32ptr(&J2G(as->J)->cur_L);
    if (ra_hasspill(irp->s))
      emit_lso(as, ARMI_LDR, pbase, RID_SP, sps_scale(irp->s));
    emit_lso(as, ARMI_LDR, RID_TMP, RID_TMP, (i & 4095));
    if (savereg)
      emit_lso(as, ARMI_STR, RID_RET, RID_SP, 0);  /* Save temp. register. */
    emit_loadi(as, RID_TMP, (i & ~4095));
  } else {
    emit_getgl(as, RID_TMP, cur_L);
  }
}

/* Restore Lua stack from on-trace state. */
static void asm_stack_restore(ASMState *as, SnapShot *snap)
{
  SnapEntry *map = &as->T->snapmap[snap->mapofs];
  SnapEntry *flinks = &as->T->snapmap[snap_nextofs(as->T, snap)-1];
  MSize n, nent = snap->nent;
  int32_t bias = 0;
  /* Store the value of all modified slots to the Lua stack. */
  for (n = 0; n < nent; n++) {
    SnapEntry sn = map[n];
    BCReg s = snap_slot(sn);
    int32_t ofs = 8*((int32_t)s-1) - bias;
    IRRef ref = snap_ref(sn);
    IRIns *ir = IR(ref);
    if ((sn & SNAP_NORESTORE))
      continue;
    if (irt_isnum(ir->t)) {
#if LJ_SOFTFP
      RegSet odd = rset_exclude(RSET_GPRODD, RID_BASE);
      Reg tmp;
      /* LJ_SOFTFP: must be a number constant. */
      lj_assertA(irref_isk(ref), "unsplit FP op");
      tmp = ra_allock(as, (int32_t)ir_knum(ir)->u32.lo,
		      rset_exclude(RSET_GPREVEN, RID_BASE));
      emit_lso(as, ARMI_STR, tmp, RID_BASE, ofs);
      if (rset_test(as->freeset, tmp+1)) odd = RID2RSET(tmp+1);
      tmp = ra_allock(as, (int32_t)ir_knum(ir)->u32.hi, odd);
      emit_lso(as, ARMI_STR, tmp, RID_BASE, ofs+4);
#else
      Reg src = ra_alloc1(as, ref, RSET_FPR);
      if (LJ_UNLIKELY(ofs < -1020 || ofs > 1020)) {
	int32_t adj = ofs & 0xffffff00;  /* K12-friendly. */
	bias += adj;
	ofs -= adj;
	emit_addptr(as, RID_BASE, -adj);
      }
      emit_vlso(as, ARMI_VSTR_D, src, RID_BASE, ofs);
#endif
    } else {
      RegSet odd = rset_exclude(RSET_GPRODD, RID_BASE);
      Reg type;
      lj_assertA(irt_ispri(ir->t) || irt_isaddr(ir->t) || irt_isinteger(ir->t),
		 "restore of IR type %d", irt_type(ir->t));
      if (!irt_ispri(ir->t)) {
	Reg src = ra_alloc1(as, ref, rset_exclude(RSET_GPREVEN, RID_BASE));
	emit_lso(as, ARMI_STR, src, RID_BASE, ofs);
	if (rset_test(as->freeset, src+1)) odd = RID2RSET(src+1);
      }
      if ((sn & (SNAP_CONT|SNAP_FRAME))) {
	if (s == 0) continue;  /* Do not overwrite link to previous frame. */
	type = ra_allock(as, (int32_t)(*flinks--), odd);
#if LJ_SOFTFP
      } else if ((sn & SNAP_SOFTFPNUM)) {
	type = ra_alloc1(as, ref+1, rset_exclude(RSET_GPRODD, RID_BASE));
#endif
      } else if ((sn & SNAP_KEYINDEX)) {
	type = ra_allock(as, (int32_t)LJ_KEYINDEX, odd);
      } else {
	type = ra_allock(as, (int32_t)irt_toitype(ir->t), odd);
      }
      emit_lso(as, ARMI_STR, type, RID_BASE, ofs+4);
    }
    checkmclim(as);
  }
  emit_addptr(as, RID_BASE, bias);
  lj_assertA(map + nent == flinks, "inconsistent frames in snapshot");
}

/* -- GC handling --------------------------------------------------------- */

/* Marker to prevent patching the GC check exit. */
#define ARM_NOPATCH_GC_CHECK	(ARMI_BIC|ARMI_K12)

/* Check GC threshold and do one or more GC steps. */
static void asm_gc_check(ASMState *as)
{
  const CCallInfo *ci = &lj_ir_callinfo[IRCALL_lj_gc_step_jit];
  IRRef args[2];
  MCLabel l_end;
  Reg tmp1, tmp2;
  ra_evictset(as, RSET_SCRATCH);
  l_end = emit_label(as);
  /* Exit trace if in GCSatomic or GCSfinalize. Avoids syncing GC objects. */
  asm_guardcc(as, CC_NE);  /* Assumes asm_snap_prep() already done. */
  *--as->mcp = ARM_NOPATCH_GC_CHECK;
  emit_n(as, ARMI_CMP|ARMI_K12|0, RID_RET);
  args[0] = ASMREF_TMP1;  /* global_State *g */
  args[1] = ASMREF_TMP2;  /* MSize steps     */
  asm_gencall(as, ci, args);
  tmp1 = ra_releasetmp(as, ASMREF_TMP1);
  tmp2 = ra_releasetmp(as, ASMREF_TMP2);
  emit_loadi(as, tmp2, as->gcsteps);
  /* Jump around GC step if GC total < GC threshold. */
  emit_branch(as, ARMF_CC(ARMI_B, CC_LS), l_end);
  emit_nm(as, ARMI_CMP, RID_TMP, tmp2);
  emit_lso(as, ARMI_LDR, tmp2, tmp1,
	   (int32_t)offsetof(global_State, gc.threshold));
  emit_lso(as, ARMI_LDR, RID_TMP, tmp1,
	   (int32_t)offsetof(global_State, gc.total));
  ra_allockreg(as, i32ptr(J2G(as->J)), tmp1);
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
    /* asm_guardcc already inverted the bcc and patched the final bl. */
    p[-2] |= ((uint32_t)(target-p) & 0x00ffffffu);
  } else {
    p[-1] = ARMI_B | ((uint32_t)((target-p)-1) & 0x00ffffffu);
  }
}

/* Fixup the tail of the loop. */
static void asm_loop_tail_fixup(ASMState *as)
{
  UNUSED(as);  /* Nothing to do. */
}

/* -- Head of trace ------------------------------------------------------- */

/* Reload L register from g->cur_L. */
static void asm_head_lreg(ASMState *as)
{
  IRIns *ir = IR(ASMREF_L);
  if (ra_used(ir)) {
    Reg r = ra_dest(as, ir, RSET_GPR);
    emit_getgl(as, r, cur_L);
    ra_evictk(as);
  }
}

/* Coalesce BASE register for a root trace. */
static void asm_head_root_base(ASMState *as)
{
  IRIns *ir;
  asm_head_lreg(as);
  ir = IR(REF_BASE);
  if (ra_hasreg(ir->r) && (rset_test(as->modset, ir->r) || irt_ismarked(ir->t)))
    ra_spill(as, ir);
  ra_destreg(as, ir, RID_BASE);
}

/* Coalesce BASE register for a side trace. */
static Reg asm_head_side_base(ASMState *as, IRIns *irp)
{
  IRIns *ir;
  asm_head_lreg(as);
  ir = IR(REF_BASE);
  if (ra_hasreg(ir->r) && (rset_test(as->modset, ir->r) || irt_ismarked(ir->t)))
    ra_spill(as, ir);
  if (ra_hasspill(irp->s)) {
    return ra_dest(as, ir, RSET_GPR);
  } else {
    Reg r = irp->r;
    lj_assertA(ra_hasreg(r), "base reg lost");
    if (r != ir->r && !rset_test(as->freeset, r))
      ra_restore(as, regcost_ref(as->cost[r]));
    ra_destreg(as, ir, r);
    return r;
  }
}

/* -- Tail of trace ------------------------------------------------------- */

/* Fixup the tail code. */
static void asm_tail_fixup(ASMState *as, TraceNo lnk)
{
  MCode *p = as->mctop;
  MCode *target;
  int32_t spadj = as->T->spadjust;
  if (spadj == 0) {
    as->mctop = --p;
  } else {
    /* Patch stack adjustment. */
    uint32_t k = emit_isk12(ARMI_ADD, spadj);
    lj_assertA(k, "stack adjustment %d does not fit in K12", spadj);
    p[-2] = (ARMI_ADD^k) | ARMF_D(RID_SP) | ARMF_N(RID_SP);
  }
  /* Patch exit branch. */
  target = lnk ? traceref(as->J, lnk)->mcode : (MCode *)lj_vm_exit_interp;
  p[-1] = ARMI_B|(((target-p)-1)&0x00ffffffu);
}

/* Prepare tail of code. */
static void asm_tail_prep(ASMState *as)
{
  MCode *p = as->mctop - 1;  /* Leave room for exit branch. */
  if (as->loopref) {
    as->invmcp = as->mcp = p;
  } else {
    as->mcp = p-1;  /* Leave room for stack pointer adjustment. */
    as->invmcp = NULL;
  }
  *p = 0;  /* Prevent load/store merging. */
}

/* -- Trace setup --------------------------------------------------------- */

/* Ensure there are enough stack slots for call arguments. */
static Reg asm_setup_call_slots(ASMState *as, IRIns *ir, const CCallInfo *ci)
{
  IRRef args[CCI_NARGS_MAX*2];
  uint32_t i, nargs = CCI_XNARGS(ci);
  int nslots = 0, ngpr = REGARG_NUMGPR, nfpr = REGARG_NUMFPR, fprodd = 0;
  asm_collectargs(as, ir, ci, args);
  for (i = 0; i < nargs; i++) {
    if (!LJ_SOFTFP && args[i] && irt_isfp(IR(args[i])->t)) {
      if (!LJ_ABI_SOFTFP && !(ci->flags & CCI_VARARG)) {
	if (irt_isnum(IR(args[i])->t)) {
	  if (nfpr > 0) nfpr--;
	  else fprodd = 0, nslots = (nslots + 3) & ~1;
	} else {
	  if (fprodd) fprodd--;
	  else if (nfpr > 0) fprodd = 1, nfpr--;
	  else nslots++;
	}
      } else if (irt_isnum(IR(args[i])->t)) {
	ngpr &= ~1;
	if (ngpr > 0) ngpr -= 2; else nslots += 2;
      } else {
	if (ngpr > 0) ngpr--; else nslots++;
      }
    } else {
      if (ngpr > 0) ngpr--; else nslots++;
    }
  }
  if (nslots > as->evenspill)  /* Leave room for args in stack slots. */
    as->evenspill = nslots;
  return REGSP_HINT(irt_isfp(ir->t) ? RID_FPRET : RID_RET);
}

static void asm_setup_target(ASMState *as)
{
  /* May need extra exit for asm_stack_check on side traces. */
  asm_exitstub_setup(as, as->T->nsnap + (as->parent ? 1 : 0));
}

/* -- Trace patching ------------------------------------------------------ */

/* Patch exit jumps of existing machine code to a new target. */
void lj_asm_patchexit(jit_State *J, GCtrace *T, ExitNo exitno, MCode *target)
{
  MCode *p = T->mcode;
  MCode *pe = (MCode *)((char *)p + T->szmcode);
  MCode *cstart = NULL, *cend = p;
  MCode *mcarea = lj_mcode_patch(J, p, 0);
  MCode *px = exitstub_addr(J, exitno) - 2;
  for (; p < pe; p++) {
    /* Look for bl_cc exitstub, replace with b_cc target. */
    uint32_t ins = *p;
    if ((ins & 0x0f000000u) == 0x0b000000u && ins < 0xf0000000u &&
	((ins ^ (px-p)) & 0x00ffffffu) == 0 &&
	p[-1] != ARM_NOPATCH_GC_CHECK) {
      *p = (ins & 0xfe000000u) | (((target-p)-2) & 0x00ffffffu);
      cend = p+1;
      if (!cstart) cstart = p;
    }
  }
  lj_assertJ(cstart != NULL, "exit stub %d not found", exitno);
  lj_mcode_sync(cstart, cend);
  lj_mcode_patch(J, mcarea, 1);
}

