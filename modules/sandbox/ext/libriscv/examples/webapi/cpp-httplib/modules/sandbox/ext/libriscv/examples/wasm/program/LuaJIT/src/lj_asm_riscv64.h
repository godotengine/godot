/*
** RISC-V IR assembler (SSA IR -> machine code).
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
**
** Contributed by gns from PLCT Lab, ISCAS.
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

/* Allocate a register or RID_ZERO. */
static Reg ra_alloc1z(ASMState *as, IRRef ref, RegSet allow)
{
  Reg r = IR(ref)->r;
  if (ra_noreg(r)) {
    if (!(allow & RSET_FPR) && irref_isk(ref) && get_kval(as, ref) == 0)
      return RID_ZERO;
    r = ra_allocref(as, ref, allow);
  } else {
    ra_noweak(as, r);
  }
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
      right = ra_alloc1z(as, ir->op2, rset_exclude(allow, left));
    else
      ra_noweak(as, right);
  } else if (ra_hasreg(right)) {
    ra_noweak(as, right);
    left = ra_alloc1z(as, ir->op1, rset_exclude(allow, right));
  } else if (ra_hashint(right)) {
    right = ra_alloc1z(as, ir->op2, allow);
    left = ra_alloc1z(as, ir->op1, rset_exclude(allow, right));
  } else {
    left = ra_alloc1z(as, ir->op1, allow);
    right = ra_alloc1z(as, ir->op2, rset_exclude(allow, left));
  }
  return left | (right << 8);
}

/* -- Guard handling ------------------------------------------------------ */

/* Copied from MIPS, AUIPC+JALR is expensive to setup in-place */
#define RISCV_SPAREJUMP		4

/* Setup spare long-range jump (trampoline?) slots per mcarea. */

static void asm_sparejump_setup(ASMState *as)
{
  MCode *mxp = as->mctop;
  if ((char *)mxp == (char *)as->J->mcarea + as->J->szmcarea) {
    for (int i = RISCV_SPAREJUMP*2; i--; )
      *--mxp = RISCVI_EBREAK;
    as->mctop = mxp;
  }
}

static MCode *asm_sparejump_use(MCode *mcarea, MCode *target)
{
  MCode *mxp = (MCode *)((char *)mcarea + ((MCLink *)mcarea)->size);
  int slot = RISCV_SPAREJUMP;
  RISCVIns tslot = RISCVI_EBREAK, tauipc, tjalr;
  while (slot--) {
    mxp -= 2;
    ptrdiff_t delta = (char *)target - (char *)mxp;
    tauipc = RISCVI_AUIPC | RISCVF_D(RID_TMP) | RISCVF_IMMU(RISCVF_HI(delta)),
    tjalr = RISCVI_JALR | RISCVF_S1(RID_TMP) | RISCVF_IMMI(RISCVF_LO(delta));
    if (mxp[0] == tauipc && mxp[1] == tjalr) {
      return mxp;
    } else if (mxp[0] == tslot) {
      mxp[0] = tauipc, mxp[1] = tjalr;
      return mxp;
    }
  }
  return NULL;
}

/* Setup exit stub after the end of each trace. */
static void asm_exitstub_setup(ASMState *as, ExitNo nexits)
{
  ExitNo i;
  MCode *mxp = as->mctop;
  if (mxp - (nexits + 4 + MCLIM_REDZONE) < as->mclim)
    asm_mclimit(as);
  for (i = nexits-1; (int32_t)i >= 0; i--)
    *--mxp = RISCVI_JAL | RISCVF_D(RID_RA) | RISCVF_IMMJ((uintptr_t)(4*(-4-i)));
  ptrdiff_t delta = (char *)lj_vm_exit_handler - (char *)(mxp-3);
  /* 1: sw ra, 0(sp); auipc+jalr ->vm_exit_handler; lui x0, traceno; jal <1; jal <1; ... */
  *--mxp = RISCVI_LUI | RISCVF_IMMU(as->T->traceno);
  *--mxp = RISCVI_JALR | RISCVF_D(RID_RA) | RISCVF_S1(RID_TMP)
         | RISCVF_IMMI(RISCVF_LO((uintptr_t)(void *)delta));
  *--mxp = RISCVI_AUIPC | RISCVF_D(RID_TMP)
         | RISCVF_IMMU(RISCVF_HI((uintptr_t)(void *)delta));
  *--mxp = RISCVI_SD | RISCVF_S2(RID_RA) | RISCVF_S1(RID_SP);
  as->mctop = mxp;
}

static MCode *asm_exitstub_addr(ASMState *as, ExitNo exitno)
{
  /* Keep this in-sync with exitstub_trace_addr(). */
  return as->mctop + exitno + 4;
}

/* Emit conditional branch to exit for guard. */
static void asm_guard(ASMState *as, RISCVIns riscvi, Reg rs1, Reg rs2)
{
  MCode *target = asm_exitstub_addr(as, as->snapno);
  MCode *p = as->mcp;
  if (LJ_UNLIKELY(p == as->invmcp)) {
    as->loopinv = 1;
    as->mcp = ++p;
    *p = RISCVI_JAL | RISCVF_IMMJ((char *)target - (char *)p);
    riscvi = riscvi^RISCVF_FUNCT3(1);  /* Invert cond. */
    target = p - 1;  /* Patch target later in asm_loop_fixup. */
  }
    ptrdiff_t delta = (char *)target - (char *)(p - 1);
    *--p = RISCVI_JAL | RISCVF_IMMJ(delta);
    *--p = (riscvi^RISCVF_FUNCT3(1)) | RISCVF_S1(rs1) | RISCVF_S2(rs2) | RISCVF_IMMB(8);
    as->mcp = p;
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
	  if (checki12(ofs)) {
	    *ofsp = ofs;
	    return ra_alloc1(as, refa, allow);
	  }
	}
      }
    } else if (ir->o == IR_HREFK) {
      if (mayfuse(as, ref)) {
	int32_t ofs = (int32_t)(IR(ir->op2)->op2 * sizeof(Node));
	if (checki12(ofs)) {
	  *ofsp = ofs;
	  return ra_alloc1(as, ir->op1, allow);
	}
      }
    } else if (ir->o == IR_UREFC) {
      if (irref_isk(ir->op1)) {
	GCfunc *fn = ir_kfunc(IR(ir->op1));
	GCupval *uv = &gcref(fn->l.uvptr[(ir->op2 >> 8)])->uv;
  intptr_t ofs = ((intptr_t)((uintptr_t)(&uv->tv) - (uintptr_t)&J2GG(as->J)->g));
	if (checki12(ofs)) {
	  *ofsp = (int32_t)ofs;
	  return RID_GL;
	}
      }
    } else if (ir->o == IR_TMPREF) {
      *ofsp = (int32_t)offsetof(global_State, tmptv);
      return RID_GL;
    }
  }
  *ofsp = 0;
  return ra_alloc1(as, ref, allow);
}

/* Fuse XLOAD/XSTORE reference into load/store operand. */
static void asm_fusexref(ASMState *as, RISCVIns riscvi, Reg rd, IRRef ref,
			 RegSet allow, int32_t ofs)
{
  IRIns *ir = IR(ref);
  Reg base;
  if (ra_noreg(ir->r) && canfuse(as, ir)) {
    intptr_t ofs2;
    if (ir->o == IR_ADD) {
      if (irref_isk(ir->op2) && (ofs2 = ofs + get_kval(as, ir->op2),
				 checki12(ofs2))) {
	ref = ir->op1;
	ofs = (int32_t)ofs2;
      }
    } else if (ir->o == IR_STRREF) {
      ofs2 = 4096;
      lj_assertA(ofs == 0, "bad usage");
      ofs = (int32_t)sizeof(GCstr);
      if (irref_isk(ir->op2)) {
	ofs2 = ofs + get_kval(as, ir->op2);
	ref = ir->op1;
      } else if (irref_isk(ir->op1)) {
	ofs2 = ofs + get_kval(as, ir->op1);
	ref = ir->op2;
      }
      if (!checki12(ofs2)) {
        /* NYI: Fuse ADD with constant. */
        Reg right, left = ra_alloc2(as, ir, allow);
        right = (left >> 8); left &= 255;
        emit_lso(as, riscvi, rd, RID_TMP, ofs);
        emit_ds1s2(as, RISCVI_ADD, RID_TMP, left, right);
        return;
      }
      ofs = ofs2;
    }
  }
  base = ra_alloc1(as, ref, allow);
  emit_lso(as, riscvi, rd, base, ofs);
}

/* Fuse Integer multiply-accumulate. */

static int asm_fusemac(ASMState *as, IRIns *ir, RISCVIns riscvi)
{
  IRRef lref = ir->op1, rref = ir->op2;
  IRIns *irm;
  if (lref != rref &&
      ((mayfuse(as, lref) && (irm = IR(lref), irm->o == IR_MUL) &&
       ra_noreg(irm->r)) ||
       (mayfuse(as, rref) && (irm = IR(rref), irm->o == IR_MUL) &&
       (rref = lref, ra_noreg(irm->r))))) {
    Reg dest = ra_dest(as, ir, RSET_GPR);
    Reg add = ra_hintalloc(as, rref, dest, RSET_GPR);
    Reg left = ra_alloc2(as, irm,
       rset_exclude(rset_exclude(RSET_GPR, dest), add));
    Reg right = (left >> 8); left &= 255;
    emit_ds1s2(as, riscvi, dest, left, right);
    if (dest != add) emit_mv(as, dest, add);
    return 1;
  }
  return 0;
}

/* Fuse FP multiply-add/sub. */

static int asm_fusemadd(ASMState *as, IRIns *ir, RISCVIns riscvi, RISCVIns riscvir)
{
  IRRef lref = ir->op1, rref = ir->op2;
  IRIns *irm;
  if ((as->flags & JIT_F_OPT_FMA) &&
      lref != rref &&
      ((mayfuse(as, lref) && (irm = IR(lref), irm->o == IR_MUL) &&
       ra_noreg(irm->r)) ||
       (mayfuse(as, rref) && (irm = IR(rref), irm->o == IR_MUL) &&
       (rref = lref, riscvi = riscvir, ra_noreg(irm->r))))) {
    Reg dest = ra_dest(as, ir, RSET_FPR);
    Reg add = ra_hintalloc(as, rref, dest, RSET_FPR);
    Reg left = ra_alloc2(as, irm,
       rset_exclude(rset_exclude(RSET_FPR, dest), add));
    Reg right = (left >> 8); left &= 255;
    emit_ds1s2s3(as, riscvi, dest, left, right, add);
    return 1;
  }
  return 0;
}
/* -- Calls --------------------------------------------------------------- */

/* Generate a call to a C function. */
static void asm_gencall(ASMState *as, const CCallInfo *ci, IRRef *args)
{
  uint32_t n, nargs = CCI_XNARGS(ci);
  int32_t ofs = 0;
  Reg gpr, fpr = REGARG_FIRSTFPR;
  if ((void *)ci->func)
    emit_call(as, (void *)ci->func, 1);
  for (gpr = REGARG_FIRSTGPR; gpr <= REGARG_LASTGPR; gpr++)
    as->cost[gpr] = REGCOST(~0u, ASMREF_L);
  gpr = REGARG_FIRSTGPR;
  for (n = 0; n < nargs; n++) { /* Setup args. */
    IRRef ref = args[n];
    IRIns *ir = IR(ref);
    if (ref) {
      if (irt_isfp(ir->t)) {
        if (fpr <= REGARG_LASTFPR) {
	  lj_assertA(rset_test(as->freeset, fpr),
	             "reg %d not free", fpr);  /* Must have been evicted. */
          ra_leftov(as, fpr, ref);
	  fpr++; if(ci->flags & CCI_VARARG) gpr++;
	} else if (!(ci->flags & CCI_VARARG) && gpr <= REGARG_LASTGPR) {
	  lj_assertA(rset_test(as->freeset, gpr),
	             "reg %d not free", gpr);  /* Must have been evicted. */
          ra_leftov(as, gpr, ref);
	  gpr++;
	} else {
	  Reg r = ra_alloc1(as, ref, RSET_FPR);
	  emit_spstore(as, ir, r, ofs);
	  ofs += 8;
	}
      } else {
        if (gpr <= REGARG_LASTGPR) {
	  lj_assertA(rset_test(as->freeset, gpr),
	             "reg %d not free", gpr);  /* Must have been evicted. */
          ra_leftov(as, gpr, ref);
	  gpr++; if(ci->flags & CCI_VARARG) fpr++;
	} else {
	  Reg r = ra_alloc1z(as, ref, RSET_GPR);
	  emit_spstore(as, ir, r, ofs);
	  ofs += 8;
	}
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
    if (irt_isfp(ir->t)) {
      if ((ci->flags & CCI_CASTU64)) {
        Reg dest = ra_dest(as, ir, RSET_FPR);
  emit_ds(as, irt_isnum(ir->t) ? RISCVI_FMV_D_X : RISCVI_FMV_W_X,
	        dest, RID_RET);
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
    ci.func = (ASMFunction)(void *)get_kval(as, func);
  } else {  /* Need specific register for indirect calls. */
    Reg r = ra_alloc1(as, func, RID2RSET(RID_CFUNCADDR));
    MCode *p = as->mcp;
    *--p = RISCVI_JALR | RISCVF_D(RID_RA) | RISCVF_S1(r);
    if (r == RID_CFUNCADDR)
      *--p = RISCVI_ADDI | RISCVF_D(RID_CFUNCADDR) | RISCVF_S1(r);
    else
      *--p = RISCVI_MV | RISCVF_D(RID_CFUNCADDR) | RISCVF_S1(r);
    as->mcp = p;
    ci.func = (ASMFunction)(void *)0;
  }
  asm_gencall(as, &ci, args);
}

static void asm_callround(ASMState *as, IRIns *ir, IRCallID id)
{
  /* The modified regs must match with the *.dasc implementation. */
  RegSet drop = RID2RSET(RID_X6)|RID2RSET(RID_X7)|RID2RSET(RID_F10)|
                RID2RSET(RID_F14)|RID2RSET(RID_F1)|RID2RSET(RID_F3)|
                RID2RSET(RID_F4);
  if (ra_hasreg(ir->r)) rset_clear(drop, ir->r);
  ra_evictset(as, drop);
  ra_destreg(as, ir, RID_FPRET);
  emit_call(as, (void *)lj_ir_callinfo[id].func, 0);
  ra_leftov(as, REGARG_FIRSTFPR, ir->op1);
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
  asm_guard(as, RISCVI_BNE, RID_TMP,
	    ra_allock(as, igcptr(pc), rset_exclude(RSET_GPR, base)));
  emit_lso(as, RISCVI_LD, RID_TMP, base, -8);
}

/* -- Buffer operations --------------------------------------------------- */

#if LJ_HASBUFFER
static void asm_bufhdr_write(ASMState *as, Reg sb)
{
  Reg tmp = ra_scratch(as, rset_exclude(RSET_GPR, sb));
  IRIns irgc;
  irgc.ot = IRT(0, IRT_PGC);  /* GC type. */
  emit_storeofs(as, &irgc, RID_TMP, sb, offsetof(SBuf, L));
  emit_ds1s2(as, RISCVI_OR, RID_TMP, RID_TMP, tmp);
  emit_dsi(as, RISCVI_ANDI, tmp, tmp, SBUF_MASK_FLAG);
  emit_getgl(as, RID_TMP, cur_L);
  emit_loadofs(as, &irgc, tmp, sb, offsetof(SBuf, L));
}
#endif

/* -- Type conversions ---------------------------------------------------- */

static void asm_tointg(ASMState *as, IRIns *ir, Reg left)
{
  Reg tmp = ra_scratch(as, rset_exclude(RSET_FPR, left));
  Reg dest = ra_dest(as, ir, RSET_GPR), cmp = ra_scratch(as, rset_exclude(RSET_GPR, dest));
  asm_guard(as, RISCVI_BEQ, cmp, RID_ZERO);
  emit_ds1s2(as, RISCVI_FEQ_D, cmp, tmp, left);
  emit_ds(as, RISCVI_FCVT_D_W, tmp, dest);
  emit_ds(as, RISCVI_FCVT_W_D, dest, left);
}

static void asm_tobit(ASMState *as, IRIns *ir)
{
  RegSet allow = RSET_FPR;
  Reg dest = ra_dest(as, ir, RSET_GPR);
  Reg left = ra_alloc1(as, ir->op1, allow);
  Reg right = ra_alloc1(as, ir->op2, rset_clear(allow, left));
  Reg tmp = ra_scratch(as, rset_clear(allow, right));
  emit_ds(as, RISCVI_FMV_X_W, dest, tmp);
  emit_ds1s2(as, RISCVI_FADD_D, tmp, left, right);
}

static void asm_conv(ASMState *as, IRIns *ir)
{
  IRType st = (IRType)(ir->op2 & IRCONV_SRCMASK);
  int st64 = (st == IRT_I64 || st == IRT_U64 || st == IRT_P64);
  int stfp = (st == IRT_NUM || st == IRT_FLOAT);
  IRRef lref = ir->op1;
  lj_assertA(irt_type(ir->t) != st, "inconsistent types for CONV");
  /* Use GPR to pass floating-point arguments */
  if (irt_isfp(ir->t) && ir->r >= RID_X10 && ir->r <= RID_X17) {
    Reg dest = ra_dest(as, ir, RSET_GPR);
    Reg ftmp = ra_scratch(as, RSET_FPR);
    if (stfp) {  /* FP to FP conversion. */
      emit_ds(as, st == IRT_NUM ? RISCVI_FMV_X_W : RISCVI_FMV_X_D, dest, ftmp);
      emit_ds(as, st == IRT_NUM ? RISCVI_FCVT_S_D : RISCVI_FCVT_D_S,
        ftmp, ra_alloc1(as, lref, RSET_FPR));
    } else {  /* Integer to FP conversion. */
      Reg left = ra_alloc1(as, lref, RSET_GPR);
      RISCVIns riscvi = irt_isfloat(ir->t) ?
  (((IRT_IS64 >> st) & 1) ?
   (st == IRT_I64 ? RISCVI_FCVT_S_L : RISCVI_FCVT_S_LU) :
   (st == IRT_INT ? RISCVI_FCVT_S_W : RISCVI_FCVT_S_WU)) :
  (((IRT_IS64 >> st) & 1) ?
   (st == IRT_I64 ? RISCVI_FCVT_D_L : RISCVI_FCVT_D_LU) :
   (st == IRT_INT ? RISCVI_FCVT_D_W : RISCVI_FCVT_D_WU));
      emit_ds(as, st64 ? RISCVI_FMV_X_D : RISCVI_FMV_X_W, dest, ftmp);
      emit_ds(as, riscvi, ftmp, left);
    }
  } else if (irt_isfp(ir->t)) {
    Reg dest = ra_dest(as, ir, RSET_FPR);
    if (stfp) {  /* FP to FP conversion. */
      emit_ds(as, st == IRT_NUM ? RISCVI_FCVT_S_D : RISCVI_FCVT_D_S,
	      dest, ra_alloc1(as, lref, RSET_FPR));
    } else {  /* Integer to FP conversion. */
      Reg left = ra_alloc1(as, lref, RSET_GPR);
      RISCVIns riscvi = irt_isfloat(ir->t) ?
  (((IRT_IS64 >> st) & 1) ?
   (st == IRT_I64 ? RISCVI_FCVT_S_L : RISCVI_FCVT_S_LU) :
   (st == IRT_INT ? RISCVI_FCVT_S_W : RISCVI_FCVT_S_WU)) :
  (((IRT_IS64 >> st) & 1) ?
   (st == IRT_I64 ? RISCVI_FCVT_D_L : RISCVI_FCVT_D_LU) :
   (st == IRT_INT ? RISCVI_FCVT_D_W : RISCVI_FCVT_D_WU));
      emit_ds(as, riscvi, dest, left);
    }
  } else if (stfp) {  /* FP to integer conversion. */
    if (irt_isguard(ir->t)) {
      /* Checked conversions are only supported from number to int. */
      lj_assertA(irt_isint(ir->t) && st == IRT_NUM,
		 "bad type for checked CONV");
      asm_tointg(as, ir, ra_alloc1(as, lref, RSET_FPR));
    } else {
      Reg left = ra_alloc1(as, lref, RSET_FPR);
      Reg dest = ra_dest(as, ir, RSET_GPR);
      RISCVIns riscvi = irt_is64(ir->t) ?
  (st == IRT_NUM ?
   (irt_isi64(ir->t) ? RISCVI_FCVT_L_D : RISCVI_FCVT_LU_D) :
   (irt_isi64(ir->t) ? RISCVI_FCVT_L_S : RISCVI_FCVT_LU_S)) :
  (st == IRT_NUM ?
   (irt_isint(ir->t) ? RISCVI_FCVT_W_D : RISCVI_FCVT_WU_D) :
   (irt_isint(ir->t) ? RISCVI_FCVT_W_S : RISCVI_FCVT_WU_S));
      emit_ds(as, riscvi|RISCVF_RM(RISCVRM_RTZ), dest, left);
    }
  } else if (st >= IRT_I8 && st <= IRT_U16) { /* Extend to 32 bit integer. */
    Reg dest = ra_dest(as, ir, RSET_GPR);
    Reg left = ra_alloc1(as, lref, RSET_GPR);
    RISCVIns riscvi = st == IRT_I8 ? RISCVI_SEXT_B :
    st == IRT_U8 ? RISCVI_ZEXT_B :
    st == IRT_I16 ? RISCVI_SEXT_H : RISCVI_ZEXT_H;
    lj_assertA(irt_isint(ir->t) || irt_isu32(ir->t), "bad type for CONV EXT");
    emit_ext(as, riscvi, dest, left);
  } else {  /* 32/64 bit integer conversions. */
    Reg dest = ra_dest(as, ir, RSET_GPR);
    if (irt_is64(ir->t)) {
	    if (st64) {
	/* 64/64 bit no-op (cast)*/
	ra_leftov(as, dest, lref);  /* Do nothing, but may need to move regs. */
      } else {  /* 32 to 64 bit sign extension. */
	Reg left = ra_alloc1(as, lref, RSET_GPR);
	  if ((ir->op2 & IRCONV_SEXT)) {  /* 32 to 64 bit sign extension. */
	    emit_ext(as, RISCVI_SEXT_W, dest, left);
	  } else {  /* 32 to 64 bit zero extension. */
	    emit_ext(as, RISCVI_ZEXT_W, dest, left);
	  }
	    }
    } else {
	    if (st64 && !(ir->op2 & IRCONV_NONE)) {
	/* This is either a 32 bit reg/reg mov which zeroes the hiword
	** or a load of the loword from a 64 bit address.
	*/
	Reg left = ra_alloc1(as, lref, RSET_GPR);
	emit_ext(as, RISCVI_ZEXT_W, dest, left);
	    } else {  /* 32/32 bit no-op (cast). */
	ra_leftov(as, dest, lref);  /* Do nothing, but may need to move regs. */
    	}
    }
  }
}

static void asm_strto(ASMState *as, IRIns *ir)
{
  const CCallInfo *ci = &lj_ir_callinfo[IRCALL_lj_strscan_num];
  IRRef args[2];
  int32_t ofs = SPOFS_TMP;
  RegSet drop = RSET_SCRATCH;
  if (ra_hasreg(ir->r)) rset_set(drop, ir->r);  /* Spill dest reg (if any). */
  ra_evictset(as, drop);
  if (ir->s) ofs = sps_scale(ir->s);
  asm_guard(as, RISCVI_BEQ, RID_RET, RID_ZERO);  /* Test return status. */
  args[0] = ir->op1;      /* GCstr *str */
  args[1] = ASMREF_TMP1;  /* TValue *n  */
  asm_gencall(as, ci, args);
  /* Store the result to the spill slot or temp slots. */
  Reg tmp = ra_releasetmp(as, ASMREF_TMP1);
  emit_opk(as, RISCVI_ADDI, tmp, RID_SP, tmp, ofs);
}

/* -- Memory references --------------------------------------------------- */

/* Store tagged value for ref at base+ofs. */
static void asm_tvstore64(ASMState *as, Reg base, int32_t ofs, IRRef ref)
{
  RegSet allow = rset_exclude(RSET_GPR, base);
  IRIns *ir = IR(ref);
  lj_assertA(irt_ispri(ir->t) || irt_isaddr(ir->t) || irt_isinteger(ir->t),
	     "store of IR type %d", irt_type(ir->t));
  if (irref_isk(ref)) {
    TValue k;
    lj_ir_kvalue(as->J->L, &k, ir);
    emit_lso(as, RISCVI_SD, ra_allock(as, (int64_t)k.u64, allow), base, ofs);
  } else {
    Reg src = ra_alloc1(as, ref, allow);
    rset_clear(allow, src);
    Reg type = ra_allock(as, (int64_t)irt_toitype(ir->t) << 47, allow);
    emit_lso(as, RISCVI_SD, RID_TMP, base, ofs);
    if (irt_isinteger(ir->t)) {
      if (as->flags & JIT_F_RVZba) {
  emit_ds1s2(as, RISCVI_ADD_UW, RID_TMP, src, type);
      } else {
  emit_ds1s2(as, RISCVI_ADD, RID_TMP, RID_TMP, type);
  emit_ext(as, RISCVI_ZEXT_W, RID_TMP, src);
      }
    } else {
      emit_ds1s2(as, RISCVI_ADD, RID_TMP, src, type);
    }
  }
}

/* Get pointer to TValue. */
static void asm_tvptr(ASMState *as, Reg dest, IRRef ref, MSize mode)	// todo-new
{
  if ((mode & IRTMPREF_IN1)) {
    IRIns *ir = IR(ref);
    if (irt_isnum(ir->t)) {
      if (irref_isk(ref) && !(mode & IRTMPREF_OUT1)) {
  /* Use the number constant itself as a TValue. */
  ra_allockreg(as, igcptr(ir_knum(ir)), dest);
  return;
      }
      emit_lso(as, RISCVI_FSD, ra_alloc1(as, ref, RSET_FPR), dest, 0);
    } else {
      asm_tvstore64(as, dest, 0, ref);
    }
  }
  /* g->tmptv holds the TValue(s). */
  emit_opk(as, RISCVI_ADDI, dest, RID_GL, dest, offsetof(global_State, tmptv));
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
    if (checki12(ofs)) {
      base = ra_alloc1(as, refa, RSET_GPR);
      emit_dsi(as, RISCVI_ADDI, dest, base, ofs);
      return;
    }
  }
  base = ra_alloc1(as, ir->op1, RSET_GPR);
  idx = ra_alloc1(as, ir->op2, rset_exclude(RSET_GPR, base));
  emit_sh3add(as, dest, base, idx, RID_TMP);
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
  Reg key = RID_NONE, type = RID_NONE, tmpnum = RID_NONE, tmp1, tmp2;
  Reg cmp64 = RID_NONE;
  IRRef refkey = ir->op2;
  IRIns *irkey = IR(refkey);
  int isk = irref_isk(refkey);
  IRType1 kt = irkey->t;
  uint32_t khash;
  MCLabel l_end, l_loop, l_next;
  rset_clear(allow, tab);
  tmp1 = ra_scratch(as, allow);
  rset_clear(allow, tmp1);
  tmp2 = ra_scratch(as, allow);
  rset_clear(allow, tmp2);

  if (irt_isnum(kt)) {
    key = ra_alloc1(as, refkey, RSET_FPR);
    tmpnum = ra_scratch(as, rset_exclude(RSET_FPR, key));
  } else {
    /* Allocate cmp64 register used for 64-bit comparisons */
    if (!isk && irt_isaddr(kt)) {
      cmp64 = tmp2;
    } else {
      int64_t k;
      if (isk && irt_isaddr(kt)) {
	k = ((int64_t)irt_toitype(kt) << 47) | irkey[1].tv.u64;
      } else {
	lj_assertA(irt_ispri(kt) && !irt_isnil(kt), "bad HREF key type");
	k = ~((int64_t)~irt_toitype(kt) << 47);
      }
      cmp64 = ra_allock(as, k, allow);
      rset_clear(allow, cmp64);
    }
    if (!irt_ispri(kt)) {
      key = ra_alloc1(as, refkey, allow);
      rset_clear(allow, key);
    }
  } 

  /* Key not found in chain: jump to exit (if merged) or load niltv. */
  l_end = emit_label(as);
  int is_lend_exit = 0;
  as->invmcp = NULL;
  if (merge == IR_NE)
    asm_guard(as, RISCVI_BEQ, RID_ZERO, RID_ZERO);
  else if (destused)
    emit_loada(as, dest, niltvg(J2G(as->J)));

  /* Follow hash chain until the end. */
  l_loop = --as->mcp;
  emit_mv(as, dest, tmp1);
  emit_lso(as, RISCVI_LD, tmp1, dest, (int32_t)offsetof(Node, next));
  l_next = emit_label(as);

  /* Type and value comparison. */
  if (merge == IR_EQ) {  /* Must match asm_guard(). */
    l_end = asm_exitstub_addr(as, as->snapno);
    is_lend_exit = 1;
  }
  if (irt_isnum(kt)) {
    emit_branch(as, RISCVI_BNE, tmp1, RID_ZERO, l_end, is_lend_exit);
    emit_ds1s2(as, RISCVI_FEQ_D, tmp1, tmpnum, key);
    emit_branch(as, RISCVI_BEQ, tmp1, RID_ZERO, l_next, 0);
    emit_dsi(as, RISCVI_SLTIU, tmp1, tmp1, ((int32_t)LJ_TISNUM));
    emit_dsshamt(as, RISCVI_SRAI, tmp1, tmp1, 47);
    emit_ds(as, RISCVI_FMV_D_X, tmpnum, tmp1);
  } else {
    emit_branch(as, RISCVI_BEQ, tmp1, cmp64, l_end, is_lend_exit);
  }
  emit_lso(as, RISCVI_LD, tmp1, dest, (int32_t)offsetof(Node, key.u64));
  *l_loop = RISCVI_BNE | RISCVF_S1(tmp1) | RISCVF_S2(RID_ZERO)
          | RISCVF_IMMB((char *)as->mcp-(char *)l_loop);
  if (!isk && irt_isaddr(kt)) {
    type = ra_allock(as, (int64_t)irt_toitype(kt) << 47, allow);
    emit_ds1s2(as, RISCVI_ADD, tmp2, key, type);
    rset_clear(allow, type);
  }

  /* Load main position relative to tab->node into dest. */
  khash = isk ? ir_khash(as, irkey) : 1;
  if (khash == 0) {
    emit_lso(as, RISCVI_LD, dest, tab, (int32_t)offsetof(GCtab, node));
  } else {
    Reg tmphash = tmp1;
    if (isk)
      tmphash = ra_allock(as, khash, allow);
    /* node = tab->node + (idx*32-idx*8) */
    emit_ds1s2(as, RISCVI_ADD, dest, dest, tmp1);
    lj_assertA(sizeof(Node) == 24, "bad Node size");
    emit_ds1s2(as, RISCVI_SUBW, tmp1, tmp2, tmp1);
    emit_dsshamt(as, RISCVI_SLLIW, tmp1, tmp1, 3);
    emit_dsshamt(as, RISCVI_SLLIW, tmp2, tmp1, 5);
    emit_ds1s2(as, RISCVI_AND, tmp1, tmp2, tmphash);	// idx = hi & tab->hmask
    emit_lso(as, RISCVI_LD, dest, tab, (int32_t)offsetof(GCtab, node));
    emit_lso(as, RISCVI_LW, tmp2, tab, (int32_t)offsetof(GCtab, hmask));
    if (isk) {
      /* Nothing to do. */
    } else if (irt_isstr(kt)) {
      emit_lso(as, RISCVI_LW, tmp1, key, (int32_t)offsetof(GCstr, sid));
    } else {  /* Must match with hash*() in lj_tab.c. */
      emit_ds1s2(as, RISCVI_SUBW, tmp1, tmp1, tmp2);
      emit_roti(as, RISCVI_RORIW, tmp2, tmp2, dest, (-HASH_ROT3)&0x1f);
      emit_ds1s2(as, RISCVI_XOR, tmp1, tmp1, tmp2);
      emit_roti(as, RISCVI_RORIW, tmp1, tmp1, dest, (-HASH_ROT2-HASH_ROT1)&0x1f);
      emit_ds1s2(as, RISCVI_SUBW, tmp2, tmp2, dest);
      emit_ds1s2(as, RISCVI_XOR, tmp2, tmp2, tmp1);
      emit_roti(as, RISCVI_RORIW, dest, tmp1, RID_TMP, (-HASH_ROT1)&0x1f);
      if (irt_isnum(kt)) {
	emit_dsshamt(as, RISCVI_SLLIW, tmp1, tmp1, 1);
	emit_dsshamt(as, RISCVI_SRAI, tmp1, tmp1, 32);	// hi
	emit_ext(as, RISCVI_SEXT_W, tmp2, tmp1);	// lo
	emit_ds(as, RISCVI_FMV_X_D, tmp1, key);
      } else {
	checkmclim(as);
	emit_dsshamt(as, RISCVI_SRAI, tmp1, tmp1, 32);	// hi
	emit_ext(as, RISCVI_SEXT_W, tmp2, key);	// lo
	emit_ds1s2(as, RISCVI_ADD, tmp1, key, type);
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
  int bigofs = !checki12(kofs);
  Reg dest = (ra_used(ir) || bigofs) ? ra_dest(as, ir, RSET_GPR) : RID_NONE;
  Reg node = ra_alloc1(as, ir->op1, RSET_GPR);
  RegSet allow = rset_exclude(RSET_GPR, node);
  Reg idx = node;
  int64_t k;
  lj_assertA(ofs % sizeof(Node) == 0, "unaligned HREFK slot");
  if (bigofs) {
    idx = dest;
    rset_clear(allow, dest);
    kofs = (int32_t)offsetof(Node, key);
  } else if (ra_hasreg(dest)) {
    emit_dsi(as, RISCVI_ADDI, dest, node, ofs);
  }
  if (irt_ispri(irkey->t)) {
    lj_assertA(!irt_isnil(irkey->t), "bad HREFK key type");
    k = ~((int64_t)~irt_toitype(irkey->t) << 47);
  } else if (irt_isnum(irkey->t)) {
    k = (int64_t)ir_knum(irkey)->u64;
  } else {
    k = ((int64_t)irt_toitype(irkey->t) << 47) | (int64_t)ir_kgc(irkey);
  }
  asm_guard(as, RISCVI_BNE, RID_TMP, ra_allock(as, k, allow));
  emit_lso(as, RISCVI_LD, RID_TMP, idx, kofs);
  if (bigofs)
    emit_ds1s2(as, RISCVI_ADD, dest, node, ra_allock(as, ofs, allow));
}

static void asm_uref(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  int guarded = (irt_t(ir->t) & (IRT_GUARD|IRT_TYPE)) == (IRT_GUARD|IRT_PGC);
  if (irref_isk(ir->op1) && !guarded) {
    GCfunc *fn = ir_kfunc(IR(ir->op1));
    MRef *v = &gcref(fn->l.uvptr[(ir->op2 >> 8)])->uv.v;
    emit_lsptr(as, RISCVI_LD, dest, v, RSET_GPR);
  } else {
    if (guarded)
      asm_guard(as, ir->o == IR_UREFC ? RISCVI_BEQ : RISCVI_BNE, RID_TMP, RID_ZERO);
    if (ir->o == IR_UREFC)
      emit_dsi(as, RISCVI_ADDI, dest, dest, (int32_t)offsetof(GCupval, tv));
    else
      emit_lso(as, RISCVI_LD, dest, dest, (int32_t)offsetof(GCupval, v));
    if (guarded)
      emit_lso(as, RISCVI_LBU, RID_TMP, dest, (int32_t)offsetof(GCupval, closed));
    if (irref_isk(ir->op1)) {
      GCfunc *fn = ir_kfunc(IR(ir->op1));
      GCobj *o = gcref(fn->l.uvptr[(ir->op2 >> 8)]);
      emit_loada(as, dest, o);
    } else {
      emit_lso(as, RISCVI_LD, dest, ra_alloc1(as, ir->op1, RSET_GPR),
         (int32_t)offsetof(GCfuncL, uvptr) +
         (int32_t)sizeof(MRef) * (int32_t)(ir->op2 >> 8));
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
  RegSet allow = RSET_GPR;
  Reg dest = ra_dest(as, ir, allow);
  Reg base = ra_alloc1(as, ir->op1, allow);
  IRIns *irr = IR(ir->op2);
  int32_t ofs = sizeof(GCstr);
  rset_clear(allow, base);
  if (irref_isk(ir->op2) && checki12(ofs + irr->i)) {
    emit_dsi(as, RISCVI_ADDI, dest, base, ofs + irr->i);
  } else {
    emit_dsi(as, RISCVI_ADDI, dest, dest, ofs);
    emit_ds1s2(as, RISCVI_ADD, dest, base, ra_alloc1(as, ir->op2, allow));
  }
}

/* -- Loads and stores ---------------------------------------------------- */

static RISCVIns asm_fxloadins(IRIns *ir)
{
  switch (irt_type(ir->t)) {
  case IRT_I8: return RISCVI_LB;
  case IRT_U8: return RISCVI_LBU;
  case IRT_I16: return RISCVI_LH;
  case IRT_U16: return RISCVI_LHU;
  case IRT_NUM: return RISCVI_FLD;
  case IRT_FLOAT: return RISCVI_FLW;
  default: return irt_is64(ir->t) ? RISCVI_LD : RISCVI_LW;
  }
}

static RISCVIns asm_fxstoreins(IRIns *ir)
{
  switch (irt_type(ir->t)) {
  case IRT_I8: case IRT_U8: return RISCVI_SB;
  case IRT_I16: case IRT_U16: return RISCVI_SH;
  case IRT_NUM: return RISCVI_FSD;
  case IRT_FLOAT: return RISCVI_FSW;
  default: return irt_is64(ir->t) ? RISCVI_SD : RISCVI_SW;
  }
}

static void asm_fload(ASMState *as, IRIns *ir)
{
  RegSet allow = RSET_GPR;
  Reg idx, dest = ra_dest(as, ir, allow);
  rset_clear(allow, dest);
  RISCVIns riscvi = asm_fxloadins(ir);
  int32_t ofs;
  if (ir->op1 == REF_NIL) {  /* FLOAD from GG_State with offset. */
    idx = RID_GL;
    ofs = (ir->op2 << 2) - GG_OFS(g);
  } else {
    idx = ra_alloc1(as, ir->op1, allow);
    if (ir->op2 == IRFL_TAB_ARRAY) {
      ofs = asm_fuseabase(as, ir->op1);
      if (ofs) {  /* Turn the t->array load into an add for colocated arrays. */
	emit_dsi(as, RISCVI_ADDI, dest, idx, ofs);
	return;
      }
    }
    ofs = field_ofs[ir->op2];
    lj_assertA(!irt_isfp(ir->t), "bad FP FLOAD");
  }
  rset_clear(allow, idx);
  emit_lso(as, riscvi, dest, idx, ofs);
}

static void asm_fstore(ASMState *as, IRIns *ir)
{
  if (ir->r != RID_SINK) {
    Reg src = ra_alloc1z(as, ir->op2, RSET_GPR);
    IRIns *irf = IR(ir->op1);
    Reg idx = ra_alloc1(as, irf->op1, rset_exclude(RSET_GPR, src));
    int32_t ofs = field_ofs[irf->op2];
    lj_assertA(!irt_isfp(ir->t), "bad FP FSTORE");
    emit_lso(as, asm_fxstoreins(ir), src, idx, ofs);
  }
}

static void asm_xload(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, (irt_isfp(ir->t)) ? RSET_FPR : RSET_GPR);
  lj_assertA(LJ_TARGET_UNALIGNED || !(ir->op2 & IRXLOAD_UNALIGNED),
	     "unaligned XLOAD");
  asm_fusexref(as, asm_fxloadins(ir), dest, ir->op1, RSET_GPR, 0);
}

static void asm_xstore_(ASMState *as, IRIns *ir, int32_t ofs)
{
  if (ir->r != RID_SINK) {
    Reg src = ra_alloc1z(as, ir->op2, irt_isfp(ir->t) ? RSET_FPR : RSET_GPR);
    asm_fusexref(as, asm_fxstoreins(ir), src, ir->op1,
	  	 rset_exclude(RSET_GPR, src), ofs);
  }
}

#define asm_xstore(as, ir)	asm_xstore_(as, ir, 0)

static void asm_ahuvload(ASMState *as, IRIns *ir)
{
  Reg dest = RID_NONE, type = RID_TMP, idx;
  RegSet allow = RSET_GPR;
  int32_t ofs = 0;
  IRType1 t = ir->t;
  if (ra_used(ir)) {
    lj_assertA((irt_isnum(ir->t)) || irt_isint(ir->t) || irt_isaddr(ir->t),
	       "bad load type %d", irt_type(ir->t));
    dest = ra_dest(as, ir, irt_isnum(t) ? RSET_FPR : allow);
    rset_clear(allow, dest);
    if (irt_isaddr(t)) {
      emit_cleartp(as, dest, dest);
    } else if (irt_isint(t))
      emit_ext(as, RISCVI_SEXT_W, dest, dest);
  }
  idx = asm_fuseahuref(as, ir->op1, &ofs, allow);
  if (ir->o == IR_VLOAD) ofs += 8 * ir->op2;
  rset_clear(allow, idx);
  if (irt_isnum(t)) {
    asm_guard(as, RISCVI_BEQ, RID_TMP, RID_ZERO);
    emit_dsi(as, RISCVI_SLTIU, RID_TMP, type, (int32_t)LJ_TISNUM);
  } else {
    asm_guard(as, RISCVI_BNE, type,
	      ra_allock(as, (int32_t)irt_toitype(t), allow));
  }
  if (ra_hasreg(dest)) {
    if (irt_isnum(t)) {
      emit_lso(as, RISCVI_FLD, dest, idx, ofs);
      dest = type;
    }
  } else {
    dest = type;
  }
  emit_dsshamt(as, RISCVI_SRAI, type, dest, 47);
  emit_lso(as, RISCVI_LD, dest, idx, ofs);
}

static void asm_ahustore(ASMState *as, IRIns *ir)
{
  RegSet allow = RSET_GPR;
  Reg idx, src = RID_NONE, type = RID_NONE;
  int32_t ofs = 0;
  if (ir->r == RID_SINK)
    return;
  if (irt_isnum(ir->t)) {
    src = ra_alloc1(as, ir->op2, RSET_FPR);
    idx = asm_fuseahuref(as, ir->op1, &ofs, allow);
    emit_lso(as, RISCVI_FSD, src, idx, ofs);
  } else {
    Reg tmp = RID_TMP;
    if (irt_ispri(ir->t)) {
      tmp = ra_allock(as, ~((int64_t)~irt_toitype(ir->t) << 47), allow);
      rset_clear(allow, tmp);
    } else {
      src = ra_alloc1(as, ir->op2, allow);
      rset_clear(allow, src);
      type = ra_allock(as, (int64_t)irt_toitype(ir->t) << 47, allow);
      rset_clear(allow, type);
    }
    idx = asm_fuseahuref(as, ir->op1, &ofs, allow);
    emit_lso(as, RISCVI_SD, tmp, idx, ofs);
    if (ra_hasreg(src)) {
      if (irt_isinteger(ir->t)) {
  if (as->flags & JIT_F_RVZba) {
    emit_ds1s2(as, RISCVI_ADD_UW, tmp, src, type);
  } else {
    emit_ds1s2(as, RISCVI_ADD, tmp, tmp, type);
    emit_ext(as, RISCVI_ZEXT_W, tmp, src);
  }
      } else {
  emit_ds1s2(as, RISCVI_ADD, tmp, src, type);
      }
    }
  }
}

static void asm_sload(ASMState *as, IRIns *ir)
{
  Reg dest = RID_NONE, type = RID_NONE, base;
  RegSet allow = RSET_GPR;
  IRType1 t = ir->t;
  int32_t ofs = 8*((int32_t)ir->op1-2);
  lj_assertA(checki12(ofs), "sload IR operand out of range");
  lj_assertA(!(ir->op2 & IRSLOAD_PARENT),
	     "bad parent SLOAD");  /* Handled by asm_head_side(). */
  lj_assertA(irt_isguard(t) || !(ir->op2 & IRSLOAD_TYPECHECK),
	     "inconsistent SLOAD variant");
  if ((ir->op2 & IRSLOAD_CONVERT) && irt_isguard(t) && irt_isint(t)) {
    dest = ra_scratch(as, RSET_FPR);
    asm_tointg(as, ir, dest);
    t.irt = IRT_NUM;  /* Continue with a regular number type check. */
  } else if (ra_used(ir)) {
    Reg tmp = RID_NONE;
    if ((ir->op2 & IRSLOAD_CONVERT))
      tmp = ra_scratch(as, irt_isint(t) ? RSET_FPR : RSET_GPR);
    lj_assertA((irt_isnum(t)) || irt_isint(t) || irt_isaddr(t),
	       "bad SLOAD type %d", irt_type(t));
    dest = ra_dest(as, ir, irt_isnum(t) ? RSET_FPR : allow);
    rset_clear(allow, dest);
    base = ra_alloc1(as, REF_BASE, allow);
    rset_clear(allow, base);
    if (irt_isaddr(t)) { /* Clear type from pointers. */
      emit_cleartp(as, dest, dest);
    } else if (ir->op2 & IRSLOAD_CONVERT) {
      if (irt_isint(t)) {
	emit_ds(as, RISCVI_FCVT_W_D|RISCVF_RM(RISCVRM_RTZ), dest, tmp);
  /* If value is already loaded for type check, move it to FPR. */
	if ((ir->op2 & IRSLOAD_TYPECHECK))
	  emit_ds(as, RISCVI_FMV_D_X, tmp, dest);
	else
	  dest = tmp;
	t.irt = IRT_NUM;  /* Check for original type. */
      } else {
	emit_ds(as, RISCVI_FCVT_D_W, dest, tmp);
	dest = tmp;
	t.irt = IRT_INT;  /* Check for original type. */
      }
    } else if (irt_isint(t) && (ir->op2 & IRSLOAD_TYPECHECK)) {
      /* Sign-extend integers. */
      emit_ext(as, RISCVI_SEXT_W, dest, dest);
    }
    goto dotypecheck;
  }
  base = ra_alloc1(as, REF_BASE, allow);
  rset_clear(allow, base);
dotypecheck:
  if ((ir->op2 & IRSLOAD_TYPECHECK)) {
    type = dest < RID_MAX_GPR ? dest : RID_TMP;
    if (irt_ispri(t)) {
      asm_guard(as, RISCVI_BNE, type,
		ra_allock(as, ~((int64_t)~irt_toitype(t) << 47) , allow));
    } else if ((ir->op2 & IRSLOAD_KEYINDEX)) {
      asm_guard(as, RISCVI_BNE, RID_TMP,
               ra_allock(as, (int32_t)LJ_KEYINDEX, allow));
      emit_dsshamt(as, RISCVI_SRAI, RID_TMP, type, 32);
    } else {
      if (irt_isnum(t)) {
        asm_guard(as, RISCVI_BEQ, RID_TMP, RID_ZERO);
        emit_dsi(as, RISCVI_SLTIU, RID_TMP, RID_TMP, LJ_TISNUM);
	if (ra_hasreg(dest)) {
	  emit_lso(as, RISCVI_FLD, dest, base, ofs);
	}
      } else {
	asm_guard(as, RISCVI_BNE, RID_TMP,
		  ra_allock(as, (int32_t)irt_toitype(t), allow));
      }
      emit_dsshamt(as, RISCVI_SRAI, RID_TMP, type, 47);
    }
    emit_lso(as, RISCVI_LD, type, base, ofs);
  } else if (ra_hasreg(dest)) {
    emit_lso(as, irt_isnum(t) ? RISCVI_FLD :
             irt_isint(t) ? RISCVI_LW : RISCVI_LD,
             dest, base, ofs);
  }
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
    emit_lso(as, sz == 8 ? RISCVI_SD : RISCVI_SW, ra_alloc1(as, ir->op2, allow),
	     RID_RET, (sizeof(GCcdata)));
    lj_assertA(sz == 4 || sz == 8, "bad CNEWI size %d", sz);
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
  emit_lso(as, RISCVI_SB, RID_RET+1, RID_RET, (offsetof(GCcdata, gct)));
  emit_lso(as, RISCVI_SH, RID_TMP, RID_RET, (offsetof(GCcdata, ctypeid)));
  emit_loadk12(as, RID_RET+1, ~LJ_TCDATA);
  emit_loadk32(as, RID_TMP, id);
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
  emit_lso(as, RISCVI_SD, link, tab, (int32_t)offsetof(GCtab, gclist));
  emit_lso(as, RISCVI_SB, mark, tab, (int32_t)offsetof(GCtab, marked));
  emit_setgl(as, tab, gc.grayagain);	// make tab gray again
  emit_getgl(as, link, gc.grayagain);
  emit_branch(as, RISCVI_BEQ, RID_TMP, RID_ZERO, l_end, 0);	// black: not jump
  emit_ds1s2(as, RISCVI_XOR, mark, mark, RID_TMP);	// mark=0: gray
  emit_dsi(as, RISCVI_ANDI, RID_TMP, mark, LJ_GC_BLACK);
  emit_lso(as, RISCVI_LBU, mark, tab, ((int32_t)offsetof(GCtab, marked)));
}

static void asm_obar(ASMState *as, IRIns *ir)
{
  const CCallInfo *ci = &lj_ir_callinfo[IRCALL_lj_gc_barrieruv];
  IRRef args[2];
  MCLabel l_end;
  Reg obj, val, tmp;
  /* No need for other object barriers (yet). */
  lj_assertA(IR(ir->op1)->o == IR_UREFC, "bad OBAR type");	// Closed upvalue
  ra_evictset(as, RSET_SCRATCH);
  l_end = emit_label(as);
  args[0] = ASMREF_TMP1;  /* global_State *g */
  args[1] = ir->op1;      /* TValue *tv      */
  asm_gencall(as, ci, args);
  emit_ds(as, RISCVI_MV, ra_releasetmp(as, ASMREF_TMP1), RID_GL);
  obj = IR(ir->op1)->r;
  tmp = ra_scratch(as, rset_exclude(RSET_GPR, obj));
  emit_branch(as, RISCVI_BEQ, tmp, RID_ZERO, l_end, 0);
  emit_branch(as, RISCVI_BEQ, RID_TMP, RID_ZERO, l_end, 0);	// black: jump
  emit_dsi(as, RISCVI_ANDI, tmp, tmp, LJ_GC_BLACK);
  emit_dsi(as, RISCVI_ANDI, RID_TMP, RID_TMP, LJ_GC_WHITES);
  val = ra_alloc1(as, ir->op2, rset_exclude(RSET_GPR, obj));
  emit_lso(as, RISCVI_LBU, tmp, obj,
	   ((int32_t)offsetof(GCupval, marked)-(int32_t)offsetof(GCupval, tv)));
  emit_lso(as, RISCVI_LBU, RID_TMP, val, ((int32_t)offsetof(GChead, marked)));
}

/* -- Arithmetic and logic operations ------------------------------------- */

static void asm_fparith(ASMState *as, IRIns *ir, RISCVIns riscvi)
{
  Reg dest = ra_dest(as, ir, RSET_FPR);
  Reg right, left = ra_alloc2(as, ir, RSET_FPR);
  right = (left >> 8); left &= 255;
  emit_ds1s2(as, riscvi, dest, left, right);
}

static void asm_fpunary(ASMState *as, IRIns *ir, RISCVIns riscvi)
{
  Reg dest = ra_dest(as, ir, RSET_FPR);
  Reg left = ra_hintalloc(as, ir->op1, dest, RSET_FPR);
  switch(riscvi) {
    case RISCVI_FSQRT_S: case RISCVI_FSQRT_D:
      emit_ds(as, riscvi, dest, left);
      break;
    case RISCVI_FMV_S: case RISCVI_FMV_D:
    case RISCVI_FABS_S: case RISCVI_FABS_D:
    case RISCVI_FNEG_S: case RISCVI_FNEG_D:
      emit_ds1s2(as, riscvi, dest, left, left);
      break;
    default:
      lj_assertA(0, "bad fp unary instruction");
      return;
  }
}

static void asm_fpmath(ASMState *as, IRIns *ir)
{
  IRFPMathOp fpm = (IRFPMathOp)ir->op2;
  if (fpm <= IRFPM_TRUNC)
    asm_callround(as, ir, IRCALL_lj_vm_floor + fpm);
  else if (fpm == IRFPM_SQRT)
    asm_fpunary(as, ir, RISCVI_FSQRT_D);
  else
    asm_callid(as, ir, IRCALL_lj_vm_floor + fpm);
}

static void asm_add(ASMState *as, IRIns *ir)
{
  IRType1 t = ir->t;
  if (irt_isnum(t)) {
    if (!asm_fusemadd(as, ir, RISCVI_FMADD_D, RISCVI_FMADD_D))
      asm_fparith(as, ir, RISCVI_FADD_D);
    return;
  } else {
    if ((as->flags & JIT_F_RVXThead) && asm_fusemac(as, ir, RISCVI_TH_MULA))
      return;
    Reg dest = ra_dest(as, ir, RSET_GPR);
    Reg left = ra_hintalloc(as, ir->op1, dest, RSET_GPR);
    if (irref_isk(ir->op2)) {
      intptr_t k = get_kval(as, ir->op2);
      if (checki12(k)) {
  if (irt_is64(t)) {
    emit_dsi(as, RISCVI_ADDI, dest, left, k);
  } else {
	  emit_dsi(as, RISCVI_ADDIW, dest, left, k);
  }
	return;
      }
    }
    Reg right = ra_alloc1(as, ir->op2, rset_exclude(RSET_GPR, left));
    emit_ds1s2(as, irt_is64(t) ? RISCVI_ADD : RISCVI_ADDW, dest,
	     left, right);
  }
}

static void asm_sub(ASMState *as, IRIns *ir)
{
  if (irt_isnum(ir->t)) {
    if (!asm_fusemadd(as, ir, RISCVI_FMSUB_D, RISCVI_FNMSUB_D))
      asm_fparith(as, ir, RISCVI_FSUB_D);
    return;
  } else {
    if ((as->flags & JIT_F_RVXThead) && asm_fusemac(as, ir, RISCVI_TH_MULS))
      return;
    Reg dest = ra_dest(as, ir, RSET_GPR);
    Reg right, left = ra_alloc2(as, ir, RSET_GPR);
    right = (left >> 8); left &= 255;
    emit_ds1s2(as, irt_is64(ir->t) ? RISCVI_SUB : RISCVI_SUBW, dest,
	     left, right);
  }
}

static void asm_mul(ASMState *as, IRIns *ir)
{
  if (irt_isnum(ir->t)) {
    asm_fparith(as, ir, RISCVI_FMUL_D);
  } else {
    Reg dest = ra_dest(as, ir, RSET_GPR);
    Reg right, left = ra_alloc2(as, ir, RSET_GPR);
    right = (left >> 8); left &= 255;
    emit_ds1s2(as, irt_is64(ir->t) ? RISCVI_MUL : RISCVI_MULW, dest,
	     left, right);
  }
}

static void asm_fpdiv(ASMState *as, IRIns *ir)
{
    asm_fparith(as, ir, RISCVI_FDIV_D);
}

static void asm_neg(ASMState *as, IRIns *ir)
{
  if (irt_isnum(ir->t)) {
    asm_fpunary(as, ir, RISCVI_FNEG_D);
  } else {
    Reg dest = ra_dest(as, ir, RSET_GPR);
    Reg left = ra_hintalloc(as, ir->op1, dest, RSET_GPR);
    emit_ds1s2(as, irt_is64(ir->t) ? RISCVI_SUB : RISCVI_SUBW, dest,
	     RID_ZERO, left);
  }
}

#define asm_abs(as, ir)		asm_fpunary(as, ir, RISCVI_FABS_D)

static void asm_arithov(ASMState *as, IRIns *ir)
{
  Reg right, left, tmp, dest = ra_dest(as, ir, RSET_GPR);
  lj_assertA(!irt_is64(ir->t), "bad usage");
  if (irref_isk(ir->op2)) {
    int k = IR(ir->op2)->i;
    if (ir->o == IR_SUBOV) k = (int)(~(unsigned int)k+1u);
    if (checki12(k)) {	/* (dest < left) == (k >= 0 ? 1 : 0) */
      left = ra_alloc1(as, ir->op1, RSET_GPR);
      asm_guard(as, k >= 0 ? RISCVI_BLT : RISCVI_BGE, dest, dest == left ? RID_TMP : left);
      emit_dsi(as, RISCVI_ADDI, dest, left, k);
      if (dest == left) emit_mv(as, RID_TMP, left);
      return;
    }
  }
  left = ra_alloc2(as, ir, RSET_GPR);
  right = (left >> 8); left &= 255;
  tmp = ra_scratch(as, rset_exclude(rset_exclude(rset_exclude(RSET_GPR, left),
						 right), dest));
  asm_guard(as, RISCVI_BLT, RID_TMP, RID_ZERO);
  emit_ds1s2(as, RISCVI_AND, RID_TMP, RID_TMP, tmp);
  if (ir->o == IR_ADDOV) {  /* ((dest^left) & (dest^right)) < 0 */
    emit_ds1s2(as, RISCVI_XOR, RID_TMP, dest, dest == right ? RID_TMP : right);
  } else {  /* ((dest^left) & (dest^~right)) < 0 */
    emit_xnor(as, RID_TMP, dest, dest == right ? RID_TMP : right);
  }
  emit_ds1s2(as, RISCVI_XOR, tmp, dest, dest == left ? RID_TMP : left);
  emit_ds1s2(as, ir->o == IR_ADDOV ? RISCVI_ADDW : RISCVI_SUBW, dest, left, right);
  if (dest == left || dest == right)
    emit_mv(as, RID_TMP, dest == left ? left : right);
}

#define asm_addov(as, ir)	asm_arithov(as, ir)
#define asm_subov(as, ir)	asm_arithov(as, ir)

static void asm_mulov(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  Reg right, left = ra_alloc2(as, ir, RSET_GPR);
  right = (left >> 8); left &= 255;
  asm_guard(as, RISCVI_BNE, dest, RID_TMP);
  emit_ext(as, RISCVI_SEXT_W, dest, RID_TMP);	// dest: [31:0]+signextend
  emit_ds1s2(as, RISCVI_MUL, RID_TMP, left, right);	// RID_TMP: [63:0]
}

static void asm_bnot(ASMState *as, IRIns *ir)
{
  Reg left, right, dest = ra_dest(as, ir, RSET_GPR);
  IRIns *irl = IR(ir->op1);
  if (as->flags & JIT_F_RVZbb && mayfuse(as, ir->op1) && irl->o == IR_BXOR) {
    left = ra_alloc2(as, irl, RSET_GPR);
    right = (left >> 8); left &= 255;
    emit_ds1s2(as, RISCVI_XNOR, dest, left, right);
  } else {
    left = ra_hintalloc(as, ir->op1, dest, RSET_GPR);
    emit_ds(as, RISCVI_NOT, dest, left);
  }
}

static void asm_bswap(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  Reg left = ra_alloc1(as, ir->op1, RSET_GPR);
  RegSet allow = rset_exclude(rset_exclude(RSET_GPR, dest), left);
  if (as->flags & JIT_F_RVZbb) {
    if (!irt_is64(ir->t))
      emit_dsshamt(as, RISCVI_SRAI, dest, dest, 32);
    emit_ds(as, RISCVI_REV8, dest, left);
  } else if (as->flags & JIT_F_RVXThead) {
    emit_ds(as, irt_is64(ir->t) ? RISCVI_TH_REV : RISCVI_TH_REVW,
       dest, left);
  } else if (irt_is64(ir->t)) {
    Reg tmp1, tmp2, tmp3, tmp4;
    tmp1 = ra_scratch(as, allow), allow = rset_exclude(allow, tmp1);
    tmp2 = ra_scratch(as, allow), allow = rset_exclude(allow, tmp2);
    tmp3 = ra_scratch(as, allow), allow = rset_exclude(allow, tmp3);
    tmp4 = ra_scratch(as, allow);
    emit_ds1s2(as, RISCVI_OR, dest, dest, tmp4);
    emit_ds1s2(as, RISCVI_OR, dest, dest, tmp3);
    emit_ds1s2(as, RISCVI_OR, dest, dest, tmp2);
    emit_dsshamt(as, RISCVI_SLLI, tmp4, tmp4, 40);
    emit_dsshamt(as, RISCVI_SLLI, dest, left, 56);
    emit_ds1s2(as, RISCVI_OR, tmp3, tmp1, tmp3);
    emit_ds1s2(as, RISCVI_AND, tmp4, left, RID_TMP);
    emit_dsshamt(as, RISCVI_SLLI, tmp3, tmp3, 32);
    emit_dsshamt(as, RISCVI_SLLI, tmp1, tmp1, 24);
    emit_dsshamt(as, RISCVI_SRLIW, tmp3, left, 24);
    emit_ds1s2(as, RISCVI_OR, tmp2, tmp3, tmp2);
    emit_ds1s2(as, RISCVI_AND, tmp1, left, tmp1);
    emit_ds1s2(as, RISCVI_OR, tmp3, tmp4, tmp3);
    emit_dsshamt(as, RISCVI_SLLI, tmp4, tmp4, 24);
    emit_dsshamt(as, RISCVI_SRLIW, tmp4, tmp4, 24);
    emit_ds1s2(as, RISCVI_AND, tmp3, tmp3, tmp1);
    emit_dsshamt(as, RISCVI_SRLI, tmp4, left, 8);
    emit_dsshamt(as, RISCVI_SRLI, tmp3, left, 24);
    emit_ds1s2(as, RISCVI_OR, tmp2, tmp2, tmp3);
    emit_du(as, RISCVI_LUI, tmp1, RISCVF_HI(0xff0000u));
    emit_ds1s2(as, RISCVI_AND, tmp2, tmp2, RID_TMP);
    emit_dsshamt(as, RISCVI_SRLI, tmp3, left, 56);
    emit_dsi(as, RISCVI_ADDI, RID_TMP, RID_TMP, RISCVF_LO(0xff00));
    emit_du(as, RISCVI_LUI, RID_TMP, RISCVF_HI(0xff00u));
    emit_dsshamt(as, RISCVI_SRLI, tmp2, left, 40);
  } else {
    Reg tmp1, tmp2;
    tmp1 = ra_scratch(as, allow), allow = rset_exclude(allow, tmp1);
    tmp2 = ra_scratch(as, allow);
    emit_ds1s2(as, RISCVI_OR, dest, dest, tmp2);
    emit_ds1s2(as, RISCVI_OR, dest, dest, tmp1);
    emit_dsshamt(as, RISCVI_SLLI, tmp2, RID_TMP, 8);
    emit_dsshamt(as, RISCVI_SLLIW, dest, left, 24);
    emit_ds1s2(as, RISCVI_OR, tmp1, tmp1, tmp2);
    emit_ds1s2(as, RISCVI_AND, RID_TMP, left, RID_TMP);
    emit_ds1s2(as, RISCVI_AND, tmp1, tmp1, RID_TMP);
    emit_dsshamt(as, RISCVI_SRLIW, tmp2, left, 24);
    emit_dsi(as, RISCVI_ADDI, RID_TMP, RID_TMP, RISCVF_LO(0xff00));
    emit_du(as, RISCVI_LUI, RID_TMP, RISCVF_HI(0xff00u));
    emit_dsshamt(as, RISCVI_SRLI, tmp1, left, 8);
  }
}

static void asm_bitop(ASMState *as, IRIns *ir, RISCVIns riscvi, RISCVIns riscvik, RISCVIns riscvin)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  Reg left, right;
  IRIns *irl = IR(ir->op1), *irr = IR(ir->op2);
  if (irref_isk(ir->op2)) {
    intptr_t k = get_kval(as, ir->op2);
    if (checki12(k)) {
      left = ra_hintalloc(as, ir->op1, dest, RSET_GPR);
      emit_dsi(as, riscvik, dest, left, k);
      return;
    }
  } else if (as->flags & JIT_F_RVZbb) {
    if (mayfuse(as, ir->op1) && irl->o == IR_BNOT) {
      left = ra_alloc1(as, irl->op1, RSET_GPR);
      right = ra_alloc1(as, ir->op2, rset_exclude(RSET_GPR, left));
      emit_ds1s2(as, riscvin, dest, right, left);
      return;
    } else if (mayfuse(as, ir->op2) && irr->o == IR_BNOT) {
      left = ra_alloc1(as, ir->op1, RSET_GPR);
      right = ra_alloc1(as, irr->op1, rset_exclude(RSET_GPR, left));
      emit_ds1s2(as, riscvin, dest, left, right);
      return;
    }
  }
  left = ra_hintalloc(as, ir->op1, dest, RSET_GPR);
  right = ra_alloc1(as, ir->op2, rset_exclude(RSET_GPR, left));
  emit_ds1s2(as, riscvi, dest, left, right);
}

#define asm_band(as, ir)	asm_bitop(as, ir, RISCVI_AND, RISCVI_ANDI, RISCVI_ANDN)
#define asm_bor(as, ir)	asm_bitop(as, ir, RISCVI_OR, RISCVI_ORI, RISCVI_ORN)
#define asm_bxor(as, ir)	asm_bitop(as, ir, RISCVI_XOR, RISCVI_XORI, RISCVI_XNOR)

static void asm_bitshift(ASMState *as, IRIns *ir, RISCVIns riscvi, RISCVIns riscvik)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  Reg left = ra_alloc1(as, ir->op1, RSET_GPR);
  uint32_t shmsk = irt_is64(ir->t) ? 63 : 31;
  if (irref_isk(ir->op2)) {  /* Constant shifts. */
    uint32_t shift = (uint32_t)(IR(ir->op2)->i & shmsk);
    switch (riscvik) {
      case RISCVI_SRAI: case RISCVI_SRLI: case RISCVI_SLLI:
      case RISCVI_SRAIW: case RISCVI_SLLIW: case RISCVI_SRLIW:
        emit_dsshamt(as, riscvik, dest, left, shift);
        break;
      case RISCVI_ADDI: shift = (-shift) & shmsk;
      case RISCVI_RORI:
        emit_roti(as, RISCVI_RORI, dest, left, RID_TMP, shift);
        break;
      case RISCVI_ADDIW: shift = (-shift) & shmsk;
      case RISCVI_RORIW:
        emit_roti(as, RISCVI_RORIW, dest, left, RID_TMP, shift);
        break;
      default:
        lj_assertA(0, "bad shift instruction");
        return;
    }
  } else {
    Reg right = ra_alloc1(as, ir->op2, rset_exclude(RSET_GPR, left));
    switch (riscvi) {
      case RISCVI_SRA: case RISCVI_SRL: case RISCVI_SLL:
      case RISCVI_SRAW: case RISCVI_SRLW: case RISCVI_SLLW:
        emit_ds1s2(as, riscvi, dest, left, right);
        break;
      case RISCVI_ROR: case RISCVI_ROL:
      case RISCVI_RORW: case RISCVI_ROLW:
        emit_rot(as, riscvi, dest, left, right, RID_TMP);
        break;
      default:
        lj_assertA(0, "bad shift instruction");
        return;
    }
  }
}

#define asm_bshl(as, ir)	(irt_is64(ir->t) ? \
  asm_bitshift(as, ir, RISCVI_SLL, RISCVI_SLLI) : \
  asm_bitshift(as, ir, RISCVI_SLLW, RISCVI_SLLIW))
#define asm_bshr(as, ir)	(irt_is64(ir->t) ? \
  asm_bitshift(as, ir, RISCVI_SRL, RISCVI_SRLI) : \
  asm_bitshift(as, ir, RISCVI_SRLW, RISCVI_SRLIW))
#define asm_bsar(as, ir)	(irt_is64(ir->t) ? \
  asm_bitshift(as, ir, RISCVI_SRA, RISCVI_SRAI) : \
  asm_bitshift(as, ir, RISCVI_SRAW, RISCVI_SRAIW))
#define asm_brol(as, ir)	(irt_is64(ir->t) ? \
  asm_bitshift(as, ir, RISCVI_ROL, RISCVI_ADDI) : \
  asm_bitshift(as, ir, RISCVI_ROLW, RISCVI_ADDIW))
  // ROLI -> ADDI, ROLIW -> ADDIW; Hacky but works.
#define asm_bror(as, ir)	(irt_is64(ir->t) ? \
  asm_bitshift(as, ir, RISCVI_ROR, RISCVI_RORI) : \
  asm_bitshift(as, ir, RISCVI_RORW, RISCVI_RORIW))

static void asm_min_max(ASMState *as, IRIns *ir, int ismax)
{
  if (irt_isnum(ir->t)) {
    Reg dest = ra_dest(as, ir, RSET_FPR);
    Reg right, left = ra_alloc2(as, ir, RSET_FPR);
    right = (left >> 8); left &= 255;
    emit_ds1s2(as, ismax ? RISCVI_FMAX_D : RISCVI_FMIN_D, dest, left, right);
  } else {
    Reg dest = ra_dest(as, ir, RSET_GPR);
    Reg left = ra_hintalloc(as, ir->op1, dest, RSET_GPR);
    Reg right = ra_alloc1(as, ir->op2, rset_exclude(RSET_GPR, left));
    if (as->flags & JIT_F_RVZbb) {
      emit_ds1s2(as, ismax ? RISCVI_MAX : RISCVI_MIN, dest, left, right);
    } else {
      if (as->flags & JIT_F_RVXThead) {
  if (left == right) {
    if (dest != left) emit_mv(as, dest, left);
  } else {
    if (dest == left) {
	    emit_ds1s2(as, RISCVI_TH_MVNEZ, dest, right, RID_TMP);
    } else {
	    emit_ds1s2(as, RISCVI_TH_MVEQZ, dest, left, RID_TMP);
	    if (dest != right) emit_mv(as, dest, right);
    }
  }
      } else if (as->flags & JIT_F_RVZicond) {
  emit_ds1s2(as, RISCVI_OR, dest, dest, RID_TMP);
  if (dest != right) {
    emit_ds1s2(as, RISCVI_CZERO_EQZ, RID_TMP, right, RID_TMP);
    emit_ds1s2(as, RISCVI_CZERO_NEZ, dest, left, RID_TMP);
  } else {
    emit_ds1s2(as, RISCVI_CZERO_NEZ, RID_TMP, left, RID_TMP);
    emit_ds1s2(as, RISCVI_CZERO_EQZ, dest, right, RID_TMP);
  }
      } else {
  if (dest != right) {
    emit_ds1s2(as, RISCVI_XOR, dest, right, dest);
    emit_ds1s2(as, RISCVI_AND, dest, dest, RID_TMP);
    emit_ds1s2(as, RISCVI_XOR, dest, right, left);
    emit_dsi(as, RISCVI_ADDI, RID_TMP, RID_TMP, -1);
  } else {
    emit_ds1s2(as, RISCVI_XOR, dest, left, dest);
    emit_ds1s2(as, RISCVI_AND, dest, dest, RID_TMP);
    emit_ds1s2(as, RISCVI_XOR, dest, left, right);
    emit_ds1s2(as, RISCVI_SUB, RID_TMP, RID_ZERO, RID_TMP);
  }
      }
      emit_ds1s2(as, RISCVI_SLT, RID_TMP,
         ismax ? left : right, ismax ? right : left);
    }
  }
}

#define asm_min(as, ir)		asm_min_max(as, ir, 0)
#define asm_max(as, ir)		asm_min_max(as, ir, 1)

/* -- Comparisons --------------------------------------------------------- */

/* FP comparisons. */
static void asm_fpcomp(ASMState *as, IRIns *ir)
{
  IROp op = ir->o;
  Reg right, left = ra_alloc2(as, ir, RSET_FPR);
  right = (left >> 8); left &= 255;
  asm_guard(as, (op < IR_EQ ? (op&4) : (op&1))
            ? RISCVI_BNE : RISCVI_BEQ, RID_TMP, RID_ZERO);
  switch (op) {
    case IR_LT: case IR_UGE:
      emit_ds1s2(as, RISCVI_FLT_D, RID_TMP, left, right);
      break;
    case IR_LE: case IR_UGT: case IR_ABC:
      emit_ds1s2(as, RISCVI_FLE_D, RID_TMP, left, right);
      break;
    case IR_GT: case IR_ULE:
      emit_ds1s2(as, RISCVI_FLT_D, RID_TMP, right, left);
      break;
    case IR_GE: case IR_ULT:
      emit_ds1s2(as, RISCVI_FLE_D, RID_TMP, right, left);
      break;
    case IR_EQ: case IR_NE:
      emit_ds1s2(as, RISCVI_FEQ_D, RID_TMP, left, right);
      break;
    default:
      break;
  }
}

/* Integer comparisons. */
static void asm_intcomp(ASMState *as, IRIns *ir)
{
  /* ORDER IR: LT GE LE GT  ULT UGE ULE UGT. */
  /*           00 01 10 11  100 101 110 111  */
  IROp op = ir->o;
  Reg right, left = ra_alloc1(as, ir->op1, RSET_GPR);
  if (op == IR_ABC) op = IR_UGT;
  if ((op&4) == 0 && irref_isk(ir->op2) && get_kval(as, ir->op2) == 0) {
    switch (op) {
      case IR_LT: asm_guard(as, RISCVI_BGE, left, RID_ZERO); break;
      case IR_GE: asm_guard(as, RISCVI_BLT, left, RID_ZERO); break;
      case IR_LE: asm_guard(as, RISCVI_BLT, RID_ZERO, left); break;
      case IR_GT: asm_guard(as, RISCVI_BGE, RID_ZERO, left); break;
      default: break;
    }
    return;
  }
  if (irref_isk(ir->op2)) {
    intptr_t k = get_kval(as, ir->op2);
    if ((op&2)) k++;
    if (checki12(k)) {
      asm_guard(as, (op&1) ? RISCVI_BNE : RISCVI_BEQ, RID_TMP, RID_ZERO);
      emit_dsi(as, (op&4) ? RISCVI_SLTIU : RISCVI_SLTI, RID_TMP, left, k);
      return;
    }
  }
  right = ra_alloc1(as, ir->op2, rset_exclude(RSET_GPR, left));
  asm_guard(as, ((op&4) ? RISCVI_BGEU : RISCVI_BGE) ^ RISCVF_FUNCT3((op^(op>>1))&1),
             (op&2) ? right : left, (op&2) ? left : right);
}

static void asm_comp(ASMState *as, IRIns *ir)
{
  if (irt_isnum(ir->t))
    asm_fpcomp(as, ir);
  else
    asm_intcomp(as, ir);
}

static void asm_equal(ASMState *as, IRIns *ir)
{
  if (irt_isnum(ir->t)) {
    asm_fpcomp(as, ir);
  } else {
    Reg right, left = ra_alloc2(as, ir, RSET_GPR);
    right = (left >> 8); left &= 255;
    asm_guard(as, (ir->o & 1) ? RISCVI_BEQ : RISCVI_BNE, left, right);
  }
}

/* -- Split register ops -------------------------------------------------- */

/* Hiword op of a split 64 bit op. Previous op must be the loword op. */
static void asm_hiop(ASMState *as, IRIns *ir)
{
  /* HIOP is marked as a store because it needs its own DCE logic. */
  int uselo = ra_used(ir-1), usehi = ra_used(ir);  /* Loword/hiword used? */
  if (LJ_UNLIKELY(!(as->flags & JIT_F_OPT_DCE))) uselo = usehi = 1;
  if (!usehi) return;  /* Skip unused hiword op for all remaining ops. */
  switch ((ir-1)->o) {
  case IR_CALLN:
  case IR_CALLL:
  case IR_CALLS:
  case IR_CALLXS:
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
  asm_guard(as, RISCVI_BNE, RID_TMP, RID_ZERO);
  emit_dsi(as, RISCVI_ANDI, RID_TMP, RID_TMP, HOOK_PROFILE);
  emit_lsglptr(as, RISCVI_LBU, RID_TMP,
         (int32_t)offsetof(global_State, hookmask));
}

/* -- Stack handling ------------------------------------------------------ */

/* Check Lua stack size for overflow. Use exit handler as fallback. */
static void asm_stack_check(ASMState *as, BCReg topslot,
			    IRIns *irp, RegSet allow, ExitNo exitno)
{
  /* Try to get an unused temp register, otherwise spill/restore RID_RET*. */
  Reg tmp, pbase = irp ? (ra_hasreg(irp->r) ? irp->r : RID_TMP) : RID_BASE;
  ExitNo oldsnap = as->snapno;
  rset_clear(allow, pbase);
  as->snapno = exitno;
  asm_guard(as, RISCVI_BNE, RID_TMP, RID_ZERO);
  as->snapno = oldsnap;
  if (allow) {
    tmp = rset_pickbot(allow);
    ra_modified(as, tmp);
  } else {	// allow == RSET_EMPTY
    tmp = RID_RET;
    emit_lso(as, RISCVI_LD, tmp, RID_SP, 0);	/* Restore tmp1 register. */
  }
  emit_dsi(as, RISCVI_SLTIU, RID_TMP, RID_TMP, (int32_t)(8*topslot));
  emit_ds1s2(as, RISCVI_SUB, RID_TMP, tmp, pbase);
  emit_lso(as, RISCVI_LD, tmp, tmp, offsetof(lua_State, maxstack));
  if (pbase == RID_TMP)
    emit_getgl(as, RID_TMP, jit_base);
  emit_getgl(as, tmp, cur_L);
  if (allow == RSET_EMPTY)  /* Spill temp register. */
    emit_lso(as, RISCVI_SD, tmp, RID_SP, 0);
}

/* Restore Lua stack from on-trace state. */
static void asm_stack_restore(ASMState *as, SnapShot *snap)
{
  SnapEntry *map = &as->T->snapmap[snap->mapofs];
#ifdef LUA_USE_ASSERT
  SnapEntry *flinks = &as->T->snapmap[snap_nextofs(as->T, snap)-1-LJ_FR2];
#endif
  MSize n, nent = snap->nent;
  /* Store the value of all modified slots to the Lua stack. */
  for (n = 0; n < nent; n++) {
    SnapEntry sn = map[n];
    BCReg s = snap_slot(sn);
    int32_t ofs = 8*((int32_t)s-1-LJ_FR2);
    IRRef ref = snap_ref(sn);
    IRIns *ir = IR(ref);
    if ((sn & SNAP_NORESTORE))
      continue;
    if (irt_isnum(ir->t)) {
      Reg src = ra_alloc1(as, ref, RSET_FPR);
      emit_lso(as, RISCVI_FSD, src, RID_BASE, ofs);
    } else {
      if ((sn & SNAP_KEYINDEX)) {
        RegSet allow = rset_exclude(RSET_GPR, RID_BASE);
	int64_t kki = (int64_t)LJ_KEYINDEX << 32;
	if (irref_isk(ref)) {
	  emit_lso(as, RISCVI_SD,
       ra_allock(as, kki | (int64_t)(uint32_t)ir->i, allow),
       RID_BASE, ofs);
	} else {
	  Reg src = ra_alloc1(as, ref, allow);
	  Reg rki = ra_allock(as, kki, rset_exclude(allow, src));
	  emit_lso(as, RISCVI_SD, RID_TMP, RID_BASE, ofs);
	  emit_ds1s2(as, RISCVI_ADD, RID_TMP, src, rki);
	}
      } else {
        asm_tvstore64(as, RID_BASE, ofs, ref);
      }
    }
    checkmclim(as);
  }
  lj_assertA(map + nent == flinks, "inconsistent frames in snapshot");
}

/* -- GC handling --------------------------------------------------------- */

/* Marker to prevent patching the GC check exit. */
#define RISCV_NOPATCH_GC_CHECK \
  (RISCVI_OR|RISCVF_D(RID_TMP)|RISCVF_S1(RID_TMP)|RISCVF_S2(RID_TMP))

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
  asm_guard(as, RISCVI_BNE, RID_RET, RID_ZERO);	/* Assumes asm_snap_prep() already done. */
  *--as->mcp = RISCV_NOPATCH_GC_CHECK;
  args[0] = ASMREF_TMP1;  /* global_State *g */
  args[1] = ASMREF_TMP2;  /* MSize steps     */
  asm_gencall(as, ci, args);
  emit_ds(as, RISCVI_MV, ra_releasetmp(as, ASMREF_TMP1), RID_GL);
  tmp = ra_releasetmp(as, ASMREF_TMP2);
  emit_loadi(as, tmp, as->gcsteps);
  /* Jump around GC step if GC total < GC threshold. */
  emit_branch(as, RISCVI_BLTU, RID_TMP, tmp, l_end, 0);
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
  ptrdiff_t delta;
  if (as->loopinv) {  /* Inverted loop branch? */
    delta = (char *)target - (char *)(p - 2);
    /* asm_guard* already inverted the branch, and patched the final b. */
    lj_assertA(checki21(delta), "branch target out of range");
    p[-2] = (p[-2]&0x00000fff) | RISCVF_IMMJ(delta);
  } else {
    /* J */
    delta = (char *)target - (char *)(p - 1);
    p[-1] = RISCVI_JAL | RISCVF_IMMJ(delta);
  }
}

/* Fixup the tail of the loop. */
static void asm_loop_tail_fixup(ASMState *as)
{
  UNUSED(as);  /* Nothing to do(?) */
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
      emit_mv(as, r, RID_BASE);
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
      emit_mv(as, r, irp->r);  /* Move from coalesced parent reg. */
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
  MCode *target = lnk ? traceref(as->J,lnk)->mcode : (MCode *)lj_vm_exit_interp;
  int32_t spadj = as->T->spadjust;
  if (spadj == 0) {
    p[-3] = RISCVI_NOP;
    // as->mctop = p-2;
  } else {
    /* Patch stack adjustment. */
    p[-3] = RISCVI_ADDI | RISCVF_D(RID_SP) | RISCVF_S1(RID_SP) | RISCVF_IMMI(spadj);
  }
  /* Patch exit jump. */
  ptrdiff_t delta = (char *)target - (char *)(p - 2);
  p[-2] = RISCVI_AUIPC | RISCVF_D(RID_TMP) | RISCVF_IMMU(RISCVF_HI(delta));
  p[-1] = RISCVI_JALR | RISCVF_S1(RID_TMP) | RISCVF_IMMI(RISCVF_LO(delta));
}

/* Prepare tail of code. */
static void asm_tail_prep(ASMState *as)
{
  MCode *p = as->mctop - 2;  /* Leave room for exitstub. */
  if (as->loopref) {
    as->invmcp = as->mcp = p;
  } else {
    as->mcp = p-1;  /* Leave room for stack pointer adjustment. */
    as->invmcp = NULL;
  }
  p[0] = p[1] = RISCVI_NOP;  /* Prevent load/store merging. */
}

/* -- Trace setup --------------------------------------------------------- */

/* Ensure there are enough stack slots for call arguments. */
static Reg asm_setup_call_slots(ASMState *as, IRIns *ir, const CCallInfo *ci)
{
  IRRef args[CCI_NARGS_MAX*2];
  uint32_t i, nargs = CCI_XNARGS(ci);
  int nslots = 0, ngpr = REGARG_NUMGPR, nfpr = REGARG_NUMFPR;
  asm_collectargs(as, ir, ci, args);
  for (i = 0; i < nargs; i++) {
    if (args[i] && irt_isfp(IR(args[i])->t)) {
      if (nfpr > 0) {
        nfpr--; if(ci->flags & CCI_VARARG) ngpr--;
      } else if (!(ci->flags & CCI_VARARG) && ngpr > 0) ngpr--;
      else nslots += 2;
    } else {
      if (ngpr > 0) {
        ngpr--; if(ci->flags & CCI_VARARG) nfpr--;
      } else nslots += 2;
    }
  }
  if (nslots > as->evenspill)  /* Leave room for args in stack slots. */
    as->evenspill = nslots;
  return REGSP_HINT(irt_isfp(ir->t) ? RID_FPRET : RID_RET);
}

static void asm_setup_target(ASMState *as)
{
  asm_sparejump_setup(as);
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

  for (; p < pe; p++) {
    /* Look for exitstub branch, replace with branch to target. */
    ptrdiff_t odelta = (char *)px - (char *)(p+1),
              ndelta = (char *)target - (char *)(p+1);
    if ((((p[0] ^ RISCVF_IMMB(8)) & 0xfe000f80u) == 0 &&
         ((p[0] & 0x0000007fu) == 0x63u) &&
         ((p[1] ^ RISCVF_IMMJ(odelta)) & 0xfffff000u) == 0 &&
         ((p[1] & 0x0000007fu) == 0x6fu) && p[-1] != RISCV_NOPATCH_GC_CHECK) ||
        (((p[1] ^ RISCVF_IMMJ(odelta)) & 0xfffff000u) == 0 &&
         ((p[1] & 0x0000007fu) == 0x6fu) && p[0] != RISCV_NOPATCH_GC_CHECK)) {
      lj_assertJ(checki32(ndelta), "branch target out of range");
      /* Patch jump, if within range. */
	    patchbranch:
      if (checki21(ndelta)) { /* Patch jump */
  p[1] = RISCVI_JAL | RISCVF_IMMJ(ndelta);
  if (!cstart) cstart = p + 1;
      } else {  /* Branch out of range. Use spare jump slot in mcarea. */
  MCode *mcjump = asm_sparejump_use(mcarea, target);
  if (mcjump) {
	  lj_mcode_sync(mcjump, mcjump+2);
    ndelta = (char *)mcjump - (char *)(p+1);
    if (checki21(ndelta)) {
      goto patchbranch;
    } else {
      lj_assertJ(0, "spare jump out of range: -Osizemcode too big");
    }
  }
	/* Ignore jump slot overflow. Child trace is simply not attached. */
      }
    } else if (p+2 == pe) {
      if (p[0] == RISCVI_NOP && p[1] == RISCVI_NOP) {
  ptrdiff_t delta = (char *)target - (char *)p;
  lj_assertJ(checki32(delta), "jump target out of range");
  p[0] = RISCVI_AUIPC | RISCVF_D(RID_TMP) | RISCVF_IMMU(RISCVF_HI(delta));
  p[1] = RISCVI_JALR | RISCVF_S1(RID_TMP) | RISCVF_IMMI(RISCVF_LO(delta));
  if (!cstart) cstart = p;
      }
    }
  }
  if (cstart) lj_mcode_sync(cstart, px+1);
  lj_mcode_patch(J, mcarea, 1);
}
