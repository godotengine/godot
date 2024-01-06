/*
** MIPS IR assembler (SSA IR -> machine code).
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

/* Need some spare long-range jump slots, for out-of-range branches. */
#define MIPS_SPAREJUMP		4

/* Setup spare long-range jump slots per mcarea. */
static void asm_sparejump_setup(ASMState *as)
{
  MCode *mxp = as->mctop;
  if ((char *)mxp == (char *)as->J->mcarea + as->J->szmcarea) {
    mxp -= MIPS_SPAREJUMP*2;
    lj_assertA(MIPSI_NOP == 0, "bad NOP");
    memset(mxp, 0, MIPS_SPAREJUMP*2*sizeof(MCode));
    as->mctop = mxp;
  }
}

static MCode *asm_sparejump_use(MCode *mcarea, MCode tjump)
{
  MCode *mxp = (MCode *)((char *)mcarea + ((MCLink *)mcarea)->size);
  int slot = MIPS_SPAREJUMP;
  while (slot--) {
    mxp -= 2;
    if (*mxp == tjump) {
      return mxp;
    } else if (*mxp == MIPSI_NOP) {
      *mxp = tjump;
      return mxp;
    }
  }
  return NULL;
}

/* Setup exit stub after the end of each trace. */
static void asm_exitstub_setup(ASMState *as)
{
  MCode *mxp = as->mctop;
  /* sw TMP, 0(sp); j ->vm_exit_handler; li TMP, traceno */
  *--mxp = MIPSI_LI|MIPSF_T(RID_TMP)|as->T->traceno;
  *--mxp = MIPSI_J|((((uintptr_t)(void *)lj_vm_exit_handler)>>2)&0x03ffffffu);
  lj_assertA(((uintptr_t)mxp ^ (uintptr_t)(void *)lj_vm_exit_handler)>>28 == 0,
	     "branch target out of range");
  *--mxp = MIPSI_SW|MIPSF_T(RID_TMP)|MIPSF_S(RID_SP)|0;
  as->mctop = mxp;
}

/* Keep this in-sync with exitstub_trace_addr(). */
#define asm_exitstub_addr(as)	((as)->mctop)

/* Emit conditional branch to exit for guard. */
static void asm_guard(ASMState *as, MIPSIns mi, Reg rs, Reg rt)
{
  MCode *target = asm_exitstub_addr(as);
  MCode *p = as->mcp;
  if (LJ_UNLIKELY(p == as->invmcp)) {
    as->invmcp = NULL;
    as->loopinv = 1;
    as->mcp = p+1;
#if !LJ_TARGET_MIPSR6
    mi = mi ^ ((mi>>28) == 1 ? 0x04000000u : 0x00010000u);  /* Invert cond. */
#else
    mi = mi ^ ((mi>>28) == 1 ? 0x04000000u :
	       (mi>>28) == 4 ? 0x00800000u : 0x00010000u);  /* Invert cond. */
#endif
    target = p;  /* Patch target later in asm_loop_fixup. */
  }
  emit_ti(as, MIPSI_LI, RID_TMP, as->snapno);
  emit_branch(as, mi, rs, rt, target);
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
	  if (checki16(ofs)) {
	    *ofsp = ofs;
	    return ra_alloc1(as, refa, allow);
	  }
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
	intptr_t ofs = (intptr_t)&gcref(fn->l.uvptr[(ir->op2 >> 8)])->uv.tv;
	intptr_t jgl = (intptr_t)J2G(as->J);
	if ((uintptr_t)(ofs-jgl) < 65536) {
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
static void asm_fusexref(ASMState *as, MIPSIns mi, Reg rt, IRRef ref,
			 RegSet allow, int32_t ofs)
{
  IRIns *ir = IR(ref);
  Reg base;
  if (ra_noreg(ir->r) && canfuse(as, ir)) {
    if (ir->o == IR_ADD) {
      intptr_t ofs2;
      if (irref_isk(ir->op2) && (ofs2 = ofs + get_kval(as, ir->op2),
				 checki16(ofs2))) {
	ref = ir->op1;
	ofs = (int32_t)ofs2;
      }
    } else if (ir->o == IR_STRREF) {
      intptr_t ofs2 = 65536;
      lj_assertA(ofs == 0, "bad usage");
      ofs = (int32_t)sizeof(GCstr);
      if (irref_isk(ir->op2)) {
	ofs2 = ofs + get_kval(as, ir->op2);
	ref = ir->op1;
      } else if (irref_isk(ir->op1)) {
	ofs2 = ofs + get_kval(as, ir->op1);
	ref = ir->op2;
      }
      if (!checki16(ofs2)) {
	/* NYI: Fuse ADD with constant. */
	Reg right, left = ra_alloc2(as, ir, allow);
	right = (left >> 8); left &= 255;
	emit_hsi(as, mi, rt, RID_TMP, ofs);
	emit_dst(as, MIPSI_AADDU, RID_TMP, left, right);
	return;
      }
      ofs = ofs2;
    }
  }
  base = ra_alloc1(as, ref, allow);
  emit_hsi(as, mi, rt, base, ofs);
}

/* -- Calls --------------------------------------------------------------- */

/* Generate a call to a C function. */
static void asm_gencall(ASMState *as, const CCallInfo *ci, IRRef *args)
{
  uint32_t n, nargs = CCI_XNARGS(ci);
  int32_t ofs = LJ_32 ? 16 : 0;
#if LJ_SOFTFP
  Reg gpr = REGARG_FIRSTGPR;
#else
  Reg gpr, fpr = REGARG_FIRSTFPR;
#endif
  if ((void *)ci->func)
    emit_call(as, (void *)ci->func, 1);
#if !LJ_SOFTFP
  for (gpr = REGARG_FIRSTGPR; gpr <= REGARG_LASTGPR; gpr++)
    as->cost[gpr] = REGCOST(~0u, ASMREF_L);
  gpr = REGARG_FIRSTGPR;
#endif
  for (n = 0; n < nargs; n++) {  /* Setup args. */
    IRRef ref = args[n];
    if (ref) {
      IRIns *ir = IR(ref);
#if !LJ_SOFTFP
      if (irt_isfp(ir->t) && fpr <= REGARG_LASTFPR &&
	  !(ci->flags & CCI_VARARG)) {
	lj_assertA(rset_test(as->freeset, fpr),
		   "reg %d not free", fpr);  /* Already evicted. */
	ra_leftov(as, fpr, ref);
	fpr += LJ_32 ? 2 : 1;
	gpr += (LJ_32 && irt_isnum(ir->t)) ? 2 : 1;
      } else
#endif
      {
#if LJ_32 && !LJ_SOFTFP
	fpr = REGARG_LASTFPR+1;
#endif
	if (LJ_32 && irt_isnum(ir->t)) gpr = (gpr+1) & ~1;
	if (gpr <= REGARG_LASTGPR) {
	  lj_assertA(rset_test(as->freeset, gpr),
		     "reg %d not free", gpr);  /* Already evicted. */
#if !LJ_SOFTFP
	  if (irt_isfp(ir->t)) {
	    RegSet of = as->freeset;
	    Reg r;
	    /* Workaround to protect argument GPRs from being used for remat. */
	    as->freeset &= ~RSET_RANGE(REGARG_FIRSTGPR, REGARG_LASTGPR+1);
	    r = ra_alloc1(as, ref, RSET_FPR);
	    as->freeset |= (of & RSET_RANGE(REGARG_FIRSTGPR, REGARG_LASTGPR+1));
	    if (irt_isnum(ir->t)) {
#if LJ_32
	      emit_tg(as, MIPSI_MFC1, gpr+(LJ_BE?0:1), r+1);
	      emit_tg(as, MIPSI_MFC1, gpr+(LJ_BE?1:0), r);
	      lj_assertA(rset_test(as->freeset, gpr+1),
			 "reg %d not free", gpr+1);  /* Already evicted. */
	      gpr += 2;
#else
	      emit_tg(as, MIPSI_DMFC1, gpr, r);
	      gpr++; fpr++;
#endif
	    } else if (irt_isfloat(ir->t)) {
	      emit_tg(as, MIPSI_MFC1, gpr, r);
	      gpr++;
#if LJ_64
	      fpr++;
#endif
	    }
	  } else
#endif
	  {
	    ra_leftov(as, gpr, ref);
	    gpr++;
#if LJ_64 && !LJ_SOFTFP
	    fpr++;
#endif
	  }
	} else {
	  Reg r = ra_alloc1z(as, ref, !LJ_SOFTFP && irt_isfp(ir->t) ? RSET_FPR : RSET_GPR);
#if LJ_32
	  if (irt_isnum(ir->t)) ofs = (ofs + 4) & ~4;
	  emit_spstore(as, ir, r, ofs);
	  ofs += irt_isnum(ir->t) ? 8 : 4;
#else
	  emit_spstore(as, ir, r, ofs + ((LJ_BE && !irt_isfp(ir->t) && !irt_is64(ir->t)) ? 4 : 0));
	  ofs += 8;
#endif
	}
      }
    } else {
#if !LJ_SOFTFP
      fpr = REGARG_LASTFPR+1;
#endif
      if (gpr <= REGARG_LASTGPR) {
	gpr++;
#if LJ_64 && !LJ_SOFTFP
	fpr++;
#endif
      } else {
	ofs += LJ_32 ? 4 : 8;
      }
    }
    checkmclim(as);
  }
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
	int32_t ofs = sps_scale(ir->s);
	Reg dest = ir->r;
	if (ra_hasreg(dest)) {
	  ra_free(as, dest);
	  ra_modified(as, dest);
#if LJ_32
	  emit_tg(as, MIPSI_MTC1, RID_RETHI, dest+1);
	  emit_tg(as, MIPSI_MTC1, RID_RETLO, dest);
#else
	  emit_tg(as, MIPSI_DMTC1, RID_RET, dest);
#endif
	}
	if (ofs) {
#if LJ_32
	  emit_tsi(as, MIPSI_SW, RID_RETLO, RID_SP, ofs+(LJ_BE?4:0));
	  emit_tsi(as, MIPSI_SW, RID_RETHI, RID_SP, ofs+(LJ_BE?0:4));
#else
	  emit_tsi(as, MIPSI_SD, RID_RET, RID_SP, ofs);
#endif
	}
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
    if (r == RID_CFUNCADDR)
      *--p = MIPSI_NOP;
    else
      *--p = MIPSI_MOVE | MIPSF_D(RID_CFUNCADDR) | MIPSF_S(r);
    *--p = MIPSI_JALR | MIPSF_S(r);
    as->mcp = p;
    ci.func = (ASMFunction)(void *)0;
  }
  asm_gencall(as, &ci, args);
}

#if !LJ_SOFTFP
static void asm_callround(ASMState *as, IRIns *ir, IRCallID id)
{
  /* The modified regs must match with the *.dasc implementation. */
  RegSet drop = RID2RSET(RID_R1)|RID2RSET(RID_R12)|RID2RSET(RID_FPRET)|
		RID2RSET(RID_F2)|RID2RSET(RID_F4)|RID2RSET(REGARG_FIRSTFPR)
#if LJ_TARGET_MIPSR6
		|RID2RSET(RID_F21)
#endif
		;
  if (ra_hasreg(ir->r)) rset_clear(drop, ir->r);
  ra_evictset(as, drop);
  ra_destreg(as, ir, RID_FPRET);
  emit_call(as, (void *)lj_ir_callinfo[id].func, 0);
  ra_leftov(as, REGARG_FIRSTFPR, ir->op1);
}
#endif

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
  asm_guard(as, MIPSI_BNE, RID_TMP,
	    ra_allock(as, igcptr(pc), rset_exclude(RSET_GPR, base)));
  emit_tsi(as, MIPSI_AL, RID_TMP, base, -8);
}

/* -- Buffer operations --------------------------------------------------- */

#if LJ_HASBUFFER
static void asm_bufhdr_write(ASMState *as, Reg sb)
{
  Reg tmp = ra_scratch(as, rset_exclude(RSET_GPR, sb));
  IRIns irgc;
  irgc.ot = IRT(0, IRT_PGC);  /* GC type. */
  emit_storeofs(as, &irgc, RID_TMP, sb, offsetof(SBuf, L));
  if ((as->flags & JIT_F_MIPSXXR2)) {
    emit_tsml(as, LJ_64 ? MIPSI_DINS : MIPSI_INS, RID_TMP, tmp,
	      lj_fls(SBUF_MASK_FLAG), 0);
  } else {
    emit_dst(as, MIPSI_OR, RID_TMP, RID_TMP, tmp);
    emit_tsi(as, MIPSI_ANDI, tmp, tmp, SBUF_MASK_FLAG);
  }
  emit_getgl(as, RID_TMP, cur_L);
  emit_loadofs(as, &irgc, tmp, sb, offsetof(SBuf, L));
}
#endif

/* -- Type conversions ---------------------------------------------------- */

#if !LJ_SOFTFP
static void asm_tointg(ASMState *as, IRIns *ir, Reg left)
{
  Reg tmp = ra_scratch(as, rset_exclude(RSET_FPR, left));
  Reg dest = ra_dest(as, ir, RSET_GPR);
#if !LJ_TARGET_MIPSR6
  asm_guard(as, MIPSI_BC1F, 0, 0);
  emit_fgh(as, MIPSI_C_EQ_D, 0, tmp, left);
#else
  asm_guard(as, MIPSI_BC1EQZ, 0, (tmp&31));
  emit_fgh(as, MIPSI_CMP_EQ_D, tmp, tmp, left);
#endif
  emit_fg(as, MIPSI_CVT_D_W, tmp, tmp);
  emit_tg(as, MIPSI_MFC1, dest, tmp);
  emit_fg(as, MIPSI_CVT_W_D, tmp, left);
}

static void asm_tobit(ASMState *as, IRIns *ir)
{
  RegSet allow = RSET_FPR;
  Reg dest = ra_dest(as, ir, RSET_GPR);
  Reg left = ra_alloc1(as, ir->op1, allow);
  Reg right = ra_alloc1(as, ir->op2, rset_clear(allow, left));
  Reg tmp = ra_scratch(as, rset_clear(allow, right));
  emit_tg(as, MIPSI_MFC1, dest, tmp);
  emit_fgh(as, MIPSI_ADD_D, tmp, left, right);
}
#elif LJ_64  /* && LJ_SOFTFP */
static void asm_tointg(ASMState *as, IRIns *ir, Reg r)
{
  /* The modified regs must match with the *.dasc implementation. */
  RegSet drop = RID2RSET(REGARG_FIRSTGPR)|RID2RSET(RID_RET)|RID2RSET(RID_RET+1)|
		RID2RSET(RID_R1)|RID2RSET(RID_R12);
  if (ra_hasreg(ir->r)) rset_clear(drop, ir->r);
  ra_evictset(as, drop);
  /* Return values are in RID_RET (converted value) and RID_RET+1 (status). */
  ra_destreg(as, ir, RID_RET);
  asm_guard(as, MIPSI_BNE, RID_RET+1, RID_ZERO);
  emit_call(as, (void *)lj_ir_callinfo[IRCALL_lj_vm_tointg].func, 0);
  if (r == RID_NONE)
    ra_leftov(as, REGARG_FIRSTGPR, ir->op1);
  else if (r != REGARG_FIRSTGPR)
    emit_move(as, REGARG_FIRSTGPR, r);
}

static void asm_tobit(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  emit_dta(as, MIPSI_SLL, dest, dest, 0);
  asm_callid(as, ir, IRCALL_lj_vm_tobit);
}
#endif

static void asm_conv(ASMState *as, IRIns *ir)
{
  IRType st = (IRType)(ir->op2 & IRCONV_SRCMASK);
#if !LJ_SOFTFP32
  int stfp = (st == IRT_NUM || st == IRT_FLOAT);
#endif
#if LJ_64
  int st64 = (st == IRT_I64 || st == IRT_U64 || st == IRT_P64);
#endif
  IRRef lref = ir->op1;
#if LJ_32
  /* 64 bit integer conversions are handled by SPLIT. */
  lj_assertA(!(irt_isint64(ir->t) || (st == IRT_I64 || st == IRT_U64)),
	     "IR %04d has unsplit 64 bit type",
	     (int)(ir - as->ir) - REF_BIAS);
#endif
#if LJ_SOFTFP32
  /* FP conversions are handled by SPLIT. */
  lj_assertA(!irt_isfp(ir->t) && !(st == IRT_NUM || st == IRT_FLOAT),
	     "IR %04d has FP type",
	     (int)(ir - as->ir) - REF_BIAS);
  /* Can't check for same types: SPLIT uses CONV int.int + BXOR for sfp NEG. */
#else
  lj_assertA(irt_type(ir->t) != st, "inconsistent types for CONV");
#if !LJ_SOFTFP
  if (irt_isfp(ir->t)) {
    Reg dest = ra_dest(as, ir, RSET_FPR);
    if (stfp) {  /* FP to FP conversion. */
      emit_fg(as, st == IRT_NUM ? MIPSI_CVT_S_D : MIPSI_CVT_D_S,
	      dest, ra_alloc1(as, lref, RSET_FPR));
    } else if (st == IRT_U32) {  /* U32 to FP conversion. */
      /* y = (x ^ 0x8000000) + 2147483648.0 */
      Reg left = ra_alloc1(as, lref, RSET_GPR);
      Reg tmp = ra_scratch(as, rset_exclude(RSET_FPR, dest));
      if (irt_isfloat(ir->t))
	emit_fg(as, MIPSI_CVT_S_D, dest, dest);
      /* Must perform arithmetic with doubles to keep the precision. */
      emit_fgh(as, MIPSI_ADD_D, dest, dest, tmp);
      emit_fg(as, MIPSI_CVT_D_W, dest, dest);
      emit_lsptr(as, MIPSI_LDC1, (tmp & 31),
		 (void *)&as->J->k64[LJ_K64_2P31], RSET_GPR);
      emit_tg(as, MIPSI_MTC1, RID_TMP, dest);
      emit_dst(as, MIPSI_XOR, RID_TMP, RID_TMP, left);
      emit_ti(as, MIPSI_LUI, RID_TMP, 0x8000);
#if LJ_64
    } else if(st == IRT_U64) {  /* U64 to FP conversion. */
      /* if (x >= 1u<<63) y = (double)(int64_t)(x&(1u<<63)-1) + pow(2.0, 63) */
      Reg left = ra_alloc1(as, lref, RSET_GPR);
      Reg tmp = ra_scratch(as, rset_exclude(RSET_FPR, dest));
      MCLabel l_end = emit_label(as);
      if (irt_isfloat(ir->t)) {
	emit_fgh(as, MIPSI_ADD_S, dest, dest, tmp);
	emit_lsptr(as, MIPSI_LWC1, (tmp & 31), (void *)&as->J->k32[LJ_K32_2P63],
		   rset_exclude(RSET_GPR, left));
	emit_fg(as, MIPSI_CVT_S_L, dest, dest);
      } else {
	emit_fgh(as, MIPSI_ADD_D, dest, dest, tmp);
	emit_lsptr(as, MIPSI_LDC1, (tmp & 31), (void *)&as->J->k64[LJ_K64_2P63],
		   rset_exclude(RSET_GPR, left));
	emit_fg(as, MIPSI_CVT_D_L, dest, dest);
      }
      emit_branch(as, MIPSI_BGEZ, left, RID_ZERO, l_end);
      emit_tg(as, MIPSI_DMTC1, RID_TMP, dest);
      emit_tsml(as, MIPSI_DEXTM, RID_TMP, left, 30, 0);
#endif
    } else {  /* Integer to FP conversion. */
      Reg left = ra_alloc1(as, lref, RSET_GPR);
#if LJ_32
      emit_fg(as, irt_isfloat(ir->t) ? MIPSI_CVT_S_W : MIPSI_CVT_D_W,
	      dest, dest);
      emit_tg(as, MIPSI_MTC1, left, dest);
#else
      MIPSIns mi = irt_isfloat(ir->t) ?
	(st64 ? MIPSI_CVT_S_L : MIPSI_CVT_S_W) :
	(st64 ? MIPSI_CVT_D_L : MIPSI_CVT_D_W);
      emit_fg(as, mi, dest, dest);
      emit_tg(as, st64 ? MIPSI_DMTC1 : MIPSI_MTC1, left, dest);
#endif
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
      if (irt_isu32(ir->t)) {  /* FP to U32 conversion. */
	/* y = (int)floor(x - 2147483648.0) ^ 0x80000000 */
	emit_dst(as, MIPSI_XOR, dest, dest, RID_TMP);
	emit_ti(as, MIPSI_LUI, RID_TMP, 0x8000);
	emit_tg(as, MIPSI_MFC1, dest, tmp);
	emit_fg(as, st == IRT_FLOAT ? MIPSI_FLOOR_W_S : MIPSI_FLOOR_W_D,
		tmp, tmp);
	emit_fgh(as, st == IRT_FLOAT ? MIPSI_SUB_S : MIPSI_SUB_D,
		 tmp, left, tmp);
	if (st == IRT_FLOAT)
	  emit_lsptr(as, MIPSI_LWC1, (tmp & 31),
		     (void *)&as->J->k32[LJ_K32_2P31], RSET_GPR);
	else
	  emit_lsptr(as, MIPSI_LDC1, (tmp & 31),
		     (void *)&as->J->k64[LJ_K64_2P31], RSET_GPR);
#if LJ_64
      } else if (irt_isu64(ir->t)) {  /* FP to U64 conversion. */
	MCLabel l_end;
	emit_tg(as, MIPSI_DMFC1, dest, tmp);
	l_end = emit_label(as);
	/* For inputs >= 2^63 add -2^64 and convert again. */
	if (st == IRT_NUM) {
	  emit_fg(as, MIPSI_TRUNC_L_D, tmp, tmp);
	  emit_fgh(as, MIPSI_ADD_D, tmp, left, tmp);
	  emit_lsptr(as, MIPSI_LDC1, (tmp & 31),
		     (void *)&as->J->k64[LJ_K64_M2P64],
		     rset_exclude(RSET_GPR, dest));
	  emit_fg(as, MIPSI_TRUNC_L_D, tmp, left);  /* Delay slot. */
#if !LJ_TARGET_MIPSR6
	 emit_branch(as, MIPSI_BC1T, 0, 0, l_end);
	 emit_fgh(as, MIPSI_C_OLT_D, 0, left, tmp);
#else
	 emit_branch(as, MIPSI_BC1NEZ, 0, (left&31), l_end);
	 emit_fgh(as, MIPSI_CMP_LT_D, left, left, tmp);
#endif
	  emit_lsptr(as, MIPSI_LDC1, (tmp & 31),
		     (void *)&as->J->k64[LJ_K64_2P63],
		     rset_exclude(RSET_GPR, dest));
	} else {
	  emit_fg(as, MIPSI_TRUNC_L_S, tmp, tmp);
	  emit_fgh(as, MIPSI_ADD_S, tmp, left, tmp);
	  emit_lsptr(as, MIPSI_LWC1, (tmp & 31),
		     (void *)&as->J->k32[LJ_K32_M2P64],
		     rset_exclude(RSET_GPR, dest));
	  emit_fg(as, MIPSI_TRUNC_L_S, tmp, left);  /* Delay slot. */
#if !LJ_TARGET_MIPSR6
	 emit_branch(as, MIPSI_BC1T, 0, 0, l_end);
	 emit_fgh(as, MIPSI_C_OLT_S, 0, left, tmp);
#else
	 emit_branch(as, MIPSI_BC1NEZ, 0, (left&31), l_end);
	 emit_fgh(as, MIPSI_CMP_LT_S, left, left, tmp);
#endif
	  emit_lsptr(as, MIPSI_LWC1, (tmp & 31),
		     (void *)&as->J->k32[LJ_K32_2P63],
		     rset_exclude(RSET_GPR, dest));
	}
#endif
      } else {
#if LJ_32
	emit_tg(as, MIPSI_MFC1, dest, tmp);
	emit_fg(as, st == IRT_FLOAT ? MIPSI_TRUNC_W_S : MIPSI_TRUNC_W_D,
		tmp, left);
#else
	MIPSIns mi = irt_is64(ir->t) ?
	  (st == IRT_NUM ? MIPSI_TRUNC_L_D : MIPSI_TRUNC_L_S) :
	  (st == IRT_NUM ? MIPSI_TRUNC_W_D : MIPSI_TRUNC_W_S);
	emit_tg(as, irt_is64(ir->t) ? MIPSI_DMFC1 : MIPSI_MFC1, dest, left);
	emit_fg(as, mi, left, left);
#endif
      }
    }
  } else
#else
  if (irt_isfp(ir->t)) {
#if LJ_64 && LJ_HASFFI
    if (stfp) {  /* FP to FP conversion. */
      asm_callid(as, ir, irt_isnum(ir->t) ? IRCALL_softfp_f2d :
					    IRCALL_softfp_d2f);
    } else {  /* Integer to FP conversion. */
      IRCallID cid = ((IRT_IS64 >> st) & 1) ?
	(irt_isnum(ir->t) ?
	 (st == IRT_I64 ? IRCALL_fp64_l2d : IRCALL_fp64_ul2d) :
	 (st == IRT_I64 ? IRCALL_fp64_l2f : IRCALL_fp64_ul2f)) :
	(irt_isnum(ir->t) ?
	 (st == IRT_INT ? IRCALL_softfp_i2d : IRCALL_softfp_ui2d) :
	 (st == IRT_INT ? IRCALL_softfp_i2f : IRCALL_softfp_ui2f));
      asm_callid(as, ir, cid);
    }
#else
    asm_callid(as, ir, IRCALL_softfp_i2d);
#endif
  } else if (stfp) {  /* FP to integer conversion. */
    if (irt_isguard(ir->t)) {
      /* Checked conversions are only supported from number to int. */
      lj_assertA(irt_isint(ir->t) && st == IRT_NUM,
		 "bad type for checked CONV");
      asm_tointg(as, ir, RID_NONE);
    } else {
      IRCallID cid = irt_is64(ir->t) ?
	((st == IRT_NUM) ?
	 (irt_isi64(ir->t) ? IRCALL_fp64_d2l : IRCALL_fp64_d2ul) :
	 (irt_isi64(ir->t) ? IRCALL_fp64_f2l : IRCALL_fp64_f2ul)) :
	((st == IRT_NUM) ?
	 (irt_isint(ir->t) ? IRCALL_softfp_d2i : IRCALL_softfp_d2ui) :
	 (irt_isint(ir->t) ? IRCALL_softfp_f2i : IRCALL_softfp_f2ui));
      asm_callid(as, ir, cid);
    }
  } else
#endif
#endif
  {
    Reg dest = ra_dest(as, ir, RSET_GPR);
    if (st >= IRT_I8 && st <= IRT_U16) {  /* Extend to 32 bit integer. */
      Reg left = ra_alloc1(as, ir->op1, RSET_GPR);
      lj_assertA(irt_isint(ir->t) || irt_isu32(ir->t), "bad type for CONV EXT");
      if ((ir->op2 & IRCONV_SEXT)) {
	if (LJ_64 || (as->flags & JIT_F_MIPSXXR2)) {
	  emit_dst(as, st == IRT_I8 ? MIPSI_SEB : MIPSI_SEH, dest, 0, left);
	} else {
	  uint32_t shift = st == IRT_I8 ? 24 : 16;
	  emit_dta(as, MIPSI_SRA, dest, dest, shift);
	  emit_dta(as, MIPSI_SLL, dest, left, shift);
	}
      } else {
	emit_tsi(as, MIPSI_ANDI, dest, left,
		 (int32_t)(st == IRT_U8 ? 0xff : 0xffff));
      }
    } else {  /* 32/64 bit integer conversions. */
#if LJ_32
      /* Only need to handle 32/32 bit no-op (cast) on 32 bit archs. */
      ra_leftov(as, dest, lref);  /* Do nothing, but may need to move regs. */
#else
      if (irt_is64(ir->t)) {
	if (st64) {
	  /* 64/64 bit no-op (cast)*/
	  ra_leftov(as, dest, lref);
	} else {
	  Reg left = ra_alloc1(as, lref, RSET_GPR);
	  if ((ir->op2 & IRCONV_SEXT)) {  /* 32 to 64 bit sign extension. */
	    emit_dta(as, MIPSI_SLL, dest, left, 0);
	  } else {  /* 32 to 64 bit zero extension. */
	    emit_tsml(as, MIPSI_DEXT, dest, left, 31, 0);
	  }
	}
      } else {
	if (st64 && !(ir->op2 & IRCONV_NONE)) {
	  /* This is either a 32 bit reg/reg mov which zeroes the hiword
	  ** or a load of the loword from a 64 bit address.
	  */
	  Reg left = ra_alloc1(as, lref, RSET_GPR);
	  emit_tsml(as, MIPSI_DEXT, dest, left, 31, 0);
	} else {  /* 32/32 bit no-op (cast). */
	  /* Do nothing, but may need to move regs. */
	  ra_leftov(as, dest, lref);
	}
      }
#endif
    }
  }
}

static void asm_strto(ASMState *as, IRIns *ir)
{
  const CCallInfo *ci = &lj_ir_callinfo[IRCALL_lj_strscan_num];
  IRRef args[2];
  int32_t ofs = 0;
#if LJ_SOFTFP32
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
      emit_tsi(as, MIPSI_LW, rhi, RID_SP, ofs+(LJ_BE?0:4));
      emit_tsi(as, MIPSI_LW, rlo, RID_SP, ofs+(LJ_BE?4:0));
    }
  }
#else
  RegSet drop = RSET_SCRATCH;
  if (ra_hasreg(ir->r)) rset_set(drop, ir->r);  /* Spill dest reg (if any). */
  ra_evictset(as, drop);
  ofs = sps_scale(ir->s);
#endif
  asm_guard(as, MIPSI_BEQ, RID_RET, RID_ZERO);  /* Test return status. */
  args[0] = ir->op1;      /* GCstr *str */
  args[1] = ASMREF_TMP1;  /* TValue *n  */
  asm_gencall(as, ci, args);
  /* Store the result to the spill slot or temp slots. */
  emit_tsi(as, MIPSI_AADDIU, ra_releasetmp(as, ASMREF_TMP1),
	   RID_SP, ofs);
}

/* -- Memory references --------------------------------------------------- */

#if LJ_64
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
    emit_tsi(as, MIPSI_SD, ra_allock(as, (int64_t)k.u64, allow), base, ofs);
  } else {
    Reg src = ra_alloc1(as, ref, allow);
    Reg type = ra_allock(as, (int64_t)irt_toitype(ir->t) << 47,
			 rset_exclude(allow, src));
    emit_tsi(as, MIPSI_SD, RID_TMP, base, ofs);
    if (irt_isinteger(ir->t)) {
      emit_dst(as, MIPSI_DADDU, RID_TMP, RID_TMP, type);
      emit_tsml(as, MIPSI_DEXT, RID_TMP, src, 31, 0);
    } else {
      emit_dst(as, MIPSI_DADDU, RID_TMP, src, type);
    }
  }
}
#endif

/* Get pointer to TValue. */
static void asm_tvptr(ASMState *as, Reg dest, IRRef ref, MSize mode)
{
  int32_t tmpofs = (int32_t)(offsetof(global_State, tmptv)-32768);
  if ((mode & IRTMPREF_IN1)) {
    IRIns *ir = IR(ref);
    if (irt_isnum(ir->t)) {
      if ((mode & IRTMPREF_OUT1)) {
#if LJ_SOFTFP
	emit_tsi(as, MIPSI_AADDIU, dest, RID_JGL, tmpofs);
#if LJ_64
	emit_setgl(as, ra_alloc1(as, ref, RSET_GPR), tmptv.u64);
#else
	lj_assertA(irref_isk(ref), "unsplit FP op");
	emit_setgl(as,
		   ra_allock(as, (int32_t)ir_knum(ir)->u32.lo, RSET_GPR),
		   tmptv.u32.lo);
	emit_setgl(as,
		   ra_allock(as, (int32_t)ir_knum(ir)->u32.hi, RSET_GPR),
		   tmptv.u32.hi);
#endif
#else
	Reg src = ra_alloc1(as, ref, RSET_FPR);
	emit_tsi(as, MIPSI_AADDIU, dest, RID_JGL, tmpofs);
	emit_tsi(as, MIPSI_SDC1, (src & 31),  RID_JGL, tmpofs);
#endif
      } else if (irref_isk(ref)) {
	/* Use the number constant itself as a TValue. */
	ra_allockreg(as, igcptr(ir_knum(ir)), dest);
      } else {
#if LJ_SOFTFP32
	lj_assertA(0, "unsplit FP op");
#else
	/* Otherwise force a spill and use the spill slot. */
	emit_tsi(as, MIPSI_AADDIU, dest, RID_SP, ra_spill(as, ir));
#endif
      }
    } else {
      /* Otherwise use g->tmptv to hold the TValue. */
#if LJ_32
      Reg type;
      emit_tsi(as, MIPSI_ADDIU, dest, RID_JGL, tmpofs);
      if (!irt_ispri(ir->t)) {
	Reg src = ra_alloc1(as, ref, RSET_GPR);
	emit_setgl(as, src, tmptv.gcr);
      }
      if (LJ_SOFTFP && (ir+1)->o == IR_HIOP && !irt_isnil((ir+1)->t))
	type = ra_alloc1(as, ref+1, RSET_GPR);
      else
	type = ra_allock(as, (int32_t)irt_toitype(ir->t), RSET_GPR);
      emit_setgl(as, type, tmptv.it);
#else
      asm_tvstore64(as, dest, 0, ref);
      emit_tsi(as, MIPSI_DADDIU, dest, RID_JGL, tmpofs);
#endif
    }
  } else {
    emit_tsi(as, MIPSI_AADDIU, dest, RID_JGL, tmpofs);
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
      emit_tsi(as, MIPSI_AADDIU, dest, base, ofs);
      return;
    }
  }
  base = ra_alloc1(as, ir->op1, RSET_GPR);
  idx = ra_alloc1(as, ir->op2, rset_exclude(RSET_GPR, base));
#if !LJ_TARGET_MIPSR6
  emit_dst(as, MIPSI_AADDU, dest, RID_TMP, base);
  emit_dta(as, MIPSI_SLL, RID_TMP, idx, 3);
#else
  emit_dst(as, MIPSI_ALSA | MIPSF_A(3-1), dest, idx, base);
#endif
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
  Reg key = RID_NONE, type = RID_NONE, tmpnum = RID_NONE, tmp1 = RID_TMP, tmp2;
#if LJ_64
  Reg cmp64 = RID_NONE;
#endif
  IRRef refkey = ir->op2;
  IRIns *irkey = IR(refkey);
  int isk = irref_isk(refkey);
  IRType1 kt = irkey->t;
  uint32_t khash;
  MCLabel l_end, l_loop, l_next;

  rset_clear(allow, tab);
  if (!LJ_SOFTFP && irt_isnum(kt)) {
    key = ra_alloc1(as, refkey, RSET_FPR);
    tmpnum = ra_scratch(as, rset_exclude(RSET_FPR, key));
  } else {
    if (!irt_ispri(kt)) {
      key = ra_alloc1(as, refkey, allow);
      rset_clear(allow, key);
    }
#if LJ_32
    if (LJ_SOFTFP && irkey[1].o == IR_HIOP) {
      if (ra_hasreg((irkey+1)->r)) {
	type = tmpnum = (irkey+1)->r;
	tmp1 = ra_scratch(as, allow);
	rset_clear(allow, tmp1);
	ra_noweak(as, tmpnum);
      } else {
	type = tmpnum = ra_allocref(as, refkey+1, allow);
      }
      rset_clear(allow, tmpnum);
    } else {
      type = ra_allock(as, (int32_t)irt_toitype(kt), allow);
      rset_clear(allow, type);
    }
#endif
  }
  tmp2 = ra_scratch(as, allow);
  rset_clear(allow, tmp2);
#if LJ_64
  if (LJ_SOFTFP || !irt_isnum(kt)) {
    /* Allocate cmp64 register used for 64-bit comparisons */
    if (LJ_SOFTFP && irt_isnum(kt)) {
      cmp64 = key;
    } else if (!isk && irt_isaddr(kt)) {
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
  }
#endif

  /* Key not found in chain: jump to exit (if merged) or load niltv. */
  l_end = emit_label(as);
  as->invmcp = NULL;
  if (merge == IR_NE)
    asm_guard(as, MIPSI_B, RID_ZERO, RID_ZERO);
  else if (destused)
    emit_loada(as, dest, niltvg(J2G(as->J)));
  /* Follow hash chain until the end. */
  emit_move(as, dest, tmp1);
  l_loop = --as->mcp;
  emit_tsi(as, MIPSI_AL, tmp1, dest, (int32_t)offsetof(Node, next));
  l_next = emit_label(as);

  /* Type and value comparison. */
  if (merge == IR_EQ) {  /* Must match asm_guard(). */
    emit_ti(as, MIPSI_LI, RID_TMP, as->snapno);
    l_end = asm_exitstub_addr(as);
  }
  if (!LJ_SOFTFP && irt_isnum(kt)) {
#if !LJ_TARGET_MIPSR6
    emit_branch(as, MIPSI_BC1T, 0, 0, l_end);
    emit_fgh(as, MIPSI_C_EQ_D, 0, tmpnum, key);
#else
    emit_branch(as, MIPSI_BC1NEZ, 0, (tmpnum&31), l_end);
    emit_fgh(as, MIPSI_CMP_EQ_D, tmpnum, tmpnum, key);
#endif
    *--as->mcp = MIPSI_NOP;  /* Avoid NaN comparison overhead. */
    emit_branch(as, MIPSI_BEQ, tmp1, RID_ZERO, l_next);
    emit_tsi(as, MIPSI_SLTIU, tmp1, tmp1, (int32_t)LJ_TISNUM);
#if LJ_32
    emit_hsi(as, MIPSI_LDC1, tmpnum, dest, (int32_t)offsetof(Node, key.n));
  } else {
    if (irt_ispri(kt)) {
      emit_branch(as, MIPSI_BEQ, tmp1, type, l_end);
    } else {
      emit_branch(as, MIPSI_BEQ, tmp2, key, l_end);
      emit_tsi(as, MIPSI_LW, tmp2, dest, (int32_t)offsetof(Node, key.gcr));
      emit_branch(as, MIPSI_BNE, tmp1, type, l_next);
    }
  }
  emit_tsi(as, MIPSI_LW, tmp1, dest, (int32_t)offsetof(Node, key.it));
  *l_loop = MIPSI_BNE | MIPSF_S(tmp1) | ((as->mcp-l_loop-1) & 0xffffu);
#else
    emit_dta(as, MIPSI_DSRA32, tmp1, tmp1, 15);
    emit_tg(as, MIPSI_DMTC1, tmp1, tmpnum);
    emit_tsi(as, MIPSI_LD, tmp1, dest, (int32_t)offsetof(Node, key.u64));
  } else {
    emit_branch(as, MIPSI_BEQ, tmp1, cmp64, l_end);
    emit_tsi(as, MIPSI_LD, tmp1, dest, (int32_t)offsetof(Node, key.u64));
  }
  *l_loop = MIPSI_BNE | MIPSF_S(tmp1) | ((as->mcp-l_loop-1) & 0xffffu);
  if (!isk && irt_isaddr(kt)) {
    type = ra_allock(as, (int64_t)irt_toitype(kt) << 47, allow);
    emit_dst(as, MIPSI_DADDU, tmp2, key, type);
    rset_clear(allow, type);
  }
#endif

  /* Load main position relative to tab->node into dest. */
  khash = isk ? ir_khash(as, irkey) : 1;
  if (khash == 0) {
    emit_tsi(as, MIPSI_AL, dest, tab, (int32_t)offsetof(GCtab, node));
  } else {
    Reg tmphash = tmp1;
    if (isk)
      tmphash = ra_allock(as, khash, allow);
    emit_dst(as, MIPSI_AADDU, dest, dest, tmp1);
    lj_assertA(sizeof(Node) == 24, "bad Node size");
    emit_dst(as, MIPSI_SUBU, tmp1, tmp2, tmp1);
    emit_dta(as, MIPSI_SLL, tmp1, tmp1, 3);
    emit_dta(as, MIPSI_SLL, tmp2, tmp1, 5);
    emit_dst(as, MIPSI_AND, tmp1, tmp2, tmphash);
    emit_tsi(as, MIPSI_AL, dest, tab, (int32_t)offsetof(GCtab, node));
    emit_tsi(as, MIPSI_LW, tmp2, tab, (int32_t)offsetof(GCtab, hmask));
    if (isk) {
      /* Nothing to do. */
    } else if (irt_isstr(kt)) {
      emit_tsi(as, MIPSI_LW, tmp1, key, (int32_t)offsetof(GCstr, sid));
    } else {  /* Must match with hash*() in lj_tab.c. */
      emit_dst(as, MIPSI_SUBU, tmp1, tmp1, tmp2);
      emit_rotr(as, tmp2, tmp2, dest, (-HASH_ROT3)&31);
      emit_dst(as, MIPSI_XOR, tmp1, tmp1, tmp2);
      emit_rotr(as, tmp1, tmp1, dest, (-HASH_ROT2-HASH_ROT1)&31);
      emit_dst(as, MIPSI_SUBU, tmp2, tmp2, dest);
#if LJ_32
      if (LJ_SOFTFP ? (irkey[1].o == IR_HIOP) : irt_isnum(kt)) {
	emit_dst(as, MIPSI_XOR, tmp2, tmp2, tmp1);
	if ((as->flags & JIT_F_MIPSXXR2)) {
	  emit_dta(as, MIPSI_ROTR, dest, tmp1, (-HASH_ROT1)&31);
	} else {
	  emit_dst(as, MIPSI_OR, dest, dest, tmp1);
	  emit_dta(as, MIPSI_SLL, tmp1, tmp1, HASH_ROT1);
	  emit_dta(as, MIPSI_SRL, dest, tmp1, (-HASH_ROT1)&31);
	}
	emit_dst(as, MIPSI_ADDU, tmp1, tmp1, tmp1);
#if LJ_SOFTFP
	emit_ds(as, MIPSI_MOVE, tmp1, type);
	emit_ds(as, MIPSI_MOVE, tmp2, key);
#else
	emit_tg(as, MIPSI_MFC1, tmp2, key);
	emit_tg(as, MIPSI_MFC1, tmp1, key+1);
#endif
      } else {
	emit_dst(as, MIPSI_XOR, tmp2, key, tmp1);
	emit_rotr(as, dest, tmp1, tmp2, (-HASH_ROT1)&31);
	emit_dst(as, MIPSI_ADDU, tmp1, key, ra_allock(as, HASH_BIAS, allow));
      }
#else
      emit_dst(as, MIPSI_XOR, tmp2, tmp2, tmp1);
      emit_dta(as, MIPSI_ROTR, dest, tmp1, (-HASH_ROT1)&31);
      if (irt_isnum(kt)) {
	emit_dst(as, MIPSI_ADDU, tmp1, tmp1, tmp1);
	emit_dta(as, MIPSI_DSRA32, tmp1, LJ_SOFTFP ? key : tmp1, 0);
	emit_dta(as, MIPSI_SLL, tmp2, LJ_SOFTFP ? key : tmp1, 0);
#if !LJ_SOFTFP
	emit_tg(as, MIPSI_DMFC1, tmp1, key);
#endif
      } else {
	checkmclim(as);
	emit_dta(as, MIPSI_DSRA32, tmp1, tmp1, 0);
	emit_dta(as, MIPSI_SLL, tmp2, key, 0);
	emit_dst(as, MIPSI_DADDU, tmp1, key, type);
      }
#endif
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
  RegSet allow = rset_exclude(RSET_GPR, node);
  Reg idx = node;
#if LJ_32
  Reg key = RID_NONE, type = RID_TMP;
  int32_t lo, hi;
#else
  Reg key = ra_scratch(as, allow);
  int64_t k;
#endif
  lj_assertA(ofs % sizeof(Node) == 0, "unaligned HREFK slot");
  if (ofs > 32736) {
    idx = dest;
    rset_clear(allow, dest);
    kofs = (int32_t)offsetof(Node, key);
  } else if (ra_hasreg(dest)) {
    emit_tsi(as, MIPSI_AADDIU, dest, node, ofs);
  }
#if LJ_32
  if (!irt_ispri(irkey->t)) {
    key = ra_scratch(as, allow);
    rset_clear(allow, key);
  }
  if (irt_isnum(irkey->t)) {
    lo = (int32_t)ir_knum(irkey)->u32.lo;
    hi = (int32_t)ir_knum(irkey)->u32.hi;
  } else {
    lo = irkey->i;
    hi = irt_toitype(irkey->t);
    if (!ra_hasreg(key))
      goto nolo;
  }
  asm_guard(as, MIPSI_BNE, key, lo ? ra_allock(as, lo, allow) : RID_ZERO);
nolo:
  asm_guard(as, MIPSI_BNE, type, hi ? ra_allock(as, hi, allow) : RID_ZERO);
  if (ra_hasreg(key)) emit_tsi(as, MIPSI_LW, key, idx, kofs+(LJ_BE?4:0));
  emit_tsi(as, MIPSI_LW, type, idx, kofs+(LJ_BE?0:4));
#else
  if (irt_ispri(irkey->t)) {
    lj_assertA(!irt_isnil(irkey->t), "bad HREFK key type");
    k = ~((int64_t)~irt_toitype(irkey->t) << 47);
  } else if (irt_isnum(irkey->t)) {
    k = (int64_t)ir_knum(irkey)->u64;
  } else {
    k = ((int64_t)irt_toitype(irkey->t) << 47) | (int64_t)ir_kgc(irkey);
  }
  asm_guard(as, MIPSI_BNE, key, ra_allock(as, k, allow));
  emit_tsi(as, MIPSI_LD, key, idx, kofs);
#endif
  if (ofs > 32736)
    emit_tsi(as, MIPSI_AADDU, dest, node, ra_allock(as, ofs, allow));
}

static void asm_uref(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  int guarded = (irt_t(ir->t) & (IRT_GUARD|IRT_TYPE)) == (IRT_GUARD|IRT_PGC);
  if (irref_isk(ir->op1) && !guarded) {
    GCfunc *fn = ir_kfunc(IR(ir->op1));
    MRef *v = &gcref(fn->l.uvptr[(ir->op2 >> 8)])->uv.v;
    emit_lsptr(as, MIPSI_AL, dest, v, RSET_GPR);
  } else {
    if (guarded)
      asm_guard(as, ir->o == IR_UREFC ? MIPSI_BEQ : MIPSI_BNE, RID_TMP, RID_ZERO);
    if (ir->o == IR_UREFC)
      emit_tsi(as, MIPSI_AADDIU, dest, dest, (int32_t)offsetof(GCupval, tv));
    else
      emit_tsi(as, MIPSI_AL, dest, dest, (int32_t)offsetof(GCupval, v));
    if (guarded)
      emit_tsi(as, MIPSI_LBU, RID_TMP, dest, (int32_t)offsetof(GCupval, closed));
    if (irref_isk(ir->op1)) {
      GCfunc *fn = ir_kfunc(IR(ir->op1));
      GCobj *o = gcref(fn->l.uvptr[(ir->op2 >> 8)]);
      emit_loada(as, dest, o);
    } else {
      emit_tsi(as, MIPSI_AL, dest, ra_alloc1(as, ir->op1, RSET_GPR),
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
#if LJ_32
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
    emit_tsi(as, MIPSI_ADDIU, dest, dest, ofs);
    emit_dst(as, MIPSI_ADDU, dest, left, right);
    return;
  }
  r = ra_alloc1(as, ref, RSET_GPR);
  ofs += IR(refk)->i;
  if (checki16(ofs))
    emit_tsi(as, MIPSI_ADDIU, dest, r, ofs);
  else
    emit_dst(as, MIPSI_ADDU, dest, r,
	     ra_allock(as, ofs, rset_exclude(RSET_GPR, r)));
#else
  RegSet allow = RSET_GPR;
  Reg dest = ra_dest(as, ir, allow);
  Reg base = ra_alloc1(as, ir->op1, allow);
  IRIns *irr = IR(ir->op2);
  int32_t ofs = sizeof(GCstr);
  rset_clear(allow, base);
  if (irref_isk(ir->op2) && checki16(ofs + irr->i)) {
    emit_tsi(as, MIPSI_DADDIU, dest, base, ofs + irr->i);
  } else {
    emit_tsi(as, MIPSI_DADDIU, dest, dest, ofs);
    emit_dst(as, MIPSI_DADDU, dest, base, ra_alloc1(as, ir->op2, allow));
  }
#endif
}

/* -- Loads and stores ---------------------------------------------------- */

static MIPSIns asm_fxloadins(ASMState *as, IRIns *ir)
{
  UNUSED(as);
  switch (irt_type(ir->t)) {
  case IRT_I8: return MIPSI_LB;
  case IRT_U8: return MIPSI_LBU;
  case IRT_I16: return MIPSI_LH;
  case IRT_U16: return MIPSI_LHU;
  case IRT_NUM:
    lj_assertA(!LJ_SOFTFP32, "unsplit FP op");
    if (!LJ_SOFTFP) return MIPSI_LDC1;
  /* fallthrough */
  case IRT_FLOAT: if (!LJ_SOFTFP) return MIPSI_LWC1;
  /* fallthrough */
  default: return (LJ_64 && irt_is64(ir->t)) ? MIPSI_LD : MIPSI_LW;
  }
}

static MIPSIns asm_fxstoreins(ASMState *as, IRIns *ir)
{
  UNUSED(as);
  switch (irt_type(ir->t)) {
  case IRT_I8: case IRT_U8: return MIPSI_SB;
  case IRT_I16: case IRT_U16: return MIPSI_SH;
  case IRT_NUM:
    lj_assertA(!LJ_SOFTFP32, "unsplit FP op");
    if (!LJ_SOFTFP) return MIPSI_SDC1;
  /* fallthrough */
  case IRT_FLOAT: if (!LJ_SOFTFP) return MIPSI_SWC1;
  /* fallthrough */
  default: return (LJ_64 && irt_is64(ir->t)) ? MIPSI_SD : MIPSI_SW;
  }
}

static void asm_fload(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  MIPSIns mi = asm_fxloadins(as, ir);
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
	emit_tsi(as, MIPSI_AADDIU, dest, idx, ofs);
	return;
      }
    }
    ofs = field_ofs[ir->op2];
    lj_assertA(!irt_isfp(ir->t), "bad FP FLOAD");
  }
  emit_tsi(as, mi, dest, idx, ofs);
}

static void asm_fstore(ASMState *as, IRIns *ir)
{
  if (ir->r != RID_SINK) {
    Reg src = ra_alloc1z(as, ir->op2, RSET_GPR);
    IRIns *irf = IR(ir->op1);
    Reg idx = ra_alloc1(as, irf->op1, rset_exclude(RSET_GPR, src));
    int32_t ofs = field_ofs[irf->op2];
    MIPSIns mi = asm_fxstoreins(as, ir);
    lj_assertA(!irt_isfp(ir->t), "bad FP FSTORE");
    emit_tsi(as, mi, src, idx, ofs);
  }
}

static void asm_xload(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir,
    (!LJ_SOFTFP && irt_isfp(ir->t)) ? RSET_FPR : RSET_GPR);
  lj_assertA(LJ_TARGET_UNALIGNED || !(ir->op2 & IRXLOAD_UNALIGNED),
	     "unaligned XLOAD");
  asm_fusexref(as, asm_fxloadins(as, ir), dest, ir->op1, RSET_GPR, 0);
}

static void asm_xstore_(ASMState *as, IRIns *ir, int32_t ofs)
{
  if (ir->r != RID_SINK) {
    Reg src = ra_alloc1z(as, ir->op2,
      (!LJ_SOFTFP && irt_isfp(ir->t)) ? RSET_FPR : RSET_GPR);
    asm_fusexref(as, asm_fxstoreins(as, ir), src, ir->op1,
		 rset_exclude(RSET_GPR, src), ofs);
  }
}

#define asm_xstore(as, ir)	asm_xstore_(as, ir, 0)

static void asm_ahuvload(ASMState *as, IRIns *ir)
{
  int hiop = (LJ_SOFTFP32 && (ir+1)->o == IR_HIOP);
  Reg dest = RID_NONE, type = RID_TMP, idx;
  RegSet allow = RSET_GPR;
  int32_t ofs = 0;
  IRType1 t = ir->t;
  if (hiop) {
    t.irt = IRT_NUM;
    if (ra_used(ir+1)) {
      type = ra_dest(as, ir+1, allow);
      rset_clear(allow, type);
    }
  }
  if (ra_used(ir)) {
    lj_assertA((LJ_SOFTFP32 ? 0 : irt_isnum(ir->t)) ||
	       irt_isint(ir->t) || irt_isaddr(ir->t),
	       "bad load type %d", irt_type(ir->t));
    dest = ra_dest(as, ir, (!LJ_SOFTFP && irt_isnum(t)) ? RSET_FPR : allow);
    rset_clear(allow, dest);
#if LJ_64
    if (irt_isaddr(t))
      emit_tsml(as, MIPSI_DEXTM, dest, dest, 14, 0);
    else if (irt_isint(t))
      emit_dta(as, MIPSI_SLL, dest, dest, 0);
#endif
  }
  idx = asm_fuseahuref(as, ir->op1, &ofs, allow);
  if (ir->o == IR_VLOAD) ofs += 8 * ir->op2;
  rset_clear(allow, idx);
  if (irt_isnum(t)) {
    asm_guard(as, MIPSI_BEQ, RID_TMP, RID_ZERO);
    emit_tsi(as, MIPSI_SLTIU, RID_TMP, type, (int32_t)LJ_TISNUM);
  } else {
    asm_guard(as, MIPSI_BNE, type,
	      ra_allock(as, (int32_t)irt_toitype(t), allow));
  }
#if LJ_32
  if (ra_hasreg(dest)) {
    if (!LJ_SOFTFP && irt_isnum(t))
      emit_hsi(as, MIPSI_LDC1, dest, idx, ofs);
    else
      emit_tsi(as, MIPSI_LW, dest, idx, ofs+(LJ_BE?4:0));
  }
  emit_tsi(as, MIPSI_LW, type, idx, ofs+(LJ_BE?0:4));
#else
  if (ra_hasreg(dest)) {
    if (!LJ_SOFTFP && irt_isnum(t)) {
      emit_hsi(as, MIPSI_LDC1, dest, idx, ofs);
      dest = type;
    }
  } else {
    dest = type;
  }
  emit_dta(as, MIPSI_DSRA32, type, dest, 15);
  emit_tsi(as, MIPSI_LD, dest, idx, ofs);
#endif
}

static void asm_ahustore(ASMState *as, IRIns *ir)
{
  RegSet allow = RSET_GPR;
  Reg idx, src = RID_NONE, type = RID_NONE;
  int32_t ofs = 0;
  if (ir->r == RID_SINK)
    return;
  if (!LJ_SOFTFP32 && irt_isnum(ir->t)) {
    src = ra_alloc1(as, ir->op2, LJ_SOFTFP ? RSET_GPR : RSET_FPR);
    idx = asm_fuseahuref(as, ir->op1, &ofs, allow);
    emit_hsi(as, LJ_SOFTFP ? MIPSI_SD : MIPSI_SDC1, src, idx, ofs);
  } else {
#if LJ_32
    if (!irt_ispri(ir->t)) {
      src = ra_alloc1(as, ir->op2, allow);
      rset_clear(allow, src);
    }
    if (LJ_SOFTFP && (ir+1)->o == IR_HIOP)
      type = ra_alloc1(as, (ir+1)->op2, allow);
    else
      type = ra_allock(as, (int32_t)irt_toitype(ir->t), allow);
    rset_clear(allow, type);
    idx = asm_fuseahuref(as, ir->op1, &ofs, allow);
    if (ra_hasreg(src))
      emit_tsi(as, MIPSI_SW, src, idx, ofs+(LJ_BE?4:0));
    emit_tsi(as, MIPSI_SW, type, idx, ofs+(LJ_BE?0:4));
#else
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
    emit_tsi(as, MIPSI_SD, tmp, idx, ofs);
    if (ra_hasreg(src)) {
      if (irt_isinteger(ir->t)) {
	emit_dst(as, MIPSI_DADDU, tmp, tmp, type);
	emit_tsml(as, MIPSI_DEXT, tmp, src, 31, 0);
      } else {
	emit_dst(as, MIPSI_DADDU, tmp, src, type);
      }
    }
#endif
  }
}

static void asm_sload(ASMState *as, IRIns *ir)
{
  Reg dest = RID_NONE, type = RID_NONE, base;
  RegSet allow = RSET_GPR;
  IRType1 t = ir->t;
#if LJ_32
  int32_t ofs = 8*((int32_t)ir->op1-1) + ((ir->op2 & IRSLOAD_FRAME) ? 4 : 0);
  int hiop = (LJ_SOFTFP32 && (ir+1)->o == IR_HIOP);
  if (hiop)
    t.irt = IRT_NUM;
#else
  int32_t ofs = 8*((int32_t)ir->op1-2);
#endif
  lj_assertA(!(ir->op2 & IRSLOAD_PARENT),
	     "bad parent SLOAD");  /* Handled by asm_head_side(). */
  lj_assertA(irt_isguard(ir->t) || !(ir->op2 & IRSLOAD_TYPECHECK),
	     "inconsistent SLOAD variant");
#if LJ_SOFTFP32
  lj_assertA(!(ir->op2 & IRSLOAD_CONVERT),
	     "unsplit SLOAD convert");  /* Handled by LJ_SOFTFP SPLIT. */
  if (hiop && ra_used(ir+1)) {
    type = ra_dest(as, ir+1, allow);
    rset_clear(allow, type);
  }
#else
  if ((ir->op2 & IRSLOAD_CONVERT) && irt_isguard(t) && irt_isint(t)) {
    dest = ra_scratch(as, LJ_SOFTFP ? allow : RSET_FPR);
    asm_tointg(as, ir, dest);
    t.irt = IRT_NUM;  /* Continue with a regular number type check. */
  } else
#endif
  if (ra_used(ir)) {
    lj_assertA((LJ_SOFTFP32 ? 0 : irt_isnum(ir->t)) ||
	       irt_isint(ir->t) || irt_isaddr(ir->t),
	       "bad SLOAD type %d", irt_type(ir->t));
    dest = ra_dest(as, ir, (!LJ_SOFTFP && irt_isnum(t)) ? RSET_FPR : allow);
    rset_clear(allow, dest);
    base = ra_alloc1(as, REF_BASE, allow);
    rset_clear(allow, base);
    if (!LJ_SOFTFP32 && (ir->op2 & IRSLOAD_CONVERT)) {
      if (irt_isint(t)) {
	Reg tmp = ra_scratch(as, LJ_SOFTFP ? RSET_GPR : RSET_FPR);
#if LJ_SOFTFP
	ra_evictset(as, rset_exclude(RSET_SCRATCH, dest));
	ra_destreg(as, ir, RID_RET);
	emit_call(as, (void *)lj_ir_callinfo[IRCALL_softfp_d2i].func, 0);
	if (tmp != REGARG_FIRSTGPR)
	  emit_move(as, REGARG_FIRSTGPR, tmp);
#else
	emit_tg(as, MIPSI_MFC1, dest, tmp);
	emit_fg(as, MIPSI_TRUNC_W_D, tmp, tmp);
#endif
	dest = tmp;
	t.irt = IRT_NUM;  /* Check for original type. */
      } else {
	Reg tmp = ra_scratch(as, RSET_GPR);
#if LJ_SOFTFP
	ra_evictset(as, rset_exclude(RSET_SCRATCH, dest));
	ra_destreg(as, ir, RID_RET);
	emit_call(as, (void *)lj_ir_callinfo[IRCALL_softfp_i2d].func, 0);
	emit_dta(as, MIPSI_SLL, REGARG_FIRSTGPR, tmp, 0);
#else
	emit_fg(as, MIPSI_CVT_D_W, dest, dest);
	emit_tg(as, MIPSI_MTC1, tmp, dest);
#endif
	dest = tmp;
	t.irt = IRT_INT;  /* Check for original type. */
      }
    }
#if LJ_64
    else if (irt_isaddr(t)) {
      /* Clear type from pointers. */
      emit_tsml(as, MIPSI_DEXTM, dest, dest, 14, 0);
    } else if (irt_isint(t) && (ir->op2 & IRSLOAD_TYPECHECK)) {
      /* Sign-extend integers. */
      emit_dta(as, MIPSI_SLL, dest, dest, 0);
    }
#endif
    goto dotypecheck;
  }
  base = ra_alloc1(as, REF_BASE, allow);
  rset_clear(allow, base);
dotypecheck:
#if LJ_32
  if ((ir->op2 & IRSLOAD_TYPECHECK)) {
    if (ra_noreg(type))
      type = RID_TMP;
    if (irt_isnum(t)) {
      asm_guard(as, MIPSI_BEQ, RID_TMP, RID_ZERO);
      emit_tsi(as, MIPSI_SLTIU, RID_TMP, type, (int32_t)LJ_TISNUM);
    } else {
      Reg ktype = ra_allock(as, (ir->op2 & IRSLOAD_KEYINDEX) ? LJ_KEYINDEX : irt_toitype(t), allow);
      asm_guard(as, MIPSI_BNE, type, ktype);
    }
  }
  if (ra_hasreg(dest)) {
    if (!LJ_SOFTFP && irt_isnum(t))
      emit_hsi(as, MIPSI_LDC1, dest, base, ofs);
    else
      emit_tsi(as, MIPSI_LW, dest, base, ofs ^ (LJ_BE?4:0));
  }
  if (ra_hasreg(type))
    emit_tsi(as, MIPSI_LW, type, base, ofs ^ (LJ_BE?0:4));
#else
  if ((ir->op2 & IRSLOAD_TYPECHECK)) {
    type = dest < RID_MAX_GPR ? dest : RID_TMP;
    if (irt_ispri(t)) {
      asm_guard(as, MIPSI_BNE, type,
		ra_allock(as, ~((int64_t)~irt_toitype(t) << 47) , allow));
    } else if ((ir->op2 & IRSLOAD_KEYINDEX)) {
      asm_guard(as, MIPSI_BNE, RID_TMP,
		ra_allock(as, (int32_t)LJ_KEYINDEX, allow));
      emit_dta(as, MIPSI_DSRA32, RID_TMP, type, 0);
    } else {
      if (irt_isnum(t)) {
	asm_guard(as, MIPSI_BEQ, RID_TMP, RID_ZERO);
	emit_tsi(as, MIPSI_SLTIU, RID_TMP, RID_TMP, (int32_t)LJ_TISNUM);
	if (!LJ_SOFTFP && ra_hasreg(dest))
	  emit_hsi(as, MIPSI_LDC1, dest, base, ofs);
      } else {
	asm_guard(as, MIPSI_BNE, RID_TMP,
		  ra_allock(as, (int32_t)irt_toitype(t), allow));
      }
      emit_dta(as, MIPSI_DSRA32, RID_TMP, type, 15);
    }
    emit_tsi(as, MIPSI_LD, type, base, ofs);
  } else if (ra_hasreg(dest)) {
    if (!LJ_SOFTFP && irt_isnum(t))
      emit_hsi(as, MIPSI_LDC1, dest, base, ofs);
    else
      emit_tsi(as, irt_isint(t) ? MIPSI_LW : MIPSI_LD, dest, base,
	       ofs ^ ((LJ_BE && irt_isint(t)) ? 4 : 0));
  }
#endif
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
#if LJ_32
    int32_t ofs = sizeof(GCcdata);
    if (sz == 8) {
      ofs += 4;
      lj_assertA((ir+1)->o == IR_HIOP, "expected HIOP for CNEWI");
      if (LJ_LE) ir++;
    }
    for (;;) {
      Reg r = ra_alloc1z(as, ir->op2, allow);
      emit_tsi(as, MIPSI_SW, r, RID_RET, ofs);
      rset_clear(allow, r);
      if (ofs == sizeof(GCcdata)) break;
      ofs -= 4; if (LJ_BE) ir++; else ir--;
    }
#else
    emit_tsi(as, sz == 8 ? MIPSI_SD : MIPSI_SW, ra_alloc1(as, ir->op2, allow),
	     RID_RET, sizeof(GCcdata));
#endif
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
  emit_tsi(as, MIPSI_SB, RID_RET+1, RID_RET, offsetof(GCcdata, gct));
  emit_tsi(as, MIPSI_SH, RID_TMP, RID_RET, offsetof(GCcdata, ctypeid));
  emit_ti(as, MIPSI_LI, RID_RET+1, ~LJ_TCDATA);
  emit_ti(as, MIPSI_LI, RID_TMP, id); /* Lower 16 bit used. Sign-ext ok. */
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
  emit_tsi(as, MIPSI_AS, link, tab, (int32_t)offsetof(GCtab, gclist));
  emit_tsi(as, MIPSI_SB, mark, tab, (int32_t)offsetof(GCtab, marked));
  emit_setgl(as, tab, gc.grayagain);
  emit_getgl(as, link, gc.grayagain);
  emit_dst(as, MIPSI_XOR, mark, mark, RID_TMP);  /* Clear black bit. */
  emit_branch(as, MIPSI_BEQ, RID_TMP, RID_ZERO, l_end);
  emit_tsi(as, MIPSI_ANDI, RID_TMP, mark, LJ_GC_BLACK);
  emit_tsi(as, MIPSI_LBU, mark, tab, (int32_t)offsetof(GCtab, marked));
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
  emit_tsi(as, MIPSI_AADDIU, ra_releasetmp(as, ASMREF_TMP1), RID_JGL, -32768);
  obj = IR(ir->op1)->r;
  tmp = ra_scratch(as, rset_exclude(RSET_GPR, obj));
  emit_branch(as, MIPSI_BEQ, RID_TMP, RID_ZERO, l_end);
  emit_tsi(as, MIPSI_ANDI, tmp, tmp, LJ_GC_BLACK);
  emit_branch(as, MIPSI_BEQ, RID_TMP, RID_ZERO, l_end);
  emit_tsi(as, MIPSI_ANDI, RID_TMP, RID_TMP, LJ_GC_WHITES);
  val = ra_alloc1(as, ir->op2, rset_exclude(RSET_GPR, obj));
  emit_tsi(as, MIPSI_LBU, tmp, obj,
	   (int32_t)offsetof(GCupval, marked)-(int32_t)offsetof(GCupval, tv));
  emit_tsi(as, MIPSI_LBU, RID_TMP, val, (int32_t)offsetof(GChead, marked));
}

/* -- Arithmetic and logic operations ------------------------------------- */

#if !LJ_SOFTFP
static void asm_fparith(ASMState *as, IRIns *ir, MIPSIns mi)
{
  Reg dest = ra_dest(as, ir, RSET_FPR);
  Reg right, left = ra_alloc2(as, ir, RSET_FPR);
  right = (left >> 8); left &= 255;
  emit_fgh(as, mi, dest, left, right);
}

static void asm_fpunary(ASMState *as, IRIns *ir, MIPSIns mi)
{
  Reg dest = ra_dest(as, ir, RSET_FPR);
  Reg left = ra_hintalloc(as, ir->op1, dest, RSET_FPR);
  emit_fg(as, mi, dest, left);
}
#endif

#if !LJ_SOFTFP32
static void asm_fpmath(ASMState *as, IRIns *ir)
{
#if !LJ_SOFTFP
  if (ir->op2 <= IRFPM_TRUNC)
    asm_callround(as, ir, IRCALL_lj_vm_floor + ir->op2);
  else if (ir->op2 == IRFPM_SQRT)
    asm_fpunary(as, ir, MIPSI_SQRT_D);
  else
#endif
    asm_callid(as, ir, IRCALL_lj_vm_floor + ir->op2);
}
#endif

#if !LJ_SOFTFP
#define asm_fpadd(as, ir)	asm_fparith(as, ir, MIPSI_ADD_D)
#define asm_fpsub(as, ir)	asm_fparith(as, ir, MIPSI_SUB_D)
#define asm_fpmul(as, ir)	asm_fparith(as, ir, MIPSI_MUL_D)
#elif LJ_64  /* && LJ_SOFTFP */
#define asm_fpadd(as, ir)	asm_callid(as, ir, IRCALL_softfp_add)
#define asm_fpsub(as, ir)	asm_callid(as, ir, IRCALL_softfp_sub)
#define asm_fpmul(as, ir)	asm_callid(as, ir, IRCALL_softfp_mul)
#endif

static void asm_add(ASMState *as, IRIns *ir)
{
  IRType1 t = ir->t;
#if !LJ_SOFTFP32
  if (irt_isnum(t)) {
    asm_fpadd(as, ir);
  } else
#endif
  {
    /* TODO MIPSR6: Fuse ADD(BSHL(a,1-4),b) or ADD(ADD(a,a),b) to MIPSI_ALSA. */
    Reg dest = ra_dest(as, ir, RSET_GPR);
    Reg right, left = ra_hintalloc(as, ir->op1, dest, RSET_GPR);
    if (irref_isk(ir->op2)) {
      intptr_t k = get_kval(as, ir->op2);
      if (checki16(k)) {
	emit_tsi(as, (LJ_64 && irt_is64(t)) ? MIPSI_DADDIU : MIPSI_ADDIU, dest,
		 left, k);
	return;
      }
    }
    right = ra_alloc1(as, ir->op2, rset_exclude(RSET_GPR, left));
    emit_dst(as, (LJ_64 && irt_is64(t)) ? MIPSI_DADDU : MIPSI_ADDU, dest,
	     left, right);
  }
}

static void asm_sub(ASMState *as, IRIns *ir)
{
#if !LJ_SOFTFP32
  if (irt_isnum(ir->t)) {
    asm_fpsub(as, ir);
  } else
#endif
  {
    Reg dest = ra_dest(as, ir, RSET_GPR);
    Reg right, left = ra_alloc2(as, ir, RSET_GPR);
    right = (left >> 8); left &= 255;
    emit_dst(as, (LJ_64 && irt_is64(ir->t)) ? MIPSI_DSUBU : MIPSI_SUBU, dest,
	     left, right);
  }
}

static void asm_mul(ASMState *as, IRIns *ir)
{
#if !LJ_SOFTFP32
  if (irt_isnum(ir->t)) {
    asm_fpmul(as, ir);
  } else
#endif
  {
    Reg dest = ra_dest(as, ir, RSET_GPR);
    Reg right, left = ra_alloc2(as, ir, RSET_GPR);
    right = (left >> 8); left &= 255;
    if (LJ_64 && irt_is64(ir->t)) {
#if !LJ_TARGET_MIPSR6
      emit_dst(as, MIPSI_MFLO, dest, 0, 0);
      emit_dst(as, MIPSI_DMULT, 0, left, right);
#else
      emit_dst(as, MIPSI_DMUL, dest, left, right);
#endif
    } else {
      emit_dst(as, MIPSI_MUL, dest, left, right);
    }
  }
}

#if !LJ_SOFTFP32
static void asm_fpdiv(ASMState *as, IRIns *ir)
{
#if !LJ_SOFTFP
    asm_fparith(as, ir, MIPSI_DIV_D);
#else
    asm_callid(as, ir, IRCALL_softfp_div);
#endif
}
#endif

static void asm_neg(ASMState *as, IRIns *ir)
{
#if !LJ_SOFTFP
  if (irt_isnum(ir->t)) {
    asm_fpunary(as, ir, MIPSI_NEG_D);
  } else
#elif LJ_64  /* && LJ_SOFTFP */
  if (irt_isnum(ir->t)) {
    Reg dest = ra_dest(as, ir, RSET_GPR);
    Reg left = ra_hintalloc(as, ir->op1, dest, RSET_GPR);
    emit_dst(as, MIPSI_XOR, dest, left,
	    ra_allock(as, 0x8000000000000000ll, rset_exclude(RSET_GPR, dest)));
  } else
#endif
  {
    Reg dest = ra_dest(as, ir, RSET_GPR);
    Reg left = ra_hintalloc(as, ir->op1, dest, RSET_GPR);
    emit_dst(as, (LJ_64 && irt_is64(ir->t)) ? MIPSI_DSUBU : MIPSI_SUBU, dest,
	     RID_ZERO, left);
  }
}

#if !LJ_SOFTFP
#define asm_abs(as, ir)		asm_fpunary(as, ir, MIPSI_ABS_D)
#elif LJ_64   /* && LJ_SOFTFP */
static void asm_abs(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  Reg left = ra_alloc1(as, ir->op1, RSET_GPR);
  emit_tsml(as, MIPSI_DEXTM, dest, left, 30, 0);
}
#endif

static void asm_arithov(ASMState *as, IRIns *ir)
{
  /* TODO MIPSR6: bovc/bnvc. Caveat: no delay slot to load RID_TMP. */
  Reg right, left, tmp, dest = ra_dest(as, ir, RSET_GPR);
  lj_assertA(!irt_is64(ir->t), "bad usage");
  if (irref_isk(ir->op2)) {
    int k = IR(ir->op2)->i;
    if (ir->o == IR_SUBOV) k = (int)(~(unsigned int)k+1u);
    if (checki16(k)) {  /* (dest < left) == (k >= 0 ? 1 : 0) */
      left = ra_alloc1(as, ir->op1, RSET_GPR);
      asm_guard(as, k >= 0 ? MIPSI_BNE : MIPSI_BEQ, RID_TMP, RID_ZERO);
      emit_dst(as, MIPSI_SLT, RID_TMP, dest, dest == left ? RID_TMP : left);
      emit_tsi(as, MIPSI_ADDIU, dest, left, k);
      if (dest == left) emit_move(as, RID_TMP, left);
      return;
    }
  }
  left = ra_alloc2(as, ir, RSET_GPR);
  right = (left >> 8); left &= 255;
  tmp = ra_scratch(as, rset_exclude(rset_exclude(rset_exclude(RSET_GPR, left),
						 right), dest));
  asm_guard(as, MIPSI_BLTZ, RID_TMP, 0);
  emit_dst(as, MIPSI_AND, RID_TMP, RID_TMP, tmp);
  if (ir->o == IR_ADDOV) {  /* ((dest^left) & (dest^right)) < 0 */
    emit_dst(as, MIPSI_XOR, RID_TMP, dest, dest == right ? RID_TMP : right);
  } else {  /* ((dest^left) & (dest^~right)) < 0 */
    emit_dst(as, MIPSI_XOR, RID_TMP, RID_TMP, dest);
    emit_dst(as, MIPSI_NOR, RID_TMP, dest == right ? RID_TMP : right, RID_ZERO);
  }
  emit_dst(as, MIPSI_XOR, tmp, dest, dest == left ? RID_TMP : left);
  emit_dst(as, ir->o == IR_ADDOV ? MIPSI_ADDU : MIPSI_SUBU, dest, left, right);
  if (dest == left || dest == right)
    emit_move(as, RID_TMP, dest == left ? left : right);
}

#define asm_addov(as, ir)	asm_arithov(as, ir)
#define asm_subov(as, ir)	asm_arithov(as, ir)

static void asm_mulov(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  Reg tmp, right, left = ra_alloc2(as, ir, RSET_GPR);
  right = (left >> 8); left &= 255;
  tmp = ra_scratch(as, rset_exclude(rset_exclude(rset_exclude(RSET_GPR, left),
						 right), dest));
  asm_guard(as, MIPSI_BNE, RID_TMP, tmp);
  emit_dta(as, MIPSI_SRA, RID_TMP, dest, 31);
#if !LJ_TARGET_MIPSR6
  emit_dst(as, MIPSI_MFHI, tmp, 0, 0);
  emit_dst(as, MIPSI_MFLO, dest, 0, 0);
  emit_dst(as, MIPSI_MULT, 0, left, right);
#else
  emit_dst(as, MIPSI_MUL, dest, left, right);
  emit_dst(as, MIPSI_MUH, tmp, left, right);
#endif
}

#if LJ_32 && LJ_HASFFI
static void asm_add64(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  Reg right, left = ra_alloc1(as, ir->op1, RSET_GPR);
  if (irref_isk(ir->op2)) {
    int32_t k = IR(ir->op2)->i;
    if (k == 0) {
      emit_dst(as, MIPSI_ADDU, dest, left, RID_TMP);
      goto loarith;
    } else if (checki16(k)) {
      emit_dst(as, MIPSI_ADDU, dest, dest, RID_TMP);
      emit_tsi(as, MIPSI_ADDIU, dest, left, k);
      goto loarith;
    }
  }
  emit_dst(as, MIPSI_ADDU, dest, dest, RID_TMP);
  right = ra_alloc1(as, ir->op2, rset_exclude(RSET_GPR, left));
  emit_dst(as, MIPSI_ADDU, dest, left, right);
loarith:
  ir--;
  dest = ra_dest(as, ir, RSET_GPR);
  left = ra_alloc1(as, ir->op1, RSET_GPR);
  if (irref_isk(ir->op2)) {
    int32_t k = IR(ir->op2)->i;
    if (k == 0) {
      if (dest != left)
	emit_move(as, dest, left);
      return;
    } else if (checki16(k)) {
      if (dest == left) {
	Reg tmp = ra_scratch(as, rset_exclude(RSET_GPR, left));
	emit_move(as, dest, tmp);
	dest = tmp;
      }
      emit_dst(as, MIPSI_SLTU, RID_TMP, dest, left);
      emit_tsi(as, MIPSI_ADDIU, dest, left, k);
      return;
    }
  }
  right = ra_alloc1(as, ir->op2, rset_exclude(RSET_GPR, left));
  if (dest == left && dest == right) {
    Reg tmp = ra_scratch(as, rset_exclude(rset_exclude(RSET_GPR, left), right));
    emit_move(as, dest, tmp);
    dest = tmp;
  }
  emit_dst(as, MIPSI_SLTU, RID_TMP, dest, dest == left ? right : left);
  emit_dst(as, MIPSI_ADDU, dest, left, right);
}

static void asm_sub64(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  Reg right, left = ra_alloc2(as, ir, RSET_GPR);
  right = (left >> 8); left &= 255;
  emit_dst(as, MIPSI_SUBU, dest, dest, RID_TMP);
  emit_dst(as, MIPSI_SUBU, dest, left, right);
  ir--;
  dest = ra_dest(as, ir, RSET_GPR);
  left = ra_alloc2(as, ir, RSET_GPR);
  right = (left >> 8); left &= 255;
  if (dest == left) {
    Reg tmp = ra_scratch(as, rset_exclude(rset_exclude(RSET_GPR, left), right));
    emit_move(as, dest, tmp);
    dest = tmp;
  }
  emit_dst(as, MIPSI_SLTU, RID_TMP, left, dest);
  emit_dst(as, MIPSI_SUBU, dest, left, right);
}

static void asm_neg64(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  Reg left = ra_alloc1(as, ir->op1, RSET_GPR);
  emit_dst(as, MIPSI_SUBU, dest, dest, RID_TMP);
  emit_dst(as, MIPSI_SUBU, dest, RID_ZERO, left);
  ir--;
  dest = ra_dest(as, ir, RSET_GPR);
  left = ra_alloc1(as, ir->op1, RSET_GPR);
  emit_dst(as, MIPSI_SLTU, RID_TMP, RID_ZERO, dest);
  emit_dst(as, MIPSI_SUBU, dest, RID_ZERO, left);
}
#endif

static void asm_bnot(ASMState *as, IRIns *ir)
{
  Reg left, right, dest = ra_dest(as, ir, RSET_GPR);
  IRIns *irl = IR(ir->op1);
  if (mayfuse(as, ir->op1) && irl->o == IR_BOR) {
    left = ra_alloc2(as, irl, RSET_GPR);
    right = (left >> 8); left &= 255;
  } else {
    left = ra_hintalloc(as, ir->op1, dest, RSET_GPR);
    right = RID_ZERO;
  }
  emit_dst(as, MIPSI_NOR, dest, left, right);
}

static void asm_bswap(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  Reg left = ra_alloc1(as, ir->op1, RSET_GPR);
#if LJ_32
  if ((as->flags & JIT_F_MIPSXXR2)) {
    emit_dta(as, MIPSI_ROTR, dest, RID_TMP, 16);
    emit_dst(as, MIPSI_WSBH, RID_TMP, 0, left);
  } else {
    Reg tmp = ra_scratch(as, rset_exclude(rset_exclude(RSET_GPR, left), dest));
    emit_dst(as, MIPSI_OR, dest, dest, tmp);
    emit_dst(as, MIPSI_OR, dest, dest, RID_TMP);
    emit_tsi(as, MIPSI_ANDI, dest, dest, 0xff00);
    emit_dta(as, MIPSI_SLL, RID_TMP, RID_TMP, 8);
    emit_dta(as, MIPSI_SRL, dest, left, 8);
    emit_tsi(as, MIPSI_ANDI, RID_TMP, left, 0xff00);
    emit_dst(as, MIPSI_OR, tmp, tmp, RID_TMP);
    emit_dta(as, MIPSI_SRL, tmp, left, 24);
    emit_dta(as, MIPSI_SLL, RID_TMP, left, 24);
  }
#else
  if (irt_is64(ir->t)) {
    emit_dst(as, MIPSI_DSHD, dest, 0, RID_TMP);
    emit_dst(as, MIPSI_DSBH, RID_TMP, 0, left);
  } else {
    emit_dta(as, MIPSI_ROTR, dest, RID_TMP, 16);
    emit_dst(as, MIPSI_WSBH, RID_TMP, 0, left);
  }
#endif
}

static void asm_bitop(ASMState *as, IRIns *ir, MIPSIns mi, MIPSIns mik)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  Reg right, left = ra_hintalloc(as, ir->op1, dest, RSET_GPR);
  if (irref_isk(ir->op2)) {
    intptr_t k = get_kval(as, ir->op2);
    if (checku16(k)) {
      emit_tsi(as, mik, dest, left, k);
      return;
    }
  }
  right = ra_alloc1(as, ir->op2, rset_exclude(RSET_GPR, left));
  emit_dst(as, mi, dest, left, right);
}

#define asm_band(as, ir)	asm_bitop(as, ir, MIPSI_AND, MIPSI_ANDI)
#define asm_bor(as, ir)		asm_bitop(as, ir, MIPSI_OR, MIPSI_ORI)
#define asm_bxor(as, ir)	asm_bitop(as, ir, MIPSI_XOR, MIPSI_XORI)

static void asm_bitshift(ASMState *as, IRIns *ir, MIPSIns mi, MIPSIns mik)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  if (irref_isk(ir->op2)) {  /* Constant shifts. */
    uint32_t shift = (uint32_t)IR(ir->op2)->i;
    if (LJ_64 && irt_is64(ir->t)) mik |= (shift & 32) ? MIPSI_D32 : MIPSI_D;
    emit_dta(as, mik, dest, ra_hintalloc(as, ir->op1, dest, RSET_GPR),
	     (shift & 31));
  } else {
    Reg right, left = ra_alloc2(as, ir, RSET_GPR);
    right = (left >> 8); left &= 255;
    if (LJ_64 && irt_is64(ir->t)) mi |= MIPSI_DV;
    emit_dst(as, mi, dest, right, left);  /* Shift amount is in rs. */
  }
}

#define asm_bshl(as, ir)	asm_bitshift(as, ir, MIPSI_SLLV, MIPSI_SLL)
#define asm_bshr(as, ir)	asm_bitshift(as, ir, MIPSI_SRLV, MIPSI_SRL)
#define asm_bsar(as, ir)	asm_bitshift(as, ir, MIPSI_SRAV, MIPSI_SRA)
#define asm_brol(as, ir)	lj_assertA(0, "unexpected BROL")

static void asm_bror(ASMState *as, IRIns *ir)
{
  if (LJ_64 || (as->flags & JIT_F_MIPSXXR2)) {
    asm_bitshift(as, ir, MIPSI_ROTRV, MIPSI_ROTR);
  } else {
    Reg dest = ra_dest(as, ir, RSET_GPR);
    if (irref_isk(ir->op2)) {  /* Constant shifts. */
      uint32_t shift = (uint32_t)(IR(ir->op2)->i & 31);
      Reg left = ra_hintalloc(as, ir->op1, dest, RSET_GPR);
      emit_rotr(as, dest, left, RID_TMP, shift);
    } else {
      Reg right, left = ra_alloc2(as, ir, RSET_GPR);
      right = (left >> 8); left &= 255;
      emit_dst(as, MIPSI_OR, dest, dest, RID_TMP);
      emit_dst(as, MIPSI_SRLV, dest, right, left);
      emit_dst(as, MIPSI_SLLV, RID_TMP, RID_TMP, left);
      emit_dst(as, MIPSI_SUBU, RID_TMP, ra_allock(as, 32, RSET_GPR), right);
    }
  }
}

#if LJ_SOFTFP
static void asm_sfpmin_max(ASMState *as, IRIns *ir)
{
  CCallInfo ci = lj_ir_callinfo[(IROp)ir->o == IR_MIN ? IRCALL_lj_vm_sfmin : IRCALL_lj_vm_sfmax];
#if LJ_64
  IRRef args[2];
  args[0] = ir->op1;
  args[1] = ir->op2;
#else
  IRRef args[4];
  args[0^LJ_BE] = ir->op1;
  args[1^LJ_BE] = (ir+1)->op1;
  args[2^LJ_BE] = ir->op2;
  args[3^LJ_BE] = (ir+1)->op2;
#endif
  asm_setupresult(as, ir, &ci);
  emit_call(as, (void *)ci.func, 0);
  ci.func = NULL;
  asm_gencall(as, &ci, args);
}
#endif

static void asm_min_max(ASMState *as, IRIns *ir, int ismax)
{
  if (!LJ_SOFTFP32 && irt_isnum(ir->t)) {
#if LJ_SOFTFP
    asm_sfpmin_max(as, ir);
#else
    Reg dest = ra_dest(as, ir, RSET_FPR);
    Reg right, left = ra_alloc2(as, ir, RSET_FPR);
    right = (left >> 8); left &= 255;
#if !LJ_TARGET_MIPSR6
    if (dest == left) {
      emit_fg(as, MIPSI_MOVF_D, dest, right);
    } else {
      emit_fg(as, MIPSI_MOVT_D, dest, left);
      if (dest != right) emit_fg(as, MIPSI_MOV_D, dest, right);
    }
    emit_fgh(as, MIPSI_C_OLT_D, 0, ismax ? right : left, ismax ? left : right);
#else
    emit_fgh(as, ismax ? MIPSI_MAX_D : MIPSI_MIN_D, dest, left, right);
#endif
#endif
  } else {
    Reg dest = ra_dest(as, ir, RSET_GPR);
    Reg right, left = ra_alloc2(as, ir, RSET_GPR);
    right = (left >> 8); left &= 255;
    if (left == right) {
      if (dest != left) emit_move(as, dest, left);
    } else {
#if !LJ_TARGET_MIPSR6
      if (dest == left) {
	emit_dst(as, MIPSI_MOVN, dest, right, RID_TMP);
      } else {
	emit_dst(as, MIPSI_MOVZ, dest, left, RID_TMP);
	if (dest != right) emit_move(as, dest, right);
      }
#else
      emit_dst(as, MIPSI_OR, dest, dest, RID_TMP);
      if (dest != right) {
	emit_dst(as, MIPSI_SELNEZ, RID_TMP, right, RID_TMP);
	emit_dst(as, MIPSI_SELEQZ, dest, left, RID_TMP);
      } else {
	emit_dst(as, MIPSI_SELEQZ, RID_TMP, left, RID_TMP);
	emit_dst(as, MIPSI_SELNEZ, dest, right, RID_TMP);
      }
#endif
      emit_dst(as, MIPSI_SLT, RID_TMP,
	       ismax ? left : right, ismax ? right : left);
    }
  }
}

#define asm_min(as, ir)		asm_min_max(as, ir, 0)
#define asm_max(as, ir)		asm_min_max(as, ir, 1)

/* -- Comparisons --------------------------------------------------------- */

#if LJ_SOFTFP
/* SFP comparisons. */
static void asm_sfpcomp(ASMState *as, IRIns *ir)
{
  const CCallInfo *ci = &lj_ir_callinfo[IRCALL_softfp_cmp];
  RegSet drop = RSET_SCRATCH;
  Reg r;
#if LJ_64
  IRRef args[2];
  args[0] = ir->op1;
  args[1] = ir->op2;
#else
  IRRef args[4];
  args[LJ_LE ? 0 : 1] = ir->op1; args[LJ_LE ? 1 : 0] = (ir+1)->op1;
  args[LJ_LE ? 2 : 3] = ir->op2; args[LJ_LE ? 3 : 2] = (ir+1)->op2;
#endif

  for (r = REGARG_FIRSTGPR; r <= REGARG_FIRSTGPR+(LJ_64?1:3); r++) {
    if (!rset_test(as->freeset, r) &&
	regcost_ref(as->cost[r]) == args[r-REGARG_FIRSTGPR])
      rset_clear(drop, r);
  }
  ra_evictset(as, drop);

  asm_setupresult(as, ir, ci);

  switch ((IROp)ir->o) {
  case IR_LT:
    asm_guard(as, MIPSI_BGEZ, RID_RET, 0);
    break;
  case IR_ULT:
    asm_guard(as, MIPSI_BEQ, RID_RET, RID_TMP);
    emit_loadi(as, RID_TMP, 1);
    asm_guard(as, MIPSI_BEQ, RID_RET, RID_ZERO);
    break;
  case IR_GE:
    asm_guard(as, MIPSI_BEQ, RID_RET, RID_TMP);
    emit_loadi(as, RID_TMP, 2);
    asm_guard(as, MIPSI_BLTZ, RID_RET, 0);
    break;
  case IR_LE:
    asm_guard(as, MIPSI_BGTZ, RID_RET, 0);
    break;
  case IR_GT:
    asm_guard(as, MIPSI_BEQ, RID_RET, RID_TMP);
    emit_loadi(as, RID_TMP, 2);
    asm_guard(as, MIPSI_BLEZ, RID_RET, 0);
    break;
  case IR_UGE:
    asm_guard(as, MIPSI_BLTZ, RID_RET, 0);
    break;
  case IR_ULE:
    asm_guard(as, MIPSI_BEQ, RID_RET, RID_TMP);
    emit_loadi(as, RID_TMP, 1);
    break;
  case IR_UGT: case IR_ABC:
    asm_guard(as, MIPSI_BLEZ, RID_RET, 0);
    break;
  case IR_EQ: case IR_NE:
    asm_guard(as, (ir->o & 1) ? MIPSI_BEQ : MIPSI_BNE, RID_RET, RID_ZERO);
  default:
    break;
  }
  asm_gencall(as, ci, args);
}
#endif

static void asm_comp(ASMState *as, IRIns *ir)
{
  /* ORDER IR: LT GE LE GT  ULT UGE ULE UGT. */
  IROp op = ir->o;
  if (!LJ_SOFTFP32 && irt_isnum(ir->t)) {
#if LJ_SOFTFP
    asm_sfpcomp(as, ir);
#else
#if !LJ_TARGET_MIPSR6
    Reg right, left = ra_alloc2(as, ir, RSET_FPR);
    right = (left >> 8); left &= 255;
    asm_guard(as, (op&1) ? MIPSI_BC1T : MIPSI_BC1F, 0, 0);
    emit_fgh(as, MIPSI_C_OLT_D + ((op&3) ^ ((op>>2)&1)), 0, left, right);
#else
    Reg tmp, right, left = ra_alloc2(as, ir, RSET_FPR);
    right = (left >> 8); left &= 255;
    tmp = ra_scratch(as, rset_exclude(rset_exclude(RSET_FPR, left), right));
    asm_guard(as, (op&1) ? MIPSI_BC1NEZ : MIPSI_BC1EQZ, 0, (tmp&31));
    emit_fgh(as, MIPSI_CMP_LT_D + ((op&3) ^ ((op>>2)&1)), tmp, left, right);
#endif
#endif
  } else {
    Reg right, left = ra_alloc1(as, ir->op1, RSET_GPR);
    if (op == IR_ABC) op = IR_UGT;
    if ((op&4) == 0 && irref_isk(ir->op2) && get_kval(as, ir->op2) == 0) {
      MIPSIns mi = (op&2) ? ((op&1) ? MIPSI_BLEZ : MIPSI_BGTZ) :
			    ((op&1) ? MIPSI_BLTZ : MIPSI_BGEZ);
      asm_guard(as, mi, left, 0);
    } else {
      if (irref_isk(ir->op2)) {
	intptr_t k = get_kval(as, ir->op2);
	if ((op&2)) k++;
	if (checki16(k)) {
	  asm_guard(as, (op&1) ? MIPSI_BNE : MIPSI_BEQ, RID_TMP, RID_ZERO);
	  emit_tsi(as, (op&4) ? MIPSI_SLTIU : MIPSI_SLTI,
		   RID_TMP, left, k);
	  return;
	}
      }
      right = ra_alloc1(as, ir->op2, rset_exclude(RSET_GPR, left));
      asm_guard(as, ((op^(op>>1))&1) ? MIPSI_BNE : MIPSI_BEQ, RID_TMP, RID_ZERO);
      emit_dst(as, (op&4) ? MIPSI_SLTU : MIPSI_SLT,
	       RID_TMP, (op&2) ? right : left, (op&2) ? left : right);
    }
  }
}

static void asm_equal(ASMState *as, IRIns *ir)
{
  Reg right, left = ra_alloc2(as, ir, (!LJ_SOFTFP && irt_isnum(ir->t)) ?
				       RSET_FPR : RSET_GPR);
  right = (left >> 8); left &= 255;
  if (!LJ_SOFTFP32 && irt_isnum(ir->t)) {
#if LJ_SOFTFP
    asm_sfpcomp(as, ir);
#elif !LJ_TARGET_MIPSR6
    asm_guard(as, (ir->o & 1) ? MIPSI_BC1T : MIPSI_BC1F, 0, 0);
    emit_fgh(as, MIPSI_C_EQ_D, 0, left, right);
#else
    Reg tmp = ra_scratch(as, rset_exclude(rset_exclude(RSET_FPR, left), right));
    asm_guard(as, (ir->o & 1) ? MIPSI_BC1NEZ : MIPSI_BC1EQZ, 0, (tmp&31));
    emit_fgh(as, MIPSI_CMP_EQ_D, tmp, left, right);
#endif
  } else {
    asm_guard(as, (ir->o & 1) ? MIPSI_BEQ : MIPSI_BNE, left, right);
  }
}

#if LJ_32 && LJ_HASFFI
/* 64 bit integer comparisons. */
static void asm_comp64(ASMState *as, IRIns *ir)
{
  /* ORDER IR: LT GE LE GT  ULT UGE ULE UGT. */
  IROp op = (ir-1)->o;
  MCLabel l_end;
  Reg rightlo, leftlo, righthi, lefthi = ra_alloc2(as, ir, RSET_GPR);
  righthi = (lefthi >> 8); lefthi &= 255;
  leftlo = ra_alloc2(as, ir-1,
		     rset_exclude(rset_exclude(RSET_GPR, lefthi), righthi));
  rightlo = (leftlo >> 8); leftlo &= 255;
  asm_guard(as, ((op^(op>>1))&1) ? MIPSI_BNE : MIPSI_BEQ, RID_TMP, RID_ZERO);
  l_end = emit_label(as);
  if (lefthi != righthi)
    emit_dst(as, (op&4) ? MIPSI_SLTU : MIPSI_SLT, RID_TMP,
	     (op&2) ? righthi : lefthi, (op&2) ? lefthi : righthi);
  emit_dst(as, MIPSI_SLTU, RID_TMP,
	   (op&2) ? rightlo : leftlo, (op&2) ? leftlo : rightlo);
  if (lefthi != righthi)
    emit_branch(as, MIPSI_BEQ, lefthi, righthi, l_end);
}

static void asm_comp64eq(ASMState *as, IRIns *ir)
{
  Reg tmp, right, left = ra_alloc2(as, ir, RSET_GPR);
  right = (left >> 8); left &= 255;
  asm_guard(as, ((ir-1)->o & 1) ? MIPSI_BEQ : MIPSI_BNE, RID_TMP, RID_ZERO);
  tmp = ra_scratch(as, rset_exclude(rset_exclude(RSET_GPR, left), right));
  emit_dst(as, MIPSI_OR, RID_TMP, RID_TMP, tmp);
  emit_dst(as, MIPSI_XOR, tmp, left, right);
  left = ra_alloc2(as, ir-1, RSET_GPR);
  right = (left >> 8); left &= 255;
  emit_dst(as, MIPSI_XOR, RID_TMP, left, right);
}
#endif

/* -- Split register ops -------------------------------------------------- */

/* Hiword op of a split 32/32 or 64/64 bit op. Previous op is the loword op. */
static void asm_hiop(ASMState *as, IRIns *ir)
{
  /* HIOP is marked as a store because it needs its own DCE logic. */
  int uselo = ra_used(ir-1), usehi = ra_used(ir);  /* Loword/hiword used? */
  if (LJ_UNLIKELY(!(as->flags & JIT_F_OPT_DCE))) uselo = usehi = 1;
#if LJ_32 && (LJ_HASFFI || LJ_SOFTFP)
  if ((ir-1)->o == IR_CONV) {  /* Conversions to/from 64 bit. */
    as->curins--;  /* Always skip the CONV. */
#if LJ_HASFFI && !LJ_SOFTFP
    if (usehi || uselo)
      asm_conv64(as, ir);
    return;
#endif
  } else if ((ir-1)->o < IR_EQ) {  /* 64 bit integer comparisons. ORDER IR. */
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
  } else if ((ir-1)->o <= IR_NE) {  /* 64 bit integer comparisons. ORDER IR. */
    as->curins--;  /* Always skip the loword comparison. */
#if LJ_SOFTFP
    if (!irt_isint(ir->t)) {
      asm_sfpcomp(as, ir-1);
      return;
    }
#endif
#if LJ_HASFFI
    asm_comp64eq(as, ir);
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
      asm_xstore_(as, ir, LJ_LE ? 4 : 0);
      asm_xstore_(as, ir-1, LJ_LE ? 0 : 4);
    }
    return;
  }
#endif
  if (!usehi) return;  /* Skip unused hiword op for all remaining ops. */
  switch ((ir-1)->o) {
#if LJ_32 && LJ_HASFFI
  case IR_ADD: as->curins--; asm_add64(as, ir); break;
  case IR_SUB: as->curins--; asm_sub64(as, ir); break;
  case IR_NEG: as->curins--; asm_neg64(as, ir); break;
  case IR_CNEWI:
    /* Nothing to do here. Handled by lo op itself. */
    break;
#endif
#if LJ_32 && LJ_SOFTFP
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
  asm_guard(as, MIPSI_BNE, RID_TMP, RID_ZERO);
  emit_tsi(as, MIPSI_ANDI, RID_TMP, RID_TMP, HOOK_PROFILE);
  emit_lsglptr(as, MIPSI_LBU, RID_TMP,
	       (int32_t)offsetof(global_State, hookmask));
}

/* -- Stack handling ------------------------------------------------------ */

/* Check Lua stack size for overflow. Use exit handler as fallback. */
static void asm_stack_check(ASMState *as, BCReg topslot,
			    IRIns *irp, RegSet allow, ExitNo exitno)
{
  /* Try to get an unused temp. register, otherwise spill/restore RID_RET*. */
  Reg tmp, pbase = irp ? (ra_hasreg(irp->r) ? irp->r : RID_TMP) : RID_BASE;
  ExitNo oldsnap = as->snapno;
  rset_clear(allow, pbase);
#if LJ_32
  tmp = allow ? rset_pickbot(allow) :
		(pbase == RID_RETHI ? RID_RETLO : RID_RETHI);
#else
  tmp = allow ? rset_pickbot(allow) : RID_RET;
#endif
  as->snapno = exitno;
  asm_guard(as, MIPSI_BNE, RID_TMP, RID_ZERO);
  as->snapno = oldsnap;
  if (allow == RSET_EMPTY)  /* Restore temp. register. */
    emit_tsi(as, MIPSI_AL, tmp, RID_SP, 0);
  else
    ra_modified(as, tmp);
  emit_tsi(as, MIPSI_SLTIU, RID_TMP, RID_TMP, (int32_t)(8*topslot));
  emit_dst(as, MIPSI_ASUBU, RID_TMP, tmp, pbase);
  emit_tsi(as, MIPSI_AL, tmp, tmp, offsetof(lua_State, maxstack));
  if (pbase == RID_TMP)
    emit_getgl(as, RID_TMP, jit_base);
  emit_getgl(as, tmp, cur_L);
  if (allow == RSET_EMPTY)  /* Spill temp. register. */
    emit_tsi(as, MIPSI_AS, tmp, RID_SP, 0);
}

/* Restore Lua stack from on-trace state. */
static void asm_stack_restore(ASMState *as, SnapShot *snap)
{
  SnapEntry *map = &as->T->snapmap[snap->mapofs];
#if LJ_32 || defined(LUA_USE_ASSERT)
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
#if LJ_SOFTFP32
      Reg tmp;
      RegSet allow = rset_exclude(RSET_GPR, RID_BASE);
      /* LJ_SOFTFP: must be a number constant. */
      lj_assertA(irref_isk(ref), "unsplit FP op");
      tmp = ra_allock(as, (int32_t)ir_knum(ir)->u32.lo, allow);
      emit_tsi(as, MIPSI_SW, tmp, RID_BASE, ofs+(LJ_BE?4:0));
      if (rset_test(as->freeset, tmp+1)) allow = RID2RSET(tmp+1);
      tmp = ra_allock(as, (int32_t)ir_knum(ir)->u32.hi, allow);
      emit_tsi(as, MIPSI_SW, tmp, RID_BASE, ofs+(LJ_BE?0:4));
#elif LJ_SOFTFP  /* && LJ_64 */
      Reg src = ra_alloc1(as, ref, rset_exclude(RSET_GPR, RID_BASE));
      emit_tsi(as, MIPSI_SD, src, RID_BASE, ofs);
#else
      Reg src = ra_alloc1(as, ref, RSET_FPR);
      emit_hsi(as, MIPSI_SDC1, src, RID_BASE, ofs);
#endif
    } else {
#if LJ_32
      RegSet allow = rset_exclude(RSET_GPR, RID_BASE);
      Reg type;
      lj_assertA(irt_ispri(ir->t) || irt_isaddr(ir->t) || irt_isinteger(ir->t),
		 "restore of IR type %d", irt_type(ir->t));
      if (!irt_ispri(ir->t)) {
	Reg src = ra_alloc1(as, ref, allow);
	rset_clear(allow, src);
	emit_tsi(as, MIPSI_SW, src, RID_BASE, ofs+(LJ_BE?4:0));
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
      emit_tsi(as, MIPSI_SW, type, RID_BASE, ofs+(LJ_BE?0:4));
#else
      if ((sn & SNAP_KEYINDEX)) {
	RegSet allow = rset_exclude(RSET_GPR, RID_BASE);
	int64_t kki = (int64_t)LJ_KEYINDEX << 32;
	if (irref_isk(ref)) {
	  emit_tsi(as, MIPSI_SD,
		   ra_allock(as, kki | (int64_t)(uint32_t)ir->i, allow),
		   RID_BASE, ofs);
	} else {
	  Reg src = ra_alloc1(as, ref, allow);
	  Reg rki = ra_allock(as, kki, rset_exclude(allow, src));
	  emit_tsi(as, MIPSI_SD, RID_TMP, RID_BASE, ofs);
	  emit_dst(as, MIPSI_DADDU, RID_TMP, src, rki);
	}
      } else {
	asm_tvstore64(as, RID_BASE, ofs, ref);
      }
#endif
    }
    checkmclim(as);
  }
  lj_assertA(map + nent == flinks, "inconsistent frames in snapshot");
}

/* -- GC handling --------------------------------------------------------- */

/* Marker to prevent patching the GC check exit. */
#define MIPS_NOPATCH_GC_CHECK	MIPSI_OR

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
  /* Assumes asm_snap_prep() already done. */
  asm_guard(as, MIPSI_BNE, RID_RET, RID_ZERO);
  args[0] = ASMREF_TMP1;  /* global_State *g */
  args[1] = ASMREF_TMP2;  /* MSize steps     */
  asm_gencall(as, ci, args);
  l_end[-3] = MIPS_NOPATCH_GC_CHECK;  /* Replace the nop after the call. */
  emit_tsi(as, MIPSI_AADDIU, ra_releasetmp(as, ASMREF_TMP1), RID_JGL, -32768);
  tmp = ra_releasetmp(as, ASMREF_TMP2);
  emit_loadi(as, tmp, as->gcsteps);
  /* Jump around GC step if GC total < GC threshold. */
  emit_branch(as, MIPSI_BNE, RID_TMP, RID_ZERO, l_end);
  emit_dst(as, MIPSI_SLTU, RID_TMP, RID_TMP, tmp);
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
  p[-1] = MIPSI_NOP;
  if (as->loopinv) {  /* Inverted loop branch? */
    /* asm_guard already inverted the cond branch. Only patch the target. */
    p[-3] |= ((target-p+2) & 0x0000ffffu);
  } else {
    p[-2] = MIPSI_J|(((uintptr_t)target>>2)&0x03ffffffu);
  }
}

/* Fixup the tail of the loop. */
static void asm_loop_tail_fixup(ASMState *as)
{
  if (as->loopinv) as->mctop--;
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
      emit_move(as, r, RID_BASE);
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
      emit_move(as, r, irp->r);  /* Move from coalesced parent reg. */
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
  MCode *target = lnk ? traceref(as->J,lnk)->mcode : (MCode *)lj_vm_exit_interp;
  int32_t spadj = as->T->spadjust;
  MCode *p = as->mctop-1;
  *p = spadj ? (MIPSI_AADDIU|MIPSF_T(RID_SP)|MIPSF_S(RID_SP)|spadj) : MIPSI_NOP;
  p[-1] = MIPSI_J|(((uintptr_t)target>>2)&0x03ffffffu);
}

/* Prepare tail of code. */
static void asm_tail_prep(ASMState *as)
{
  as->mcp = as->mctop-2;  /* Leave room for branch plus nop or stack adj. */
  as->invmcp = as->loopref ? as->mcp : NULL;
}

/* -- Trace setup --------------------------------------------------------- */

/* Ensure there are enough stack slots for call arguments. */
static Reg asm_setup_call_slots(ASMState *as, IRIns *ir, const CCallInfo *ci)
{
  IRRef args[CCI_NARGS_MAX*2];
  uint32_t i, nargs = CCI_XNARGS(ci);
#if LJ_32
  int nslots = 4, ngpr = REGARG_NUMGPR, nfpr = REGARG_NUMFPR;
#else
  int nslots = 0, ngpr = REGARG_NUMGPR;
#endif
  asm_collectargs(as, ir, ci, args);
  for (i = 0; i < nargs; i++) {
#if LJ_32
    if (!LJ_SOFTFP && args[i] && irt_isfp(IR(args[i])->t) &&
	nfpr > 0 && !(ci->flags & CCI_VARARG)) {
      nfpr--;
      ngpr -= irt_isnum(IR(args[i])->t) ? 2 : 1;
    } else if (!LJ_SOFTFP && args[i] && irt_isnum(IR(args[i])->t)) {
      nfpr = 0;
      ngpr = ngpr & ~1;
      if (ngpr > 0) ngpr -= 2; else nslots = (nslots+3) & ~1;
    } else {
      nfpr = 0;
      if (ngpr > 0) ngpr--; else nslots++;
    }
#else
    if (ngpr > 0) ngpr--; else nslots += 2;
#endif
  }
  if (nslots > as->evenspill)  /* Leave room for args in stack slots. */
    as->evenspill = nslots;
  return irt_isfp(ir->t) ? REGSP_HINT(RID_FPRET) : REGSP_HINT(RID_RET);
}

static void asm_setup_target(ASMState *as)
{
  asm_sparejump_setup(as);
  asm_exitstub_setup(as);
}

/* -- Trace patching ------------------------------------------------------ */

/* Patch exit jumps of existing machine code to a new target. */
void lj_asm_patchexit(jit_State *J, GCtrace *T, ExitNo exitno, MCode *target)
{
  MCode *p = T->mcode;
  MCode *pe = (MCode *)((char *)p + T->szmcode);
  MCode *px = exitstub_trace_addr(T, exitno);
  MCode *cstart = NULL, *cstop = NULL;
  MCode *mcarea = lj_mcode_patch(J, p, 0);
  MCode exitload = MIPSI_LI | MIPSF_T(RID_TMP) | exitno;
  MCode tjump = MIPSI_J|(((uintptr_t)target>>2)&0x03ffffffu);
  for (p++; p < pe; p++) {
    if (*p == exitload) {  /* Look for load of exit number. */
      /* Look for exitstub branch. Yes, this covers all used branch variants. */
      if (((p[-1] ^ (px-p)) & 0xffffu) == 0 &&
	  ((p[-1] & 0xf0000000u) == MIPSI_BEQ ||
	   (p[-1] & 0xfc1e0000u) == MIPSI_BLTZ ||
#if !LJ_TARGET_MIPSR6
	   (p[-1] & 0xffe00000u) == MIPSI_BC1F
#else
	   (p[-1] & 0xff600000u) == MIPSI_BC1EQZ
#endif
	  ) && p[-2] != MIPS_NOPATCH_GC_CHECK) {
	ptrdiff_t delta = target - p;
	if (((delta + 0x8000) >> 16) == 0) {  /* Patch in-range branch. */
	patchbranch:
	  p[-1] = (p[-1] & 0xffff0000u) | (delta & 0xffffu);
	  *p = MIPSI_NOP;  /* Replace the load of the exit number. */
	  cstop = p+1;
	  if (!cstart) cstart = p-1;
	} else {  /* Branch out of range. Use spare jump slot in mcarea. */
	  MCode *mcjump = asm_sparejump_use(mcarea, tjump);
	  if (mcjump) {
	    lj_mcode_sync(mcjump, mcjump+1);
	    delta = mcjump - p;
	    if (((delta + 0x8000) >> 16) == 0) {
	      goto patchbranch;
	    } else {
	      lj_assertJ(0, "spare jump out of range: -Osizemcode too big");
	    }
	  }
	  /* Ignore jump slot overflow. Child trace is simply not attached. */
	}
      } else if (p+1 == pe) {
	/* Patch NOP after code for inverted loop branch. Use of J is ok. */
	lj_assertJ(p[1] == MIPSI_NOP, "expected NOP");
	p[1] = tjump;
	*p = MIPSI_NOP;  /* Replace the load of the exit number. */
	cstop = p+2;
	if (!cstart) cstart = p+1;
      }
    }
  }
  if (cstart) lj_mcode_sync(cstart, cstop);
  lj_mcode_patch(J, mcarea, 1);
}

