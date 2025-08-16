/*
** x86/x64 IR assembler (SSA IR -> machine code).
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
*/

/* -- Guard handling ------------------------------------------------------ */

/* Generate an exit stub group at the bottom of the reserved MCode memory. */
static MCode *asm_exitstub_gen(ASMState *as, ExitNo group)
{
  ExitNo i, groupofs = (group*EXITSTUBS_PER_GROUP) & 0xff;
  MCode *mxp = as->mcbot;
  MCode *mxpstart = mxp;
  if (mxp + (2+2)*EXITSTUBS_PER_GROUP+8+5 >= as->mctop)
    asm_mclimit(as);
  /* Push low byte of exitno for each exit stub. */
  *mxp++ = XI_PUSHi8; *mxp++ = (MCode)groupofs;
  for (i = 1; i < EXITSTUBS_PER_GROUP; i++) {
    *mxp++ = XI_JMPs; *mxp++ = (MCode)((2+2)*(EXITSTUBS_PER_GROUP - i) - 2);
    *mxp++ = XI_PUSHi8; *mxp++ = (MCode)(groupofs + i);
  }
  /* Push the high byte of the exitno for each exit stub group. */
  *mxp++ = XI_PUSHi8; *mxp++ = (MCode)((group*EXITSTUBS_PER_GROUP)>>8);
#if !LJ_GC64
  /* Store DISPATCH at original stack slot 0. Account for the two push ops. */
  *mxp++ = XI_MOVmi;
  *mxp++ = MODRM(XM_OFS8, 0, RID_ESP);
  *mxp++ = MODRM(XM_SCALE1, RID_ESP, RID_ESP);
  *mxp++ = 2*sizeof(void *);
  *(int32_t *)mxp = ptr2addr(J2GG(as->J)->dispatch); mxp += 4;
#endif
  /* Jump to exit handler which fills in the ExitState. */
  *mxp++ = XI_JMP; mxp += 4;
  *((int32_t *)(mxp-4)) = jmprel(as->J, mxp, (MCode *)(void *)lj_vm_exit_handler);
  /* Commit the code for this group (even if assembly fails later on). */
  lj_mcode_commitbot(as->J, mxp);
  as->mcbot = mxp;
  as->mclim = as->mcbot + MCLIM_REDZONE;
  return mxpstart;
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

/* Emit conditional branch to exit for guard.
** It's important to emit this *after* all registers have been allocated,
** because rematerializations may invalidate the flags.
*/
static void asm_guardcc(ASMState *as, int cc)
{
  MCode *target = exitstub_addr(as->J, as->snapno);
  MCode *p = as->mcp;
  if (LJ_UNLIKELY(p == as->invmcp)) {
    as->loopinv = 1;
    *(int32_t *)(p+1) = jmprel(as->J, p+5, target);
    target = p;
    cc ^= 1;
    if (as->realign) {
      if (LJ_GC64 && LJ_UNLIKELY(as->mrm.base == RID_RIP))
	as->mrm.ofs += 2;  /* Fixup RIP offset for pending fused load. */
      emit_sjcc(as, cc, target);
      return;
    }
  }
  if (LJ_GC64 && LJ_UNLIKELY(as->mrm.base == RID_RIP))
    as->mrm.ofs += 6;  /* Fixup RIP offset for pending fused load. */
  emit_jcc(as, cc, target);
}

/* -- Memory operand fusion ----------------------------------------------- */

/* Limit linear search to this distance. Avoids O(n^2) behavior. */
#define CONFLICT_SEARCH_LIM	31

/* Check if a reference is a signed 32 bit constant. */
static int asm_isk32(ASMState *as, IRRef ref, int32_t *k)
{
  if (irref_isk(ref)) {
    IRIns *ir = IR(ref);
#if LJ_GC64
    if (ir->o == IR_KNULL || !irt_is64(ir->t)) {
      *k = ir->i;
      return 1;
    } else if (checki32((int64_t)ir_k64(ir)->u64)) {
      *k = (int32_t)ir_k64(ir)->u64;
      return 1;
    }
#else
    if (ir->o != IR_KINT64) {
      *k = ir->i;
      return 1;
    } else if (checki32((int64_t)ir_kint64(ir)->u64)) {
      *k = (int32_t)ir_kint64(ir)->u64;
      return 1;
    }
#endif
  }
  return 0;
}

/* Check if there's no conflicting instruction between curins and ref.
** Also avoid fusing loads if there are multiple references.
*/
static int noconflict(ASMState *as, IRRef ref, IROp conflict, int check)
{
  IRIns *ir = as->ir;
  IRRef i = as->curins;
  if (i > ref + CONFLICT_SEARCH_LIM)
    return 0;  /* Give up, ref is too far away. */
  while (--i > ref) {
    if (ir[i].o == conflict)
      return 0;  /* Conflict found. */
    else if ((check & 1) && (ir[i].o == IR_NEWREF || ir[i].o == IR_CALLS))
      return 0;
    else if ((check & 2) && (ir[i].op1 == ref || ir[i].op2 == ref))
      return 0;
  }
  return 1;  /* Ok, no conflict. */
}

/* Fuse array base into memory operand. */
static IRRef asm_fuseabase(ASMState *as, IRRef ref)
{
  IRIns *irb = IR(ref);
  as->mrm.ofs = 0;
  if (irb->o == IR_FLOAD) {
    IRIns *ira = IR(irb->op1);
    lj_assertA(irb->op2 == IRFL_TAB_ARRAY, "expected FLOAD TAB_ARRAY");
    /* We can avoid the FLOAD of t->array for colocated arrays. */
    if (ira->o == IR_TNEW && ira->op1 <= LJ_MAX_COLOSIZE &&
	!neverfuse(as) && noconflict(as, irb->op1, IR_NEWREF, 0)) {
      as->mrm.ofs = (int32_t)sizeof(GCtab);  /* Ofs to colocated array. */
      return irb->op1;  /* Table obj. */
    }
  } else if (irb->o == IR_ADD && irref_isk(irb->op2)) {
    /* Fuse base offset (vararg load). */
    IRIns *irk = IR(irb->op2);
    as->mrm.ofs = irk->o == IR_KINT ? irk->i : (int32_t)ir_kint64(irk)->u64;
    return irb->op1;
  }
  return ref;  /* Otherwise use the given array base. */
}

/* Fuse array reference into memory operand. */
static void asm_fusearef(ASMState *as, IRIns *ir, RegSet allow)
{
  IRIns *irx;
  lj_assertA(ir->o == IR_AREF, "expected AREF");
  as->mrm.base = (uint8_t)ra_alloc1(as, asm_fuseabase(as, ir->op1), allow);
  irx = IR(ir->op2);
  if (irref_isk(ir->op2)) {
    as->mrm.ofs += 8*irx->i;
    as->mrm.idx = RID_NONE;
  } else {
    rset_clear(allow, as->mrm.base);
    as->mrm.scale = XM_SCALE8;
    /* Fuse a constant ADD (e.g. t[i+1]) into the offset.
    ** Doesn't help much without ABCelim, but reduces register pressure.
    */
    if (!LJ_64 &&  /* Has bad effects with negative index on x64. */
	mayfuse(as, ir->op2) && ra_noreg(irx->r) &&
	irx->o == IR_ADD && irref_isk(irx->op2)) {
      as->mrm.ofs += 8*IR(irx->op2)->i;
      as->mrm.idx = (uint8_t)ra_alloc1(as, irx->op1, allow);
    } else {
      as->mrm.idx = (uint8_t)ra_alloc1(as, ir->op2, allow);
    }
  }
}

/* Fuse array/hash/upvalue reference into memory operand.
** Caveat: this may allocate GPRs for the base/idx registers. Be sure to
** pass the final allow mask, excluding any GPRs used for other inputs.
** In particular: 2-operand GPR instructions need to call ra_dest() first!
*/
static void asm_fuseahuref(ASMState *as, IRRef ref, RegSet allow)
{
  IRIns *ir = IR(ref);
  if (ra_noreg(ir->r)) {
    switch ((IROp)ir->o) {
    case IR_AREF:
      if (mayfuse(as, ref)) {
	asm_fusearef(as, ir, allow);
	return;
      }
      break;
    case IR_HREFK:
      if (mayfuse(as, ref)) {
	as->mrm.base = (uint8_t)ra_alloc1(as, ir->op1, allow);
	as->mrm.ofs = (int32_t)(IR(ir->op2)->op2 * sizeof(Node));
	as->mrm.idx = RID_NONE;
	return;
      }
      break;
    case IR_UREFC:
      if (irref_isk(ir->op1)) {
	GCfunc *fn = ir_kfunc(IR(ir->op1));
	GCupval *uv = &gcref(fn->l.uvptr[(ir->op2 >> 8)])->uv;
#if LJ_GC64
	int64_t ofs = dispofs(as, &uv->tv);
	if (checki32(ofs) && checki32(ofs+4)) {
	  as->mrm.ofs = (int32_t)ofs;
	  as->mrm.base = RID_DISPATCH;
	  as->mrm.idx = RID_NONE;
	  return;
	}
#else
	as->mrm.ofs = ptr2addr(&uv->tv);
	as->mrm.base = as->mrm.idx = RID_NONE;
	return;
#endif
      }
      break;
    case IR_TMPREF:
#if LJ_GC64
      as->mrm.ofs = (int32_t)dispofs(as, &J2G(as->J)->tmptv);
      as->mrm.base = RID_DISPATCH;
      as->mrm.idx = RID_NONE;
#else
      as->mrm.ofs = igcptr(&J2G(as->J)->tmptv);
      as->mrm.base = as->mrm.idx = RID_NONE;
#endif
      return;
    default:
      break;
    }
  }
  as->mrm.base = (uint8_t)ra_alloc1(as, ref, allow);
  as->mrm.ofs = 0;
  as->mrm.idx = RID_NONE;
}

/* Fuse FLOAD/FREF reference into memory operand. */
static void asm_fusefref(ASMState *as, IRIns *ir, RegSet allow)
{
  lj_assertA(ir->o == IR_FLOAD || ir->o == IR_FREF,
	     "bad IR op %d", ir->o);
  as->mrm.idx = RID_NONE;
  if (ir->op1 == REF_NIL) {  /* FLOAD from GG_State with offset. */
#if LJ_GC64
    as->mrm.ofs = (int32_t)(ir->op2 << 2) - GG_OFS(dispatch);
    as->mrm.base = RID_DISPATCH;
#else
    as->mrm.ofs = (int32_t)(ir->op2 << 2) + ptr2addr(J2GG(as->J));
    as->mrm.base = RID_NONE;
#endif
    return;
  }
  as->mrm.ofs = field_ofs[ir->op2];
  if (irref_isk(ir->op1)) {
    IRIns *op1 = IR(ir->op1);
#if LJ_GC64
    if (ir->op1 == REF_NIL) {
      as->mrm.ofs -= GG_OFS(dispatch);
      as->mrm.base = RID_DISPATCH;
      return;
    } else if (op1->o == IR_KPTR || op1->o == IR_KKPTR) {
      intptr_t ofs = dispofs(as, ir_kptr(op1));
      if (checki32(as->mrm.ofs + ofs)) {
	as->mrm.ofs += (int32_t)ofs;
	as->mrm.base = RID_DISPATCH;
	return;
      }
    }
#else
    as->mrm.ofs += op1->i;
    as->mrm.base = RID_NONE;
    return;
#endif
  }
  as->mrm.base = (uint8_t)ra_alloc1(as, ir->op1, allow);
}

/* Fuse string reference into memory operand. */
static void asm_fusestrref(ASMState *as, IRIns *ir, RegSet allow)
{
  IRIns *irr;
  lj_assertA(ir->o == IR_STRREF, "bad IR op %d", ir->o);
  as->mrm.base = as->mrm.idx = RID_NONE;
  as->mrm.scale = XM_SCALE1;
  as->mrm.ofs = sizeof(GCstr);
  if (!LJ_GC64 && irref_isk(ir->op1)) {
    as->mrm.ofs += IR(ir->op1)->i;
  } else {
    Reg r = ra_alloc1(as, ir->op1, allow);
    rset_clear(allow, r);
    as->mrm.base = (uint8_t)r;
  }
  irr = IR(ir->op2);
  if (irref_isk(ir->op2)) {
    as->mrm.ofs += irr->i;
  } else {
    Reg r;
    /* Fuse a constant add into the offset, e.g. string.sub(s, i+10). */
    if (!LJ_64 &&  /* Has bad effects with negative index on x64. */
	mayfuse(as, ir->op2) && irr->o == IR_ADD && irref_isk(irr->op2)) {
      as->mrm.ofs += IR(irr->op2)->i;
      r = ra_alloc1(as, irr->op1, allow);
    } else {
      r = ra_alloc1(as, ir->op2, allow);
    }
    if (as->mrm.base == RID_NONE)
      as->mrm.base = (uint8_t)r;
    else
      as->mrm.idx = (uint8_t)r;
  }
}

static void asm_fusexref(ASMState *as, IRRef ref, RegSet allow)
{
  IRIns *ir = IR(ref);
  as->mrm.idx = RID_NONE;
  if (ir->o == IR_KPTR || ir->o == IR_KKPTR) {
#if LJ_GC64
    intptr_t ofs = dispofs(as, ir_kptr(ir));
    if (checki32(ofs)) {
      as->mrm.ofs = (int32_t)ofs;
      as->mrm.base = RID_DISPATCH;
      return;
    }
  } if (0) {
#else
    as->mrm.ofs = ir->i;
    as->mrm.base = RID_NONE;
  } else if (ir->o == IR_STRREF) {
    asm_fusestrref(as, ir, allow);
#endif
  } else {
    as->mrm.ofs = 0;
    if (canfuse(as, ir) && ir->o == IR_ADD && ra_noreg(ir->r)) {
      /* Gather (base+idx*sz)+ofs as emitted by cdata ptr/array indexing. */
      IRIns *irx;
      IRRef idx;
      Reg r;
      if (asm_isk32(as, ir->op2, &as->mrm.ofs)) {  /* Recognize x+ofs. */
	ref = ir->op1;
	ir = IR(ref);
	if (!(ir->o == IR_ADD && canfuse(as, ir) && ra_noreg(ir->r)))
	  goto noadd;
      }
      as->mrm.scale = XM_SCALE1;
      idx = ir->op1;
      ref = ir->op2;
      irx = IR(idx);
      if (!(irx->o == IR_BSHL || irx->o == IR_ADD)) {  /* Try other operand. */
	idx = ir->op2;
	ref = ir->op1;
	irx = IR(idx);
      }
      if (canfuse(as, irx) && ra_noreg(irx->r)) {
	if (irx->o == IR_BSHL && irref_isk(irx->op2) && IR(irx->op2)->i <= 3) {
	  /* Recognize idx<<b with b = 0-3, corresponding to sz = (1),2,4,8. */
	  idx = irx->op1;
	  as->mrm.scale = (uint8_t)(IR(irx->op2)->i << 6);
	} else if (irx->o == IR_ADD && irx->op1 == irx->op2) {
	  /* FOLD does idx*2 ==> idx<<1 ==> idx+idx. */
	  idx = irx->op1;
	  as->mrm.scale = XM_SCALE2;
	}
      }
      r = ra_alloc1(as, idx, allow);
      rset_clear(allow, r);
      as->mrm.idx = (uint8_t)r;
    }
  noadd:
    as->mrm.base = (uint8_t)ra_alloc1(as, ref, allow);
  }
}

/* Fuse load of 64 bit IR constant into memory operand. */
static Reg asm_fuseloadk64(ASMState *as, IRIns *ir)
{
  const uint64_t *k = &ir_k64(ir)->u64;
  if (!LJ_GC64 || checki32((intptr_t)k)) {
    as->mrm.ofs = ptr2addr(k);
    as->mrm.base = RID_NONE;
#if LJ_GC64
  } else if (checki32(dispofs(as, k))) {
    as->mrm.ofs = (int32_t)dispofs(as, k);
    as->mrm.base = RID_DISPATCH;
  } else if (checki32(mcpofs(as, k)) && checki32(mcpofs(as, k+1)) &&
	     checki32(mctopofs(as, k)) && checki32(mctopofs(as, k+1))) {
    as->mrm.ofs = (int32_t)mcpofs(as, k);
    as->mrm.base = RID_RIP;
  } else {  /* Intern 64 bit constant at bottom of mcode. */
    if (ir->i) {
      lj_assertA(*k == *(uint64_t*)(as->mctop - ir->i),
		 "bad interned 64 bit constant");
    } else {
      while ((uintptr_t)as->mcbot & 7) *as->mcbot++ = XI_INT3;
      *(uint64_t*)as->mcbot = *k;
      ir->i = (int32_t)(as->mctop - as->mcbot);
      as->mcbot += 8;
      as->mclim = as->mcbot + MCLIM_REDZONE;
      lj_mcode_commitbot(as->J, as->mcbot);
    }
    as->mrm.ofs = (int32_t)mcpofs(as, as->mctop - ir->i);
    as->mrm.base = RID_RIP;
#endif
  }
  as->mrm.idx = RID_NONE;
  return RID_MRM;
}

/* Fuse load into memory operand.
**
** Important caveat: this may emit RIP-relative loads! So don't place any
** code emitters between this function and the use of its result.
** The only permitted exception is asm_guardcc().
*/
static Reg asm_fuseload(ASMState *as, IRRef ref, RegSet allow)
{
  IRIns *ir = IR(ref);
  if (ra_hasreg(ir->r)) {
    if (allow != RSET_EMPTY) {  /* Fast path. */
      ra_noweak(as, ir->r);
      return ir->r;
    }
  fusespill:
    /* Force a spill if only memory operands are allowed (asm_x87load). */
    as->mrm.base = RID_ESP;
    as->mrm.ofs = ra_spill(as, ir);
    as->mrm.idx = RID_NONE;
    return RID_MRM;
  }
  if (ir->o == IR_KNUM) {
    RegSet avail = as->freeset & ~as->modset & RSET_FPR;
    lj_assertA(allow != RSET_EMPTY, "no register allowed");
    if (!(avail & (avail-1)))  /* Fuse if less than two regs available. */
      return asm_fuseloadk64(as, ir);
  } else if (ref == REF_BASE || ir->o == IR_KINT64) {
    RegSet avail = as->freeset & ~as->modset & RSET_GPR;
    lj_assertA(allow != RSET_EMPTY, "no register allowed");
    if (!(avail & (avail-1))) {  /* Fuse if less than two regs available. */
      if (ref == REF_BASE) {
#if LJ_GC64
	as->mrm.ofs = (int32_t)dispofs(as, &J2G(as->J)->jit_base);
	as->mrm.base = RID_DISPATCH;
#else
	as->mrm.ofs = ptr2addr(&J2G(as->J)->jit_base);
	as->mrm.base = RID_NONE;
#endif
	as->mrm.idx = RID_NONE;
	return RID_MRM;
      } else {
	return asm_fuseloadk64(as, ir);
      }
    }
  } else if (mayfuse(as, ref)) {
    RegSet xallow = (allow & RSET_GPR) ? allow : RSET_GPR;
    if (ir->o == IR_SLOAD) {
      if (!(ir->op2 & (IRSLOAD_PARENT|IRSLOAD_CONVERT)) &&
	  noconflict(as, ref, IR_RETF, 2) &&
	  !(LJ_GC64 && irt_isaddr(ir->t))) {
	as->mrm.base = (uint8_t)ra_alloc1(as, REF_BASE, xallow);
	as->mrm.ofs = 8*((int32_t)ir->op1-1-LJ_FR2) +
		      (!LJ_FR2 && (ir->op2 & IRSLOAD_FRAME) ? 4 : 0);
	as->mrm.idx = RID_NONE;
	return RID_MRM;
      }
    } else if (ir->o == IR_FLOAD) {
      /* Generic fusion is only ok for 32 bit operand (but see asm_comp). */
      if ((irt_isint(ir->t) || irt_isu32(ir->t) || irt_isaddr(ir->t)) &&
	  noconflict(as, ref, IR_FSTORE, 2)) {
	asm_fusefref(as, ir, xallow);
	return RID_MRM;
      }
    } else if (ir->o == IR_ALOAD || ir->o == IR_HLOAD || ir->o == IR_ULOAD) {
      if (noconflict(as, ref, ir->o + IRDELTA_L2S, 2+(ir->o != IR_ULOAD)) &&
	  !(LJ_GC64 && irt_isaddr(ir->t))) {
	asm_fuseahuref(as, ir->op1, xallow);
	return RID_MRM;
      }
    } else if (ir->o == IR_XLOAD) {
      /* Generic fusion is not ok for 8/16 bit operands (but see asm_comp).
      ** Fusing unaligned memory operands is ok on x86 (except for SIMD types).
      */
      if ((!irt_typerange(ir->t, IRT_I8, IRT_U16)) &&
	  noconflict(as, ref, IR_XSTORE, 2)) {
	asm_fusexref(as, ir->op1, xallow);
	return RID_MRM;
      }
    } else if (ir->o == IR_VLOAD && IR(ir->op1)->o == IR_AREF &&
	       !(LJ_GC64 && irt_isaddr(ir->t))) {
      asm_fuseahuref(as, ir->op1, xallow);
      as->mrm.ofs += 8 * ir->op2;
      return RID_MRM;
    }
  }
  if (ir->o == IR_FLOAD && ir->op1 == REF_NIL) {
    asm_fusefref(as, ir, RSET_EMPTY);
    return RID_MRM;
  }
  if (!(as->freeset & allow) && !emit_canremat(ref) &&
      (allow == RSET_EMPTY || ra_hasspill(ir->s) || iscrossref(as, ref)))
    goto fusespill;
  return ra_allocref(as, ref, allow);
}

#if LJ_64
/* Don't fuse a 32 bit load into a 64 bit operation. */
static Reg asm_fuseloadm(ASMState *as, IRRef ref, RegSet allow, int is64)
{
  if (is64 && !irt_is64(IR(ref)->t))
    return ra_alloc1(as, ref, allow);
  return asm_fuseload(as, ref, allow);
}
#else
#define asm_fuseloadm(as, ref, allow, is64)  asm_fuseload(as, (ref), (allow))
#endif

/* -- Calls --------------------------------------------------------------- */

/* Count the required number of stack slots for a call. */
static int asm_count_call_slots(ASMState *as, const CCallInfo *ci, IRRef *args)
{
  uint32_t i, nargs = CCI_XNARGS(ci);
  int nslots = 0;
#if LJ_64
  if (LJ_ABI_WIN) {
    nslots = (int)(nargs*2);  /* Only matters for more than four args. */
  } else {
    int ngpr = REGARG_NUMGPR, nfpr = REGARG_NUMFPR;
    for (i = 0; i < nargs; i++)
      if (args[i] && irt_isfp(IR(args[i])->t)) {
	if (nfpr > 0) nfpr--; else nslots += 2;
      } else {
	if (ngpr > 0) ngpr--; else nslots += 2;
      }
  }
#else
  int ngpr = 0;
  if ((ci->flags & CCI_CC_MASK) == CCI_CC_FASTCALL)
    ngpr = 2;
  else if ((ci->flags & CCI_CC_MASK) == CCI_CC_THISCALL)
    ngpr = 1;
  for (i = 0; i < nargs; i++)
    if (args[i] && irt_isfp(IR(args[i])->t)) {
      nslots += irt_isnum(IR(args[i])->t) ? 2 : 1;
    } else {
      if (ngpr > 0) ngpr--; else nslots++;
    }
#endif
  return nslots;
}

/* Generate a call to a C function. */
static void asm_gencall(ASMState *as, const CCallInfo *ci, IRRef *args)
{
  uint32_t n, nargs = CCI_XNARGS(ci);
  int32_t ofs = STACKARG_OFS;
#if LJ_64
  uint32_t gprs = REGARG_GPRS;
  Reg fpr = REGARG_FIRSTFPR;
#if !LJ_ABI_WIN
  MCode *patchnfpr = NULL;
#endif
#else
  uint32_t gprs = 0;
  if ((ci->flags & CCI_CC_MASK) != CCI_CC_CDECL) {
    if ((ci->flags & CCI_CC_MASK) == CCI_CC_THISCALL)
      gprs = (REGARG_GPRS & 31);
    else if ((ci->flags & CCI_CC_MASK) == CCI_CC_FASTCALL)
      gprs = REGARG_GPRS;
  }
#endif
  if ((void *)ci->func)
    emit_call(as, ci->func);
#if LJ_64
  if ((ci->flags & CCI_VARARG)) {  /* Special handling for vararg calls. */
#if LJ_ABI_WIN
    for (n = 0; n < 4 && n < nargs; n++) {
      IRIns *ir = IR(args[n]);
      if (irt_isfp(ir->t))  /* Duplicate FPRs in GPRs. */
	emit_rr(as, XO_MOVDto, (irt_isnum(ir->t) ? REX_64 : 0) | (fpr+n),
		((gprs >> (n*5)) & 31));  /* Either MOVD or MOVQ. */
    }
#else
    patchnfpr = --as->mcp;  /* Indicate number of used FPRs in register al. */
    *--as->mcp = XI_MOVrib | RID_EAX;
#endif
  }
#endif
  for (n = 0; n < nargs; n++) {  /* Setup args. */
    IRRef ref = args[n];
    IRIns *ir = IR(ref);
    Reg r;
#if LJ_64 && LJ_ABI_WIN
    /* Windows/x64 argument registers are strictly positional. */
    r = irt_isfp(ir->t) ? (fpr <= REGARG_LASTFPR ? fpr : 0) : (gprs & 31);
    fpr++; gprs >>= 5;
#elif LJ_64
    /* POSIX/x64 argument registers are used in order of appearance. */
    if (irt_isfp(ir->t)) {
      r = fpr <= REGARG_LASTFPR ? fpr++ : 0;
    } else {
      r = gprs & 31; gprs >>= 5;
    }
#else
    if (ref && irt_isfp(ir->t)) {
      r = 0;
    } else {
      r = gprs & 31; gprs >>= 5;
      if (!ref) continue;
    }
#endif
    if (r) {  /* Argument is in a register. */
      if (r < RID_MAX_GPR && ref < ASMREF_TMP1) {
#if LJ_64
	if (LJ_GC64 ? !(ir->o == IR_KINT || ir->o == IR_KNULL) : ir->o == IR_KINT64)
	  emit_loadu64(as, r, ir_k64(ir)->u64);
	else
#endif
	  emit_loadi(as, r, ir->i);
      } else {
	/* Must have been evicted. */
	lj_assertA(rset_test(as->freeset, r), "reg %d not free", r);
	if (ra_hasreg(ir->r)) {
	  ra_noweak(as, ir->r);
	  emit_movrr(as, ir, r, ir->r);
	} else {
	  ra_allocref(as, ref, RID2RSET(r));
	}
      }
    } else if (irt_isfp(ir->t)) {  /* FP argument is on stack. */
      lj_assertA(!(irt_isfloat(ir->t) && irref_isk(ref)),
		 "unexpected float constant");
      if (LJ_32 && (ofs & 4) && irref_isk(ref)) {
	/* Split stores for unaligned FP consts. */
	emit_movmroi(as, RID_ESP, ofs, (int32_t)ir_knum(ir)->u32.lo);
	emit_movmroi(as, RID_ESP, ofs+4, (int32_t)ir_knum(ir)->u32.hi);
      } else {
	r = ra_alloc1(as, ref, RSET_FPR);
	emit_rmro(as, irt_isnum(ir->t) ? XO_MOVSDto : XO_MOVSSto,
		  r, RID_ESP, ofs);
      }
      ofs += (LJ_32 && irt_isfloat(ir->t)) ? 4 : 8;
    } else {  /* Non-FP argument is on stack. */
      if (LJ_32 && ref < ASMREF_TMP1) {
	emit_movmroi(as, RID_ESP, ofs, ir->i);
      } else {
	r = ra_alloc1(as, ref, RSET_GPR);
	emit_movtomro(as, REX_64 + r, RID_ESP, ofs);
      }
      ofs += sizeof(intptr_t);
    }
    checkmclim(as);
  }
#if LJ_64 && !LJ_ABI_WIN
  if (patchnfpr) *patchnfpr = fpr - REGARG_FIRSTFPR;
#endif
}

/* Setup result reg/sp for call. Evict scratch regs. */
static void asm_setupresult(ASMState *as, IRIns *ir, const CCallInfo *ci)
{
  RegSet drop = RSET_SCRATCH;
  int hiop = ((ir+1)->o == IR_HIOP && !irt_isnil((ir+1)->t));
  if ((ci->flags & CCI_NOFPRCLOBBER))
    drop &= ~RSET_FPR;
  if (ra_hasreg(ir->r))
    rset_clear(drop, ir->r);  /* Dest reg handled below. */
  if (hiop && ra_hasreg((ir+1)->r))
    rset_clear(drop, (ir+1)->r);  /* Dest reg handled below. */
  ra_evictset(as, drop);  /* Evictions must be performed first. */
  if (ra_used(ir)) {
    if (irt_isfp(ir->t)) {
      int32_t ofs = sps_scale(ir->s);  /* Use spill slot or temp slots. */
#if LJ_64
      if ((ci->flags & CCI_CASTU64)) {
	Reg dest = ir->r;
	if (ra_hasreg(dest)) {
	  ra_free(as, dest);
	  ra_modified(as, dest);
	  emit_rr(as, XO_MOVD, dest|REX_64, RID_RET);  /* Really MOVQ. */
	}
	if (ofs) emit_movtomro(as, RID_RET|REX_64, RID_ESP, ofs);
      } else {
	ra_destreg(as, ir, RID_FPRET);
      }
#else
      /* Number result is in x87 st0 for x86 calling convention. */
      Reg dest = ir->r;
      if (ra_hasreg(dest)) {
	ra_free(as, dest);
	ra_modified(as, dest);
	emit_rmro(as, irt_isnum(ir->t) ? XO_MOVSD : XO_MOVSS,
		  dest, RID_ESP, ofs);
      }
      if ((ci->flags & CCI_CASTU64)) {
	emit_movtomro(as, RID_RETLO, RID_ESP, ofs);
	emit_movtomro(as, RID_RETHI, RID_ESP, ofs+4);
      } else {
	emit_rmro(as, irt_isnum(ir->t) ? XO_FSTPq : XO_FSTPd,
		  irt_isnum(ir->t) ? XOg_FSTPq : XOg_FSTPd, RID_ESP, ofs);
      }
#endif
    } else if (hiop) {
      ra_destpair(as, ir);
    } else {
      lj_assertA(!irt_ispri(ir->t), "PRI dest");
      ra_destreg(as, ir, RID_RET);
    }
  } else if (LJ_32 && irt_isfp(ir->t) && !(ci->flags & CCI_CASTU64)) {
    emit_x87op(as, XI_FPOP);  /* Pop unused result from x87 st0. */
  }
}

/* Return a constant function pointer or NULL for indirect calls. */
static void *asm_callx_func(ASMState *as, IRIns *irf, IRRef func)
{
#if LJ_32
  UNUSED(as);
  if (irref_isk(func))
    return (void *)irf->i;
#else
  if (irref_isk(func)) {
    MCode *p;
    if (irf->o == IR_KINT64)
      p = (MCode *)(void *)ir_k64(irf)->u64;
    else
      p = (MCode *)(void *)(uintptr_t)(uint32_t)irf->i;
    if (p - as->mcp == (int32_t)(p - as->mcp))
      return p;  /* Call target is still in +-2GB range. */
    /* Avoid the indirect case of emit_call(). Try to hoist func addr. */
  }
#endif
  return NULL;
}

static void asm_callx(ASMState *as, IRIns *ir)
{
  IRRef args[CCI_NARGS_MAX*2];
  CCallInfo ci;
  IRRef func;
  IRIns *irf;
  int32_t spadj = 0;
  ci.flags = asm_callx_flags(as, ir);
  asm_collectargs(as, ir, &ci, args);
  asm_setupresult(as, ir, &ci);
#if LJ_32
  /* Have to readjust stack after non-cdecl calls due to callee cleanup. */
  if ((ci.flags & CCI_CC_MASK) != CCI_CC_CDECL)
    spadj = 4 * asm_count_call_slots(as, &ci, args);
#endif
  func = ir->op2; irf = IR(func);
  if (irf->o == IR_CARG) { func = irf->op1; irf = IR(func); }
  ci.func = (ASMFunction)asm_callx_func(as, irf, func);
  if (!(void *)ci.func) {
    /* Use a (hoistable) non-scratch register for indirect calls. */
    RegSet allow = (RSET_GPR & ~RSET_SCRATCH);
    Reg r = ra_alloc1(as, func, allow);
    if (LJ_32) emit_spsub(as, spadj);  /* Above code may cause restores! */
    emit_rr(as, XO_GROUP5, XOg_CALL, r);
  } else if (LJ_32) {
    emit_spsub(as, spadj);
  }
  asm_gencall(as, &ci, args);
}

/* -- Returns ------------------------------------------------------------- */

/* Return to lower frame. Guard that it goes to the right spot. */
static void asm_retf(ASMState *as, IRIns *ir)
{
  Reg base = ra_alloc1(as, REF_BASE, RSET_GPR);
#if LJ_FR2
  Reg rpc = ra_scratch(as, rset_exclude(RSET_GPR, base));
#endif
  void *pc = ir_kptr(IR(ir->op2));
  int32_t delta = 1+LJ_FR2+bc_a(*((const BCIns *)pc - 1));
  as->topslot -= (BCReg)delta;
  if ((int32_t)as->topslot < 0) as->topslot = 0;
  irt_setmark(IR(REF_BASE)->t);  /* Children must not coalesce with BASE reg. */
  emit_setgl(as, base, jit_base);
  emit_addptr(as, base, -8*delta);
  asm_guardcc(as, CC_NE);
#if LJ_FR2
  emit_rmro(as, XO_CMP, rpc|REX_GC64, base, -8);
  emit_loadu64(as, rpc, u64ptr(pc));
#else
  emit_gmroi(as, XG_ARITHi(XOg_CMP), base, -4, ptr2addr(pc));
#endif
}

/* -- Buffer operations --------------------------------------------------- */

#if LJ_HASBUFFER
static void asm_bufhdr_write(ASMState *as, Reg sb)
{
  Reg tmp = ra_scratch(as, rset_exclude(RSET_GPR, sb));
  IRIns irgc;
  irgc.ot = IRT(0, IRT_PGC);  /* GC type. */
  emit_storeofs(as, &irgc, tmp, sb, offsetof(SBuf, L));
  emit_opgl(as, XO_ARITH(XOg_OR), tmp|REX_GC64, cur_L);
  emit_gri(as, XG_ARITHi(XOg_AND), tmp, SBUF_MASK_FLAG);
  emit_loadofs(as, &irgc, tmp, sb, offsetof(SBuf, L));
}
#endif

/* -- Type conversions ---------------------------------------------------- */

static void asm_tointg(ASMState *as, IRIns *ir, Reg left)
{
  Reg tmp = ra_scratch(as, rset_exclude(RSET_FPR, left));
  Reg dest = ra_dest(as, ir, RSET_GPR);
  asm_guardcc(as, CC_P);
  asm_guardcc(as, CC_NE);
  emit_rr(as, XO_UCOMISD, left, tmp);
  emit_rr(as, XO_CVTSI2SD, tmp, dest);
  emit_rr(as, XO_XORPS, tmp, tmp);  /* Avoid partial register stall. */
  checkmclim(as);
  emit_rr(as, XO_CVTTSD2SI, dest, left);
  /* Can't fuse since left is needed twice. */
}

static void asm_tobit(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  Reg tmp = ra_noreg(IR(ir->op1)->r) ?
	      ra_alloc1(as, ir->op1, RSET_FPR) :
	      ra_scratch(as, RSET_FPR);
  Reg right;
  emit_rr(as, XO_MOVDto, tmp, dest);
  right = asm_fuseload(as, ir->op2, rset_exclude(RSET_FPR, tmp));
  emit_mrm(as, XO_ADDSD, tmp, right);
  ra_left(as, tmp, ir->op1);
}

static void asm_conv(ASMState *as, IRIns *ir)
{
  IRType st = (IRType)(ir->op2 & IRCONV_SRCMASK);
  int st64 = (st == IRT_I64 || st == IRT_U64 || (LJ_64 && st == IRT_P64));
  int stfp = (st == IRT_NUM || st == IRT_FLOAT);
  IRRef lref = ir->op1;
  lj_assertA(irt_type(ir->t) != st, "inconsistent types for CONV");
  lj_assertA(!(LJ_32 && (irt_isint64(ir->t) || st64)),
	     "IR %04d has unsplit 64 bit type",
	     (int)(ir - as->ir) - REF_BIAS);
  if (irt_isfp(ir->t)) {
    Reg dest = ra_dest(as, ir, RSET_FPR);
    if (stfp) {  /* FP to FP conversion. */
      Reg left = asm_fuseload(as, lref, RSET_FPR);
      emit_mrm(as, st == IRT_NUM ? XO_CVTSD2SS : XO_CVTSS2SD, dest, left);
      if (left == dest) return;  /* Avoid the XO_XORPS. */
    } else if (LJ_32 && st == IRT_U32) {  /* U32 to FP conversion on x86. */
      /* number = (2^52+2^51 .. u32) - (2^52+2^51) */
      cTValue *k = &as->J->k64[LJ_K64_TOBIT];
      Reg bias = ra_scratch(as, rset_exclude(RSET_FPR, dest));
      if (irt_isfloat(ir->t))
	emit_rr(as, XO_CVTSD2SS, dest, dest);
      emit_rr(as, XO_SUBSD, dest, bias);  /* Subtract 2^52+2^51 bias. */
      emit_rr(as, XO_XORPS, dest, bias);  /* Merge bias and integer. */
      emit_rma(as, XO_MOVSD, bias, k);
      checkmclim(as);
      emit_mrm(as, XO_MOVD, dest, asm_fuseload(as, lref, RSET_GPR));
      return;
    } else {  /* Integer to FP conversion. */
      Reg left = (LJ_64 && (st == IRT_U32 || st == IRT_U64)) ?
		 ra_alloc1(as, lref, RSET_GPR) :
		 asm_fuseloadm(as, lref, RSET_GPR, st64);
      if (LJ_64 && st == IRT_U64) {
	MCLabel l_end = emit_label(as);
	cTValue *k = &as->J->k64[LJ_K64_2P64];
	emit_rma(as, XO_ADDSD, dest, k);  /* Add 2^64 to compensate. */
	emit_sjcc(as, CC_NS, l_end);
	emit_rr(as, XO_TEST, left|REX_64, left);  /* Check if u64 >= 2^63. */
      }
      emit_mrm(as, irt_isnum(ir->t) ? XO_CVTSI2SD : XO_CVTSI2SS,
	       dest|((LJ_64 && (st64 || st == IRT_U32)) ? REX_64 : 0), left);
    }
    emit_rr(as, XO_XORPS, dest, dest);  /* Avoid partial register stall. */
  } else if (stfp) {  /* FP to integer conversion. */
    if (irt_isguard(ir->t)) {
      /* Checked conversions are only supported from number to int. */
      lj_assertA(irt_isint(ir->t) && st == IRT_NUM,
		 "bad type for checked CONV");
      asm_tointg(as, ir, ra_alloc1(as, lref, RSET_FPR));
    } else {
      Reg dest = ra_dest(as, ir, RSET_GPR);
      x86Op op = st == IRT_NUM ? XO_CVTTSD2SI : XO_CVTTSS2SI;
      if (LJ_64 ? irt_isu64(ir->t) : irt_isu32(ir->t)) {
	/* LJ_64: For inputs >= 2^63 add -2^64, convert again. */
	/* LJ_32: For inputs >= 2^31 add -2^31, convert again and add 2^31. */
	Reg tmp = ra_noreg(IR(lref)->r) ? ra_alloc1(as, lref, RSET_FPR) :
					  ra_scratch(as, RSET_FPR);
	MCLabel l_end = emit_label(as);
	if (LJ_32)
	  emit_gri(as, XG_ARITHi(XOg_ADD), dest, (int32_t)0x80000000);
	emit_rr(as, op, dest|REX_64, tmp);
	if (st == IRT_NUM)
	  emit_rma(as, XO_ADDSD, tmp, &as->J->k64[LJ_K64_M2P64_31]);
	else
	  emit_rma(as, XO_ADDSS, tmp, &as->J->k32[LJ_K32_M2P64_31]);
	emit_sjcc(as, CC_NS, l_end);
	emit_rr(as, XO_TEST, dest|REX_64, dest);  /* Check if dest negative. */
	emit_rr(as, op, dest|REX_64, tmp);
	ra_left(as, tmp, lref);
      } else {
	if (LJ_64 && irt_isu32(ir->t))
	  emit_rr(as, XO_MOV, dest, dest);  /* Zero hiword. */
	emit_mrm(as, op,
		 dest|((LJ_64 &&
			(irt_is64(ir->t) || irt_isu32(ir->t))) ? REX_64 : 0),
		 asm_fuseload(as, lref, RSET_FPR));
      }
    }
  } else if (st >= IRT_I8 && st <= IRT_U16) {  /* Extend to 32 bit integer. */
    Reg left, dest = ra_dest(as, ir, RSET_GPR);
    RegSet allow = RSET_GPR;
    x86Op op;
    lj_assertA(irt_isint(ir->t) || irt_isu32(ir->t), "bad type for CONV EXT");
    if (st == IRT_I8) {
      op = XO_MOVSXb; allow = RSET_GPR8; dest |= FORCE_REX;
    } else if (st == IRT_U8) {
      op = XO_MOVZXb; allow = RSET_GPR8; dest |= FORCE_REX;
    } else if (st == IRT_I16) {
      op = XO_MOVSXw;
    } else {
      op = XO_MOVZXw;
    }
    left = asm_fuseload(as, lref, allow);
    /* Add extra MOV if source is already in wrong register. */
    if (!LJ_64 && left != RID_MRM && !rset_test(allow, left)) {
      Reg tmp = ra_scratch(as, allow);
      emit_rr(as, op, dest, tmp);
      emit_rr(as, XO_MOV, tmp, left);
    } else {
      emit_mrm(as, op, dest, left);
    }
  } else {  /* 32/64 bit integer conversions. */
    if (LJ_32) {  /* Only need to handle 32/32 bit no-op (cast) on x86. */
      Reg dest = ra_dest(as, ir, RSET_GPR);
      ra_left(as, dest, lref);  /* Do nothing, but may need to move regs. */
    } else if (irt_is64(ir->t)) {
      Reg dest = ra_dest(as, ir, RSET_GPR);
      if (st64 || !(ir->op2 & IRCONV_SEXT)) {
	/* 64/64 bit no-op (cast) or 32 to 64 bit zero extension. */
	ra_left(as, dest, lref);  /* Do nothing, but may need to move regs. */
      } else {  /* 32 to 64 bit sign extension. */
	Reg left = asm_fuseload(as, lref, RSET_GPR);
	emit_mrm(as, XO_MOVSXd, dest|REX_64, left);
      }
    } else {
      Reg dest = ra_dest(as, ir, RSET_GPR);
      if (st64 && !(ir->op2 & IRCONV_NONE)) {
	Reg left = asm_fuseload(as, lref, RSET_GPR);
	/* This is either a 32 bit reg/reg mov which zeroes the hiword
	** or a load of the loword from a 64 bit address.
	*/
	emit_mrm(as, XO_MOV, dest, left);
      } else {  /* 32/32 bit no-op (cast). */
	ra_left(as, dest, lref);  /* Do nothing, but may need to move regs. */
      }
    }
  }
}

#if LJ_32 && LJ_HASFFI
/* No SSE conversions to/from 64 bit on x86, so resort to ugly x87 code. */

/* 64 bit integer to FP conversion in 32 bit mode. */
static void asm_conv_fp_int64(ASMState *as, IRIns *ir)
{
  Reg hi = ra_alloc1(as, ir->op1, RSET_GPR);
  Reg lo = ra_alloc1(as, (ir-1)->op1, rset_exclude(RSET_GPR, hi));
  int32_t ofs = sps_scale(ir->s);  /* Use spill slot or temp slots. */
  Reg dest = ir->r;
  if (ra_hasreg(dest)) {
    ra_free(as, dest);
    ra_modified(as, dest);
    emit_rmro(as, irt_isnum(ir->t) ? XO_MOVSD : XO_MOVSS, dest, RID_ESP, ofs);
  }
  emit_rmro(as, irt_isnum(ir->t) ? XO_FSTPq : XO_FSTPd,
	    irt_isnum(ir->t) ? XOg_FSTPq : XOg_FSTPd, RID_ESP, ofs);
  if (((ir-1)->op2 & IRCONV_SRCMASK) == IRT_U64) {
    /* For inputs in [2^63,2^64-1] add 2^64 to compensate. */
    MCLabel l_end = emit_label(as);
    emit_rma(as, XO_FADDq, XOg_FADDq, &as->J->k64[LJ_K64_2P64]);
    emit_sjcc(as, CC_NS, l_end);
    emit_rr(as, XO_TEST, hi, hi);  /* Check if u64 >= 2^63. */
  } else {
    lj_assertA(((ir-1)->op2 & IRCONV_SRCMASK) == IRT_I64, "bad type for CONV");
  }
  emit_rmro(as, XO_FILDq, XOg_FILDq, RID_ESP, 0);
  /* NYI: Avoid narrow-to-wide store-to-load forwarding stall. */
  emit_rmro(as, XO_MOVto, hi, RID_ESP, 4);
  emit_rmro(as, XO_MOVto, lo, RID_ESP, 0);
}

/* FP to 64 bit integer conversion in 32 bit mode. */
static void asm_conv_int64_fp(ASMState *as, IRIns *ir)
{
  IRType st = (IRType)((ir-1)->op2 & IRCONV_SRCMASK);
  IRType dt = (((ir-1)->op2 & IRCONV_DSTMASK) >> IRCONV_DSH);
  Reg lo, hi;
  lj_assertA(st == IRT_NUM || st == IRT_FLOAT, "bad type for CONV");
  lj_assertA(dt == IRT_I64 || dt == IRT_U64, "bad type for CONV");
  hi = ra_dest(as, ir, RSET_GPR);
  lo = ra_dest(as, ir-1, rset_exclude(RSET_GPR, hi));
  if (ra_used(ir-1)) emit_rmro(as, XO_MOV, lo, RID_ESP, 0);
  /* NYI: Avoid wide-to-narrow store-to-load forwarding stall. */
  if (!(as->flags & JIT_F_SSE3)) {  /* Set FPU rounding mode to default. */
    emit_rmro(as, XO_FLDCW, XOg_FLDCW, RID_ESP, 4);
    emit_rmro(as, XO_MOVto, lo, RID_ESP, 4);
    emit_gri(as, XG_ARITHi(XOg_AND), lo, 0xf3ff);
  }
  if (dt == IRT_U64) {
    /* For inputs in [2^63,2^64-1] add -2^64 and convert again. */
    MCLabel l_pop, l_end = emit_label(as);
    emit_x87op(as, XI_FPOP);
    l_pop = emit_label(as);
    emit_sjmp(as, l_end);
    emit_rmro(as, XO_MOV, hi, RID_ESP, 4);
    if ((as->flags & JIT_F_SSE3))
      emit_rmro(as, XO_FISTTPq, XOg_FISTTPq, RID_ESP, 0);
    else
      emit_rmro(as, XO_FISTPq, XOg_FISTPq, RID_ESP, 0);
    emit_rma(as, XO_FADDq, XOg_FADDq, &as->J->k64[LJ_K64_M2P64]);
    emit_sjcc(as, CC_NS, l_pop);
    emit_rr(as, XO_TEST, hi, hi);  /* Check if out-of-range (2^63). */
  }
  emit_rmro(as, XO_MOV, hi, RID_ESP, 4);
  if ((as->flags & JIT_F_SSE3)) {  /* Truncation is easy with SSE3. */
    emit_rmro(as, XO_FISTTPq, XOg_FISTTPq, RID_ESP, 0);
  } else {  /* Otherwise set FPU rounding mode to truncate before the store. */
    emit_rmro(as, XO_FISTPq, XOg_FISTPq, RID_ESP, 0);
    emit_rmro(as, XO_FLDCW, XOg_FLDCW, RID_ESP, 0);
    emit_rmro(as, XO_MOVtow, lo, RID_ESP, 0);
    emit_rmro(as, XO_ARITHw(XOg_OR), lo, RID_ESP, 0);
    emit_loadi(as, lo, 0xc00);
    emit_rmro(as, XO_FNSTCW, XOg_FNSTCW, RID_ESP, 0);
  }
  if (dt == IRT_U64)
    emit_x87op(as, XI_FDUP);
  emit_mrm(as, st == IRT_NUM ? XO_FLDq : XO_FLDd,
	   st == IRT_NUM ? XOg_FLDq: XOg_FLDd,
	   asm_fuseload(as, ir->op1, RSET_EMPTY));
}

static void asm_conv64(ASMState *as, IRIns *ir)
{
  if (irt_isfp(ir->t))
    asm_conv_fp_int64(as, ir);
  else
    asm_conv_int64_fp(as, ir);
}
#endif

static void asm_strto(ASMState *as, IRIns *ir)
{
  /* Force a spill slot for the destination register (if any). */
  const CCallInfo *ci = &lj_ir_callinfo[IRCALL_lj_strscan_num];
  IRRef args[2];
  RegSet drop = RSET_SCRATCH;
  if ((drop & RSET_FPR) != RSET_FPR && ra_hasreg(ir->r))
    rset_set(drop, ir->r);  /* WIN64 doesn't spill all FPRs. */
  ra_evictset(as, drop);
  asm_guardcc(as, CC_E);
  emit_rr(as, XO_TEST, RID_RET, RID_RET);  /* Test return status. */
  args[0] = ir->op1;      /* GCstr *str */
  args[1] = ASMREF_TMP1;  /* TValue *n  */
  asm_gencall(as, ci, args);
  /* Store the result to the spill slot or temp slots. */
  emit_rmro(as, XO_LEA, ra_releasetmp(as, ASMREF_TMP1)|REX_64,
	    RID_ESP, sps_scale(ir->s));
}

/* -- Memory references --------------------------------------------------- */

/* Get pointer to TValue. */
static void asm_tvptr(ASMState *as, Reg dest, IRRef ref, MSize mode)
{
  if ((mode & IRTMPREF_IN1)) {
    IRIns *ir = IR(ref);
    if (irt_isnum(ir->t)) {
      if (irref_isk(ref) && !(mode & IRTMPREF_OUT1)) {
	/* Use the number constant itself as a TValue. */
	emit_loada(as, dest, ir_knum(ir));
	return;
      }
      emit_rmro(as, XO_MOVSDto, ra_alloc1(as, ref, RSET_FPR), dest, 0);
    } else {
#if LJ_GC64
      if (irref_isk(ref)) {
	TValue k;
	lj_ir_kvalue(as->J->L, &k, ir);
	emit_movmroi(as, dest, 4, k.u32.hi);
	emit_movmroi(as, dest, 0, k.u32.lo);
      } else {
	/* TODO: 64 bit store + 32 bit load-modify-store is suboptimal. */
	Reg src = ra_alloc1(as, ref, rset_exclude(RSET_GPR, dest));
	if (irt_is64(ir->t)) {
	  emit_u32(as, irt_toitype(ir->t) << 15);
	  emit_rmro(as, XO_ARITHi, XOg_OR, dest, 4);
	} else {
	  emit_movmroi(as, dest, 4, (irt_toitype(ir->t) << 15));
	}
	emit_movtomro(as, REX_64IR(ir, src), dest, 0);
      }
#else
      if (!irref_isk(ref)) {
	Reg src = ra_alloc1(as, ref, rset_exclude(RSET_GPR, dest));
	emit_movtomro(as, REX_64IR(ir, src), dest, 0);
      } else if (!irt_ispri(ir->t)) {
	emit_movmroi(as, dest, 0, ir->i);
      }
      if (!(LJ_64 && irt_islightud(ir->t)))
	emit_movmroi(as, dest, 4, irt_toitype(ir->t));
#endif
    }
  }
  emit_loada(as, dest, &J2G(as->J)->tmptv); /* g->tmptv holds the TValue(s). */
}

static void asm_aref(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  asm_fusearef(as, ir, RSET_GPR);
  if (!(as->mrm.idx == RID_NONE && as->mrm.ofs == 0))
    emit_mrm(as, XO_LEA, dest|REX_GC64, RID_MRM);
  else if (as->mrm.base != dest)
    emit_rr(as, XO_MOV, dest|REX_GC64, as->mrm.base);
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
  Reg key = RID_NONE, tmp = RID_NONE;
  IRIns *irkey = IR(ir->op2);
  int isk = irref_isk(ir->op2);
  IRType1 kt = irkey->t;
  uint32_t khash;
  MCLabel l_end, l_loop, l_next;

  if (!isk) {
    rset_clear(allow, tab);
    key = ra_alloc1(as, ir->op2, irt_isnum(kt) ? RSET_FPR : allow);
    if (LJ_GC64 || !irt_isstr(kt))
      tmp = ra_scratch(as, rset_exclude(allow, key));
  }

  /* Key not found in chain: jump to exit (if merged) or load niltv. */
  l_end = emit_label(as);
  if (merge == IR_NE)
    asm_guardcc(as, CC_E);  /* XI_JMP is not found by lj_asm_patchexit. */
  else if (destused)
    emit_loada(as, dest, niltvg(J2G(as->J)));

  /* Follow hash chain until the end. */
  l_loop = emit_sjcc_label(as, CC_NZ);
  emit_rr(as, XO_TEST, dest|REX_GC64, dest);
  emit_rmro(as, XO_MOV, dest|REX_GC64, dest, offsetof(Node, next));
  l_next = emit_label(as);

  /* Type and value comparison. */
  if (merge == IR_EQ)
    asm_guardcc(as, CC_E);
  else
    emit_sjcc(as, CC_E, l_end);
  checkmclim(as);
  if (irt_isnum(kt)) {
    if (isk) {
      /* Assumes -0.0 is already canonicalized to +0.0. */
      emit_gmroi(as, XG_ARITHi(XOg_CMP), dest, offsetof(Node, key.u32.lo),
		 (int32_t)ir_knum(irkey)->u32.lo);
      emit_sjcc(as, CC_NE, l_next);
      emit_gmroi(as, XG_ARITHi(XOg_CMP), dest, offsetof(Node, key.u32.hi),
		 (int32_t)ir_knum(irkey)->u32.hi);
    } else {
      emit_sjcc(as, CC_P, l_next);
      emit_rmro(as, XO_UCOMISD, key, dest, offsetof(Node, key.n));
      emit_sjcc(as, CC_AE, l_next);
      /* The type check avoids NaN penalties and complaints from Valgrind. */
#if LJ_64 && !LJ_GC64
      emit_u32(as, LJ_TISNUM);
      emit_rmro(as, XO_ARITHi, XOg_CMP, dest, offsetof(Node, key.it));
#else
      emit_i8(as, LJ_TISNUM);
      emit_rmro(as, XO_ARITHi8, XOg_CMP, dest, offsetof(Node, key.it));
#endif
    }
#if LJ_64 && !LJ_GC64
  } else if (irt_islightud(kt)) {
    emit_rmro(as, XO_CMP, key|REX_64, dest, offsetof(Node, key.u64));
#endif
#if LJ_GC64
  } else if (irt_isaddr(kt)) {
    if (isk) {
      TValue k;
      k.u64 = ((uint64_t)irt_toitype(irkey->t) << 47) | irkey[1].tv.u64;
      emit_gmroi(as, XG_ARITHi(XOg_CMP), dest, offsetof(Node, key.u32.lo),
		 k.u32.lo);
      emit_sjcc(as, CC_NE, l_next);
      emit_gmroi(as, XG_ARITHi(XOg_CMP), dest, offsetof(Node, key.u32.hi),
		 k.u32.hi);
    } else {
      emit_rmro(as, XO_CMP, tmp|REX_64, dest, offsetof(Node, key.u64));
    }
  } else {
    lj_assertA(irt_ispri(kt) && !irt_isnil(kt), "bad HREF key type");
    emit_u32(as, (irt_toitype(kt)<<15)|0x7fff);
    emit_rmro(as, XO_ARITHi, XOg_CMP, dest, offsetof(Node, key.it));
#else
  } else {
    if (!irt_ispri(kt)) {
      lj_assertA(irt_isaddr(kt), "bad HREF key type");
      if (isk)
	emit_gmroi(as, XG_ARITHi(XOg_CMP), dest, offsetof(Node, key.gcr),
		   ptr2addr(ir_kgc(irkey)));
      else
	emit_rmro(as, XO_CMP, key, dest, offsetof(Node, key.gcr));
      emit_sjcc(as, CC_NE, l_next);
    }
    lj_assertA(!irt_isnil(kt), "bad HREF key type");
    emit_i8(as, irt_toitype(kt));
    emit_rmro(as, XO_ARITHi8, XOg_CMP, dest, offsetof(Node, key.it));
#endif
  }
  emit_sfixup(as, l_loop);
#if LJ_GC64
  if (!isk && irt_isaddr(kt)) {
    emit_rr(as, XO_OR, tmp|REX_64, key);
    emit_loadu64(as, tmp, (uint64_t)irt_toitype(kt) << 47);
  }
#endif

  /* Load main position relative to tab->node into dest. */
  khash = isk ? ir_khash(as, irkey) : 1;
  if (khash == 0) {
    emit_rmro(as, XO_MOV, dest|REX_GC64, tab, offsetof(GCtab, node));
  } else {
    emit_rmro(as, XO_ARITH(XOg_ADD), dest|REX_GC64, tab, offsetof(GCtab,node));
    emit_shifti(as, XOg_SHL, dest, 3);
    emit_rmrxo(as, XO_LEA, dest, dest, dest, XM_SCALE2, 0);
    if (isk) {
      emit_gri(as, XG_ARITHi(XOg_AND), dest, (int32_t)khash);
      emit_rmro(as, XO_MOV, dest, tab, offsetof(GCtab, hmask));
    } else if (irt_isstr(kt)) {
      emit_rmro(as, XO_ARITH(XOg_AND), dest, key, offsetof(GCstr, sid));
      emit_rmro(as, XO_MOV, dest, tab, offsetof(GCtab, hmask));
    } else {  /* Must match with hashrot() in lj_tab.c. */
      emit_rmro(as, XO_ARITH(XOg_AND), dest, tab, offsetof(GCtab, hmask));
      emit_rr(as, XO_ARITH(XOg_SUB), dest, tmp);
      emit_shifti(as, XOg_ROL, tmp, HASH_ROT3);
      emit_rr(as, XO_ARITH(XOg_XOR), dest, tmp);
      checkmclim(as);
      emit_shifti(as, XOg_ROL, dest, HASH_ROT2);
      emit_rr(as, XO_ARITH(XOg_SUB), tmp, dest);
      emit_shifti(as, XOg_ROL, dest, HASH_ROT1);
      emit_rr(as, XO_ARITH(XOg_XOR), tmp, dest);
      if (irt_isnum(kt)) {
	emit_rr(as, XO_ARITH(XOg_ADD), dest, dest);
#if LJ_64
	emit_shifti(as, XOg_SHR|REX_64, dest, 32);
	emit_rr(as, XO_MOV, tmp, dest);
	emit_rr(as, XO_MOVDto, key|REX_64, dest);
#else
	emit_rmro(as, XO_MOV, dest, RID_ESP, ra_spill(as, irkey)+4);
	emit_rr(as, XO_MOVDto, key, tmp);
#endif
      } else {
	emit_rr(as, XO_MOV, tmp, key);
#if LJ_GC64
	emit_gri(as, XG_ARITHi(XOg_XOR), dest, irt_toitype(kt) << 15);
	if ((as->flags & JIT_F_BMI2)) {
	  emit_i8(as, 32);
	  emit_mrm(as, XV_RORX|VEX_64, dest, key);
	} else {
	  emit_shifti(as, XOg_SHR|REX_64, dest, 32);
	  emit_rr(as, XO_MOV, dest|REX_64, key|REX_64);
	}
#else
	emit_rmro(as, XO_LEA, dest, key, HASH_BIAS);
#endif
      }
    }
  }
}

static void asm_hrefk(ASMState *as, IRIns *ir)
{
  IRIns *kslot = IR(ir->op2);
  IRIns *irkey = IR(kslot->op1);
  int32_t ofs = (int32_t)(kslot->op2 * sizeof(Node));
  Reg dest = ra_used(ir) ? ra_dest(as, ir, RSET_GPR) : RID_NONE;
  Reg node = ra_alloc1(as, ir->op1, RSET_GPR);
#if !LJ_64
  MCLabel l_exit;
#endif
  lj_assertA(ofs % sizeof(Node) == 0, "unaligned HREFK slot");
  if (ra_hasreg(dest)) {
    if (ofs != 0) {
      if (dest == node)
	emit_gri(as, XG_ARITHi(XOg_ADD), dest|REX_GC64, ofs);
      else
	emit_rmro(as, XO_LEA, dest|REX_GC64, node, ofs);
    } else if (dest != node) {
      emit_rr(as, XO_MOV, dest|REX_GC64, node);
    }
  }
  asm_guardcc(as, CC_NE);
#if LJ_64
  if (!irt_ispri(irkey->t)) {
    Reg key = ra_scratch(as, rset_exclude(RSET_GPR, node));
    emit_rmro(as, XO_CMP, key|REX_64, node,
	       ofs + (int32_t)offsetof(Node, key.u64));
    lj_assertA(irt_isnum(irkey->t) || irt_isgcv(irkey->t),
	       "bad HREFK key type");
    /* Assumes -0.0 is already canonicalized to +0.0. */
    emit_loadu64(as, key, irt_isnum(irkey->t) ? ir_knum(irkey)->u64 :
#if LJ_GC64
			  ((uint64_t)irt_toitype(irkey->t) << 47) |
			  (uint64_t)ir_kgc(irkey));
#else
			  ((uint64_t)irt_toitype(irkey->t) << 32) |
			  (uint64_t)(uint32_t)ptr2addr(ir_kgc(irkey)));
#endif
  } else {
    lj_assertA(!irt_isnil(irkey->t), "bad HREFK key type");
#if LJ_GC64
    emit_i32(as, (irt_toitype(irkey->t)<<15)|0x7fff);
    emit_rmro(as, XO_ARITHi, XOg_CMP, node,
	      ofs + (int32_t)offsetof(Node, key.it));
#else
    emit_i8(as, irt_toitype(irkey->t));
    emit_rmro(as, XO_ARITHi8, XOg_CMP, node,
	      ofs + (int32_t)offsetof(Node, key.it));
#endif
  }
#else
  l_exit = emit_label(as);
  if (irt_isnum(irkey->t)) {
    /* Assumes -0.0 is already canonicalized to +0.0. */
    emit_gmroi(as, XG_ARITHi(XOg_CMP), node,
	       ofs + (int32_t)offsetof(Node, key.u32.lo),
	       (int32_t)ir_knum(irkey)->u32.lo);
    emit_sjcc(as, CC_NE, l_exit);
    emit_gmroi(as, XG_ARITHi(XOg_CMP), node,
	       ofs + (int32_t)offsetof(Node, key.u32.hi),
	       (int32_t)ir_knum(irkey)->u32.hi);
  } else {
    if (!irt_ispri(irkey->t)) {
      lj_assertA(irt_isgcv(irkey->t), "bad HREFK key type");
      emit_gmroi(as, XG_ARITHi(XOg_CMP), node,
		 ofs + (int32_t)offsetof(Node, key.gcr),
		 ptr2addr(ir_kgc(irkey)));
      emit_sjcc(as, CC_NE, l_exit);
    }
    lj_assertA(!irt_isnil(irkey->t), "bad HREFK key type");
    emit_i8(as, irt_toitype(irkey->t));
    emit_rmro(as, XO_ARITHi8, XOg_CMP, node,
	      ofs + (int32_t)offsetof(Node, key.it));
  }
#endif
}

static void asm_uref(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  int guarded = (irt_t(ir->t) & (IRT_GUARD|IRT_TYPE)) == (IRT_GUARD|IRT_PGC);
  if (irref_isk(ir->op1) && !guarded) {
    GCfunc *fn = ir_kfunc(IR(ir->op1));
    MRef *v = &gcref(fn->l.uvptr[(ir->op2 >> 8)])->uv.v;
    emit_rma(as, XO_MOV, dest|REX_GC64, v);
  } else {
    Reg uv = ra_scratch(as, RSET_GPR);
    if (ir->o == IR_UREFC)
      emit_rmro(as, XO_LEA, dest|REX_GC64, uv, offsetof(GCupval, tv));
    else
      emit_rmro(as, XO_MOV, dest|REX_GC64, uv, offsetof(GCupval, v));
    if (guarded) {
      asm_guardcc(as, ir->o == IR_UREFC ? CC_E : CC_NE);
      emit_i8(as, 0);
      emit_rmro(as, XO_ARITHib, XOg_CMP, uv, offsetof(GCupval, closed));
    }
    if (irref_isk(ir->op1)) {
      GCfunc *fn = ir_kfunc(IR(ir->op1));
      GCobj *o = gcref(fn->l.uvptr[(ir->op2 >> 8)]);
      emit_loada(as, uv, o);
    } else {
      emit_rmro(as, XO_MOV, uv|REX_GC64, ra_alloc1(as, ir->op1, RSET_GPR),
	        (int32_t)offsetof(GCfuncL, uvptr) +
	        (int32_t)sizeof(MRef) * (int32_t)(ir->op2 >> 8));
    }
  }
}

static void asm_fref(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  asm_fusefref(as, ir, RSET_GPR);
  emit_mrm(as, XO_LEA, dest, RID_MRM);
}

static void asm_strref(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  asm_fusestrref(as, ir, RSET_GPR);
  if (as->mrm.base == RID_NONE)
    emit_loadi(as, dest, as->mrm.ofs);
  else if (as->mrm.base == dest && as->mrm.idx == RID_NONE)
    emit_gri(as, XG_ARITHi(XOg_ADD), dest|REX_GC64, as->mrm.ofs);
  else
    emit_mrm(as, XO_LEA, dest|REX_GC64, RID_MRM);
}

/* -- Loads and stores ---------------------------------------------------- */

static void asm_fxload(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, irt_isfp(ir->t) ? RSET_FPR : RSET_GPR);
  x86Op xo;
  if (ir->o == IR_FLOAD)
    asm_fusefref(as, ir, RSET_GPR);
  else
    asm_fusexref(as, ir->op1, RSET_GPR);
  /* ir->op2 is ignored -- unaligned loads are ok on x86. */
  switch (irt_type(ir->t)) {
  case IRT_I8: xo = XO_MOVSXb; break;
  case IRT_U8: xo = XO_MOVZXb; break;
  case IRT_I16: xo = XO_MOVSXw; break;
  case IRT_U16: xo = XO_MOVZXw; break;
  case IRT_NUM: xo = XO_MOVSD; break;
  case IRT_FLOAT: xo = XO_MOVSS; break;
  default:
    if (LJ_64 && irt_is64(ir->t))
      dest |= REX_64;
    else
      lj_assertA(irt_isint(ir->t) || irt_isu32(ir->t) || irt_isaddr(ir->t),
		 "unsplit 64 bit load");
    xo = XO_MOV;
    break;
  }
  emit_mrm(as, xo, dest, RID_MRM);
}

#define asm_fload(as, ir)	asm_fxload(as, ir)
#define asm_xload(as, ir)	asm_fxload(as, ir)

static void asm_fxstore(ASMState *as, IRIns *ir)
{
  RegSet allow = RSET_GPR;
  Reg src = RID_NONE, osrc = RID_NONE;
  int32_t k = 0;
  if (ir->r == RID_SINK)
    return;
  /* The IRT_I16/IRT_U16 stores should never be simplified for constant
  ** values since mov word [mem], imm16 has a length-changing prefix.
  */
  if (irt_isi16(ir->t) || irt_isu16(ir->t) || irt_isfp(ir->t) ||
      !asm_isk32(as, ir->op2, &k)) {
    RegSet allow8 = irt_isfp(ir->t) ? RSET_FPR :
		    (irt_isi8(ir->t) || irt_isu8(ir->t)) ? RSET_GPR8 : RSET_GPR;
    src = osrc = ra_alloc1(as, ir->op2, allow8);
    if (!LJ_64 && !rset_test(allow8, src)) {  /* Already in wrong register. */
      rset_clear(allow, osrc);
      src = ra_scratch(as, allow8);
    }
    rset_clear(allow, src);
  }
  if (ir->o == IR_FSTORE) {
    asm_fusefref(as, IR(ir->op1), allow);
  } else {
    asm_fusexref(as, ir->op1, allow);
    if (LJ_32 && ir->o == IR_HIOP) as->mrm.ofs += 4;
  }
  if (ra_hasreg(src)) {
    x86Op xo;
    switch (irt_type(ir->t)) {
    case IRT_I8: case IRT_U8: xo = XO_MOVtob; src |= FORCE_REX; break;
    case IRT_I16: case IRT_U16: xo = XO_MOVtow; break;
    case IRT_NUM: xo = XO_MOVSDto; break;
    case IRT_FLOAT: xo = XO_MOVSSto; break;
#if LJ_64 && !LJ_GC64
    case IRT_LIGHTUD:
      /* NYI: mask 64 bit lightuserdata. */
      lj_assertA(0, "store of lightuserdata");
#endif
    default:
      if (LJ_64 && irt_is64(ir->t))
	src |= REX_64;
      else
	lj_assertA(irt_isint(ir->t) || irt_isu32(ir->t) || irt_isaddr(ir->t),
		   "unsplit 64 bit store");
      xo = XO_MOVto;
      break;
    }
    emit_mrm(as, xo, src, RID_MRM);
    if (!LJ_64 && src != osrc) {
      ra_noweak(as, osrc);
      emit_rr(as, XO_MOV, src, osrc);
    }
  } else {
    if (irt_isi8(ir->t) || irt_isu8(ir->t)) {
      emit_i8(as, k);
      emit_mrm(as, XO_MOVmib, 0, RID_MRM);
    } else {
      lj_assertA(irt_is64(ir->t) || irt_isint(ir->t) || irt_isu32(ir->t) ||
		 irt_isaddr(ir->t), "bad store type");
      emit_i32(as, k);
      emit_mrm(as, XO_MOVmi, REX_64IR(ir, 0), RID_MRM);
    }
  }
}

#define asm_fstore(as, ir)	asm_fxstore(as, ir)
#define asm_xstore(as, ir)	asm_fxstore(as, ir)

#if LJ_64 && !LJ_GC64
static Reg asm_load_lightud64(ASMState *as, IRIns *ir, int typecheck)
{
  if (ra_used(ir) || typecheck) {
    Reg dest = ra_dest(as, ir, RSET_GPR);
    if (typecheck) {
      Reg tmp = ra_scratch(as, rset_exclude(RSET_GPR, dest));
      asm_guardcc(as, CC_NE);
      emit_i8(as, -2);
      emit_rr(as, XO_ARITHi8, XOg_CMP, tmp);
      emit_shifti(as, XOg_SAR|REX_64, tmp, 47);
      emit_rr(as, XO_MOV, tmp|REX_64, dest);
    }
    return dest;
  } else {
    return RID_NONE;
  }
}
#endif

static void asm_ahuvload(ASMState *as, IRIns *ir)
{
#if LJ_GC64
  Reg tmp = RID_NONE;
#endif
  lj_assertA(irt_isnum(ir->t) || irt_ispri(ir->t) || irt_isaddr(ir->t) ||
	     (LJ_DUALNUM && irt_isint(ir->t)),
	     "bad load type %d", irt_type(ir->t));
#if LJ_64 && !LJ_GC64
  if (irt_islightud(ir->t)) {
    Reg dest = asm_load_lightud64(as, ir, 1);
    if (ra_hasreg(dest)) {
      checkmclim(as);
      asm_fuseahuref(as, ir->op1, RSET_GPR);
      if (ir->o == IR_VLOAD) as->mrm.ofs += 8 * ir->op2;
      emit_mrm(as, XO_MOV, dest|REX_64, RID_MRM);
    }
    return;
  } else
#endif
  if (ra_used(ir)) {
    RegSet allow = irt_isnum(ir->t) ? RSET_FPR : RSET_GPR;
    Reg dest = ra_dest(as, ir, allow);
    asm_fuseahuref(as, ir->op1, RSET_GPR);
    if (ir->o == IR_VLOAD) as->mrm.ofs += 8 * ir->op2;
#if LJ_GC64
    if (irt_isaddr(ir->t)) {
      emit_shifti(as, XOg_SHR|REX_64, dest, 17);
      asm_guardcc(as, CC_NE);
      emit_i8(as, irt_toitype(ir->t));
      emit_rr(as, XO_ARITHi8, XOg_CMP, dest);
      emit_i8(as, XI_O16);
      if ((as->flags & JIT_F_BMI2)) {
	emit_i8(as, 47);
	emit_mrm(as, XV_RORX|VEX_64, dest, RID_MRM);
      } else {
	emit_shifti(as, XOg_ROR|REX_64, dest, 47);
	emit_mrm(as, XO_MOV, dest|REX_64, RID_MRM);
      }
      return;
    } else
#endif
    emit_mrm(as, dest < RID_MAX_GPR ? XO_MOV : XO_MOVSD, dest, RID_MRM);
  } else {
    RegSet gpr = RSET_GPR;
#if LJ_GC64
    if (irt_isaddr(ir->t)) {
      tmp = ra_scratch(as, RSET_GPR);
      gpr = rset_exclude(gpr, tmp);
    }
#endif
    asm_fuseahuref(as, ir->op1, gpr);
    if (ir->o == IR_VLOAD) as->mrm.ofs += 8 * ir->op2;
  }
  /* Always do the type check, even if the load result is unused. */
  as->mrm.ofs += 4;
  asm_guardcc(as, irt_isnum(ir->t) ? CC_AE : CC_NE);
  if (LJ_64 && irt_type(ir->t) >= IRT_NUM) {
    lj_assertA(irt_isinteger(ir->t) || irt_isnum(ir->t),
	       "bad load type %d", irt_type(ir->t));
    checkmclim(as);
#if LJ_GC64
    emit_u32(as, LJ_TISNUM << 15);
#else
    emit_u32(as, LJ_TISNUM);
#endif
    emit_mrm(as, XO_ARITHi, XOg_CMP, RID_MRM);
#if LJ_GC64
  } else if (irt_isaddr(ir->t)) {
    as->mrm.ofs -= 4;
    emit_i8(as, irt_toitype(ir->t));
    emit_mrm(as, XO_ARITHi8, XOg_CMP, tmp);
    emit_shifti(as, XOg_SAR|REX_64, tmp, 47);
    emit_mrm(as, XO_MOV, tmp|REX_64, RID_MRM);
  } else if (irt_isnil(ir->t)) {
    as->mrm.ofs -= 4;
    emit_i8(as, -1);
    emit_mrm(as, XO_ARITHi8, XOg_CMP|REX_64, RID_MRM);
  } else {
    emit_u32(as, (irt_toitype(ir->t) << 15) | 0x7fff);
    emit_mrm(as, XO_ARITHi, XOg_CMP, RID_MRM);
#else
  } else {
    emit_i8(as, irt_toitype(ir->t));
    emit_mrm(as, XO_ARITHi8, XOg_CMP, RID_MRM);
#endif
  }
}

static void asm_ahustore(ASMState *as, IRIns *ir)
{
  if (ir->r == RID_SINK)
    return;
  if (irt_isnum(ir->t)) {
    Reg src = ra_alloc1(as, ir->op2, RSET_FPR);
    asm_fuseahuref(as, ir->op1, RSET_GPR);
    emit_mrm(as, XO_MOVSDto, src, RID_MRM);
#if LJ_64 && !LJ_GC64
  } else if (irt_islightud(ir->t)) {
    Reg src = ra_alloc1(as, ir->op2, RSET_GPR);
    asm_fuseahuref(as, ir->op1, rset_exclude(RSET_GPR, src));
    emit_mrm(as, XO_MOVto, src|REX_64, RID_MRM);
#endif
#if LJ_GC64
  } else if (irref_isk(ir->op2)) {
    TValue k;
    lj_ir_kvalue(as->J->L, &k, IR(ir->op2));
    asm_fuseahuref(as, ir->op1, RSET_GPR);
    if (tvisnil(&k)) {
      emit_i32(as, -1);
      emit_mrm(as, XO_MOVmi, REX_64, RID_MRM);
    } else {
      emit_u32(as, k.u32.lo);
      emit_mrm(as, XO_MOVmi, 0, RID_MRM);
      as->mrm.ofs += 4;
      emit_u32(as, k.u32.hi);
      emit_mrm(as, XO_MOVmi, 0, RID_MRM);
    }
#endif
  } else {
    IRIns *irr = IR(ir->op2);
    RegSet allow = RSET_GPR;
    Reg src = RID_NONE;
    if (!irref_isk(ir->op2)) {
      src = ra_alloc1(as, ir->op2, allow);
      rset_clear(allow, src);
    }
    asm_fuseahuref(as, ir->op1, allow);
    if (ra_hasreg(src)) {
#if LJ_GC64
      if (!(LJ_DUALNUM && irt_isinteger(ir->t))) {
	/* TODO: 64 bit store + 32 bit load-modify-store is suboptimal. */
	as->mrm.ofs += 4;
	emit_u32(as, irt_toitype(ir->t) << 15);
	emit_mrm(as, XO_ARITHi, XOg_OR, RID_MRM);
	as->mrm.ofs -= 4;
	emit_mrm(as, XO_MOVto, src|REX_64, RID_MRM);
	return;
      }
#endif
      emit_mrm(as, XO_MOVto, src, RID_MRM);
    } else if (!irt_ispri(irr->t)) {
      lj_assertA(irt_isaddr(ir->t) || (LJ_DUALNUM && irt_isinteger(ir->t)),
		 "bad store type");
      emit_i32(as, irr->i);
      emit_mrm(as, XO_MOVmi, 0, RID_MRM);
    }
    as->mrm.ofs += 4;
#if LJ_GC64
    lj_assertA(LJ_DUALNUM && irt_isinteger(ir->t), "bad store type");
    emit_i32(as, LJ_TNUMX << 15);
#else
    emit_i32(as, (int32_t)irt_toitype(ir->t));
#endif
    emit_mrm(as, XO_MOVmi, 0, RID_MRM);
  }
}

static void asm_sload(ASMState *as, IRIns *ir)
{
  int32_t ofs = 8*((int32_t)ir->op1-1-LJ_FR2) +
		(!LJ_FR2 && (ir->op2 & IRSLOAD_FRAME) ? 4 : 0);
  IRType1 t = ir->t;
  Reg base;
  lj_assertA(!(ir->op2 & IRSLOAD_PARENT),
	     "bad parent SLOAD"); /* Handled by asm_head_side(). */
  lj_assertA(irt_isguard(t) || !(ir->op2 & IRSLOAD_TYPECHECK),
	     "inconsistent SLOAD variant");
  lj_assertA(LJ_DUALNUM ||
	     !irt_isint(t) ||
	     (ir->op2 & (IRSLOAD_CONVERT|IRSLOAD_FRAME|IRSLOAD_KEYINDEX)),
	     "bad SLOAD type");
  if ((ir->op2 & IRSLOAD_CONVERT) && irt_isguard(t) && irt_isint(t)) {
    Reg left = ra_scratch(as, RSET_FPR);
    asm_tointg(as, ir, left);  /* Frees dest reg. Do this before base alloc. */
    base = ra_alloc1(as, REF_BASE, RSET_GPR);
    emit_rmro(as, XO_MOVSD, left, base, ofs);
    t.irt = IRT_NUM;  /* Continue with a regular number type check. */
#if LJ_64 && !LJ_GC64
  } else if (irt_islightud(t)) {
    Reg dest = asm_load_lightud64(as, ir, (ir->op2 & IRSLOAD_TYPECHECK));
    if (ra_hasreg(dest)) {
      base = ra_alloc1(as, REF_BASE, RSET_GPR);
      emit_rmro(as, XO_MOV, dest|REX_64, base, ofs);
    }
    return;
#endif
  } else if (ra_used(ir)) {
    RegSet allow = irt_isnum(t) ? RSET_FPR : RSET_GPR;
    Reg dest = ra_dest(as, ir, allow);
    base = ra_alloc1(as, REF_BASE, RSET_GPR);
    lj_assertA(irt_isnum(t) || irt_isint(t) || irt_isaddr(t),
	       "bad SLOAD type %d", irt_type(t));
    if ((ir->op2 & IRSLOAD_CONVERT)) {
      t.irt = irt_isint(t) ? IRT_NUM : IRT_INT;  /* Check for original type. */
      emit_rmro(as, irt_isint(t) ? XO_CVTSI2SD : XO_CVTTSD2SI, dest, base, ofs);
    } else {
#if LJ_GC64
      if (irt_isaddr(t)) {
	/* LJ_GC64 type check + tag removal without BMI2 and with BMI2:
	**
	**  mov r64, [addr]    rorx r64, [addr], 47
	**  ror r64, 47
	**  cmp r16, itype     cmp r16, itype
	**  jne ->exit         jne ->exit
	**  shr r64, 16        shr r64, 16
	*/
	emit_shifti(as, XOg_SHR|REX_64, dest, 17);
	if ((ir->op2 & IRSLOAD_TYPECHECK)) {
	  asm_guardcc(as, CC_NE);
	  emit_i8(as, irt_toitype(t));
	  emit_rr(as, XO_ARITHi8, XOg_CMP, dest);
	  emit_i8(as, XI_O16);
	}
	if ((as->flags & JIT_F_BMI2)) {
	  emit_i8(as, 47);
	  emit_rmro(as, XV_RORX|VEX_64, dest, base, ofs);
	} else {
	  if ((ir->op2 & IRSLOAD_TYPECHECK))
	    emit_shifti(as, XOg_ROR|REX_64, dest, 47);
	  else
	    emit_shifti(as, XOg_SHL|REX_64, dest, 17);
	  emit_rmro(as, XO_MOV, dest|REX_64, base, ofs);
	}
	return;
      } else
#endif
      emit_rmro(as, irt_isnum(t) ? XO_MOVSD : XO_MOV, dest, base, ofs);
    }
  } else {
    if (!(ir->op2 & IRSLOAD_TYPECHECK))
      return;  /* No type check: avoid base alloc. */
    base = ra_alloc1(as, REF_BASE, RSET_GPR);
  }
  if ((ir->op2 & IRSLOAD_TYPECHECK)) {
    /* Need type check, even if the load result is unused. */
    asm_guardcc(as, irt_isnum(t) ? CC_AE : CC_NE);
    if ((LJ_64 && irt_type(t) >= IRT_NUM) || (ir->op2 & IRSLOAD_KEYINDEX)) {
      lj_assertA(irt_isinteger(t) || irt_isnum(t),
		 "bad SLOAD type %d", irt_type(t));
      emit_u32(as, (ir->op2 & IRSLOAD_KEYINDEX) ? LJ_KEYINDEX :
		   LJ_GC64 ? (LJ_TISNUM << 15) : LJ_TISNUM);
      emit_rmro(as, XO_ARITHi, XOg_CMP, base, ofs+4);
#if LJ_GC64
    } else if (irt_isnil(t)) {
      /* LJ_GC64 type check for nil:
      **
      **   cmp qword [addr], -1
      **   jne ->exit
      */
      emit_i8(as, -1);
      emit_rmro(as, XO_ARITHi8, XOg_CMP|REX_64, base, ofs);
    } else if (irt_ispri(t)) {
      emit_u32(as, (irt_toitype(t) << 15) | 0x7fff);
      emit_rmro(as, XO_ARITHi, XOg_CMP, base, ofs+4);
    } else {
      /* LJ_GC64 type check only:
      **
      **   mov r64, [addr]
      **   sar r64, 47
      **   cmp r32, itype
      **   jne ->exit
      */
      Reg tmp = ra_scratch(as, rset_exclude(RSET_GPR, base));
      emit_i8(as, irt_toitype(t));
      emit_rr(as, XO_ARITHi8, XOg_CMP, tmp);
      emit_shifti(as, XOg_SAR|REX_64, tmp, 47);
      emit_rmro(as, XO_MOV, tmp|REX_64, base, ofs);
#else
    } else {
      emit_i8(as, irt_toitype(t));
      emit_rmro(as, XO_ARITHi8, XOg_CMP, base, ofs+4);
#endif
    }
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
  lj_assertA(sz != CTSIZE_INVALID || (ir->o == IR_CNEW && ir->op2 != REF_NIL),
	     "bad CNEW/CNEWI operands");

  as->gcsteps++;
  asm_setupresult(as, ir, ci);  /* GCcdata * */

  /* Initialize immutable cdata object. */
  if (ir->o == IR_CNEWI) {
    RegSet allow = (RSET_GPR & ~RSET_SCRATCH);
#if LJ_64
    Reg r64 = sz == 8 ? REX_64 : 0;
    if (irref_isk(ir->op2)) {
      IRIns *irk = IR(ir->op2);
      uint64_t k = (irk->o == IR_KINT64 ||
		    (LJ_GC64 && (irk->o == IR_KPTR || irk->o == IR_KKPTR))) ?
		   ir_k64(irk)->u64 : (uint64_t)(uint32_t)irk->i;
      if (sz == 4 || checki32((int64_t)k)) {
	emit_i32(as, (int32_t)k);
	emit_rmro(as, XO_MOVmi, r64, RID_RET, sizeof(GCcdata));
      } else {
	emit_movtomro(as, RID_ECX + r64, RID_RET, sizeof(GCcdata));
	emit_loadu64(as, RID_ECX, k);
      }
    } else {
      Reg r = ra_alloc1(as, ir->op2, allow);
      emit_movtomro(as, r + r64, RID_RET, sizeof(GCcdata));
    }
#else
    int32_t ofs = sizeof(GCcdata);
    if (sz == 8) {
      ofs += 4; ir++;
      lj_assertA(ir->o == IR_HIOP, "missing CNEWI HIOP");
    }
    do {
      if (irref_isk(ir->op2)) {
	emit_movmroi(as, RID_RET, ofs, IR(ir->op2)->i);
      } else {
	Reg r = ra_alloc1(as, ir->op2, allow);
	emit_movtomro(as, r, RID_RET, ofs);
	rset_clear(allow, r);
      }
      if (ofs == sizeof(GCcdata)) break;
      ofs -= 4; ir--;
    } while (1);
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

  /* Combine initialization of marked, gct and ctypeid. */
  emit_movtomro(as, RID_ECX, RID_RET, offsetof(GCcdata, marked));
  emit_gri(as, XG_ARITHi(XOg_OR), RID_ECX,
	   (int32_t)((~LJ_TCDATA<<8)+(id<<16)));
  emit_gri(as, XG_ARITHi(XOg_AND), RID_ECX, LJ_GC_WHITES);
  emit_opgl(as, XO_MOVZXb, RID_ECX, gc.currentwhite);

  args[0] = ASMREF_L;     /* lua_State *L */
  args[1] = ASMREF_TMP1;  /* MSize size   */
  asm_gencall(as, ci, args);
  emit_loadi(as, ra_releasetmp(as, ASMREF_TMP1), (int32_t)(sz+sizeof(GCcdata)));
}
#endif

/* -- Write barriers ------------------------------------------------------ */

static void asm_tbar(ASMState *as, IRIns *ir)
{
  Reg tab = ra_alloc1(as, ir->op1, RSET_GPR);
  Reg tmp = ra_scratch(as, rset_exclude(RSET_GPR, tab));
  MCLabel l_end = emit_label(as);
  emit_movtomro(as, tmp|REX_GC64, tab, offsetof(GCtab, gclist));
  emit_setgl(as, tab, gc.grayagain);
  emit_getgl(as, tmp, gc.grayagain);
  emit_i8(as, ~LJ_GC_BLACK);
  emit_rmro(as, XO_ARITHib, XOg_AND, tab, offsetof(GCtab, marked));
  emit_sjcc(as, CC_Z, l_end);
  emit_i8(as, LJ_GC_BLACK);
  emit_rmro(as, XO_GROUP3b, XOg_TEST, tab, offsetof(GCtab, marked));
}

static void asm_obar(ASMState *as, IRIns *ir)
{
  const CCallInfo *ci = &lj_ir_callinfo[IRCALL_lj_gc_barrieruv];
  IRRef args[2];
  MCLabel l_end;
  Reg obj;
  /* No need for other object barriers (yet). */
  lj_assertA(IR(ir->op1)->o == IR_UREFC, "bad OBAR type");
  ra_evictset(as, RSET_SCRATCH);
  l_end = emit_label(as);
  args[0] = ASMREF_TMP1;  /* global_State *g */
  args[1] = ir->op1;      /* TValue *tv      */
  asm_gencall(as, ci, args);
  emit_loada(as, ra_releasetmp(as, ASMREF_TMP1), J2G(as->J));
  obj = IR(ir->op1)->r;
  emit_sjcc(as, CC_Z, l_end);
  emit_i8(as, LJ_GC_WHITES);
  if (irref_isk(ir->op2)) {
    GCobj *vp = ir_kgc(IR(ir->op2));
    emit_rma(as, XO_GROUP3b, XOg_TEST, &vp->gch.marked);
  } else {
    Reg val = ra_alloc1(as, ir->op2, rset_exclude(RSET_SCRATCH&RSET_GPR, obj));
    emit_rmro(as, XO_GROUP3b, XOg_TEST, val, (int32_t)offsetof(GChead, marked));
  }
  emit_sjcc(as, CC_Z, l_end);
  emit_i8(as, LJ_GC_BLACK);
  emit_rmro(as, XO_GROUP3b, XOg_TEST, obj,
	    (int32_t)offsetof(GCupval, marked)-(int32_t)offsetof(GCupval, tv));
}

/* -- FP/int arithmetic and logic operations ------------------------------ */

/* Load reference onto x87 stack. Force a spill to memory if needed. */
static void asm_x87load(ASMState *as, IRRef ref)
{
  IRIns *ir = IR(ref);
  if (ir->o == IR_KNUM) {
    cTValue *tv = ir_knum(ir);
    if (tvispzero(tv))  /* Use fldz only for +0. */
      emit_x87op(as, XI_FLDZ);
    else if (tvispone(tv))
      emit_x87op(as, XI_FLD1);
    else
      emit_rma(as, XO_FLDq, XOg_FLDq, tv);
  } else if (ir->o == IR_CONV && ir->op2 == IRCONV_NUM_INT && !ra_used(ir) &&
	     !irref_isk(ir->op1) && mayfuse(as, ir->op1)) {
    IRIns *iri = IR(ir->op1);
    emit_rmro(as, XO_FILDd, XOg_FILDd, RID_ESP, ra_spill(as, iri));
  } else {
    emit_mrm(as, XO_FLDq, XOg_FLDq, asm_fuseload(as, ref, RSET_EMPTY));
  }
}

static void asm_fpmath(ASMState *as, IRIns *ir)
{
  IRFPMathOp fpm = (IRFPMathOp)ir->op2;
  if (fpm == IRFPM_SQRT) {
    Reg dest = ra_dest(as, ir, RSET_FPR);
    Reg left = asm_fuseload(as, ir->op1, RSET_FPR);
    emit_mrm(as, XO_SQRTSD, dest, left);
  } else if (fpm <= IRFPM_TRUNC) {
    if (as->flags & JIT_F_SSE4_1) {  /* SSE4.1 has a rounding instruction. */
      Reg dest = ra_dest(as, ir, RSET_FPR);
      Reg left = asm_fuseload(as, ir->op1, RSET_FPR);
      /* ROUNDSD has a 4-byte opcode which doesn't fit in x86Op.
      ** Let's pretend it's a 3-byte opcode, and compensate afterwards.
      ** This is atrocious, but the alternatives are much worse.
      */
      /* Round down/up/trunc == 1001/1010/1011. */
      emit_i8(as, 0x09 + fpm);
      emit_mrm(as, XO_ROUNDSD, dest, left);
      if (LJ_64 && as->mcp[1] != (MCode)(XO_ROUNDSD >> 16)) {
	as->mcp[0] = as->mcp[1]; as->mcp[1] = 0x0f;  /* Swap 0F and REX. */
      }
      *--as->mcp = 0x66;  /* 1st byte of ROUNDSD opcode. */
    } else {  /* Call helper functions for SSE2 variant. */
      /* The modified regs must match with the *.dasc implementation. */
      RegSet drop = RSET_RANGE(RID_XMM0, RID_XMM3+1)|RID2RSET(RID_EAX);
      if (ra_hasreg(ir->r))
	rset_clear(drop, ir->r);  /* Dest reg handled below. */
      ra_evictset(as, drop);
      ra_destreg(as, ir, RID_XMM0);
      emit_call(as, fpm == IRFPM_FLOOR ? lj_vm_floor_sse :
		    fpm == IRFPM_CEIL ? lj_vm_ceil_sse : lj_vm_trunc_sse);
      ra_left(as, RID_XMM0, ir->op1);
    }
  } else {
    asm_callid(as, ir, IRCALL_lj_vm_floor + fpm);
  }
}

static void asm_ldexp(ASMState *as, IRIns *ir)
{
  int32_t ofs = sps_scale(ir->s);  /* Use spill slot or temp slots. */
  Reg dest = ir->r;
  if (ra_hasreg(dest)) {
    ra_free(as, dest);
    ra_modified(as, dest);
    emit_rmro(as, XO_MOVSD, dest, RID_ESP, ofs);
  }
  emit_rmro(as, XO_FSTPq, XOg_FSTPq, RID_ESP, ofs);
  emit_x87op(as, XI_FPOP1);
  emit_x87op(as, XI_FSCALE);
  asm_x87load(as, ir->op1);
  asm_x87load(as, ir->op2);
}

static int asm_swapops(ASMState *as, IRIns *ir)
{
  IRIns *irl = IR(ir->op1);
  IRIns *irr = IR(ir->op2);
  lj_assertA(ra_noreg(irr->r), "bad usage");
  if (!irm_iscomm(lj_ir_mode[ir->o]))
    return 0;  /* Can't swap non-commutative operations. */
  if (irref_isk(ir->op2))
    return 0;  /* Don't swap constants to the left. */
  if (ra_hasreg(irl->r))
    return 1;  /* Swap if left already has a register. */
  if (ra_samehint(ir->r, irr->r))
    return 1;  /* Swap if dest and right have matching hints. */
  if (as->curins > as->loopref) {  /* In variant part? */
    if (ir->op2 < as->loopref && !irt_isphi(irr->t))
      return 0;  /* Keep invariants on the right. */
    if (ir->op1 < as->loopref && !irt_isphi(irl->t))
      return 1;  /* Swap invariants to the right. */
  }
  if (opisfusableload(irl->o))
    return 1;  /* Swap fusable loads to the right. */
  return 0;  /* Otherwise don't swap. */
}

static void asm_fparith(ASMState *as, IRIns *ir, x86Op xo)
{
  IRRef lref = ir->op1;
  IRRef rref = ir->op2;
  RegSet allow = RSET_FPR;
  Reg dest;
  Reg right = IR(rref)->r;
  if (ra_hasreg(right)) {
    rset_clear(allow, right);
    ra_noweak(as, right);
  }
  dest = ra_dest(as, ir, allow);
  if (lref == rref) {
    right = dest;
  } else if (ra_noreg(right)) {
    if (asm_swapops(as, ir)) {
      IRRef tmp = lref; lref = rref; rref = tmp;
    }
    right = asm_fuseload(as, rref, rset_clear(allow, dest));
  }
  emit_mrm(as, xo, dest, right);
  ra_left(as, dest, lref);
}

static void asm_intarith(ASMState *as, IRIns *ir, x86Arith xa)
{
  IRRef lref = ir->op1;
  IRRef rref = ir->op2;
  RegSet allow = RSET_GPR;
  Reg dest, right;
  int32_t k = 0;
  if (as->flagmcp == as->mcp) {  /* Drop test r,r instruction. */
    MCode *p = as->mcp + ((LJ_64 && *as->mcp < XI_TESTb) ? 3 : 2);
    MCode *q = p[0] == 0x0f ? p+1 : p;
    if ((*q & 15) < 14) {
      if ((*q & 15) >= 12) *q -= 4;  /* L <->S, NL <-> NS */
      as->flagmcp = NULL;
      as->mcp = p;
    }  /* else: cannot transform LE/NLE to cc without use of OF. */
  }
  right = IR(rref)->r;
  if (ra_hasreg(right)) {
    rset_clear(allow, right);
    ra_noweak(as, right);
  }
  dest = ra_dest(as, ir, allow);
  if (lref == rref) {
    right = dest;
  } else if (ra_noreg(right) && !asm_isk32(as, rref, &k)) {
    if (asm_swapops(as, ir)) {
      IRRef tmp = lref; lref = rref; rref = tmp;
    }
    right = asm_fuseloadm(as, rref, rset_clear(allow, dest), irt_is64(ir->t));
  }
  if (irt_isguard(ir->t))  /* For IR_ADDOV etc. */
    asm_guardcc(as, CC_O);
  if (xa != XOg_X_IMUL) {
    if (ra_hasreg(right))
      emit_mrm(as, XO_ARITH(xa), REX_64IR(ir, dest), right);
    else
      emit_gri(as, XG_ARITHi(xa), REX_64IR(ir, dest), k);
  } else if (ra_hasreg(right)) {  /* IMUL r, mrm. */
    emit_mrm(as, XO_IMUL, REX_64IR(ir, dest), right);
  } else {  /* IMUL r, r, k. */
    /* NYI: use lea/shl/add/sub (FOLD only does 2^k) depending on CPU. */
    Reg left = asm_fuseloadm(as, lref, RSET_GPR, irt_is64(ir->t));
    x86Op xo;
    if (checki8(k)) { emit_i8(as, k); xo = XO_IMULi8;
    } else { emit_i32(as, k); xo = XO_IMULi; }
    emit_mrm(as, xo, REX_64IR(ir, dest), left);
    return;
  }
  ra_left(as, dest, lref);
}

/* LEA is really a 4-operand ADD with an independent destination register,
** up to two source registers and an immediate. One register can be scaled
** by 1, 2, 4 or 8. This can be used to avoid moves or to fuse several
** instructions.
**
** Currently only a few common cases are supported:
** - 3-operand ADD:    y = a+b; y = a+k   with a and b already allocated
** - Left ADD fusion:  y = (a+b)+k; y = (a+k)+b
** - Right ADD fusion: y = a+(b+k)
** The ommited variants have already been reduced by FOLD.
**
** There are more fusion opportunities, like gathering shifts or joining
** common references. But these are probably not worth the trouble, since
** array indexing is not decomposed and already makes use of all fields
** of the ModRM operand.
*/
static int asm_lea(ASMState *as, IRIns *ir)
{
  IRIns *irl = IR(ir->op1);
  IRIns *irr = IR(ir->op2);
  RegSet allow = RSET_GPR;
  Reg dest;
  as->mrm.base = as->mrm.idx = RID_NONE;
  as->mrm.scale = XM_SCALE1;
  as->mrm.ofs = 0;
  if (ra_hasreg(irl->r)) {
    rset_clear(allow, irl->r);
    ra_noweak(as, irl->r);
    as->mrm.base = irl->r;
    if (irref_isk(ir->op2) || ra_hasreg(irr->r)) {
      /* The PHI renaming logic does a better job in some cases. */
      if (ra_hasreg(ir->r) &&
	  ((irt_isphi(irl->t) && as->phireg[ir->r] == ir->op1) ||
	   (irt_isphi(irr->t) && as->phireg[ir->r] == ir->op2)))
	return 0;
      if (irref_isk(ir->op2)) {
	as->mrm.ofs = irr->i;
      } else {
	rset_clear(allow, irr->r);
	ra_noweak(as, irr->r);
	as->mrm.idx = irr->r;
      }
    } else if (irr->o == IR_ADD && mayfuse(as, ir->op2) &&
	       irref_isk(irr->op2)) {
      Reg idx = ra_alloc1(as, irr->op1, allow);
      rset_clear(allow, idx);
      as->mrm.idx = (uint8_t)idx;
      as->mrm.ofs = IR(irr->op2)->i;
    } else {
      return 0;
    }
  } else if (ir->op1 != ir->op2 && irl->o == IR_ADD && mayfuse(as, ir->op1) &&
	     (irref_isk(ir->op2) || irref_isk(irl->op2))) {
    Reg idx, base = ra_alloc1(as, irl->op1, allow);
    rset_clear(allow, base);
    as->mrm.base = (uint8_t)base;
    if (irref_isk(ir->op2)) {
      as->mrm.ofs = irr->i;
      idx = ra_alloc1(as, irl->op2, allow);
    } else {
      as->mrm.ofs = IR(irl->op2)->i;
      idx = ra_alloc1(as, ir->op2, allow);
    }
    rset_clear(allow, idx);
    as->mrm.idx = (uint8_t)idx;
  } else {
    return 0;
  }
  dest = ra_dest(as, ir, allow);
  emit_mrm(as, XO_LEA, dest, RID_MRM);
  return 1;  /* Success. */
}

static void asm_add(ASMState *as, IRIns *ir)
{
  if (irt_isnum(ir->t))
    asm_fparith(as, ir, XO_ADDSD);
  else if (as->flagmcp == as->mcp || irt_is64(ir->t) || !asm_lea(as, ir))
    asm_intarith(as, ir, XOg_ADD);
}

static void asm_sub(ASMState *as, IRIns *ir)
{
  if (irt_isnum(ir->t))
    asm_fparith(as, ir, XO_SUBSD);
  else  /* Note: no need for LEA trick here. i-k is encoded as i+(-k). */
    asm_intarith(as, ir, XOg_SUB);
}

static void asm_mul(ASMState *as, IRIns *ir)
{
  if (irt_isnum(ir->t))
    asm_fparith(as, ir, XO_MULSD);
  else
    asm_intarith(as, ir, XOg_X_IMUL);
}

#define asm_fpdiv(as, ir)	asm_fparith(as, ir, XO_DIVSD)

static void asm_neg_not(ASMState *as, IRIns *ir, x86Group3 xg)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  emit_rr(as, XO_GROUP3, REX_64IR(ir, xg), dest);
  ra_left(as, dest, ir->op1);
}

static void asm_neg(ASMState *as, IRIns *ir)
{
  if (irt_isnum(ir->t))
    asm_fparith(as, ir, XO_XORPS);
  else
    asm_neg_not(as, ir, XOg_NEG);
}

#define asm_abs(as, ir)		asm_fparith(as, ir, XO_ANDPS)

static void asm_intmin_max(ASMState *as, IRIns *ir, int cc)
{
  Reg right, dest = ra_dest(as, ir, RSET_GPR);
  IRRef lref = ir->op1, rref = ir->op2;
  if (irref_isk(rref)) { lref = rref; rref = ir->op1; }
  right = ra_alloc1(as, rref, rset_exclude(RSET_GPR, dest));
  emit_rr(as, XO_CMOV + (cc<<24), REX_64IR(ir, dest), right);
  emit_rr(as, XO_CMP, REX_64IR(ir, dest), right);
  ra_left(as, dest, lref);
}

static void asm_min(ASMState *as, IRIns *ir)
{
  if (irt_isnum(ir->t))
    asm_fparith(as, ir, XO_MINSD);
  else
    asm_intmin_max(as, ir, CC_G);
}

static void asm_max(ASMState *as, IRIns *ir)
{
  if (irt_isnum(ir->t))
    asm_fparith(as, ir, XO_MAXSD);
  else
    asm_intmin_max(as, ir, CC_L);
}

/* Note: don't use LEA for overflow-checking arithmetic! */
#define asm_addov(as, ir)	asm_intarith(as, ir, XOg_ADD)
#define asm_subov(as, ir)	asm_intarith(as, ir, XOg_SUB)
#define asm_mulov(as, ir)	asm_intarith(as, ir, XOg_X_IMUL)

#define asm_bnot(as, ir)	asm_neg_not(as, ir, XOg_NOT)

static void asm_bswap(ASMState *as, IRIns *ir)
{
  Reg dest = ra_dest(as, ir, RSET_GPR);
  as->mcp = emit_op(XO_BSWAP + ((dest&7) << 24),
		    REX_64IR(ir, 0), dest, 0, as->mcp, 1);
  ra_left(as, dest, ir->op1);
}

#define asm_band(as, ir)	asm_intarith(as, ir, XOg_AND)
#define asm_bor(as, ir)		asm_intarith(as, ir, XOg_OR)
#define asm_bxor(as, ir)	asm_intarith(as, ir, XOg_XOR)

static void asm_bitshift(ASMState *as, IRIns *ir, x86Shift xs, x86Op xv)
{
  IRRef rref = ir->op2;
  IRIns *irr = IR(rref);
  Reg dest;
  if (irref_isk(rref)) {  /* Constant shifts. */
    int shift;
    dest = ra_dest(as, ir, RSET_GPR);
    shift = irr->i & (irt_is64(ir->t) ? 63 : 31);
    if (!xv && shift && (as->flags & JIT_F_BMI2)) {
      Reg left = asm_fuseloadm(as, ir->op1, RSET_GPR, irt_is64(ir->t));
      if (left != dest) {  /* BMI2 rotate right by constant. */
	emit_i8(as, xs == XOg_ROL ? -shift : shift);
	emit_mrm(as, VEX_64IR(ir, XV_RORX), dest, left);
	return;
      }
    }
    switch (shift) {
    case 0: break;
    case 1: emit_rr(as, XO_SHIFT1, REX_64IR(ir, xs), dest); break;
    default: emit_shifti(as, REX_64IR(ir, xs), dest, shift); break;
    }
  } else if ((as->flags & JIT_F_BMI2) && xv) {	/* BMI2 variable shifts. */
    Reg left, right;
    dest = ra_dest(as, ir, RSET_GPR);
    right = ra_alloc1(as, rref, RSET_GPR);
    left = asm_fuseloadm(as, ir->op1, rset_exclude(RSET_GPR, right),
			 irt_is64(ir->t));
    emit_mrm(as, VEX_64IR(ir, xv) ^ (right << 19), dest, left);
    return;
  } else {  /* Variable shifts implicitly use register cl (i.e. ecx). */
    Reg right;
    dest = ra_dest(as, ir, rset_exclude(RSET_GPR, RID_ECX));
    if (dest == RID_ECX) {
      dest = ra_scratch(as, rset_exclude(RSET_GPR, RID_ECX));
      emit_rr(as, XO_MOV, REX_64IR(ir, RID_ECX), dest);
    }
    right = irr->r;
    if (ra_noreg(right))
      right = ra_allocref(as, rref, RID2RSET(RID_ECX));
    else if (right != RID_ECX)
      ra_scratch(as, RID2RSET(RID_ECX));
    emit_rr(as, XO_SHIFTcl, REX_64IR(ir, xs), dest);
    ra_noweak(as, right);
    if (right != RID_ECX)
      emit_rr(as, XO_MOV, RID_ECX, right);
  }
  ra_left(as, dest, ir->op1);
  /*
  ** Note: avoid using the flags resulting from a shift or rotate!
  ** All of them cause a partial flag stall, except for r,1 shifts
  ** (but not rotates). And a shift count of 0 leaves the flags unmodified.
  */
}

#define asm_bshl(as, ir)	asm_bitshift(as, ir, XOg_SHL, XV_SHLX)
#define asm_bshr(as, ir)	asm_bitshift(as, ir, XOg_SHR, XV_SHRX)
#define asm_bsar(as, ir)	asm_bitshift(as, ir, XOg_SAR, XV_SARX)
#define asm_brol(as, ir)	asm_bitshift(as, ir, XOg_ROL, 0)
#define asm_bror(as, ir)	asm_bitshift(as, ir, XOg_ROR, 0)

/* -- Comparisons --------------------------------------------------------- */

/* Virtual flags for unordered FP comparisons. */
#define VCC_U	0x1000		/* Unordered. */
#define VCC_P	0x2000		/* Needs extra CC_P branch. */
#define VCC_S	0x4000		/* Swap avoids CC_P branch. */
#define VCC_PS	(VCC_P|VCC_S)

/* Map of comparisons to flags. ORDER IR. */
#define COMPFLAGS(ci, cin, cu, cf)	((ci)+((cu)<<4)+((cin)<<8)+(cf))
static const uint16_t asm_compmap[IR_ABC+1] = {
  /*                 signed non-eq unsigned flags */
  /* LT  */ COMPFLAGS(CC_GE, CC_G,  CC_AE, VCC_PS),
  /* GE  */ COMPFLAGS(CC_L,  CC_L,  CC_B,  0),
  /* LE  */ COMPFLAGS(CC_G,  CC_G,  CC_A,  VCC_PS),
  /* GT  */ COMPFLAGS(CC_LE, CC_L,  CC_BE, 0),
  /* ULT */ COMPFLAGS(CC_AE, CC_A,  CC_AE, VCC_U),
  /* UGE */ COMPFLAGS(CC_B,  CC_B,  CC_B,  VCC_U|VCC_PS),
  /* ULE */ COMPFLAGS(CC_A,  CC_A,  CC_A,  VCC_U),
  /* UGT */ COMPFLAGS(CC_BE, CC_B,  CC_BE, VCC_U|VCC_PS),
  /* EQ  */ COMPFLAGS(CC_NE, CC_NE, CC_NE, VCC_P),
  /* NE  */ COMPFLAGS(CC_E,  CC_E,  CC_E,  VCC_U|VCC_P),
  /* ABC */ COMPFLAGS(CC_BE, CC_B,  CC_BE, VCC_U|VCC_PS)  /* Same as UGT. */
};

/* FP and integer comparisons. */
static void asm_comp(ASMState *as, IRIns *ir)
{
  uint32_t cc = asm_compmap[ir->o];
  if (irt_isnum(ir->t)) {
    IRRef lref = ir->op1;
    IRRef rref = ir->op2;
    Reg left, right;
    MCLabel l_around;
    /*
    ** An extra CC_P branch is required to preserve ordered/unordered
    ** semantics for FP comparisons. This can be avoided by swapping
    ** the operands and inverting the condition (except for EQ and UNE).
    ** So always try to swap if possible.
    **
    ** Another option would be to swap operands to achieve better memory
    ** operand fusion. But it's unlikely that this outweighs the cost
    ** of the extra branches.
    */
    if (cc & VCC_S) {  /* Swap? */
      IRRef tmp = lref; lref = rref; rref = tmp;
      cc ^= (VCC_PS|(5<<4));  /* A <-> B, AE <-> BE, PS <-> none */
    }
    left = ra_alloc1(as, lref, RSET_FPR);
    l_around = emit_label(as);
    asm_guardcc(as, cc >> 4);
    if (cc & VCC_P) {  /* Extra CC_P branch required? */
      if (!(cc & VCC_U)) {
	asm_guardcc(as, CC_P);  /* Branch to exit for ordered comparisons. */
      } else if (l_around != as->invmcp) {
	emit_sjcc(as, CC_P, l_around);  /* Branch around for unordered. */
      } else {
	/* Patched to mcloop by asm_loop_fixup. */
	as->loopinv = 2;
	if (as->realign)
	  emit_sjcc(as, CC_P, as->mcp);
	else
	  emit_jcc(as, CC_P, as->mcp);
      }
    }
    right = asm_fuseload(as, rref, rset_exclude(RSET_FPR, left));
    emit_mrm(as, XO_UCOMISD, left, right);
  } else {
    IRRef lref = ir->op1, rref = ir->op2;
    IROp leftop = (IROp)(IR(lref)->o);
    Reg r64 = REX_64IR(ir, 0);
    int32_t imm = 0;
    lj_assertA(irt_is64(ir->t) || irt_isint(ir->t) ||
	       irt_isu32(ir->t) || irt_isaddr(ir->t) || irt_isu8(ir->t),
	       "bad comparison data type %d", irt_type(ir->t));
    /* Swap constants (only for ABC) and fusable loads to the right. */
    if (irref_isk(lref) || (!irref_isk(rref) && opisfusableload(leftop))) {
      if ((cc & 0xc) == 0xc) cc ^= 0x53;  /* L <-> G, LE <-> GE */
      else if ((cc & 0xa) == 0x2) cc ^= 0x55;  /* A <-> B, AE <-> BE */
      lref = ir->op2; rref = ir->op1;
    }
    if (asm_isk32(as, rref, &imm)) {
      IRIns *irl = IR(lref);
      /* Check wether we can use test ins. Not for unsigned, since CF=0. */
      int usetest = (imm == 0 && (cc & 0xa) != 0x2);
      if (usetest && irl->o == IR_BAND && irl+1 == ir && !ra_used(irl)) {
	/* Combine comp(BAND(ref, r/imm), 0) into test mrm, r/imm. */
	Reg right, left = RID_NONE;
	RegSet allow = RSET_GPR;
	if (!asm_isk32(as, irl->op2, &imm)) {
	  left = ra_alloc1(as, irl->op2, allow);
	  rset_clear(allow, left);
	} else {  /* Try to Fuse IRT_I8/IRT_U8 loads, too. See below. */
	  IRIns *irll = IR(irl->op1);
	  if (opisfusableload((IROp)irll->o) &&
	      (irt_isi8(irll->t) || irt_isu8(irll->t))) {
	    IRType1 origt = irll->t;  /* Temporarily flip types. */
	    irll->t.irt = (irll->t.irt & ~IRT_TYPE) | IRT_INT;
	    as->curins--;  /* Skip to BAND to avoid failing in noconflict(). */
	    right = asm_fuseload(as, irl->op1, RSET_GPR);
	    as->curins++;
	    irll->t = origt;
	    if (right != RID_MRM) goto test_nofuse;
	    /* Fusion succeeded, emit test byte mrm, imm8. */
	    asm_guardcc(as, cc);
	    emit_i8(as, (imm & 0xff));
	    emit_mrm(as, XO_GROUP3b, XOg_TEST, RID_MRM);
	    return;
	  }
	}
	as->curins--;  /* Skip to BAND to avoid failing in noconflict(). */
	right = asm_fuseloadm(as, irl->op1, allow, r64);
	as->curins++;  /* Undo the above. */
      test_nofuse:
	asm_guardcc(as, cc);
	if (ra_noreg(left)) {
	  emit_i32(as, imm);
	  emit_mrm(as, XO_GROUP3, r64 + XOg_TEST, right);
	} else {
	  emit_mrm(as, XO_TEST, r64 + left, right);
	}
      } else {
	Reg left;
	if (opisfusableload((IROp)irl->o) &&
	    ((irt_isu8(irl->t) && checku8(imm)) ||
	     ((irt_isi8(irl->t) || irt_isi16(irl->t)) && checki8(imm)) ||
	     (irt_isu16(irl->t) && checku16(imm) && checki8((int16_t)imm)))) {
	  /* Only the IRT_INT case is fused by asm_fuseload.
	  ** The IRT_I8/IRT_U8 loads and some IRT_I16/IRT_U16 loads
	  ** are handled here.
	  ** Note that cmp word [mem], imm16 should not be generated,
	  ** since it has a length-changing prefix. Compares of a word
	  ** against a sign-extended imm8 are ok, however.
	  */
	  IRType1 origt = irl->t;  /* Temporarily flip types. */
	  irl->t.irt = (irl->t.irt & ~IRT_TYPE) | IRT_INT;
	  left = asm_fuseload(as, lref, RSET_GPR);
	  irl->t = origt;
	  if (left == RID_MRM) {  /* Fusion succeeded? */
	    if (irt_isu8(irl->t) || irt_isu16(irl->t))
	      cc >>= 4;  /* Need unsigned compare. */
	    asm_guardcc(as, cc);
	    emit_i8(as, imm);
	    emit_mrm(as, (irt_isi8(origt) || irt_isu8(origt)) ?
			 XO_ARITHib : XO_ARITHiw8, r64 + XOg_CMP, RID_MRM);
	    return;
	  }  /* Otherwise handle register case as usual. */
	} else {
	  left = asm_fuseloadm(as, lref,
			       irt_isu8(ir->t) ? RSET_GPR8 : RSET_GPR, r64);
	}
	asm_guardcc(as, cc);
	if (usetest && left != RID_MRM) {
	  /* Use test r,r instead of cmp r,0. */
	  x86Op xo = XO_TEST;
	  if (irt_isu8(ir->t)) {
	    lj_assertA(ir->o == IR_EQ || ir->o == IR_NE, "bad usage");
	    xo = XO_TESTb;
	    if (!rset_test(RSET_RANGE(RID_EAX, RID_EBX+1), left)) {
	      if (LJ_64) {
		left |= FORCE_REX;
	      } else {
		emit_i32(as, 0xff);
		emit_mrm(as, XO_GROUP3, XOg_TEST, left);
		return;
	      }
	    }
	  }
	  emit_rr(as, xo, r64 + left, left);
	  if (irl+1 == ir)  /* Referencing previous ins? */
	    as->flagmcp = as->mcp;  /* Set flag to drop test r,r if possible. */
	} else {
	  emit_gmrmi(as, XG_ARITHi(XOg_CMP), r64 + left, imm);
	}
      }
    } else {
      Reg left = ra_alloc1(as, lref, RSET_GPR);
      Reg right = asm_fuseloadm(as, rref, rset_exclude(RSET_GPR, left), r64);
      asm_guardcc(as, cc);
      emit_mrm(as, XO_CMP, r64 + left, right);
    }
  }
}

#define asm_equal(as, ir)	asm_comp(as, ir)

#if LJ_32 && LJ_HASFFI
/* 64 bit integer comparisons in 32 bit mode. */
static void asm_comp_int64(ASMState *as, IRIns *ir)
{
  uint32_t cc = asm_compmap[(ir-1)->o];
  RegSet allow = RSET_GPR;
  Reg lefthi = RID_NONE, leftlo = RID_NONE;
  Reg righthi = RID_NONE, rightlo = RID_NONE;
  MCLabel l_around;
  x86ModRM mrm;

  as->curins--;  /* Skip loword ins. Avoids failing in noconflict(), too. */

  /* Allocate/fuse hiword operands. */
  if (irref_isk(ir->op2)) {
    lefthi = asm_fuseload(as, ir->op1, allow);
  } else {
    lefthi = ra_alloc1(as, ir->op1, allow);
    rset_clear(allow, lefthi);
    righthi = asm_fuseload(as, ir->op2, allow);
    if (righthi == RID_MRM) {
      if (as->mrm.base != RID_NONE) rset_clear(allow, as->mrm.base);
      if (as->mrm.idx != RID_NONE) rset_clear(allow, as->mrm.idx);
    } else {
      rset_clear(allow, righthi);
    }
  }
  mrm = as->mrm;  /* Save state for hiword instruction. */

  /* Allocate/fuse loword operands. */
  if (irref_isk((ir-1)->op2)) {
    leftlo = asm_fuseload(as, (ir-1)->op1, allow);
  } else {
    leftlo = ra_alloc1(as, (ir-1)->op1, allow);
    rset_clear(allow, leftlo);
    rightlo = asm_fuseload(as, (ir-1)->op2, allow);
  }

  /* All register allocations must be performed _before_ this point. */
  l_around = emit_label(as);
  as->invmcp = as->flagmcp = NULL;  /* Cannot use these optimizations. */

  /* Loword comparison and branch. */
  asm_guardcc(as, cc >> 4);  /* Always use unsigned compare for loword. */
  if (ra_noreg(rightlo)) {
    int32_t imm = IR((ir-1)->op2)->i;
    if (imm == 0 && ((cc >> 4) & 0xa) != 0x2 && leftlo != RID_MRM)
      emit_rr(as, XO_TEST, leftlo, leftlo);
    else
      emit_gmrmi(as, XG_ARITHi(XOg_CMP), leftlo, imm);
  } else {
    emit_mrm(as, XO_CMP, leftlo, rightlo);
  }

  /* Hiword comparison and branches. */
  if ((cc & 15) != CC_NE)
    emit_sjcc(as, CC_NE, l_around);  /* Hiword unequal: skip loword compare. */
  if ((cc & 15) != CC_E)
    asm_guardcc(as, cc >> 8);  /* Hiword compare without equality check. */
  as->mrm = mrm;  /* Restore state. */
  if (ra_noreg(righthi)) {
    int32_t imm = IR(ir->op2)->i;
    if (imm == 0 && (cc & 0xa) != 0x2 && lefthi != RID_MRM)
      emit_rr(as, XO_TEST, lefthi, lefthi);
    else
      emit_gmrmi(as, XG_ARITHi(XOg_CMP), lefthi, imm);
  } else {
    emit_mrm(as, XO_CMP, lefthi, righthi);
  }
}
#endif

/* -- Split register ops -------------------------------------------------- */

/* Hiword op of a split 32/32 or 64/64 bit op. Previous op is the loword op. */
static void asm_hiop(ASMState *as, IRIns *ir)
{
  /* HIOP is marked as a store because it needs its own DCE logic. */
  int uselo = ra_used(ir-1), usehi = ra_used(ir);  /* Loword/hiword used? */
  if (LJ_UNLIKELY(!(as->flags & JIT_F_OPT_DCE))) uselo = usehi = 1;
#if LJ_32 && LJ_HASFFI
  if ((ir-1)->o == IR_CONV) {  /* Conversions to/from 64 bit. */
    as->curins--;  /* Always skip the CONV. */
    if (usehi || uselo)
      asm_conv64(as, ir);
    return;
  } else if ((ir-1)->o <= IR_NE) {  /* 64 bit integer comparisons. ORDER IR. */
    asm_comp_int64(as, ir);
    return;
  } else if ((ir-1)->o == IR_XSTORE) {
    if ((ir-1)->r != RID_SINK)
      asm_fxstore(as, ir);
    return;
  }
#endif
  if (!usehi) return;  /* Skip unused hiword op for all remaining ops. */
  switch ((ir-1)->o) {
#if LJ_32 && LJ_HASFFI
  case IR_ADD:
    as->flagmcp = NULL;
    as->curins--;
    asm_intarith(as, ir, XOg_ADC);
    asm_intarith(as, ir-1, XOg_ADD);
    break;
  case IR_SUB:
    as->flagmcp = NULL;
    as->curins--;
    asm_intarith(as, ir, XOg_SBB);
    asm_intarith(as, ir-1, XOg_SUB);
    break;
  case IR_NEG: {
    Reg dest = ra_dest(as, ir, RSET_GPR);
    emit_rr(as, XO_GROUP3, XOg_NEG, dest);
    emit_i8(as, 0);
    emit_rr(as, XO_ARITHi8, XOg_ADC, dest);
    ra_left(as, dest, ir->op1);
    as->curins--;
    asm_neg_not(as, ir-1, XOg_NEG);
    break;
    }
  case IR_CNEWI:
    /* Nothing to do here. Handled by CNEWI itself. */
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
  emit_i8(as, HOOK_PROFILE);
  emit_rma(as, XO_GROUP3b, XOg_TEST, &J2G(as->J)->hookmask);
}

/* -- Stack handling ------------------------------------------------------ */

/* Check Lua stack size for overflow. Use exit handler as fallback. */
static void asm_stack_check(ASMState *as, BCReg topslot,
			    IRIns *irp, RegSet allow, ExitNo exitno)
{
  /* Try to get an unused temp. register, otherwise spill/restore eax. */
  Reg pbase = irp ? irp->r : RID_BASE;
  Reg r = allow ? rset_pickbot(allow) : RID_EAX;
  emit_jcc(as, CC_B, exitstub_addr(as->J, exitno));
  if (allow == RSET_EMPTY)  /* Restore temp. register. */
    emit_rmro(as, XO_MOV, r|REX_64, RID_ESP, 0);
  else
    ra_modified(as, r);
  emit_gri(as, XG_ARITHi(XOg_CMP), r|REX_GC64, (int32_t)(8*topslot));
  if (ra_hasreg(pbase) && pbase != r)
    emit_rr(as, XO_ARITH(XOg_SUB), r|REX_GC64, pbase);
  else
#if LJ_GC64
    emit_rmro(as, XO_ARITH(XOg_SUB), r|REX_64, RID_DISPATCH,
	      (int32_t)dispofs(as, &J2G(as->J)->jit_base));
#else
    emit_rmro(as, XO_ARITH(XOg_SUB), r, RID_NONE,
	      ptr2addr(&J2G(as->J)->jit_base));
#endif
  emit_rmro(as, XO_MOV, r|REX_GC64, r, offsetof(lua_State, maxstack));
  emit_getgl(as, r, cur_L);
  if (allow == RSET_EMPTY)  /* Spill temp. register. */
    emit_rmro(as, XO_MOVto, r|REX_64, RID_ESP, 0);
}

/* Restore Lua stack from on-trace state. */
static void asm_stack_restore(ASMState *as, SnapShot *snap)
{
  SnapEntry *map = &as->T->snapmap[snap->mapofs];
#if !LJ_FR2 || defined(LUA_USE_ASSERT)
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
    if ((sn & SNAP_KEYINDEX)) {
      emit_movmroi(as, RID_BASE, ofs+4, LJ_KEYINDEX);
      if (irref_isk(ref)) {
	emit_movmroi(as, RID_BASE, ofs, ir->i);
      } else {
	Reg src = ra_alloc1(as, ref, rset_exclude(RSET_GPR, RID_BASE));
	emit_movtomro(as, src, RID_BASE, ofs);
      }
    } else if (irt_isnum(ir->t)) {
      Reg src = ra_alloc1(as, ref, RSET_FPR);
      emit_rmro(as, XO_MOVSDto, src, RID_BASE, ofs);
    } else {
      lj_assertA(irt_ispri(ir->t) || irt_isaddr(ir->t) ||
		 (LJ_DUALNUM && irt_isinteger(ir->t)),
		 "restore of IR type %d", irt_type(ir->t));
      if (!irref_isk(ref)) {
	Reg src = ra_alloc1(as, ref, rset_exclude(RSET_GPR, RID_BASE));
#if LJ_GC64
	if (irt_is64(ir->t)) {
	  /* TODO: 64 bit store + 32 bit load-modify-store is suboptimal. */
	  emit_u32(as, irt_toitype(ir->t) << 15);
	  emit_rmro(as, XO_ARITHi, XOg_OR, RID_BASE, ofs+4);
	} else if (LJ_DUALNUM && irt_isinteger(ir->t)) {
	  emit_movmroi(as, RID_BASE, ofs+4, LJ_TISNUM << 15);
	} else {
	  emit_movmroi(as, RID_BASE, ofs+4, (irt_toitype(ir->t)<<15)|0x7fff);
	}
#endif
	emit_movtomro(as, REX_64IR(ir, src), RID_BASE, ofs);
#if LJ_GC64
      } else {
	TValue k;
	lj_ir_kvalue(as->J->L, &k, ir);
	if (tvisnil(&k)) {
	  emit_i32(as, -1);
	  emit_rmro(as, XO_MOVmi, REX_64, RID_BASE, ofs);
	} else {
	  emit_movmroi(as, RID_BASE, ofs+4, k.u32.hi);
	  emit_movmroi(as, RID_BASE, ofs, k.u32.lo);
	}
#else
      } else if (!irt_ispri(ir->t)) {
	emit_movmroi(as, RID_BASE, ofs, ir->i);
#endif
      }
      if ((sn & (SNAP_CONT|SNAP_FRAME))) {
#if !LJ_FR2
	if (s != 0)  /* Do not overwrite link to previous frame. */
	  emit_movmroi(as, RID_BASE, ofs+4, (int32_t)(*flinks--));
#endif
#if !LJ_GC64
      } else {
	if (!(LJ_64 && irt_islightud(ir->t)))
	  emit_movmroi(as, RID_BASE, ofs+4, irt_toitype(ir->t));
#endif
      }
    }
    checkmclim(as);
  }
  lj_assertA(map + nent == flinks, "inconsistent frames in snapshot");
}

/* -- GC handling --------------------------------------------------------- */

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
  emit_rr(as, XO_TEST, RID_RET, RID_RET);
  args[0] = ASMREF_TMP1;  /* global_State *g */
  args[1] = ASMREF_TMP2;  /* MSize steps     */
  asm_gencall(as, ci, args);
  tmp = ra_releasetmp(as, ASMREF_TMP1);
#if LJ_GC64
  emit_rmro(as, XO_LEA, tmp|REX_64, RID_DISPATCH, GG_DISP2G);
#else
  emit_loada(as, tmp, J2G(as->J));
#endif
  emit_loadi(as, ra_releasetmp(as, ASMREF_TMP2), as->gcsteps);
  /* Jump around GC step if GC total < GC threshold. */
  emit_sjcc(as, CC_B, l_end);
  emit_opgl(as, XO_ARITH(XOg_CMP), tmp|REX_GC64, gc.threshold);
  emit_getgl(as, tmp, gc.total);
  as->gcsteps = 0;
  checkmclim(as);
}

/* -- Loop handling ------------------------------------------------------- */

/* Fixup the loop branch. */
static void asm_loop_fixup(ASMState *as)
{
  MCode *p = as->mctop;
  MCode *target = as->mcp;
  if (as->realign) {  /* Realigned loops use short jumps. */
    as->realign = NULL;  /* Stop another retry. */
    lj_assertA(((intptr_t)target & 15) == 0, "loop realign failed");
    if (as->loopinv) {  /* Inverted loop branch? */
      p -= 5;
      p[0] = XI_JMP;
      lj_assertA(target - p >= -128, "loop realign failed");
      p[-1] = (MCode)(target - p);  /* Patch sjcc. */
      if (as->loopinv == 2)
	p[-3] = (MCode)(target - p + 2);  /* Patch opt. short jp. */
    } else {
      lj_assertA(target - p >= -128, "loop realign failed");
      p[-1] = (MCode)(int8_t)(target - p);  /* Patch short jmp. */
      p[-2] = XI_JMPs;
    }
  } else {
    MCode *newloop;
    p[-5] = XI_JMP;
    if (as->loopinv) {  /* Inverted loop branch? */
      /* asm_guardcc already inverted the jcc and patched the jmp. */
      p -= 5;
      newloop = target+4;
      *(int32_t *)(p-4) = (int32_t)(target - p);  /* Patch jcc. */
      if (as->loopinv == 2) {
	*(int32_t *)(p-10) = (int32_t)(target - p + 6);  /* Patch opt. jp. */
	newloop = target+8;
      }
    } else {  /* Otherwise just patch jmp. */
      *(int32_t *)(p-4) = (int32_t)(target - p);
      newloop = target+3;
    }
    /* Realign small loops and shorten the loop branch. */
    if (newloop >= p - 128) {
      as->realign = newloop;  /* Force a retry and remember alignment. */
      as->curins = as->stopins;  /* Abort asm_trace now. */
      as->T->nins = as->orignins;  /* Remove any added renames. */
    }
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
      emit_rr(as, XO_MOV, r|REX_GC64, RID_BASE);
  }
}

/* Coalesce or reload BASE register for a side trace. */
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
      /* Move from coalesced parent reg. */
      emit_rr(as, XO_MOV, r|REX_GC64, irp->r);
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
  /* Note: don't use as->mcp swap + emit_*: emit_op overwrites more bytes. */
  MCode *p = as->mctop;
  MCode *target, *q;
  int32_t spadj = as->T->spadjust;
  if (spadj == 0) {
    p -= LJ_64 ? 7 : 6;
  } else {
    MCode *p1;
    /* Patch stack adjustment. */
    if (checki8(spadj)) {
      p -= 3;
      p1 = p-6;
      *p1 = (MCode)spadj;
    } else {
      p1 = p-9;
      *(int32_t *)p1 = spadj;
    }
#if LJ_64
    p1[-3] = 0x48;
#endif
    p1[-2] = (MCode)(checki8(spadj) ? XI_ARITHi8 : XI_ARITHi);
    p1[-1] = MODRM(XM_REG, XOg_ADD, RID_ESP);
  }
  /* Patch exit branch. */
  target = lnk ? traceref(as->J, lnk)->mcode : (MCode *)lj_vm_exit_interp;
  *(int32_t *)(p-4) = jmprel(as->J, p, target);
  p[-5] = XI_JMP;
  /* Drop unused mcode tail. Fill with NOPs to make the prefetcher happy. */
  for (q = as->mctop-1; q >= p; q--)
    *q = XI_NOP;
  as->mctop = p;
}

/* Prepare tail of code. */
static void asm_tail_prep(ASMState *as)
{
  MCode *p = as->mctop;
  /* Realign and leave room for backwards loop branch or exit branch. */
  if (as->realign) {
    int i = ((int)(intptr_t)as->realign) & 15;
    /* Fill unused mcode tail with NOPs to make the prefetcher happy. */
    while (i-- > 0)
      *--p = XI_NOP;
    as->mctop = p;
    p -= (as->loopinv ? 5 : 2);  /* Space for short/near jmp. */
  } else {
    p -= 5;  /* Space for exit branch (near jmp). */
  }
  if (as->loopref) {
    as->invmcp = as->mcp = p;
  } else {
    /* Leave room for ESP adjustment: add esp, imm or lea esp, [esp+imm] */
    as->mcp = p - (LJ_64 ? 7 : 6);
    as->invmcp = NULL;
  }
}

/* -- Trace setup --------------------------------------------------------- */

/* Ensure there are enough stack slots for call arguments. */
static Reg asm_setup_call_slots(ASMState *as, IRIns *ir, const CCallInfo *ci)
{
  IRRef args[CCI_NARGS_MAX*2];
  int nslots;
  asm_collectargs(as, ir, ci, args);
  nslots = asm_count_call_slots(as, ci, args);
  if (nslots > as->evenspill)  /* Leave room for args in stack slots. */
    as->evenspill = nslots;
#if LJ_64
  return irt_isfp(ir->t) ? REGSP_HINT(RID_FPRET) : REGSP_HINT(RID_RET);
#else
  return irt_isfp(ir->t) ? REGSP_INIT : REGSP_HINT(RID_RET);
#endif
}

/* Target-specific setup. */
static void asm_setup_target(ASMState *as)
{
  asm_exitstub_setup(as, as->T->nsnap);
  as->mrm.base = 0;
}

/* -- Trace patching ------------------------------------------------------ */

static const uint8_t map_op1[256] = {
0x92,0x92,0x92,0x92,0x52,0x45,0x51,0x51,0x92,0x92,0x92,0x92,0x52,0x45,0x51,0x20,
0x92,0x92,0x92,0x92,0x52,0x45,0x51,0x51,0x92,0x92,0x92,0x92,0x52,0x45,0x51,0x51,
0x92,0x92,0x92,0x92,0x52,0x45,0x10,0x51,0x92,0x92,0x92,0x92,0x52,0x45,0x10,0x51,
0x92,0x92,0x92,0x92,0x52,0x45,0x10,0x51,0x92,0x92,0x92,0x92,0x52,0x45,0x10,0x51,
#if LJ_64
0x10,0x10,0x10,0x10,0x10,0x10,0x10,0x10,0x14,0x14,0x14,0x14,0x14,0x14,0x14,0x14,
#else
0x51,0x51,0x51,0x51,0x51,0x51,0x51,0x51,0x51,0x51,0x51,0x51,0x51,0x51,0x51,0x51,
#endif
0x51,0x51,0x51,0x51,0x51,0x51,0x51,0x51,0x51,0x51,0x51,0x51,0x51,0x51,0x51,0x51,
0x51,0x51,0x92,0x92,0x10,0x10,0x12,0x11,0x45,0x86,0x52,0x93,0x51,0x51,0x51,0x51,
0x52,0x52,0x52,0x52,0x52,0x52,0x52,0x52,0x52,0x52,0x52,0x52,0x52,0x52,0x52,0x52,
0x93,0x86,0x93,0x93,0x92,0x92,0x92,0x92,0x92,0x92,0x92,0x92,0x92,0x92,0x92,0x92,
0x51,0x51,0x51,0x51,0x51,0x51,0x51,0x51,0x51,0x51,0x47,0x51,0x51,0x51,0x51,0x51,
#if LJ_64
0x59,0x59,0x59,0x59,0x51,0x51,0x51,0x51,0x52,0x45,0x51,0x51,0x51,0x51,0x51,0x51,
#else
0x55,0x55,0x55,0x55,0x51,0x51,0x51,0x51,0x52,0x45,0x51,0x51,0x51,0x51,0x51,0x51,
#endif
0x52,0x52,0x52,0x52,0x52,0x52,0x52,0x52,0x05,0x05,0x05,0x05,0x05,0x05,0x05,0x05,
0x93,0x93,0x53,0x51,0x70,0x71,0x93,0x86,0x54,0x51,0x53,0x51,0x51,0x52,0x51,0x51,
0x92,0x92,0x92,0x92,0x52,0x52,0x51,0x51,0x92,0x92,0x92,0x92,0x92,0x92,0x92,0x92,
0x52,0x52,0x52,0x52,0x52,0x52,0x52,0x52,0x45,0x45,0x47,0x52,0x51,0x51,0x51,0x51,
0x10,0x51,0x10,0x10,0x51,0x51,0x63,0x66,0x51,0x51,0x51,0x51,0x51,0x51,0x92,0x92
};

static const uint8_t map_op2[256] = {
0x93,0x93,0x93,0x93,0x52,0x52,0x52,0x52,0x52,0x52,0x51,0x52,0x51,0x93,0x52,0x94,
0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,
0x53,0x53,0x53,0x53,0x53,0x53,0x53,0x53,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,
0x52,0x52,0x52,0x52,0x52,0x52,0x52,0x52,0x34,0x51,0x35,0x51,0x51,0x51,0x51,0x51,
0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,
0x53,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,
0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,
0x94,0x54,0x54,0x54,0x93,0x93,0x93,0x52,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,
0x46,0x46,0x46,0x46,0x46,0x46,0x46,0x46,0x46,0x46,0x46,0x46,0x46,0x46,0x46,0x46,
0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,
0x52,0x52,0x52,0x93,0x94,0x93,0x51,0x51,0x52,0x52,0x52,0x93,0x94,0x93,0x93,0x93,
0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x94,0x93,0x93,0x93,0x93,0x93,
0x93,0x93,0x94,0x93,0x94,0x94,0x94,0x93,0x52,0x52,0x52,0x52,0x52,0x52,0x52,0x52,
0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,
0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,
0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x93,0x52
};

static uint32_t asm_x86_inslen(const uint8_t* p)
{
  uint32_t result = 0;
  uint32_t prefixes = 0;
  uint32_t x = map_op1[*p];
  for (;;) {
    switch (x >> 4) {
    case 0: return result + x + (prefixes & 4);
    case 1: prefixes |= x; x = map_op1[*++p]; result++; break;
    case 2: x = map_op2[*++p]; break;
    case 3: p++; goto mrm;
    case 4: result -= (prefixes & 2);  /* fallthrough */
    case 5: return result + (x & 15);
    case 6:  /* Group 3. */
      if (p[1] & 0x38) x = 2;
      else if ((prefixes & 2) && (x == 0x66)) x = 4;
      goto mrm;
    case 7: /* VEX c4/c5. */
      if (LJ_32 && p[1] < 0xc0) {
	x = 2;
	goto mrm;
      }
      if (x == 0x70) {
	x = *++p & 0x1f;
	result++;
	if (x >= 2) {
	  p += 2;
	  result += 2;
	  goto mrm;
	}
      }
      p++;
      result++;
      x = map_op2[*++p];
      break;
    case 8: result -= (prefixes & 2);  /* fallthrough */
    case 9: mrm:  /* ModR/M and possibly SIB. */
      result += (x & 15);
      x = *++p;
      switch (x >> 6) {
      case 0: if ((x & 7) == 5) return result + 4; break;
      case 1: result++; break;
      case 2: result += 4; break;
      case 3: return result;
      }
      if ((x & 7) == 4) {
	result++;
	if (x < 0x40 && (p[1] & 7) == 5) result += 4;
      }
      return result;
    }
  }
}

/* Patch exit jumps of existing machine code to a new target. */
void lj_asm_patchexit(jit_State *J, GCtrace *T, ExitNo exitno, MCode *target)
{
  MCode *p = T->mcode;
  MCode *mcarea = lj_mcode_patch(J, p, 0);
  MSize len = T->szmcode;
  MCode *px = exitstub_addr(J, exitno) - 6;
  MCode *pe = p+len-6;
  MCode *pgc = NULL;
#if LJ_GC64
  uint32_t statei = (uint32_t)(GG_OFS(g.vmstate) - GG_OFS(dispatch));
#else
  uint32_t statei = u32ptr(&J2G(J)->vmstate);
#endif
  if (len > 5 && p[len-5] == XI_JMP && p+len-6 + *(int32_t *)(p+len-4) == px)
    *(int32_t *)(p+len-4) = jmprel(J, p+len, target);
  /* Do not patch parent exit for a stack check. Skip beyond vmstate update. */
  for (; p < pe; p += asm_x86_inslen(p)) {
    intptr_t ofs = LJ_GC64 ? (p[0] & 0xf0) == 0x40 : LJ_64;
    if (*(uint32_t *)(p+2+ofs) == statei && p[ofs+LJ_GC64-LJ_64] == XI_MOVmi)
      break;
  }
  lj_assertJ(p < pe, "instruction length decoder failed");
  for (; p < pe; p += asm_x86_inslen(p)) {
    if ((*(uint16_t *)p & 0xf0ff) == 0x800f && p + *(int32_t *)(p+2) == px &&
	p != pgc) {
      *(int32_t *)(p+2) = jmprel(J, p+6, target);
    } else if (*p == XI_CALL &&
	      (void *)(p+5+*(int32_t *)(p+1)) == (void *)lj_gc_step_jit) {
      pgc = p+7;  /* Do not patch GC check exit. */
    }
  }
  lj_mcode_sync(T->mcode, T->mcode + T->szmcode);
  lj_mcode_patch(J, mcarea, 1);
}

