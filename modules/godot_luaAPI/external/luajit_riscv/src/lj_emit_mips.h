/*
** MIPS instruction emitter.
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
*/

#if LJ_64
static intptr_t get_k64val(ASMState *as, IRRef ref)
{
  IRIns *ir = IR(ref);
  if (ir->o == IR_KINT64) {
    return (intptr_t)ir_kint64(ir)->u64;
  } else if (ir->o == IR_KGC) {
    return (intptr_t)ir_kgc(ir);
  } else if (ir->o == IR_KPTR || ir->o == IR_KKPTR) {
    return (intptr_t)ir_kptr(ir);
  } else if (LJ_SOFTFP && ir->o == IR_KNUM) {
    return (intptr_t)ir_knum(ir)->u64;
  } else {
    lj_assertA(ir->o == IR_KINT || ir->o == IR_KNULL,
	       "bad 64 bit const IR op %d", ir->o);
    return ir->i;  /* Sign-extended. */
  }
}
#endif

#if LJ_64
#define get_kval(as, ref)	get_k64val(as, ref)
#else
#define get_kval(as, ref)	(IR((ref))->i)
#endif

/* -- Emit basic instructions --------------------------------------------- */

static void emit_dst(ASMState *as, MIPSIns mi, Reg rd, Reg rs, Reg rt)
{
  *--as->mcp = mi | MIPSF_D(rd) | MIPSF_S(rs) | MIPSF_T(rt);
}

static void emit_dta(ASMState *as, MIPSIns mi, Reg rd, Reg rt, uint32_t a)
{
  *--as->mcp = mi | MIPSF_D(rd) | MIPSF_T(rt) | MIPSF_A(a);
}

#define emit_ds(as, mi, rd, rs)		emit_dst(as, (mi), (rd), (rs), 0)
#define emit_tg(as, mi, rt, rg)		emit_dst(as, (mi), (rg)&31, 0, (rt))

static void emit_tsi(ASMState *as, MIPSIns mi, Reg rt, Reg rs, int32_t i)
{
  *--as->mcp = mi | MIPSF_T(rt) | MIPSF_S(rs) | (i & 0xffff);
}

#define emit_ti(as, mi, rt, i)		emit_tsi(as, (mi), (rt), 0, (i))
#define emit_hsi(as, mi, rh, rs, i)	emit_tsi(as, (mi), (rh) & 31, (rs), (i))

static void emit_fgh(ASMState *as, MIPSIns mi, Reg rf, Reg rg, Reg rh)
{
  *--as->mcp = mi | MIPSF_F(rf&31) | MIPSF_G(rg&31) | MIPSF_H(rh&31);
}

#define emit_fg(as, mi, rf, rg)		emit_fgh(as, (mi), (rf), (rg), 0)

static void emit_rotr(ASMState *as, Reg dest, Reg src, Reg tmp, uint32_t shift)
{
  if (LJ_64 || (as->flags & JIT_F_MIPSXXR2)) {
    emit_dta(as, MIPSI_ROTR, dest, src, shift);
  } else {
    emit_dst(as, MIPSI_OR, dest, dest, tmp);
    emit_dta(as, MIPSI_SLL, dest, src, (-shift)&31);
    emit_dta(as, MIPSI_SRL, tmp, src, shift);
  }
}

#if LJ_64 || LJ_HASBUFFER
static void emit_tsml(ASMState *as, MIPSIns mi, Reg rt, Reg rs, uint32_t msb,
		      uint32_t lsb)
{
  *--as->mcp = mi | MIPSF_T(rt) | MIPSF_S(rs) | MIPSF_M(msb) | MIPSF_L(lsb);
}
#endif

/* -- Emit loads/stores --------------------------------------------------- */

/* Prefer rematerialization of BASE/L from global_State over spills. */
#define emit_canremat(ref)	((ref) <= REF_BASE)

/* Try to find a one step delta relative to another constant. */
static int emit_kdelta1(ASMState *as, Reg rd, intptr_t i)
{
  RegSet work = ~as->freeset & RSET_GPR;
  while (work) {
    Reg r = rset_picktop(work);
    IRRef ref = regcost_ref(as->cost[r]);
    lj_assertA(r != rd, "dest reg %d not free", rd);
    if (ref < ASMREF_L) {
      intptr_t delta = (intptr_t)((uintptr_t)i -
	(uintptr_t)(ra_iskref(ref) ? ra_krefk(as, ref) : get_kval(as, ref)));
      if (checki16(delta)) {
	emit_tsi(as, MIPSI_AADDIU, rd, r, delta);
	return 1;
      }
    }
    rset_clear(work, r);
  }
  return 0;  /* Failed. */
}

/* Load a 32 bit constant into a GPR. */
static void emit_loadi(ASMState *as, Reg r, int32_t i)
{
  if (checki16(i)) {
    emit_ti(as, MIPSI_LI, r, i);
  } else {
    if ((i & 0xffff)) {
      intptr_t jgl = (intptr_t)(void *)J2G(as->J);
      if ((uintptr_t)(i-jgl) < 65536) {
	emit_tsi(as, MIPSI_ADDIU, r, RID_JGL, i-jgl-32768);
	return;
      } else if (emit_kdelta1(as, r, i)) {
	return;
      } else if ((i >> 16) == 0) {
	emit_tsi(as, MIPSI_ORI, r, RID_ZERO, i);
	return;
      }
      emit_tsi(as, MIPSI_ORI, r, r, i);
    }
    emit_ti(as, MIPSI_LUI, r, (i >> 16));
  }
}

#if LJ_64
/* Load a 64 bit constant into a GPR. */
static void emit_loadu64(ASMState *as, Reg r, uint64_t u64)
{
  if (checki32((int64_t)u64)) {
    emit_loadi(as, r, (int32_t)u64);
  } else {
    uint64_t delta = u64 - (uint64_t)(void *)J2G(as->J);
    if (delta < 65536) {
      emit_tsi(as, MIPSI_DADDIU, r, RID_JGL, (int32_t)(delta-32768));
    } else if (emit_kdelta1(as, r, (intptr_t)u64)) {
      return;
    } else {
      /* TODO MIPSR6: Use DAHI & DATI. Caveat: sign-extension. */
      if ((u64 & 0xffff)) {
	emit_tsi(as, MIPSI_ORI, r, r, u64 & 0xffff);
      }
      if (((u64 >> 16) & 0xffff)) {
	emit_dta(as, MIPSI_DSLL, r, r, 16);
	emit_tsi(as, MIPSI_ORI, r, r, (u64 >> 16) & 0xffff);
	emit_dta(as, MIPSI_DSLL, r, r, 16);
      } else {
	emit_dta(as, MIPSI_DSLL32, r, r, 0);
      }
      emit_loadi(as, r, (int32_t)(u64 >> 32));
    }
    /* TODO: There are probably more optimization opportunities. */
  }
}

#define emit_loada(as, r, addr)		emit_loadu64(as, (r), u64ptr((addr)))
#else
#define emit_loada(as, r, addr)		emit_loadi(as, (r), i32ptr((addr)))
#endif

static Reg ra_allock(ASMState *as, intptr_t k, RegSet allow);
static void ra_allockreg(ASMState *as, intptr_t k, Reg r);

/* Get/set from constant pointer. */
static void emit_lsptr(ASMState *as, MIPSIns mi, Reg r, void *p, RegSet allow)
{
  intptr_t jgl = (intptr_t)(J2G(as->J));
  intptr_t i = (intptr_t)(p);
  Reg base;
  if ((uint32_t)(i-jgl) < 65536) {
    i = i-jgl-32768;
    base = RID_JGL;
  } else {
    base = ra_allock(as, i-(int16_t)i, allow);
  }
  emit_tsi(as, mi, r, base, i);
}

#if LJ_64
static void emit_loadk64(ASMState *as, Reg r, IRIns *ir)
{
  const uint64_t *k = &ir_k64(ir)->u64;
  Reg r64 = r;
  if (rset_test(RSET_FPR, r)) {
    r64 = RID_TMP;
    emit_tg(as, MIPSI_DMTC1, r64, r);
  }
  if ((uint32_t)((intptr_t)k-(intptr_t)J2G(as->J)) < 65536)
    emit_lsptr(as, MIPSI_LD, r64, (void *)k, 0);
  else
    emit_loadu64(as, r64, *k);
}
#else
#define emit_loadk64(as, r, ir) \
  emit_lsptr(as, MIPSI_LDC1, ((r) & 31), (void *)&ir_knum((ir))->u64, RSET_GPR)
#endif

/* Get/set global_State fields. */
static void emit_lsglptr(ASMState *as, MIPSIns mi, Reg r, int32_t ofs)
{
  emit_tsi(as, mi, r, RID_JGL, ofs-32768);
}

#define emit_getgl(as, r, field) \
  emit_lsglptr(as, MIPSI_AL, (r), (int32_t)offsetof(global_State, field))
#define emit_setgl(as, r, field) \
  emit_lsglptr(as, MIPSI_AS, (r), (int32_t)offsetof(global_State, field))

/* Trace number is determined from per-trace exit stubs. */
#define emit_setvmstate(as, i)		UNUSED(i)

/* -- Emit control-flow instructions -------------------------------------- */

/* Label for internal jumps. */
typedef MCode *MCLabel;

/* Return label pointing to current PC. */
#define emit_label(as)		((as)->mcp)

static void emit_branch(ASMState *as, MIPSIns mi, Reg rs, Reg rt, MCode *target)
{
  MCode *p = as->mcp;
  ptrdiff_t delta = target - p;
  lj_assertA(((delta + 0x8000) >> 16) == 0, "branch target out of range");
  *--p = mi | MIPSF_S(rs) | MIPSF_T(rt) | ((uint32_t)delta & 0xffffu);
  as->mcp = p;
}

static void emit_jmp(ASMState *as, MCode *target)
{
  *--as->mcp = MIPSI_NOP;
  emit_branch(as, MIPSI_B, RID_ZERO, RID_ZERO, (target));
}

static void emit_call(ASMState *as, void *target, int needcfa)
{
  MCode *p = as->mcp;
#if LJ_TARGET_MIPSR6
  ptrdiff_t delta = (char *)target - (char *)p;
  if ((((delta>>2) + 0x02000000) >> 26) == 0) {  /* Try compact call first. */
    *--p = MIPSI_BALC | (((uintptr_t)delta >>2) & 0x03ffffffu);
    as->mcp = p;
    return;
  }
#endif
  *--p = MIPSI_NOP;  /* Delay slot. */
  if ((((uintptr_t)target ^ (uintptr_t)p) >> 28) == 0) {
#if !LJ_TARGET_MIPSR6
    *--p = (((uintptr_t)target & 1) ? MIPSI_JALX : MIPSI_JAL) |
	   (((uintptr_t)target >>2) & 0x03ffffffu);
#else
    *--p = MIPSI_JAL | (((uintptr_t)target >>2) & 0x03ffffffu);
#endif
  } else {  /* Target out of range: need indirect call. */
    *--p = MIPSI_JALR | MIPSF_S(RID_CFUNCADDR);
    needcfa = 1;
  }
  as->mcp = p;
  if (needcfa) ra_allockreg(as, (intptr_t)target, RID_CFUNCADDR);
}

/* -- Emit generic operations --------------------------------------------- */

#define emit_move(as, dst, src) \
  emit_ds(as, MIPSI_MOVE, (dst), (src))

/* Generic move between two regs. */
static void emit_movrr(ASMState *as, IRIns *ir, Reg dst, Reg src)
{
  if (dst < RID_MAX_GPR)
    emit_move(as, dst, src);
  else
    emit_fg(as, irt_isnum(ir->t) ? MIPSI_MOV_D : MIPSI_MOV_S, dst, src);
}

/* Generic load of register with base and (small) offset address. */
static void emit_loadofs(ASMState *as, IRIns *ir, Reg r, Reg base, int32_t ofs)
{
  if (r < RID_MAX_GPR)
    emit_tsi(as, irt_is64(ir->t) ? MIPSI_LD : MIPSI_LW, r, base, ofs);
  else
    emit_tsi(as, irt_isnum(ir->t) ? MIPSI_LDC1 : MIPSI_LWC1,
	     (r & 31), base, ofs);
}

/* Generic store of register with base and (small) offset address. */
static void emit_storeofs(ASMState *as, IRIns *ir, Reg r, Reg base, int32_t ofs)
{
  if (r < RID_MAX_GPR)
    emit_tsi(as, irt_is64(ir->t) ? MIPSI_SD : MIPSI_SW, r, base, ofs);
  else
    emit_tsi(as, irt_isnum(ir->t) ? MIPSI_SDC1 : MIPSI_SWC1,
	     (r&31), base, ofs);
}

/* Add offset to pointer. */
static void emit_addptr(ASMState *as, Reg r, int32_t ofs)
{
  if (ofs) {
    lj_assertA(checki16(ofs), "offset %d out of range", ofs);
    emit_tsi(as, MIPSI_AADDIU, r, r, ofs);
  }
}

#define emit_spsub(as, ofs)	emit_addptr(as, RID_SP, -(ofs))

