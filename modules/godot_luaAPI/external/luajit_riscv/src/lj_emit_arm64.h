/*
** ARM64 instruction emitter.
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
**
** Contributed by Djordje Kovacevic and Stefan Pejic from RT-RK.com.
** Sponsored by Cisco Systems, Inc.
*/

/* -- Constant encoding --------------------------------------------------- */

static uint64_t get_k64val(ASMState *as, IRRef ref)
{
  IRIns *ir = IR(ref);
  if (ir->o == IR_KINT64) {
    return ir_kint64(ir)->u64;
  } else if (ir->o == IR_KGC) {
    return (uint64_t)ir_kgc(ir);
  } else if (ir->o == IR_KPTR || ir->o == IR_KKPTR) {
    return (uint64_t)ir_kptr(ir);
  } else {
    lj_assertA(ir->o == IR_KINT || ir->o == IR_KNULL,
	       "bad 64 bit const IR op %d", ir->o);
    return (uint32_t)ir->i;  /* Zero-extended. */
  }
}

/* Encode constant in K12 format for data processing instructions. */
static uint32_t emit_isk12(int64_t n)
{
  uint64_t k = n < 0 ? ~(uint64_t)n+1u : (uint64_t)n;
  uint32_t m = n < 0 ? 0x40000000 : 0;
  if (k < 0x1000) {
    return (uint32_t)(A64I_K12|m|A64F_U12(k));
  } else if ((k & 0xfff000) == k) {
    return (uint32_t)(A64I_K12|m|0x400000|A64F_U12(k>>12));
  }
  return 0;
}

#define emit_clz64(n)	(lj_fls64(n)^63)
#define emit_ctz64(n)	lj_ffs64(n)

/* Encode constant in K13 format for logical data processing instructions. */
static uint32_t emit_isk13(uint64_t n, int is64)
{
  /* Thanks to: https://dougallj.wordpress.com/2021/10/30/ */
  int rot, ones, size, immr, imms;
  if (!is64) n = ((uint64_t)n << 32) | (uint32_t)n;
  if ((n+1u) <= 1u) return 0;  /* Neither all-zero nor all-ones are allowed. */
  rot = (n & (n+1u)) ? emit_ctz64(n & (n+1u)) : 64;
  n = lj_ror(n, rot & 63);
  ones = emit_ctz64(~n);
  size = emit_clz64(n) + ones;
  if (lj_ror(n, size & 63) != n) return 0;  /* Non-repeating? */
  immr = -rot & (size - 1);
  imms = (-(size << 1) | (ones - 1)) & 63;
  return A64I_K13 | A64F_IMMR(immr | (size & 64)) | A64F_IMMS(imms);
}

static uint32_t emit_isfpk64(uint64_t n)
{
  uint64_t etop9 = ((n >> 54) & 0x1ff);
  if ((n << 16) == 0 && (etop9 == 0x100 || etop9 == 0x0ff)) {
    return (uint32_t)(((n >> 48) & 0x7f) | ((n >> 56) & 0x80));
  }
  return ~0u;
}

/* -- Emit basic instructions --------------------------------------------- */

static void emit_dnma(ASMState *as, A64Ins ai, Reg rd, Reg rn, Reg rm, Reg ra)
{
  *--as->mcp = ai | A64F_D(rd) | A64F_N(rn) | A64F_M(rm) | A64F_A(ra);
}

static void emit_dnm(ASMState *as, A64Ins ai, Reg rd, Reg rn, Reg rm)
{
  *--as->mcp = ai | A64F_D(rd) | A64F_N(rn) | A64F_M(rm);
}

static void emit_dm(ASMState *as, A64Ins ai, Reg rd, Reg rm)
{
  *--as->mcp = ai | A64F_D(rd) | A64F_M(rm);
}

static void emit_dn(ASMState *as, A64Ins ai, Reg rd, Reg rn)
{
  *--as->mcp = ai | A64F_D(rd) | A64F_N(rn);
}

static void emit_nm(ASMState *as, A64Ins ai, Reg rn, Reg rm)
{
  *--as->mcp = ai | A64F_N(rn) | A64F_M(rm);
}

static void emit_d(ASMState *as, A64Ins ai, Reg rd)
{
  *--as->mcp = ai | A64F_D(rd);
}

static void emit_n(ASMState *as, A64Ins ai, Reg rn)
{
  *--as->mcp = ai | A64F_N(rn);
}

static int emit_checkofs(A64Ins ai, int64_t ofs)
{
  int scale = (ai >> 30) & 3;
  if (ofs < 0 || (ofs & ((1<<scale)-1))) {
    return (ofs >= -256 && ofs <= 255) ? -1 : 0;
  } else {
    return (ofs < (4096<<scale)) ? 1 : 0;
  }
}

static LJ_AINLINE uint32_t emit_lso_pair_candidate(A64Ins ai, int ofs, int sc)
{
  if (ofs >= 0) {
    return ai | A64F_U12(ofs>>sc);  /* Subsequent lj_ror checks ofs. */
  } else if (ofs >= -256) {
    return (ai^A64I_LS_U) | A64F_S9(ofs & 0x1ff);
  } else {
    return A64F_D(31);  /* Will mismatch prev. */
  }
}

static void emit_lso(ASMState *as, A64Ins ai, Reg rd, Reg rn, int64_t ofs64)
{
  int ot = emit_checkofs(ai, ofs64), sc = (ai >> 30) & 3, ofs = (int)ofs64;
  lj_assertA(ot, "load/store offset %d out of range", ofs);
  /* Combine LDR/STR pairs to LDP/STP. */
  if ((sc == 2 || sc == 3) &&
      (!(ai & 0x400000) || rd != rn) &&
      as->mcp != as->mcloop) {
    uint32_t prev = *as->mcp & ~A64F_D(31);
    int ofsm = ofs - (1<<sc), ofsp = ofs + (1<<sc);
    A64Ins aip;
    if (prev == emit_lso_pair_candidate(ai | A64F_N(rn), ofsm, sc)) {
      aip = (A64F_A(rd) | A64F_D(*as->mcp & 31));
    } else if (prev == emit_lso_pair_candidate(ai | A64F_N(rn), ofsp, sc)) {
      aip = (A64F_D(rd) | A64F_A(*as->mcp & 31));
      ofsm = ofs;
    } else {
      goto nopair;
    }
    if (lj_ror((unsigned int)ofsm + (64u<<sc), sc) <= 127u) {
      *as->mcp = aip | A64F_N(rn) | (((ofsm >> sc) & 0x7f) << 15) |
	(ai ^ ((ai == A64I_LDRx || ai == A64I_STRx) ? 0x50000000 : 0x90000000));
      return;
    }
  }
nopair:
  if (ot == 1)
    *--as->mcp = ai | A64F_D(rd) | A64F_N(rn) | A64F_U12(ofs >> sc);
  else
    *--as->mcp = (ai^A64I_LS_U) | A64F_D(rd) | A64F_N(rn) | A64F_S9(ofs & 0x1ff);
}

/* -- Emit loads/stores --------------------------------------------------- */

/* Prefer rematerialization of BASE/L from global_State over spills. */
#define emit_canremat(ref)	((ref) <= REF_BASE)

/* Try to find a one-step delta relative to other consts. */
static int emit_kdelta(ASMState *as, Reg rd, uint64_t k, int is64)
{
  RegSet work = (~as->freeset & RSET_GPR) | RID2RSET(RID_GL);
  while (work) {
    Reg r = rset_picktop(work);
    IRRef ref = regcost_ref(as->cost[r]);
    lj_assertA(r != rd, "dest reg %d not free", rd);
    if (ref < REF_TRUE) {
      uint64_t kx = ra_iskref(ref) ? (uint64_t)ra_krefk(as, ref) :
				     get_k64val(as, ref);
      int64_t delta = (int64_t)(k - kx);
      if (!is64) delta = (int64_t)(int32_t)delta;  /* Sign-extend. */
      if (delta == 0) {
	emit_dm(as, is64|A64I_MOVw, rd, r);
	return 1;
      } else {
	uint32_t k12 = emit_isk12(delta < 0 ? (int64_t)(~(uint64_t)delta+1u) : delta);
	if (k12) {
	  emit_dn(as, (delta < 0 ? A64I_SUBw : A64I_ADDw)^is64^k12, rd, r);
	  return 1;
	}
	/* Do other ops or multi-step deltas pay off? Probably not.
	** E.g. XOR rarely helps with pointer consts.
	*/
      }
    }
    rset_clear(work, r);
  }
  return 0;  /* Failed. */
}

#define glofs(as, k) \
  ((intptr_t)((uintptr_t)(k) - (uintptr_t)&J2GG(as->J)->g))
#define mcpofs(as, k) \
  ((intptr_t)((uintptr_t)(k) - (uintptr_t)(as->mcp - 1)))
#define checkmcpofs(as, k) \
  (A64F_S_OK(mcpofs(as, k)>>2, 19))

/* Try to form a const as ADR or ADRP or ADRP + ADD. */
static int emit_kadrp(ASMState *as, Reg rd, uint64_t k)
{
  A64Ins ai = A64I_ADR;
  int64_t ofs = mcpofs(as, k);
  if (!A64F_S_OK((uint64_t)ofs, 21)) {
    uint64_t kpage = k & ~0xfffull;
    MCode *adrp = as->mcp - 1 - (k != kpage);
    ofs = (int64_t)(kpage - ((uint64_t)adrp & ~0xfffull)) >> 12;
    if (!A64F_S_OK(ofs, 21))
      return 0;  /* Failed. */
    if (k != kpage)
      emit_dn(as, (A64I_ADDx^A64I_K12)|A64F_U12(k - kpage), rd, rd);
    ai = A64I_ADRP;
  }
  emit_d(as, ai|(((uint32_t)ofs&3)<<29)|A64F_S19(ofs>>2), rd);
  return 1;
}

static void emit_loadk(ASMState *as, Reg rd, uint64_t u64)
{
  int zeros = 0, ones = 0, neg, lshift = 0;
  int is64 = (u64 >> 32) ? A64I_X : 0, i = is64 ? 4 : 2;
  /* Count non-homogeneous 16 bit fragments. */
  while (--i >= 0) {
    uint32_t frag = (u64 >> i*16) & 0xffff;
    zeros += (frag != 0);
    ones += (frag != 0xffff);
  }
  neg = ones < zeros;  /* Use MOVN if it pays off. */
  if ((neg ? ones : zeros) > 1) {  /* Need 2+ ins. Try 1 ins encodings. */
    uint32_t k13 = emit_isk13(u64, is64);
    if (k13) {
      emit_dn(as, (is64|A64I_ORRw)^k13, rd, RID_ZERO);
      return;
    }
    if (emit_kdelta(as, rd, u64, is64)) {
      return;
    }
    if (emit_kadrp(as, rd, u64)) {  /* Either 1 or 2 ins. */
      return;
    }
  }
  if (neg) {
    u64 = ~u64;
    if (!is64) u64 = (uint32_t)u64;
  }
  if (u64) {
    /* Find first/last fragment to be filled. */
    int shift = (63-emit_clz64(u64)) & ~15;
    lshift = emit_ctz64(u64) & ~15;
    for (; shift > lshift; shift -= 16) {
      uint32_t frag = (u64 >> shift) & 0xffff;
      if (frag == 0) continue; /* Will be correctly filled by MOVN/MOVZ. */
      if (neg) frag ^= 0xffff; /* MOVK requires the original value. */
      emit_d(as, is64 | A64I_MOVKw | A64F_U16(frag) | A64F_LSL16(shift), rd);
    }
  }
  /* But MOVN needs an inverted value. */
  emit_d(as, is64 | (neg ? A64I_MOVNw : A64I_MOVZw) |
	     A64F_U16((u64 >> lshift) & 0xffff) | A64F_LSL16(lshift), rd);
}

/* Load a 32 bit constant into a GPR. */
#define emit_loadi(as, rd, i)	emit_loadk(as, rd, (uint32_t)i)

/* Load a 64 bit constant into a GPR. */
#define emit_loadu64(as, rd, i)	emit_loadk(as, rd, i)

static Reg ra_allock(ASMState *as, intptr_t k, RegSet allow);

/* Get/set from constant pointer. */
static void emit_lsptr(ASMState *as, A64Ins ai, Reg r, void *p)
{
  Reg base = RID_GL;
  int64_t ofs = glofs(as, p);
  if (emit_checkofs(ai, ofs)) {
    /* GL + offset, might subsequently fuse to LDP/STP. */
  } else if (ai == A64I_LDRx && checkmcpofs(as, p)) {
    /* IP + offset is cheaper than allock, but address must be in range. */
    emit_d(as, A64I_LDRLx | A64F_S19(mcpofs(as, p)>>2), r);
    return;
  } else {  /* Split up into base reg + offset. */
    int64_t i64 = i64ptr(p);
    base = ra_allock(as, (i64 & ~0x7fffull), rset_exclude(RSET_GPR, r));
    ofs = i64 & 0x7fffull;
  }
  emit_lso(as, ai, r, base, ofs);
}

/* Load 64 bit IR constant into register. */
static void emit_loadk64(ASMState *as, Reg r, IRIns *ir)
{
  const uint64_t *k = &ir_k64(ir)->u64;
  int64_t ofs;
  if (r >= RID_MAX_GPR) {
    uint32_t fpk = emit_isfpk64(*k);
    if (fpk != ~0u) {
      emit_d(as, A64I_FMOV_DI | A64F_FP8(fpk), (r & 31));
      return;
    }
  }
  ofs = glofs(as, k);
  if (emit_checkofs(A64I_LDRx, ofs)) {
    emit_lso(as, r >= RID_MAX_GPR ? A64I_LDRd : A64I_LDRx,
	     (r & 31), RID_GL, ofs);
  } else {
    if (r >= RID_MAX_GPR) {
      emit_dn(as, A64I_FMOV_D_R, (r & 31), RID_TMP);
      r = RID_TMP;
    }
    if (checkmcpofs(as, k))
      emit_d(as, A64I_LDRLx | A64F_S19(mcpofs(as, k)>>2), r);
    else
      emit_loadu64(as, r, *k);
  }
}

/* Get/set global_State fields. */
#define emit_getgl(as, r, field) \
  emit_lsptr(as, A64I_LDRx, (r), (void *)&J2G(as->J)->field)
#define emit_setgl(as, r, field) \
  emit_lsptr(as, A64I_STRx, (r), (void *)&J2G(as->J)->field)

/* Trace number is determined from pc of exit instruction. */
#define emit_setvmstate(as, i)	UNUSED(i)

/* -- Emit control-flow instructions -------------------------------------- */

/* Label for internal jumps. */
typedef MCode *MCLabel;

/* Return label pointing to current PC. */
#define emit_label(as)		((as)->mcp)

static void emit_cond_branch(ASMState *as, A64CC cond, MCode *target)
{
  MCode *p = --as->mcp;
  ptrdiff_t delta = target - p;
  lj_assertA(A64F_S_OK(delta, 19), "branch target out of range");
  *p = A64I_BCC | A64F_S19(delta) | cond;
}

static void emit_branch(ASMState *as, A64Ins ai, MCode *target)
{
  MCode *p = --as->mcp;
  ptrdiff_t delta = target - p;
  lj_assertA(A64F_S_OK(delta, 26), "branch target out of range");
  *p = ai | A64F_S26(delta);
}

static void emit_tnb(ASMState *as, A64Ins ai, Reg r, uint32_t bit, MCode *target)
{
  MCode *p = --as->mcp;
  ptrdiff_t delta = target - p;
  lj_assertA(bit < 63, "bit number out of range");
  lj_assertA(A64F_S_OK(delta, 14), "branch target out of range");
  if (bit > 31) ai |= A64I_X;
  *p = ai | A64F_BIT(bit & 31) | A64F_S14(delta) | r;
}

static void emit_cnb(ASMState *as, A64Ins ai, Reg r, MCode *target)
{
  MCode *p = --as->mcp;
  ptrdiff_t delta = target - p;
  lj_assertA(A64F_S_OK(delta, 19), "branch target out of range");
  *p = ai | A64F_S19(delta) | r;
}

#define emit_jmp(as, target)	emit_branch(as, A64I_B, (target))

static void emit_call(ASMState *as, ASMFunction target)
{
  MCode *p = --as->mcp;
#if LJ_ABI_PAUTH
  char *targetp = ptrauth_auth_data((char *)target,
				    ptrauth_key_function_pointer, 0);
#else
  char *targetp = (char *)target;
#endif
  ptrdiff_t delta = targetp - (char *)p;
  if (A64F_S_OK(delta>>2, 26)) {
    *p = A64I_BL | A64F_S26(delta>>2);
  } else {  /* Target out of range: need indirect call. But don't use R0-R7. */
    Reg r = ra_allock(as, i64ptr(target),
		      RSET_RANGE(RID_X8, RID_MAX_GPR)-RSET_FIXED);
    *p = A64I_BLR_AUTH | A64F_N(r);
  }
}

/* -- Emit generic operations --------------------------------------------- */

/* Generic move between two regs. */
static void emit_movrr(ASMState *as, IRIns *ir, Reg dst, Reg src)
{
  if (dst >= RID_MAX_GPR) {
    emit_dn(as, irt_isnum(ir->t) ? A64I_FMOV_D : A64I_FMOV_S,
	    (dst & 31), (src & 31));
    return;
  }
  if (as->mcp != as->mcloop) {  /* Swap early registers for loads/stores. */
    MCode ins = *as->mcp, swp = (src^dst);
    if ((ins & 0xbf800000) == 0xb9000000) {
      if (!((ins ^ (dst << 5)) & 0x000003e0))
	*as->mcp = ins ^ (swp << 5);  /* Swap N in load/store. */
      if (!(ins & 0x00400000) && !((ins ^ dst) & 0x0000001f))
	*as->mcp = ins ^ swp;  /* Swap D in store. */
    }
  }
  emit_dm(as, A64I_MOVx, dst, src);
}

/* Generic load of register with base and (small) offset address. */
static void emit_loadofs(ASMState *as, IRIns *ir, Reg r, Reg base, int32_t ofs)
{
  if (r >= RID_MAX_GPR)
    emit_lso(as, irt_isnum(ir->t) ? A64I_LDRd : A64I_LDRs, (r & 31), base, ofs);
  else
    emit_lso(as, irt_is64(ir->t) ? A64I_LDRx : A64I_LDRw, r, base, ofs);
}

/* Generic store of register with base and (small) offset address. */
static void emit_storeofs(ASMState *as, IRIns *ir, Reg r, Reg base, int32_t ofs)
{
  if (r >= RID_MAX_GPR)
    emit_lso(as, irt_isnum(ir->t) ? A64I_STRd : A64I_STRs, (r & 31), base, ofs);
  else
    emit_lso(as, irt_is64(ir->t) ? A64I_STRx : A64I_STRw, r, base, ofs);
}

/* Emit an arithmetic operation with a constant operand. */
static void emit_opk(ASMState *as, A64Ins ai, Reg dest, Reg src,
		     int32_t i, RegSet allow)
{
  uint32_t k = emit_isk12(i);
  if (k)
    emit_dn(as, ai^k, dest, src);
  else
    emit_dnm(as, ai, dest, src, ra_allock(as, i, allow));
}

/* Add offset to pointer. */
static void emit_addptr(ASMState *as, Reg r, int32_t ofs)
{
  if (ofs)
    emit_opk(as, ofs < 0 ? A64I_SUBx : A64I_ADDx, r, r,
		 ofs < 0 ? (int32_t)(~(uint32_t)ofs+1u) : ofs,
		 rset_exclude(RSET_GPR, r));
}

#define emit_spsub(as, ofs)	emit_addptr(as, RID_SP, -(ofs))

