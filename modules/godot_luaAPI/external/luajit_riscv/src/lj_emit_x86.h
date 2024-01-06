/*
** x86/x64 instruction emitter.
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
*/

/* -- Emit basic instructions --------------------------------------------- */

#define MODRM(mode, r1, r2)	((MCode)((mode)+(((r1)&7)<<3)+((r2)&7)))

#if LJ_64
#define REXRB(p, rr, rb) \
    { MCode rex = 0x40 + (((rr)>>1)&4) + (((rb)>>3)&1); \
      if (rex != 0x40) *--(p) = rex; }
#define FORCE_REX		0x200
#define REX_64			(FORCE_REX|0x080000)
#define VEX_64			0x800000
#else
#define REXRB(p, rr, rb)	((void)0)
#define FORCE_REX		0
#define REX_64			0
#define VEX_64			0
#endif
#if LJ_GC64
#define REX_GC64		REX_64
#else
#define REX_GC64		0
#endif

#define emit_i8(as, i)		(*--as->mcp = (MCode)(i))
#define emit_i32(as, i)		(*(int32_t *)(as->mcp-4) = (i), as->mcp -= 4)
#define emit_u32(as, u)		(*(uint32_t *)(as->mcp-4) = (u), as->mcp -= 4)

#define emit_x87op(as, xo) \
  (*(uint16_t *)(as->mcp-2) = (uint16_t)(xo), as->mcp -= 2)

/* op */
static LJ_AINLINE MCode *emit_op(x86Op xo, Reg rr, Reg rb, Reg rx,
				 MCode *p, int delta)
{
  int n = (int8_t)xo;
  if (n == -60) {  /* VEX-encoded instruction */
#if LJ_64
    xo ^= (((rr>>1)&4)+((rx>>2)&2)+((rb>>3)&1))<<13;
#endif
    *(uint32_t *)(p+delta-5) = (uint32_t)xo;
    return p+delta-5;
  }
#if defined(__GNUC__) || defined(__clang__)
  if (__builtin_constant_p(xo) && n == -2)
    p[delta-2] = (MCode)(xo >> 24);
  else if (__builtin_constant_p(xo) && n == -3)
    *(uint16_t *)(p+delta-3) = (uint16_t)(xo >> 16);
  else
#endif
    *(uint32_t *)(p+delta-5) = (uint32_t)xo;
  p += n + delta;
#if LJ_64
  {
    uint32_t rex = 0x40 + ((rr>>1)&(4+(FORCE_REX>>1)))+((rx>>2)&2)+((rb>>3)&1);
    if (rex != 0x40) {
      rex |= (rr >> 16);
      if (n == -4) { *p = (MCode)rex; rex = (MCode)(xo >> 8); }
      else if ((xo & 0xffffff) == 0x6600fd) { *p = (MCode)rex; rex = 0x66; }
      *--p = (MCode)rex;
    }
  }
#else
  UNUSED(rr); UNUSED(rb); UNUSED(rx);
#endif
  return p;
}

/* op + modrm */
#define emit_opm(xo, mode, rr, rb, p, delta) \
  (p[(delta)-1] = MODRM((mode), (rr), (rb)), \
   emit_op((xo), (rr), (rb), 0, (p), (delta)))

/* op + modrm + sib */
#define emit_opmx(xo, mode, scale, rr, rb, rx, p) \
  (p[-1] = MODRM((scale), (rx), (rb)), \
   p[-2] = MODRM((mode), (rr), RID_ESP), \
   emit_op((xo), (rr), (rb), (rx), (p), -1))

/* op r1, r2 */
static void emit_rr(ASMState *as, x86Op xo, Reg r1, Reg r2)
{
  MCode *p = as->mcp;
  as->mcp = emit_opm(xo, XM_REG, r1, r2, p, 0);
}

#if LJ_64 && defined(LUA_USE_ASSERT)
/* [addr] is sign-extended in x64 and must be in lower 2G (not 4G). */
static int32_t ptr2addr(const void *p)
{
  lj_assertX((uintptr_t)p < (uintptr_t)0x80000000, "pointer outside 2G range");
  return i32ptr(p);
}
#else
#define ptr2addr(p)	(i32ptr((p)))
#endif

/* op r, [base+ofs] */
static void emit_rmro(ASMState *as, x86Op xo, Reg rr, Reg rb, int32_t ofs)
{
  MCode *p = as->mcp;
  x86Mode mode;
  if (ra_hasreg(rb)) {
    if (LJ_GC64 && rb == RID_RIP) {
      mode = XM_OFS0;
      p -= 4;
      *(int32_t *)p = ofs;
    } else if (ofs == 0 && (rb&7) != RID_EBP) {
      mode = XM_OFS0;
    } else if (checki8(ofs)) {
      *--p = (MCode)ofs;
      mode = XM_OFS8;
    } else {
      p -= 4;
      *(int32_t *)p = ofs;
      mode = XM_OFS32;
    }
    if ((rb&7) == RID_ESP)
      *--p = MODRM(XM_SCALE1, RID_ESP, RID_ESP);
  } else {
    *(int32_t *)(p-4) = ofs;
#if LJ_64
    p[-5] = MODRM(XM_SCALE1, RID_ESP, RID_EBP);
    p -= 5;
    rb = RID_ESP;
#else
    p -= 4;
    rb = RID_EBP;
#endif
    mode = XM_OFS0;
  }
  as->mcp = emit_opm(xo, mode, rr, rb, p, 0);
}

/* op r, [base+idx*scale+ofs] */
static void emit_rmrxo(ASMState *as, x86Op xo, Reg rr, Reg rb, Reg rx,
		       x86Mode scale, int32_t ofs)
{
  MCode *p = as->mcp;
  x86Mode mode;
  if (ofs == 0 && (rb&7) != RID_EBP) {
    mode = XM_OFS0;
  } else if (checki8(ofs)) {
    mode = XM_OFS8;
    *--p = (MCode)ofs;
  } else {
    mode = XM_OFS32;
    p -= 4;
    *(int32_t *)p = ofs;
  }
  as->mcp = emit_opmx(xo, mode, scale, rr, rb, rx, p);
}

/* op r, i */
static void emit_gri(ASMState *as, x86Group xg, Reg rb, int32_t i)
{
  MCode *p = as->mcp;
  x86Op xo;
  if (checki8(i)) {
    *--p = (MCode)i;
    xo = XG_TOXOi8(xg);
  } else {
    p -= 4;
    *(int32_t *)p = i;
    xo = XG_TOXOi(xg);
  }
  as->mcp = emit_opm(xo, XM_REG, (Reg)(xg & 7) | (rb & REX_64), rb, p, 0);
}

/* op [base+ofs], i */
static void emit_gmroi(ASMState *as, x86Group xg, Reg rb, int32_t ofs,
		       int32_t i)
{
  x86Op xo;
  if (checki8(i)) {
    emit_i8(as, i);
    xo = XG_TOXOi8(xg);
  } else {
    emit_i32(as, i);
    xo = XG_TOXOi(xg);
  }
  emit_rmro(as, xo, (Reg)(xg & 7), rb, ofs);
}

#define emit_shifti(as, xg, r, i) \
  (emit_i8(as, (i)), emit_rr(as, XO_SHIFTi, (Reg)(xg), (r)))

/* op r, rm/mrm */
static void emit_mrm(ASMState *as, x86Op xo, Reg rr, Reg rb)
{
  MCode *p = as->mcp;
  x86Mode mode = XM_REG;
  if (rb == RID_MRM) {
    rb = as->mrm.base;
    if (rb == RID_NONE) {
      rb = RID_EBP;
      mode = XM_OFS0;
      p -= 4;
      *(int32_t *)p = as->mrm.ofs;
      if (as->mrm.idx != RID_NONE)
	goto mrmidx;
#if LJ_64
      *--p = MODRM(XM_SCALE1, RID_ESP, RID_EBP);
      rb = RID_ESP;
#endif
    } else if (LJ_GC64 && rb == RID_RIP) {
      lj_assertA(as->mrm.idx == RID_NONE, "RIP-rel mrm cannot have index");
      mode = XM_OFS0;
      p -= 4;
      *(int32_t *)p = as->mrm.ofs;
    } else {
      if (as->mrm.ofs == 0 && (rb&7) != RID_EBP) {
	mode = XM_OFS0;
      } else if (checki8(as->mrm.ofs)) {
	*--p = (MCode)as->mrm.ofs;
	mode = XM_OFS8;
      } else {
	p -= 4;
	*(int32_t *)p = as->mrm.ofs;
	mode = XM_OFS32;
      }
      if (as->mrm.idx != RID_NONE) {
      mrmidx:
	as->mcp = emit_opmx(xo, mode, as->mrm.scale, rr, rb, as->mrm.idx, p);
	return;
      }
      if ((rb&7) == RID_ESP)
	*--p = MODRM(XM_SCALE1, RID_ESP, RID_ESP);
    }
  }
  as->mcp = emit_opm(xo, mode, rr, rb, p, 0);
}

/* op rm/mrm, i */
static void emit_gmrmi(ASMState *as, x86Group xg, Reg rb, int32_t i)
{
  x86Op xo;
  if (checki8(i)) {
    emit_i8(as, i);
    xo = XG_TOXOi8(xg);
  } else {
    emit_i32(as, i);
    xo = XG_TOXOi(xg);
  }
  emit_mrm(as, xo, (Reg)(xg & 7) | (rb & REX_64), (rb & ~REX_64));
}

/* -- Emit loads/stores --------------------------------------------------- */

/* mov [base+ofs], i */
static void emit_movmroi(ASMState *as, Reg base, int32_t ofs, int32_t i)
{
  emit_i32(as, i);
  emit_rmro(as, XO_MOVmi, 0, base, ofs);
}

/* mov [base+ofs], r */
#define emit_movtomro(as, r, base, ofs) \
  emit_rmro(as, XO_MOVto, (r), (base), (ofs))

/* Get/set global_State fields. */
#define emit_opgl(as, xo, r, field) \
  emit_rma(as, (xo), (r), (void *)&J2G(as->J)->field)
#define emit_getgl(as, r, field) emit_opgl(as, XO_MOV, (r)|REX_GC64, field)
#define emit_setgl(as, r, field) emit_opgl(as, XO_MOVto, (r)|REX_GC64, field)

#define emit_setvmstate(as, i) \
  (emit_i32(as, i), emit_opgl(as, XO_MOVmi, 0, vmstate))

/* mov r, i / xor r, r */
static void emit_loadi(ASMState *as, Reg r, int32_t i)
{
  /* XOR r,r is shorter, but modifies the flags. This is bad for HIOP/jcc. */
  if (i == 0 && !(LJ_32 && (IR(as->curins)->o == IR_HIOP ||
			    (as->curins+1 < as->T->nins &&
			     IR(as->curins+1)->o == IR_HIOP))) &&
		!((*as->mcp == 0x0f && (as->mcp[1] & 0xf0) == XI_JCCn) ||
		  (*as->mcp & 0xf0) == XI_JCCs)) {
    emit_rr(as, XO_ARITH(XOg_XOR), r, r);
  } else {
    MCode *p = as->mcp;
    *(int32_t *)(p-4) = i;
    p[-5] = (MCode)(XI_MOVri+(r&7));
    p -= 5;
    REXRB(p, 0, r);
    as->mcp = p;
  }
}

#if LJ_GC64
#define dispofs(as, k) \
  ((intptr_t)((uintptr_t)(k) - (uintptr_t)J2GG(as->J)->dispatch))
#define mcpofs(as, k) \
  ((intptr_t)((uintptr_t)(k) - (uintptr_t)as->mcp))
#define mctopofs(as, k) \
  ((intptr_t)((uintptr_t)(k) - (uintptr_t)as->mctop))
/* mov r, addr */
#define emit_loada(as, r, addr) \
  emit_loadu64(as, (r), (uintptr_t)(addr))
#else
/* mov r, addr */
#define emit_loada(as, r, addr) \
  emit_loadi(as, (r), ptr2addr((addr)))
#endif

#if LJ_64
/* mov r, imm64 or shorter 32 bit extended load. */
static void emit_loadu64(ASMState *as, Reg r, uint64_t u64)
{
  if (checku32(u64)) {  /* 32 bit load clears upper 32 bits. */
    emit_loadi(as, r, (int32_t)u64);
  } else if (checki32((int64_t)u64)) {  /* Sign-extended 32 bit load. */
    MCode *p = as->mcp;
    *(int32_t *)(p-4) = (int32_t)u64;
    as->mcp = emit_opm(XO_MOVmi, XM_REG, REX_64, r, p, -4);
#if LJ_GC64
  } else if (checki32(dispofs(as, u64))) {
    emit_rmro(as, XO_LEA, r|REX_64, RID_DISPATCH, (int32_t)dispofs(as, u64));
  } else if (checki32(mcpofs(as, u64)) && checki32(mctopofs(as, u64))) {
    /* Since as->realign assumes the code size doesn't change, check
    ** RIP-relative addressing reachability for both as->mcp and as->mctop.
    */
    emit_rmro(as, XO_LEA, r|REX_64, RID_RIP, (int32_t)mcpofs(as, u64));
#endif
  } else {  /* Full-size 64 bit load. */
    MCode *p = as->mcp;
    *(uint64_t *)(p-8) = u64;
    p[-9] = (MCode)(XI_MOVri+(r&7));
    p[-10] = 0x48 + ((r>>3)&1);
    p -= 10;
    as->mcp = p;
  }
}
#endif

/* op r, [addr] */
static void emit_rma(ASMState *as, x86Op xo, Reg rr, const void *addr)
{
#if LJ_GC64
  if (checki32(dispofs(as, addr))) {
    emit_rmro(as, xo, rr, RID_DISPATCH, (int32_t)dispofs(as, addr));
  } else if (checki32(mcpofs(as, addr)) && checki32(mctopofs(as, addr))) {
    emit_rmro(as, xo, rr, RID_RIP, (int32_t)mcpofs(as, addr));
  } else if (!checki32((intptr_t)addr)) {
    Reg ra = (rr & 15);
    if (xo != XO_MOV) {
      /* We can't allocate a register here. Use and restore DISPATCH. Ugly. */
      uint64_t dispaddr = (uintptr_t)J2GG(as->J)->dispatch;
      uint8_t i8 = xo == XO_GROUP3b ? *as->mcp++ : 0;
      ra = RID_DISPATCH;
      if (checku32(dispaddr)) {
	emit_loadi(as, ra, (int32_t)dispaddr);
      } else {  /* Full-size 64 bit load. */
	MCode *p = as->mcp;
	*(uint64_t *)(p-8) = dispaddr;
	p[-9] = (MCode)(XI_MOVri+(ra&7));
	p[-10] = 0x48 + ((ra>>3)&1);
	p -= 10;
	as->mcp = p;
      }
      if (xo == XO_GROUP3b) emit_i8(as, i8);
    }
    emit_rmro(as, xo, rr, ra, 0);
    emit_loadu64(as, ra, (uintptr_t)addr);
  } else
#endif
  {
    MCode *p = as->mcp;
    *(int32_t *)(p-4) = ptr2addr(addr);
#if LJ_64
    p[-5] = MODRM(XM_SCALE1, RID_ESP, RID_EBP);
    as->mcp = emit_opm(xo, XM_OFS0, rr, RID_ESP, p, -5);
#else
    as->mcp = emit_opm(xo, XM_OFS0, rr, RID_EBP, p, -4);
#endif
  }
}

/* Load 64 bit IR constant into register. */
static void emit_loadk64(ASMState *as, Reg r, IRIns *ir)
{
  Reg r64;
  x86Op xo;
  const uint64_t *k = &ir_k64(ir)->u64;
  if (rset_test(RSET_FPR, r)) {
    r64 = r;
    xo = XO_MOVSD;
  } else {
    r64 = r | REX_64;
    xo = XO_MOV;
  }
  if (*k == 0) {
    emit_rr(as, rset_test(RSET_FPR, r) ? XO_XORPS : XO_ARITH(XOg_XOR), r, r);
#if LJ_GC64
  } else if (checki32((intptr_t)k) || checki32(dispofs(as, k)) ||
	     (checki32(mcpofs(as, k)) && checki32(mctopofs(as, k)))) {
    emit_rma(as, xo, r64, k);
  } else {
    if (ir->i) {
      lj_assertA(*k == *(uint64_t*)(as->mctop - ir->i),
		 "bad interned 64 bit constant");
    } else if (as->curins <= as->stopins && rset_test(RSET_GPR, r)) {
      emit_loadu64(as, r, *k);
      return;
    } else {
      /* If all else fails, add the FP constant at the MCode area bottom. */
      while ((uintptr_t)as->mcbot & 7) *as->mcbot++ = XI_INT3;
      *(uint64_t *)as->mcbot = *k;
      ir->i = (int32_t)(as->mctop - as->mcbot);
      as->mcbot += 8;
      as->mclim = as->mcbot + MCLIM_REDZONE;
      lj_mcode_commitbot(as->J, as->mcbot);
    }
    emit_rmro(as, xo, r64, RID_RIP, (int32_t)mcpofs(as, as->mctop - ir->i));
#else
  } else {
    emit_rma(as, xo, r64, k);
#endif
  }
}

/* -- Emit control-flow instructions -------------------------------------- */

/* Label for short jumps. */
typedef MCode *MCLabel;

#if LJ_32 && LJ_HASFFI
/* jmp short target */
static void emit_sjmp(ASMState *as, MCLabel target)
{
  MCode *p = as->mcp;
  ptrdiff_t delta = target - p;
  lj_assertA(delta == (int8_t)delta, "short jump target out of range");
  p[-1] = (MCode)(int8_t)delta;
  p[-2] = XI_JMPs;
  as->mcp = p - 2;
}
#endif

/* jcc short target */
static void emit_sjcc(ASMState *as, int cc, MCLabel target)
{
  MCode *p = as->mcp;
  ptrdiff_t delta = target - p;
  lj_assertA(delta == (int8_t)delta, "short jump target out of range");
  p[-1] = (MCode)(int8_t)delta;
  p[-2] = (MCode)(XI_JCCs+(cc&15));
  as->mcp = p - 2;
}

/* jcc short (pending target) */
static MCLabel emit_sjcc_label(ASMState *as, int cc)
{
  MCode *p = as->mcp;
  p[-1] = 0;
  p[-2] = (MCode)(XI_JCCs+(cc&15));
  as->mcp = p - 2;
  return p;
}

/* Fixup jcc short target. */
static void emit_sfixup(ASMState *as, MCLabel source)
{
  source[-1] = (MCode)(as->mcp-source);
}

/* Return label pointing to current PC. */
#define emit_label(as)		((as)->mcp)

/* Compute relative 32 bit offset for jump and call instructions. */
static LJ_AINLINE int32_t jmprel(jit_State *J, MCode *p, MCode *target)
{
  ptrdiff_t delta = target - p;
  UNUSED(J);
  lj_assertJ(delta == (int32_t)delta, "jump target out of range");
  return (int32_t)delta;
}

/* jcc target */
static void emit_jcc(ASMState *as, int cc, MCode *target)
{
  MCode *p = as->mcp;
  *(int32_t *)(p-4) = jmprel(as->J, p, target);
  p[-5] = (MCode)(XI_JCCn+(cc&15));
  p[-6] = 0x0f;
  as->mcp = p - 6;
}

/* jmp target */
static void emit_jmp(ASMState *as, MCode *target)
{
  MCode *p = as->mcp;
  *(int32_t *)(p-4) = jmprel(as->J, p, target);
  p[-5] = XI_JMP;
  as->mcp = p - 5;
}

/* call target */
static void emit_call_(ASMState *as, MCode *target)
{
  MCode *p = as->mcp;
#if LJ_64
  if (target-p != (int32_t)(target-p)) {
    /* Assumes RID_RET is never an argument to calls and always clobbered. */
    emit_rr(as, XO_GROUP5, XOg_CALL, RID_RET);
    emit_loadu64(as, RID_RET, (uint64_t)target);
    return;
  }
#endif
  *(int32_t *)(p-4) = jmprel(as->J, p, target);
  p[-5] = XI_CALL;
  as->mcp = p - 5;
}

#define emit_call(as, f)	emit_call_(as, (MCode *)(void *)(f))

/* -- Emit generic operations --------------------------------------------- */

/* Use 64 bit operations to handle 64 bit IR types. */
#if LJ_64
#define REX_64IR(ir, r)		((r) + (irt_is64((ir)->t) ? REX_64 : 0))
#define VEX_64IR(ir, r)		((r) + (irt_is64((ir)->t) ? VEX_64 : 0))
#else
#define REX_64IR(ir, r)		(r)
#define VEX_64IR(ir, r)		(r)
#endif

/* Generic move between two regs. */
static void emit_movrr(ASMState *as, IRIns *ir, Reg dst, Reg src)
{
  UNUSED(ir);
  if (dst < RID_MAX_GPR)
    emit_rr(as, XO_MOV, REX_64IR(ir, dst), src);
  else
    emit_rr(as, XO_MOVAPS, dst, src);
}

/* Generic load of register with base and (small) offset address. */
static void emit_loadofs(ASMState *as, IRIns *ir, Reg r, Reg base, int32_t ofs)
{
  if (r < RID_MAX_GPR)
    emit_rmro(as, XO_MOV, REX_64IR(ir, r), base, ofs);
  else
    emit_rmro(as, irt_isnum(ir->t) ? XO_MOVSD : XO_MOVSS, r, base, ofs);
}

/* Generic store of register with base and (small) offset address. */
static void emit_storeofs(ASMState *as, IRIns *ir, Reg r, Reg base, int32_t ofs)
{
  if (r < RID_MAX_GPR)
    emit_rmro(as, XO_MOVto, REX_64IR(ir, r), base, ofs);
  else
    emit_rmro(as, irt_isnum(ir->t) ? XO_MOVSDto : XO_MOVSSto, r, base, ofs);
}

/* Add offset to pointer. */
static void emit_addptr(ASMState *as, Reg r, int32_t ofs)
{
  if (ofs) {
    emit_gri(as, XG_ARITHi(XOg_ADD), r|REX_GC64, ofs);
  }
}

#define emit_spsub(as, ofs)	emit_addptr(as, RID_ESP|REX_64, -(ofs))

/* Prefer rematerialization of BASE/L from global_State over spills. */
#define emit_canremat(ref)	((ref) <= REF_BASE)

