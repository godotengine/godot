/*
** RISC-V instruction emitter.
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
**
** Contributed by gns from PLCT Lab, ISCAS.
*/

static intptr_t get_k64val(ASMState *as, IRRef ref)
{
  IRIns *ir = IR(ref);
  if (ir->o == IR_KINT64) {
    return (intptr_t)ir_kint64(ir)->u64;
  } else if (ir->o == IR_KGC) {
    return (intptr_t)ir_kgc(ir);
  } else if (ir->o == IR_KPTR || ir->o == IR_KKPTR) {
    return (intptr_t)ir_kptr(ir);
  } else {
    lj_assertA(ir->o == IR_KINT || ir->o == IR_KNULL,
               "bad 64 bit const IR op %d", ir->o);
    return ir->i;  /* Sign-extended. */
  }
}

#define get_kval(as, ref)       get_k64val(as, ref)

/* -- Emit basic instructions --------------------------------------------- */

static void emit_r(ASMState *as, RISCVIns riscvi, Reg rd, Reg rs1, Reg rs2)
{
  *--as->mcp = riscvi | RISCVF_D(rd) | RISCVF_S1(rs1) | RISCVF_S2(rs2);
}

#define emit_ds(as, riscvi, rd, rs1)         emit_r(as, riscvi, rd, rs1, 0)
#define emit_ds2(as, riscvi, rd, rs2)         emit_r(as, riscvi, rd, 0, rs2)
#define emit_ds1s2(as, riscvi, rd, rs1, rs2)         emit_r(as, riscvi, rd, rs1, rs2)

static void emit_r4(ASMState *as, RISCVIns riscvi, Reg rd, Reg rs1, Reg rs2, Reg rs3)
{
  *--as->mcp = riscvi | RISCVF_D(rd) | RISCVF_S1(rs1) | RISCVF_S2(rs2) | RISCVF_S3(rs3);
}

#define emit_ds1s2s3(as, riscvi, rd, rs1, rs2, rs3)         emit_r4(as, riscvi, rd, rs1, rs2, rs3)

static void emit_i(ASMState *as, RISCVIns riscvi, Reg rd, Reg rs1, int32_t i)
{
  *--as->mcp = riscvi | RISCVF_D(rd) | RISCVF_S1(rs1) | RISCVF_IMMI((uint32_t)i & 0xfff);
}

#define emit_di(as, riscvi, rd, i)         emit_i(as, riscvi, rd, 0, i)
#define emit_dsi(as, riscvi, rd, rs1, i)     emit_i(as, riscvi, rd, rs1, i)
#define emit_dsshamt(as, riscvi, rd, rs1, i) emit_i(as, riscvi, rd, rs1, i&0x3f)

static void emit_s(ASMState *as, RISCVIns riscvi, Reg rs1, Reg rs2, int32_t i)
{
  *--as->mcp = riscvi | RISCVF_S1(rs1) | RISCVF_S2(rs2) | RISCVF_IMMS((uint32_t)i & 0xfff);
}

#define emit_s1s2i(as, riscvi, rs1, rs2, i)  emit_s(as, riscvi, rs1, rs2, i)

/*
static void emit_b(ASMState *as, RISCVIns riscvi, Reg rs1, Reg rs2, int32_t i)
{
  *--as->mcp = riscvi | RISCVF_S1(rs1) | RISCVF_S2(rs2) | RISCVF_IMMB((uint32_t)i & 0x1ffe);
}
*/

static void emit_u(ASMState *as, RISCVIns riscvi, Reg rd, uint32_t i)
{
  *--as->mcp = riscvi | RISCVF_D(rd) | RISCVF_IMMU(i & 0xfffff);
}

#define emit_du(as, riscvi, rd, i)           emit_u(as, riscvi, rd, i)

/*
static void emit_j(ASMState *as, RISCVIns riscvi, Reg rd, int32_t i)
{
  *--as->mcp = riscvi | RISCVF_D(rd) | RISCVF_IMMJ((uint32_t)i & 0x1fffffe);
}
*/

static Reg ra_allock(ASMState *as, intptr_t k, RegSet allow);
static void ra_allockreg(ASMState *as, intptr_t k, Reg r);
static Reg ra_scratch(ASMState *as, RegSet allow);

static void emit_lso(ASMState *as, RISCVIns riscvi, Reg data, Reg base, int32_t ofs)
{
  lj_assertA(checki12(ofs), "load/store offset %d out of range", ofs);
  switch (riscvi) {
    case RISCVI_LD: case RISCVI_LW: case RISCVI_LH: case RISCVI_LB:
    case RISCVI_LWU: case RISCVI_LHU: case RISCVI_LBU:
    case RISCVI_FLW: case RISCVI_FLD:
      emit_dsi(as, riscvi, data, base, ofs);
      break;
    case RISCVI_SD: case RISCVI_SW: case RISCVI_SH: case RISCVI_SB:
    case RISCVI_FSW: case RISCVI_FSD:
      emit_s1s2i(as, riscvi, base, data, ofs);
      break;
    default: lj_assertA(0, "invalid lso"); break;
  }
}

static void emit_roti(ASMState *as, RISCVIns riscvi, Reg rd, Reg rs1, Reg tmp,
                       int32_t shamt)
{
  if (as->flags & JIT_F_RVZbb || as->flags & JIT_F_RVXThead) {
    if (!(as->flags & JIT_F_RVZbb)) switch (riscvi) {
      case RISCVI_RORI: riscvi = RISCVI_TH_SRRI; break;
      case RISCVI_RORIW: riscvi = RISCVI_TH_SRRIW; break;
      default: lj_assertA(0, "invalid roti op"); break;
    }
    emit_dsshamt(as, riscvi, rd, rs1, shamt);
  } else {
    RISCVIns ai, bi;
    int32_t shwid, shmsk;
    switch (riscvi) {
      case RISCVI_RORI:
        ai = RISCVI_SRLI, bi = RISCVI_SLLI;
        shwid = 64, shmsk = 63;
        break;
      case RISCVI_RORIW:
        ai = RISCVI_SRLIW, bi = RISCVI_SLLIW;
        shwid = 32, shmsk = 31;
        break;
      default:
        lj_assertA(0, "invalid roti op");
        return;
    }
    emit_ds1s2(as, RISCVI_OR, rd, rd, tmp);
    emit_dsshamt(as, bi, rd, rs1, (shwid - shamt)&shmsk);
    emit_dsshamt(as, ai, tmp, rs1, shamt&shmsk);
  }
}

static void emit_rot(ASMState *as, RISCVIns riscvi, Reg rd, Reg rs1, Reg rs2, Reg tmp)
{
  if (as->flags & JIT_F_RVZbb) {
    emit_ds1s2(as, riscvi, rd, rs1, rs2);
  } else {
    RISCVIns sai, sbi;
    switch (riscvi) {
      case RISCVI_ROL:
        sai = RISCVI_SLL, sbi = RISCVI_SRL;
        break;
      case RISCVI_ROR:
        sai = RISCVI_SRL, sbi = RISCVI_SLL;
        break;
      case RISCVI_ROLW:
        sai = RISCVI_SLLW, sbi = RISCVI_SRLW;
        break;
      case RISCVI_RORW:
        sai = RISCVI_SRLW, sbi = RISCVI_SLLW;
        break;
      default:
        lj_assertA(0, "invalid rot op");
        return;
    }
    if (rd == rs2) {
      emit_ds1s2(as, RISCVI_OR, rd, rd, tmp);
      emit_ds1s2(as, sbi, tmp, rs1, tmp);
      emit_ds1s2(as, sai, rd, rs1, rs2);
      emit_ds2(as, RISCVI_NEG, tmp, rs2);
    } else {
      emit_ds1s2(as, RISCVI_OR, rd, rd, tmp);
      emit_ds1s2(as, sai, rd, rs1, rs2);
      emit_ds1s2(as, sbi, tmp, rs1, tmp);
      emit_ds2(as, RISCVI_NEG, tmp, rs2);
    }
  }
}

static void emit_ext(ASMState *as, RISCVIns riscvi, Reg rd, Reg rs1)
{
  if ((riscvi != RISCVI_ZEXT_W && as->flags & JIT_F_RVZbb) ||
      (riscvi == RISCVI_ZEXT_W && as->flags & JIT_F_RVZba)) {
    emit_ds(as, riscvi, rd, rs1);
  } else if (as->flags & JIT_F_RVXThead) {
    uint32_t hi, sext;
    switch (riscvi) {
      case RISCVI_ZEXT_B:
      case RISCVI_SEXT_W:
        emit_ds(as, riscvi, rd, rs1);
        return;
      case RISCVI_ZEXT_H:
        hi = 15, sext = 0;
        break;
      case RISCVI_ZEXT_W:
        hi = 31, sext = 0;
        break;
      case RISCVI_SEXT_B:
        hi = 7, sext = 1;
        break;
      case RISCVI_SEXT_H:
        hi = 15, sext = 1;
        break;
      default:
        lj_assertA(0, "invalid ext op");
        return;
    }
    emit_dsi(as, sext ? RISCVI_TH_EXT : RISCVI_TH_EXTU,
      rd, rs1, hi << 6);
  } else {
    RISCVIns sli, sri;
    int32_t shamt;
    switch (riscvi) {
      case RISCVI_ZEXT_B:
      case RISCVI_SEXT_W:
        emit_ds(as, riscvi, rd, rs1);
        return;
      case RISCVI_ZEXT_H:
        sli = RISCVI_SLLI, sri = RISCVI_SRLI;
        shamt = 48;
        break;
      case RISCVI_ZEXT_W:
        sli = RISCVI_SLLI, sri = RISCVI_SRLI;
        shamt = 32;
        break;
      case RISCVI_SEXT_B:
        sli = RISCVI_SLLI, sri = RISCVI_SRAI;
        shamt = 56;
        break;
      case RISCVI_SEXT_H:
        sli = RISCVI_SLLI, sri = RISCVI_SRAI;
        shamt = 48;
        break;
      default:
        lj_assertA(0, "invalid ext op");
        return;
    }
    emit_dsshamt(as, sri, rd, rd, shamt);   
    emit_dsshamt(as, sli, rd, rs1, shamt);
  }
}

static void emit_cleartp(ASMState *as, Reg rd, Reg rs1)
{
  if (as->flags & JIT_F_RVXThead) {
    emit_dsi(as, RISCVI_TH_EXTU, rd, rs1, 46u << 6);
  } else {
    emit_dsshamt(as, RISCVI_SRLI, rd, rd, 17);
    emit_dsshamt(as, RISCVI_SLLI, rd, rs1, 17);
  }
}

/*
static void emit_andn(ASMState *as, Reg rd, Reg rs1, Reg rs2, Reg tmp)
{
  if (as->flags & JIT_F_RVZbb) {
    emit_ds1s2(as, RISCVI_ANDN, rd, rs1, rs2);
  } else {
    emit_ds1s2(as, RISCVI_AND, rd, rs1, tmp);
    emit_ds(as, RISCVI_NOT, tmp, rs2);
  }
}
*/

/*
static void emit_orn(ASMState *as, Reg rd, Reg rs1, Reg rs2, Reg tmp)
{
  if (as->flags & JIT_F_RVZbb) {
    emit_ds1s2(as, RISCVI_ORN, rd, rs1, rs2);
  } else {
    emit_ds1s2(as, RISCVI_OR, rd, rs1, tmp);
    emit_ds(as, RISCVI_NOT, tmp, rs2);
  }
}
*/

static void emit_xnor(ASMState *as, Reg rd, Reg rs1, Reg rs2)
{
  if (as->flags & JIT_F_RVZbb) {
    emit_ds1s2(as, RISCVI_XNOR, rd, rs1, rs2);
  } else {
    emit_ds(as, RISCVI_NOT, rd, rd);
    emit_ds1s2(as, RISCVI_XOR, rd, rs1, rs2);
  }
}

static void emit_shxadd(ASMState *as, Reg rd, Reg rs1, Reg rs2, Reg tmp, unsigned int shamt)
{
  if (as->flags & JIT_F_RVZba) {
    switch (shamt) {
      case 1: emit_ds1s2(as, RISCVI_SH1ADD, rd, rs2, rs1); break;
      case 2: emit_ds1s2(as, RISCVI_SH2ADD, rd, rs2, rs1); break;
      case 3: emit_ds1s2(as, RISCVI_SH3ADD, rd, rs2, rs1); break;
      default: return;
    }
  } else if (as->flags & JIT_F_RVXThead) {
    emit_dsi(as, RISCVI_TH_ADDSL|RISCVF_IMMI(shamt<<5), rd, rs1, rs2);
  } else {
    emit_ds1s2(as, RISCVI_ADD, rd, rs1, tmp);
    emit_dsshamt(as, RISCVI_SLLI, tmp, rs2, shamt);
  }
}

#define emit_sh1add(as, rd, rs1, rs2, tmp) emit_shxadd(as, rd, rs1, rs2, tmp, 1)
#define emit_sh2add(as, rd, rs1, rs2, tmp) emit_shxadd(as, rd, rs1, rs2, tmp, 2)
#define emit_sh3add(as, rd, rs1, rs2, tmp) emit_shxadd(as, rd, rs1, rs2, tmp, 3)

static void emit_loadk12(ASMState *as, Reg rd, int32_t i)
{
  emit_di(as, RISCVI_ADDI, rd, i);
}

static void emit_loadk32(ASMState *as, Reg rd, int32_t i)
{
  if (checki12((int64_t)i)) {
    emit_loadk12(as, rd, i);
  } else {
    if(LJ_UNLIKELY(RISCVF_HI((uint32_t)i) == 0x80000u && i > 0))
      emit_dsi(as, RISCVI_XORI, rd, rd, RISCVF_LO(i));
    else
    emit_dsi(as, RISCVI_ADDI, rd, rd, RISCVF_LO(i));
    emit_du(as, RISCVI_LUI, rd, RISCVF_HI((uint32_t)i));
  }
}

/* -- Emit loads/stores --------------------------------------------------- */

/* Prefer rematerialization of BASE/L from global_State over spills. */
#define emit_canremat(ref)	((ref) <= REF_BASE)


/* Load a 32 bit constant into a GPR. */
#define emit_loadi(as, r, i)	emit_loadk32(as, r, i);

/* Load a 64 bit constant into a GPR. */
static void emit_loadu64(ASMState *as, Reg r, uint64_t u64)
{
  int64_t u64_delta = (int64_t)((intptr_t)u64 - (intptr_t)(as->mcp - 2));
  if (checki32((int64_t)u64)) {
    emit_loadk32(as, r, (int32_t)u64);
  } else if (checki32auipc(u64_delta)) {
    emit_dsi(as, RISCVI_ADDI, r, r, RISCVF_LO(u64_delta));
    emit_du(as, RISCVI_AUIPC, r, RISCVF_HI(u64_delta));
  } else {
    uint32_t lo32 = u64 & 0xfffffffful;
    if (checku11(lo32)) {
      if (lo32 > 0) emit_dsi(as, RISCVI_ADDI, r, r, lo32);
      emit_dsshamt(as, RISCVI_SLLI, r, r, 32);
    } else {
      RISCVIns li_insn[7] = {0};
      int shamt = 0, step = 0;
      for(int bit = 0; bit < 32; bit++) {
  if (lo32 & (1u << bit)) {
    if (shamt) li_insn[step++] = RISCVI_SLLI | RISCVF_D(r) | RISCVF_S1(r) | RISCVF_IMMI(shamt);
    int inc = bit+10 > 31 ? 31-bit : 10;
    bit += inc, shamt = inc+1;
    uint32_t msk = ((1ul << (bit+1))-1)^((1ul << (((bit-inc) >= 0) ? (bit-inc) : 0))-1);
    uint16_t payload = (lo32 & msk) >> (((bit-inc) >= 0) ? (bit-inc) : 0);
    li_insn[step++] = RISCVI_ADDI | RISCVF_D(r) | RISCVF_S1(r) | RISCVF_IMMI(payload);
  } else shamt++;
      }
      if (shamt) li_insn[step++] = RISCVI_SLLI | RISCVF_D(r) | RISCVF_S1(r) | RISCVF_IMMI(shamt);

      if (step < 6) {
  for(int i = 0; i < step; i++)
    *--as->mcp = li_insn[i];
      } else {
  emit_dsi(as, RISCVI_ADDI, r, r, u64 & 0x3ff);
  emit_dsshamt(as, RISCVI_SLLI, r, r, 10);
  emit_dsi(as, RISCVI_ADDI, r, r, (u64 >> 10) & 0x7ff);
  emit_dsshamt(as, RISCVI_SLLI, r, r, 11);
  emit_dsi(as, RISCVI_ADDI, r, r, (u64 >> 21) & 0x7ff);
  emit_dsshamt(as, RISCVI_SLLI, r, r, 11);
      }
    }

    uint32_t hi32 = u64 >> 32;
    if (hi32 & 0xfff) emit_loadk32(as, r, hi32);
    else emit_du(as, RISCVI_LUI, r, hi32 >> 12);
  }
}

#define emit_loada(as, r, addr)	emit_loadu64(as, (r), u64ptr((addr)))

/* Get/set from constant pointer. */
static void emit_lsptr(ASMState *as, RISCVIns riscvi, Reg r, void *p, RegSet allow)
{
  emit_lso(as, riscvi, r, ra_allock(as, igcptr(p), allow), 0);
}

/* Load 64 bit IR constant into register. */
static void emit_loadk64(ASMState *as, Reg r, IRIns *ir)
{
  const uint64_t *k = &ir_k64(ir)->u64;
  Reg r64 = r;
  if (rset_test(RSET_FPR, r)) {
    r64 = RID_TMP;
    emit_ds(as, RISCVI_FMV_D_X, r, r64);
  }
  emit_loadu64(as, r64, *k);
}

/* Get/set global_State fields. */
static void emit_lsglptr(ASMState *as, RISCVIns riscvi, Reg r, int32_t ofs)
{
  emit_lso(as, riscvi, r, RID_GL, ofs);
}

#define emit_getgl(as, r, field) \
  emit_lsglptr(as, RISCVI_LD, (r), (int32_t)offsetof(global_State, field))
#define emit_setgl(as, r, field) \
  emit_lsglptr(as, RISCVI_SD, (r), (int32_t)offsetof(global_State, field))

/* Trace number is determined from per-trace exit stubs. */
#define emit_setvmstate(as, i)		UNUSED(i)

/* -- Emit control-flow instructions -------------------------------------- */

/* Label for internal jumps. */
typedef MCode *MCLabel;

/* Return label pointing to current PC. */
#define emit_label(as)		((as)->mcp)

static void emit_branch(ASMState *as, RISCVIns riscvi, Reg rs1, Reg rs2, MCode *target, int jump)
{
  MCode *p = as->mcp;
  ptrdiff_t delta = (char *)target - (char *)(p - 1);
  // lj_assertA(((delta + 0x10000) >> 13) == 0, "branch target out of range"); /* B */
  lj_assertA(((delta + 0x100000) >> 21) == 0, "branch target out of range"); /* ^B+J */
  if (checki13(delta) && !jump) {
    *--p = riscvi | RISCVF_S1(rs1) | RISCVF_S2(rs2) | RISCVF_IMMB(delta);
    *--p = RISCVI_NOP;
  } else {
    *--p = RISCVI_JAL | RISCVF_IMMJ(delta); /* Poorman's trampoline */
    *--p = (riscvi^0x00001000) | RISCVF_S1(rs1) | RISCVF_S2(rs2) | RISCVF_IMMB(8);
  }
  as->mcp = p;
}

static void emit_jmp(ASMState *as, MCode *target)
{
  MCode *p = as->mcp;
  ptrdiff_t delta = (char *)target - (char *)(p - 2);
  // lj_assertA(((delta + 0x100000) >> 21) == 0, "jump target out of range"); /* J */
  lj_assertA(checki32(delta), "jump target out of range"); /* AUIPC+JALR */
  if (checki21(delta)) {
    *--p = RISCVI_NOP;
    *--p = RISCVI_JAL | RISCVF_IMMJ(delta);
  } else {
    *--p = RISCVI_JALR | RISCVF_S1(RID_TMP) | RISCVF_IMMI(RISCVF_LO(delta));
    *--p = RISCVI_AUIPC | RISCVF_D(RID_TMP) | RISCVF_IMMU(RISCVF_HI(delta));
  }
  as->mcp = p;
}

#define emit_mv(as, dst, src) \
  emit_ds(as, RISCVI_MV, (dst), (src))

static void emit_call(ASMState *as, void *target, int needcfa)
{
  MCode *p = as->mcp;
  ptrdiff_t delta = (char *)target - (char *)(p - 2);
  if (checki21(delta)) {
    *--p = RISCVI_NOP;
    *--p = RISCVI_JAL | RISCVF_D(RID_RA) | RISCVF_IMMJ(delta);
  } else if (checki32(delta)) {
    *--p = RISCVI_JALR | RISCVF_D(RID_RA) | RISCVF_S1(RID_TMP) | RISCVF_IMMI(RISCVF_LO(delta));
    *--p = RISCVI_AUIPC | RISCVF_D(RID_TMP) | RISCVF_IMMU(RISCVF_HI(delta));
    needcfa = 1;
  } else {
    *--p = RISCVI_JALR | RISCVF_D(RID_RA) | RISCVF_S1(RID_CFUNCADDR) | RISCVF_IMMI(0);
    needcfa = 2;
  }
  as->mcp = p;
  if (needcfa > 1)
    ra_allockreg(as, (intptr_t)target, RID_CFUNCADDR);
}

/* -- Emit generic operations --------------------------------------------- */

/* Generic move between two regs. */
static void emit_movrr(ASMState *as, IRIns *ir, Reg dst, Reg src)
{
  if (src < RID_MAX_GPR && dst < RID_MAX_GPR)
    emit_mv(as, dst, src);
  else if (src < RID_MAX_GPR)
    emit_ds(as, irt_isnum(ir->t) ? RISCVI_FMV_D_X : RISCVI_FMV_W_X, dst, src);
  else if (dst < RID_MAX_GPR)
    emit_ds(as, irt_isnum(ir->t) ? RISCVI_FMV_X_D : RISCVI_FMV_X_W, dst, src);
  else
    emit_ds1s2(as, irt_isnum(ir->t) ? RISCVI_FMV_D : RISCVI_FMV_S, dst, src, src);
}

/* Emit an arithmetic operation with a constant operand. */
static void emit_opk(ASMState *as, RISCVIns riscvi, Reg dest, Reg src,
         Reg tmp, intptr_t k)
{
  if (checki12(k)) emit_dsi(as, riscvi, dest, src, k);
  else {
    switch (riscvi) {
      case RISCVI_ADDI: riscvi = RISCVI_ADD; break;
      case RISCVI_XORI: riscvi = RISCVI_XOR; break;
      case RISCVI_ORI: riscvi = RISCVI_OR; break;
      case RISCVI_ANDI: riscvi = RISCVI_AND; break;
      default: lj_assertA(0, "NYI arithmetic RISCVIns"); return;
    }
    emit_ds1s2(as, riscvi, dest, src, tmp);
    emit_loadu64(as, tmp, (uintptr_t)k);
  }
}

/* Generic load of register with base and (small) offset address. */
static void emit_loadofs(ASMState *as, IRIns *ir, Reg r, Reg base, int32_t ofs)
{
  if (r < RID_MAX_GPR)
    emit_lso(as, irt_is64(ir->t) ? RISCVI_LD : RISCVI_LW, r, base, ofs);
  else
    emit_lso(as, irt_isnum(ir->t) ? RISCVI_FLD : RISCVI_FLW, r, base, ofs);
}

/* Generic store of register with base and (small) offset address. */
static void emit_storeofs(ASMState *as, IRIns *ir, Reg r, Reg base, int32_t ofs)
{
  if (r < RID_MAX_GPR)
    emit_lso(as, irt_is64(ir->t) ? RISCVI_SD : RISCVI_SW, r, base, ofs);
  else
    emit_lso(as, irt_isnum(ir->t) ? RISCVI_FSD : RISCVI_FSW, r, base, ofs);
}

/* Add offset to pointer. */
static void emit_addptr(ASMState *as, Reg r, int32_t ofs)
{
  if (ofs)
    emit_opk(as, RISCVI_ADDI, r, r, RID_TMP, ofs);
}


#define emit_spsub(as, ofs)	emit_addptr(as, RID_SP, -(ofs))
