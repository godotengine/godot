/*
** LuaJIT VM builder: PE object emitter.
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
**
** Only used for building on Windows, since we cannot assume the presence
** of a suitable assembler. The host and target byte order must match.
*/

#include "buildvm.h"
#include "lj_bc.h"

#if LJ_TARGET_WINDOWS || LJ_TARGET_CYGWIN

/* Context for PE object emitter. */
static char *strtab;
static size_t strtabofs;

/* -- PE object definitions ----------------------------------------------- */

/* PE header. */
typedef struct PEheader {
  uint16_t arch;
  uint16_t nsects;
  uint32_t time;
  uint32_t symtabofs;
  uint32_t nsyms;
  uint16_t opthdrsz;
  uint16_t flags;
} PEheader;

/* PE section. */
typedef struct PEsection {
  char name[8];
  uint32_t vsize;
  uint32_t vaddr;
  uint32_t size;
  uint32_t ofs;
  uint32_t relocofs;
  uint32_t lineofs;
  uint16_t nreloc;
  uint16_t nline;
  uint32_t flags;
} PEsection;

/* PE relocation. */
typedef struct PEreloc {
  uint32_t vaddr;
  uint32_t symidx;
  uint16_t type;
} PEreloc;

/* Cannot use sizeof, because it pads up to the max. alignment. */
#define PEOBJ_RELOC_SIZE	(4+4+2)

/* PE symbol table entry. */
typedef struct PEsym {
  union {
    char name[8];
    uint32_t nameref[2];
  } n;
  uint32_t value;
  int16_t sect;
  uint16_t type;
  uint8_t scl;
  uint8_t naux;
} PEsym;

/* PE symbol table auxiliary entry for a section. */
typedef struct PEsymaux {
  uint32_t size;
  uint16_t nreloc;
  uint16_t nline;
  uint32_t cksum;
  uint16_t assoc;
  uint8_t comdatsel;
  uint8_t unused[3];
} PEsymaux;

/* Cannot use sizeof, because it pads up to the max. alignment. */
#define PEOBJ_SYM_SIZE	(8+4+2+2+1+1)

/* PE object CPU specific defines. */
#if LJ_TARGET_X86
#define PEOBJ_ARCH_TARGET	0x014c
#define PEOBJ_RELOC_REL32	0x14  /* MS: REL32, GNU: DISP32. */
#define PEOBJ_RELOC_DIR32	0x06
#define PEOBJ_RELOC_OFS		0
#define PEOBJ_TEXT_FLAGS	0x60500020  /* 60=r+x, 50=align16, 20=code. */
#elif LJ_TARGET_X64
#define PEOBJ_ARCH_TARGET	0x8664
#define PEOBJ_RELOC_REL32	0x04  /* MS: REL32, GNU: DISP32. */
#define PEOBJ_RELOC_DIR32	0x02
#define PEOBJ_RELOC_ADDR32NB	0x03
#define PEOBJ_RELOC_OFS		0
#define PEOBJ_TEXT_FLAGS	0x60500020  /* 60=r+x, 50=align16, 20=code. */
#define PEOBJ_PDATA_NRELOC	6
#define PEOBJ_XDATA_SIZE	(8*2+4+6*2)
#elif LJ_TARGET_ARM64
#define PEOBJ_ARCH_TARGET	0xaa64
#define PEOBJ_RELOC_REL32	0x03  /* MS: BRANCH26. */
#define PEOBJ_RELOC_DIR32	0x01
#define PEOBJ_RELOC_ADDR32NB	0x02
#define PEOBJ_RELOC_OFS		(-4)
#define PEOBJ_TEXT_FLAGS	0x60500020  /* 60=r+x, 50=align16, 20=code. */
#define PEOBJ_PDATA_NRELOC	4
#define PEOBJ_XDATA_SIZE	(4+24+4 +4+8)
#endif

/* Section numbers (0-based). */
enum {
  PEOBJ_SECT_ABS = -2,
  PEOBJ_SECT_UNDEF = -1,
  PEOBJ_SECT_TEXT,
#ifdef PEOBJ_PDATA_NRELOC
  PEOBJ_SECT_PDATA,
  PEOBJ_SECT_XDATA,
#elif LJ_TARGET_X86
  PEOBJ_SECT_SXDATA,
#endif
  PEOBJ_SECT_RDATA_Z,
  PEOBJ_NSECTIONS
};

/* Symbol types. */
#define PEOBJ_TYPE_NULL		0
#define PEOBJ_TYPE_FUNC		0x20

/* Symbol storage class. */
#define PEOBJ_SCL_EXTERN	2
#define PEOBJ_SCL_STATIC	3

/* -- PE object emitter --------------------------------------------------- */

/* Emit PE object symbol. */
static void emit_peobj_sym(BuildCtx *ctx, const char *name, uint32_t value,
			   int sect, int type, int scl)
{
  PEsym sym;
  size_t len = strlen(name);
  if (!strtab) {  /* Pass 1: only calculate string table length. */
    if (len > 8) strtabofs += len+1;
    return;
  }
  if (len <= 8) {
    memcpy(sym.n.name, name, len);
    memset(sym.n.name+len, 0, 8-len);
  } else {
    sym.n.nameref[0] = 0;
    sym.n.nameref[1] = (uint32_t)strtabofs;
    memcpy(strtab + strtabofs, name, len);
    strtab[strtabofs+len] = 0;
    strtabofs += len+1;
  }
  sym.value = value;
  sym.sect = (int16_t)(sect+1);  /* 1-based section number. */
  sym.type = (uint16_t)type;
  sym.scl = (uint8_t)scl;
  sym.naux = 0;
  owrite(ctx, &sym, PEOBJ_SYM_SIZE);
}

/* Emit PE object section symbol. */
static void emit_peobj_sym_sect(BuildCtx *ctx, PEsection *pesect, int sect)
{
  PEsym sym;
  PEsymaux aux;
  if (!strtab) return;  /* Pass 1: no output. */
  memcpy(sym.n.name, pesect[sect].name, 8);
  sym.value = 0;
  sym.sect = (int16_t)(sect+1);  /* 1-based section number. */
  sym.type = PEOBJ_TYPE_NULL;
  sym.scl = PEOBJ_SCL_STATIC;
  sym.naux = 1;
  owrite(ctx, &sym, PEOBJ_SYM_SIZE);
  memset(&aux, 0, sizeof(PEsymaux));
  aux.size = pesect[sect].size;
  aux.nreloc = pesect[sect].nreloc;
  owrite(ctx, &aux, PEOBJ_SYM_SIZE);
}

/* Emit Windows PE object file. */
void emit_peobj(BuildCtx *ctx)
{
  PEheader pehdr;
  PEsection pesect[PEOBJ_NSECTIONS];
  uint32_t sofs;
  int i, nrsym;
  union { uint8_t b; uint32_t u; } host_endian;
#ifdef PEOBJ_PDATA_NRELOC
  uint32_t fcofs = (uint32_t)ctx->sym[ctx->nsym-1].ofs;
#endif

  sofs = sizeof(PEheader) + PEOBJ_NSECTIONS*sizeof(PEsection);

  /* Fill in PE sections. */
  memset(&pesect, 0, PEOBJ_NSECTIONS*sizeof(PEsection));
  memcpy(pesect[PEOBJ_SECT_TEXT].name, ".text", sizeof(".text")-1);
  pesect[PEOBJ_SECT_TEXT].ofs = sofs;
  sofs += (pesect[PEOBJ_SECT_TEXT].size = (uint32_t)ctx->codesz);
  pesect[PEOBJ_SECT_TEXT].relocofs = sofs;
  sofs += (pesect[PEOBJ_SECT_TEXT].nreloc = (uint16_t)ctx->nreloc) * PEOBJ_RELOC_SIZE;
  /* Flags: 60 = read+execute, 50 = align16, 20 = code. */
  pesect[PEOBJ_SECT_TEXT].flags = PEOBJ_TEXT_FLAGS;

#ifdef PEOBJ_PDATA_NRELOC
  memcpy(pesect[PEOBJ_SECT_PDATA].name, ".pdata", sizeof(".pdata")-1);
  pesect[PEOBJ_SECT_PDATA].ofs = sofs;
  sofs += (pesect[PEOBJ_SECT_PDATA].size = PEOBJ_PDATA_NRELOC*4);
  pesect[PEOBJ_SECT_PDATA].relocofs = sofs;
  sofs += (pesect[PEOBJ_SECT_PDATA].nreloc = PEOBJ_PDATA_NRELOC) * PEOBJ_RELOC_SIZE;
  /* Flags: 40 = read, 30 = align4, 40 = initialized data. */
  pesect[PEOBJ_SECT_PDATA].flags = 0x40300040;

  memcpy(pesect[PEOBJ_SECT_XDATA].name, ".xdata", sizeof(".xdata")-1);
  pesect[PEOBJ_SECT_XDATA].ofs = sofs;
  sofs += (pesect[PEOBJ_SECT_XDATA].size = PEOBJ_XDATA_SIZE);  /* See below. */
  pesect[PEOBJ_SECT_XDATA].relocofs = sofs;
  sofs += (pesect[PEOBJ_SECT_XDATA].nreloc = 1) * PEOBJ_RELOC_SIZE;
  /* Flags: 40 = read, 30 = align4, 40 = initialized data. */
  pesect[PEOBJ_SECT_XDATA].flags = 0x40300040;
#elif LJ_TARGET_X86
  memcpy(pesect[PEOBJ_SECT_SXDATA].name, ".sxdata", sizeof(".sxdata")-1);
  pesect[PEOBJ_SECT_SXDATA].ofs = sofs;
  sofs += (pesect[PEOBJ_SECT_SXDATA].size = 4);
  pesect[PEOBJ_SECT_SXDATA].relocofs = sofs;
  /* Flags: 40 = read, 30 = align4, 02 = lnk_info, 40 = initialized data. */
  pesect[PEOBJ_SECT_SXDATA].flags = 0x40300240;
#endif

  memcpy(pesect[PEOBJ_SECT_RDATA_Z].name, ".rdata$Z", sizeof(".rdata$Z")-1);
  pesect[PEOBJ_SECT_RDATA_Z].ofs = sofs;
  sofs += (pesect[PEOBJ_SECT_RDATA_Z].size = (uint32_t)strlen(ctx->dasm_ident)+1);
  /* Flags: 40 = read, 30 = align4, 40 = initialized data. */
  pesect[PEOBJ_SECT_RDATA_Z].flags = 0x40300040;

  /* Fill in PE header. */
  pehdr.arch = PEOBJ_ARCH_TARGET;
  pehdr.nsects = PEOBJ_NSECTIONS;
  pehdr.time = 0;  /* Timestamp is optional. */
  pehdr.symtabofs = sofs;
  pehdr.opthdrsz = 0;
  pehdr.flags = 0;

  /* Compute the size of the symbol table:
  ** @feat.00 + nsections*2
  ** + asm_start + nsym
  ** + nrsym
  */
  nrsym = ctx->nrelocsym;
  pehdr.nsyms = 1+PEOBJ_NSECTIONS*2 + 1+ctx->nsym + nrsym;
#ifdef PEOBJ_PDATA_NRELOC
  pehdr.nsyms += 1;  /* Symbol for lj_err_unwind_win. */
#endif

  /* Write PE object header and all sections. */
  owrite(ctx, &pehdr, sizeof(PEheader));
  owrite(ctx, &pesect, sizeof(PEsection)*PEOBJ_NSECTIONS);

  /* Write .text section. */
  host_endian.u = 1;
  if (host_endian.b != LJ_ENDIAN_SELECT(1, 0)) {
    fprintf(stderr, "Error: different byte order for host and target\n");
    exit(1);
  }
  owrite(ctx, ctx->code, ctx->codesz);
  for (i = 0; i < ctx->nreloc; i++) {
    PEreloc reloc;
    reloc.vaddr = (uint32_t)ctx->reloc[i].ofs + PEOBJ_RELOC_OFS;
    reloc.symidx = 1+2+ctx->reloc[i].sym;  /* Reloc syms are after .text sym. */
    reloc.type = ctx->reloc[i].type ? PEOBJ_RELOC_REL32 : PEOBJ_RELOC_DIR32;
    owrite(ctx, &reloc, PEOBJ_RELOC_SIZE);
  }

#if LJ_TARGET_X64
  { /* Write .pdata section. */
    uint32_t pdata[3];  /* Start of .text, end of .text and .xdata. */
    PEreloc reloc;
    pdata[0] = 0; pdata[1] = fcofs; pdata[2] = 0;
    owrite(ctx, &pdata, sizeof(pdata));
    pdata[0] = fcofs; pdata[1] = (uint32_t)ctx->codesz; pdata[2] = 20;
    owrite(ctx, &pdata, sizeof(pdata));
    reloc.vaddr = 0; reloc.symidx = 1+2+nrsym+2+2+1;
    reloc.type = PEOBJ_RELOC_ADDR32NB;
    owrite(ctx, &reloc, PEOBJ_RELOC_SIZE);
    reloc.vaddr = 4; reloc.symidx = 1+2+nrsym+2+2+1;
    reloc.type = PEOBJ_RELOC_ADDR32NB;
    owrite(ctx, &reloc, PEOBJ_RELOC_SIZE);
    reloc.vaddr = 8; reloc.symidx = 1+2+nrsym+2;
    reloc.type = PEOBJ_RELOC_ADDR32NB;
    owrite(ctx, &reloc, PEOBJ_RELOC_SIZE);
    reloc.vaddr = 12; reloc.symidx = 1+2+nrsym+2+2+1;
    reloc.type = PEOBJ_RELOC_ADDR32NB;
    owrite(ctx, &reloc, PEOBJ_RELOC_SIZE);
    reloc.vaddr = 16; reloc.symidx = 1+2+nrsym+2+2+1;
    reloc.type = PEOBJ_RELOC_ADDR32NB;
    owrite(ctx, &reloc, PEOBJ_RELOC_SIZE);
    reloc.vaddr = 20; reloc.symidx = 1+2+nrsym+2;
    reloc.type = PEOBJ_RELOC_ADDR32NB;
    owrite(ctx, &reloc, PEOBJ_RELOC_SIZE);
  }
  { /* Write .xdata section. */
    uint16_t xdata[8+2+6];
    PEreloc reloc;
    xdata[0] = 0x01|0x08|0x10;  /* Ver. 1, uhandler/ehandler, prolog size 0. */
    xdata[1] = 0x0005;  /* Number of unwind codes, no frame pointer. */
    xdata[2] = 0x4200;  /* Stack offset 4*8+8 = aword*5. */
    xdata[3] = 0x3000;  /* Push rbx. */
    xdata[4] = 0x6000;  /* Push rsi. */
    xdata[5] = 0x7000;  /* Push rdi. */
    xdata[6] = 0x5000;  /* Push rbp. */
    xdata[7] = 0;  /* Alignment. */
    xdata[8] = xdata[9] = 0;  /* Relocated address of exception handler. */
    xdata[10] = 0x01;  /* Ver. 1, no handler, prolog size 0. */
    xdata[11] = 0x1504;  /* Number of unwind codes, fp = rbp, fpofs = 16. */
    xdata[12] = 0x0300;  /* set_fpreg. */
    xdata[13] = 0x0200;  /* stack offset 0*8+8 = aword*1. */
    xdata[14] = 0x3000;  /* Push rbx. */
    xdata[15] = 0x5000;  /* Push rbp. */
    owrite(ctx, &xdata, sizeof(xdata));
    reloc.vaddr = 2*8; reloc.symidx = 1+2+nrsym+2+2;
    reloc.type = PEOBJ_RELOC_ADDR32NB;
    owrite(ctx, &reloc, PEOBJ_RELOC_SIZE);
  }
#elif LJ_TARGET_ARM64
  /* https://learn.microsoft.com/en-us/cpp/build/arm64-exception-handling */
  { /* Write .pdata section. */
    uint32_t pdata[4];
    PEreloc reloc;
    pdata[0] = 0;
    pdata[1] = 0;
    pdata[2] = fcofs;
    pdata[3] = 4+24+4;
    owrite(ctx, &pdata, sizeof(pdata));
    /* Start of .text and start of .xdata. */
    reloc.vaddr = 0; reloc.symidx = 1+2+nrsym+2+2+1;
    reloc.type = PEOBJ_RELOC_ADDR32NB;
    owrite(ctx, &reloc, PEOBJ_RELOC_SIZE);
    reloc.vaddr = 4; reloc.symidx = 1+2+nrsym+2;
    reloc.type = PEOBJ_RELOC_ADDR32NB;
    owrite(ctx, &reloc, PEOBJ_RELOC_SIZE);
    /* Start of vm_ffi_call and start of second part of .xdata. */
    reloc.vaddr = 8; reloc.symidx = 1+2+nrsym+2+2+1;
    reloc.type = PEOBJ_RELOC_ADDR32NB;
    owrite(ctx, &reloc, PEOBJ_RELOC_SIZE);
    reloc.vaddr = 12; reloc.symidx = 1+2+nrsym+2;
    reloc.type = PEOBJ_RELOC_ADDR32NB;
    owrite(ctx, &reloc, PEOBJ_RELOC_SIZE);
  }
  { /* Write .xdata section. */
    uint32_t u32;
    uint8_t *p, uwc[24];
    PEreloc reloc;

#define CBE16(x)	(*p = ((x) >> 8) & 0xff, p[1] = (x) & 0xff, p += 2)
#define CALLOC_S(s)	(*p++ = ((s) >> 4))  /* s < 512 */
#define CSAVE_FPLR(o)	(*p++ = 0x40 | ((o) >> 3))  /* o <= 504 */
#define CSAVE_REGP(r,o)	CBE16(0xc800 | (((r) - 19) << 6) | ((o) >> 3))
#define CSAVE_REGS(r1,r2,o1) do { \
  int r, o; for (r = r1, o = o1; r <= r2; r += 2, o -= 16) CSAVE_REGP(r, o); \
} while (0)
#define CSAVE_REGPX(r,o) CBE16(0xcc00 | (((r) - 19) << 6) | (~(o) >> 3))
#define CSAVE_FREGP(r,o) CBE16(0xd800 | (((r) - 8) << 6) | ((o) >> 3))
#define CSAVE_FREGS(r1,r2,o1) do { \
  int r, o; for (r = r1, o = o1; r <= r2; r += 2, o -= 16) CSAVE_FREGP(r, o); \
} while (0)
#define CADD_FP(s)	CBE16(0xe200 | ((s) >> 3))  /* s < 8*256 */
#define CODE_NOP	0xe3
#define CODE_END	0xe4
#define CEND_ALIGN	do { \
  *p++ = CODE_END; \
  while ((p - uwc) & 3) *p++ = CODE_NOP; \
} while (0)

    /* Unwind codes for .text section with handler. */
    p = uwc;
    CADD_FP(192);		/* +2 */
    CSAVE_REGS(19, 28, 176);	/* +5*2 */
    CSAVE_FREGS(8, 15, 96);	/* +4*2 */
    CSAVE_FPLR(192);		/* +1 */
    CALLOC_S(208);		/* +1 */
    CEND_ALIGN;			/* +1 +1 -> 24 */

    u32 = ((24u >> 2) << 27) | (1u << 20) | (fcofs >> 2);
    owrite(ctx, &u32, 4);
    owrite(ctx, &uwc, 24);

    u32 = 0;  /* Handler RVA to be relocated at 4 + 24. */
    owrite(ctx, &u32, 4);

    /* Unwind codes for vm_ffi_call without handler. */
    p = uwc;
    CADD_FP(16);		/* +2 */
    CSAVE_FPLR(16);		/* +1 */
    CSAVE_REGPX(19, -32);	/* +2 */
    CEND_ALIGN;			/* +1 +2 -> 8 */

    u32 = ((8u >> 2) << 27) | (((uint32_t)ctx->codesz - fcofs) >> 2);
    owrite(ctx, &u32, 4);
    owrite(ctx, &uwc, 8);

    reloc.vaddr = 4 + 24; reloc.symidx = 1+2+nrsym+2+2;
    reloc.type = PEOBJ_RELOC_ADDR32NB;
    owrite(ctx, &reloc, PEOBJ_RELOC_SIZE);
  }
#elif LJ_TARGET_X86
  /* Write .sxdata section. */
  for (i = 0; i < nrsym; i++) {
    if (!strcmp(ctx->relocsym[i], "_lj_err_unwind_win")) {
      uint32_t symidx = 1+2+i;
      owrite(ctx, &symidx, 4);
      break;
    }
  }
  if (i == nrsym) {
    fprintf(stderr, "Error: extern lj_err_unwind_win not used\n");
    exit(1);
  }
#endif

  /* Write .rdata$Z section. */
  owrite(ctx, ctx->dasm_ident, strlen(ctx->dasm_ident)+1);

  /* Write symbol table. */
  strtab = NULL;  /* 1st pass: collect string sizes. */
  for (;;) {
    strtabofs = 4;
    /* Mark as SafeSEH compliant. */
    emit_peobj_sym(ctx, "@feat.00", 1,
		   PEOBJ_SECT_ABS, PEOBJ_TYPE_NULL, PEOBJ_SCL_STATIC);

    emit_peobj_sym_sect(ctx, pesect, PEOBJ_SECT_TEXT);
    for (i = 0; i < nrsym; i++)
      emit_peobj_sym(ctx, ctx->relocsym[i], 0,
		     PEOBJ_SECT_UNDEF, PEOBJ_TYPE_FUNC, PEOBJ_SCL_EXTERN);

#ifdef PEOBJ_PDATA_NRELOC
    emit_peobj_sym_sect(ctx, pesect, PEOBJ_SECT_PDATA);
    emit_peobj_sym_sect(ctx, pesect, PEOBJ_SECT_XDATA);
    emit_peobj_sym(ctx, "lj_err_unwind_win", 0,
		   PEOBJ_SECT_UNDEF, PEOBJ_TYPE_FUNC, PEOBJ_SCL_EXTERN);
#elif LJ_TARGET_X86
    emit_peobj_sym_sect(ctx, pesect, PEOBJ_SECT_SXDATA);
#endif

    emit_peobj_sym(ctx, ctx->beginsym, 0,
		   PEOBJ_SECT_TEXT, PEOBJ_TYPE_NULL, PEOBJ_SCL_EXTERN);
    for (i = 0; i < ctx->nsym; i++)
      emit_peobj_sym(ctx, ctx->sym[i].name, (uint32_t)ctx->sym[i].ofs,
		     PEOBJ_SECT_TEXT, PEOBJ_TYPE_FUNC, PEOBJ_SCL_EXTERN);

    emit_peobj_sym_sect(ctx, pesect, PEOBJ_SECT_RDATA_Z);

    if (strtab)
      break;
    /* 2nd pass: alloc strtab, write syms and copy strings. */
    strtab = (char *)malloc(strtabofs);
    *(uint32_t *)strtab = (uint32_t)strtabofs;
  }

  /* Write string table. */
  owrite(ctx, strtab, strtabofs);
}

#else

void emit_peobj(BuildCtx *ctx)
{
  UNUSED(ctx);
  fprintf(stderr, "Error: no PE object support for this target\n");
  exit(1);
}

#endif
