/*
** DynASM ARM64 encoding engine.
** Copyright (C) 2005-2023 Mike Pall. All rights reserved.
** Released under the MIT license. See dynasm.lua for full copyright notice.
*/

#include <stddef.h>
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>

#define DASM_ARCH		"arm64"

#ifndef DASM_EXTERN
#define DASM_EXTERN(a,b,c,d)	0
#endif

/* Action definitions. */
enum {
  DASM_STOP, DASM_SECTION, DASM_ESC, DASM_REL_EXT,
  /* The following actions need a buffer position. */
  DASM_ALIGN, DASM_REL_LG, DASM_LABEL_LG,
  /* The following actions also have an argument. */
  DASM_REL_PC, DASM_LABEL_PC, DASM_REL_A,
  DASM_IMM, DASM_IMM6, DASM_IMM12, DASM_IMM13W, DASM_IMM13X, DASM_IMML,
  DASM_IMMV, DASM_VREG,
  DASM__MAX
};

/* Maximum number of section buffer positions for a single dasm_put() call. */
#define DASM_MAXSECPOS		25

/* DynASM encoder status codes. Action list offset or number are or'ed in. */
#define DASM_S_OK		0x00000000
#define DASM_S_NOMEM		0x01000000
#define DASM_S_PHASE		0x02000000
#define DASM_S_MATCH_SEC	0x03000000
#define DASM_S_RANGE_I		0x11000000
#define DASM_S_RANGE_SEC	0x12000000
#define DASM_S_RANGE_LG		0x13000000
#define DASM_S_RANGE_PC		0x14000000
#define DASM_S_RANGE_REL	0x15000000
#define DASM_S_RANGE_VREG	0x16000000
#define DASM_S_UNDEF_LG		0x21000000
#define DASM_S_UNDEF_PC		0x22000000

/* Macros to convert positions (8 bit section + 24 bit index). */
#define DASM_POS2IDX(pos)	((pos)&0x00ffffff)
#define DASM_POS2BIAS(pos)	((pos)&0xff000000)
#define DASM_SEC2POS(sec)	((sec)<<24)
#define DASM_POS2SEC(pos)	((pos)>>24)
#define DASM_POS2PTR(D, pos)	(D->sections[DASM_POS2SEC(pos)].rbuf + (pos))

/* Action list type. */
typedef const unsigned int *dasm_ActList;

/* Per-section structure. */
typedef struct dasm_Section {
  int *rbuf;		/* Biased buffer pointer (negative section bias). */
  int *buf;		/* True buffer pointer. */
  size_t bsize;		/* Buffer size in bytes. */
  int pos;		/* Biased buffer position. */
  int epos;		/* End of biased buffer position - max single put. */
  int ofs;		/* Byte offset into section. */
} dasm_Section;

/* Core structure holding the DynASM encoding state. */
struct dasm_State {
  size_t psize;			/* Allocated size of this structure. */
  dasm_ActList actionlist;	/* Current actionlist pointer. */
  int *lglabels;		/* Local/global chain/pos ptrs. */
  size_t lgsize;
  int *pclabels;		/* PC label chains/pos ptrs. */
  size_t pcsize;
  void **globals;		/* Array of globals. */
  dasm_Section *section;	/* Pointer to active section. */
  size_t codesize;		/* Total size of all code sections. */
  int maxsection;		/* 0 <= sectionidx < maxsection. */
  int status;			/* Status code. */
  dasm_Section sections[1];	/* All sections. Alloc-extended. */
};

/* The size of the core structure depends on the max. number of sections. */
#define DASM_PSZ(ms)	(sizeof(dasm_State)+(ms-1)*sizeof(dasm_Section))


/* Initialize DynASM state. */
void dasm_init(Dst_DECL, int maxsection)
{
  dasm_State *D;
  size_t psz = 0;
  Dst_REF = NULL;
  DASM_M_GROW(Dst, struct dasm_State, Dst_REF, psz, DASM_PSZ(maxsection));
  D = Dst_REF;
  D->psize = psz;
  D->lglabels = NULL;
  D->lgsize = 0;
  D->pclabels = NULL;
  D->pcsize = 0;
  D->globals = NULL;
  D->maxsection = maxsection;
  memset((void *)D->sections, 0, maxsection * sizeof(dasm_Section));
}

/* Free DynASM state. */
void dasm_free(Dst_DECL)
{
  dasm_State *D = Dst_REF;
  int i;
  for (i = 0; i < D->maxsection; i++)
    if (D->sections[i].buf)
      DASM_M_FREE(Dst, D->sections[i].buf, D->sections[i].bsize);
  if (D->pclabels) DASM_M_FREE(Dst, D->pclabels, D->pcsize);
  if (D->lglabels) DASM_M_FREE(Dst, D->lglabels, D->lgsize);
  DASM_M_FREE(Dst, D, D->psize);
}

/* Setup global label array. Must be called before dasm_setup(). */
void dasm_setupglobal(Dst_DECL, void **gl, unsigned int maxgl)
{
  dasm_State *D = Dst_REF;
  D->globals = gl;
  DASM_M_GROW(Dst, int, D->lglabels, D->lgsize, (10+maxgl)*sizeof(int));
}

/* Grow PC label array. Can be called after dasm_setup(), too. */
void dasm_growpc(Dst_DECL, unsigned int maxpc)
{
  dasm_State *D = Dst_REF;
  size_t osz = D->pcsize;
  DASM_M_GROW(Dst, int, D->pclabels, D->pcsize, maxpc*sizeof(int));
  memset((void *)(((unsigned char *)D->pclabels)+osz), 0, D->pcsize-osz);
}

/* Setup encoder. */
void dasm_setup(Dst_DECL, const void *actionlist)
{
  dasm_State *D = Dst_REF;
  int i;
  D->actionlist = (dasm_ActList)actionlist;
  D->status = DASM_S_OK;
  D->section = &D->sections[0];
  memset((void *)D->lglabels, 0, D->lgsize);
  if (D->pclabels) memset((void *)D->pclabels, 0, D->pcsize);
  for (i = 0; i < D->maxsection; i++) {
    D->sections[i].pos = DASM_SEC2POS(i);
    D->sections[i].rbuf = D->sections[i].buf - D->sections[i].pos;
    D->sections[i].ofs = 0;
  }
}


#ifdef DASM_CHECKS
#define CK(x, st) \
  do { if (!(x)) { \
    D->status = DASM_S_##st|(int)(p-D->actionlist-1); return; } } while (0)
#define CKPL(kind, st) \
  do { if ((size_t)((char *)pl-(char *)D->kind##labels) >= D->kind##size) { \
    D->status = DASM_S_RANGE_##st|(int)(p-D->actionlist-1); return; } } while (0)
#else
#define CK(x, st)	((void)0)
#define CKPL(kind, st)	((void)0)
#endif

static int dasm_imm12(unsigned int n)
{
  if ((n >> 12) == 0)
    return n;
  else if ((n & 0xff000fff) == 0)
    return (n >> 12) | 0x1000;
  else
    return -1;
}

static int dasm_ffs(unsigned long long x)
{
  int n = -1;
  while (x) { x >>= 1; n++; }
  return n;
}

static int dasm_imm13(int lo, int hi)
{
  int inv = 0, w = 64, s = 0xfff, xa, xb;
  unsigned long long n = (((unsigned long long)hi) << 32) | (unsigned int)lo;
  unsigned long long m = 1ULL, a, b, c;
  if (n & 1) { n = ~n; inv = 1; }
  a = n & (unsigned long long)-(long long)n;
  b = (n+a)&(unsigned long long)-(long long)(n+a);
  c = (n+a-b)&(unsigned long long)-(long long)(n+a-b);
  xa = dasm_ffs(a); xb = dasm_ffs(b);
  if (c) {
    w = dasm_ffs(c) - xa;
    if (w == 32) m = 0x0000000100000001UL;
    else if (w == 16) m = 0x0001000100010001UL;
    else if (w == 8) m = 0x0101010101010101UL;
    else if (w == 4) m = 0x1111111111111111UL;
    else if (w == 2) m = 0x5555555555555555UL;
    else return -1;
    s = (-2*w & 0x3f) - 1;
  } else if (!a) {
    return -1;
  } else if (xb == -1) {
    xb = 64;
  }
  if ((b-a) * m != n) return -1;
  if (inv) {
    return ((w - xb) << 6) | (s+w+xa-xb);
  } else {
    return ((w - xa) << 6) | (s+xb-xa);
  }
  return -1;
}

/* Pass 1: Store actions and args, link branches/labels, estimate offsets. */
void dasm_put(Dst_DECL, int start, ...)
{
  va_list ap;
  dasm_State *D = Dst_REF;
  dasm_ActList p = D->actionlist + start;
  dasm_Section *sec = D->section;
  int pos = sec->pos, ofs = sec->ofs;
  int *b;

  if (pos >= sec->epos) {
    DASM_M_GROW(Dst, int, sec->buf, sec->bsize,
      sec->bsize + 2*DASM_MAXSECPOS*sizeof(int));
    sec->rbuf = sec->buf - DASM_POS2BIAS(pos);
    sec->epos = (int)sec->bsize/sizeof(int) - DASM_MAXSECPOS+DASM_POS2BIAS(pos);
  }

  b = sec->rbuf;
  b[pos++] = start;

  va_start(ap, start);
  while (1) {
    unsigned int ins = *p++;
    unsigned int action = (ins >> 16);
    if (action >= DASM__MAX) {
      ofs += 4;
    } else {
      int *pl, n = action >= DASM_REL_PC ? va_arg(ap, int) : 0;
      switch (action) {
      case DASM_STOP: goto stop;
      case DASM_SECTION:
	n = (ins & 255); CK(n < D->maxsection, RANGE_SEC);
	D->section = &D->sections[n]; goto stop;
      case DASM_ESC: p++; ofs += 4; break;
      case DASM_REL_EXT: if ((ins & 0x8000)) ofs += 8; break;
      case DASM_ALIGN: ofs += (ins & 255); b[pos++] = ofs; break;
      case DASM_REL_LG:
	n = (ins & 2047) - 10; pl = D->lglabels + n;
	/* Bkwd rel or global. */
	if (n >= 0) { CK(n>=10||*pl<0, RANGE_LG); CKPL(lg, LG); goto putrel; }
	pl += 10; n = *pl;
	if (n < 0) n = 0;  /* Start new chain for fwd rel if label exists. */
	goto linkrel;
      case DASM_REL_PC:
	pl = D->pclabels + n; CKPL(pc, PC);
      putrel:
	n = *pl;
	if (n < 0) {  /* Label exists. Get label pos and store it. */
	  b[pos] = -n;
	} else {
      linkrel:
	  b[pos] = n;  /* Else link to rel chain, anchored at label. */
	  *pl = pos;
	}
	pos++;
	if ((ins & 0x8000)) ofs += 8;
	break;
      case DASM_REL_A:
	b[pos++] = n;
	b[pos++] = va_arg(ap, int);
	break;
      case DASM_LABEL_LG:
	pl = D->lglabels + (ins & 2047) - 10; CKPL(lg, LG); goto putlabel;
      case DASM_LABEL_PC:
	pl = D->pclabels + n; CKPL(pc, PC);
      putlabel:
	n = *pl;  /* n > 0: Collapse rel chain and replace with label pos. */
	while (n > 0) { int *pb = DASM_POS2PTR(D, n); n = *pb; *pb = pos;
	}
	*pl = -pos;  /* Label exists now. */
	b[pos++] = ofs;  /* Store pass1 offset estimate. */
	break;
      case DASM_IMM:
	CK((n & ((1<<((ins>>10)&31))-1)) == 0, RANGE_I);
	n >>= ((ins>>10)&31);
#ifdef DASM_CHECKS
	if ((ins & 0x8000))
	  CK(((n + (1<<(((ins>>5)&31)-1)))>>((ins>>5)&31)) == 0, RANGE_I);
	else
	  CK((n>>((ins>>5)&31)) == 0, RANGE_I);
#endif
	b[pos++] = n;
	break;
      case DASM_IMM6:
	CK((n >> 6) == 0, RANGE_I);
	b[pos++] = n;
	break;
      case DASM_IMM12:
	CK(dasm_imm12((unsigned int)n) != -1, RANGE_I);
	b[pos++] = n;
	break;
      case DASM_IMM13W:
	CK(dasm_imm13(n, n) != -1, RANGE_I);
	b[pos++] = n;
	break;
      case DASM_IMM13X: {
	int m = va_arg(ap, int);
	CK(dasm_imm13(n, m) != -1, RANGE_I);
	b[pos++] = n;
	b[pos++] = m;
	break;
	}
      case DASM_IMML: {
#ifdef DASM_CHECKS
	int scale = (ins & 3);
	CK((!(n & ((1<<scale)-1)) && (unsigned int)(n>>scale) < 4096) ||
	   (unsigned int)(n+256) < 512, RANGE_I);
#endif
	b[pos++] = n;
	break;
	}
      case DASM_IMMV:
	ofs += 4;
	b[pos++] = n;
	break;
      case DASM_VREG:
	CK(n < 32, RANGE_VREG);
	b[pos++] = n;
	break;
      }
    }
  }
stop:
  va_end(ap);
  sec->pos = pos;
  sec->ofs = ofs;
}
#undef CK

/* Pass 2: Link sections, shrink aligns, fix label offsets. */
int dasm_link(Dst_DECL, size_t *szp)
{
  dasm_State *D = Dst_REF;
  int secnum;
  int ofs = 0;

#ifdef DASM_CHECKS
  *szp = 0;
  if (D->status != DASM_S_OK) return D->status;
  {
    int pc;
    for (pc = 0; pc*sizeof(int) < D->pcsize; pc++)
      if (D->pclabels[pc] > 0) return DASM_S_UNDEF_PC|pc;
  }
#endif

  { /* Handle globals not defined in this translation unit. */
    int idx;
    for (idx = 10; idx*sizeof(int) < D->lgsize; idx++) {
      int n = D->lglabels[idx];
      /* Undefined label: Collapse rel chain and replace with marker (< 0). */
      while (n > 0) { int *pb = DASM_POS2PTR(D, n); n = *pb; *pb = -idx; }
    }
  }

  /* Combine all code sections. No support for data sections (yet). */
  for (secnum = 0; secnum < D->maxsection; secnum++) {
    dasm_Section *sec = D->sections + secnum;
    int *b = sec->rbuf;
    int pos = DASM_SEC2POS(secnum);
    int lastpos = sec->pos;

    while (pos != lastpos) {
      dasm_ActList p = D->actionlist + b[pos++];
      while (1) {
	unsigned int ins = *p++;
	unsigned int action = (ins >> 16);
	switch (action) {
	case DASM_STOP: case DASM_SECTION: goto stop;
	case DASM_ESC: p++; break;
	case DASM_REL_EXT: break;
	case DASM_ALIGN: ofs -= (b[pos++] + ofs) & (ins & 255); break;
	case DASM_REL_LG: case DASM_REL_PC: pos++; break;
	case DASM_LABEL_LG: case DASM_LABEL_PC: b[pos++] += ofs; break;
	case DASM_IMM: case DASM_IMM6: case DASM_IMM12: case DASM_IMM13W:
	case DASM_IMML: case DASM_IMMV: case DASM_VREG: pos++; break;
	case DASM_IMM13X: case DASM_REL_A: pos += 2; break;
	}
      }
      stop: (void)0;
    }
    ofs += sec->ofs;  /* Next section starts right after current section. */
  }

  D->codesize = ofs;  /* Total size of all code sections */
  *szp = ofs;
  return DASM_S_OK;
}

#ifdef DASM_CHECKS
#define CK(x, st) \
  do { if (!(x)) return DASM_S_##st|(int)(p-D->actionlist-1); } while (0)
#else
#define CK(x, st)	((void)0)
#endif

/* Pass 3: Encode sections. */
int dasm_encode(Dst_DECL, void *buffer)
{
  dasm_State *D = Dst_REF;
  char *base = (char *)buffer;
  unsigned int *cp = (unsigned int *)buffer;
  int secnum;

  /* Encode all code sections. No support for data sections (yet). */
  for (secnum = 0; secnum < D->maxsection; secnum++) {
    dasm_Section *sec = D->sections + secnum;
    int *b = sec->buf;
    int *endb = sec->rbuf + sec->pos;

    while (b != endb) {
      dasm_ActList p = D->actionlist + *b++;
      while (1) {
	unsigned int ins = *p++;
	unsigned int action = (ins >> 16);
	int n = (action >= DASM_ALIGN && action < DASM__MAX) ? *b++ : 0;
	switch (action) {
	case DASM_STOP: case DASM_SECTION: goto stop;
	case DASM_ESC: *cp++ = *p++; break;
	case DASM_REL_EXT:
	  n = DASM_EXTERN(Dst, (unsigned char *)cp, (ins&2047), !(ins&2048));
	  goto patchrel;
	case DASM_ALIGN:
	  ins &= 255; while ((((char *)cp - base) & ins)) *cp++ = 0xd503201f;
	  break;
	case DASM_REL_LG:
	  if (n < 0) {
	    ptrdiff_t na = (ptrdiff_t)D->globals[-n-10] - (ptrdiff_t)cp + 4;
	    n = (int)na;
	    CK((ptrdiff_t)n == na, RANGE_REL);
	    goto patchrel;
	  }
	  /* fallthrough */
	case DASM_REL_PC:
	  CK(n >= 0, UNDEF_PC);
	  n = *DASM_POS2PTR(D, n) - (int)((char *)cp - base) + 4;
	patchrel:
	  if (!(ins & 0xf800)) {  /* B, BL */
	    CK((n & 3) == 0 && ((n+0x08000000) >> 28) == 0, RANGE_REL);
	    cp[-1] |= ((n >> 2) & 0x03ffffff);
	  } else if ((ins & 0x800)) {  /* B.cond, CBZ, CBNZ, LDR* literal */
	    CK((n & 3) == 0 && ((n+0x00100000) >> 21) == 0, RANGE_REL);
	    cp[-1] |= ((n << 3) & 0x00ffffe0);
	  } else if ((ins & 0x3000) == 0x2000) {  /* ADR */
	    CK(((n+0x00100000) >> 21) == 0, RANGE_REL);
	    cp[-1] |= ((n << 3) & 0x00ffffe0) | ((n & 3) << 29);
	  } else if ((ins & 0x3000) == 0x3000) {  /* ADRP */
	    cp[-1] |= ((n >> 9) & 0x00ffffe0) | (((n >> 12) & 3) << 29);
	  } else if ((ins & 0x1000)) {  /* TBZ, TBNZ */
	    CK((n & 3) == 0 && ((n+0x00008000) >> 16) == 0, RANGE_REL);
	    cp[-1] |= ((n << 3) & 0x0007ffe0);
	  } else if ((ins & 0x8000)) {  /* absolute */
	    cp[0] = (unsigned int)((ptrdiff_t)cp - 4 + n);
	    cp[1] = (unsigned int)(((ptrdiff_t)cp - 4 + n) >> 32);
	    cp += 2;
	  }
	  break;
	case DASM_REL_A: {
	  ptrdiff_t na = (((ptrdiff_t)(*b++) << 32) | (unsigned int)n);
	  if ((ins & 0x3000) == 0x3000) {  /* ADRP */
	    ins &= ~0x1000;
	    na = (na >> 12) - (((ptrdiff_t)cp - 4) >> 12);
	  } else {
	    na = na - (ptrdiff_t)cp + 4;
	  }
	  n = (int)na;
	  CK((ptrdiff_t)n == na, RANGE_REL);
	  goto patchrel;
	}
	case DASM_LABEL_LG:
	  ins &= 2047; if (ins >= 20) D->globals[ins-20] = (void *)(base + n);
	  break;
	case DASM_LABEL_PC: break;
	case DASM_IMM:
	  cp[-1] |= (n & ((1<<((ins>>5)&31))-1)) << (ins&31);
	  break;
	case DASM_IMM6:
	  cp[-1] |= ((n&31) << 19) | ((n&32) << 26);
	  break;
	case DASM_IMM12:
	  cp[-1] |= (dasm_imm12((unsigned int)n) << 10);
	  break;
	case DASM_IMM13W:
	  cp[-1] |= (dasm_imm13(n, n) << 10);
	  break;
	case DASM_IMM13X:
	  cp[-1] |= (dasm_imm13(n, *b++) << 10);
	  break;
	case DASM_IMML: {
	  int scale = (ins & 3);
	  cp[-1] |= (!(n & ((1<<scale)-1)) && (unsigned int)(n>>scale) < 4096) ?
	    ((n << (10-scale)) | 0x01000000) : ((n & 511) << 12);
	  break;
	  }
	case DASM_IMMV:
	  *cp++ = n;
	  break;
	case DASM_VREG:
	  cp[-1] |= (n & 0x1f) << (ins & 0x1f);
	  break;
	default: *cp++ = ins; break;
	}
      }
      stop: (void)0;
    }
  }

  if (base + D->codesize != (char *)cp)  /* Check for phase errors. */
    return DASM_S_PHASE;
  return DASM_S_OK;
}
#undef CK

/* Get PC label offset. */
int dasm_getpclabel(Dst_DECL, unsigned int pc)
{
  dasm_State *D = Dst_REF;
  if (pc*sizeof(int) < D->pcsize) {
    int pos = D->pclabels[pc];
    if (pos < 0) return *DASM_POS2PTR(D, -pos);
    if (pos > 0) return -1;  /* Undefined. */
  }
  return -2;  /* Unused or out of range. */
}

#ifdef DASM_CHECKS
/* Optional sanity checker to call between isolated encoding steps. */
int dasm_checkstep(Dst_DECL, int secmatch)
{
  dasm_State *D = Dst_REF;
  if (D->status == DASM_S_OK) {
    int i;
    for (i = 1; i <= 9; i++) {
      if (D->lglabels[i] > 0) { D->status = DASM_S_UNDEF_LG|i; break; }
      D->lglabels[i] = 0;
    }
  }
  if (D->status == DASM_S_OK && secmatch >= 0 &&
      D->section != &D->sections[secmatch])
    D->status = DASM_S_MATCH_SEC|(int)(D->section-D->sections);
  return D->status;
}
#endif

