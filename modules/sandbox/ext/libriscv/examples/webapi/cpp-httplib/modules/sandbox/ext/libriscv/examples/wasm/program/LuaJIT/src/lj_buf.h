/*
** Buffer handling.
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
*/

#ifndef _LJ_BUF_H
#define _LJ_BUF_H

#include "lj_obj.h"
#include "lj_gc.h"
#include "lj_str.h"

/* Resizable string buffers. */

/* The SBuf struct definition is in lj_obj.h:
**   char *w;	Write pointer.
**   char *e;	End pointer.
**   char *b;	Base pointer.
**   MRef L;	lua_State, used for buffer resizing. Extension bits in 3 LSB.
*/

/* Extended string buffer. */
typedef struct SBufExt {
  SBufHeader;
  union {
    GCRef cowref;	/* Copy-on-write object reference. */
    MRef bsb;		/* Borrowed string buffer. */
  };
  char *r;		/* Read pointer. */
  GCRef dict_str;	/* Serialization string dictionary table. */
  GCRef dict_mt;	/* Serialization metatable dictionary table. */
  int depth;		/* Remaining recursion depth. */
} SBufExt;

#define sbufsz(sb)		((MSize)((sb)->e - (sb)->b))
#define sbuflen(sb)		((MSize)((sb)->w - (sb)->b))
#define sbufleft(sb)		((MSize)((sb)->e - (sb)->w))
#define sbufxlen(sbx)		((MSize)((sbx)->w - (sbx)->r))
#define sbufxslack(sbx)		((MSize)((sbx)->r - (sbx)->b))

#define SBUF_MASK_FLAG		(7)
#define SBUF_MASK_L		(~(GCSize)SBUF_MASK_FLAG)
#define SBUF_FLAG_EXT		1	/* Extended string buffer. */
#define SBUF_FLAG_COW		2	/* Copy-on-write buffer. */
#define SBUF_FLAG_BORROW	4	/* Borrowed string buffer. */

#define sbufL(sb) \
  ((lua_State *)(void *)(uintptr_t)(mrefu((sb)->L) & SBUF_MASK_L))
#define setsbufL(sb, l)		(setmref((sb)->L, (l)))
#define setsbufXL(sb, l, flag) \
  (setmrefu((sb)->L, (GCSize)(uintptr_t)(void *)(l) + (flag)))
#define setsbufXL_(sb, l) \
  (setmrefu((sb)->L, (GCSize)(uintptr_t)(void *)(l) | (mrefu((sb)->L) & SBUF_MASK_FLAG)))

#define sbufflag(sb)		(mrefu((sb)->L))
#define sbufisext(sb)		(sbufflag((sb)) & SBUF_FLAG_EXT)
#define sbufiscow(sb)		(sbufflag((sb)) & SBUF_FLAG_COW)
#define sbufisborrow(sb)	(sbufflag((sb)) & SBUF_FLAG_BORROW)
#define sbufiscoworborrow(sb)	(sbufflag((sb)) & (SBUF_FLAG_COW|SBUF_FLAG_BORROW))
#define sbufX(sb) \
  (lj_assertG_(G(sbufL(sb)), sbufisext(sb), "not an SBufExt"), (SBufExt *)(sb))
#define setsbufflag(sb, flag)	(setmrefu((sb)->L, (flag)))

#define tvisbuf(o) \
  (LJ_HASBUFFER && tvisudata(o) && udataV(o)->udtype == UDTYPE_BUFFER)
#define bufV(o)		check_exp(tvisbuf(o), ((SBufExt *)uddata(udataV(o))))

/* Buffer management */
LJ_FUNC char *LJ_FASTCALL lj_buf_need2(SBuf *sb, MSize sz);
LJ_FUNC char *LJ_FASTCALL lj_buf_more2(SBuf *sb, MSize sz);
LJ_FUNC void LJ_FASTCALL lj_buf_shrink(lua_State *L, SBuf *sb);
LJ_FUNC char * LJ_FASTCALL lj_buf_tmp(lua_State *L, MSize sz);

static LJ_AINLINE void lj_buf_init(lua_State *L, SBuf *sb)
{
  setsbufL(sb, L);
  sb->w = sb->e = sb->b = NULL;
}

static LJ_AINLINE void lj_buf_reset(SBuf *sb)
{
  sb->w = sb->b;
}

static LJ_AINLINE SBuf *lj_buf_tmp_(lua_State *L)
{
  SBuf *sb = &G(L)->tmpbuf;
  setsbufL(sb, L);
  lj_buf_reset(sb);
  return sb;
}

static LJ_AINLINE void lj_buf_free(global_State *g, SBuf *sb)
{
  lj_assertG(!sbufisext(sb), "bad free of SBufExt");
  lj_mem_free(g, sb->b, sbufsz(sb));
}

static LJ_AINLINE char *lj_buf_need(SBuf *sb, MSize sz)
{
  if (LJ_UNLIKELY(sz > sbufsz(sb)))
    return lj_buf_need2(sb, sz);
  return sb->b;
}

static LJ_AINLINE char *lj_buf_more(SBuf *sb, MSize sz)
{
  if (LJ_UNLIKELY(sz > sbufleft(sb)))
    return lj_buf_more2(sb, sz);
  return sb->w;
}

/* Extended buffer management */
static LJ_AINLINE void lj_bufx_init(lua_State *L, SBufExt *sbx)
{
  memset(sbx, 0, sizeof(SBufExt));
  setsbufXL(sbx, L, SBUF_FLAG_EXT);
}

static LJ_AINLINE void lj_bufx_set_borrow(lua_State *L, SBufExt *sbx, SBuf *sb)
{
  setsbufXL(sbx, L, SBUF_FLAG_EXT | SBUF_FLAG_BORROW);
  setmref(sbx->bsb, sb);
  sbx->r = sbx->w = sbx->b = sb->b;
  sbx->e = sb->e;
}

static LJ_AINLINE void lj_bufx_set_cow(lua_State *L, SBufExt *sbx,
				       const char *p, MSize len)
{
  setsbufXL(sbx, L, SBUF_FLAG_EXT | SBUF_FLAG_COW);
  sbx->r = sbx->b = (char *)p;
  sbx->w = sbx->e = (char *)p + len;
}

static LJ_AINLINE void lj_bufx_reset(SBufExt *sbx)
{
  if (sbufiscow(sbx)) {
    setmrefu(sbx->L, (mrefu(sbx->L) & ~(GCSize)SBUF_FLAG_COW));
    setgcrefnull(sbx->cowref);
    sbx->b = sbx->e = NULL;
  }
  sbx->r = sbx->w = sbx->b;
}

static LJ_AINLINE void lj_bufx_free(lua_State *L, SBufExt *sbx)
{
  if (!sbufiscoworborrow(sbx)) lj_mem_free(G(L), sbx->b, sbufsz(sbx));
  setsbufXL(sbx, L, SBUF_FLAG_EXT);
  setgcrefnull(sbx->cowref);
  sbx->r = sbx->w = sbx->b = sbx->e = NULL;
}

#if LJ_HASBUFFER && LJ_HASJIT
LJ_FUNC void lj_bufx_set(SBufExt *sbx, const char *p, MSize len, GCobj *o);
#if LJ_HASFFI
LJ_FUNC MSize LJ_FASTCALL lj_bufx_more(SBufExt *sbx, MSize sz);
#endif
#endif

/* Low-level buffer put operations */
LJ_FUNC SBuf *lj_buf_putmem(SBuf *sb, const void *q, MSize len);
#if LJ_HASJIT || LJ_HASFFI
LJ_FUNC SBuf * LJ_FASTCALL lj_buf_putchar(SBuf *sb, int c);
#endif
LJ_FUNC SBuf * LJ_FASTCALL lj_buf_putstr(SBuf *sb, GCstr *s);

static LJ_AINLINE char *lj_buf_wmem(char *p, const void *q, MSize len)
{
  return (char *)memcpy(p, q, len) + len;
}

static LJ_AINLINE void lj_buf_putb(SBuf *sb, int c)
{
  char *w = lj_buf_more(sb, 1);
  *w++ = (char)c;
  sb->w = w;
}

/* High-level buffer put operations */
LJ_FUNCA SBuf * LJ_FASTCALL lj_buf_putstr_reverse(SBuf *sb, GCstr *s);
LJ_FUNCA SBuf * LJ_FASTCALL lj_buf_putstr_lower(SBuf *sb, GCstr *s);
LJ_FUNCA SBuf * LJ_FASTCALL lj_buf_putstr_upper(SBuf *sb, GCstr *s);
LJ_FUNC SBuf *lj_buf_putstr_rep(SBuf *sb, GCstr *s, int32_t rep);
LJ_FUNC SBuf *lj_buf_puttab(SBuf *sb, GCtab *t, GCstr *sep,
			    int32_t i, int32_t e);

/* Miscellaneous buffer operations */
LJ_FUNCA GCstr * LJ_FASTCALL lj_buf_tostr(SBuf *sb);
LJ_FUNC GCstr *lj_buf_cat2str(lua_State *L, GCstr *s1, GCstr *s2);
LJ_FUNC uint32_t LJ_FASTCALL lj_buf_ruleb128(const char **pp);

static LJ_AINLINE GCstr *lj_buf_str(lua_State *L, SBuf *sb)
{
  return lj_str_new(L, sb->b, sbuflen(sb));
}

#endif
