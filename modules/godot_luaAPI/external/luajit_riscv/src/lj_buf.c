/*
** Buffer handling.
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
*/

#define lj_buf_c
#define LUA_CORE

#include "lj_obj.h"
#include "lj_gc.h"
#include "lj_err.h"
#include "lj_buf.h"
#include "lj_str.h"
#include "lj_tab.h"
#include "lj_strfmt.h"

/* -- Buffer management --------------------------------------------------- */

static void buf_grow(SBuf *sb, MSize sz)
{
  MSize osz = sbufsz(sb), len = sbuflen(sb), nsz = osz;
  char *b;
  GCSize flag;
  if (nsz < LJ_MIN_SBUF) nsz = LJ_MIN_SBUF;
  while (nsz < sz) nsz += nsz;
  flag = sbufflag(sb);
  if ((flag & SBUF_FLAG_COW)) {  /* Copy-on-write semantics. */
    lj_assertG_(G(sbufL(sb)), sb->w == sb->e, "bad SBuf COW");
    b = (char *)lj_mem_new(sbufL(sb), nsz);
    setsbufflag(sb, flag & ~(GCSize)SBUF_FLAG_COW);
    setgcrefnull(sbufX(sb)->cowref);
    memcpy(b, sb->b, osz);
  } else {
    b = (char *)lj_mem_realloc(sbufL(sb), sb->b, osz, nsz);
  }
  if ((flag & SBUF_FLAG_EXT)) {
    sbufX(sb)->r = sbufX(sb)->r - sb->b + b;  /* Adjust read pointer, too. */
  }
  /* Adjust buffer pointers. */
  sb->b = b;
  sb->w = b + len;
  sb->e = b + nsz;
  if ((flag & SBUF_FLAG_BORROW)) {  /* Adjust borrowed buffer pointers. */
    SBuf *bsb = mref(sbufX(sb)->bsb, SBuf);
    bsb->b = b;
    bsb->w = b + len;
    bsb->e = b + nsz;
  }
}

LJ_NOINLINE char *LJ_FASTCALL lj_buf_need2(SBuf *sb, MSize sz)
{
  lj_assertG_(G(sbufL(sb)), sz > sbufsz(sb), "SBuf overflow");
  if (LJ_UNLIKELY(sz > LJ_MAX_BUF))
    lj_err_mem(sbufL(sb));
  buf_grow(sb, sz);
  return sb->b;
}

LJ_NOINLINE char *LJ_FASTCALL lj_buf_more2(SBuf *sb, MSize sz)
{
  if (sbufisext(sb)) {
    SBufExt *sbx = (SBufExt *)sb;
    MSize len = sbufxlen(sbx);
    if (LJ_UNLIKELY(sz > LJ_MAX_BUF || len + sz > LJ_MAX_BUF))
      lj_err_mem(sbufL(sbx));
    if (len + sz > sbufsz(sbx)) {  /* Must grow. */
      buf_grow((SBuf *)sbx, len + sz);
    } else if (sbufiscow(sb) || sbufxslack(sbx) < (sbufsz(sbx) >> 3)) {
      /* Also grow to avoid excessive compactions, if slack < size/8. */
      buf_grow((SBuf *)sbx, sbuflen(sbx) + sz);  /* Not sbufxlen! */
      return sbx->w;
    }
    if (sbx->r != sbx->b) {  /* Compact by moving down. */
      memmove(sbx->b, sbx->r, len);
      sbx->r = sbx->b;
      sbx->w = sbx->b + len;
      lj_assertG_(G(sbufL(sbx)), len + sz <= sbufsz(sbx), "bad SBuf compact");
    }
  } else {
    MSize len = sbuflen(sb);
    lj_assertG_(G(sbufL(sb)), sz > sbufleft(sb), "SBuf overflow");
    if (LJ_UNLIKELY(sz > LJ_MAX_BUF || len + sz > LJ_MAX_BUF))
      lj_err_mem(sbufL(sb));
    buf_grow(sb, len + sz);
  }
  return sb->w;
}

void LJ_FASTCALL lj_buf_shrink(lua_State *L, SBuf *sb)
{
  char *b = sb->b;
  MSize osz = (MSize)(sb->e - b);
  if (osz > 2*LJ_MIN_SBUF) {
    MSize n = (MSize)(sb->w - b);
    b = lj_mem_realloc(L, b, osz, (osz >> 1));
    sb->b = b;
    sb->w = b + n;
    sb->e = b + (osz >> 1);
  }
  lj_assertG_(G(sbufL(sb)), !sbufisext(sb), "YAGNI shrink SBufExt");
}

char * LJ_FASTCALL lj_buf_tmp(lua_State *L, MSize sz)
{
  SBuf *sb = &G(L)->tmpbuf;
  setsbufL(sb, L);
  return lj_buf_need(sb, sz);
}

#if LJ_HASBUFFER && LJ_HASJIT
void lj_bufx_set(SBufExt *sbx, const char *p, MSize len, GCobj *ref)
{
  lua_State *L = sbufL(sbx);
  lj_bufx_free(L, sbx);
  lj_bufx_set_cow(L, sbx, p, len);
  setgcref(sbx->cowref, ref);
  lj_gc_objbarrier(L, (GCudata *)sbx - 1, ref);
}

#if LJ_HASFFI
MSize LJ_FASTCALL lj_bufx_more(SBufExt *sbx, MSize sz)
{
  lj_buf_more((SBuf *)sbx, sz);
  return sbufleft(sbx);
}
#endif
#endif

/* -- Low-level buffer put operations ------------------------------------- */

SBuf *lj_buf_putmem(SBuf *sb, const void *q, MSize len)
{
  char *w = lj_buf_more(sb, len);
  w = lj_buf_wmem(w, q, len);
  sb->w = w;
  return sb;
}

#if LJ_HASJIT || LJ_HASFFI
static LJ_NOINLINE SBuf * LJ_FASTCALL lj_buf_putchar2(SBuf *sb, int c)
{
  char *w = lj_buf_more2(sb, 1);
  *w++ = (char)c;
  sb->w = w;
  return sb;
}

SBuf * LJ_FASTCALL lj_buf_putchar(SBuf *sb, int c)
{
  char *w = sb->w;
  if (LJ_LIKELY(w < sb->e)) {
    *w++ = (char)c;
    sb->w = w;
    return sb;
  }
  return lj_buf_putchar2(sb, c);
}
#endif

SBuf * LJ_FASTCALL lj_buf_putstr(SBuf *sb, GCstr *s)
{
  MSize len = s->len;
  char *w = lj_buf_more(sb, len);
  w = lj_buf_wmem(w, strdata(s), len);
  sb->w = w;
  return sb;
}

/* -- High-level buffer put operations ------------------------------------ */

SBuf * LJ_FASTCALL lj_buf_putstr_reverse(SBuf *sb, GCstr *s)
{
  MSize len = s->len;
  char *w = lj_buf_more(sb, len), *e = w+len;
  const char *q = strdata(s)+len-1;
  while (w < e)
    *w++ = *q--;
  sb->w = w;
  return sb;
}

SBuf * LJ_FASTCALL lj_buf_putstr_lower(SBuf *sb, GCstr *s)
{
  MSize len = s->len;
  char *w = lj_buf_more(sb, len), *e = w+len;
  const char *q = strdata(s);
  for (; w < e; w++, q++) {
    uint32_t c = *(unsigned char *)q;
#if LJ_TARGET_PPC
    *w = c + ((c >= 'A' && c <= 'Z') << 5);
#else
    if (c >= 'A' && c <= 'Z') c += 0x20;
    *w = c;
#endif
  }
  sb->w = w;
  return sb;
}

SBuf * LJ_FASTCALL lj_buf_putstr_upper(SBuf *sb, GCstr *s)
{
  MSize len = s->len;
  char *w = lj_buf_more(sb, len), *e = w+len;
  const char *q = strdata(s);
  for (; w < e; w++, q++) {
    uint32_t c = *(unsigned char *)q;
#if LJ_TARGET_PPC
    *w = c - ((c >= 'a' && c <= 'z') << 5);
#else
    if (c >= 'a' && c <= 'z') c -= 0x20;
    *w = c;
#endif
  }
  sb->w = w;
  return sb;
}

SBuf *lj_buf_putstr_rep(SBuf *sb, GCstr *s, int32_t rep)
{
  MSize len = s->len;
  if (rep > 0 && len) {
    uint64_t tlen = (uint64_t)rep * len;
    char *w;
    if (LJ_UNLIKELY(tlen > LJ_MAX_STR))
      lj_err_mem(sbufL(sb));
    w = lj_buf_more(sb, (MSize)tlen);
    if (len == 1) {  /* Optimize a common case. */
      uint32_t c = strdata(s)[0];
      do { *w++ = c; } while (--rep > 0);
    } else {
      const char *e = strdata(s) + len;
      do {
	const char *q = strdata(s);
	do { *w++ = *q++; } while (q < e);
      } while (--rep > 0);
    }
    sb->w = w;
  }
  return sb;
}

SBuf *lj_buf_puttab(SBuf *sb, GCtab *t, GCstr *sep, int32_t i, int32_t e)
{
  MSize seplen = sep ? sep->len : 0;
  if (i <= e) {
    for (;;) {
      cTValue *o = lj_tab_getint(t, i);
      char *w;
      if (!o) {
      badtype:  /* Error: bad element type. */
	sb->w = (char *)(intptr_t)i;  /* Store failing index. */
	return NULL;
      } else if (tvisstr(o)) {
	MSize len = strV(o)->len;
	w = lj_buf_wmem(lj_buf_more(sb, len + seplen), strVdata(o), len);
      } else if (tvisint(o)) {
	w = lj_strfmt_wint(lj_buf_more(sb, STRFMT_MAXBUF_INT+seplen), intV(o));
      } else if (tvisnum(o)) {
	w = lj_buf_more(lj_strfmt_putfnum(sb, STRFMT_G14, numV(o)), seplen);
      } else {
	goto badtype;
      }
      if (i++ == e) {
	sb->w = w;
	break;
      }
      if (seplen) w = lj_buf_wmem(w, strdata(sep), seplen);
      sb->w = w;
    }
  }
  return sb;
}

/* -- Miscellaneous buffer operations ------------------------------------- */

GCstr * LJ_FASTCALL lj_buf_tostr(SBuf *sb)
{
  return lj_str_new(sbufL(sb), sb->b, sbuflen(sb));
}

/* Concatenate two strings. */
GCstr *lj_buf_cat2str(lua_State *L, GCstr *s1, GCstr *s2)
{
  MSize len1 = s1->len, len2 = s2->len;
  char *buf = lj_buf_tmp(L, len1 + len2);
  memcpy(buf, strdata(s1), len1);
  memcpy(buf+len1, strdata(s2), len2);
  return lj_str_new(L, buf, len1 + len2);
}

/* Read ULEB128 from buffer. */
uint32_t LJ_FASTCALL lj_buf_ruleb128(const char **pp)
{
  const uint8_t *w = (const uint8_t *)*pp;
  uint32_t v = *w++;
  if (LJ_UNLIKELY(v >= 0x80)) {
    int sh = 0;
    v &= 0x7f;
    do { v |= ((*w & 0x7f) << (sh += 7)); } while (*w++ >= 0x80);
  }
  *pp = (const char *)w;
  return v;
}

