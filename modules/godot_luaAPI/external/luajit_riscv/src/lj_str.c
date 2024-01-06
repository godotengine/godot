/*
** String handling.
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
*/

#define lj_str_c
#define LUA_CORE

#include "lj_obj.h"
#include "lj_gc.h"
#include "lj_err.h"
#include "lj_str.h"
#include "lj_char.h"
#include "lj_prng.h"

/* -- String helpers ------------------------------------------------------ */

/* Ordered compare of strings. Assumes string data is 4-byte aligned. */
int32_t LJ_FASTCALL lj_str_cmp(GCstr *a, GCstr *b)
{
  MSize i, n = a->len > b->len ? b->len : a->len;
  for (i = 0; i < n; i += 4) {
    /* Note: innocuous access up to end of string + 3. */
    uint32_t va = *(const uint32_t *)(strdata(a)+i);
    uint32_t vb = *(const uint32_t *)(strdata(b)+i);
    if (va != vb) {
#if LJ_LE
      va = lj_bswap(va); vb = lj_bswap(vb);
#endif
      i -= n;
      if ((int32_t)i >= -3) {
	va >>= 32+(i<<3); vb >>= 32+(i<<3);
	if (va == vb) break;
      }
      return va < vb ? -1 : 1;
    }
  }
  return (int32_t)(a->len - b->len);
}

/* Find fixed string p inside string s. */
const char *lj_str_find(const char *s, const char *p, MSize slen, MSize plen)
{
  if (plen <= slen) {
    if (plen == 0) {
      return s;
    } else {
      int c = *(const uint8_t *)p++;
      plen--; slen -= plen;
      while (slen) {
	const char *q = (const char *)memchr(s, c, slen);
	if (!q) break;
	if (memcmp(q+1, p, plen) == 0) return q;
	q++; slen -= (MSize)(q-s); s = q;
      }
    }
  }
  return NULL;
}

/* Check whether a string has a pattern matching character. */
int lj_str_haspattern(GCstr *s)
{
  const char *p = strdata(s), *q = p + s->len;
  while (p < q) {
    int c = *(const uint8_t *)p++;
    if (lj_char_ispunct(c) && strchr("^$*+?.([%-", c))
      return 1;  /* Found a pattern matching char. */
  }
  return 0;  /* No pattern matching chars found. */
}

/* -- String hashing ------------------------------------------------------ */

/* Keyed sparse ARX string hash. Constant time. */
static StrHash hash_sparse(uint64_t seed, const char *str, MSize len)
{
  /* Constants taken from lookup3 hash by Bob Jenkins. */
  StrHash a, b, h = len ^ (StrHash)seed;
  if (len >= 4) {  /* Caveat: unaligned access! */
    a = lj_getu32(str);
    h ^= lj_getu32(str+len-4);
    b = lj_getu32(str+(len>>1)-2);
    h ^= b; h -= lj_rol(b, 14);
    b += lj_getu32(str+(len>>2)-1);
  } else {
    a = *(const uint8_t *)str;
    h ^= *(const uint8_t *)(str+len-1);
    b = *(const uint8_t *)(str+(len>>1));
    h ^= b; h -= lj_rol(b, 14);
  }
  a ^= h; a -= lj_rol(h, 11);
  b ^= a; b -= lj_rol(a, 25);
  h ^= b; h -= lj_rol(b, 16);
  return h;
}

#if LUAJIT_SECURITY_STRHASH
/* Keyed dense ARX string hash. Linear time. */
static LJ_NOINLINE StrHash hash_dense(uint64_t seed, StrHash h,
				      const char *str, MSize len)
{
  StrHash b = lj_bswap(lj_rol(h ^ (StrHash)(seed >> 32), 4));
  if (len > 12) {
    StrHash a = (StrHash)seed;
    const char *pe = str+len-12, *p = pe, *q = str;
    do {
      a += lj_getu32(p);
      b += lj_getu32(p+4);
      h += lj_getu32(p+8);
      p = q; q += 12;
      h ^= b; h -= lj_rol(b, 14);
      a ^= h; a -= lj_rol(h, 11);
      b ^= a; b -= lj_rol(a, 25);
    } while (p < pe);
    h ^= b; h -= lj_rol(b, 16);
    a ^= h; a -= lj_rol(h, 4);
    b ^= a; b -= lj_rol(a, 14);
  }
  return b;
}
#endif

/* -- String interning ---------------------------------------------------- */

#define LJ_STR_MAXCOLL		32

/* Resize the string interning hash table (grow and shrink). */
void lj_str_resize(lua_State *L, MSize newmask)
{
  global_State *g = G(L);
  GCRef *newtab, *oldtab = g->str.tab;
  MSize i;

  /* No resizing during GC traversal or if already too big. */
  if (g->gc.state == GCSsweepstring || newmask >= LJ_MAX_STRTAB-1)
    return;

  newtab = lj_mem_newvec(L, newmask+1, GCRef);
  memset(newtab, 0, (newmask+1)*sizeof(GCRef));

#if LUAJIT_SECURITY_STRHASH
  /* Check which chains need secondary hashes. */
  if (g->str.second) {
    int newsecond = 0;
    /* Compute primary chain lengths. */
    for (i = g->str.mask; i != ~(MSize)0; i--) {
      GCobj *o = (GCobj *)(gcrefu(oldtab[i]) & ~(uintptr_t)1);
      while (o) {
	GCstr *s = gco2str(o);
	MSize hash = s->hashalg ? hash_sparse(g->str.seed, strdata(s), s->len) :
				  s->hash;
	hash &= newmask;
	setgcrefp(newtab[hash], gcrefu(newtab[hash]) + 1);
	o = gcnext(o);
      }
    }
    /* Mark secondary chains. */
    for (i = newmask; i != ~(MSize)0; i--) {
      int secondary = gcrefu(newtab[i]) > LJ_STR_MAXCOLL;
      newsecond |= secondary;
      setgcrefp(newtab[i], secondary);
    }
    g->str.second = newsecond;
  }
#endif

  /* Reinsert all strings from the old table into the new table. */
  for (i = g->str.mask; i != ~(MSize)0; i--) {
    GCobj *o = (GCobj *)(gcrefu(oldtab[i]) & ~(uintptr_t)1);
    while (o) {
      GCobj *next = gcnext(o);
      GCstr *s = gco2str(o);
      MSize hash = s->hash;
#if LUAJIT_SECURITY_STRHASH
      uintptr_t u;
      if (LJ_LIKELY(!s->hashalg)) {  /* String hashed with primary hash. */
	hash &= newmask;
	u = gcrefu(newtab[hash]);
	if (LJ_UNLIKELY(u & 1)) {  /* Switch string to secondary hash. */
	  s->hash = hash = hash_dense(g->str.seed, s->hash, strdata(s), s->len);
	  s->hashalg = 1;
	  hash &= newmask;
	  u = gcrefu(newtab[hash]);
	}
      } else {  /* String hashed with secondary hash. */
	MSize shash = hash_sparse(g->str.seed, strdata(s), s->len);
	u = gcrefu(newtab[shash & newmask]);
	if (u & 1) {
	  hash &= newmask;
	  u = gcrefu(newtab[hash]);
	} else {  /* Revert string back to primary hash. */
	  s->hash = shash;
	  s->hashalg = 0;
	  hash = (shash & newmask);
	}
      }
      /* NOBARRIER: The string table is a GC root. */
      setgcrefp(o->gch.nextgc, (u & ~(uintptr_t)1));
      setgcrefp(newtab[hash], ((uintptr_t)o | (u & 1)));
#else
      hash &= newmask;
      /* NOBARRIER: The string table is a GC root. */
      setgcrefr(o->gch.nextgc, newtab[hash]);
      setgcref(newtab[hash], o);
#endif
      o = next;
    }
  }

  /* Free old table and replace with new table. */
  lj_str_freetab(g);
  g->str.tab = newtab;
  g->str.mask = newmask;
}

#if LUAJIT_SECURITY_STRHASH
/* Rehash and rechain all strings in a chain. */
static LJ_NOINLINE GCstr *lj_str_rehash_chain(lua_State *L, StrHash hashc,
					      const char *str, MSize len)
{
  global_State *g = G(L);
  int ow = g->gc.state == GCSsweepstring ? otherwhite(g) : 0;  /* Sweeping? */
  GCRef *strtab = g->str.tab;
  MSize strmask = g->str.mask;
  GCobj *o = gcref(strtab[hashc & strmask]);
  setgcrefp(strtab[hashc & strmask], (void *)((uintptr_t)1));
  g->str.second = 1;
  while (o) {
    uintptr_t u;
    GCobj *next = gcnext(o);
    GCstr *s = gco2str(o);
    StrHash hash;
    if (ow) {  /* Must sweep while rechaining. */
      if (((o->gch.marked ^ LJ_GC_WHITES) & ow)) {  /* String alive? */
	lj_assertG(!isdead(g, o) || (o->gch.marked & LJ_GC_FIXED),
		   "sweep of undead string");
	makewhite(g, o);
      } else {  /* Free dead string. */
	lj_assertG(isdead(g, o) || ow == LJ_GC_SFIXED,
		   "sweep of unlive string");
	lj_str_free(g, s);
	o = next;
	continue;
      }
    }
    hash = s->hash;
    if (!s->hashalg) {  /* Rehash with secondary hash. */
      hash = hash_dense(g->str.seed, hash, strdata(s), s->len);
      s->hash = hash;
      s->hashalg = 1;
    }
    /* Rechain. */
    hash &= strmask;
    u = gcrefu(strtab[hash]);
    setgcrefp(o->gch.nextgc, (u & ~(uintptr_t)1));
    setgcrefp(strtab[hash], ((uintptr_t)o | (u & 1)));
    o = next;
  }
  /* Try to insert the pending string again. */
  return lj_str_new(L, str, len);
}
#endif

/* Reseed String ID from PRNG after random interval < 2^bits. */
#if LUAJIT_SECURITY_STRID == 1
#define STRID_RESEED_INTERVAL	8
#elif LUAJIT_SECURITY_STRID == 2
#define STRID_RESEED_INTERVAL	4
#elif LUAJIT_SECURITY_STRID >= 3
#define STRID_RESEED_INTERVAL	0
#endif

/* Allocate a new string and add to string interning table. */
static GCstr *lj_str_alloc(lua_State *L, const char *str, MSize len,
			   StrHash hash, int hashalg)
{
  GCstr *s = lj_mem_newt(L, lj_str_size(len), GCstr);
  global_State *g = G(L);
  uintptr_t u;
  newwhite(g, s);
  s->gct = ~LJ_TSTR;
  s->len = len;
  s->hash = hash;
#ifndef STRID_RESEED_INTERVAL
  s->sid = g->str.id++;
#elif STRID_RESEED_INTERVAL
  if (!g->str.idreseed--) {
    uint64_t r = lj_prng_u64(&g->prng);
    g->str.id = (StrID)r;
    g->str.idreseed = (uint8_t)(r >> (64 - STRID_RESEED_INTERVAL));
  }
  s->sid = g->str.id++;
#else
  s->sid = (StrID)lj_prng_u64(&g->prng);
#endif
  s->reserved = 0;
  s->hashalg = (uint8_t)hashalg;
  /* Clear last 4 bytes of allocated memory. Implies zero-termination, too. */
  *(uint32_t *)(strdatawr(s)+(len & ~(MSize)3)) = 0;
  memcpy(strdatawr(s), str, len);
  /* Add to string hash table. */
  hash &= g->str.mask;
  u = gcrefu(g->str.tab[hash]);
  setgcrefp(s->nextgc, (u & ~(uintptr_t)1));
  /* NOBARRIER: The string table is a GC root. */
  setgcrefp(g->str.tab[hash], ((uintptr_t)s | (u & 1)));
  if (g->str.num++ > g->str.mask)  /* Allow a 100% load factor. */
    lj_str_resize(L, (g->str.mask<<1)+1);  /* Grow string table. */
  return s;  /* Return newly interned string. */
}

/* Intern a string and return string object. */
GCstr *lj_str_new(lua_State *L, const char *str, size_t lenx)
{
  global_State *g = G(L);
  if (lenx-1 < LJ_MAX_STR-1) {
    MSize len = (MSize)lenx;
    StrHash hash = hash_sparse(g->str.seed, str, len);
    MSize coll = 0;
    int hashalg = 0;
    /* Check if the string has already been interned. */
    GCobj *o = gcref(g->str.tab[hash & g->str.mask]);
#if LUAJIT_SECURITY_STRHASH
    if (LJ_UNLIKELY((uintptr_t)o & 1)) {  /* Secondary hash for this chain? */
      hashalg = 1;
      hash = hash_dense(g->str.seed, hash, str, len);
      o = (GCobj *)(gcrefu(g->str.tab[hash & g->str.mask]) & ~(uintptr_t)1);
    }
#endif
    while (o != NULL) {
      GCstr *sx = gco2str(o);
      if (sx->hash == hash && sx->len == len) {
	if (memcmp(str, strdata(sx), len) == 0) {
	  if (isdead(g, o)) flipwhite(o);  /* Resurrect if dead. */
	  return sx;  /* Return existing string. */
	}
	coll++;
      }
      coll++;
      o = gcnext(o);
    }
#if LUAJIT_SECURITY_STRHASH
    /* Rehash chain if there are too many collisions. */
    if (LJ_UNLIKELY(coll > LJ_STR_MAXCOLL) && !hashalg) {
      return lj_str_rehash_chain(L, hash, str, len);
    }
#endif
    /* Otherwise allocate a new string. */
    return lj_str_alloc(L, str, len, hash, hashalg);
  } else {
    if (lenx)
      lj_err_msg(L, LJ_ERR_STROV);
    return &g->strempty;
  }
}

void LJ_FASTCALL lj_str_free(global_State *g, GCstr *s)
{
  g->str.num--;
  lj_mem_free(g, s, lj_str_size(s->len));
}

void LJ_FASTCALL lj_str_init(lua_State *L)
{
  global_State *g = G(L);
  g->str.seed = lj_prng_u64(&g->prng);
  lj_str_resize(L, LJ_MIN_STRTAB-1);
}

