/*
** Table handling.
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
**
** Major portions taken verbatim or adapted from the Lua interpreter.
** Copyright (C) 1994-2008 Lua.org, PUC-Rio. See Copyright Notice in lua.h
*/

#define lj_tab_c
#define LUA_CORE

#include "lj_obj.h"
#include "lj_gc.h"
#include "lj_err.h"
#include "lj_tab.h"

/* -- Object hashing ------------------------------------------------------ */

/* Hash an arbitrary key and return its anchor position in the hash table. */
static Node *hashkey(const GCtab *t, cTValue *key)
{
  lj_assertX(!tvisint(key), "attempt to hash integer");
  if (tvisstr(key))
    return hashstr(t, strV(key));
  else if (tvisnum(key))
    return hashnum(t, key);
  else if (tvisbool(key))
    return hashmask(t, boolV(key));
  else
    return hashgcref(t, key->gcr);
  /* Only hash 32 bits of lightuserdata on a 64 bit CPU. Good enough? */
}

/* -- Table creation and destruction -------------------------------------- */

/* Create new hash part for table. */
static LJ_AINLINE void newhpart(lua_State *L, GCtab *t, uint32_t hbits)
{
  uint32_t hsize;
  Node *node;
  lj_assertL(hbits != 0, "zero hash size");
  if (hbits > LJ_MAX_HBITS)
    lj_err_msg(L, LJ_ERR_TABOV);
  hsize = 1u << hbits;
  node = lj_mem_newvec(L, hsize, Node);
  setmref(t->node, node);
  setfreetop(t, node, &node[hsize]);
  t->hmask = hsize-1;
}

/*
** Q: Why all of these copies of t->hmask, t->node etc. to local variables?
** A: Because alias analysis for C is _really_ tough.
**    Even state-of-the-art C compilers won't produce good code without this.
*/

/* Clear hash part of table. */
static LJ_AINLINE void clearhpart(GCtab *t)
{
  uint32_t i, hmask = t->hmask;
  Node *node = noderef(t->node);
  lj_assertX(t->hmask != 0, "empty hash part");
  for (i = 0; i <= hmask; i++) {
    Node *n = &node[i];
    setmref(n->next, NULL);
    setnilV(&n->key);
    setnilV(&n->val);
  }
}

/* Clear array part of table. */
static LJ_AINLINE void clearapart(GCtab *t)
{
  uint32_t i, asize = t->asize;
  TValue *array = tvref(t->array);
  for (i = 0; i < asize; i++)
    setnilV(&array[i]);
}

/* Create a new table. Note: the slots are not initialized (yet). */
static GCtab *newtab(lua_State *L, uint32_t asize, uint32_t hbits)
{
  GCtab *t;
  /* First try to colocate the array part. */
  if (LJ_MAX_COLOSIZE != 0 && asize > 0 && asize <= LJ_MAX_COLOSIZE) {
    Node *nilnode;
    lj_assertL((sizeof(GCtab) & 7) == 0, "bad GCtab size");
    t = (GCtab *)lj_mem_newgco(L, sizetabcolo(asize));
    t->gct = ~LJ_TTAB;
    t->nomm = (uint8_t)~0;
    t->colo = (int8_t)asize;
    setmref(t->array, (TValue *)((char *)t + sizeof(GCtab)));
    setgcrefnull(t->metatable);
    t->asize = asize;
    t->hmask = 0;
    nilnode = &G(L)->nilnode;
    setmref(t->node, nilnode);
#if LJ_GC64
    setmref(t->freetop, nilnode);
#endif
  } else {  /* Otherwise separately allocate the array part. */
    Node *nilnode;
    t = lj_mem_newobj(L, GCtab);
    t->gct = ~LJ_TTAB;
    t->nomm = (uint8_t)~0;
    t->colo = 0;
    setmref(t->array, NULL);
    setgcrefnull(t->metatable);
    t->asize = 0;  /* In case the array allocation fails. */
    t->hmask = 0;
    nilnode = &G(L)->nilnode;
    setmref(t->node, nilnode);
#if LJ_GC64
    setmref(t->freetop, nilnode);
#endif
    if (asize > 0) {
      if (asize > LJ_MAX_ASIZE)
	lj_err_msg(L, LJ_ERR_TABOV);
      setmref(t->array, lj_mem_newvec(L, asize, TValue));
      t->asize = asize;
    }
  }
  if (hbits)
    newhpart(L, t, hbits);
  return t;
}

/* Create a new table.
**
** IMPORTANT NOTE: The API differs from lua_createtable()!
**
** The array size is non-inclusive. E.g. asize=128 creates array slots
** for 0..127, but not for 128. If you need slots 1..128, pass asize=129
** (slot 0 is wasted in this case).
**
** The hash size is given in hash bits. hbits=0 means no hash part.
** hbits=1 creates 2 hash slots, hbits=2 creates 4 hash slots and so on.
*/
GCtab *lj_tab_new(lua_State *L, uint32_t asize, uint32_t hbits)
{
  GCtab *t = newtab(L, asize, hbits);
  clearapart(t);
  if (t->hmask > 0) clearhpart(t);
  return t;
}

/* The API of this function conforms to lua_createtable(). */
GCtab *lj_tab_new_ah(lua_State *L, int32_t a, int32_t h)
{
  return lj_tab_new(L, (uint32_t)(a > 0 ? a+1 : 0), hsize2hbits(h));
}

#if LJ_HASJIT
GCtab * LJ_FASTCALL lj_tab_new1(lua_State *L, uint32_t ahsize)
{
  GCtab *t = newtab(L, ahsize & 0xffffff, ahsize >> 24);
  clearapart(t);
  if (t->hmask > 0) clearhpart(t);
  return t;
}
#endif

/* Duplicate a table. */
GCtab * LJ_FASTCALL lj_tab_dup(lua_State *L, const GCtab *kt)
{
  GCtab *t;
  uint32_t asize, hmask;
  t = newtab(L, kt->asize, kt->hmask > 0 ? lj_fls(kt->hmask)+1 : 0);
  lj_assertL(kt->asize == t->asize && kt->hmask == t->hmask,
	     "mismatched size of table and template");
  t->nomm = 0;  /* Keys with metamethod names may be present. */
  asize = kt->asize;
  if (asize > 0) {
    TValue *array = tvref(t->array);
    TValue *karray = tvref(kt->array);
    if (asize < 64) {  /* An inlined loop beats memcpy for < 512 bytes. */
      uint32_t i;
      for (i = 0; i < asize; i++)
	copyTV(L, &array[i], &karray[i]);
    } else {
      memcpy(array, karray, asize*sizeof(TValue));
    }
  }
  hmask = kt->hmask;
  if (hmask > 0) {
    uint32_t i;
    Node *node = noderef(t->node);
    Node *knode = noderef(kt->node);
    ptrdiff_t d = (char *)node - (char *)knode;
    setfreetop(t, node, (Node *)((char *)getfreetop(kt, knode) + d));
    for (i = 0; i <= hmask; i++) {
      Node *kn = &knode[i];
      Node *n = &node[i];
      Node *next = nextnode(kn);
      /* Don't use copyTV here, since it asserts on a copy of a dead key. */
      n->val = kn->val; n->key = kn->key;
      setmref(n->next, next == NULL? next : (Node *)((char *)next + d));
    }
  }
  return t;
}

/* Clear a table. */
void LJ_FASTCALL lj_tab_clear(GCtab *t)
{
  clearapart(t);
  if (t->hmask > 0) {
    Node *node = noderef(t->node);
    setfreetop(t, node, &node[t->hmask+1]);
    clearhpart(t);
  }
}

/* Free a table. */
void LJ_FASTCALL lj_tab_free(global_State *g, GCtab *t)
{
  if (t->hmask > 0)
    lj_mem_freevec(g, noderef(t->node), t->hmask+1, Node);
  if (t->asize > 0 && LJ_MAX_COLOSIZE != 0 && t->colo <= 0)
    lj_mem_freevec(g, tvref(t->array), t->asize, TValue);
  if (LJ_MAX_COLOSIZE != 0 && t->colo)
    lj_mem_free(g, t, sizetabcolo((uint32_t)t->colo & 0x7f));
  else
    lj_mem_freet(g, t);
}

/* -- Table resizing ------------------------------------------------------ */

/* Resize a table to fit the new array/hash part sizes. */
void lj_tab_resize(lua_State *L, GCtab *t, uint32_t asize, uint32_t hbits)
{
  Node *oldnode = noderef(t->node);
  uint32_t oldasize = t->asize;
  uint32_t oldhmask = t->hmask;
  if (asize > oldasize) {  /* Array part grows? */
    TValue *array;
    uint32_t i;
    if (asize > LJ_MAX_ASIZE)
      lj_err_msg(L, LJ_ERR_TABOV);
    if (LJ_MAX_COLOSIZE != 0 && t->colo > 0) {
      /* A colocated array must be separated and copied. */
      TValue *oarray = tvref(t->array);
      array = lj_mem_newvec(L, asize, TValue);
      t->colo = (int8_t)(t->colo | 0x80);  /* Mark as separated (colo < 0). */
      for (i = 0; i < oldasize; i++)
	copyTV(L, &array[i], &oarray[i]);
    } else {
      array = (TValue *)lj_mem_realloc(L, tvref(t->array),
			  oldasize*sizeof(TValue), asize*sizeof(TValue));
    }
    setmref(t->array, array);
    t->asize = asize;
    for (i = oldasize; i < asize; i++)  /* Clear newly allocated slots. */
      setnilV(&array[i]);
  }
  /* Create new (empty) hash part. */
  if (hbits) {
    newhpart(L, t, hbits);
    clearhpart(t);
  } else {
    global_State *g = G(L);
    setmref(t->node, &g->nilnode);
#if LJ_GC64
    setmref(t->freetop, &g->nilnode);
#endif
    t->hmask = 0;
  }
  if (asize < oldasize) {  /* Array part shrinks? */
    TValue *array = tvref(t->array);
    uint32_t i;
    t->asize = asize;  /* Note: This 'shrinks' even colocated arrays. */
    for (i = asize; i < oldasize; i++)  /* Reinsert old array values. */
      if (!tvisnil(&array[i]))
	copyTV(L, lj_tab_setinth(L, t, (int32_t)i), &array[i]);
    /* Physically shrink only separated arrays. */
    if (LJ_MAX_COLOSIZE != 0 && t->colo <= 0)
      setmref(t->array, lj_mem_realloc(L, array,
	      oldasize*sizeof(TValue), asize*sizeof(TValue)));
  }
  if (oldhmask > 0) {  /* Reinsert pairs from old hash part. */
    global_State *g;
    uint32_t i;
    for (i = 0; i <= oldhmask; i++) {
      Node *n = &oldnode[i];
      if (!tvisnil(&n->val))
	copyTV(L, lj_tab_set(L, t, &n->key), &n->val);
    }
    g = G(L);
    lj_mem_freevec(g, oldnode, oldhmask+1, Node);
  }
}

static uint32_t countint(cTValue *key, uint32_t *bins)
{
  lj_assertX(!tvisint(key), "bad integer key");
  if (tvisnum(key)) {
    lua_Number nk = numV(key);
    int32_t k = lj_num2int(nk);
    if ((uint32_t)k < LJ_MAX_ASIZE && nk == (lua_Number)k) {
      bins[(k > 2 ? lj_fls((uint32_t)(k-1)) : 0)]++;
      return 1;
    }
  }
  return 0;
}

static uint32_t countarray(const GCtab *t, uint32_t *bins)
{
  uint32_t na, b, i;
  if (t->asize == 0) return 0;
  for (na = i = b = 0; b < LJ_MAX_ABITS; b++) {
    uint32_t n, top = 2u << b;
    TValue *array;
    if (top >= t->asize) {
      top = t->asize-1;
      if (i > top)
	break;
    }
    array = tvref(t->array);
    for (n = 0; i <= top; i++)
      if (!tvisnil(&array[i]))
	n++;
    bins[b] += n;
    na += n;
  }
  return na;
}

static uint32_t counthash(const GCtab *t, uint32_t *bins, uint32_t *narray)
{
  uint32_t total, na, i, hmask = t->hmask;
  Node *node = noderef(t->node);
  for (total = na = 0, i = 0; i <= hmask; i++) {
    Node *n = &node[i];
    if (!tvisnil(&n->val)) {
      na += countint(&n->key, bins);
      total++;
    }
  }
  *narray += na;
  return total;
}

static uint32_t bestasize(uint32_t bins[], uint32_t *narray)
{
  uint32_t b, sum, na = 0, sz = 0, nn = *narray;
  for (b = 0, sum = 0; 2*nn > (1u<<b) && sum != nn; b++)
    if (bins[b] > 0 && 2*(sum += bins[b]) > (1u<<b)) {
      sz = (2u<<b)+1;
      na = sum;
    }
  *narray = sz;
  return na;
}

static void rehashtab(lua_State *L, GCtab *t, cTValue *ek)
{
  uint32_t bins[LJ_MAX_ABITS];
  uint32_t total, asize, na, i;
  for (i = 0; i < LJ_MAX_ABITS; i++) bins[i] = 0;
  asize = countarray(t, bins);
  total = 1 + asize;
  total += counthash(t, bins, &asize);
  asize += countint(ek, bins);
  na = bestasize(bins, &asize);
  total -= na;
  lj_tab_resize(L, t, asize, hsize2hbits(total));
}

#if LJ_HASFFI
void lj_tab_rehash(lua_State *L, GCtab *t)
{
  rehashtab(L, t, niltv(L));
}
#endif

void lj_tab_reasize(lua_State *L, GCtab *t, uint32_t nasize)
{
  lj_tab_resize(L, t, nasize+1, t->hmask > 0 ? lj_fls(t->hmask)+1 : 0);
}

/* -- Table getters ------------------------------------------------------- */

cTValue * LJ_FASTCALL lj_tab_getinth(GCtab *t, int32_t key)
{
  TValue k;
  Node *n;
  k.n = (lua_Number)key;
  n = hashnum(t, &k);
  do {
    if (tvisnum(&n->key) && n->key.n == k.n)
      return &n->val;
  } while ((n = nextnode(n)));
  return NULL;
}

cTValue *lj_tab_getstr(GCtab *t, const GCstr *key)
{
  Node *n = hashstr(t, key);
  do {
    if (tvisstr(&n->key) && strV(&n->key) == key)
      return &n->val;
  } while ((n = nextnode(n)));
  return NULL;
}

cTValue *lj_tab_get(lua_State *L, GCtab *t, cTValue *key)
{
  if (tvisstr(key)) {
    cTValue *tv = lj_tab_getstr(t, strV(key));
    if (tv)
      return tv;
  } else if (tvisint(key)) {
    cTValue *tv = lj_tab_getint(t, intV(key));
    if (tv)
      return tv;
  } else if (tvisnum(key)) {
    lua_Number nk = numV(key);
    int32_t k = lj_num2int(nk);
    if (nk == (lua_Number)k) {
      cTValue *tv = lj_tab_getint(t, k);
      if (tv)
	return tv;
    } else {
      goto genlookup;  /* Else use the generic lookup. */
    }
  } else if (!tvisnil(key)) {
    Node *n;
  genlookup:
    n = hashkey(t, key);
    do {
      if (lj_obj_equal(&n->key, key))
	return &n->val;
    } while ((n = nextnode(n)));
  }
  return niltv(L);
}

/* -- Table setters ------------------------------------------------------- */

/* Insert new key. Use Brent's variation to optimize the chain length. */
TValue *lj_tab_newkey(lua_State *L, GCtab *t, cTValue *key)
{
  Node *n = hashkey(t, key);
  if (!tvisnil(&n->val) || t->hmask == 0) {
    Node *nodebase = noderef(t->node);
    Node *collide, *freenode = getfreetop(t, nodebase);
    lj_assertL(freenode >= nodebase && freenode <= nodebase+t->hmask+1,
	       "bad freenode");
    do {
      if (freenode == nodebase) {  /* No free node found? */
	rehashtab(L, t, key);  /* Rehash table. */
	return lj_tab_set(L, t, key);  /* Retry key insertion. */
      }
    } while (!tvisnil(&(--freenode)->key));
    setfreetop(t, nodebase, freenode);
    lj_assertL(freenode != &G(L)->nilnode, "store to fallback hash");
    collide = hashkey(t, &n->key);
    if (collide != n) {  /* Colliding node not the main node? */
      while (noderef(collide->next) != n)  /* Find predecessor. */
	collide = nextnode(collide);
      setmref(collide->next, freenode);  /* Relink chain. */
      /* Copy colliding node into free node and free main node. */
      freenode->val = n->val;
      freenode->key = n->key;
      freenode->next = n->next;
      setmref(n->next, NULL);
      setnilV(&n->val);
      /* Rechain pseudo-resurrected string keys with colliding hashes. */
      while (nextnode(freenode)) {
	Node *nn = nextnode(freenode);
	if (!tvisnil(&nn->val) && hashkey(t, &nn->key) == n) {
	  freenode->next = nn->next;
	  nn->next = n->next;
	  setmref(n->next, nn);
	  /*
	  ** Rechaining a resurrected string key creates a new dilemma:
	  ** Another string key may have originally been resurrected via
	  ** _any_ of the previous nodes as a chain anchor. Including
	  ** a node that had to be moved, which makes them unreachable.
	  ** It's not feasible to check for all previous nodes, so rechain
	  ** any string key that's currently in a non-main positions.
	  */
	  while ((nn = nextnode(freenode))) {
	    if (!tvisnil(&nn->val)) {
	      Node *mn = hashkey(t, &nn->key);
	      if (mn != freenode && mn != nn) {
		freenode->next = nn->next;
		nn->next = mn->next;
		setmref(mn->next, nn);
	      } else {
		freenode = nn;
	      }
	    } else {
	      freenode = nn;
	    }
	  }
	  break;
	} else {
	  freenode = nn;
	}
      }
    } else {  /* Otherwise use free node. */
      setmrefr(freenode->next, n->next);  /* Insert into chain. */
      setmref(n->next, freenode);
      n = freenode;
    }
  }
  n->key.u64 = key->u64;
  if (LJ_UNLIKELY(tvismzero(&n->key)))
    n->key.u64 = 0;
  lj_gc_anybarriert(L, t);
  lj_assertL(tvisnil(&n->val), "new hash slot is not empty");
  return &n->val;
}

TValue *lj_tab_setinth(lua_State *L, GCtab *t, int32_t key)
{
  TValue k;
  Node *n;
  k.n = (lua_Number)key;
  n = hashnum(t, &k);
  do {
    if (tvisnum(&n->key) && n->key.n == k.n)
      return &n->val;
  } while ((n = nextnode(n)));
  return lj_tab_newkey(L, t, &k);
}

TValue *lj_tab_setstr(lua_State *L, GCtab *t, const GCstr *key)
{
  TValue k;
  Node *n = hashstr(t, key);
  do {
    if (tvisstr(&n->key) && strV(&n->key) == key)
      return &n->val;
  } while ((n = nextnode(n)));
  setstrV(L, &k, key);
  return lj_tab_newkey(L, t, &k);
}

TValue *lj_tab_set(lua_State *L, GCtab *t, cTValue *key)
{
  Node *n;
  t->nomm = 0;  /* Invalidate negative metamethod cache. */
  if (tvisstr(key)) {
    return lj_tab_setstr(L, t, strV(key));
  } else if (tvisint(key)) {
    return lj_tab_setint(L, t, intV(key));
  } else if (tvisnum(key)) {
    lua_Number nk = numV(key);
    int32_t k = lj_num2int(nk);
    if (nk == (lua_Number)k)
      return lj_tab_setint(L, t, k);
    if (tvisnan(key))
      lj_err_msg(L, LJ_ERR_NANIDX);
    /* Else use the generic lookup. */
  } else if (tvisnil(key)) {
    lj_err_msg(L, LJ_ERR_NILIDX);
  }
  n = hashkey(t, key);
  do {
    if (lj_obj_equal(&n->key, key))
      return &n->val;
  } while ((n = nextnode(n)));
  return lj_tab_newkey(L, t, key);
}

/* -- Table traversal ----------------------------------------------------- */

/* Table traversal indexes:
**
** Array key index: [0 .. t->asize-1]
** Hash key index:  [t->asize .. t->asize+t->hmask]
** Invalid key:     ~0
*/

/* Get the successor traversal index of a key. */
uint32_t LJ_FASTCALL lj_tab_keyindex(GCtab *t, cTValue *key)
{
  TValue tmp;
  if (tvisint(key)) {
    int32_t k = intV(key);
    if ((uint32_t)k < t->asize)
      return (uint32_t)k + 1;
    setnumV(&tmp, (lua_Number)k);
    key = &tmp;
  } else if (tvisnum(key)) {
    lua_Number nk = numV(key);
    int32_t k = lj_num2int(nk);
    if ((uint32_t)k < t->asize && nk == (lua_Number)k)
      return (uint32_t)k + 1;
  }
  if (!tvisnil(key)) {
    Node *n = hashkey(t, key);
    do {
      if (lj_obj_equal(&n->key, key))
	return t->asize + (uint32_t)((n+1) - noderef(t->node));
    } while ((n = nextnode(n)));
    if (key->u32.hi == LJ_KEYINDEX)  /* Despecialized ITERN while running. */
      return key->u32.lo;
    return ~0u;  /* Invalid key to next. */
  }
  return 0;  /* A nil key starts the traversal. */
}

/* Get the next key/value pair of a table traversal. */
int lj_tab_next(GCtab *t, cTValue *key, TValue *o)
{
  uint32_t idx = lj_tab_keyindex(t, key);  /* Find successor index of key. */
  /* First traverse the array part. */
  for (; idx < t->asize; idx++) {
    cTValue *a = arrayslot(t, idx);
    if (LJ_LIKELY(!tvisnil(a))) {
      setintV(o, idx);
      o[1] = *a;
      return 1;
    }
  }
  idx -= t->asize;
  /* Then traverse the hash part. */
  for (; idx <= t->hmask; idx++) {
    Node *n = &noderef(t->node)[idx];
    if (!tvisnil(&n->val)) {
      o[0] = n->key;
      o[1] = n->val;
      return 1;
    }
  }
  return (int32_t)idx < 0 ? -1 : 0;  /* Invalid key or end of traversal. */
}

/* -- Table length calculation -------------------------------------------- */

/* Compute table length. Slow path with mixed array/hash lookups. */
LJ_NOINLINE static MSize tab_len_slow(GCtab *t, size_t hi)
{
  cTValue *tv;
  size_t lo = hi;
  hi++;
  /* Widening search for an upper bound. */
  while ((tv = lj_tab_getint(t, (int32_t)hi)) && !tvisnil(tv)) {
    lo = hi;
    hi += hi;
    if (hi > (size_t)(INT_MAX-2)) {  /* Punt and do a linear search. */
      lo = 1;
      while ((tv = lj_tab_getint(t, (int32_t)lo)) && !tvisnil(tv)) lo++;
      return (MSize)(lo - 1);
    }
  }
  /* Binary search to find a non-nil to nil transition. */
  while (hi - lo > 1) {
    size_t mid = (lo+hi) >> 1;
    cTValue *tvb = lj_tab_getint(t, (int32_t)mid);
    if (tvb && !tvisnil(tvb)) lo = mid; else hi = mid;
  }
  return (MSize)lo;
}

/* Compute table length. Fast path. */
MSize LJ_FASTCALL lj_tab_len(GCtab *t)
{
  size_t hi = (size_t)t->asize;
  if (hi) hi--;
  /* In a growing array the last array element is very likely nil. */
  if (hi > 0 && LJ_LIKELY(tvisnil(arrayslot(t, hi)))) {
    /* Binary search to find a non-nil to nil transition in the array. */
    size_t lo = 0;
    while (hi - lo > 1) {
      size_t mid = (lo+hi) >> 1;
      if (tvisnil(arrayslot(t, mid))) hi = mid; else lo = mid;
    }
    return (MSize)lo;
  }
  /* Without a hash part, there's an implicit nil after the last element. */
  return t->hmask ? tab_len_slow(t, hi) : (MSize)hi;
}

#if LJ_HASJIT
/* Verify hinted table length or compute it. */
MSize LJ_FASTCALL lj_tab_len_hint(GCtab *t, size_t hint)
{
  size_t asize = (size_t)t->asize;
  cTValue *tv = arrayslot(t, hint);
  if (LJ_LIKELY(hint+1 < asize)) {
    if (LJ_LIKELY(!tvisnil(tv) && tvisnil(tv+1))) return (MSize)hint;
  } else if (hint+1 <= asize && LJ_LIKELY(t->hmask == 0) && !tvisnil(tv)) {
    return (MSize)hint;
  }
  return lj_tab_len(t);
}
#endif

