/*
** Library function support.
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
*/

#define lj_lib_c
#define LUA_CORE

#include "lauxlib.h"

#include "lj_obj.h"
#include "lj_gc.h"
#include "lj_err.h"
#include "lj_str.h"
#include "lj_tab.h"
#include "lj_func.h"
#include "lj_bc.h"
#include "lj_dispatch.h"
#if LJ_HASFFI
#include "lj_ctype.h"
#endif
#include "lj_vm.h"
#include "lj_strscan.h"
#include "lj_strfmt.h"
#include "lj_lex.h"
#include "lj_bcdump.h"
#include "lj_lib.h"

/* -- Library initialization ---------------------------------------------- */

static GCtab *lib_create_table(lua_State *L, const char *libname, int hsize)
{
  if (libname) {
    luaL_findtable(L, LUA_REGISTRYINDEX, "_LOADED", 16);
    lua_getfield(L, -1, libname);
    if (!tvistab(L->top-1)) {
      L->top--;
      if (luaL_findtable(L, LUA_GLOBALSINDEX, libname, hsize) != NULL)
	lj_err_callerv(L, LJ_ERR_BADMODN, libname);
      settabV(L, L->top, tabV(L->top-1));
      L->top++;
      lua_setfield(L, -3, libname);  /* _LOADED[libname] = new table */
    }
    L->top--;
    settabV(L, L->top-1, tabV(L->top));
  } else {
    lua_createtable(L, 0, hsize);
  }
  return tabV(L->top-1);
}

static const uint8_t *lib_read_lfunc(lua_State *L, const uint8_t *p, GCtab *tab)
{
  int len = *p++;
  GCstr *name = lj_str_new(L, (const char *)p, len);
  LexState ls;
  GCproto *pt;
  GCfunc *fn;
  memset(&ls, 0, sizeof(ls));
  ls.L = L;
  ls.p = (const char *)(p+len);
  ls.pe = (const char *)~(uintptr_t)0;
  ls.c = -1;
  ls.level = (BCDUMP_F_STRIP|(LJ_BE*BCDUMP_F_BE));
  ls.chunkname = name;
  pt = lj_bcread_proto(&ls);
  pt->firstline = ~(BCLine)0;
  fn = lj_func_newL_empty(L, pt, tabref(L->env));
  /* NOBARRIER: See below for common barrier. */
  setfuncV(L, lj_tab_setstr(L, tab, name), fn);
  return (const uint8_t *)ls.p;
}

void lj_lib_register(lua_State *L, const char *libname,
		     const uint8_t *p, const lua_CFunction *cf)
{
  GCtab *env = tabref(L->env);
  GCfunc *ofn = NULL;
  int ffid = *p++;
  BCIns *bcff = &L2GG(L)->bcff[*p++];
  GCtab *tab = lib_create_table(L, libname, *p++);
  ptrdiff_t tpos = L->top - L->base;

  /* Avoid barriers further down. */
  lj_gc_anybarriert(L, tab);
  tab->nomm = 0;

  for (;;) {
    uint32_t tag = *p++;
    MSize len = tag & LIBINIT_LENMASK;
    tag &= LIBINIT_TAGMASK;
    if (tag != LIBINIT_STRING) {
      const char *name;
      MSize nuv = (MSize)(L->top - L->base - tpos);
      GCfunc *fn = lj_func_newC(L, nuv, env);
      if (nuv) {
	L->top = L->base + tpos;
	memcpy(fn->c.upvalue, L->top, sizeof(TValue)*nuv);
      }
      fn->c.ffid = (uint8_t)(ffid++);
      name = (const char *)p;
      p += len;
      if (tag == LIBINIT_CF)
	setmref(fn->c.pc, &G(L)->bc_cfunc_int);
      else
	setmref(fn->c.pc, bcff++);
      if (tag == LIBINIT_ASM_)
	fn->c.f = ofn->c.f;  /* Copy handler from previous function. */
      else
	fn->c.f = *cf++;  /* Get cf or handler from C function table. */
      if (len) {
	/* NOBARRIER: See above for common barrier. */
	setfuncV(L, lj_tab_setstr(L, tab, lj_str_new(L, name, len)), fn);
      }
      ofn = fn;
    } else {
      switch (tag | len) {
      case LIBINIT_LUA:
	p = lib_read_lfunc(L, p, tab);
	break;
      case LIBINIT_SET:
	L->top -= 2;
	if (tvisstr(L->top+1) && strV(L->top+1)->len == 0)
	  env = tabV(L->top);
	else  /* NOBARRIER: See above for common barrier. */
	  copyTV(L, lj_tab_set(L, tab, L->top+1), L->top);
	break;
      case LIBINIT_NUMBER:
	memcpy(&L->top->n, p, sizeof(double));
	L->top++;
	p += sizeof(double);
	break;
      case LIBINIT_COPY:
	copyTV(L, L->top, L->top - *p++);
	L->top++;
	break;
      case LIBINIT_LASTCL:
	setfuncV(L, L->top++, ofn);
	break;
      case LIBINIT_FFID:
	ffid++;
	break;
      case LIBINIT_END:
	return;
      default:
	setstrV(L, L->top++, lj_str_new(L, (const char *)p, len));
	p += len;
	break;
      }
    }
  }
}

/* Push internal function on the stack. */
GCfunc *lj_lib_pushcc(lua_State *L, lua_CFunction f, int id, int n)
{
  GCfunc *fn;
  lua_pushcclosure(L, f, n);
  fn = funcV(L->top-1);
  fn->c.ffid = (uint8_t)id;
  setmref(fn->c.pc, &G(L)->bc_cfunc_int);
  return fn;
}

void lj_lib_prereg(lua_State *L, const char *name, lua_CFunction f, GCtab *env)
{
  luaL_findtable(L, LUA_REGISTRYINDEX, "_PRELOAD", 4);
  lua_pushcfunction(L, f);
  /* NOBARRIER: The function is new (marked white). */
  setgcref(funcV(L->top-1)->c.env, obj2gco(env));
  lua_setfield(L, -2, name);
  L->top--;
}

int lj_lib_postreg(lua_State *L, lua_CFunction cf, int id, const char *name)
{
  GCfunc *fn = lj_lib_pushcf(L, cf, id);
  GCtab *t = tabref(curr_func(L)->c.env);  /* Reference to parent table. */
  setfuncV(L, lj_tab_setstr(L, t, lj_str_newz(L, name)), fn);
  lj_gc_anybarriert(L, t);
  setfuncV(L, L->top++, fn);
  return 1;
}

/* -- Type checks --------------------------------------------------------- */

TValue *lj_lib_checkany(lua_State *L, int narg)
{
  TValue *o = L->base + narg-1;
  if (o >= L->top)
    lj_err_arg(L, narg, LJ_ERR_NOVAL);
  return o;
}

GCstr *lj_lib_checkstr(lua_State *L, int narg)
{
  TValue *o = L->base + narg-1;
  if (o < L->top) {
    if (LJ_LIKELY(tvisstr(o))) {
      return strV(o);
    } else if (tvisnumber(o)) {
      GCstr *s = lj_strfmt_number(L, o);
      setstrV(L, o, s);
      return s;
    }
  }
  lj_err_argt(L, narg, LUA_TSTRING);
  return NULL;  /* unreachable */
}

GCstr *lj_lib_optstr(lua_State *L, int narg)
{
  TValue *o = L->base + narg-1;
  return (o < L->top && !tvisnil(o)) ? lj_lib_checkstr(L, narg) : NULL;
}

#if LJ_DUALNUM
void lj_lib_checknumber(lua_State *L, int narg)
{
  TValue *o = L->base + narg-1;
  if (!(o < L->top && lj_strscan_numberobj(o)))
    lj_err_argt(L, narg, LUA_TNUMBER);
}
#endif

lua_Number lj_lib_checknum(lua_State *L, int narg)
{
  TValue *o = L->base + narg-1;
  if (!(o < L->top &&
	(tvisnumber(o) || (tvisstr(o) && lj_strscan_num(strV(o), o)))))
    lj_err_argt(L, narg, LUA_TNUMBER);
  if (LJ_UNLIKELY(tvisint(o))) {
    lua_Number n = (lua_Number)intV(o);
    setnumV(o, n);
    return n;
  } else {
    return numV(o);
  }
}

int32_t lj_lib_checkint(lua_State *L, int narg)
{
  TValue *o = L->base + narg-1;
  if (!(o < L->top && lj_strscan_numberobj(o)))
    lj_err_argt(L, narg, LUA_TNUMBER);
  if (LJ_LIKELY(tvisint(o))) {
    return intV(o);
  } else {
    int32_t i = lj_num2int(numV(o));
    if (LJ_DUALNUM) setintV(o, i);
    return i;
  }
}

int32_t lj_lib_optint(lua_State *L, int narg, int32_t def)
{
  TValue *o = L->base + narg-1;
  return (o < L->top && !tvisnil(o)) ? lj_lib_checkint(L, narg) : def;
}

GCfunc *lj_lib_checkfunc(lua_State *L, int narg)
{
  TValue *o = L->base + narg-1;
  if (!(o < L->top && tvisfunc(o)))
    lj_err_argt(L, narg, LUA_TFUNCTION);
  return funcV(o);
}

GCtab *lj_lib_checktab(lua_State *L, int narg)
{
  TValue *o = L->base + narg-1;
  if (!(o < L->top && tvistab(o)))
    lj_err_argt(L, narg, LUA_TTABLE);
  return tabV(o);
}

GCtab *lj_lib_checktabornil(lua_State *L, int narg)
{
  TValue *o = L->base + narg-1;
  if (o < L->top) {
    if (tvistab(o))
      return tabV(o);
    else if (tvisnil(o))
      return NULL;
  }
  lj_err_arg(L, narg, LJ_ERR_NOTABN);
  return NULL;  /* unreachable */
}

int lj_lib_checkopt(lua_State *L, int narg, int def, const char *lst)
{
  GCstr *s = def >= 0 ? lj_lib_optstr(L, narg) : lj_lib_checkstr(L, narg);
  if (s) {
    const char *opt = strdata(s);
    MSize len = s->len;
    int i;
    for (i = 0; *(const uint8_t *)lst; i++) {
      if (*(const uint8_t *)lst == len && memcmp(opt, lst+1, len) == 0)
	return i;
      lst += 1+*(const uint8_t *)lst;
    }
    lj_err_argv(L, narg, LJ_ERR_INVOPTM, opt);
  }
  return def;
}

/* -- Strict type checks -------------------------------------------------- */

/* The following type checks do not coerce between strings and numbers.
** And they handle plain int64_t/uint64_t FFI numbers, too.
*/

#if LJ_HASBUFFER
GCstr *lj_lib_checkstrx(lua_State *L, int narg)
{
  TValue *o = L->base + narg-1;
  if (!(o < L->top && tvisstr(o))) lj_err_argt(L, narg, LUA_TSTRING);
  return strV(o);
}

int32_t lj_lib_checkintrange(lua_State *L, int narg, int32_t a, int32_t b)
{
  TValue *o = L->base + narg-1;
  lj_assertL(b >= 0, "expected range must be non-negative");
  if (o < L->top) {
    if (LJ_LIKELY(tvisint(o))) {
      int32_t i = intV(o);
      if (i >= a && i <= b) return i;
    } else if (LJ_LIKELY(tvisnum(o))) {
      /* For performance reasons, this doesn't check for integerness or
      ** integer overflow. Overflow detection still works, since all FPUs
      ** return either MININT or MAXINT, which is then out of range.
      */
      int32_t i = (int32_t)numV(o);
      if (i >= a && i <= b) return i;
#if LJ_HASFFI
    } else if (tviscdata(o)) {
      GCcdata *cd = cdataV(o);
      if (cd->ctypeid == CTID_INT64) {
	int64_t i = *(int64_t *)cdataptr(cd);
	if (i >= (int64_t)a && i <= (int64_t)b) return (int32_t)i;
      } else if (cd->ctypeid == CTID_UINT64) {
	uint64_t i = *(uint64_t *)cdataptr(cd);
	if ((a < 0 || i >= (uint64_t)a) && i <= (uint64_t)b) return (int32_t)i;
      } else {
	goto badtype;
      }
#endif
    } else {
      goto badtype;
    }
    lj_err_arg(L, narg, LJ_ERR_NUMRNG);
  }
badtype:
  lj_err_argt(L, narg, LUA_TNUMBER);
  return 0;  /* unreachable */
}
#endif

