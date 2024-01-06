/*
** Table library.
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
**
** Major portions taken verbatim or adapted from the Lua interpreter.
** Copyright (C) 1994-2008 Lua.org, PUC-Rio. See Copyright Notice in lua.h
*/

#define lib_table_c
#define LUA_LIB

#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"

#include "lj_obj.h"
#include "lj_gc.h"
#include "lj_err.h"
#include "lj_buf.h"
#include "lj_tab.h"
#include "lj_ff.h"
#include "lj_lib.h"

/* ------------------------------------------------------------------------ */

#define LJLIB_MODULE_table

LJLIB_LUA(table_foreachi) /*
  function(t, f)
    CHECK_tab(t)
    CHECK_func(f)
    for i=1,#t do
      local r = f(i, t[i])
      if r ~= nil then return r end
    end
  end
*/

LJLIB_LUA(table_foreach) /*
  function(t, f)
    CHECK_tab(t)
    CHECK_func(f)
    for k, v in PAIRS(t) do
      local r = f(k, v)
      if r ~= nil then return r end
    end
  end
*/

LJLIB_LUA(table_getn) /*
  function(t)
    CHECK_tab(t)
    return #t
  end
*/

LJLIB_CF(table_maxn)
{
  GCtab *t = lj_lib_checktab(L, 1);
  TValue *array = tvref(t->array);
  Node *node;
  lua_Number m = 0;
  ptrdiff_t i;
  for (i = (ptrdiff_t)t->asize - 1; i >= 0; i--)
    if (!tvisnil(&array[i])) {
      m = (lua_Number)(int32_t)i;
      break;
    }
  node = noderef(t->node);
  for (i = (ptrdiff_t)t->hmask; i >= 0; i--)
    if (!tvisnil(&node[i].val) && tvisnumber(&node[i].key)) {
      lua_Number n = numberVnum(&node[i].key);
      if (n > m) m = n;
    }
  setnumV(L->top-1, m);
  return 1;
}

LJLIB_CF(table_insert)		LJLIB_REC(.)
{
  GCtab *t = lj_lib_checktab(L, 1);
  int32_t n, i = (int32_t)lj_tab_len(t) + 1;
  int nargs = (int)((char *)L->top - (char *)L->base);
  if (nargs != 2*sizeof(TValue)) {
    if (nargs != 3*sizeof(TValue))
      lj_err_caller(L, LJ_ERR_TABINS);
    /* NOBARRIER: This just moves existing elements around. */
    for (n = lj_lib_checkint(L, 2); i > n; i--) {
      /* The set may invalidate the get pointer, so need to do it first! */
      TValue *dst = lj_tab_setint(L, t, i);
      cTValue *src = lj_tab_getint(t, i-1);
      if (src) {
	copyTV(L, dst, src);
      } else {
	setnilV(dst);
      }
    }
    i = n;
  }
  {
    TValue *dst = lj_tab_setint(L, t, i);
    copyTV(L, dst, L->top-1);  /* Set new value. */
    lj_gc_barriert(L, t, dst);
  }
  return 0;
}

LJLIB_LUA(table_remove) /*
  function(t, pos)
    CHECK_tab(t)
    local len = #t
    if pos == nil then
      if len ~= 0 then
	local old = t[len]
	t[len] = nil
	return old
      end
    else
      CHECK_int(pos)
      if pos >= 1 and pos <= len then
	local old = t[pos]
	for i=pos+1,len do
	  t[i-1] = t[i]
	end
	t[len] = nil
	return old
      end
    end
  end
*/

LJLIB_LUA(table_move) /*
  function(a1, f, e, t, a2)
    CHECK_tab(a1)
    CHECK_int(f)
    CHECK_int(e)
    CHECK_int(t)
    if a2 == nil then a2 = a1 end
    CHECK_tab(a2)
    if e >= f then
      local d = t - f
      if t > e or t <= f or a2 ~= a1 then
	for i=f,e do a2[i+d] = a1[i] end
      else
	for i=e,f,-1 do a2[i+d] = a1[i] end
      end
    end
    return a2
  end
*/

LJLIB_CF(table_concat)		LJLIB_REC(.)
{
  GCtab *t = lj_lib_checktab(L, 1);
  GCstr *sep = lj_lib_optstr(L, 2);
  int32_t i = lj_lib_optint(L, 3, 1);
  int32_t e = (L->base+3 < L->top && !tvisnil(L->base+3)) ?
	      lj_lib_checkint(L, 4) : (int32_t)lj_tab_len(t);
  SBuf *sb = lj_buf_tmp_(L);
  SBuf *sbx = lj_buf_puttab(sb, t, sep, i, e);
  if (LJ_UNLIKELY(!sbx)) {  /* Error: bad element type. */
    int32_t idx = (int32_t)(intptr_t)sb->w;
    cTValue *o = lj_tab_getint(t, idx);
    lj_err_callerv(L, LJ_ERR_TABCAT,
		   lj_obj_itypename[o ? itypemap(o) : ~LJ_TNIL], idx);
  }
  setstrV(L, L->top-1, lj_buf_str(L, sbx));
  lj_gc_check(L);
  return 1;
}

/* ------------------------------------------------------------------------ */

static void set2(lua_State *L, int i, int j)
{
  lua_rawseti(L, 1, i);
  lua_rawseti(L, 1, j);
}

static int sort_comp(lua_State *L, int a, int b)
{
  if (!lua_isnil(L, 2)) {  /* function? */
    int res;
    lua_pushvalue(L, 2);
    lua_pushvalue(L, a-1);  /* -1 to compensate function */
    lua_pushvalue(L, b-2);  /* -2 to compensate function and `a' */
    lua_call(L, 2, 1);
    res = lua_toboolean(L, -1);
    lua_pop(L, 1);
    return res;
  } else {  /* a < b? */
    return lua_lessthan(L, a, b);
  }
}

static void auxsort(lua_State *L, int l, int u)
{
  while (l < u) {  /* for tail recursion */
    int i, j;
    /* sort elements a[l], a[(l+u)/2] and a[u] */
    lua_rawgeti(L, 1, l);
    lua_rawgeti(L, 1, u);
    if (sort_comp(L, -1, -2))  /* a[u] < a[l]? */
      set2(L, l, u);  /* swap a[l] - a[u] */
    else
      lua_pop(L, 2);
    if (u-l == 1) break;  /* only 2 elements */
    i = (l+u)/2;
    lua_rawgeti(L, 1, i);
    lua_rawgeti(L, 1, l);
    if (sort_comp(L, -2, -1)) {  /* a[i]<a[l]? */
      set2(L, i, l);
    } else {
      lua_pop(L, 1);  /* remove a[l] */
      lua_rawgeti(L, 1, u);
      if (sort_comp(L, -1, -2))  /* a[u]<a[i]? */
	set2(L, i, u);
      else
	lua_pop(L, 2);
    }
    if (u-l == 2) break;  /* only 3 elements */
    lua_rawgeti(L, 1, i);  /* Pivot */
    lua_pushvalue(L, -1);
    lua_rawgeti(L, 1, u-1);
    set2(L, i, u-1);
    /* a[l] <= P == a[u-1] <= a[u], only need to sort from l+1 to u-2 */
    i = l; j = u-1;
    for (;;) {  /* invariant: a[l..i] <= P <= a[j..u] */
      /* repeat ++i until a[i] >= P */
      while (lua_rawgeti(L, 1, ++i), sort_comp(L, -1, -2)) {
	if (i>=u) lj_err_caller(L, LJ_ERR_TABSORT);
	lua_pop(L, 1);  /* remove a[i] */
      }
      /* repeat --j until a[j] <= P */
      while (lua_rawgeti(L, 1, --j), sort_comp(L, -3, -1)) {
	if (j<=l) lj_err_caller(L, LJ_ERR_TABSORT);
	lua_pop(L, 1);  /* remove a[j] */
      }
      if (j<i) {
	lua_pop(L, 3);  /* pop pivot, a[i], a[j] */
	break;
      }
      set2(L, i, j);
    }
    lua_rawgeti(L, 1, u-1);
    lua_rawgeti(L, 1, i);
    set2(L, u-1, i);  /* swap pivot (a[u-1]) with a[i] */
    /* a[l..i-1] <= a[i] == P <= a[i+1..u] */
    /* adjust so that smaller half is in [j..i] and larger one in [l..u] */
    if (i-l < u-i) {
      j=l; i=i-1; l=i+2;
    } else {
      j=i+1; i=u; u=j-2;
    }
    auxsort(L, j, i);  /* call recursively the smaller one */
  }  /* repeat the routine for the larger one */
}

LJLIB_CF(table_sort)
{
  GCtab *t = lj_lib_checktab(L, 1);
  int32_t n = (int32_t)lj_tab_len(t);
  lua_settop(L, 2);
  if (!tvisnil(L->base+1))
    lj_lib_checkfunc(L, 2);
  auxsort(L, 1, n);
  return 0;
}

#if LJ_52
LJLIB_PUSH("n")
LJLIB_CF(table_pack)
{
  TValue *array, *base = L->base;
  MSize i, n = (uint32_t)(L->top - base);
  GCtab *t = lj_tab_new(L, n ? n+1 : 0, 1);
  /* NOBARRIER: The table is new (marked white). */
  setintV(lj_tab_setstr(L, t, strV(lj_lib_upvalue(L, 1))), (int32_t)n);
  for (array = tvref(t->array) + 1, i = 0; i < n; i++)
    copyTV(L, &array[i], &base[i]);
  settabV(L, base, t);
  L->top = base+1;
  lj_gc_check(L);
  return 1;
}
#endif

LJLIB_NOREG LJLIB_CF(table_new)		LJLIB_REC(.)
{
  int32_t a = lj_lib_checkint(L, 1);
  int32_t h = lj_lib_checkint(L, 2);
  lua_createtable(L, a, h);
  return 1;
}

LJLIB_NOREG LJLIB_CF(table_clear)	LJLIB_REC(.)
{
  lj_tab_clear(lj_lib_checktab(L, 1));
  return 0;
}

static int luaopen_table_new(lua_State *L)
{
  return lj_lib_postreg(L, lj_cf_table_new, FF_table_new, "new");
}

static int luaopen_table_clear(lua_State *L)
{
  return lj_lib_postreg(L, lj_cf_table_clear, FF_table_clear, "clear");
}

/* ------------------------------------------------------------------------ */

#include "lj_libdef.h"

LUALIB_API int luaopen_table(lua_State *L)
{
  LJ_LIB_REG(L, LUA_TABLIBNAME, table);
#if LJ_52
  lua_getglobal(L, "unpack");
  lua_setfield(L, -2, "unpack");
#endif
  lj_lib_prereg(L, LUA_TABLIBNAME ".new", luaopen_table_new, tabV(L->top-1));
  lj_lib_prereg(L, LUA_TABLIBNAME ".clear", luaopen_table_clear, tabV(L->top-1));
  return 1;
}

