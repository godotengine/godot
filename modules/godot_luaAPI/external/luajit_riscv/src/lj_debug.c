/*
** Debugging and introspection.
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
*/

#define lj_debug_c
#define LUA_CORE

#include "lj_obj.h"
#include "lj_err.h"
#include "lj_debug.h"
#include "lj_buf.h"
#include "lj_tab.h"
#include "lj_state.h"
#include "lj_frame.h"
#include "lj_bc.h"
#include "lj_strfmt.h"
#if LJ_HASJIT
#include "lj_jit.h"
#endif

/* -- Frames -------------------------------------------------------------- */

/* Get frame corresponding to a level. */
cTValue *lj_debug_frame(lua_State *L, int level, int *size)
{
  cTValue *frame, *nextframe, *bot = tvref(L->stack)+LJ_FR2;
  /* Traverse frames backwards. */
  for (nextframe = frame = L->base-1; frame > bot; ) {
    if (frame_gc(frame) == obj2gco(L))
      level++;  /* Skip dummy frames. See lj_err_optype_call(). */
    if (level-- == 0) {
      *size = (int)(nextframe - frame);
      return frame;  /* Level found. */
    }
    nextframe = frame;
    if (frame_islua(frame)) {
      frame = frame_prevl(frame);
    } else {
      if (frame_isvarg(frame))
	level++;  /* Skip vararg pseudo-frame. */
      frame = frame_prevd(frame);
    }
  }
  *size = level;
  return NULL;  /* Level not found. */
}

/* Invalid bytecode position. */
#define NO_BCPOS	(~(BCPos)0)

/* Return bytecode position for function/frame or NO_BCPOS. */
static BCPos debug_framepc(lua_State *L, GCfunc *fn, cTValue *nextframe)
{
  const BCIns *ins;
  GCproto *pt;
  BCPos pos;
  lj_assertL(fn->c.gct == ~LJ_TFUNC || fn->c.gct == ~LJ_TTHREAD,
	     "function or frame expected");
  if (!isluafunc(fn)) {  /* Cannot derive a PC for non-Lua functions. */
    return NO_BCPOS;
  } else if (nextframe == NULL) {  /* Lua function on top. */
    void *cf = cframe_raw(L->cframe);
    if (cf == NULL || (char *)cframe_pc(cf) == (char *)cframe_L(cf))
      return NO_BCPOS;
    ins = cframe_pc(cf);  /* Only happens during error/hook handling. */
  } else {
    if (frame_islua(nextframe)) {
      ins = frame_pc(nextframe);
    } else if (frame_iscont(nextframe)) {
      ins = frame_contpc(nextframe);
    } else {
      /* Lua function below errfunc/gc/hook: find cframe to get the PC. */
      void *cf = cframe_raw(L->cframe);
      TValue *f = L->base-1;
      for (;;) {
	if (cf == NULL)
	  return NO_BCPOS;
	while (cframe_nres(cf) < 0) {
	  if (f >= restorestack(L, -cframe_nres(cf)))
	    break;
	  cf = cframe_raw(cframe_prev(cf));
	  if (cf == NULL)
	    return NO_BCPOS;
	}
	if (f < nextframe)
	  break;
	if (frame_islua(f)) {
	  f = frame_prevl(f);
	} else {
	  if (frame_isc(f) || (frame_iscont(f) && frame_iscont_fficb(f)))
	    cf = cframe_raw(cframe_prev(cf));
	  f = frame_prevd(f);
	}
      }
      ins = cframe_pc(cf);
      if (!ins) return NO_BCPOS;
    }
  }
  pt = funcproto(fn);
  pos = proto_bcpos(pt, ins) - 1;
#if LJ_HASJIT
  if (pos > pt->sizebc) {  /* Undo the effects of lj_trace_exit for JLOOP. */
    if (bc_isret(bc_op(ins[-1]))) {
      GCtrace *T = (GCtrace *)((char *)(ins-1) - offsetof(GCtrace, startins));
      pos = proto_bcpos(pt, mref(T->startpc, const BCIns));
    } else {
      pos = NO_BCPOS;  /* Punt in case of stack overflow for stitched trace. */
    }
  }
#endif
  return pos;
}

/* -- Line numbers -------------------------------------------------------- */

/* Get line number for a bytecode position. */
BCLine LJ_FASTCALL lj_debug_line(GCproto *pt, BCPos pc)
{
  const void *lineinfo = proto_lineinfo(pt);
  if (pc <= pt->sizebc && lineinfo) {
    BCLine first = pt->firstline;
    if (pc == pt->sizebc) return first + pt->numline;
    if (pc-- == 0) return first;
    if (pt->numline < 256)
      return first + (BCLine)((const uint8_t *)lineinfo)[pc];
    else if (pt->numline < 65536)
      return first + (BCLine)((const uint16_t *)lineinfo)[pc];
    else
      return first + (BCLine)((const uint32_t *)lineinfo)[pc];
  }
  return 0;
}

/* Get line number for function/frame. */
static BCLine debug_frameline(lua_State *L, GCfunc *fn, cTValue *nextframe)
{
  BCPos pc = debug_framepc(L, fn, nextframe);
  if (pc != NO_BCPOS) {
    GCproto *pt = funcproto(fn);
    lj_assertL(pc <= pt->sizebc, "PC out of range");
    return lj_debug_line(pt, pc);
  }
  return -1;
}

/* -- Variable names ------------------------------------------------------ */

/* Get name of a local variable from slot number and PC. */
static const char *debug_varname(const GCproto *pt, BCPos pc, BCReg slot)
{
  const char *p = (const char *)proto_varinfo(pt);
  if (p) {
    BCPos lastpc = 0;
    for (;;) {
      const char *name = p;
      uint32_t vn = *(const uint8_t *)p;
      BCPos startpc, endpc;
      if (vn < VARNAME__MAX) {
	if (vn == VARNAME_END) break;  /* End of varinfo. */
      } else {
	do { p++; } while (*(const uint8_t *)p);  /* Skip over variable name. */
      }
      p++;
      lastpc = startpc = lastpc + lj_buf_ruleb128(&p);
      if (startpc > pc) break;
      endpc = startpc + lj_buf_ruleb128(&p);
      if (pc < endpc && slot-- == 0) {
	if (vn < VARNAME__MAX) {
#define VARNAMESTR(name, str)	str "\0"
	  name = VARNAMEDEF(VARNAMESTR);
#undef VARNAMESTR
	  if (--vn) while (*name++ || --vn) ;
	}
	return name;
      }
    }
  }
  return NULL;
}

/* Get name of local variable from 1-based slot number and function/frame. */
static TValue *debug_localname(lua_State *L, const lua_Debug *ar,
			       const char **name, BCReg slot1)
{
  uint32_t offset = (uint32_t)ar->i_ci & 0xffff;
  uint32_t size = (uint32_t)ar->i_ci >> 16;
  TValue *frame = tvref(L->stack) + offset;
  TValue *nextframe = size ? frame + size : NULL;
  GCfunc *fn = frame_func(frame);
  BCPos pc = debug_framepc(L, fn, nextframe);
  if (!nextframe) nextframe = L->top+LJ_FR2;
  if ((int)slot1 < 0) {  /* Negative slot number is for varargs. */
    if (pc != NO_BCPOS) {
      GCproto *pt = funcproto(fn);
      if ((pt->flags & PROTO_VARARG)) {
	slot1 = pt->numparams + (BCReg)(-(int)slot1);
	if (frame_isvarg(frame)) {  /* Vararg frame has been set up? (pc!=0) */
	  nextframe = frame;
	  frame = frame_prevd(frame);
	}
	if (frame + slot1+LJ_FR2 < nextframe) {
	  *name = "(*vararg)";
	  return frame+slot1;
	}
      }
    }
    return NULL;
  }
  if (pc != NO_BCPOS &&
      (*name = debug_varname(funcproto(fn), pc, slot1-1)) != NULL)
    ;
  else if (slot1 > 0 && frame + slot1+LJ_FR2 < nextframe)
    *name = "(*temporary)";
  return frame+slot1;
}

/* Get name of upvalue. */
const char *lj_debug_uvname(GCproto *pt, uint32_t idx)
{
  const uint8_t *p = proto_uvinfo(pt);
  lj_assertX(idx < pt->sizeuv, "bad upvalue index");
  if (!p) return "";
  if (idx) while (*p++ || --idx) ;
  return (const char *)p;
}

/* Get name and value of upvalue. */
const char *lj_debug_uvnamev(cTValue *o, uint32_t idx, TValue **tvp, GCobj **op)
{
  if (tvisfunc(o)) {
    GCfunc *fn = funcV(o);
    if (isluafunc(fn)) {
      GCproto *pt = funcproto(fn);
      if (idx < pt->sizeuv) {
	GCobj *uvo = gcref(fn->l.uvptr[idx]);
	*tvp = uvval(&uvo->uv);
	*op = uvo;
	return lj_debug_uvname(pt, idx);
      }
    } else {
      if (idx < fn->c.nupvalues) {
	*tvp = &fn->c.upvalue[idx];
	*op = obj2gco(fn);
	return "";
      }
    }
  }
  return NULL;
}

/* Deduce name of an object from slot number and PC. */
const char *lj_debug_slotname(GCproto *pt, const BCIns *ip, BCReg slot,
			      const char **name)
{
  const char *lname;
restart:
  lname = debug_varname(pt, proto_bcpos(pt, ip), slot);
  if (lname != NULL) { *name = lname; return "local"; }
  while (--ip > proto_bc(pt)) {
    BCIns ins = *ip;
    BCOp op = bc_op(ins);
    BCReg ra = bc_a(ins);
    if (bcmode_a(op) == BCMbase) {
      if (slot >= ra && (op != BC_KNIL || slot <= bc_d(ins)))
	return NULL;
    } else if (bcmode_a(op) == BCMdst && ra == slot) {
      switch (bc_op(ins)) {
      case BC_MOV:
	if (ra == slot) { slot = bc_d(ins); goto restart; }
	break;
      case BC_GGET:
	*name = strdata(gco2str(proto_kgc(pt, ~(ptrdiff_t)bc_d(ins))));
	return "global";
      case BC_TGETS:
	*name = strdata(gco2str(proto_kgc(pt, ~(ptrdiff_t)bc_c(ins))));
	if (ip > proto_bc(pt)) {
	  BCIns insp = ip[-1];
	  if (bc_op(insp) == BC_MOV && bc_a(insp) == ra+1+LJ_FR2 &&
	      bc_d(insp) == bc_b(ins))
	    return "method";
	}
	return "field";
      case BC_UGET:
	*name = lj_debug_uvname(pt, bc_d(ins));
	return "upvalue";
      default:
	return NULL;
      }
    }
  }
  return NULL;
}

/* Deduce function name from caller of a frame. */
const char *lj_debug_funcname(lua_State *L, cTValue *frame, const char **name)
{
  cTValue *pframe;
  GCfunc *fn;
  BCPos pc;
  if (frame <= tvref(L->stack)+LJ_FR2)
    return NULL;
  if (frame_isvarg(frame))
    frame = frame_prevd(frame);
  pframe = frame_prev(frame);
  fn = frame_func(pframe);
  pc = debug_framepc(L, fn, frame);
  if (pc != NO_BCPOS) {
    GCproto *pt = funcproto(fn);
    const BCIns *ip = &proto_bc(pt)[check_exp(pc < pt->sizebc, pc)];
    MMS mm = bcmode_mm(bc_op(*ip));
    if (mm == MM_call) {
      BCReg slot = bc_a(*ip);
      if (bc_op(*ip) == BC_ITERC) slot -= 3;
      return lj_debug_slotname(pt, ip, slot, name);
    } else if (mm != MM__MAX) {
      *name = strdata(mmname_str(G(L), mm));
      return "metamethod";
    }
  }
  return NULL;
}

/* -- Source code locations ----------------------------------------------- */

/* Generate shortened source name. */
void lj_debug_shortname(char *out, GCstr *str, BCLine line)
{
  const char *src = strdata(str);
  if (*src == '=') {
    strncpy(out, src+1, LUA_IDSIZE);  /* Remove first char. */
    out[LUA_IDSIZE-1] = '\0';  /* Ensures null termination. */
  } else if (*src == '@') {  /* Output "source", or "...source". */
    size_t len = str->len-1;
    src++;  /* Skip the `@' */
    if (len >= LUA_IDSIZE) {
      src += len-(LUA_IDSIZE-4);  /* Get last part of file name. */
      *out++ = '.'; *out++ = '.'; *out++ = '.';
    }
    strcpy(out, src);
  } else {  /* Output [string "string"] or [builtin:name]. */
    size_t len;  /* Length, up to first control char. */
    for (len = 0; len < LUA_IDSIZE-12; len++)
      if (((const unsigned char *)src)[len] < ' ') break;
    strcpy(out, line == ~(BCLine)0 ? "[builtin:" : "[string \""); out += 9;
    if (src[len] != '\0') {  /* Must truncate? */
      if (len > LUA_IDSIZE-15) len = LUA_IDSIZE-15;
      strncpy(out, src, len); out += len;
      strcpy(out, "..."); out += 3;
    } else {
      strcpy(out, src); out += len;
    }
    strcpy(out, line == ~(BCLine)0 ? "]" : "\"]");
  }
}

/* Add current location of a frame to error message. */
void lj_debug_addloc(lua_State *L, const char *msg,
		     cTValue *frame, cTValue *nextframe)
{
  if (frame) {
    GCfunc *fn = frame_func(frame);
    if (isluafunc(fn)) {
      BCLine line = debug_frameline(L, fn, nextframe);
      if (line >= 0) {
	GCproto *pt = funcproto(fn);
	char buf[LUA_IDSIZE];
	lj_debug_shortname(buf, proto_chunkname(pt), pt->firstline);
	lj_strfmt_pushf(L, "%s:%d: %s", buf, line, msg);
	return;
      }
    }
  }
  lj_strfmt_pushf(L, "%s", msg);
}

/* Push location string for a bytecode position to Lua stack. */
void lj_debug_pushloc(lua_State *L, GCproto *pt, BCPos pc)
{
  GCstr *name = proto_chunkname(pt);
  const char *s = strdata(name);
  MSize i, len = name->len;
  BCLine line = lj_debug_line(pt, pc);
  if (pt->firstline == ~(BCLine)0) {
    lj_strfmt_pushf(L, "builtin:%s", s);
  } else if (*s == '@') {
    s++; len--;
    for (i = len; i > 0; i--)
      if (s[i] == '/' || s[i] == '\\') {
	s += i+1;
	break;
      }
    lj_strfmt_pushf(L, "%s:%d", s, line);
  } else if (len > 40) {
    lj_strfmt_pushf(L, "%p:%d", pt, line);
  } else if (*s == '=') {
    lj_strfmt_pushf(L, "%s:%d", s+1, line);
  } else {
    lj_strfmt_pushf(L, "\"%s\":%d", s, line);
  }
}

/* -- Public debug API ---------------------------------------------------- */

/* lua_getupvalue() and lua_setupvalue() are in lj_api.c. */

LUA_API const char *lua_getlocal(lua_State *L, const lua_Debug *ar, int n)
{
  const char *name = NULL;
  if (ar) {
    TValue *o = debug_localname(L, ar, &name, (BCReg)n);
    if (name) {
      copyTV(L, L->top, o);
      incr_top(L);
    }
  } else if (tvisfunc(L->top-1) && isluafunc(funcV(L->top-1))) {
    name = debug_varname(funcproto(funcV(L->top-1)), 0, (BCReg)n-1);
  }
  return name;
}

LUA_API const char *lua_setlocal(lua_State *L, const lua_Debug *ar, int n)
{
  const char *name = NULL;
  TValue *o = debug_localname(L, ar, &name, (BCReg)n);
  if (name)
    copyTV(L, o, L->top-1);
  L->top--;
  return name;
}

int lj_debug_getinfo(lua_State *L, const char *what, lj_Debug *ar, int ext)
{
  int opt_f = 0, opt_L = 0;
  TValue *frame = NULL;
  TValue *nextframe = NULL;
  GCfunc *fn;
  if (*what == '>') {
    TValue *func = L->top - 1;
    if (!tvisfunc(func)) return 0;
    fn = funcV(func);
    L->top--;
    what++;
  } else {
    uint32_t offset = (uint32_t)ar->i_ci & 0xffff;
    uint32_t size = (uint32_t)ar->i_ci >> 16;
    lj_assertL(offset != 0, "bad frame offset");
    frame = tvref(L->stack) + offset;
    if (size) nextframe = frame + size;
    lj_assertL(frame <= tvref(L->maxstack) &&
	       (!nextframe || nextframe <= tvref(L->maxstack)),
	       "broken frame chain");
    fn = frame_func(frame);
    lj_assertL(fn->c.gct == ~LJ_TFUNC, "bad frame function");
  }
  for (; *what; what++) {
    if (*what == 'S') {
      if (isluafunc(fn)) {
	GCproto *pt = funcproto(fn);
	BCLine firstline = pt->firstline;
	GCstr *name = proto_chunkname(pt);
	ar->source = strdata(name);
	lj_debug_shortname(ar->short_src, name, pt->firstline);
	ar->linedefined = (int)firstline;
	ar->lastlinedefined = (int)(firstline + pt->numline);
	ar->what = (firstline || !pt->numline) ? "Lua" : "main";
      } else {
	ar->source = "=[C]";
	ar->short_src[0] = '[';
	ar->short_src[1] = 'C';
	ar->short_src[2] = ']';
	ar->short_src[3] = '\0';
	ar->linedefined = -1;
	ar->lastlinedefined = -1;
	ar->what = "C";
      }
    } else if (*what == 'l') {
      ar->currentline = frame ? debug_frameline(L, fn, nextframe) : -1;
    } else if (*what == 'u') {
      ar->nups = fn->c.nupvalues;
      if (ext) {
	if (isluafunc(fn)) {
	  GCproto *pt = funcproto(fn);
	  ar->nparams = pt->numparams;
	  ar->isvararg = !!(pt->flags & PROTO_VARARG);
	} else {
	  ar->nparams = 0;
	  ar->isvararg = 1;
	}
      }
    } else if (*what == 'n') {
      ar->namewhat = frame ? lj_debug_funcname(L, frame, &ar->name) : NULL;
      if (ar->namewhat == NULL) {
	ar->namewhat = "";
	ar->name = NULL;
      }
    } else if (*what == 'f') {
      opt_f = 1;
    } else if (*what == 'L') {
      opt_L = 1;
    } else {
      return 0;  /* Bad option. */
    }
  }
  if (opt_f) {
    setfuncV(L, L->top, fn);
    incr_top(L);
  }
  if (opt_L) {
    if (isluafunc(fn)) {
      GCtab *t = lj_tab_new(L, 0, 0);
      GCproto *pt = funcproto(fn);
      const void *lineinfo = proto_lineinfo(pt);
      if (lineinfo) {
	BCLine first = pt->firstline;
	int sz = pt->numline < 256 ? 1 : pt->numline < 65536 ? 2 : 4;
	MSize i, szl = pt->sizebc-1;
	for (i = 0; i < szl; i++) {
	  BCLine line = first +
	    (sz == 1 ? (BCLine)((const uint8_t *)lineinfo)[i] :
	     sz == 2 ? (BCLine)((const uint16_t *)lineinfo)[i] :
	     (BCLine)((const uint32_t *)lineinfo)[i]);
	  setboolV(lj_tab_setint(L, t, line), 1);
	}
      }
      settabV(L, L->top, t);
    } else {
      setnilV(L->top);
    }
    incr_top(L);
  }
  return 1;  /* Ok. */
}

LUA_API int lua_getinfo(lua_State *L, const char *what, lua_Debug *ar)
{
  return lj_debug_getinfo(L, what, (lj_Debug *)ar, 0);
}

LUA_API int lua_getstack(lua_State *L, int level, lua_Debug *ar)
{
  int size;
  cTValue *frame = lj_debug_frame(L, level, &size);
  if (frame) {
    ar->i_ci = (size << 16) + (int)(frame - tvref(L->stack));
    return 1;
  } else {
    ar->i_ci = level - size;
    return 0;
  }
}

#if LJ_HASPROFILE
/* Put the chunkname into a buffer. */
static int debug_putchunkname(SBuf *sb, GCproto *pt, int pathstrip)
{
  GCstr *name = proto_chunkname(pt);
  const char *p = strdata(name);
  if (pt->firstline == ~(BCLine)0) {
    lj_buf_putmem(sb, "[builtin:", 9);
    lj_buf_putstr(sb, name);
    lj_buf_putb(sb, ']');
    return 0;
  }
  if (*p == '=' || *p == '@') {
    MSize len = name->len-1;
    p++;
    if (pathstrip) {
      int i;
      for (i = len-1; i >= 0; i--)
	if (p[i] == '/' || p[i] == '\\') {
	  len -= i+1;
	  p = p+i+1;
	  break;
	}
    }
    lj_buf_putmem(sb, p, len);
  } else {
    lj_buf_putmem(sb, "[string]", 8);
  }
  return 1;
}

/* Put a compact stack dump into a buffer. */
void lj_debug_dumpstack(lua_State *L, SBuf *sb, const char *fmt, int depth)
{
  int level = 0, dir = 1, pathstrip = 1;
  MSize lastlen = 0;
  if (depth < 0) { level = ~depth; depth = dir = -1; }  /* Reverse frames. */
  while (level != depth) {  /* Loop through all frame. */
    int size;
    cTValue *frame = lj_debug_frame(L, level, &size);
    if (frame) {
      cTValue *nextframe = size ? frame+size : NULL;
      GCfunc *fn = frame_func(frame);
      const uint8_t *p = (const uint8_t *)fmt;
      int c;
      while ((c = *p++)) {
	switch (c) {
	case 'p':  /* Preserve full path. */
	  pathstrip = 0;
	  break;
	case 'F': case 'f': {  /* Dump function name. */
	  const char *name;
	  const char *what = lj_debug_funcname(L, frame, &name);
	  if (what) {
	    if (c == 'F' && isluafunc(fn)) {  /* Dump module:name for 'F'. */
	      GCproto *pt = funcproto(fn);
	      if (pt->firstline != ~(BCLine)0) {  /* Not a bytecode builtin. */
		debug_putchunkname(sb, pt, pathstrip);
		lj_buf_putb(sb, ':');
	      }
	    }
	    lj_buf_putmem(sb, name, (MSize)strlen(name));
	    break;
	  }  /* else: can't derive a name, dump module:line. */
	  }
	  /* fallthrough */
	case 'l':  /* Dump module:line. */
	  if (isluafunc(fn)) {
	    GCproto *pt = funcproto(fn);
	    if (debug_putchunkname(sb, pt, pathstrip)) {
	      /* Regular Lua function. */
	      BCLine line = c == 'l' ? debug_frameline(L, fn, nextframe) :
				       pt->firstline;
	      lj_buf_putb(sb, ':');
	      lj_strfmt_putint(sb, line >= 0 ? line : pt->firstline);
	    }
	  } else if (isffunc(fn)) {  /* Dump numbered builtins. */
	    lj_buf_putmem(sb, "[builtin#", 9);
	    lj_strfmt_putint(sb, fn->c.ffid);
	    lj_buf_putb(sb, ']');
	  } else {  /* Dump C function address. */
	    lj_buf_putb(sb, '@');
	    lj_strfmt_putptr(sb, fn->c.f);
	  }
	  break;
	case 'Z':  /* Zap trailing separator. */
	  lastlen = sbuflen(sb);
	  break;
	default:
	  lj_buf_putb(sb, c);
	  break;
	}
      }
    } else if (dir == 1) {
      break;
    } else {
      level -= size;  /* Reverse frame order: quickly skip missing level. */
    }
    level += dir;
  }
  if (lastlen)
    sb->w = sb->b + lastlen;  /* Zap trailing separator. */
}
#endif

/* Number of frames for the leading and trailing part of a traceback. */
#define TRACEBACK_LEVELS1	12
#define TRACEBACK_LEVELS2	10

LUALIB_API void luaL_traceback (lua_State *L, lua_State *L1, const char *msg,
				int level)
{
  int top = (int)(L->top - L->base);
  int lim = TRACEBACK_LEVELS1;
  lua_Debug ar;
  if (msg) lua_pushfstring(L, "%s\n", msg);
  lua_pushliteral(L, "stack traceback:");
  while (lua_getstack(L1, level++, &ar)) {
    GCfunc *fn;
    if (level > lim) {
      if (!lua_getstack(L1, level + TRACEBACK_LEVELS2, &ar)) {
	level--;
      } else {
	lua_pushliteral(L, "\n\t...");
	lua_getstack(L1, -10, &ar);
	level = ar.i_ci - TRACEBACK_LEVELS2;
      }
      lim = 2147483647;
      continue;
    }
    lua_getinfo(L1, "Snlf", &ar);
    fn = funcV(L1->top-1); L1->top--;
    if (isffunc(fn) && !*ar.namewhat)
      lua_pushfstring(L, "\n\t[builtin#%d]:", fn->c.ffid);
    else
      lua_pushfstring(L, "\n\t%s:", ar.short_src);
    if (ar.currentline > 0)
      lua_pushfstring(L, "%d:", ar.currentline);
    if (*ar.namewhat) {
      lua_pushfstring(L, " in function " LUA_QS, ar.name);
    } else {
      if (*ar.what == 'm') {
	lua_pushliteral(L, " in main chunk");
      } else if (*ar.what == 'C') {
	lua_pushfstring(L, " at %p", fn->c.f);
      } else {
	lua_pushfstring(L, " in function <%s:%d>",
			ar.short_src, ar.linedefined);
      }
    }
    if ((int)(L->top - L->base) - top >= 15)
      lua_concat(L, (int)(L->top - L->base) - top);
  }
  lua_concat(L, (int)(L->top - L->base) - top);
}

