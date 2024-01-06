/*
** Fast function call recorder.
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
*/

#define lj_ffrecord_c
#define LUA_CORE

#include "lj_obj.h"

#if LJ_HASJIT

#include "lj_err.h"
#include "lj_buf.h"
#include "lj_str.h"
#include "lj_tab.h"
#include "lj_frame.h"
#include "lj_bc.h"
#include "lj_ff.h"
#include "lj_ir.h"
#include "lj_jit.h"
#include "lj_ircall.h"
#include "lj_iropt.h"
#include "lj_trace.h"
#include "lj_record.h"
#include "lj_ffrecord.h"
#include "lj_crecord.h"
#include "lj_dispatch.h"
#include "lj_vm.h"
#include "lj_strscan.h"
#include "lj_strfmt.h"
#include "lj_serialize.h"

/* Some local macros to save typing. Undef'd at the end. */
#define IR(ref)			(&J->cur.ir[(ref)])

/* Pass IR on to next optimization in chain (FOLD). */
#define emitir(ot, a, b)	(lj_ir_set(J, (ot), (a), (b)), lj_opt_fold(J))

/* -- Fast function recording handlers ------------------------------------ */

/* Conventions for fast function call handlers:
**
** The argument slots start at J->base[0]. All of them are guaranteed to be
** valid and type-specialized references. J->base[J->maxslot] is set to 0
** as a sentinel. The runtime argument values start at rd->argv[0].
**
** In general fast functions should check for presence of all of their
** arguments and for the correct argument types. Some simplifications
** are allowed if the interpreter throws instead. But even if recording
** is aborted, the generated IR must be consistent (no zero-refs).
**
** The number of results in rd->nres is set to 1. Handlers that return
** a different number of results need to override it. A negative value
** prevents return processing (e.g. for pending calls).
**
** Results need to be stored starting at J->base[0]. Return processing
** moves them to the right slots later.
**
** The per-ffid auxiliary data is the value of the 2nd part of the
** LJLIB_REC() annotation. This allows handling similar functionality
** in a common handler.
*/

/* Type of handler to record a fast function. */
typedef void (LJ_FASTCALL *RecordFunc)(jit_State *J, RecordFFData *rd);

/* Get runtime value of int argument. */
static int32_t argv2int(jit_State *J, TValue *o)
{
  if (!lj_strscan_numberobj(o))
    lj_trace_err(J, LJ_TRERR_BADTYPE);
  return tvisint(o) ? intV(o) : lj_num2int(numV(o));
}

/* Get runtime value of string argument. */
static GCstr *argv2str(jit_State *J, TValue *o)
{
  if (LJ_LIKELY(tvisstr(o))) {
    return strV(o);
  } else {
    GCstr *s;
    if (!tvisnumber(o))
      lj_trace_err(J, LJ_TRERR_BADTYPE);
    s = lj_strfmt_number(J->L, o);
    setstrV(J->L, o, s);
    return s;
  }
}

/* Return number of results wanted by caller. */
static ptrdiff_t results_wanted(jit_State *J)
{
  TValue *frame = J->L->base-1;
  if (frame_islua(frame))
    return (ptrdiff_t)bc_b(frame_pc(frame)[-1]) - 1;
  else
    return -1;
}

/* Trace stitching: add continuation below frame to start a new trace. */
static void recff_stitch(jit_State *J)
{
  ASMFunction cont = lj_cont_stitch;
  lua_State *L = J->L;
  TValue *base = L->base;
  BCReg nslot = J->maxslot + 1 + LJ_FR2;
  TValue *nframe = base + 1 + LJ_FR2;
  const BCIns *pc = frame_pc(base-1);
  TValue *pframe = frame_prevl(base-1);

  /* Check for this now. Throwing in lj_record_stop messes up the stack. */
  if (J->cur.nsnap >= (MSize)J->param[JIT_P_maxsnap])
    lj_trace_err(J, LJ_TRERR_SNAPOV);

  /* Move func + args up in Lua stack and insert continuation. */
  memmove(&base[1], &base[-1-LJ_FR2], sizeof(TValue)*nslot);
  setframe_ftsz(nframe, ((char *)nframe - (char *)pframe) + FRAME_CONT);
  setcont(base-LJ_FR2, cont);
  setframe_pc(base, pc);
  setnilV(base-1-LJ_FR2);  /* Incorrect, but rec_check_slots() won't run anymore. */
  L->base += 2 + LJ_FR2;
  L->top += 2 + LJ_FR2;

  /* Ditto for the IR. */
  memmove(&J->base[1], &J->base[-1-LJ_FR2], sizeof(TRef)*nslot);
#if LJ_FR2
  J->base[2] = TREF_FRAME;
  J->base[-1] = lj_ir_k64(J, IR_KNUM, u64ptr(contptr(cont)));
  J->base[0] = lj_ir_k64(J, IR_KNUM, u64ptr(pc)) | TREF_CONT;
#else
  J->base[0] = lj_ir_kptr(J, contptr(cont)) | TREF_CONT;
#endif
  J->ktrace = tref_ref((J->base[-1-LJ_FR2] = lj_ir_ktrace(J)));
  J->base += 2 + LJ_FR2;
  J->baseslot += 2 + LJ_FR2;
  J->framedepth++;

  lj_record_stop(J, LJ_TRLINK_STITCH, 0);

  /* Undo Lua stack changes. */
  memmove(&base[-1-LJ_FR2], &base[1], sizeof(TValue)*nslot);
  setframe_pc(base-1, pc);
  L->base -= 2 + LJ_FR2;
  L->top -= 2 + LJ_FR2;
}

/* Fallback handler for fast functions that are not recorded (yet). */
static void LJ_FASTCALL recff_nyi(jit_State *J, RecordFFData *rd)
{
  if (J->cur.nins < (IRRef)J->param[JIT_P_minstitch] + REF_BASE) {
    lj_trace_err_info(J, LJ_TRERR_TRACEUV);
  } else {
    /* Can only stitch from Lua call. */
    if (J->framedepth && frame_islua(J->L->base-1)) {
      BCOp op = bc_op(*frame_pc(J->L->base-1));
      /* Stitched trace cannot start with *M op with variable # of args. */
      if (!(op == BC_CALLM || op == BC_CALLMT ||
	    op == BC_RETM || op == BC_TSETM)) {
	switch (J->fn->c.ffid) {
	case FF_error:
	case FF_debug_sethook:
	case FF_jit_flush:
	  break;  /* Don't stitch across special builtins. */
	default:
	  recff_stitch(J);  /* Use trace stitching. */
	  rd->nres = -1;
	  return;
	}
      }
    }
    /* Otherwise stop trace and return to interpreter. */
    lj_record_stop(J, LJ_TRLINK_RETURN, 0);
    rd->nres = -1;
  }
}

/* Fallback handler for unsupported variants of fast functions. */
#define recff_nyiu	recff_nyi

/* Must stop the trace for classic C functions with arbitrary side-effects. */
#define recff_c		recff_nyi

/* Emit BUFHDR for the global temporary buffer. */
static TRef recff_bufhdr(jit_State *J)
{
  return emitir(IRT(IR_BUFHDR, IRT_PGC),
		lj_ir_kptr(J, &J2G(J)->tmpbuf), IRBUFHDR_RESET);
}

/* Emit TMPREF. */
static TRef recff_tmpref(jit_State *J, TRef tr, int mode)
{
  if (!LJ_DUALNUM && tref_isinteger(tr))
    tr = emitir(IRTN(IR_CONV), tr, IRCONV_NUM_INT);
  return emitir(IRT(IR_TMPREF, IRT_PGC), tr, mode);
}

/* -- Base library fast functions ----------------------------------------- */

static void LJ_FASTCALL recff_assert(jit_State *J, RecordFFData *rd)
{
  /* Arguments already specialized. The interpreter throws for nil/false. */
  rd->nres = J->maxslot;  /* Pass through all arguments. */
}

static void LJ_FASTCALL recff_type(jit_State *J, RecordFFData *rd)
{
  /* Arguments already specialized. Result is a constant string. Neat, huh? */
  uint32_t t;
  if (tvisnumber(&rd->argv[0]))
    t = ~LJ_TNUMX;
  else if (LJ_64 && !LJ_GC64 && tvislightud(&rd->argv[0]))
    t = ~LJ_TLIGHTUD;
  else
    t = ~itype(&rd->argv[0]);
  J->base[0] = lj_ir_kstr(J, strV(&J->fn->c.upvalue[t]));
  UNUSED(rd);
}

static void LJ_FASTCALL recff_getmetatable(jit_State *J, RecordFFData *rd)
{
  TRef tr = J->base[0];
  if (tr) {
    RecordIndex ix;
    ix.tab = tr;
    copyTV(J->L, &ix.tabv, &rd->argv[0]);
    if (lj_record_mm_lookup(J, &ix, MM_metatable))
      J->base[0] = ix.mobj;
    else
      J->base[0] = ix.mt;
  }  /* else: Interpreter will throw. */
}

static void LJ_FASTCALL recff_setmetatable(jit_State *J, RecordFFData *rd)
{
  TRef tr = J->base[0];
  TRef mt = J->base[1];
  if (tref_istab(tr) && (tref_istab(mt) || (mt && tref_isnil(mt)))) {
    TRef fref, mtref;
    RecordIndex ix;
    ix.tab = tr;
    copyTV(J->L, &ix.tabv, &rd->argv[0]);
    lj_record_mm_lookup(J, &ix, MM_metatable); /* Guard for no __metatable. */
    fref = emitir(IRT(IR_FREF, IRT_PGC), tr, IRFL_TAB_META);
    mtref = tref_isnil(mt) ? lj_ir_knull(J, IRT_TAB) : mt;
    emitir(IRT(IR_FSTORE, IRT_TAB), fref, mtref);
    if (!tref_isnil(mt))
      emitir(IRT(IR_TBAR, IRT_TAB), tr, 0);
    J->base[0] = tr;
    J->needsnap = 1;
  }  /* else: Interpreter will throw. */
}

static void LJ_FASTCALL recff_rawget(jit_State *J, RecordFFData *rd)
{
  RecordIndex ix;
  ix.tab = J->base[0]; ix.key = J->base[1];
  if (tref_istab(ix.tab) && ix.key) {
    ix.val = 0; ix.idxchain = 0;
    settabV(J->L, &ix.tabv, tabV(&rd->argv[0]));
    copyTV(J->L, &ix.keyv, &rd->argv[1]);
    J->base[0] = lj_record_idx(J, &ix);
  }  /* else: Interpreter will throw. */
}

static void LJ_FASTCALL recff_rawset(jit_State *J, RecordFFData *rd)
{
  RecordIndex ix;
  ix.tab = J->base[0]; ix.key = J->base[1]; ix.val = J->base[2];
  if (tref_istab(ix.tab) && ix.key && ix.val) {
    ix.idxchain = 0;
    settabV(J->L, &ix.tabv, tabV(&rd->argv[0]));
    copyTV(J->L, &ix.keyv, &rd->argv[1]);
    copyTV(J->L, &ix.valv, &rd->argv[2]);
    lj_record_idx(J, &ix);
    /* Pass through table at J->base[0] as result. */
  }  /* else: Interpreter will throw. */
}

static void LJ_FASTCALL recff_rawequal(jit_State *J, RecordFFData *rd)
{
  TRef tra = J->base[0];
  TRef trb = J->base[1];
  if (tra && trb) {
    int diff = lj_record_objcmp(J, tra, trb, &rd->argv[0], &rd->argv[1]);
    J->base[0] = diff ? TREF_FALSE : TREF_TRUE;
  }  /* else: Interpreter will throw. */
}

#if LJ_52
static void LJ_FASTCALL recff_rawlen(jit_State *J, RecordFFData *rd)
{
  TRef tr = J->base[0];
  if (tref_isstr(tr))
    J->base[0] = emitir(IRTI(IR_FLOAD), tr, IRFL_STR_LEN);
  else if (tref_istab(tr))
    J->base[0] = emitir(IRTI(IR_ALEN), tr, TREF_NIL);
  /* else: Interpreter will throw. */
  UNUSED(rd);
}
#endif

/* Determine mode of select() call. */
int32_t lj_ffrecord_select_mode(jit_State *J, TRef tr, TValue *tv)
{
  if (tref_isstr(tr) && *strVdata(tv) == '#') {  /* select('#', ...) */
    if (strV(tv)->len == 1) {
      emitir(IRTG(IR_EQ, IRT_STR), tr, lj_ir_kstr(J, strV(tv)));
    } else {
      TRef trptr = emitir(IRT(IR_STRREF, IRT_PGC), tr, lj_ir_kint(J, 0));
      TRef trchar = emitir(IRT(IR_XLOAD, IRT_U8), trptr, IRXLOAD_READONLY);
      emitir(IRTGI(IR_EQ), trchar, lj_ir_kint(J, '#'));
    }
    return 0;
  } else {  /* select(n, ...) */
    int32_t start = argv2int(J, tv);
    if (start == 0) lj_trace_err(J, LJ_TRERR_BADTYPE);  /* A bit misleading. */
    return start;
  }
}

static void LJ_FASTCALL recff_select(jit_State *J, RecordFFData *rd)
{
  TRef tr = J->base[0];
  if (tr) {
    ptrdiff_t start = lj_ffrecord_select_mode(J, tr, &rd->argv[0]);
    if (start == 0) {  /* select('#', ...) */
      J->base[0] = lj_ir_kint(J, J->maxslot - 1);
    } else if (tref_isk(tr)) {  /* select(k, ...) */
      ptrdiff_t n = (ptrdiff_t)J->maxslot;
      if (start < 0) start += n;
      else if (start > n) start = n;
      if (start >= 1) {
	ptrdiff_t i;
	rd->nres = n - start;
	for (i = 0; i < n - start; i++)
	  J->base[i] = J->base[start+i];
      }  /* else: Interpreter will throw. */
    } else {
      recff_nyiu(J, rd);
      return;
    }
  }  /* else: Interpreter will throw. */
}

static void LJ_FASTCALL recff_tonumber(jit_State *J, RecordFFData *rd)
{
  TRef tr = J->base[0];
  TRef base = J->base[1];
  if (tr && !tref_isnil(base)) {
    base = lj_opt_narrow_toint(J, base);
    if (!tref_isk(base) || IR(tref_ref(base))->i != 10) {
      recff_nyiu(J, rd);
      return;
    }
  }
  if (tref_isnumber_str(tr)) {
    if (tref_isstr(tr)) {
      TValue tmp;
      if (!lj_strscan_num(strV(&rd->argv[0]), &tmp)) {
	recff_nyiu(J, rd);  /* Would need an inverted STRTO for this case. */
	return;
      }
      tr = emitir(IRTG(IR_STRTO, IRT_NUM), tr, 0);
    }
#if LJ_HASFFI
  } else if (tref_iscdata(tr)) {
    lj_crecord_tonumber(J, rd);
    return;
#endif
  } else {
    tr = TREF_NIL;
  }
  J->base[0] = tr;
  UNUSED(rd);
}

static TValue *recff_metacall_cp(lua_State *L, lua_CFunction dummy, void *ud)
{
  jit_State *J = (jit_State *)ud;
  lj_record_tailcall(J, 0, 1);
  UNUSED(L); UNUSED(dummy);
  return NULL;
}

static int recff_metacall(jit_State *J, RecordFFData *rd, MMS mm)
{
  RecordIndex ix;
  ix.tab = J->base[0];
  copyTV(J->L, &ix.tabv, &rd->argv[0]);
  if (lj_record_mm_lookup(J, &ix, mm)) {  /* Has metamethod? */
    int errcode;
    TValue argv0;
    /* Temporarily insert metamethod below object. */
    J->base[1+LJ_FR2] = J->base[0];
    J->base[0] = ix.mobj;
    copyTV(J->L, &argv0, &rd->argv[0]);
    copyTV(J->L, &rd->argv[1+LJ_FR2], &rd->argv[0]);
    copyTV(J->L, &rd->argv[0], &ix.mobjv);
    /* Need to protect lj_record_tailcall because it may throw. */
    errcode = lj_vm_cpcall(J->L, NULL, J, recff_metacall_cp);
    /* Always undo Lua stack changes to avoid confusing the interpreter. */
    copyTV(J->L, &rd->argv[0], &argv0);
    if (errcode)
      lj_err_throw(J->L, errcode);  /* Propagate errors. */
    rd->nres = -1;  /* Pending call. */
    return 1;  /* Tailcalled to metamethod. */
  }
  return 0;
}

static void LJ_FASTCALL recff_tostring(jit_State *J, RecordFFData *rd)
{
  TRef tr = J->base[0];
  if (tref_isstr(tr)) {
    /* Ignore __tostring in the string base metatable. */
    /* Pass on result in J->base[0]. */
  } else if (tr && !recff_metacall(J, rd, MM_tostring)) {
    if (tref_isnumber(tr)) {
      J->base[0] = emitir(IRT(IR_TOSTR, IRT_STR), tr,
			  tref_isnum(tr) ? IRTOSTR_NUM : IRTOSTR_INT);
    } else if (tref_ispri(tr)) {
      J->base[0] = lj_ir_kstr(J, lj_strfmt_obj(J->L, &rd->argv[0]));
    } else {
      recff_nyiu(J, rd);
      return;
    }
  }
}

static void LJ_FASTCALL recff_ipairs_aux(jit_State *J, RecordFFData *rd)
{
  RecordIndex ix;
  ix.tab = J->base[0];
  if (tref_istab(ix.tab)) {
    if (!tvisnumber(&rd->argv[1]))  /* No support for string coercion. */
      lj_trace_err(J, LJ_TRERR_BADTYPE);
    setintV(&ix.keyv, numberVint(&rd->argv[1])+1);
    settabV(J->L, &ix.tabv, tabV(&rd->argv[0]));
    ix.val = 0; ix.idxchain = 0;
    ix.key = lj_opt_narrow_toint(J, J->base[1]);
    J->base[0] = ix.key = emitir(IRTI(IR_ADD), ix.key, lj_ir_kint(J, 1));
    J->base[1] = lj_record_idx(J, &ix);
    rd->nres = tref_isnil(J->base[1]) ? 0 : 2;
  }  /* else: Interpreter will throw. */
}

static void LJ_FASTCALL recff_xpairs(jit_State *J, RecordFFData *rd)
{
  TRef tr = J->base[0];
  if (!((LJ_52 || (LJ_HASFFI && tref_iscdata(tr))) &&
	recff_metacall(J, rd, MM_pairs + rd->data))) {
    if (tref_istab(tr)) {
      J->base[0] = lj_ir_kfunc(J, funcV(&J->fn->c.upvalue[0]));
      J->base[1] = tr;
      J->base[2] = rd->data ? lj_ir_kint(J, 0) : TREF_NIL;
      rd->nres = 3;
    }  /* else: Interpreter will throw. */
  }
}

static void LJ_FASTCALL recff_pcall(jit_State *J, RecordFFData *rd)
{
  if (J->maxslot >= 1) {
#if LJ_FR2
    /* Shift function arguments up. */
    memmove(J->base + 1, J->base, sizeof(TRef) * J->maxslot);
#endif
    lj_record_call(J, 0, J->maxslot - 1);
    rd->nres = -1;  /* Pending call. */
    J->needsnap = 1;  /* Start catching on-trace errors. */
  }  /* else: Interpreter will throw. */
}

static TValue *recff_xpcall_cp(lua_State *L, lua_CFunction dummy, void *ud)
{
  jit_State *J = (jit_State *)ud;
  lj_record_call(J, 1, J->maxslot - 2);
  UNUSED(L); UNUSED(dummy);
  return NULL;
}

static void LJ_FASTCALL recff_xpcall(jit_State *J, RecordFFData *rd)
{
  if (J->maxslot >= 2) {
    TValue argv0, argv1;
    TRef tmp;
    int errcode;
    /* Swap function and traceback. */
    tmp = J->base[0]; J->base[0] = J->base[1]; J->base[1] = tmp;
    copyTV(J->L, &argv0, &rd->argv[0]);
    copyTV(J->L, &argv1, &rd->argv[1]);
    copyTV(J->L, &rd->argv[0], &argv1);
    copyTV(J->L, &rd->argv[1], &argv0);
#if LJ_FR2
    /* Shift function arguments up. */
    memmove(J->base + 2, J->base + 1, sizeof(TRef) * (J->maxslot-1));
#endif
    /* Need to protect lj_record_call because it may throw. */
    errcode = lj_vm_cpcall(J->L, NULL, J, recff_xpcall_cp);
    /* Always undo Lua stack swap to avoid confusing the interpreter. */
    copyTV(J->L, &rd->argv[0], &argv0);
    copyTV(J->L, &rd->argv[1], &argv1);
    if (errcode)
      lj_err_throw(J->L, errcode);  /* Propagate errors. */
    rd->nres = -1;  /* Pending call. */
    J->needsnap = 1;  /* Start catching on-trace errors. */
  }  /* else: Interpreter will throw. */
}

static void LJ_FASTCALL recff_getfenv(jit_State *J, RecordFFData *rd)
{
  TRef tr = J->base[0];
  /* Only support getfenv(0) for now. */
  if (tref_isint(tr) && tref_isk(tr) && IR(tref_ref(tr))->i == 0) {
    TRef trl = emitir(IRT(IR_LREF, IRT_THREAD), 0, 0);
    J->base[0] = emitir(IRT(IR_FLOAD, IRT_TAB), trl, IRFL_THREAD_ENV);
    return;
  }
  recff_nyiu(J, rd);
}

static void LJ_FASTCALL recff_next(jit_State *J, RecordFFData *rd)
{
#if LJ_BE
  /* YAGNI: Disabled on big-endian due to issues with lj_vm_next,
  ** IR_HIOP, RID_RETLO/RID_RETHI and ra_destpair.
  */
  recff_nyi(J, rd);
#else
  TRef tab = J->base[0];
  if (tref_istab(tab)) {
    RecordIndex ix;
    cTValue *keyv;
    ix.tab = tab;
    if (tref_isnil(J->base[1])) {  /* Shortcut for start of traversal. */
      ix.key = lj_ir_kint(J, 0);
      keyv = niltvg(J2G(J));
    } else {
      TRef tmp = recff_tmpref(J, J->base[1], IRTMPREF_IN1);
      ix.key = lj_ir_call(J, IRCALL_lj_tab_keyindex, tab, tmp);
      keyv = &rd->argv[1];
    }
    copyTV(J->L, &ix.tabv, &rd->argv[0]);
    ix.keyv.u32.lo = lj_tab_keyindex(tabV(&ix.tabv), keyv);
    /* Omit the value, if not used by the caller. */
    ix.idxchain = (J->framedepth && frame_islua(J->L->base-1) &&
		   bc_b(frame_pc(J->L->base-1)[-1])-1 < 2);
    ix.mobj = 0;  /* We don't need the next index. */
    rd->nres = lj_record_next(J, &ix);
    J->base[0] = ix.key;
    J->base[1] = ix.val;
  }  /* else: Interpreter will throw. */
#endif
}

/* -- Math library fast functions ----------------------------------------- */

static void LJ_FASTCALL recff_math_abs(jit_State *J, RecordFFData *rd)
{
  TRef tr = lj_ir_tonum(J, J->base[0]);
  J->base[0] = emitir(IRTN(IR_ABS), tr, lj_ir_ksimd(J, LJ_KSIMD_ABS));
  UNUSED(rd);
}

/* Record rounding functions math.floor and math.ceil. */
static void LJ_FASTCALL recff_math_round(jit_State *J, RecordFFData *rd)
{
  TRef tr = J->base[0];
  if (!tref_isinteger(tr)) {  /* Pass through integers unmodified. */
    tr = emitir(IRTN(IR_FPMATH), lj_ir_tonum(J, tr), rd->data);
    /* Result is integral (or NaN/Inf), but may not fit an int32_t. */
    if (LJ_DUALNUM) {  /* Try to narrow using a guarded conversion to int. */
      lua_Number n = lj_vm_foldfpm(numberVnum(&rd->argv[0]), rd->data);
      if (n == (lua_Number)lj_num2int(n))
	tr = emitir(IRTGI(IR_CONV), tr, IRCONV_INT_NUM|IRCONV_CHECK);
    }
    J->base[0] = tr;
  }
}

/* Record unary math.* functions, mapped to IR_FPMATH opcode. */
static void LJ_FASTCALL recff_math_unary(jit_State *J, RecordFFData *rd)
{
  J->base[0] = emitir(IRTN(IR_FPMATH), lj_ir_tonum(J, J->base[0]), rd->data);
}

/* Record math.log. */
static void LJ_FASTCALL recff_math_log(jit_State *J, RecordFFData *rd)
{
  TRef tr = lj_ir_tonum(J, J->base[0]);
  if (J->base[1]) {
#ifdef LUAJIT_NO_LOG2
    uint32_t fpm = IRFPM_LOG;
#else
    uint32_t fpm = IRFPM_LOG2;
#endif
    TRef trb = lj_ir_tonum(J, J->base[1]);
    tr = emitir(IRTN(IR_FPMATH), tr, fpm);
    trb = emitir(IRTN(IR_FPMATH), trb, fpm);
    trb = emitir(IRTN(IR_DIV), lj_ir_knum_one(J), trb);
    tr = emitir(IRTN(IR_MUL), tr, trb);
  } else {
    tr = emitir(IRTN(IR_FPMATH), tr, IRFPM_LOG);
  }
  J->base[0] = tr;
  UNUSED(rd);
}

/* Record math.atan2. */
static void LJ_FASTCALL recff_math_atan2(jit_State *J, RecordFFData *rd)
{
  TRef tr = lj_ir_tonum(J, J->base[0]);
  TRef tr2 = lj_ir_tonum(J, J->base[1]);
  J->base[0] = lj_ir_call(J, IRCALL_atan2, tr, tr2);
  UNUSED(rd);
}

/* Record math.ldexp. */
static void LJ_FASTCALL recff_math_ldexp(jit_State *J, RecordFFData *rd)
{
  TRef tr = lj_ir_tonum(J, J->base[0]);
#if LJ_TARGET_X86ORX64
  TRef tr2 = lj_ir_tonum(J, J->base[1]);
#else
  TRef tr2 = lj_opt_narrow_toint(J, J->base[1]);
#endif
  J->base[0] = emitir(IRTN(IR_LDEXP), tr, tr2);
  UNUSED(rd);
}

static void LJ_FASTCALL recff_math_call(jit_State *J, RecordFFData *rd)
{
  TRef tr = lj_ir_tonum(J, J->base[0]);
  J->base[0] = emitir(IRTN(IR_CALLN), tr, rd->data);
}

static void LJ_FASTCALL recff_math_pow(jit_State *J, RecordFFData *rd)
{
  J->base[0] = lj_opt_narrow_arith(J, J->base[0], J->base[1],
				   &rd->argv[0], &rd->argv[1], IR_POW);
  UNUSED(rd);
}

static void LJ_FASTCALL recff_math_minmax(jit_State *J, RecordFFData *rd)
{
  TRef tr = lj_ir_tonumber(J, J->base[0]);
  uint32_t op = rd->data;
  BCReg i;
  for (i = 1; J->base[i] != 0; i++) {
    TRef tr2 = lj_ir_tonumber(J, J->base[i]);
    IRType t = IRT_INT;
    if (!(tref_isinteger(tr) && tref_isinteger(tr2))) {
      if (tref_isinteger(tr)) tr = emitir(IRTN(IR_CONV), tr, IRCONV_NUM_INT);
      if (tref_isinteger(tr2)) tr2 = emitir(IRTN(IR_CONV), tr2, IRCONV_NUM_INT);
      t = IRT_NUM;
    }
    tr = emitir(IRT(op, t), tr, tr2);
  }
  J->base[0] = tr;
}

static void LJ_FASTCALL recff_math_random(jit_State *J, RecordFFData *rd)
{
  GCudata *ud = udataV(&J->fn->c.upvalue[0]);
  TRef tr, one;
  lj_ir_kgc(J, obj2gco(ud), IRT_UDATA);  /* Prevent collection. */
  tr = lj_ir_call(J, IRCALL_lj_prng_u64d, lj_ir_kptr(J, uddata(ud)));
  one = lj_ir_knum_one(J);
  tr = emitir(IRTN(IR_SUB), tr, one);
  if (J->base[0]) {
    TRef tr1 = lj_ir_tonum(J, J->base[0]);
    if (J->base[1]) {  /* d = floor(d*(r2-r1+1.0)) + r1 */
      TRef tr2 = lj_ir_tonum(J, J->base[1]);
      tr2 = emitir(IRTN(IR_SUB), tr2, tr1);
      tr2 = emitir(IRTN(IR_ADD), tr2, one);
      tr = emitir(IRTN(IR_MUL), tr, tr2);
      tr = emitir(IRTN(IR_FPMATH), tr, IRFPM_FLOOR);
      tr = emitir(IRTN(IR_ADD), tr, tr1);
    } else {  /* d = floor(d*r1) + 1.0 */
      tr = emitir(IRTN(IR_MUL), tr, tr1);
      tr = emitir(IRTN(IR_FPMATH), tr, IRFPM_FLOOR);
      tr = emitir(IRTN(IR_ADD), tr, one);
    }
  }
  J->base[0] = tr;
  UNUSED(rd);
}

/* -- Bit library fast functions ------------------------------------------ */

/* Record bit.tobit. */
static void LJ_FASTCALL recff_bit_tobit(jit_State *J, RecordFFData *rd)
{
  TRef tr = J->base[0];
#if LJ_HASFFI
  if (tref_iscdata(tr)) { recff_bit64_tobit(J, rd); return; }
#endif
  J->base[0] = lj_opt_narrow_tobit(J, tr);
  UNUSED(rd);
}

/* Record unary bit.bnot, bit.bswap. */
static void LJ_FASTCALL recff_bit_unary(jit_State *J, RecordFFData *rd)
{
#if LJ_HASFFI
  if (recff_bit64_unary(J, rd))
    return;
#endif
  J->base[0] = emitir(IRTI(rd->data), lj_opt_narrow_tobit(J, J->base[0]), 0);
}

/* Record N-ary bit.band, bit.bor, bit.bxor. */
static void LJ_FASTCALL recff_bit_nary(jit_State *J, RecordFFData *rd)
{
#if LJ_HASFFI
  if (recff_bit64_nary(J, rd))
    return;
#endif
  {
    TRef tr = lj_opt_narrow_tobit(J, J->base[0]);
    uint32_t ot = IRTI(rd->data);
    BCReg i;
    for (i = 1; J->base[i] != 0; i++)
      tr = emitir(ot, tr, lj_opt_narrow_tobit(J, J->base[i]));
    J->base[0] = tr;
  }
}

/* Record bit shifts. */
static void LJ_FASTCALL recff_bit_shift(jit_State *J, RecordFFData *rd)
{
#if LJ_HASFFI
  if (recff_bit64_shift(J, rd))
    return;
#endif
  {
    TRef tr = lj_opt_narrow_tobit(J, J->base[0]);
    TRef tsh = lj_opt_narrow_tobit(J, J->base[1]);
    IROp op = (IROp)rd->data;
    if (!(op < IR_BROL ? LJ_TARGET_MASKSHIFT : LJ_TARGET_MASKROT) &&
	!tref_isk(tsh))
      tsh = emitir(IRTI(IR_BAND), tsh, lj_ir_kint(J, 31));
#ifdef LJ_TARGET_UNIFYROT
    if (op == (LJ_TARGET_UNIFYROT == 1 ? IR_BROR : IR_BROL)) {
      op = LJ_TARGET_UNIFYROT == 1 ? IR_BROL : IR_BROR;
      tsh = emitir(IRTI(IR_NEG), tsh, tsh);
    }
#endif
    J->base[0] = emitir(IRTI(op), tr, tsh);
  }
}

static void LJ_FASTCALL recff_bit_tohex(jit_State *J, RecordFFData *rd)
{
#if LJ_HASFFI
  TRef hdr = recff_bufhdr(J);
  TRef tr = recff_bit64_tohex(J, rd, hdr);
  J->base[0] = emitir(IRTG(IR_BUFSTR, IRT_STR), tr, hdr);
#else
  recff_nyiu(J, rd);  /* Don't bother working around this NYI. */
#endif
}

/* -- String library fast functions --------------------------------------- */

/* Specialize to relative starting position for string. */
static TRef recff_string_start(jit_State *J, GCstr *s, int32_t *st, TRef tr,
			       TRef trlen, TRef tr0)
{
  int32_t start = *st;
  if (start < 0) {
    emitir(IRTGI(IR_LT), tr, tr0);
    tr = emitir(IRTI(IR_ADD), trlen, tr);
    start = start + (int32_t)s->len;
    emitir(start < 0 ? IRTGI(IR_LT) : IRTGI(IR_GE), tr, tr0);
    if (start < 0) {
      tr = tr0;
      start = 0;
    }
  } else if (start == 0) {
    emitir(IRTGI(IR_EQ), tr, tr0);
    tr = tr0;
  } else {
    tr = emitir(IRTI(IR_ADD), tr, lj_ir_kint(J, -1));
    emitir(IRTGI(IR_GE), tr, tr0);
    start--;
  }
  *st = start;
  return tr;
}

/* Handle string.byte (rd->data = 0) and string.sub (rd->data = 1). */
static void LJ_FASTCALL recff_string_range(jit_State *J, RecordFFData *rd)
{
  TRef trstr = lj_ir_tostr(J, J->base[0]);
  TRef trlen = emitir(IRTI(IR_FLOAD), trstr, IRFL_STR_LEN);
  TRef tr0 = lj_ir_kint(J, 0);
  TRef trstart, trend;
  GCstr *str = argv2str(J, &rd->argv[0]);
  int32_t start, end;
  if (rd->data) {  /* string.sub(str, start [,end]) */
    start = argv2int(J, &rd->argv[1]);
    trstart = lj_opt_narrow_toint(J, J->base[1]);
    trend = J->base[2];
    if (tref_isnil(trend)) {
      trend = lj_ir_kint(J, -1);
      end = -1;
    } else {
      trend = lj_opt_narrow_toint(J, trend);
      end = argv2int(J, &rd->argv[2]);
    }
  } else {  /* string.byte(str, [,start [,end]]) */
    if (tref_isnil(J->base[1])) {
      start = 1;
      trstart = lj_ir_kint(J, 1);
    } else {
      start = argv2int(J, &rd->argv[1]);
      trstart = lj_opt_narrow_toint(J, J->base[1]);
    }
    if (J->base[1] && !tref_isnil(J->base[2])) {
      trend = lj_opt_narrow_toint(J, J->base[2]);
      end = argv2int(J, &rd->argv[2]);
    } else {
      trend = trstart;
      end = start;
    }
  }
  if (end < 0) {
    emitir(IRTGI(IR_LT), trend, tr0);
    trend = emitir(IRTI(IR_ADD), emitir(IRTI(IR_ADD), trlen, trend),
		   lj_ir_kint(J, 1));
    end = end+(int32_t)str->len+1;
  } else if ((MSize)end <= str->len) {
    emitir(IRTGI(IR_ULE), trend, trlen);
  } else {
    emitir(IRTGI(IR_UGT), trend, trlen);
    end = (int32_t)str->len;
    trend = trlen;
  }
  trstart = recff_string_start(J, str, &start, trstart, trlen, tr0);
  if (rd->data) {  /* Return string.sub result. */
    if (end - start >= 0) {
      /* Also handle empty range here, to avoid extra traces. */
      TRef trptr, trslen = emitir(IRTI(IR_SUB), trend, trstart);
      emitir(IRTGI(IR_GE), trslen, tr0);
      trptr = emitir(IRT(IR_STRREF, IRT_PGC), trstr, trstart);
      J->base[0] = emitir(IRT(IR_SNEW, IRT_STR), trptr, trslen);
    } else {  /* Range underflow: return empty string. */
      emitir(IRTGI(IR_LT), trend, trstart);
      J->base[0] = lj_ir_kstr(J, &J2G(J)->strempty);
    }
  } else {  /* Return string.byte result(s). */
    ptrdiff_t i, len = end - start;
    if (len > 0) {
      TRef trslen = emitir(IRTI(IR_SUB), trend, trstart);
      emitir(IRTGI(IR_EQ), trslen, lj_ir_kint(J, (int32_t)len));
      if (J->baseslot + len > LJ_MAX_JSLOTS)
	lj_trace_err_info(J, LJ_TRERR_STACKOV);
      rd->nres = len;
      for (i = 0; i < len; i++) {
	TRef tmp = emitir(IRTI(IR_ADD), trstart, lj_ir_kint(J, (int32_t)i));
	tmp = emitir(IRT(IR_STRREF, IRT_PGC), trstr, tmp);
	J->base[i] = emitir(IRT(IR_XLOAD, IRT_U8), tmp, IRXLOAD_READONLY);
      }
    } else {  /* Empty range or range underflow: return no results. */
      emitir(IRTGI(IR_LE), trend, trstart);
      rd->nres = 0;
    }
  }
}

static void LJ_FASTCALL recff_string_char(jit_State *J, RecordFFData *rd)
{
  TRef k255 = lj_ir_kint(J, 255);
  BCReg i;
  for (i = 0; J->base[i] != 0; i++) {  /* Convert char values to strings. */
    TRef tr = lj_opt_narrow_toint(J, J->base[i]);
    emitir(IRTGI(IR_ULE), tr, k255);
    J->base[i] = emitir(IRT(IR_TOSTR, IRT_STR), tr, IRTOSTR_CHAR);
  }
  if (i > 1) {  /* Concatenate the strings, if there's more than one. */
    TRef hdr = recff_bufhdr(J), tr = hdr;
    for (i = 0; J->base[i] != 0; i++)
      tr = emitir(IRTG(IR_BUFPUT, IRT_PGC), tr, J->base[i]);
    J->base[0] = emitir(IRTG(IR_BUFSTR, IRT_STR), tr, hdr);
  } else if (i == 0) {
    J->base[0] = lj_ir_kstr(J, &J2G(J)->strempty);
  }
  UNUSED(rd);
}

static void LJ_FASTCALL recff_string_rep(jit_State *J, RecordFFData *rd)
{
  TRef str = lj_ir_tostr(J, J->base[0]);
  TRef rep = lj_opt_narrow_toint(J, J->base[1]);
  TRef hdr, tr, str2 = 0;
  if (!tref_isnil(J->base[2])) {
    TRef sep = lj_ir_tostr(J, J->base[2]);
    int32_t vrep = argv2int(J, &rd->argv[1]);
    emitir(IRTGI(vrep > 1 ? IR_GT : IR_LE), rep, lj_ir_kint(J, 1));
    if (vrep > 1) {
      TRef hdr2 = recff_bufhdr(J);
      TRef tr2 = emitir(IRTG(IR_BUFPUT, IRT_PGC), hdr2, sep);
      tr2 = emitir(IRTG(IR_BUFPUT, IRT_PGC), tr2, str);
      str2 = emitir(IRTG(IR_BUFSTR, IRT_STR), tr2, hdr2);
    }
  }
  tr = hdr = recff_bufhdr(J);
  if (str2) {
    tr = emitir(IRTG(IR_BUFPUT, IRT_PGC), tr, str);
    str = str2;
    rep = emitir(IRTI(IR_ADD), rep, lj_ir_kint(J, -1));
  }
  tr = lj_ir_call(J, IRCALL_lj_buf_putstr_rep, tr, str, rep);
  J->base[0] = emitir(IRTG(IR_BUFSTR, IRT_STR), tr, hdr);
}

static void LJ_FASTCALL recff_string_op(jit_State *J, RecordFFData *rd)
{
  TRef str = lj_ir_tostr(J, J->base[0]);
  TRef hdr = recff_bufhdr(J);
  TRef tr = lj_ir_call(J, rd->data, hdr, str);
  J->base[0] = emitir(IRTG(IR_BUFSTR, IRT_STR), tr, hdr);
}

static void LJ_FASTCALL recff_string_find(jit_State *J, RecordFFData *rd)
{
  TRef trstr = lj_ir_tostr(J, J->base[0]);
  TRef trpat = lj_ir_tostr(J, J->base[1]);
  TRef trlen = emitir(IRTI(IR_FLOAD), trstr, IRFL_STR_LEN);
  TRef tr0 = lj_ir_kint(J, 0);
  TRef trstart;
  GCstr *str = argv2str(J, &rd->argv[0]);
  GCstr *pat = argv2str(J, &rd->argv[1]);
  int32_t start;
  J->needsnap = 1;
  if (tref_isnil(J->base[2])) {
    trstart = lj_ir_kint(J, 1);
    start = 1;
  } else {
    trstart = lj_opt_narrow_toint(J, J->base[2]);
    start = argv2int(J, &rd->argv[2]);
  }
  trstart = recff_string_start(J, str, &start, trstart, trlen, tr0);
  if ((MSize)start <= str->len) {
    emitir(IRTGI(IR_ULE), trstart, trlen);
  } else {
    emitir(IRTGI(IR_UGT), trstart, trlen);
#if LJ_52
    J->base[0] = TREF_NIL;
    return;
#else
    trstart = trlen;
    start = str->len;
#endif
  }
  /* Fixed arg or no pattern matching chars? (Specialized to pattern string.) */
  if ((J->base[2] && tref_istruecond(J->base[3])) ||
      (emitir(IRTG(IR_EQ, IRT_STR), trpat, lj_ir_kstr(J, pat)),
       !lj_str_haspattern(pat))) {  /* Search for fixed string. */
    TRef trsptr = emitir(IRT(IR_STRREF, IRT_PGC), trstr, trstart);
    TRef trpptr = emitir(IRT(IR_STRREF, IRT_PGC), trpat, tr0);
    TRef trslen = emitir(IRTI(IR_SUB), trlen, trstart);
    TRef trplen = emitir(IRTI(IR_FLOAD), trpat, IRFL_STR_LEN);
    TRef tr = lj_ir_call(J, IRCALL_lj_str_find, trsptr, trpptr, trslen, trplen);
    TRef trp0 = lj_ir_kkptr(J, NULL);
    if (lj_str_find(strdata(str)+(MSize)start, strdata(pat),
		    str->len-(MSize)start, pat->len)) {
      TRef pos;
      emitir(IRTG(IR_NE, IRT_PGC), tr, trp0);
      /* Recompute offset. trsptr may not point into trstr after folding. */
      pos = emitir(IRTI(IR_ADD), emitir(IRTI(IR_SUB), tr, trsptr), trstart);
      J->base[0] = emitir(IRTI(IR_ADD), pos, lj_ir_kint(J, 1));
      J->base[1] = emitir(IRTI(IR_ADD), pos, trplen);
      rd->nres = 2;
    } else {
      emitir(IRTG(IR_EQ, IRT_PGC), tr, trp0);
      J->base[0] = TREF_NIL;
    }
  } else {  /* Search for pattern. */
    recff_nyiu(J, rd);
    return;
  }
}

static void recff_format(jit_State *J, RecordFFData *rd, TRef hdr, int sbufx)
{
  ptrdiff_t arg = sbufx;
  TRef tr = hdr, trfmt = lj_ir_tostr(J, J->base[arg]);
  GCstr *fmt = argv2str(J, &rd->argv[arg]);
  FormatState fs;
  SFormat sf;
  /* Specialize to the format string. */
  emitir(IRTG(IR_EQ, IRT_STR), trfmt, lj_ir_kstr(J, fmt));
  lj_strfmt_init(&fs, strdata(fmt), fmt->len);
  while ((sf = lj_strfmt_parse(&fs)) != STRFMT_EOF) {  /* Parse format. */
    TRef tra = sf == STRFMT_LIT ? 0 : J->base[++arg];
    TRef trsf = lj_ir_kint(J, (int32_t)sf);
    IRCallID id;
    switch (STRFMT_TYPE(sf)) {
    case STRFMT_LIT:
      tr = emitir(IRTG(IR_BUFPUT, IRT_PGC), tr,
		  lj_ir_kstr(J, lj_str_new(J->L, fs.str, fs.len)));
      break;
    case STRFMT_INT:
      id = IRCALL_lj_strfmt_putfnum_int;
    handle_int:
      if (!tref_isinteger(tra)) {
#if LJ_HASFFI
	if (tref_iscdata(tra)) {
	  tra = lj_crecord_loadiu64(J, tra, &rd->argv[arg]);
	  tr = lj_ir_call(J, IRCALL_lj_strfmt_putfxint, tr, trsf, tra);
	  break;
	}
#endif
	goto handle_num;
      }
      if (sf == STRFMT_INT) { /* Shortcut for plain %d. */
	tr = emitir(IRTG(IR_BUFPUT, IRT_PGC), tr,
		    emitir(IRT(IR_TOSTR, IRT_STR), tra, IRTOSTR_INT));
      } else {
#if LJ_HASFFI
	tra = emitir(IRT(IR_CONV, IRT_U64), tra,
		     (IRT_INT|(IRT_U64<<5)|IRCONV_SEXT));
	tr = lj_ir_call(J, IRCALL_lj_strfmt_putfxint, tr, trsf, tra);
	lj_needsplit(J);
#else
	recff_nyiu(J, rd);  /* Don't bother working around this NYI. */
	return;
#endif
      }
      break;
    case STRFMT_UINT:
      id = IRCALL_lj_strfmt_putfnum_uint;
      goto handle_int;
    case STRFMT_NUM:
      id = IRCALL_lj_strfmt_putfnum;
    handle_num:
      tra = lj_ir_tonum(J, tra);
      tr = lj_ir_call(J, id, tr, trsf, tra);
      if (LJ_SOFTFP32) lj_needsplit(J);
      break;
    case STRFMT_STR:
      if (!tref_isstr(tra)) {
	recff_nyiu(J, rd);  /* NYI: __tostring and non-string types for %s. */
	/* NYI: also buffers. */
	return;
      }
      if (sf == STRFMT_STR)  /* Shortcut for plain %s. */
	tr = emitir(IRTG(IR_BUFPUT, IRT_PGC), tr, tra);
      else if ((sf & STRFMT_T_QUOTED))
	tr = lj_ir_call(J, IRCALL_lj_strfmt_putquoted, tr, tra);
      else
	tr = lj_ir_call(J, IRCALL_lj_strfmt_putfstr, tr, trsf, tra);
      break;
    case STRFMT_CHAR:
      tra = lj_opt_narrow_toint(J, tra);
      if (sf == STRFMT_CHAR)  /* Shortcut for plain %c. */
	tr = emitir(IRTG(IR_BUFPUT, IRT_PGC), tr,
		    emitir(IRT(IR_TOSTR, IRT_STR), tra, IRTOSTR_CHAR));
      else
	tr = lj_ir_call(J, IRCALL_lj_strfmt_putfchar, tr, trsf, tra);
      break;
    case STRFMT_PTR:  /* NYI */
    case STRFMT_ERR:
    default:
      recff_nyiu(J, rd);
      return;
    }
  }
  if (sbufx) {
    emitir(IRT(IR_USE, IRT_NIL), tr, 0);
  } else {
    J->base[0] = emitir(IRTG(IR_BUFSTR, IRT_STR), tr, hdr);
  }
}

static void LJ_FASTCALL recff_string_format(jit_State *J, RecordFFData *rd)
{
  recff_format(J, rd, recff_bufhdr(J), 0);
}

/* -- Buffer library fast functions --------------------------------------- */

#if LJ_HASBUFFER

static LJ_AINLINE TRef recff_sbufx_get_L(jit_State *J, TRef ud)
{
  return emitir(IRT(IR_FLOAD, IRT_PGC), ud, IRFL_SBUF_L);
}

static LJ_AINLINE void recff_sbufx_set_L(jit_State *J, TRef ud, TRef val)
{
  TRef fref = emitir(IRT(IR_FREF, IRT_PGC), ud, IRFL_SBUF_L);
  emitir(IRT(IR_FSTORE, IRT_PGC), fref, val);
}

static LJ_AINLINE TRef recff_sbufx_get_ptr(jit_State *J, TRef ud, IRFieldID fl)
{
  return emitir(IRT(IR_FLOAD, IRT_PTR), ud, fl);
}

static LJ_AINLINE void recff_sbufx_set_ptr(jit_State *J, TRef ud, IRFieldID fl, TRef val)
{
  TRef fref = emitir(IRT(IR_FREF, IRT_PTR), ud, fl);
  emitir(IRT(IR_FSTORE, IRT_PTR), fref, val);
}

static LJ_AINLINE TRef recff_sbufx_len(jit_State *J, TRef trr, TRef trw)
{
  TRef len = emitir(IRT(IR_SUB, IRT_INTP), trw, trr);
  if (LJ_64)
    len = emitir(IRTI(IR_CONV), len, (IRT_INT<<5)|IRT_INTP|IRCONV_NONE);
  return len;
}

/* Emit typecheck for string buffer. */
static TRef recff_sbufx_check(jit_State *J, RecordFFData *rd, ptrdiff_t arg)
{
  TRef trtype, ud = J->base[arg];
  if (!tvisbuf(&rd->argv[arg])) lj_trace_err(J, LJ_TRERR_BADTYPE);
  trtype = emitir(IRT(IR_FLOAD, IRT_U8), ud, IRFL_UDATA_UDTYPE);
  emitir(IRTGI(IR_EQ), trtype, lj_ir_kint(J, UDTYPE_BUFFER));
  J->needsnap = 1;
  return ud;
}

/* Emit BUFHDR for write to extended string buffer. */
static TRef recff_sbufx_write(jit_State *J, TRef ud)
{
  TRef trbuf = emitir(IRT(IR_ADD, IRT_PGC), ud, lj_ir_kintpgc(J, sizeof(GCudata)));
  return emitir(IRT(IR_BUFHDR, IRT_PGC), trbuf, IRBUFHDR_WRITE);
}

/* Check for integer in range for the buffer API. */
static TRef recff_sbufx_checkint(jit_State *J, RecordFFData *rd, ptrdiff_t arg)
{
  TRef tr = J->base[arg];
  TRef trlim = lj_ir_kint(J, LJ_MAX_BUF);
  if (tref_isinteger(tr)) {
    emitir(IRTGI(IR_ULE), tr, trlim);
  } else if (tref_isnum(tr)) {
    tr = emitir(IRTI(IR_CONV), tr, IRCONV_INT_NUM|IRCONV_ANY);
    emitir(IRTGI(IR_ULE), tr, trlim);
#if LJ_HASFFI
  } else if (tref_iscdata(tr)) {
    tr = lj_crecord_loadiu64(J, tr, &rd->argv[arg]);
    emitir(IRTG(IR_ULE, IRT_U64), tr, lj_ir_kint64(J, LJ_MAX_BUF));
    tr = emitir(IRTI(IR_CONV), tr, (IRT_INT<<5)|IRT_I64|IRCONV_NONE);
#else
    UNUSED(rd);
#endif
  } else {
    lj_trace_err(J, LJ_TRERR_BADTYPE);
  }
  return tr;
}

static void LJ_FASTCALL recff_buffer_method_reset(jit_State *J, RecordFFData *rd)
{
  TRef ud = recff_sbufx_check(J, rd, 0);
  SBufExt *sbx = bufV(&rd->argv[0]);
  int iscow = (int)sbufiscow(sbx);
  TRef trl = recff_sbufx_get_L(J, ud);
  TRef trcow = emitir(IRT(IR_BAND, IRT_IGC), trl, lj_ir_kintpgc(J, SBUF_FLAG_COW));
  TRef zeropgc = lj_ir_kintpgc(J, 0);
  emitir(IRTG(iscow ? IR_NE : IR_EQ, IRT_IGC), trcow, zeropgc);
  if (iscow) {
    TRef zerop = lj_ir_kintp(J, 0);
    trl = emitir(IRT(IR_BXOR, IRT_IGC), trl, lj_ir_kintpgc(J, SBUF_FLAG_COW));
    recff_sbufx_set_ptr(J, ud, IRFL_SBUF_W, zerop);
    recff_sbufx_set_ptr(J, ud, IRFL_SBUF_E, zerop);
    recff_sbufx_set_ptr(J, ud, IRFL_SBUF_B, zerop);
    recff_sbufx_set_L(J, ud, trl);
    emitir(IRT(IR_FSTORE, IRT_PGC),
	   emitir(IRT(IR_FREF, IRT_PGC), ud, IRFL_SBUF_REF), zeropgc);
    recff_sbufx_set_ptr(J, ud, IRFL_SBUF_R, zerop);
  } else {
    TRef trb = recff_sbufx_get_ptr(J, ud, IRFL_SBUF_B);
    recff_sbufx_set_ptr(J, ud, IRFL_SBUF_W, trb);
    recff_sbufx_set_ptr(J, ud, IRFL_SBUF_R, trb);
  }
}

static void LJ_FASTCALL recff_buffer_method_skip(jit_State *J, RecordFFData *rd)
{
  TRef ud = recff_sbufx_check(J, rd, 0);
  TRef trr = recff_sbufx_get_ptr(J, ud, IRFL_SBUF_R);
  TRef trw = recff_sbufx_get_ptr(J, ud, IRFL_SBUF_W);
  TRef len = recff_sbufx_len(J, trr, trw);
  TRef trn = recff_sbufx_checkint(J, rd, 1);
  len = emitir(IRTI(IR_MIN), len, trn);
  trr = emitir(IRT(IR_ADD, IRT_PTR), trr, len);
  recff_sbufx_set_ptr(J, ud, IRFL_SBUF_R, trr);
}

static void LJ_FASTCALL recff_buffer_method_set(jit_State *J, RecordFFData *rd)
{
  TRef ud = recff_sbufx_check(J, rd, 0);
  TRef trbuf = recff_sbufx_write(J, ud);
  TRef tr = J->base[1];
  if (tref_isstr(tr)) {
    TRef trp = emitir(IRT(IR_STRREF, IRT_PGC), tr, lj_ir_kint(J, 0));
    TRef len = emitir(IRTI(IR_FLOAD), tr, IRFL_STR_LEN);
    IRIns *irp = IR(tref_ref(trp));
    /* trp must point into the anchored obj, even after folding. */
    if (irp->o == IR_STRREF)
      tr = irp->op1;
    else if (!tref_isk(tr))
      trp = emitir(IRT(IR_ADD, IRT_PGC), tr, lj_ir_kintpgc(J, sizeof(GCstr)));
    lj_ir_call(J, IRCALL_lj_bufx_set, trbuf, trp, len, tr);
#if LJ_HASFFI
  } else if (tref_iscdata(tr)) {
    TRef trp = lj_crecord_topcvoid(J, tr, &rd->argv[1]);
    TRef len = recff_sbufx_checkint(J, rd, 2);
    lj_ir_call(J, IRCALL_lj_bufx_set, trbuf, trp, len, tr);
#endif
  }  /* else: Interpreter will throw. */
}

static void LJ_FASTCALL recff_buffer_method_put(jit_State *J, RecordFFData *rd)
{
  TRef ud = recff_sbufx_check(J, rd, 0);
  TRef trbuf = recff_sbufx_write(J, ud);
  TRef tr;
  ptrdiff_t arg;
  if (!J->base[1]) return;
  for (arg = 1; (tr = J->base[arg]); arg++) {
    if (tref_isudata(tr)) {
      TRef ud2 = recff_sbufx_check(J, rd, arg);
      emitir(IRTG(IR_NE, IRT_PGC), ud, ud2);
    }
  }
  for (arg = 1; (tr = J->base[arg]); arg++) {
    if (tref_isstr(tr)) {
      trbuf = emitir(IRTG(IR_BUFPUT, IRT_PGC), trbuf, tr);
    } else if (tref_isnumber(tr)) {
      trbuf = emitir(IRTG(IR_BUFPUT, IRT_PGC), trbuf,
		     emitir(IRT(IR_TOSTR, IRT_STR), tr,
			    tref_isnum(tr) ? IRTOSTR_NUM : IRTOSTR_INT));
    } else if (tref_isudata(tr)) {
      TRef trr = recff_sbufx_get_ptr(J, tr, IRFL_SBUF_R);
      TRef trw = recff_sbufx_get_ptr(J, tr, IRFL_SBUF_W);
      TRef len = recff_sbufx_len(J, trr, trw);
      trbuf = lj_ir_call(J, IRCALL_lj_buf_putmem, trbuf, trr, len);
    } else {
      recff_nyiu(J, rd);
    }
  }
  emitir(IRT(IR_USE, IRT_NIL), trbuf, 0);
}

static void LJ_FASTCALL recff_buffer_method_putf(jit_State *J, RecordFFData *rd)
{
  TRef ud = recff_sbufx_check(J, rd, 0);
  TRef trbuf = recff_sbufx_write(J, ud);
  recff_format(J, rd, trbuf, 1);
}

static void LJ_FASTCALL recff_buffer_method_get(jit_State *J, RecordFFData *rd)
{
  TRef ud = recff_sbufx_check(J, rd, 0);
  TRef trr = recff_sbufx_get_ptr(J, ud, IRFL_SBUF_R);
  TRef trw = recff_sbufx_get_ptr(J, ud, IRFL_SBUF_W);
  TRef tr;
  ptrdiff_t arg;
  if (!J->base[1]) { J->base[1] = TREF_NIL; J->base[2] = 0; }
  for (arg = 0; (tr = J->base[arg+1]); arg++) {
    if (!tref_isnil(tr)) {
      J->base[arg+1] = recff_sbufx_checkint(J, rd, arg+1);
    }
  }
  for (arg = 0; (tr = J->base[arg+1]); arg++) {
    TRef len = recff_sbufx_len(J, trr, trw);
    if (tref_isnil(tr)) {
      J->base[arg] = emitir(IRT(IR_XSNEW, IRT_STR), trr, len);
      trr = trw;
    } else {
      TRef tru;
      len = emitir(IRTI(IR_MIN), len, tr);
      tru = emitir(IRT(IR_ADD, IRT_PTR), trr, len);
      J->base[arg] = emitir(IRT(IR_XSNEW, IRT_STR), trr, len);
      trr = tru;  /* Doing the ADD before the SNEW generates better code. */
    }
    recff_sbufx_set_ptr(J, ud, IRFL_SBUF_R, trr);
  }
  rd->nres = arg;
}

static void LJ_FASTCALL recff_buffer_method___tostring(jit_State *J, RecordFFData *rd)
{
  TRef ud = recff_sbufx_check(J, rd, 0);
  TRef trr = recff_sbufx_get_ptr(J, ud, IRFL_SBUF_R);
  TRef trw = recff_sbufx_get_ptr(J, ud, IRFL_SBUF_W);
  J->base[0] = emitir(IRT(IR_XSNEW, IRT_STR), trr, recff_sbufx_len(J, trr, trw));
}

static void LJ_FASTCALL recff_buffer_method___len(jit_State *J, RecordFFData *rd)
{
  TRef ud = recff_sbufx_check(J, rd, 0);
  TRef trr = recff_sbufx_get_ptr(J, ud, IRFL_SBUF_R);
  TRef trw = recff_sbufx_get_ptr(J, ud, IRFL_SBUF_W);
  J->base[0] = recff_sbufx_len(J, trr, trw);
}

#if LJ_HASFFI
static void LJ_FASTCALL recff_buffer_method_putcdata(jit_State *J, RecordFFData *rd)
{
  TRef ud = recff_sbufx_check(J, rd, 0);
  TRef trbuf = recff_sbufx_write(J, ud);
  TRef tr = lj_crecord_topcvoid(J, J->base[1], &rd->argv[1]);
  TRef len = recff_sbufx_checkint(J, rd, 2);
  trbuf = lj_ir_call(J, IRCALL_lj_buf_putmem, trbuf, tr, len);
  emitir(IRT(IR_USE, IRT_NIL), trbuf, 0);
}

static void LJ_FASTCALL recff_buffer_method_reserve(jit_State *J, RecordFFData *rd)
{
  TRef ud = recff_sbufx_check(J, rd, 0);
  TRef trbuf = recff_sbufx_write(J, ud);
  TRef trsz = recff_sbufx_checkint(J, rd, 1);
  J->base[1] = lj_ir_call(J, IRCALL_lj_bufx_more, trbuf, trsz);
  J->base[0] = lj_crecord_topuint8(J, recff_sbufx_get_ptr(J, ud, IRFL_SBUF_W));
  rd->nres = 2;
}

static void LJ_FASTCALL recff_buffer_method_commit(jit_State *J, RecordFFData *rd)
{
  TRef ud = recff_sbufx_check(J, rd, 0);
  TRef len = recff_sbufx_checkint(J, rd, 1);
  TRef trw = recff_sbufx_get_ptr(J, ud, IRFL_SBUF_W);
  TRef tre = recff_sbufx_get_ptr(J, ud, IRFL_SBUF_E);
  TRef left = emitir(IRT(IR_SUB, IRT_INTP), tre, trw);
  if (LJ_64)
    left = emitir(IRTI(IR_CONV), left, (IRT_INT<<5)|IRT_INTP|IRCONV_NONE);
  emitir(IRTGI(IR_ULE), len, left);
  trw = emitir(IRT(IR_ADD, IRT_PTR), trw, len);
  recff_sbufx_set_ptr(J, ud, IRFL_SBUF_W, trw);
}

static void LJ_FASTCALL recff_buffer_method_ref(jit_State *J, RecordFFData *rd)
{
  TRef ud = recff_sbufx_check(J, rd, 0);
  TRef trr = recff_sbufx_get_ptr(J, ud, IRFL_SBUF_R);
  TRef trw = recff_sbufx_get_ptr(J, ud, IRFL_SBUF_W);
  J->base[0] = lj_crecord_topuint8(J, trr);
  J->base[1] = recff_sbufx_len(J, trr, trw);
  rd->nres = 2;
}
#endif

static void LJ_FASTCALL recff_buffer_method_encode(jit_State *J, RecordFFData *rd)
{
  TRef ud = recff_sbufx_check(J, rd, 0);
  TRef trbuf = recff_sbufx_write(J, ud);
  TRef tmp = recff_tmpref(J, J->base[1], IRTMPREF_IN1);
  lj_ir_call(J, IRCALL_lj_serialize_put, trbuf, tmp);
  /* No IR_USE needed, since the call is a store. */
}

static void LJ_FASTCALL recff_buffer_method_decode(jit_State *J, RecordFFData *rd)
{
  TRef ud = recff_sbufx_check(J, rd, 0);
  TRef trbuf = recff_sbufx_write(J, ud);
  TRef tmp = recff_tmpref(J, TREF_NIL, IRTMPREF_OUT1);
  TRef trr = lj_ir_call(J, IRCALL_lj_serialize_get, trbuf, tmp);
  IRType t = (IRType)lj_serialize_peektype(bufV(&rd->argv[0]));
  /* No IR_USE needed, since the call is a store. */
  J->base[0] = lj_record_vload(J, tmp, 0, t);
  /* The sbx->r store must be after the VLOAD type check, in case it fails. */
  recff_sbufx_set_ptr(J, ud, IRFL_SBUF_R, trr);
}

static void LJ_FASTCALL recff_buffer_encode(jit_State *J, RecordFFData *rd)
{
  TRef tmp = recff_tmpref(J, J->base[0], IRTMPREF_IN1);
  J->base[0] = lj_ir_call(J, IRCALL_lj_serialize_encode, tmp);
  /* IR_USE needed for IR_CALLA, because the encoder may throw non-OOM. */
  emitir(IRT(IR_USE, IRT_NIL), J->base[0], 0);
  UNUSED(rd);
}

static void LJ_FASTCALL recff_buffer_decode(jit_State *J, RecordFFData *rd)
{
  if (tvisstr(&rd->argv[0])) {
    GCstr *str = strV(&rd->argv[0]);
    SBufExt sbx;
    IRType t;
    TRef tmp = recff_tmpref(J, TREF_NIL, IRTMPREF_OUT1);
    TRef tr = lj_ir_call(J, IRCALL_lj_serialize_decode, tmp, J->base[0]);
    /* IR_USE needed for IR_CALLA, because the decoder may throw non-OOM.
    ** That's why IRCALL_lj_serialize_decode needs a fake INT result.
    */
    emitir(IRT(IR_USE, IRT_NIL), tr, 0);
    memset(&sbx, 0, sizeof(SBufExt));
    lj_bufx_set_cow(J->L, &sbx, strdata(str), str->len);
    t = (IRType)lj_serialize_peektype(&sbx);
    J->base[0] = lj_record_vload(J, tmp, 0, t);
  }  /* else: Interpreter will throw. */
}

#endif

/* -- Table library fast functions ---------------------------------------- */

static void LJ_FASTCALL recff_table_insert(jit_State *J, RecordFFData *rd)
{
  RecordIndex ix;
  ix.tab = J->base[0];
  ix.val = J->base[1];
  rd->nres = 0;
  if (tref_istab(ix.tab) && ix.val) {
    if (!J->base[2]) {  /* Simple push: t[#t+1] = v */
      TRef trlen = emitir(IRTI(IR_ALEN), ix.tab, TREF_NIL);
      GCtab *t = tabV(&rd->argv[0]);
      ix.key = emitir(IRTI(IR_ADD), trlen, lj_ir_kint(J, 1));
      settabV(J->L, &ix.tabv, t);
      setintV(&ix.keyv, lj_tab_len(t) + 1);
      ix.idxchain = 0;
      lj_record_idx(J, &ix);  /* Set new value. */
    } else {  /* Complex case: insert in the middle. */
      recff_nyiu(J, rd);
      return;
    }
  }  /* else: Interpreter will throw. */
}

static void LJ_FASTCALL recff_table_concat(jit_State *J, RecordFFData *rd)
{
  TRef tab = J->base[0];
  if (tref_istab(tab)) {
    TRef sep = !tref_isnil(J->base[1]) ?
	       lj_ir_tostr(J, J->base[1]) : lj_ir_knull(J, IRT_STR);
    TRef tri = (J->base[1] && !tref_isnil(J->base[2])) ?
	       lj_opt_narrow_toint(J, J->base[2]) : lj_ir_kint(J, 1);
    TRef tre = (J->base[1] && J->base[2] && !tref_isnil(J->base[3])) ?
	       lj_opt_narrow_toint(J, J->base[3]) :
	       emitir(IRTI(IR_ALEN), tab, TREF_NIL);
    TRef hdr = recff_bufhdr(J);
    TRef tr = lj_ir_call(J, IRCALL_lj_buf_puttab, hdr, tab, sep, tri, tre);
    emitir(IRTG(IR_NE, IRT_PTR), tr, lj_ir_kptr(J, NULL));
    J->base[0] = emitir(IRTG(IR_BUFSTR, IRT_STR), tr, hdr);
  }  /* else: Interpreter will throw. */
  UNUSED(rd);
}

static void LJ_FASTCALL recff_table_new(jit_State *J, RecordFFData *rd)
{
  TRef tra = lj_opt_narrow_toint(J, J->base[0]);
  TRef trh = lj_opt_narrow_toint(J, J->base[1]);
  if (tref_isk(tra) && tref_isk(trh)) {
    int32_t a = IR(tref_ref(tra))->i;
    if (a < 0x7fff) {
      uint32_t hbits = hsize2hbits(IR(tref_ref(trh))->i);
      a = a > 0 ? a+1 : 0;
      J->base[0] = emitir(IRTG(IR_TNEW, IRT_TAB), (uint32_t)a, hbits);
      return;
    }
  }
  J->base[0] = lj_ir_call(J, IRCALL_lj_tab_new_ah, tra, trh);
  UNUSED(rd);
}

static void LJ_FASTCALL recff_table_clear(jit_State *J, RecordFFData *rd)
{
  TRef tr = J->base[0];
  if (tref_istab(tr)) {
    rd->nres = 0;
    lj_ir_call(J, IRCALL_lj_tab_clear, tr);
    J->needsnap = 1;
  }  /* else: Interpreter will throw. */
}

/* -- I/O library fast functions ------------------------------------------ */

/* Get FILE* for I/O function. Any I/O error aborts recording, so there's
** no need to encode the alternate cases for any of the guards.
*/
static TRef recff_io_fp(jit_State *J, TRef *udp, int32_t id)
{
  TRef tr, ud, fp;
  if (id) {  /* io.func() */
    ud = lj_ir_ggfload(J, IRT_UDATA, GG_OFS(g.gcroot[id]));
  } else {  /* fp:method() */
    ud = J->base[0];
    if (!tref_isudata(ud))
      lj_trace_err(J, LJ_TRERR_BADTYPE);
    tr = emitir(IRT(IR_FLOAD, IRT_U8), ud, IRFL_UDATA_UDTYPE);
    emitir(IRTGI(IR_EQ), tr, lj_ir_kint(J, UDTYPE_IO_FILE));
  }
  *udp = ud;
  fp = emitir(IRT(IR_FLOAD, IRT_PTR), ud, IRFL_UDATA_FILE);
  emitir(IRTG(IR_NE, IRT_PTR), fp, lj_ir_knull(J, IRT_PTR));
  return fp;
}

static void LJ_FASTCALL recff_io_write(jit_State *J, RecordFFData *rd)
{
  TRef ud, fp = recff_io_fp(J, &ud, rd->data);
  TRef zero = lj_ir_kint(J, 0);
  TRef one = lj_ir_kint(J, 1);
  ptrdiff_t i = rd->data == 0 ? 1 : 0;
  for (; J->base[i]; i++) {
    TRef str = lj_ir_tostr(J, J->base[i]);
    TRef buf = emitir(IRT(IR_STRREF, IRT_PGC), str, zero);
    TRef len = emitir(IRTI(IR_FLOAD), str, IRFL_STR_LEN);
    if (tref_isk(len) && IR(tref_ref(len))->i == 1) {
      IRIns *irs = IR(tref_ref(str));
      TRef tr = (irs->o == IR_TOSTR && irs->op2 == IRTOSTR_CHAR) ?
		irs->op1 :
		emitir(IRT(IR_XLOAD, IRT_U8), buf, IRXLOAD_READONLY);
      tr = lj_ir_call(J, IRCALL_fputc, tr, fp);
      if (results_wanted(J) != 0)  /* Check result only if not ignored. */
	emitir(IRTGI(IR_NE), tr, lj_ir_kint(J, -1));
    } else {
      TRef tr = lj_ir_call(J, IRCALL_fwrite, buf, one, len, fp);
      if (results_wanted(J) != 0)  /* Check result only if not ignored. */
	emitir(IRTGI(IR_EQ), tr, len);
    }
  }
  J->base[0] = LJ_52 ? ud : TREF_TRUE;
}

static void LJ_FASTCALL recff_io_flush(jit_State *J, RecordFFData *rd)
{
  TRef ud, fp = recff_io_fp(J, &ud, rd->data);
  TRef tr = lj_ir_call(J, IRCALL_fflush, fp);
  if (results_wanted(J) != 0)  /* Check result only if not ignored. */
    emitir(IRTGI(IR_EQ), tr, lj_ir_kint(J, 0));
  J->base[0] = TREF_TRUE;
}

/* -- Debug library fast functions ---------------------------------------- */

static void LJ_FASTCALL recff_debug_getmetatable(jit_State *J, RecordFFData *rd)
{
  GCtab *mt;
  TRef mtref;
  TRef tr = J->base[0];
  if (tref_istab(tr)) {
    mt = tabref(tabV(&rd->argv[0])->metatable);
    mtref = emitir(IRT(IR_FLOAD, IRT_TAB), tr, IRFL_TAB_META);
  } else if (tref_isudata(tr)) {
    mt = tabref(udataV(&rd->argv[0])->metatable);
    mtref = emitir(IRT(IR_FLOAD, IRT_TAB), tr, IRFL_UDATA_META);
  } else {
    mt = tabref(basemt_obj(J2G(J), &rd->argv[0]));
    J->base[0] = mt ? lj_ir_ktab(J, mt) : TREF_NIL;
    return;
  }
  emitir(IRTG(mt ? IR_NE : IR_EQ, IRT_TAB), mtref, lj_ir_knull(J, IRT_TAB));
  J->base[0] = mt ? mtref : TREF_NIL;
}

/* -- Record calls to fast functions -------------------------------------- */

#include "lj_recdef.h"

static uint32_t recdef_lookup(GCfunc *fn)
{
  if (fn->c.ffid < sizeof(recff_idmap)/sizeof(recff_idmap[0]))
    return recff_idmap[fn->c.ffid];
  else
    return 0;
}

/* Record entry to a fast function or C function. */
void lj_ffrecord_func(jit_State *J)
{
  RecordFFData rd;
  uint32_t m = recdef_lookup(J->fn);
  rd.data = m & 0xff;
  rd.nres = 1;  /* Default is one result. */
  rd.argv = J->L->base;
  J->base[J->maxslot] = 0;  /* Mark end of arguments. */
  (recff_func[m >> 8])(J, &rd);  /* Call recff_* handler. */
  if (rd.nres >= 0) {
    if (J->postproc == LJ_POST_NONE) J->postproc = LJ_POST_FFRETRY;
    lj_record_ret(J, 0, rd.nres);
  }
}

#undef IR
#undef emitir

#endif
