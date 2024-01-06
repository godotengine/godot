/*
** Snapshot handling.
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
*/

#define lj_snap_c
#define LUA_CORE

#include "lj_obj.h"

#if LJ_HASJIT

#include "lj_gc.h"
#include "lj_tab.h"
#include "lj_state.h"
#include "lj_frame.h"
#include "lj_bc.h"
#include "lj_ir.h"
#include "lj_jit.h"
#include "lj_iropt.h"
#include "lj_trace.h"
#include "lj_snap.h"
#include "lj_target.h"
#if LJ_HASFFI
#include "lj_ctype.h"
#include "lj_cdata.h"
#endif

/* Pass IR on to next optimization in chain (FOLD). */
#define emitir(ot, a, b)	(lj_ir_set(J, (ot), (a), (b)), lj_opt_fold(J))

/* Emit raw IR without passing through optimizations. */
#define emitir_raw(ot, a, b)	(lj_ir_set(J, (ot), (a), (b)), lj_ir_emit(J))

/* -- Snapshot buffer allocation ------------------------------------------ */

/* Grow snapshot buffer. */
void lj_snap_grow_buf_(jit_State *J, MSize need)
{
  MSize maxsnap = (MSize)J->param[JIT_P_maxsnap];
  if (need > maxsnap)
    lj_trace_err(J, LJ_TRERR_SNAPOV);
  lj_mem_growvec(J->L, J->snapbuf, J->sizesnap, maxsnap, SnapShot);
  J->cur.snap = J->snapbuf;
}

/* Grow snapshot map buffer. */
void lj_snap_grow_map_(jit_State *J, MSize need)
{
  if (need < 2*J->sizesnapmap)
    need = 2*J->sizesnapmap;
  else if (need < 64)
    need = 64;
  J->snapmapbuf = (SnapEntry *)lj_mem_realloc(J->L, J->snapmapbuf,
		    J->sizesnapmap*sizeof(SnapEntry), need*sizeof(SnapEntry));
  J->cur.snapmap = J->snapmapbuf;
  J->sizesnapmap = need;
}

/* -- Snapshot generation ------------------------------------------------- */

/* Add all modified slots to the snapshot. */
static MSize snapshot_slots(jit_State *J, SnapEntry *map, BCReg nslots)
{
  IRRef retf = J->chain[IR_RETF];  /* Limits SLOAD restore elimination. */
  BCReg s;
  MSize n = 0;
  for (s = 0; s < nslots; s++) {
    TRef tr = J->slot[s];
    IRRef ref = tref_ref(tr);
#if LJ_FR2
    if (s == 1) {  /* Ignore slot 1 in LJ_FR2 mode, except if tailcalled. */
      if ((tr & TREF_FRAME))
	map[n++] = SNAP(1, SNAP_FRAME | SNAP_NORESTORE, REF_NIL);
      continue;
    }
    if ((tr & (TREF_FRAME | TREF_CONT)) && !ref) {
      cTValue *base = J->L->base - J->baseslot;
      tr = J->slot[s] = (tr & 0xff0000) | lj_ir_k64(J, IR_KNUM, base[s].u64);
      ref = tref_ref(tr);
    }
#endif
    if (ref) {
      SnapEntry sn = SNAP_TR(s, tr);
      IRIns *ir = &J->cur.ir[ref];
      if ((LJ_FR2 || !(sn & (SNAP_CONT|SNAP_FRAME))) &&
	  ir->o == IR_SLOAD && ir->op1 == s && ref > retf) {
	/*
	** No need to snapshot unmodified non-inherited slots.
	** But always snapshot the function below a frame in LJ_FR2 mode.
	*/
	if (!(ir->op2 & IRSLOAD_INHERIT) &&
	    (!LJ_FR2 || s == 0 || s+1 == nslots ||
	     !(J->slot[s+1] & (TREF_CONT|TREF_FRAME))))
	  continue;
	/* No need to restore readonly slots and unmodified non-parent slots. */
	if (!(LJ_DUALNUM && (ir->op2 & IRSLOAD_CONVERT)) &&
	    (ir->op2 & (IRSLOAD_READONLY|IRSLOAD_PARENT)) != IRSLOAD_PARENT)
	  sn |= SNAP_NORESTORE;
      }
      if (LJ_SOFTFP32 && irt_isnum(ir->t))
	sn |= SNAP_SOFTFPNUM;
      map[n++] = sn;
    }
  }
  return n;
}

/* Add frame links at the end of the snapshot. */
static MSize snapshot_framelinks(jit_State *J, SnapEntry *map, uint8_t *topslot)
{
  cTValue *frame = J->L->base - 1;
  cTValue *lim = J->L->base - J->baseslot + LJ_FR2;
  GCfunc *fn = frame_func(frame);
  cTValue *ftop = isluafunc(fn) ? (frame+funcproto(fn)->framesize) : J->L->top;
#if LJ_FR2
  uint64_t pcbase = (u64ptr(J->pc) << 8) | (J->baseslot - 2);
  lj_assertJ(2 <= J->baseslot && J->baseslot <= 257, "bad baseslot");
  memcpy(map, &pcbase, sizeof(uint64_t));
#else
  MSize f = 0;
  map[f++] = SNAP_MKPC(J->pc);  /* The current PC is always the first entry. */
#endif
  lj_assertJ(!J->pt ||
	     (J->pc >= proto_bc(J->pt) &&
	      J->pc < proto_bc(J->pt) + J->pt->sizebc), "bad snapshot PC");
  while (frame > lim) {  /* Backwards traversal of all frames above base. */
    if (frame_islua(frame)) {
#if !LJ_FR2
      map[f++] = SNAP_MKPC(frame_pc(frame));
#endif
      frame = frame_prevl(frame);
    } else if (frame_iscont(frame)) {
#if !LJ_FR2
      map[f++] = SNAP_MKFTSZ(frame_ftsz(frame));
      map[f++] = SNAP_MKPC(frame_contpc(frame));
#endif
      frame = frame_prevd(frame);
    } else {
      lj_assertJ(!frame_isc(frame), "broken frame chain");
#if !LJ_FR2
      map[f++] = SNAP_MKFTSZ(frame_ftsz(frame));
#endif
      frame = frame_prevd(frame);
      continue;
    }
    if (frame + funcproto(frame_func(frame))->framesize > ftop)
      ftop = frame + funcproto(frame_func(frame))->framesize;
  }
  *topslot = (uint8_t)(ftop - lim);
#if LJ_FR2
  lj_assertJ(sizeof(SnapEntry) * 2 == sizeof(uint64_t), "bad SnapEntry def");
  return 2;
#else
  lj_assertJ(f == (MSize)(1 + J->framedepth), "miscalculated snapshot size");
  return f;
#endif
}

/* Take a snapshot of the current stack. */
static void snapshot_stack(jit_State *J, SnapShot *snap, MSize nsnapmap)
{
  BCReg nslots = J->baseslot + J->maxslot;
  MSize nent;
  SnapEntry *p;
  /* Conservative estimate. */
  lj_snap_grow_map(J, nsnapmap + nslots + (MSize)(LJ_FR2?2:J->framedepth+1));
  p = &J->cur.snapmap[nsnapmap];
  nent = snapshot_slots(J, p, nslots);
  snap->nent = (uint8_t)nent;
  nent += snapshot_framelinks(J, p + nent, &snap->topslot);
  snap->mapofs = (uint32_t)nsnapmap;
  snap->ref = (IRRef1)J->cur.nins;
  snap->mcofs = 0;
  snap->nslots = (uint8_t)nslots;
  snap->count = 0;
  J->cur.nsnapmap = (uint32_t)(nsnapmap + nent);
}

/* Add or merge a snapshot. */
void lj_snap_add(jit_State *J)
{
  MSize nsnap = J->cur.nsnap;
  MSize nsnapmap = J->cur.nsnapmap;
  /* Merge if no ins. inbetween or if requested and no guard inbetween. */
  if ((nsnap > 0 && J->cur.snap[nsnap-1].ref == J->cur.nins) ||
      (J->mergesnap && !irt_isguard(J->guardemit))) {
    if (nsnap == 1) {  /* But preserve snap #0 PC. */
      emitir_raw(IRT(IR_NOP, IRT_NIL), 0, 0);
      goto nomerge;
    }
    nsnapmap = J->cur.snap[--nsnap].mapofs;
  } else {
  nomerge:
    lj_snap_grow_buf(J, nsnap+1);
    J->cur.nsnap = (uint16_t)(nsnap+1);
  }
  J->mergesnap = 0;
  J->guardemit.irt = 0;
  snapshot_stack(J, &J->cur.snap[nsnap], nsnapmap);
}

/* -- Snapshot modification ----------------------------------------------- */

#define SNAP_USEDEF_SLOTS	(LJ_MAX_JSLOTS+LJ_STACK_EXTRA)

/* Find unused slots with reaching-definitions bytecode data-flow analysis. */
static BCReg snap_usedef(jit_State *J, uint8_t *udf,
			 const BCIns *pc, BCReg maxslot)
{
  BCReg s;
  GCobj *o;

  if (maxslot == 0) return 0;
#ifdef LUAJIT_USE_VALGRIND
  /* Avoid errors for harmless reads beyond maxslot. */
  memset(udf, 1, SNAP_USEDEF_SLOTS);
#else
  memset(udf, 1, maxslot);
#endif

  /* Treat open upvalues as used. */
  o = gcref(J->L->openupval);
  while (o) {
    if (uvval(gco2uv(o)) < J->L->base) break;
    udf[uvval(gco2uv(o)) - J->L->base] = 0;
    o = gcref(o->gch.nextgc);
  }

#define USE_SLOT(s)		udf[(s)] &= ~1
#define DEF_SLOT(s)		udf[(s)] *= 3

  /* Scan through following bytecode and check for uses/defs. */
  lj_assertJ(pc >= proto_bc(J->pt) && pc < proto_bc(J->pt) + J->pt->sizebc,
	     "snapshot PC out of range");
  for (;;) {
    BCIns ins = *pc++;
    BCOp op = bc_op(ins);
    switch (bcmode_b(op)) {
    case BCMvar: USE_SLOT(bc_b(ins)); break;
    default: break;
    }
    switch (bcmode_c(op)) {
    case BCMvar: USE_SLOT(bc_c(ins)); break;
    case BCMrbase:
      lj_assertJ(op == BC_CAT, "unhandled op %d with RC rbase", op);
      for (s = bc_b(ins); s <= bc_c(ins); s++) USE_SLOT(s);
      for (; s < maxslot; s++) DEF_SLOT(s);
      break;
    case BCMjump:
    handle_jump: {
      BCReg minslot = bc_a(ins);
      if (op >= BC_FORI && op <= BC_JFORL) minslot += FORL_EXT;
      else if (op >= BC_ITERL && op <= BC_JITERL) minslot += bc_b(pc[-2])-1;
      else if (op == BC_UCLO) {
	ptrdiff_t delta = bc_j(ins);
	if (delta < 0) return maxslot;  /* Prevent loop. */
	pc += delta;
	break;
      }
      for (s = minslot; s < maxslot; s++) DEF_SLOT(s);
      return minslot < maxslot ? minslot : maxslot;
      }
    case BCMlit:
      if (op == BC_JFORL || op == BC_JITERL || op == BC_JLOOP) {
	goto handle_jump;
      } else if (bc_isret(op)) {
	BCReg top = op == BC_RETM ? maxslot : (bc_a(ins) + bc_d(ins)-1);
	for (s = 0; s < bc_a(ins); s++) DEF_SLOT(s);
	for (; s < top; s++) USE_SLOT(s);
	for (; s < maxslot; s++) DEF_SLOT(s);
	return 0;
      }
      break;
    case BCMfunc: return maxslot;  /* NYI: will abort, anyway. */
    default: break;
    }
    switch (bcmode_a(op)) {
    case BCMvar: USE_SLOT(bc_a(ins)); break;
    case BCMdst:
       if (!(op == BC_ISTC || op == BC_ISFC)) DEF_SLOT(bc_a(ins));
       break;
    case BCMbase:
      if (op >= BC_CALLM && op <= BC_ITERN) {
	BCReg top = (op == BC_CALLM || op == BC_CALLMT || bc_c(ins) == 0) ?
		    maxslot : (bc_a(ins) + bc_c(ins)+LJ_FR2);
	if (LJ_FR2) DEF_SLOT(bc_a(ins)+1);
	s = bc_a(ins) - ((op == BC_ITERC || op == BC_ITERN) ? 3 : 0);
	for (; s < top; s++) USE_SLOT(s);
	for (; s < maxslot; s++) DEF_SLOT(s);
	if (op == BC_CALLT || op == BC_CALLMT) {
	  for (s = 0; s < bc_a(ins); s++) DEF_SLOT(s);
	  return 0;
	}
      } else if (op == BC_VARG) {
	return maxslot;  /* NYI: punt. */
      } else if (op == BC_KNIL) {
	for (s = bc_a(ins); s <= bc_d(ins); s++) DEF_SLOT(s);
      } else if (op == BC_TSETM) {
	for (s = bc_a(ins)-1; s < maxslot; s++) USE_SLOT(s);
      }
      break;
    default: break;
    }
    lj_assertJ(pc >= proto_bc(J->pt) && pc < proto_bc(J->pt) + J->pt->sizebc,
	       "use/def analysis PC out of range");
  }

#undef USE_SLOT
#undef DEF_SLOT

  return 0;  /* unreachable */
}

/* Mark slots used by upvalues of child prototypes as used. */
static void snap_useuv(GCproto *pt, uint8_t *udf)
{
  /* This is a coarse check, because it's difficult to correlate the lifetime
  ** of slots and closures. But the number of false positives is quite low.
  ** A false positive may cause a slot not to be purged, which is just
  ** a missed optimization.
  */
  if ((pt->flags & PROTO_CHILD)) {
    ptrdiff_t i, j, n = pt->sizekgc;
    GCRef *kr = mref(pt->k, GCRef) - 1;
    for (i = 0; i < n; i++, kr--) {
      GCobj *o = gcref(*kr);
      if (o->gch.gct == ~LJ_TPROTO) {
	for (j = 0; j < gco2pt(o)->sizeuv; j++) {
	  uint32_t v = proto_uv(gco2pt(o))[j];
	  if ((v & PROTO_UV_LOCAL)) {
	    udf[(v & 0xff)] = 0;
	  }
	}
      }
    }
  }
}

/* Purge dead slots before the next snapshot. */
void lj_snap_purge(jit_State *J)
{
  uint8_t udf[SNAP_USEDEF_SLOTS];
  BCReg s, maxslot = J->maxslot;
  if (bc_op(*J->pc) == BC_FUNCV && maxslot > J->pt->numparams)
    maxslot = J->pt->numparams;
  s = snap_usedef(J, udf, J->pc, maxslot);
  if (s < maxslot) {
    snap_useuv(J->pt, udf);
    for (; s < maxslot; s++)
      if (udf[s] != 0)
	J->base[s] = 0;  /* Purge dead slots. */
  }
}

/* Shrink last snapshot. */
void lj_snap_shrink(jit_State *J)
{
  SnapShot *snap = &J->cur.snap[J->cur.nsnap-1];
  SnapEntry *map = &J->cur.snapmap[snap->mapofs];
  MSize n, m, nlim, nent = snap->nent;
  uint8_t udf[SNAP_USEDEF_SLOTS];
  BCReg maxslot = J->maxslot;
  BCReg baseslot = J->baseslot;
  BCReg minslot = snap_usedef(J, udf, snap_pc(&map[nent]), maxslot);
  if (minslot < maxslot) snap_useuv(J->pt, udf);
  maxslot += baseslot;
  minslot += baseslot;
  snap->nslots = (uint8_t)maxslot;
  for (n = m = 0; n < nent; n++) {  /* Remove unused slots from snapshot. */
    BCReg s = snap_slot(map[n]);
    if (s < minslot || (s < maxslot && udf[s-baseslot] == 0))
      map[m++] = map[n];  /* Only copy used slots. */
  }
  snap->nent = (uint8_t)m;
  nlim = J->cur.nsnapmap - snap->mapofs - 1;
  while (n <= nlim) map[m++] = map[n++];  /* Move PC + frame links down. */
  J->cur.nsnapmap = (uint32_t)(snap->mapofs + m);  /* Free up space in map. */
}

/* -- Snapshot access ----------------------------------------------------- */

/* Initialize a Bloom Filter with all renamed refs.
** There are very few renames (often none), so the filter has
** very few bits set. This makes it suitable for negative filtering.
*/
static BloomFilter snap_renamefilter(GCtrace *T, SnapNo lim)
{
  BloomFilter rfilt = 0;
  IRIns *ir;
  for (ir = &T->ir[T->nins-1]; ir->o == IR_RENAME; ir--)
    if (ir->op2 <= lim)
      bloomset(rfilt, ir->op1);
  return rfilt;
}

/* Process matching renames to find the original RegSP. */
static RegSP snap_renameref(GCtrace *T, SnapNo lim, IRRef ref, RegSP rs)
{
  IRIns *ir;
  for (ir = &T->ir[T->nins-1]; ir->o == IR_RENAME; ir--)
    if (ir->op1 == ref && ir->op2 <= lim)
      rs = ir->prev;
  return rs;
}

/* Copy RegSP from parent snapshot to the parent links of the IR. */
IRIns *lj_snap_regspmap(jit_State *J, GCtrace *T, SnapNo snapno, IRIns *ir)
{
  SnapShot *snap = &T->snap[snapno];
  SnapEntry *map = &T->snapmap[snap->mapofs];
  BloomFilter rfilt = snap_renamefilter(T, snapno);
  MSize n = 0;
  IRRef ref = 0;
  UNUSED(J);
  for ( ; ; ir++) {
    uint32_t rs;
    if (ir->o == IR_SLOAD) {
      if (!(ir->op2 & IRSLOAD_PARENT)) break;
      for ( ; ; n++) {
	lj_assertJ(n < snap->nent, "slot %d not found in snapshot", ir->op1);
	if (snap_slot(map[n]) == ir->op1) {
	  ref = snap_ref(map[n++]);
	  break;
	}
      }
    } else if (LJ_SOFTFP32 && ir->o == IR_HIOP) {
      ref++;
    } else if (ir->o == IR_PVAL) {
      ref = ir->op1 + REF_BIAS;
    } else {
      break;
    }
    rs = T->ir[ref].prev;
    if (bloomtest(rfilt, ref))
      rs = snap_renameref(T, snapno, ref, rs);
    ir->prev = (uint16_t)rs;
    lj_assertJ(regsp_used(rs), "unused IR %04d in snapshot", ref - REF_BIAS);
  }
  return ir;
}

/* -- Snapshot replay ----------------------------------------------------- */

/* Replay constant from parent trace. */
static TRef snap_replay_const(jit_State *J, IRIns *ir)
{
  /* Only have to deal with constants that can occur in stack slots. */
  switch ((IROp)ir->o) {
  case IR_KPRI: return TREF_PRI(irt_type(ir->t));
  case IR_KINT: return lj_ir_kint(J, ir->i);
  case IR_KGC: return lj_ir_kgc(J, ir_kgc(ir), irt_t(ir->t));
  case IR_KNUM: case IR_KINT64:
    return lj_ir_k64(J, (IROp)ir->o, ir_k64(ir)->u64);
  case IR_KPTR: return lj_ir_kptr(J, ir_kptr(ir));  /* Continuation. */
  default: lj_assertJ(0, "bad IR constant op %d", ir->o); return TREF_NIL;
  }
}

/* De-duplicate parent reference. */
static TRef snap_dedup(jit_State *J, SnapEntry *map, MSize nmax, IRRef ref)
{
  MSize j;
  for (j = 0; j < nmax; j++)
    if (snap_ref(map[j]) == ref)
      return J->slot[snap_slot(map[j])] & ~(SNAP_KEYINDEX|SNAP_CONT|SNAP_FRAME);
  return 0;
}

/* Emit parent reference with de-duplication. */
static TRef snap_pref(jit_State *J, GCtrace *T, SnapEntry *map, MSize nmax,
		      BloomFilter seen, IRRef ref)
{
  IRIns *ir = &T->ir[ref];
  TRef tr;
  if (irref_isk(ref))
    tr = snap_replay_const(J, ir);
  else if (!regsp_used(ir->prev))
    tr = 0;
  else if (!bloomtest(seen, ref) || (tr = snap_dedup(J, map, nmax, ref)) == 0)
    tr = emitir(IRT(IR_PVAL, irt_type(ir->t)), ref - REF_BIAS, 0);
  return tr;
}

/* Check whether a sunk store corresponds to an allocation. Slow path. */
static int snap_sunk_store2(GCtrace *T, IRIns *ira, IRIns *irs)
{
  if (irs->o == IR_ASTORE || irs->o == IR_HSTORE ||
      irs->o == IR_FSTORE || irs->o == IR_XSTORE) {
    IRIns *irk = &T->ir[irs->op1];
    if (irk->o == IR_AREF || irk->o == IR_HREFK)
      irk = &T->ir[irk->op1];
    return (&T->ir[irk->op1] == ira);
  }
  return 0;
}

/* Check whether a sunk store corresponds to an allocation. Fast path. */
static LJ_AINLINE int snap_sunk_store(GCtrace *T, IRIns *ira, IRIns *irs)
{
  if (irs->s != 255)
    return (ira + irs->s == irs);  /* Fast check. */
  return snap_sunk_store2(T, ira, irs);
}

/* Replay snapshot state to setup side trace. */
void lj_snap_replay(jit_State *J, GCtrace *T)
{
  SnapShot *snap = &T->snap[J->exitno];
  SnapEntry *map = &T->snapmap[snap->mapofs];
  MSize n, nent = snap->nent;
  BloomFilter seen = 0;
  int pass23 = 0;
  J->framedepth = 0;
  /* Emit IR for slots inherited from parent snapshot. */
  for (n = 0; n < nent; n++) {
    SnapEntry sn = map[n];
    BCReg s = snap_slot(sn);
    IRRef ref = snap_ref(sn);
    IRIns *ir = &T->ir[ref];
    TRef tr;
    /* The bloom filter avoids O(nent^2) overhead for de-duping slots. */
    if (bloomtest(seen, ref) && (tr = snap_dedup(J, map, n, ref)) != 0)
      goto setslot;
    bloomset(seen, ref);
    if (irref_isk(ref)) {
      /* See special treatment of LJ_FR2 slot 1 in snapshot_slots() above. */
      if (LJ_FR2 && (sn == SNAP(1, SNAP_FRAME | SNAP_NORESTORE, REF_NIL)))
	tr = 0;
      else
	tr = snap_replay_const(J, ir);
    } else if (!regsp_used(ir->prev)) {
      pass23 = 1;
      lj_assertJ(s != 0, "unused slot 0 in snapshot");
      tr = s;
    } else {
      IRType t = irt_type(ir->t);
      uint32_t mode = IRSLOAD_INHERIT|IRSLOAD_PARENT;
      if (LJ_SOFTFP32 && (sn & SNAP_SOFTFPNUM)) t = IRT_NUM;
      if (ir->o == IR_SLOAD) mode |= (ir->op2 & IRSLOAD_READONLY);
      if ((sn & SNAP_KEYINDEX)) mode |= IRSLOAD_KEYINDEX;
      tr = emitir_raw(IRT(IR_SLOAD, t), s, mode);
    }
  setslot:
    /* Same as TREF_* flags. */
    J->slot[s] = tr | (sn&(SNAP_KEYINDEX|SNAP_CONT|SNAP_FRAME));
    J->framedepth += ((sn & (SNAP_CONT|SNAP_FRAME)) && (s != LJ_FR2));
    if ((sn & SNAP_FRAME))
      J->baseslot = s+1;
  }
  if (pass23) {
    IRIns *irlast = &T->ir[snap->ref];
    pass23 = 0;
    /* Emit dependent PVALs. */
    for (n = 0; n < nent; n++) {
      SnapEntry sn = map[n];
      IRRef refp = snap_ref(sn);
      IRIns *ir = &T->ir[refp];
      if (regsp_reg(ir->r) == RID_SUNK) {
	uint8_t m;
	if (J->slot[snap_slot(sn)] != snap_slot(sn)) continue;
	pass23 = 1;
	lj_assertJ(ir->o == IR_TNEW || ir->o == IR_TDUP ||
		   ir->o == IR_CNEW || ir->o == IR_CNEWI,
		   "sunk parent IR %04d has bad op %d", refp - REF_BIAS, ir->o);
	m = lj_ir_mode[ir->o];
	if (irm_op1(m) == IRMref) snap_pref(J, T, map, nent, seen, ir->op1);
	if (irm_op2(m) == IRMref) snap_pref(J, T, map, nent, seen, ir->op2);
	if (LJ_HASFFI && ir->o == IR_CNEWI) {
	  if (LJ_32 && refp+1 < T->nins && (ir+1)->o == IR_HIOP)
	    snap_pref(J, T, map, nent, seen, (ir+1)->op2);
	} else {
	  IRIns *irs;
	  for (irs = ir+1; irs < irlast; irs++)
	    if (irs->r == RID_SINK && snap_sunk_store(T, ir, irs)) {
	      if (snap_pref(J, T, map, nent, seen, irs->op2) == 0)
		snap_pref(J, T, map, nent, seen, T->ir[irs->op2].op1);
	      else if ((LJ_SOFTFP32 || (LJ_32 && LJ_HASFFI)) &&
		       irs+1 < irlast && (irs+1)->o == IR_HIOP)
		snap_pref(J, T, map, nent, seen, (irs+1)->op2);
	    }
	}
      } else if (!irref_isk(refp) && !regsp_used(ir->prev)) {
	lj_assertJ(ir->o == IR_CONV && ir->op2 == IRCONV_NUM_INT,
		   "sunk parent IR %04d has bad op %d", refp - REF_BIAS, ir->o);
	J->slot[snap_slot(sn)] = snap_pref(J, T, map, nent, seen, ir->op1);
      }
    }
    /* Replay sunk instructions. */
    for (n = 0; pass23 && n < nent; n++) {
      SnapEntry sn = map[n];
      IRRef refp = snap_ref(sn);
      IRIns *ir = &T->ir[refp];
      if (regsp_reg(ir->r) == RID_SUNK) {
	TRef op1, op2;
	uint8_t m;
	if (J->slot[snap_slot(sn)] != snap_slot(sn)) {  /* De-dup allocs. */
	  J->slot[snap_slot(sn)] = J->slot[J->slot[snap_slot(sn)]];
	  continue;
	}
	op1 = ir->op1;
	m = lj_ir_mode[ir->o];
	if (irm_op1(m) == IRMref) op1 = snap_pref(J, T, map, nent, seen, op1);
	op2 = ir->op2;
	if (irm_op2(m) == IRMref) op2 = snap_pref(J, T, map, nent, seen, op2);
	if (LJ_HASFFI && ir->o == IR_CNEWI) {
	  if (LJ_32 && refp+1 < T->nins && (ir+1)->o == IR_HIOP) {
	    lj_needsplit(J);  /* Emit joining HIOP. */
	    op2 = emitir_raw(IRT(IR_HIOP, IRT_I64), op2,
			     snap_pref(J, T, map, nent, seen, (ir+1)->op2));
	  }
	  J->slot[snap_slot(sn)] = emitir(ir->ot & ~(IRT_MARK|IRT_ISPHI), op1, op2);
	} else {
	  IRIns *irs;
	  TRef tr = emitir(ir->ot, op1, op2);
	  J->slot[snap_slot(sn)] = tr;
	  for (irs = ir+1; irs < irlast; irs++)
	    if (irs->r == RID_SINK && snap_sunk_store(T, ir, irs)) {
	      IRIns *irr = &T->ir[irs->op1];
	      TRef val, key = irr->op2, tmp = tr;
	      if (irr->o != IR_FREF) {
		IRIns *irk = &T->ir[key];
		if (irr->o == IR_HREFK)
		  key = lj_ir_kslot(J, snap_replay_const(J, &T->ir[irk->op1]),
				    irk->op2);
		else
		  key = snap_replay_const(J, irk);
		if (irr->o == IR_HREFK || irr->o == IR_AREF) {
		  IRIns *irf = &T->ir[irr->op1];
		  tmp = emitir(irf->ot, tmp, irf->op2);
		} else if (irr->o == IR_NEWREF) {
		  IRRef allocref = tref_ref(tr);
		  IRRef keyref = tref_ref(key);
		  IRRef newref_ref = J->chain[IR_NEWREF];
		  IRIns *newref = &J->cur.ir[newref_ref];
		  lj_assertJ(irref_isk(keyref),
			     "sunk store for parent IR %04d with bad key %04d",
			     refp - REF_BIAS, keyref - REF_BIAS);
		  if (newref_ref > allocref && newref->op2 == keyref) {
		    lj_assertJ(newref->op1 == allocref,
			       "sunk store for parent IR %04d with bad tab %04d",
			       refp - REF_BIAS, allocref - REF_BIAS);
		    tmp = newref_ref;
		    goto skip_newref;
		  }
		}
	      }
	      tmp = emitir(irr->ot, tmp, key);
	    skip_newref:
	      val = snap_pref(J, T, map, nent, seen, irs->op2);
	      if (val == 0) {
		IRIns *irc = &T->ir[irs->op2];
		lj_assertJ(irc->o == IR_CONV && irc->op2 == IRCONV_NUM_INT,
			   "sunk store for parent IR %04d with bad op %d",
			   refp - REF_BIAS, irc->o);
		val = snap_pref(J, T, map, nent, seen, irc->op1);
		val = emitir(IRTN(IR_CONV), val, IRCONV_NUM_INT);
	      } else if ((LJ_SOFTFP32 || (LJ_32 && LJ_HASFFI)) &&
			 irs+1 < irlast && (irs+1)->o == IR_HIOP) {
		IRType t = IRT_I64;
		if (LJ_SOFTFP32 && irt_type((irs+1)->t) == IRT_SOFTFP)
		  t = IRT_NUM;
		lj_needsplit(J);
		if (irref_isk(irs->op2) && irref_isk((irs+1)->op2)) {
		  uint64_t k = (uint32_t)T->ir[irs->op2].i +
			       ((uint64_t)T->ir[(irs+1)->op2].i << 32);
		  val = lj_ir_k64(J, t == IRT_I64 ? IR_KINT64 : IR_KNUM, k);
		} else {
		  val = emitir_raw(IRT(IR_HIOP, t), val,
			  snap_pref(J, T, map, nent, seen, (irs+1)->op2));
		}
		tmp = emitir(IRT(irs->o, t), tmp, val);
		continue;
	      }
	      tmp = emitir(irs->ot, tmp, val);
	    } else if (LJ_HASFFI && irs->o == IR_XBAR && ir->o == IR_CNEW) {
	      emitir(IRT(IR_XBAR, IRT_NIL), 0, 0);
	    }
	}
      }
    }
  }
  J->base = J->slot + J->baseslot;
  J->maxslot = snap->nslots - J->baseslot;
  lj_snap_add(J);
  if (pass23)  /* Need explicit GC step _after_ initial snapshot. */
    emitir_raw(IRTG(IR_GCSTEP, IRT_NIL), 0, 0);
}

/* -- Snapshot restore ---------------------------------------------------- */

static void snap_unsink(jit_State *J, GCtrace *T, ExitState *ex,
			SnapNo snapno, BloomFilter rfilt,
			IRIns *ir, TValue *o);

/* Restore a value from the trace exit state. */
static void snap_restoreval(jit_State *J, GCtrace *T, ExitState *ex,
			    SnapNo snapno, BloomFilter rfilt,
			    IRRef ref, TValue *o)
{
  IRIns *ir = &T->ir[ref];
  IRType1 t = ir->t;
  RegSP rs = ir->prev;
  if (irref_isk(ref)) {  /* Restore constant slot. */
    if (ir->o == IR_KPTR) {
      o->u64 = (uint64_t)(uintptr_t)ir_kptr(ir);
    } else {
      lj_assertJ(!(ir->o == IR_KKPTR || ir->o == IR_KNULL),
		 "restore of const from IR %04d with bad op %d",
		 ref - REF_BIAS, ir->o);
      lj_ir_kvalue(J->L, o, ir);
    }
    return;
  }
  if (LJ_UNLIKELY(bloomtest(rfilt, ref)))
    rs = snap_renameref(T, snapno, ref, rs);
  if (ra_hasspill(regsp_spill(rs))) {  /* Restore from spill slot. */
    int32_t *sps = &ex->spill[regsp_spill(rs)];
    if (irt_isinteger(t)) {
      setintV(o, *sps);
#if !LJ_SOFTFP32
    } else if (irt_isnum(t)) {
      o->u64 = *(uint64_t *)sps;
#endif
#if LJ_64 && !LJ_GC64
    } else if (irt_islightud(t)) {
      /* 64 bit lightuserdata which may escape already has the tag bits. */
      o->u64 = *(uint64_t *)sps;
#endif
    } else {
      lj_assertJ(!irt_ispri(t), "PRI ref with spill slot");
      setgcV(J->L, o, (GCobj *)(uintptr_t)*(GCSize *)sps, irt_toitype(t));
    }
  } else {  /* Restore from register. */
    Reg r = regsp_reg(rs);
    if (ra_noreg(r)) {
      lj_assertJ(ir->o == IR_CONV && ir->op2 == IRCONV_NUM_INT,
		 "restore from IR %04d has no reg", ref - REF_BIAS);
      snap_restoreval(J, T, ex, snapno, rfilt, ir->op1, o);
      if (LJ_DUALNUM) setnumV(o, (lua_Number)intV(o));
      return;
    } else if (irt_isinteger(t)) {
      setintV(o, (int32_t)ex->gpr[r-RID_MIN_GPR]);
#if !LJ_SOFTFP
    } else if (irt_isnum(t)) {
      setnumV(o, ex->fpr[r-RID_MIN_FPR]);
#elif LJ_64  /* && LJ_SOFTFP */
    } else if (irt_isnum(t)) {
      o->u64 = ex->gpr[r-RID_MIN_GPR];
#endif
#if LJ_64 && !LJ_GC64
    } else if (irt_is64(t)) {
      /* 64 bit values that already have the tag bits. */
      o->u64 = ex->gpr[r-RID_MIN_GPR];
#endif
    } else if (irt_ispri(t)) {
      setpriV(o, irt_toitype(t));
    } else {
      setgcV(J->L, o, (GCobj *)ex->gpr[r-RID_MIN_GPR], irt_toitype(t));
    }
  }
}

#if LJ_HASFFI
/* Restore raw data from the trace exit state. */
static void snap_restoredata(jit_State *J, GCtrace *T, ExitState *ex,
			     SnapNo snapno, BloomFilter rfilt,
			     IRRef ref, void *dst, CTSize sz)
{
  IRIns *ir = &T->ir[ref];
  RegSP rs = ir->prev;
  int32_t *src;
  uint64_t tmp;
  UNUSED(J);
  if (irref_isk(ref)) {
    if (ir_isk64(ir)) {
      src = (int32_t *)&ir[1];
    } else if (sz == 8) {
      tmp = (uint64_t)(uint32_t)ir->i;
      src = (int32_t *)&tmp;
    } else {
      src = &ir->i;
    }
  } else {
    if (LJ_UNLIKELY(bloomtest(rfilt, ref)))
      rs = snap_renameref(T, snapno, ref, rs);
    if (ra_hasspill(regsp_spill(rs))) {
      src = &ex->spill[regsp_spill(rs)];
      if (sz == 8 && !irt_is64(ir->t)) {
	tmp = (uint64_t)(uint32_t)*src;
	src = (int32_t *)&tmp;
      }
    } else {
      Reg r = regsp_reg(rs);
      if (ra_noreg(r)) {
	/* Note: this assumes CNEWI is never used for SOFTFP split numbers. */
	lj_assertJ(sz == 8 && ir->o == IR_CONV && ir->op2 == IRCONV_NUM_INT,
		   "restore from IR %04d has no reg", ref - REF_BIAS);
	snap_restoredata(J, T, ex, snapno, rfilt, ir->op1, dst, 4);
	*(lua_Number *)dst = (lua_Number)*(int32_t *)dst;
	return;
      }
      src = (int32_t *)&ex->gpr[r-RID_MIN_GPR];
#if !LJ_SOFTFP
      if (r >= RID_MAX_GPR) {
	src = (int32_t *)&ex->fpr[r-RID_MIN_FPR];
#if LJ_TARGET_PPC
	if (sz == 4) {  /* PPC FPRs are always doubles. */
	  *(float *)dst = (float)*(double *)src;
	  return;
	}
#else
	if (LJ_BE && sz == 4) src++;
#endif
      } else
#endif
      if (LJ_64 && LJ_BE && sz == 4) src++;
    }
  }
  lj_assertJ(sz == 1 || sz == 2 || sz == 4 || sz == 8,
	     "restore from IR %04d with bad size %d", ref - REF_BIAS, sz);
  if (sz == 4) *(int32_t *)dst = *src;
  else if (sz == 8) *(int64_t *)dst = *(int64_t *)src;
  else if (sz == 1) *(int8_t *)dst = (int8_t)*src;
  else *(int16_t *)dst = (int16_t)*src;
}
#endif

/* Unsink allocation from the trace exit state. Unsink sunk stores. */
static void snap_unsink(jit_State *J, GCtrace *T, ExitState *ex,
			SnapNo snapno, BloomFilter rfilt,
			IRIns *ir, TValue *o)
{
  lj_assertJ(ir->o == IR_TNEW || ir->o == IR_TDUP ||
	     ir->o == IR_CNEW || ir->o == IR_CNEWI,
	     "sunk allocation with bad op %d", ir->o);
#if LJ_HASFFI
  if (ir->o == IR_CNEW || ir->o == IR_CNEWI) {
    CTState *cts = ctype_cts(J->L);
    CTypeID id = (CTypeID)T->ir[ir->op1].i;
    CTSize sz;
    CTInfo info = lj_ctype_info(cts, id, &sz);
    GCcdata *cd = lj_cdata_newx(cts, id, sz, info);
    setcdataV(J->L, o, cd);
    if (ir->o == IR_CNEWI) {
      uint8_t *p = (uint8_t *)cdataptr(cd);
      lj_assertJ(sz == 4 || sz == 8, "sunk cdata with bad size %d", sz);
      if (LJ_32 && sz == 8 && ir+1 < T->ir + T->nins && (ir+1)->o == IR_HIOP) {
	snap_restoredata(J, T, ex, snapno, rfilt, (ir+1)->op2,
			 LJ_LE ? p+4 : p, 4);
	if (LJ_BE) p += 4;
	sz = 4;
      }
      snap_restoredata(J, T, ex, snapno, rfilt, ir->op2, p, sz);
    } else {
      IRIns *irs, *irlast = &T->ir[T->snap[snapno].ref];
      for (irs = ir+1; irs < irlast; irs++)
	if (irs->r == RID_SINK && snap_sunk_store(T, ir, irs)) {
	  IRIns *iro = &T->ir[T->ir[irs->op1].op2];
	  uint8_t *p = (uint8_t *)cd;
	  CTSize szs;
	  lj_assertJ(irs->o == IR_XSTORE, "sunk store with bad op %d", irs->o);
	  lj_assertJ(T->ir[irs->op1].o == IR_ADD,
		     "sunk store with bad add op %d", T->ir[irs->op1].o);
	  lj_assertJ(iro->o == IR_KINT || iro->o == IR_KINT64,
		     "sunk store with bad const offset op %d", iro->o);
	  if (irt_is64(irs->t)) szs = 8;
	  else if (irt_isi8(irs->t) || irt_isu8(irs->t)) szs = 1;
	  else if (irt_isi16(irs->t) || irt_isu16(irs->t)) szs = 2;
	  else szs = 4;
	  if (LJ_64 && iro->o == IR_KINT64)
	    p += (int64_t)ir_k64(iro)->u64;
	  else
	    p += iro->i;
	  lj_assertJ(p >= (uint8_t *)cdataptr(cd) &&
		     p + szs <= (uint8_t *)cdataptr(cd) + sz,
		     "sunk store with offset out of range");
	  if (LJ_32 && irs+1 < T->ir + T->nins && (irs+1)->o == IR_HIOP) {
	    lj_assertJ(szs == 4, "sunk store with bad size %d", szs);
	    snap_restoredata(J, T, ex, snapno, rfilt, (irs+1)->op2,
			     LJ_LE ? p+4 : p, 4);
	    if (LJ_BE) p += 4;
	  }
	  snap_restoredata(J, T, ex, snapno, rfilt, irs->op2, p, szs);
	}
    }
  } else
#endif
  {
    IRIns *irs, *irlast;
    GCtab *t = ir->o == IR_TNEW ? lj_tab_new(J->L, ir->op1, ir->op2) :
				  lj_tab_dup(J->L, ir_ktab(&T->ir[ir->op1]));
    settabV(J->L, o, t);
    irlast = &T->ir[T->snap[snapno].ref];
    for (irs = ir+1; irs < irlast; irs++)
      if (irs->r == RID_SINK && snap_sunk_store(T, ir, irs)) {
	IRIns *irk = &T->ir[irs->op1];
	TValue tmp, *val;
	lj_assertJ(irs->o == IR_ASTORE || irs->o == IR_HSTORE ||
		   irs->o == IR_FSTORE,
		   "sunk store with bad op %d", irs->o);
	if (irk->o == IR_FREF) {
	  switch (irk->op2) {
	  case IRFL_TAB_META:
	    snap_restoreval(J, T, ex, snapno, rfilt, irs->op2, &tmp);
	    /* NOBARRIER: The table is new (marked white). */
	    setgcref(t->metatable, obj2gco(tabV(&tmp)));
	    break;
	  case IRFL_TAB_NOMM:
	    /* Negative metamethod cache invalidated by lj_tab_set() below. */
	    break;
	  default:
	    lj_assertJ(0, "sunk store with bad field %d", irk->op2);
	    break;
	  }
	} else {
	  irk = &T->ir[irk->op2];
	  if (irk->o == IR_KSLOT) irk = &T->ir[irk->op1];
	  lj_ir_kvalue(J->L, &tmp, irk);
	  val = lj_tab_set(J->L, t, &tmp);
	  /* NOBARRIER: The table is new (marked white). */
	  snap_restoreval(J, T, ex, snapno, rfilt, irs->op2, val);
	  if (LJ_SOFTFP32 && irs+1 < T->ir + T->nins && (irs+1)->o == IR_HIOP) {
	    snap_restoreval(J, T, ex, snapno, rfilt, (irs+1)->op2, &tmp);
	    val->u32.hi = tmp.u32.lo;
	  }
	}
      }
  }
}

/* Restore interpreter state from exit state with the help of a snapshot. */
const BCIns *lj_snap_restore(jit_State *J, void *exptr)
{
  ExitState *ex = (ExitState *)exptr;
  SnapNo snapno = J->exitno;  /* For now, snapno == exitno. */
  GCtrace *T = traceref(J, J->parent);
  SnapShot *snap = &T->snap[snapno];
  MSize n, nent = snap->nent;
  SnapEntry *map = &T->snapmap[snap->mapofs];
#if !LJ_FR2 || defined(LUA_USE_ASSERT)
  SnapEntry *flinks = &T->snapmap[snap_nextofs(T, snap)-1-LJ_FR2];
#endif
#if !LJ_FR2
  ptrdiff_t ftsz0;
#endif
  TValue *frame;
  BloomFilter rfilt = snap_renamefilter(T, snapno);
  const BCIns *pc = snap_pc(&map[nent]);
  lua_State *L = J->L;

  /* Set interpreter PC to the next PC to get correct error messages. */
  setcframe_pc(cframe_raw(L->cframe), pc+1);

  /* Make sure the stack is big enough for the slots from the snapshot. */
  if (LJ_UNLIKELY(L->base + snap->topslot >= tvref(L->maxstack))) {
    L->top = curr_topL(L);
    lj_state_growstack(L, snap->topslot - curr_proto(L)->framesize);
  }

  /* Fill stack slots with data from the registers and spill slots. */
  frame = L->base-1-LJ_FR2;
#if !LJ_FR2
  ftsz0 = frame_ftsz(frame);  /* Preserve link to previous frame in slot #0. */
#endif
  for (n = 0; n < nent; n++) {
    SnapEntry sn = map[n];
    if (!(sn & SNAP_NORESTORE)) {
      TValue *o = &frame[snap_slot(sn)];
      IRRef ref = snap_ref(sn);
      IRIns *ir = &T->ir[ref];
      if (ir->r == RID_SUNK) {
	MSize j;
	for (j = 0; j < n; j++)
	  if (snap_ref(map[j]) == ref) {  /* De-duplicate sunk allocations. */
	    copyTV(L, o, &frame[snap_slot(map[j])]);
	    goto dupslot;
	  }
	snap_unsink(J, T, ex, snapno, rfilt, ir, o);
      dupslot:
	continue;
      }
      snap_restoreval(J, T, ex, snapno, rfilt, ref, o);
      if (LJ_SOFTFP32 && (sn & SNAP_SOFTFPNUM) && tvisint(o)) {
	TValue tmp;
	snap_restoreval(J, T, ex, snapno, rfilt, ref+1, &tmp);
	o->u32.hi = tmp.u32.lo;
#if !LJ_FR2
      } else if ((sn & (SNAP_CONT|SNAP_FRAME))) {
	/* Overwrite tag with frame link. */
	setframe_ftsz(o, snap_slot(sn) != 0 ? (int32_t)*flinks-- : ftsz0);
	L->base = o+1;
#endif
      } else if ((sn & SNAP_KEYINDEX)) {
	/* A IRT_INT key index slot is restored as a number. Undo this. */
	o->u32.lo = (uint32_t)(LJ_DUALNUM ? intV(o) : lj_num2int(numV(o)));
	o->u32.hi = LJ_KEYINDEX;
      }
    }
  }
#if LJ_FR2
  L->base += (map[nent+LJ_BE] & 0xff);
#endif
  lj_assertJ(map + nent == flinks, "inconsistent frames in snapshot");

  /* Compute current stack top. */
  switch (bc_op(*pc)) {
  default:
    if (bc_op(*pc) < BC_FUNCF) {
      L->top = curr_topL(L);
      break;
    }
    /* fallthrough */
  case BC_CALLM: case BC_CALLMT: case BC_RETM: case BC_TSETM:
    L->top = frame + snap->nslots;
    break;
  }
  return pc;
}

#undef emitir_raw
#undef emitir

#endif
