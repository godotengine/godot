/*
** Machine code management.
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
*/

#define lj_mcode_c
#define LUA_CORE

#include "lj_obj.h"
#if LJ_HASJIT
#include "lj_gc.h"
#include "lj_err.h"
#include "lj_jit.h"
#include "lj_mcode.h"
#include "lj_trace.h"
#include "lj_dispatch.h"
#include "lj_prng.h"
#endif
#if LJ_HASJIT || LJ_HASFFI
#include "lj_vm.h"
#endif

/* -- OS-specific functions ----------------------------------------------- */

#if LJ_HASJIT || LJ_HASFFI

/* Define this if you want to run LuaJIT with Valgrind. */
#ifdef LUAJIT_USE_VALGRIND
#include <valgrind/valgrind.h>
#endif

#if LJ_TARGET_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#if LJ_TARGET_IOS
void sys_icache_invalidate(void *start, size_t len);
#endif

#if LJ_TARGET_RISCV64 && LJ_TARGET_LINUX
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/cachectl.h>
#endif

/* Synchronize data/instruction cache. */
void lj_mcode_sync(void *start, void *end)
{
#ifdef LUAJIT_USE_VALGRIND
  VALGRIND_DISCARD_TRANSLATIONS(start, (char *)end-(char *)start);
#endif
#if LJ_TARGET_X86ORX64
  UNUSED(start); UNUSED(end);
#elif LJ_TARGET_WINDOWS
  FlushInstructionCache(GetCurrentProcess(), start, (char *)end-(char *)start);
#elif LJ_TARGET_IOS
  sys_icache_invalidate(start, (char *)end-(char *)start);
#elif LJ_TARGET_PPC
  lj_vm_cachesync(start, end);
#elif LJ_TARGET_RISCV64 && LJ_TARGET_LINUX
#if (defined(__GNUC__) || defined(__clang__))
  __asm__ volatile("fence rw, rw");
#else
  lj_vm_fence_rw_rw();
#endif
#ifdef __GLIBC__
  __riscv_flush_icache(start, end, 0);
#else
  syscall(__NR_riscv_flush_icache, start, end, 0UL);
#endif
#elif defined(__GNUC__) || defined(__clang__)
  __clear_cache(start, end);
#else
#error "Missing builtin to flush instruction cache"
#endif
}

#endif

#if LJ_HASJIT

#if LJ_TARGET_WINDOWS

#define MCPROT_RW	PAGE_READWRITE
#define MCPROT_RX	PAGE_EXECUTE_READ
#define MCPROT_RWX	PAGE_EXECUTE_READWRITE

static void *mcode_alloc_at(jit_State *J, uintptr_t hint, size_t sz, DWORD prot)
{
  void *p = LJ_WIN_VALLOC((void *)hint, sz,
			  MEM_RESERVE|MEM_COMMIT|MEM_TOP_DOWN, prot);
  if (!p && !hint)
    lj_trace_err(J, LJ_TRERR_MCODEAL);
  return p;
}

static void mcode_free(jit_State *J, void *p, size_t sz)
{
  UNUSED(J); UNUSED(sz);
  VirtualFree(p, 0, MEM_RELEASE);
}

static int mcode_setprot(void *p, size_t sz, DWORD prot)
{
  DWORD oprot;
  return !LJ_WIN_VPROTECT(p, sz, prot, &oprot);
}

#elif LJ_TARGET_POSIX

#include <sys/mman.h>

#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS	MAP_ANON
#endif

#define MCPROT_RW	(PROT_READ|PROT_WRITE)
#define MCPROT_RX	(PROT_READ|PROT_EXEC)
#define MCPROT_RWX	(PROT_READ|PROT_WRITE|PROT_EXEC)
#ifdef PROT_MPROTECT
#define MCPROT_CREATE	(PROT_MPROTECT(MCPROT_RWX))
#else
#define MCPROT_CREATE	0
#endif

static void *mcode_alloc_at(jit_State *J, uintptr_t hint, size_t sz, int prot)
{
  void *p = mmap((void *)hint, sz, prot|MCPROT_CREATE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) {
    if (!hint) lj_trace_err(J, LJ_TRERR_MCODEAL);
    p = NULL;
  }
  return p;
}

static void mcode_free(jit_State *J, void *p, size_t sz)
{
  UNUSED(J);
  munmap(p, sz);
}

static int mcode_setprot(void *p, size_t sz, int prot)
{
  return mprotect(p, sz, prot);
}

#else

#error "Missing OS support for explicit placement of executable memory"

#endif

/* -- MCode area protection ----------------------------------------------- */

#if LUAJIT_SECURITY_MCODE == 0

/* Define this ONLY if page protection twiddling becomes a bottleneck.
**
** It's generally considered to be a potential security risk to have
** pages with simultaneous write *and* execute access in a process.
**
** Do not even think about using this mode for server processes or
** apps handling untrusted external data.
**
** The security risk is not in LuaJIT itself -- but if an adversary finds
** any *other* flaw in your C application logic, then any RWX memory pages
** simplify writing an exploit considerably.
*/
#define MCPROT_GEN	MCPROT_RWX
#define MCPROT_RUN	MCPROT_RWX

static void mcode_protect(jit_State *J, int prot)
{
  UNUSED(J); UNUSED(prot); UNUSED(mcode_setprot);
}

#else

/* This is the default behaviour and much safer:
**
** Most of the time the memory pages holding machine code are executable,
** but NONE of them is writable.
**
** The current memory area is marked read-write (but NOT executable) only
** during the short time window while the assembler generates machine code.
*/
#define MCPROT_GEN	MCPROT_RW
#define MCPROT_RUN	MCPROT_RX

/* Protection twiddling failed. Probably due to kernel security. */
static LJ_NORET LJ_NOINLINE void mcode_protfail(jit_State *J)
{
  lua_CFunction panic = J2G(J)->panic;
  if (panic) {
    lua_State *L = J->L;
    setstrV(L, L->top++, lj_err_str(L, LJ_ERR_JITPROT));
    panic(L);
  }
  exit(EXIT_FAILURE);
}

/* Change protection of MCode area. */
static void mcode_protect(jit_State *J, int prot)
{
  if (J->mcprot != prot) {
    if (LJ_UNLIKELY(mcode_setprot(J->mcarea, J->szmcarea, prot)))
      mcode_protfail(J);
    J->mcprot = prot;
  }
}

#endif

/* -- MCode area allocation ----------------------------------------------- */

#if LJ_64
#define mcode_validptr(p)	(p)
#else
#define mcode_validptr(p)	((p) && (uintptr_t)(p) < 0xffff0000)
#endif

#ifdef LJ_TARGET_JUMPRANGE

/* Get memory within relative jump distance of our code in 64 bit mode. */
static void *mcode_alloc(jit_State *J, size_t sz)
{
  /* Target an address in the static assembler code (64K aligned).
  ** Try addresses within a distance of target-range/2+1MB..target+range/2-1MB.
  ** Use half the jump range so every address in the range can reach any other.
  */
#if LJ_TARGET_MIPS
  /* Use the middle of the 256MB-aligned region. */
  uintptr_t target = ((uintptr_t)(void *)lj_vm_exit_handler &
		      ~(uintptr_t)0x0fffffffu) + 0x08000000u;
#else
  uintptr_t target = (uintptr_t)(void *)lj_vm_exit_handler & ~(uintptr_t)0xffff;
#endif
  const uintptr_t range = (1u << (LJ_TARGET_JUMPRANGE-1)) - (1u << 21);
  /* First try a contiguous area below the last one. */
  uintptr_t hint = J->mcarea ? (uintptr_t)J->mcarea - sz : 0;
  int i;
  /* Limit probing iterations, depending on the available pool size. */
  for (i = 0; i < LJ_TARGET_JUMPRANGE; i++) {
    if (mcode_validptr(hint)) {
      void *p = mcode_alloc_at(J, hint, sz, MCPROT_GEN);

      if (mcode_validptr(p) &&
	  ((uintptr_t)p + sz - target < range || target - (uintptr_t)p < range))
	return p;
      if (p) mcode_free(J, p, sz);  /* Free badly placed area. */
    }
    /* Next try probing 64K-aligned pseudo-random addresses. */
    do {
      hint = lj_prng_u64(&J2G(J)->prng) & ((1u<<LJ_TARGET_JUMPRANGE)-0x10000);
    } while (!(hint + sz < range+range));
    hint = target + hint - range;
  }
  lj_trace_err(J, LJ_TRERR_MCODEAL);  /* Give up. OS probably ignores hints? */
  return NULL;
}

#else

/* All memory addresses are reachable by relative jumps. */
static void *mcode_alloc(jit_State *J, size_t sz)
{
#if defined(__OpenBSD__) || defined(__NetBSD__) || LJ_TARGET_UWP
  /* Allow better executable memory allocation for OpenBSD W^X mode. */
  void *p = mcode_alloc_at(J, 0, sz, MCPROT_RUN);
  if (p && mcode_setprot(p, sz, MCPROT_GEN)) {
    mcode_free(J, p, sz);
    return NULL;
  }
  return p;
#else
  return mcode_alloc_at(J, 0, sz, MCPROT_GEN);
#endif
}

#endif

/* -- MCode area management ----------------------------------------------- */

/* Allocate a new MCode area. */
static void mcode_allocarea(jit_State *J)
{
  MCode *oldarea = J->mcarea;
  size_t sz = (size_t)J->param[JIT_P_sizemcode] << 10;
  sz = (sz + LJ_PAGESIZE-1) & ~(size_t)(LJ_PAGESIZE - 1);
  J->mcarea = (MCode *)mcode_alloc(J, sz);
  J->szmcarea = sz;
  J->mcprot = MCPROT_GEN;
  J->mctop = (MCode *)((char *)J->mcarea + J->szmcarea);
  J->mcbot = (MCode *)((char *)J->mcarea + sizeof(MCLink));
  ((MCLink *)J->mcarea)->next = oldarea;
  ((MCLink *)J->mcarea)->size = sz;
  J->szallmcarea += sz;
  J->mcbot = (MCode *)lj_err_register_mcode(J->mcarea, sz, (uint8_t *)J->mcbot);
}

/* Free all MCode areas. */
void lj_mcode_free(jit_State *J)
{
  MCode *mc = J->mcarea;
  J->mcarea = NULL;
  J->szallmcarea = 0;
  while (mc) {
    MCode *next = ((MCLink *)mc)->next;
    size_t sz = ((MCLink *)mc)->size;
    lj_err_deregister_mcode(mc, sz, (uint8_t *)mc + sizeof(MCLink));
    mcode_free(J, mc, sz);
    mc = next;
  }
}

/* -- MCode transactions -------------------------------------------------- */

/* Reserve the remainder of the current MCode area. */
MCode *lj_mcode_reserve(jit_State *J, MCode **lim)
{
  if (!J->mcarea)
    mcode_allocarea(J);
  else
    mcode_protect(J, MCPROT_GEN);
  *lim = J->mcbot;
  return J->mctop;
}

/* Commit the top part of the current MCode area. */
void lj_mcode_commit(jit_State *J, MCode *top)
{
  J->mctop = top;
  mcode_protect(J, MCPROT_RUN);
}

/* Abort the reservation. */
void lj_mcode_abort(jit_State *J)
{
  if (J->mcarea)
    mcode_protect(J, MCPROT_RUN);
}

/* Set/reset protection to allow patching of MCode areas. */
MCode *lj_mcode_patch(jit_State *J, MCode *ptr, int finish)
{
  if (finish) {
#if LUAJIT_SECURITY_MCODE
    if (J->mcarea == ptr)
      mcode_protect(J, MCPROT_RUN);
    else if (LJ_UNLIKELY(mcode_setprot(ptr, ((MCLink *)ptr)->size, MCPROT_RUN)))
      mcode_protfail(J);
#endif
    return NULL;
  } else {
    MCode *mc = J->mcarea;
    /* Try current area first to use the protection cache. */
    if (ptr >= mc && ptr < (MCode *)((char *)mc + J->szmcarea)) {
#if LUAJIT_SECURITY_MCODE
      mcode_protect(J, MCPROT_GEN);
#endif
      return mc;
    }
    /* Otherwise search through the list of MCode areas. */
    for (;;) {
      mc = ((MCLink *)mc)->next;
      lj_assertJ(mc != NULL, "broken MCode area chain");
      if (ptr >= mc && ptr < (MCode *)((char *)mc + ((MCLink *)mc)->size)) {
#if LUAJIT_SECURITY_MCODE
	if (LJ_UNLIKELY(mcode_setprot(mc, ((MCLink *)mc)->size, MCPROT_GEN)))
	  mcode_protfail(J);
#endif
	return mc;
      }
    }
  }
}

/* Limit of MCode reservation reached. */
void lj_mcode_limiterr(jit_State *J, size_t need)
{
  size_t sizemcode, maxmcode;
  lj_mcode_abort(J);
  sizemcode = (size_t)J->param[JIT_P_sizemcode] << 10;
  sizemcode = (sizemcode + LJ_PAGESIZE-1) & ~(size_t)(LJ_PAGESIZE - 1);
  maxmcode = (size_t)J->param[JIT_P_maxmcode] << 10;
  if (need * sizeof(MCode) > sizemcode)
    lj_trace_err(J, LJ_TRERR_MCODEOV);  /* Too long for any area. */
  if (J->szallmcarea + sizemcode > maxmcode)
    lj_trace_err(J, LJ_TRERR_MCODEAL);
  mcode_allocarea(J);
  lj_trace_err(J, LJ_TRERR_MCODELM);  /* Retry with new area. */
}

#endif
