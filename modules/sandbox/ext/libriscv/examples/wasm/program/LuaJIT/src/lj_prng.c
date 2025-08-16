/*
** Pseudo-random number generation.
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
*/

#define lj_prng_c
#define LUA_CORE

/* To get the syscall prototype. */
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "lj_def.h"
#include "lj_arch.h"
#include "lj_prng.h"

/* -- PRNG step function -------------------------------------------------- */

/* This implements a Tausworthe PRNG with period 2^223. Based on:
**   Tables of maximally-equidistributed combined LFSR generators,
**   Pierre L'Ecuyer, 1991, table 3, 1st entry.
** Full-period ME-CF generator with L=64, J=4, k=223, N1=49.
**
** Important note: This PRNG is NOT suitable for cryptographic use!
**
** But it works fine for math.random(), which has an API that's not
** suitable for cryptography, anyway.
**
** When used as a securely seeded global PRNG, it substantially raises
** the difficulty for various attacks on the VM.
*/

/* Update generator i and compute a running xor of all states. */
#define TW223_GEN(rs, z, r, i, k, q, s) \
  z = rs->u[i]; \
  z = (((z<<q)^z) >> (k-s)) ^ ((z&((uint64_t)(int64_t)-1 << (64-k)))<<s); \
  r ^= z; rs->u[i] = z;

#define TW223_STEP(rs, z, r) \
  TW223_GEN(rs, z, r, 0, 63, 31, 18) \
  TW223_GEN(rs, z, r, 1, 58, 19, 28) \
  TW223_GEN(rs, z, r, 2, 55, 24,  7) \
  TW223_GEN(rs, z, r, 3, 47, 21,  8)

/* PRNG step function with uint64_t result. */
LJ_NOINLINE uint64_t LJ_FASTCALL lj_prng_u64(PRNGState *rs)
{
  uint64_t z, r = 0;
  TW223_STEP(rs, z, r)
  return r;
}

/* PRNG step function with double in uint64_t result. */
LJ_NOINLINE uint64_t LJ_FASTCALL lj_prng_u64d(PRNGState *rs)
{
  uint64_t z, r = 0;
  TW223_STEP(rs, z, r)
  /* Returns a double bit pattern in the range 1.0 <= d < 2.0. */
  return (r & U64x(000fffff,ffffffff)) | U64x(3ff00000,00000000);
}

/* Condition seed: ensure k[i] MSB of u[i] are non-zero. */
static LJ_AINLINE void lj_prng_condition(PRNGState *rs)
{
  if (rs->u[0] < (1u << 1)) rs->u[0] += (1u << 1);
  if (rs->u[1] < (1u << 6)) rs->u[1] += (1u << 6);
  if (rs->u[2] < (1u << 9)) rs->u[2] += (1u << 9);
  if (rs->u[3] < (1u << 17)) rs->u[3] += (1u << 17);
}

/* -- PRNG seeding from OS ------------------------------------------------ */

#if LUAJIT_SECURITY_PRNG == 0

/* Nothing to define. */

#elif LJ_TARGET_XBOX360

extern int XNetRandom(void *buf, unsigned int len);

#elif LJ_TARGET_PS3

extern int sys_get_random_number(void *buf, uint64_t len);

#elif LJ_TARGET_PS4 || LJ_TARGET_PS5 || LJ_TARGET_PSVITA

extern int sceRandomGetRandomNumber(void *buf, size_t len);

#elif LJ_TARGET_NX

#include <unistd.h>

#elif LJ_TARGET_WINDOWS || LJ_TARGET_XBOXONE

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#if LJ_TARGET_UWP || LJ_TARGET_XBOXONE
/* Must use BCryptGenRandom. */
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#else
/* If you wonder about this mess, then search online for RtlGenRandom. */
typedef BOOLEAN (WINAPI *PRGR)(void *buf, ULONG len);
static PRGR libfunc_rgr;
#endif

#elif LJ_TARGET_POSIX

#if LJ_TARGET_LINUX
/* Avoid a dependency on glibc 2.25+ and use the getrandom syscall instead. */
#include <sys/syscall.h>
#else

#if LJ_TARGET_OSX && !LJ_TARGET_IOS
/*
** In their infinite wisdom Apple decided to disallow getentropy() in the
** iOS App Store. Even though the call is common to all BSD-ish OS, it's
** recommended by Apple in their own security-related docs, and, to top
** off the foolery, /dev/urandom is handled by the same kernel code,
** yet accessing it is actually permitted (but less efficient).
*/
#include <Availability.h>
#if __MAC_OS_X_VERSION_MIN_REQUIRED >= 101200
#define LJ_TARGET_HAS_GETENTROPY	1
#endif
#elif (LJ_TARGET_BSD && !defined(__NetBSD__)) || LJ_TARGET_SOLARIS || LJ_TARGET_CYGWIN || LJ_TARGET_QNX
#define LJ_TARGET_HAS_GETENTROPY	1
#endif

#if LJ_TARGET_HAS_GETENTROPY
extern int getentropy(void *buf, size_t len)
#ifdef __ELF__
  __attribute__((weak))
#endif
;
#endif

#endif

/* For the /dev/urandom fallback. */
#include <fcntl.h>
#include <unistd.h>

#endif

#if LUAJIT_SECURITY_PRNG == 0

/* If you really don't care about security, then define
** LUAJIT_SECURITY_PRNG=0. This yields a predictable seed
** and provides NO SECURITY against various attacks on the VM.
**
** BTW: This is NOT the way to get predictable table iteration,
** predictable trace generation, predictable bytecode generation, etc.
*/
int LJ_FASTCALL lj_prng_seed_secure(PRNGState *rs)
{
  lj_prng_seed_fixed(rs);  /* The fixed seed is already conditioned. */
  return 1;
}

#else

/* Securely seed PRNG from system entropy. Returns 0 on failure. */
int LJ_FASTCALL lj_prng_seed_secure(PRNGState *rs)
{
#if LJ_TARGET_XBOX360

  if (XNetRandom(rs->u, (unsigned int)sizeof(rs->u)) == 0)
    goto ok;

#elif LJ_TARGET_PS3

  if (sys_get_random_number(rs->u, sizeof(rs->u)) == 0)
    goto ok;

#elif LJ_TARGET_PS4 || LJ_TARGET_PS5 || LJ_TARGET_PSVITA

  if (sceRandomGetRandomNumber(rs->u, sizeof(rs->u)) == 0)
    goto ok;

#elif LJ_TARGET_NX

  if (getentropy(rs->u, sizeof(rs->u)) == 0)
    goto ok;

#elif LJ_TARGET_UWP || LJ_TARGET_XBOXONE

  if (BCryptGenRandom(NULL, (PUCHAR)(rs->u), (ULONG)sizeof(rs->u),
		      BCRYPT_USE_SYSTEM_PREFERRED_RNG) >= 0)
    goto ok;

#elif LJ_TARGET_WINDOWS

  /* Keep the library loaded in case multiple VMs are started. */
  if (!libfunc_rgr) {
    HMODULE lib = LJ_WIN_LOADLIBA("advapi32.dll");
    if (!lib) return 0;
    libfunc_rgr = (PRGR)GetProcAddress(lib, "SystemFunction036");
    if (!libfunc_rgr) return 0;
  }
  if (libfunc_rgr(rs->u, (ULONG)sizeof(rs->u)))
    goto ok;

#elif LJ_TARGET_POSIX

#if LJ_TARGET_LINUX && defined(SYS_getrandom)

  if (syscall(SYS_getrandom, rs->u, sizeof(rs->u), 0) == (long)sizeof(rs->u))
    goto ok;

#elif LJ_TARGET_HAS_GETENTROPY

#ifdef __ELF__
  if (&getentropy && getentropy(rs->u, sizeof(rs->u)) == 0)
    goto ok;
#else
  if (getentropy(rs->u, sizeof(rs->u)) == 0)
    goto ok;
#endif

#endif

  /* Fallback to /dev/urandom. This may fail if the device is not
  ** existent or accessible in a chroot or container, or if the process
  ** or the OS ran out of file descriptors.
  */
  {
    int fd = open("/dev/urandom", O_RDONLY|O_CLOEXEC);
    if (fd != -1) {
      ssize_t n = read(fd, rs->u, sizeof(rs->u));
      (void)close(fd);
      if (n == (ssize_t)sizeof(rs->u))
	goto ok;
    }
  }

#else

  /* Add an elif above for your OS with a secure PRNG seed.
  ** Note that fiddling around with rand(), getpid(), time() or coercing
  ** ASLR to yield a few bits of randomness is not helpful.
  ** If you don't want any security, then don't pretend you have any
  ** and simply define LUAJIT_SECURITY_PRNG=0 for the build.
  */
#error "Missing secure PRNG seed for this OS"

#endif
  return 0;  /* Fail. */

ok:
  lj_prng_condition(rs);
  (void)lj_prng_u64(rs);
  return 1;  /* Success. */
}

#endif

