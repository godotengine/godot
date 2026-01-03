/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/

/**
 * # CategoryAtomic
 *
 * Atomic operations.
 *
 * IMPORTANT: If you are not an expert in concurrent lockless programming, you
 * should not be using any functions in this file. You should be protecting
 * your data structures with full mutexes instead.
 *
 * ***Seriously, here be dragons!***
 *
 * You can find out a little more about lockless programming and the subtle
 * issues that can arise here:
 * https://learn.microsoft.com/en-us/windows/win32/dxtecharts/lockless-programming
 *
 * There's also lots of good information here:
 *
 * - https://www.1024cores.net/home/lock-free-algorithms
 * - https://preshing.com/
 *
 * These operations may or may not actually be implemented using processor
 * specific atomic operations. When possible they are implemented as true
 * processor specific atomic operations. When that is not possible the are
 * implemented using locks that *do* use the available atomic operations.
 *
 * All of the atomic operations that modify memory are full memory barriers.
 */

#ifndef SDL_atomic_h_
#define SDL_atomic_h_

#include <SDL3/SDL_stdinc.h>
#include <SDL3/SDL_platform_defines.h>

#include <SDL3/SDL_begin_code.h>

/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * An atomic spinlock.
 *
 * The atomic locks are efficient spinlocks using CPU instructions, but are
 * vulnerable to starvation and can spin forever if a thread holding a lock
 * has been terminated. For this reason you should minimize the code executed
 * inside an atomic lock and never do expensive things like API or system
 * calls while holding them.
 *
 * They are also vulnerable to starvation if the thread holding the lock is
 * lower priority than other threads and doesn't get scheduled. In general you
 * should use mutexes instead, since they have better performance and
 * contention behavior.
 *
 * The atomic locks are not safe to lock recursively.
 *
 * Porting Note: The spin lock functions and type are required and can not be
 * emulated because they are used in the atomic emulation code.
 */
typedef int SDL_SpinLock;

/**
 * Try to lock a spin lock by setting it to a non-zero value.
 *
 * ***Please note that spinlocks are dangerous if you don't know what you're
 * doing. Please be careful using any sort of spinlock!***
 *
 * \param lock a pointer to a lock variable.
 * \returns true if the lock succeeded, false if the lock is already held.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_LockSpinlock
 * \sa SDL_UnlockSpinlock
 */
extern SDL_DECLSPEC bool SDLCALL SDL_TryLockSpinlock(SDL_SpinLock *lock);

/**
 * Lock a spin lock by setting it to a non-zero value.
 *
 * ***Please note that spinlocks are dangerous if you don't know what you're
 * doing. Please be careful using any sort of spinlock!***
 *
 * \param lock a pointer to a lock variable.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_TryLockSpinlock
 * \sa SDL_UnlockSpinlock
 */
extern SDL_DECLSPEC void SDLCALL SDL_LockSpinlock(SDL_SpinLock *lock);

/**
 * Unlock a spin lock by setting it to 0.
 *
 * Always returns immediately.
 *
 * ***Please note that spinlocks are dangerous if you don't know what you're
 * doing. Please be careful using any sort of spinlock!***
 *
 * \param lock a pointer to a lock variable.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_LockSpinlock
 * \sa SDL_TryLockSpinlock
 */
extern SDL_DECLSPEC void SDLCALL SDL_UnlockSpinlock(SDL_SpinLock *lock);


#ifdef SDL_WIKI_DOCUMENTATION_SECTION

/**
 * Mark a compiler barrier.
 *
 * A compiler barrier prevents the compiler from reordering reads and writes
 * to globally visible variables across the call.
 *
 * This macro only prevents the compiler from reordering reads and writes, it
 * does not prevent the CPU from reordering reads and writes. However, all of
 * the atomic operations that modify memory are full memory barriers.
 *
 * \threadsafety Obviously this macro is safe to use from any thread at any
 *               time, but if you find yourself needing this, you are probably
 *               dealing with some very sensitive code; be careful!
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_CompilerBarrier() DoCompilerSpecificReadWriteBarrier()

#elif defined(_MSC_VER) && (_MSC_VER > 1200) && !defined(__clang__)
void _ReadWriteBarrier(void);
#pragma intrinsic(_ReadWriteBarrier)
#define SDL_CompilerBarrier()   _ReadWriteBarrier()
#elif (defined(__GNUC__) && !defined(SDL_PLATFORM_EMSCRIPTEN)) || (defined(__SUNPRO_C) && (__SUNPRO_C >= 0x5120))
/* This is correct for all CPUs when using GCC or Solaris Studio 12.1+. */
#define SDL_CompilerBarrier()   __asm__ __volatile__ ("" : : : "memory")
#elif defined(__WATCOMC__)
extern __inline void SDL_CompilerBarrier(void);
#pragma aux SDL_CompilerBarrier = "" parm [] modify exact [];
#else
#define SDL_CompilerBarrier()   \
{ SDL_SpinLock _tmp = 0; SDL_LockSpinlock(&_tmp); SDL_UnlockSpinlock(&_tmp); }
#endif

/**
 * Insert a memory release barrier (function version).
 *
 * Please refer to SDL_MemoryBarrierRelease for details. This is a function
 * version, which might be useful if you need to use this functionality from a
 * scripting language, etc. Also, some of the macro versions call this
 * function behind the scenes, where more heavy lifting can happen inside of
 * SDL. Generally, though, an app written in C/C++/etc should use the macro
 * version, as it will be more efficient.
 *
 * \threadsafety Obviously this function is safe to use from any thread at any
 *               time, but if you find yourself needing this, you are probably
 *               dealing with some very sensitive code; be careful!
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_MemoryBarrierRelease
 */
extern SDL_DECLSPEC void SDLCALL SDL_MemoryBarrierReleaseFunction(void);

/**
 * Insert a memory acquire barrier (function version).
 *
 * Please refer to SDL_MemoryBarrierRelease for details. This is a function
 * version, which might be useful if you need to use this functionality from a
 * scripting language, etc. Also, some of the macro versions call this
 * function behind the scenes, where more heavy lifting can happen inside of
 * SDL. Generally, though, an app written in C/C++/etc should use the macro
 * version, as it will be more efficient.
 *
 * \threadsafety Obviously this function is safe to use from any thread at any
 *               time, but if you find yourself needing this, you are probably
 *               dealing with some very sensitive code; be careful!
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_MemoryBarrierAcquire
 */
extern SDL_DECLSPEC void SDLCALL SDL_MemoryBarrierAcquireFunction(void);


#ifdef SDL_WIKI_DOCUMENTATION_SECTION

/**
 * Insert a memory release barrier (macro version).
 *
 * Memory barriers are designed to prevent reads and writes from being
 * reordered by the compiler and being seen out of order on multi-core CPUs.
 *
 * A typical pattern would be for thread A to write some data and a flag, and
 * for thread B to read the flag and get the data. In this case you would
 * insert a release barrier between writing the data and the flag,
 * guaranteeing that the data write completes no later than the flag is
 * written, and you would insert an acquire barrier between reading the flag
 * and reading the data, to ensure that all the reads associated with the flag
 * have completed.
 *
 * In this pattern you should always see a release barrier paired with an
 * acquire barrier and you should gate the data reads/writes with a single
 * flag variable.
 *
 * For more information on these semantics, take a look at the blog post:
 * http://preshing.com/20120913/acquire-and-release-semantics
 *
 * This is the macro version of this functionality; if possible, SDL will use
 * compiler intrinsics or inline assembly, but some platforms might need to
 * call the function version of this, SDL_MemoryBarrierReleaseFunction to do
 * the heavy lifting. Apps that can use the macro should favor it over the
 * function.
 *
 * \threadsafety Obviously this macro is safe to use from any thread at any
 *               time, but if you find yourself needing this, you are probably
 *               dealing with some very sensitive code; be careful!
 *
 * \since This macro is available since SDL 3.2.0.
 *
 * \sa SDL_MemoryBarrierAcquire
 * \sa SDL_MemoryBarrierReleaseFunction
 */
#define SDL_MemoryBarrierRelease() SDL_MemoryBarrierReleaseFunction()

/**
 * Insert a memory acquire barrier (macro version).
 *
 * Please see SDL_MemoryBarrierRelease for the details on what memory barriers
 * are and when to use them.
 *
 * This is the macro version of this functionality; if possible, SDL will use
 * compiler intrinsics or inline assembly, but some platforms might need to
 * call the function version of this, SDL_MemoryBarrierAcquireFunction, to do
 * the heavy lifting. Apps that can use the macro should favor it over the
 * function.
 *
 * \threadsafety Obviously this macro is safe to use from any thread at any
 *               time, but if you find yourself needing this, you are probably
 *               dealing with some very sensitive code; be careful!
 *
 * \since This macro is available since SDL 3.2.0.
 *
 * \sa SDL_MemoryBarrierRelease
 * \sa SDL_MemoryBarrierAcquireFunction
 */
#define SDL_MemoryBarrierAcquire() SDL_MemoryBarrierAcquireFunction()

#elif defined(__GNUC__) && (defined(__powerpc__) || defined(__ppc__))
#define SDL_MemoryBarrierRelease()   __asm__ __volatile__ ("lwsync" : : : "memory")
#define SDL_MemoryBarrierAcquire()   __asm__ __volatile__ ("lwsync" : : : "memory")
#elif defined(__GNUC__) && defined(__aarch64__)
#define SDL_MemoryBarrierRelease()   __asm__ __volatile__ ("dmb ish" : : : "memory")
#define SDL_MemoryBarrierAcquire()   __asm__ __volatile__ ("dmb ish" : : : "memory")
#elif defined(__GNUC__) && defined(__arm__)
#if 0 /* defined(SDL_PLATFORM_LINUX) || defined(SDL_PLATFORM_ANDROID) */
/* Information from:
   https://chromium.googlesource.com/chromium/chromium/+/trunk/base/atomicops_internals_arm_gcc.h#19

   The Linux kernel provides a helper function which provides the right code for a memory barrier,
   hard-coded at address 0xffff0fa0
*/
typedef void (*SDL_KernelMemoryBarrierFunc)();
#define SDL_MemoryBarrierRelease()  ((SDL_KernelMemoryBarrierFunc)0xffff0fa0)()
#define SDL_MemoryBarrierAcquire()  ((SDL_KernelMemoryBarrierFunc)0xffff0fa0)()
#else
#if defined(__ARM_ARCH_7__) || defined(__ARM_ARCH_7A__) || defined(__ARM_ARCH_7EM__) || defined(__ARM_ARCH_7R__) || defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_7S__) || defined(__ARM_ARCH_8A__)
#define SDL_MemoryBarrierRelease()   __asm__ __volatile__ ("dmb ish" : : : "memory")
#define SDL_MemoryBarrierAcquire()   __asm__ __volatile__ ("dmb ish" : : : "memory")
#elif defined(__ARM_ARCH_6__) || defined(__ARM_ARCH_6J__) || defined(__ARM_ARCH_6K__) || defined(__ARM_ARCH_6T2__) || defined(__ARM_ARCH_6Z__) || defined(__ARM_ARCH_6ZK__)
#ifdef __thumb__
/* The mcr instruction isn't available in thumb mode, use real functions */
#define SDL_MEMORY_BARRIER_USES_FUNCTION
#define SDL_MemoryBarrierRelease()   SDL_MemoryBarrierReleaseFunction()
#define SDL_MemoryBarrierAcquire()   SDL_MemoryBarrierAcquireFunction()
#else
#define SDL_MemoryBarrierRelease()   __asm__ __volatile__ ("mcr p15, 0, %0, c7, c10, 5" : : "r"(0) : "memory")
#define SDL_MemoryBarrierAcquire()   __asm__ __volatile__ ("mcr p15, 0, %0, c7, c10, 5" : : "r"(0) : "memory")
#endif /* __thumb__ */
#else
#define SDL_MemoryBarrierRelease()   __asm__ __volatile__ ("" : : : "memory")
#define SDL_MemoryBarrierAcquire()   __asm__ __volatile__ ("" : : : "memory")
#endif /* SDL_PLATFORM_LINUX || SDL_PLATFORM_ANDROID */
#endif /* __GNUC__ && __arm__ */
#else
#if (defined(__SUNPRO_C) && (__SUNPRO_C >= 0x5120))
/* This is correct for all CPUs on Solaris when using Solaris Studio 12.1+. */
#include <mbarrier.h>
#define SDL_MemoryBarrierRelease()  __machine_rel_barrier()
#define SDL_MemoryBarrierAcquire()  __machine_acq_barrier()
#else
/* This is correct for the x86 and x64 CPUs, and we'll expand this over time. */
#define SDL_MemoryBarrierRelease()  SDL_CompilerBarrier()
#define SDL_MemoryBarrierAcquire()  SDL_CompilerBarrier()
#endif
#endif

/* "REP NOP" is PAUSE, coded for tools that don't know it by that name. */
#ifdef SDL_WIKI_DOCUMENTATION_SECTION

/**
 * A macro to insert a CPU-specific "pause" instruction into the program.
 *
 * This can be useful in busy-wait loops, as it serves as a hint to the CPU as
 * to the program's intent; some CPUs can use this to do more efficient
 * processing. On some platforms, this doesn't do anything, so using this
 * macro might just be a harmless no-op.
 *
 * Note that if you are busy-waiting, there are often more-efficient
 * approaches with other synchronization primitives: mutexes, semaphores,
 * condition variables, etc.
 *
 * \threadsafety This macro is safe to use from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_CPUPauseInstruction() DoACPUPauseInACompilerAndArchitectureSpecificWay

#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__i386__) || defined(__x86_64__))
    #define SDL_CPUPauseInstruction() __asm__ __volatile__("pause\n")  /* Some assemblers can't do REP NOP, so go with PAUSE. */
#elif (defined(__arm__) && defined(__ARM_ARCH) && __ARM_ARCH >= 7) || defined(__aarch64__)
    #define SDL_CPUPauseInstruction() __asm__ __volatile__("yield" ::: "memory")
#elif (defined(__powerpc__) || defined(__powerpc64__))
    #define SDL_CPUPauseInstruction() __asm__ __volatile__("or 27,27,27");
#elif (defined(__riscv) && __riscv_xlen == 64)
    #define SDL_CPUPauseInstruction() __asm__ __volatile__(".insn i 0x0F, 0, x0, x0, 0x010");
#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
    #define SDL_CPUPauseInstruction() _mm_pause()  /* this is actually "rep nop" and not a SIMD instruction. No inline asm in MSVC x86-64! */
#elif defined(_MSC_VER) && (defined(_M_ARM) || defined(_M_ARM64))
    #define SDL_CPUPauseInstruction() __yield()
#elif defined(__WATCOMC__) && defined(__386__)
    extern __inline void SDL_CPUPauseInstruction(void);
    #pragma aux SDL_CPUPauseInstruction = ".686p" ".xmm2" "pause"
#else
    #define SDL_CPUPauseInstruction()
#endif


/**
 * A type representing an atomic integer value.
 *
 * This can be used to manage a value that is synchronized across multiple
 * CPUs without a race condition; when an app sets a value with
 * SDL_SetAtomicInt all other threads, regardless of the CPU it is running on,
 * will see that value when retrieved with SDL_GetAtomicInt, regardless of CPU
 * caches, etc.
 *
 * This is also useful for atomic compare-and-swap operations: a thread can
 * change the value as long as its current value matches expectations. When
 * done in a loop, one can guarantee data consistency across threads without a
 * lock (but the usual warnings apply: if you don't know what you're doing, or
 * you don't do it carefully, you can confidently cause any number of
 * disasters with this, so in most cases, you _should_ use a mutex instead of
 * this!).
 *
 * This is a struct so people don't accidentally use numeric operations on it
 * directly. You have to use SDL atomic functions.
 *
 * \since This struct is available since SDL 3.2.0.
 *
 * \sa SDL_CompareAndSwapAtomicInt
 * \sa SDL_GetAtomicInt
 * \sa SDL_SetAtomicInt
 * \sa SDL_AddAtomicInt
 */
typedef struct SDL_AtomicInt { int value; } SDL_AtomicInt;

/**
 * Set an atomic variable to a new value if it is currently an old value.
 *
 * ***Note: If you don't know what this function is for, you shouldn't use
 * it!***
 *
 * \param a a pointer to an SDL_AtomicInt variable to be modified.
 * \param oldval the old value.
 * \param newval the new value.
 * \returns true if the atomic variable was set, false otherwise.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetAtomicInt
 * \sa SDL_SetAtomicInt
 */
extern SDL_DECLSPEC bool SDLCALL SDL_CompareAndSwapAtomicInt(SDL_AtomicInt *a, int oldval, int newval);

/**
 * Set an atomic variable to a value.
 *
 * This function also acts as a full memory barrier.
 *
 * ***Note: If you don't know what this function is for, you shouldn't use
 * it!***
 *
 * \param a a pointer to an SDL_AtomicInt variable to be modified.
 * \param v the desired value.
 * \returns the previous value of the atomic variable.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetAtomicInt
 */
extern SDL_DECLSPEC int SDLCALL SDL_SetAtomicInt(SDL_AtomicInt *a, int v);

/**
 * Get the value of an atomic variable.
 *
 * ***Note: If you don't know what this function is for, you shouldn't use
 * it!***
 *
 * \param a a pointer to an SDL_AtomicInt variable.
 * \returns the current value of an atomic variable.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_SetAtomicInt
 */
extern SDL_DECLSPEC int SDLCALL SDL_GetAtomicInt(SDL_AtomicInt *a);

/**
 * Add to an atomic variable.
 *
 * This function also acts as a full memory barrier.
 *
 * ***Note: If you don't know what this function is for, you shouldn't use
 * it!***
 *
 * \param a a pointer to an SDL_AtomicInt variable to be modified.
 * \param v the desired value to add.
 * \returns the previous value of the atomic variable.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_AtomicDecRef
 * \sa SDL_AtomicIncRef
 */
extern SDL_DECLSPEC int SDLCALL SDL_AddAtomicInt(SDL_AtomicInt *a, int v);

#ifndef SDL_AtomicIncRef

/**
 * Increment an atomic variable used as a reference count.
 *
 * ***Note: If you don't know what this macro is for, you shouldn't use it!***
 *
 * \param a a pointer to an SDL_AtomicInt to increment.
 * \returns the previous value of the atomic variable.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 *
 * \sa SDL_AtomicDecRef
 */
#define SDL_AtomicIncRef(a)    SDL_AddAtomicInt(a, 1)
#endif

#ifndef SDL_AtomicDecRef

/**
 * Decrement an atomic variable used as a reference count.
 *
 * ***Note: If you don't know what this macro is for, you shouldn't use it!***
 *
 * \param a a pointer to an SDL_AtomicInt to decrement.
 * \returns true if the variable reached zero after decrementing, false
 *          otherwise.
 *
 * \threadsafety It is safe to call this macro from any thread.
 *
 * \since This macro is available since SDL 3.2.0.
 *
 * \sa SDL_AtomicIncRef
 */
#define SDL_AtomicDecRef(a)    (SDL_AddAtomicInt(a, -1) == 1)
#endif

/**
 * A type representing an atomic unsigned 32-bit value.
 *
 * This can be used to manage a value that is synchronized across multiple
 * CPUs without a race condition; when an app sets a value with
 * SDL_SetAtomicU32 all other threads, regardless of the CPU it is running on,
 * will see that value when retrieved with SDL_GetAtomicU32, regardless of CPU
 * caches, etc.
 *
 * This is also useful for atomic compare-and-swap operations: a thread can
 * change the value as long as its current value matches expectations. When
 * done in a loop, one can guarantee data consistency across threads without a
 * lock (but the usual warnings apply: if you don't know what you're doing, or
 * you don't do it carefully, you can confidently cause any number of
 * disasters with this, so in most cases, you _should_ use a mutex instead of
 * this!).
 *
 * This is a struct so people don't accidentally use numeric operations on it
 * directly. You have to use SDL atomic functions.
 *
 * \since This struct is available since SDL 3.2.0.
 *
 * \sa SDL_CompareAndSwapAtomicU32
 * \sa SDL_GetAtomicU32
 * \sa SDL_SetAtomicU32
 */
typedef struct SDL_AtomicU32 { Uint32 value; } SDL_AtomicU32;

/**
 * Set an atomic variable to a new value if it is currently an old value.
 *
 * ***Note: If you don't know what this function is for, you shouldn't use
 * it!***
 *
 * \param a a pointer to an SDL_AtomicU32 variable to be modified.
 * \param oldval the old value.
 * \param newval the new value.
 * \returns true if the atomic variable was set, false otherwise.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetAtomicU32
 * \sa SDL_SetAtomicU32
 */
extern SDL_DECLSPEC bool SDLCALL SDL_CompareAndSwapAtomicU32(SDL_AtomicU32 *a, Uint32 oldval, Uint32 newval);

/**
 * Set an atomic variable to a value.
 *
 * This function also acts as a full memory barrier.
 *
 * ***Note: If you don't know what this function is for, you shouldn't use
 * it!***
 *
 * \param a a pointer to an SDL_AtomicU32 variable to be modified.
 * \param v the desired value.
 * \returns the previous value of the atomic variable.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_GetAtomicU32
 */
extern SDL_DECLSPEC Uint32 SDLCALL SDL_SetAtomicU32(SDL_AtomicU32 *a, Uint32 v);

/**
 * Get the value of an atomic variable.
 *
 * ***Note: If you don't know what this function is for, you shouldn't use
 * it!***
 *
 * \param a a pointer to an SDL_AtomicU32 variable.
 * \returns the current value of an atomic variable.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_SetAtomicU32
 */
extern SDL_DECLSPEC Uint32 SDLCALL SDL_GetAtomicU32(SDL_AtomicU32 *a);

/**
 * Set a pointer to a new value if it is currently an old value.
 *
 * ***Note: If you don't know what this function is for, you shouldn't use
 * it!***
 *
 * \param a a pointer to a pointer.
 * \param oldval the old pointer value.
 * \param newval the new pointer value.
 * \returns true if the pointer was set, false otherwise.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_CompareAndSwapAtomicInt
 * \sa SDL_GetAtomicPointer
 * \sa SDL_SetAtomicPointer
 */
extern SDL_DECLSPEC bool SDLCALL SDL_CompareAndSwapAtomicPointer(void **a, void *oldval, void *newval);

/**
 * Set a pointer to a value atomically.
 *
 * ***Note: If you don't know what this function is for, you shouldn't use
 * it!***
 *
 * \param a a pointer to a pointer.
 * \param v the desired pointer value.
 * \returns the previous value of the pointer.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_CompareAndSwapAtomicPointer
 * \sa SDL_GetAtomicPointer
 */
extern SDL_DECLSPEC void * SDLCALL SDL_SetAtomicPointer(void **a, void *v);

/**
 * Get the value of a pointer atomically.
 *
 * ***Note: If you don't know what this function is for, you shouldn't use
 * it!***
 *
 * \param a a pointer to a pointer.
 * \returns the current value of a pointer.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_CompareAndSwapAtomicPointer
 * \sa SDL_SetAtomicPointer
 */
extern SDL_DECLSPEC void * SDLCALL SDL_GetAtomicPointer(void **a);

/* Ends C function definitions when using C++ */
#ifdef __cplusplus
}
#endif

#include <SDL3/SDL_close_code.h>

#endif /* SDL_atomic_h_ */
