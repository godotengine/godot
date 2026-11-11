/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2026 Sam Lantinga <slouken@libsdl.org>

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

/* WIKI CATEGORY: CPUInfo */

/**
 * # CategoryCPUInfo
 *
 * CPU feature detection for SDL.
 *
 * These functions are largely concerned with reporting if the system has
 * access to various SIMD instruction sets, but also has other important info
 * to share, such as system RAM size and number of logical CPU cores.
 *
 * CPU instruction set checks, like SDL_HasSSE() and SDL_HasNEON(), are
 * available on all platforms, even if they don't make sense (an ARM processor
 * will never have SSE and an x86 processor will never have NEON, for example,
 * but these functions still exist and will simply return false in these
 * cases).
 */

#ifndef SDL_cpuinfo_h_
#define SDL_cpuinfo_h_

#include <SDL3/SDL_stdinc.h>

#include <SDL3/SDL_begin_code.h>
/* Set up for C function definitions, even when using C++ */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * A guess for the cacheline size used for padding.
 *
 * Most x86 processors have a 64 byte cache line. The 64-bit PowerPC
 * processors have a 128 byte cache line. We use the larger value to be
 * generally safe.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_CACHELINE_SIZE  128

/**
 * Get the number of logical CPU cores available.
 *
 * \returns the total number of logical CPU cores. On CPUs that include
 *          technologies such as hyperthreading, the number of logical cores
 *          may be more than the number of physical cores.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC int SDLCALL SDL_GetNumLogicalCPUCores(void);

/**
 * Determine the L1 cache line size of the CPU.
 *
 * This is useful for determining multi-threaded structure padding or SIMD
 * prefetch sizes.
 *
 * \returns the L1 cache line size of the CPU, in bytes.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC int SDLCALL SDL_GetCPUCacheLineSize(void);

/**
 * Determine whether the CPU has AltiVec features.
 *
 * This always returns false on CPUs that aren't using PowerPC instruction
 * sets.
 *
 * \returns true if the CPU has AltiVec features or false if not.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC bool SDLCALL SDL_HasAltiVec(void);

/**
 * Determine whether the CPU has MMX features.
 *
 * This always returns false on CPUs that aren't using Intel instruction sets.
 *
 * \returns true if the CPU has MMX features or false if not.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC bool SDLCALL SDL_HasMMX(void);

/**
 * Determine whether the CPU has SSE features.
 *
 * This always returns false on CPUs that aren't using Intel instruction sets.
 *
 * \returns true if the CPU has SSE features or false if not.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_HasSSE2
 * \sa SDL_HasSSE3
 * \sa SDL_HasSSE41
 * \sa SDL_HasSSE42
 */
extern SDL_DECLSPEC bool SDLCALL SDL_HasSSE(void);

/**
 * Determine whether the CPU has SSE2 features.
 *
 * This always returns false on CPUs that aren't using Intel instruction sets.
 *
 * \returns true if the CPU has SSE2 features or false if not.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_HasSSE
 * \sa SDL_HasSSE3
 * \sa SDL_HasSSE41
 * \sa SDL_HasSSE42
 */
extern SDL_DECLSPEC bool SDLCALL SDL_HasSSE2(void);

/**
 * Determine whether the CPU has SSE3 features.
 *
 * This always returns false on CPUs that aren't using Intel instruction sets.
 *
 * \returns true if the CPU has SSE3 features or false if not.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_HasSSE
 * \sa SDL_HasSSE2
 * \sa SDL_HasSSE41
 * \sa SDL_HasSSE42
 */
extern SDL_DECLSPEC bool SDLCALL SDL_HasSSE3(void);

/**
 * Determine whether the CPU has SSE4.1 features.
 *
 * This always returns false on CPUs that aren't using Intel instruction sets.
 *
 * \returns true if the CPU has SSE4.1 features or false if not.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_HasSSE
 * \sa SDL_HasSSE2
 * \sa SDL_HasSSE3
 * \sa SDL_HasSSE42
 */
extern SDL_DECLSPEC bool SDLCALL SDL_HasSSE41(void);

/**
 * Determine whether the CPU has SSE4.2 features.
 *
 * This always returns false on CPUs that aren't using Intel instruction sets.
 *
 * \returns true if the CPU has SSE4.2 features or false if not.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_HasSSE
 * \sa SDL_HasSSE2
 * \sa SDL_HasSSE3
 * \sa SDL_HasSSE41
 */
extern SDL_DECLSPEC bool SDLCALL SDL_HasSSE42(void);

/**
 * Determine whether the CPU has AVX features.
 *
 * This always returns false on CPUs that aren't using Intel instruction sets.
 *
 * \returns true if the CPU has AVX features or false if not.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_HasAVX2
 * \sa SDL_HasAVX512F
 */
extern SDL_DECLSPEC bool SDLCALL SDL_HasAVX(void);

/**
 * Determine whether the CPU has AVX2 features.
 *
 * This always returns false on CPUs that aren't using Intel instruction sets.
 *
 * \returns true if the CPU has AVX2 features or false if not.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_HasAVX
 * \sa SDL_HasAVX512F
 */
extern SDL_DECLSPEC bool SDLCALL SDL_HasAVX2(void);

/**
 * Determine whether the CPU has AVX-512F (foundation) features.
 *
 * This always returns false on CPUs that aren't using Intel instruction sets.
 *
 * \returns true if the CPU has AVX-512F features or false if not.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_HasAVX
 * \sa SDL_HasAVX2
 */
extern SDL_DECLSPEC bool SDLCALL SDL_HasAVX512F(void);

/**
 * Determine whether the CPU has ARM SIMD (ARMv6) features.
 *
 * This is different from ARM NEON, which is a different instruction set.
 *
 * This always returns false on CPUs that aren't using ARM instruction sets.
 *
 * \returns true if the CPU has ARM SIMD features or false if not.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_HasNEON
 */
extern SDL_DECLSPEC bool SDLCALL SDL_HasARMSIMD(void);

/**
 * Determine whether the CPU has NEON (ARM SIMD) features.
 *
 * This always returns false on CPUs that aren't using ARM instruction sets.
 *
 * \returns true if the CPU has ARM NEON features or false if not.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC bool SDLCALL SDL_HasNEON(void);

/**
 * Determine whether the CPU has LSX (LOONGARCH SIMD) features.
 *
 * This always returns false on CPUs that aren't using LOONGARCH instruction
 * sets.
 *
 * \returns true if the CPU has LOONGARCH LSX features or false if not.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC bool SDLCALL SDL_HasLSX(void);

/**
 * Determine whether the CPU has LASX (LOONGARCH SIMD) features.
 *
 * This always returns false on CPUs that aren't using LOONGARCH instruction
 * sets.
 *
 * \returns true if the CPU has LOONGARCH LASX features or false if not.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC bool SDLCALL SDL_HasLASX(void);

/**
 * Get the amount of RAM configured in the system.
 *
 * \returns the amount of RAM configured in the system in MiB.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC int SDLCALL SDL_GetSystemRAM(void);

/**
 * Report the alignment this system needs for SIMD allocations.
 *
 * This will return the minimum number of bytes to which a pointer must be
 * aligned to be compatible with SIMD instructions on the current machine. For
 * example, if the machine supports SSE only, it will return 16, but if it
 * supports AVX-512F, it'll return 64 (etc). This only reports values for
 * instruction sets SDL knows about, so if your SDL build doesn't have
 * SDL_HasAVX512F(), then it might return 16 for the SSE support it sees and
 * not 64 for the AVX-512 instructions that exist but SDL doesn't know about.
 * Plan accordingly.
 *
 * \returns the alignment in bytes needed for available, known SIMD
 *          instructions.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_aligned_alloc
 * \sa SDL_aligned_free
 */
extern SDL_DECLSPEC size_t SDLCALL SDL_GetSIMDAlignment(void);

/**
 * Report the size of a page of memory.
 *
 * Different platforms might have different memory page sizes. In current
 * times, 4 kilobytes is not unusual, but newer systems are moving to larger
 * page sizes, and esoteric platforms might have any unexpected size.
 *
 * Note that this function can return 0, which means SDL can't determine the
 * page size on this platform. It will _not_ set an error string to be
 * retrieved with SDL_GetError() in this case! In this case, defaulting to
 * 4096 is often a reasonable option.
 *
 * \returns the size of a single page of memory, in bytes, or 0 if SDL can't
 *          determine this information.
 *
 * \threadsafety It is safe to call this function from any thread.
 *
 * \since This function is available since SDL 3.4.0.
 */
extern SDL_DECLSPEC int SDLCALL SDL_GetSystemPageSize(void);

/* Ends C function definitions when using C++ */
#ifdef __cplusplus
}
#endif
#include <SDL3/SDL_close_code.h>

#endif /* SDL_cpuinfo_h_ */
