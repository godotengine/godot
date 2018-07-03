/*
 *    Stack-less Just-In-Time compiler
 *
 *    Copyright Zoltan Herczeg (hzmester@freemail.hu). All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are
 * permitted provided that the following conditions are met:
 *
 *   1. Redistributions of source code must retain the above copyright notice, this list of
 *      conditions and the following disclaimer.
 *
 *   2. Redistributions in binary form must reproduce the above copyright notice, this list
 *      of conditions and the following disclaimer in the documentation and/or other materials
 *      provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER(S) AND CONTRIBUTORS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDER(S) OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _SLJIT_CONFIG_H_
#define _SLJIT_CONFIG_H_

/* --------------------------------------------------------------------- */
/*  Custom defines                                                       */
/* --------------------------------------------------------------------- */

/* Put your custom defines here. This empty section will never change
   which helps maintaining patches (with diff / patch utilities). */

/* --------------------------------------------------------------------- */
/*  Architecture                                                         */
/* --------------------------------------------------------------------- */

/* Architecture selection. */
/* #define SLJIT_CONFIG_X86_32 1 */
/* #define SLJIT_CONFIG_X86_64 1 */
/* #define SLJIT_CONFIG_ARM_V5 1 */
/* #define SLJIT_CONFIG_ARM_V7 1 */
/* #define SLJIT_CONFIG_ARM_THUMB2 1 */
/* #define SLJIT_CONFIG_ARM_64 1 */
/* #define SLJIT_CONFIG_PPC_32 1 */
/* #define SLJIT_CONFIG_PPC_64 1 */
/* #define SLJIT_CONFIG_MIPS_32 1 */
/* #define SLJIT_CONFIG_MIPS_64 1 */
/* #define SLJIT_CONFIG_SPARC_32 1 */
/* #define SLJIT_CONFIG_TILEGX 1 */

/* #define SLJIT_CONFIG_AUTO 1 */
/* #define SLJIT_CONFIG_UNSUPPORTED 1 */

/* --------------------------------------------------------------------- */
/*  Utilities                                                            */
/* --------------------------------------------------------------------- */

/* Useful for thread-safe compiling of global functions. */
#ifndef SLJIT_UTIL_GLOBAL_LOCK
/* Enabled by default */
#define SLJIT_UTIL_GLOBAL_LOCK 1
#endif

/* Implements a stack like data structure (by using mmap / VirtualAlloc). */
#ifndef SLJIT_UTIL_STACK
/* Enabled by default */
#define SLJIT_UTIL_STACK 1
#endif

/* Single threaded application. Does not require any locks. */
#ifndef SLJIT_SINGLE_THREADED
/* Disabled by default. */
#define SLJIT_SINGLE_THREADED 0
#endif

/* --------------------------------------------------------------------- */
/*  Configuration                                                        */
/* --------------------------------------------------------------------- */

/* If SLJIT_STD_MACROS_DEFINED is not defined, the application should
   define SLJIT_MALLOC, SLJIT_FREE, SLJIT_MEMCPY, and NULL. */
#ifndef SLJIT_STD_MACROS_DEFINED
/* Disabled by default. */
#define SLJIT_STD_MACROS_DEFINED 0
#endif

/* Executable code allocation:
   If SLJIT_EXECUTABLE_ALLOCATOR is not defined, the application should
   define SLJIT_MALLOC_EXEC, SLJIT_FREE_EXEC, and SLJIT_EXEC_OFFSET. */
#ifndef SLJIT_EXECUTABLE_ALLOCATOR
/* Enabled by default. */
#define SLJIT_EXECUTABLE_ALLOCATOR 1

/* When SLJIT_PROT_EXECUTABLE_ALLOCATOR is enabled SLJIT uses
   an allocator which does not set writable and executable
   permission flags at the same time. The trade-of is increased
   memory consumption and disabled dynamic code modifications. */
#ifndef SLJIT_PROT_EXECUTABLE_ALLOCATOR
/* Disabled by default. */
#define SLJIT_PROT_EXECUTABLE_ALLOCATOR 0
#endif

#endif

/* Force cdecl calling convention even if a better calling
   convention (e.g. fastcall) is supported by the C compiler.
   If this option is disabled (this is the default), functions
   called from JIT should be defined with SLJIT_FUNC attribute.
   Standard C functions can still be called by using the
   SLJIT_CALL_CDECL jump type. */
#ifndef SLJIT_USE_CDECL_CALLING_CONVENTION
/* Disabled by default */
#define SLJIT_USE_CDECL_CALLING_CONVENTION 0
#endif

/* Return with error when an invalid argument is passed. */
#ifndef SLJIT_ARGUMENT_CHECKS
/* Disabled by default */
#define SLJIT_ARGUMENT_CHECKS 0
#endif

/* Debug checks (assertions, etc.). */
#ifndef SLJIT_DEBUG
/* Enabled by default */
#define SLJIT_DEBUG 1
#endif

/* Verbose operations. */
#ifndef SLJIT_VERBOSE
/* Enabled by default */
#define SLJIT_VERBOSE 1
#endif

/*
  SLJIT_IS_FPU_AVAILABLE
    The availability of the FPU can be controlled by SLJIT_IS_FPU_AVAILABLE.
      zero value - FPU is NOT present.
      nonzero value - FPU is present.
*/

/* For further configurations, see the beginning of sljitConfigInternal.h */

#endif
