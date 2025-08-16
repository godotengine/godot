/*
** Target architecture selection.
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
*/

#ifndef _LJ_ARCH_H
#define _LJ_ARCH_H

#include "lua.h"

/* -- Target definitions -------------------------------------------------- */

/* Target endianess. */
#define LUAJIT_LE	0
#define LUAJIT_BE	1

/* Target architectures. */
#define LUAJIT_ARCH_X86		1
#define LUAJIT_ARCH_x86		1
#define LUAJIT_ARCH_X64		2
#define LUAJIT_ARCH_x64		2
#define LUAJIT_ARCH_ARM		3
#define LUAJIT_ARCH_arm		3
#define LUAJIT_ARCH_ARM64	4
#define LUAJIT_ARCH_arm64	4
#define LUAJIT_ARCH_PPC		5
#define LUAJIT_ARCH_ppc		5
#define LUAJIT_ARCH_MIPS	6
#define LUAJIT_ARCH_mips	6
#define LUAJIT_ARCH_MIPS32	6
#define LUAJIT_ARCH_mips32	6
#define LUAJIT_ARCH_MIPS64	7
#define LUAJIT_ARCH_mips64	7
#define LUAJIT_ARCH_riscv64	8
#define LUAJIT_ARCH_RISCV64	8

/* Target OS. */
#define LUAJIT_OS_OTHER		0
#define LUAJIT_OS_WINDOWS	1
#define LUAJIT_OS_LINUX		2
#define LUAJIT_OS_OSX		3
#define LUAJIT_OS_BSD		4
#define LUAJIT_OS_POSIX		5

/* Number mode. */
#define LJ_NUMMODE_SINGLE	0	/* Single-number mode only. */
#define LJ_NUMMODE_SINGLE_DUAL	1	/* Default to single-number mode. */
#define LJ_NUMMODE_DUAL		2	/* Dual-number mode only. */
#define LJ_NUMMODE_DUAL_SINGLE	3	/* Default to dual-number mode. */

/* -- Target detection ---------------------------------------------------- */

/* Select native target if no target defined. */
#ifndef LUAJIT_TARGET

#if defined(__i386) || defined(__i386__) || defined(_M_IX86)
#define LUAJIT_TARGET	LUAJIT_ARCH_X86
#elif defined(__x86_64__) || defined(__x86_64) || defined(_M_X64) || defined(_M_AMD64)
#define LUAJIT_TARGET	LUAJIT_ARCH_X64
#elif defined(__arm__) || defined(__arm) || defined(__ARM__) || defined(__ARM)
#define LUAJIT_TARGET	LUAJIT_ARCH_ARM
#elif defined(__aarch64__) || defined(_M_ARM64)
#define LUAJIT_TARGET	LUAJIT_ARCH_ARM64
#elif defined(__ppc__) || defined(__ppc) || defined(__PPC__) || defined(__PPC) || defined(__powerpc__) || defined(__powerpc) || defined(__POWERPC__) || defined(__POWERPC) || defined(_M_PPC)
#define LUAJIT_TARGET	LUAJIT_ARCH_PPC
#elif defined(__mips64__) || defined(__mips64) || defined(__MIPS64__) || defined(__MIPS64)
#define LUAJIT_TARGET	LUAJIT_ARCH_MIPS64
#elif defined(__mips__) || defined(__mips) || defined(__MIPS__) || defined(__MIPS)
#define LUAJIT_TARGET	LUAJIT_ARCH_MIPS32
#elif (defined(__riscv) || defined(__riscv__)) && __riscv_xlen == 64
#define LUAJIT_TARGET LUAJIT_ARCH_RISCV64
#else
#error "Architecture not supported (in this version), see: https://luajit.org/status.html#architectures"
#endif

#endif

/* Select native OS if no target OS defined. */
#ifndef LUAJIT_OS

#if defined(_WIN32) && !defined(_XBOX_VER)
#define LUAJIT_OS	LUAJIT_OS_WINDOWS
#elif defined(__linux__)
#define LUAJIT_OS	LUAJIT_OS_LINUX
#elif defined(__MACH__) && defined(__APPLE__)
#include "TargetConditionals.h"
#define LUAJIT_OS	LUAJIT_OS_OSX
#elif (defined(__FreeBSD__) || defined(__FreeBSD_kernel__) || \
       defined(__NetBSD__) || defined(__OpenBSD__) || \
       defined(__DragonFly__)) && !defined(__ORBIS__) && !defined(__PROSPERO__)
#define LUAJIT_OS	LUAJIT_OS_BSD
#elif (defined(__sun__) && defined(__svr4__))
#define LJ_TARGET_SOLARIS	1
#define LUAJIT_OS	LUAJIT_OS_POSIX
#elif defined(__HAIKU__)
#define LUAJIT_OS	LUAJIT_OS_POSIX
#elif defined(__CYGWIN__)
#define LJ_TARGET_CYGWIN	1
#define LUAJIT_OS	LUAJIT_OS_POSIX
#elif defined(__QNX__)
#define LJ_TARGET_QNX		1
#define LUAJIT_OS	LUAJIT_OS_POSIX
#else
#define LUAJIT_OS	LUAJIT_OS_OTHER
#endif

#endif

/* Set target OS properties. */
#if LUAJIT_OS == LUAJIT_OS_WINDOWS
#define LJ_OS_NAME	"Windows"
#elif LUAJIT_OS == LUAJIT_OS_LINUX
#define LJ_OS_NAME	"Linux"
#elif LUAJIT_OS == LUAJIT_OS_OSX
#define LJ_OS_NAME	"OSX"
#elif LUAJIT_OS == LUAJIT_OS_BSD
#define LJ_OS_NAME	"BSD"
#elif LUAJIT_OS == LUAJIT_OS_POSIX
#define LJ_OS_NAME	"POSIX"
#else
#define LJ_OS_NAME	"Other"
#endif

#define LJ_TARGET_WINDOWS	(LUAJIT_OS == LUAJIT_OS_WINDOWS)
#define LJ_TARGET_LINUX		(LUAJIT_OS == LUAJIT_OS_LINUX)
#define LJ_TARGET_OSX		(LUAJIT_OS == LUAJIT_OS_OSX)
#define LJ_TARGET_BSD		(LUAJIT_OS == LUAJIT_OS_BSD)
#define LJ_TARGET_POSIX		(LUAJIT_OS > LUAJIT_OS_WINDOWS)
#define LJ_TARGET_DLOPEN	LJ_TARGET_POSIX

#if defined(TARGET_OS_IPHONE) && TARGET_OS_IPHONE
#define LJ_TARGET_IOS		1
#else
#define LJ_TARGET_IOS		0
#endif

#ifdef __CELLOS_LV2__
#define LJ_TARGET_PS3		1
#define LJ_TARGET_CONSOLE	1
#endif

#ifdef __ORBIS__
#define LJ_TARGET_PS4		1
#define LJ_TARGET_CONSOLE	1
#undef NULL
#define NULL ((void*)0)
#endif

#ifdef __PROSPERO__
#define LJ_TARGET_PS5		1
#define LJ_TARGET_CONSOLE	1
#undef NULL
#define NULL ((void*)0)
#endif

#ifdef __psp2__
#define LJ_TARGET_PSVITA	1
#define LJ_TARGET_CONSOLE	1
#endif

#if _XBOX_VER >= 200
#define LJ_TARGET_XBOX360	1
#define LJ_TARGET_CONSOLE	1
#endif

#ifdef _DURANGO
#define LJ_TARGET_XBOXONE	1
#define LJ_TARGET_CONSOLE	1
#define LJ_TARGET_GC64		1
#endif

#ifdef __NX__
#define LJ_TARGET_NX		1
#define LJ_TARGET_CONSOLE	1
#undef NULL
#define NULL ((void*)0)
#endif

#ifdef _UWP
#define LJ_TARGET_UWP		1
#if LUAJIT_TARGET == LUAJIT_ARCH_X64
#define LJ_TARGET_GC64		1
#endif
#endif

/* -- Arch-specific settings ---------------------------------------------- */

/* Set target architecture properties. */
#if LUAJIT_TARGET == LUAJIT_ARCH_X86

#define LJ_ARCH_NAME		"x86"
#define LJ_ARCH_BITS		32
#define LJ_ARCH_ENDIAN		LUAJIT_LE
#define LJ_TARGET_X86		1
#define LJ_TARGET_X86ORX64	1
#define LJ_TARGET_EHRETREG	0
#define LJ_TARGET_EHRAREG	8
#define LJ_TARGET_MASKSHIFT	1
#define LJ_TARGET_MASKROT	1
#define LJ_TARGET_UNALIGNED	1
#define LJ_ARCH_NUMMODE		LJ_NUMMODE_SINGLE_DUAL

#elif LUAJIT_TARGET == LUAJIT_ARCH_X64

#define LJ_ARCH_NAME		"x64"
#define LJ_ARCH_BITS		64
#define LJ_ARCH_ENDIAN		LUAJIT_LE
#define LJ_TARGET_X64		1
#define LJ_TARGET_X86ORX64	1
#define LJ_TARGET_EHRETREG	0
#define LJ_TARGET_EHRAREG	16
#define LJ_TARGET_JUMPRANGE	31	/* +-2^31 = +-2GB */
#define LJ_TARGET_MASKSHIFT	1
#define LJ_TARGET_MASKROT	1
#define LJ_TARGET_UNALIGNED	1
#define LJ_ARCH_NUMMODE		LJ_NUMMODE_SINGLE_DUAL
#ifndef LUAJIT_DISABLE_GC64
#define LJ_TARGET_GC64		1
#elif LJ_TARGET_OSX
#error "macOS requires GC64 -- don't disable it"
#endif

#elif LUAJIT_TARGET == LUAJIT_ARCH_ARM

#define LJ_ARCH_NAME		"arm"
#define LJ_ARCH_BITS		32
#define LJ_ARCH_ENDIAN		LUAJIT_LE
#if !defined(LJ_ARCH_HASFPU) && __SOFTFP__
#define LJ_ARCH_HASFPU		0
#endif
#if !defined(LJ_ABI_SOFTFP) && !__ARM_PCS_VFP
#define LJ_ABI_SOFTFP		1
#endif
#define LJ_ABI_EABI		1
#define LJ_TARGET_ARM		1
#define LJ_TARGET_EHRETREG	0
#define LJ_TARGET_EHRAREG	14
#define LJ_TARGET_JUMPRANGE	25	/* +-2^25 = +-32MB */
#define LJ_TARGET_MASKSHIFT	0
#define LJ_TARGET_MASKROT	1
#define LJ_TARGET_UNIFYROT	2	/* Want only IR_BROR. */
#define LJ_ARCH_NUMMODE		LJ_NUMMODE_DUAL

#if __ARM_ARCH >= 8 || __ARM_ARCH_8__ || __ARM_ARCH_8A__
#define LJ_ARCH_VERSION		80
#elif __ARM_ARCH == 7 || __ARM_ARCH_7__ || __ARM_ARCH_7A__ || __ARM_ARCH_7R__ || __ARM_ARCH_7S__ || __ARM_ARCH_7VE__
#define LJ_ARCH_VERSION		70
#elif __ARM_ARCH_6T2__
#define LJ_ARCH_VERSION		61
#elif __ARM_ARCH == 6 || __ARM_ARCH_6__ || __ARM_ARCH_6J__ || __ARM_ARCH_6K__ || __ARM_ARCH_6Z__ || __ARM_ARCH_6ZK__
#define LJ_ARCH_VERSION		60
#else
#define LJ_ARCH_VERSION		50
#endif

#elif LUAJIT_TARGET == LUAJIT_ARCH_ARM64

#define LJ_ARCH_BITS		64
#if defined(__AARCH64EB__)
#define LJ_ARCH_NAME		"arm64be"
#define LJ_ARCH_ENDIAN		LUAJIT_BE
#else
#define LJ_ARCH_NAME		"arm64"
#define LJ_ARCH_ENDIAN		LUAJIT_LE
#endif
#if !defined(LJ_ABI_PAUTH) && defined(__arm64e__)
#define LJ_ABI_PAUTH		1
#endif
#define LJ_TARGET_ARM64		1
#define LJ_TARGET_EHRETREG	0
#define LJ_TARGET_EHRAREG	30
#define LJ_TARGET_JUMPRANGE	27	/* +-2^27 = +-128MB */
#define LJ_TARGET_MASKSHIFT	1
#define LJ_TARGET_MASKROT	1
#define LJ_TARGET_UNIFYROT	2	/* Want only IR_BROR. */
#define LJ_TARGET_GC64		1
#define LJ_ARCH_NUMMODE		LJ_NUMMODE_DUAL

#define LJ_ARCH_VERSION		80

#elif LUAJIT_TARGET == LUAJIT_ARCH_PPC

#ifndef LJ_ARCH_ENDIAN
#if __BYTE_ORDER__ != __ORDER_BIG_ENDIAN__
#define LJ_ARCH_ENDIAN		LUAJIT_LE
#else
#define LJ_ARCH_ENDIAN		LUAJIT_BE
#endif
#endif

#if _LP64
#define LJ_ARCH_BITS		64
#if LJ_ARCH_ENDIAN == LUAJIT_LE
#define LJ_ARCH_NAME		"ppc64le"
#else
#define LJ_ARCH_NAME		"ppc64"
#endif
#else
#define LJ_ARCH_BITS		32
#define LJ_ARCH_NAME		"ppc"

#if !defined(LJ_ARCH_HASFPU)
#if defined(_SOFT_FLOAT) || defined(_SOFT_DOUBLE)
#define LJ_ARCH_HASFPU		0
#else
#define LJ_ARCH_HASFPU		1
#endif
#endif

#if !defined(LJ_ABI_SOFTFP)
#if defined(_SOFT_FLOAT) || defined(_SOFT_DOUBLE)
#define LJ_ABI_SOFTFP		1
#else
#define LJ_ABI_SOFTFP		0
#endif
#endif
#endif

#if LJ_ABI_SOFTFP
#define LJ_ARCH_NUMMODE		LJ_NUMMODE_DUAL
#else
#define LJ_ARCH_NUMMODE		LJ_NUMMODE_DUAL_SINGLE
#endif

#define LJ_TARGET_PPC		1
#define LJ_TARGET_EHRETREG	3
#define LJ_TARGET_EHRAREG	65
#define LJ_TARGET_JUMPRANGE	25	/* +-2^25 = +-32MB */
#define LJ_TARGET_MASKSHIFT	0
#define LJ_TARGET_MASKROT	1
#define LJ_TARGET_UNIFYROT	1	/* Want only IR_BROL. */

#if LJ_TARGET_CONSOLE
#define LJ_ARCH_PPC32ON64	1
#define LJ_ARCH_NOFFI		1
#elif LJ_ARCH_BITS == 64
#error "No support for PPC64"
#undef LJ_TARGET_PPC
#endif

#if _ARCH_PWR7
#define LJ_ARCH_VERSION		70
#elif _ARCH_PWR6
#define LJ_ARCH_VERSION		60
#elif _ARCH_PWR5X
#define LJ_ARCH_VERSION		51
#elif _ARCH_PWR5
#define LJ_ARCH_VERSION		50
#elif _ARCH_PWR4
#define LJ_ARCH_VERSION		40
#else
#define LJ_ARCH_VERSION		0
#endif
#if _ARCH_PPCSQ
#define LJ_ARCH_SQRT		1
#endif
#if _ARCH_PWR5X
#define LJ_ARCH_ROUND		1
#endif
#if __PPU__
#define LJ_ARCH_CELL		1
#endif
#if LJ_TARGET_XBOX360
#define LJ_ARCH_XENON		1
#endif

#elif LUAJIT_TARGET == LUAJIT_ARCH_MIPS32 || LUAJIT_TARGET == LUAJIT_ARCH_MIPS64

#if defined(__MIPSEL__) || defined(__MIPSEL) || defined(_MIPSEL)
#if __mips_isa_rev >= 6
#define LJ_TARGET_MIPSR6	1
#define LJ_TARGET_UNALIGNED	1
#endif
#if LUAJIT_TARGET == LUAJIT_ARCH_MIPS32
#if LJ_TARGET_MIPSR6
#define LJ_ARCH_NAME		"mips32r6el"
#else
#define LJ_ARCH_NAME		"mipsel"
#endif
#else
#if LJ_TARGET_MIPSR6
#define LJ_ARCH_NAME		"mips64r6el"
#else
#define LJ_ARCH_NAME		"mips64el"
#endif
#endif
#define LJ_ARCH_ENDIAN		LUAJIT_LE
#else
#if LUAJIT_TARGET == LUAJIT_ARCH_MIPS32
#if LJ_TARGET_MIPSR6
#define LJ_ARCH_NAME		"mips32r6"
#else
#define LJ_ARCH_NAME		"mips"
#endif
#else
#if LJ_TARGET_MIPSR6
#define LJ_ARCH_NAME		"mips64r6"
#else
#define LJ_ARCH_NAME		"mips64"
#endif
#endif
#define LJ_ARCH_ENDIAN		LUAJIT_BE
#endif

#if !defined(LJ_ARCH_HASFPU)
#ifdef __mips_soft_float
#define LJ_ARCH_HASFPU		0
#else
#define LJ_ARCH_HASFPU		1
#endif
#endif

#if !defined(LJ_ABI_SOFTFP)
#ifdef __mips_soft_float
#define LJ_ABI_SOFTFP		1
#else
#define LJ_ABI_SOFTFP		0
#endif
#endif

#if LUAJIT_TARGET == LUAJIT_ARCH_MIPS32
#define LJ_ARCH_BITS		32
#define LJ_TARGET_MIPS32	1
#else
#define LJ_ARCH_BITS		64
#define LJ_TARGET_MIPS64	1
#define LJ_TARGET_GC64		1
#endif
#define LJ_TARGET_MIPS		1
#define LJ_TARGET_EHRETREG	4
#define LJ_TARGET_EHRAREG	31
#define LJ_TARGET_JUMPRANGE	27	/* 2*2^27 = 256MB-aligned region */
#define LJ_TARGET_MASKSHIFT	1
#define LJ_TARGET_MASKROT	1
#define LJ_TARGET_UNIFYROT	2	/* Want only IR_BROR. */
#define LJ_ARCH_NUMMODE		LJ_NUMMODE_DUAL

#if LJ_TARGET_MIPSR6
#define LJ_ARCH_VERSION		60
#elif _MIPS_ARCH_MIPS32R2 || _MIPS_ARCH_MIPS64R2
#define LJ_ARCH_VERSION		20
#else
#define LJ_ARCH_VERSION		10
#endif

#elif LUAJIT_TARGET == LUAJIT_ARCH_RISCV64

#define LJ_ARCH_NAME		"riscv64"
#define LJ_ARCH_BITS		64
#define LJ_ARCH_ENDIAN		LUAJIT_LE	/* Forget about BE for now */
#define LJ_TARGET_RISCV64	1
#define LJ_TARGET_GC64		1
#define LJ_TARGET_EHRETREG	10
#define LJ_TARGET_EHRAREG	1
#define LJ_TARGET_JUMPRANGE	30	/* JAL +-2^20 = +-1MB,\
        AUIPC+JALR +-2^31 = +-2GB, leave 1 bit to avoid AUIPC corner case */
#define LJ_TARGET_MASKSHIFT	1
#define LJ_TARGET_MASKROT	1
#define LJ_ARCH_NUMMODE		LJ_NUMMODE_DUAL

#else
#error "No target architecture defined"
#endif

/* -- Checks for requirements --------------------------------------------- */

/* Check for minimum required compiler versions. */
#if defined(__GNUC__)
#if LJ_TARGET_X86
#if (__GNUC__ < 3) || ((__GNUC__ == 3) && __GNUC_MINOR__ < 4)
#error "Need at least GCC 3.4 or newer"
#endif
#elif LJ_TARGET_X64
#if __GNUC__ < 4
#error "Need at least GCC 4.0 or newer"
#endif
#elif LJ_TARGET_ARM
#if (__GNUC__ < 4) || ((__GNUC__ == 4) && __GNUC_MINOR__ < 2)
#error "Need at least GCC 4.2 or newer"
#endif
#elif LJ_TARGET_ARM64
#if __clang__
#if ((__clang_major__ < 3) || ((__clang_major__ == 3) && __clang_minor__ < 5)) && !defined(__NX_TOOLCHAIN_MAJOR__)
#error "Need at least Clang 3.5 or newer"
#endif
#else
#if (__GNUC__ < 4) || ((__GNUC__ == 4) && __GNUC_MINOR__ < 8)
#error "Need at least GCC 4.8 or newer"
#endif
#endif
#elif !LJ_TARGET_PS3
#if __clang__
#if ((__clang_major__ < 3) || ((__clang_major__ == 3) && __clang_minor__ < 5))
#error "Need at least Clang 3.5 or newer"
#endif
#else
#if (__GNUC__ < 4) || ((__GNUC__ == 4) && __GNUC_MINOR__ < 3)
#error "Need at least GCC 4.3 or newer"
#endif
#endif
#endif
#endif

/* Check target-specific constraints. */
#ifndef _BUILDVM_H
#if LJ_TARGET_X64
#if __USING_SJLJ_EXCEPTIONS__
#error "Need a C compiler with native exception handling on x64"
#endif
#elif LJ_TARGET_ARM
#if defined(__ARMEB__)
#error "No support for big-endian ARM"
#undef LJ_TARGET_ARM
#endif
#if __ARM_ARCH_6M__ || __ARM_ARCH_7M__ || __ARM_ARCH_7EM__
#error "No support for Cortex-M CPUs"
#undef LJ_TARGET_ARM
#endif
#if !(__ARM_EABI__ || LJ_TARGET_IOS)
#error "Only ARM EABI or iOS 3.0+ ABI is supported"
#undef LJ_TARGET_ARM
#endif
#elif LJ_TARGET_ARM64
#if defined(_ILP32)
#error "No support for ILP32 model on ARM64"
#undef LJ_TARGET_ARM64
#endif
#elif LJ_TARGET_PPC
#if defined(_LITTLE_ENDIAN) && (!defined(_BYTE_ORDER) || (_BYTE_ORDER == _LITTLE_ENDIAN))
#error "No support for little-endian PPC32"
#undef LJ_TARGET_PPC
#endif
#if defined(__NO_FPRS__) && !defined(_SOFT_FLOAT)
#error "No support for PPC/e500, use LuaJIT 2.0"
#undef LJ_TARGET_PPC
#endif
#elif LJ_TARGET_MIPS32
#if !((defined(_MIPS_SIM_ABI32) && _MIPS_SIM == _MIPS_SIM_ABI32) || (defined(_ABIO32) && _MIPS_SIM == _ABIO32))
#error "Only o32 ABI supported for MIPS32"
#undef LJ_TARGET_MIPS
#endif
#if LJ_TARGET_MIPSR6
/* Not that useful, since most available r6 CPUs are 64 bit. */
#error "No support for MIPS32R6"
#undef LJ_TARGET_MIPS
#endif
#elif LJ_TARGET_MIPS64
#if !((defined(_MIPS_SIM_ABI64) && _MIPS_SIM == _MIPS_SIM_ABI64) || (defined(_ABI64) && _MIPS_SIM == _ABI64))
/* MIPS32ON64 aka n32 ABI support might be desirable, but difficult. */
#error "Only n64 ABI supported for MIPS64"
#undef LJ_TARGET_MIPS
#endif
#elif LJ_TARGET_RISCV64
#if !defined(__riscv_float_abi_double)
#error "Only RISC-V 64 double float supported for now"
#endif
#endif
#endif

/* -- Derived defines ----------------------------------------------------- */

/* Enable or disable the dual-number mode for the VM. */
#if (LJ_ARCH_NUMMODE == LJ_NUMMODE_SINGLE && LUAJIT_NUMMODE == 2) || \
    (LJ_ARCH_NUMMODE == LJ_NUMMODE_DUAL && LUAJIT_NUMMODE == 1)
#error "No support for this number mode on this architecture"
#endif
#if LJ_ARCH_NUMMODE == LJ_NUMMODE_DUAL || \
    (LJ_ARCH_NUMMODE == LJ_NUMMODE_DUAL_SINGLE && LUAJIT_NUMMODE != 1) || \
    (LJ_ARCH_NUMMODE == LJ_NUMMODE_SINGLE_DUAL && LUAJIT_NUMMODE == 2)
#define LJ_DUALNUM		1
#else
#define LJ_DUALNUM		0
#endif

#if LJ_TARGET_IOS || LJ_TARGET_CONSOLE
/* Runtime code generation is restricted on iOS. Complain to Apple, not me. */
/* Ditto for the consoles. Complain to Sony or MS, not me. */
#ifndef LUAJIT_ENABLE_JIT
#define LJ_OS_NOJIT		1
#endif
#endif

/* 64 bit GC references. */
#if LJ_TARGET_GC64
#define LJ_GC64			1
#else
#define LJ_GC64			0
#endif

/* 2-slot frame info. */
#if LJ_GC64
#define LJ_FR2			1
#else
#define LJ_FR2			0
#endif

/* Disable or enable the JIT compiler. */
#if defined(LUAJIT_DISABLE_JIT) || defined(LJ_ARCH_NOJIT) || defined(LJ_OS_NOJIT)
#define LJ_HASJIT		0
#else
#define LJ_HASJIT		1
#endif

/* Disable or enable the FFI extension. */
#if defined(LUAJIT_DISABLE_FFI) || defined(LJ_ARCH_NOFFI)
#define LJ_HASFFI		0
#else
#define LJ_HASFFI		1
#endif

/* Disable or enable the string buffer extension. */
#if defined(LUAJIT_DISABLE_BUFFER)
#define LJ_HASBUFFER		0
#else
#define LJ_HASBUFFER		1
#endif

#if defined(LUAJIT_DISABLE_PROFILE)
#define LJ_HASPROFILE		0
#elif LJ_TARGET_POSIX
#define LJ_HASPROFILE		1
#define LJ_PROFILE_SIGPROF	1
#elif LJ_TARGET_PS3
#define LJ_HASPROFILE		1
#define LJ_PROFILE_PTHREAD	1
#elif LJ_TARGET_WINDOWS || LJ_TARGET_XBOX360
#define LJ_HASPROFILE		1
#define LJ_PROFILE_WTHREAD	1
#else
#define LJ_HASPROFILE		0
#endif

#ifndef LJ_ARCH_HASFPU
#define LJ_ARCH_HASFPU		1
#endif
#ifndef LJ_ABI_SOFTFP
#define LJ_ABI_SOFTFP		0
#endif
#define LJ_SOFTFP		(!LJ_ARCH_HASFPU)
#define LJ_SOFTFP32		(LJ_SOFTFP && LJ_32)

#ifndef LJ_ABI_PAUTH
#define LJ_ABI_PAUTH		0
#endif

#if LJ_ARCH_ENDIAN == LUAJIT_BE
#define LJ_LE			0
#define LJ_BE			1
#define LJ_ENDIAN_SELECT(le, be)	be
#define LJ_ENDIAN_LOHI(lo, hi)		hi lo
#else
#define LJ_LE			1
#define LJ_BE			0
#define LJ_ENDIAN_SELECT(le, be)	le
#define LJ_ENDIAN_LOHI(lo, hi)		lo hi
#endif

#if LJ_ARCH_BITS == 32
#define LJ_32			1
#define LJ_64			0
#else
#define LJ_32			0
#define LJ_64			1
#endif

#ifndef LJ_TARGET_UNALIGNED
#define LJ_TARGET_UNALIGNED	0
#endif

#ifndef LJ_PAGESIZE
#define LJ_PAGESIZE		4096
#endif

/* Various workarounds for embedded operating systems or weak C runtimes. */
#if defined(__ANDROID__) || defined(__symbian__) || LJ_TARGET_XBOX360 || LJ_TARGET_WINDOWS
#define LUAJIT_NO_LOG2
#endif
#if LJ_TARGET_CONSOLE || (LJ_TARGET_IOS && __IPHONE_OS_VERSION_MIN_REQUIRED >= __IPHONE_8_0)
#define LJ_NO_SYSTEM		1
#endif

#if LJ_TARGET_WINDOWS || LJ_TARGET_CYGWIN
#define LJ_ABI_WIN		1
#else
#define LJ_ABI_WIN		0
#endif

#if LJ_TARGET_WINDOWS
#if LJ_TARGET_UWP
#define LJ_WIN_VALLOC	VirtualAllocFromApp
#define LJ_WIN_VPROTECT	VirtualProtectFromApp
extern void *LJ_WIN_LOADLIBA(const char *path);
#else
#define LJ_WIN_VALLOC	VirtualAlloc
#define LJ_WIN_VPROTECT	VirtualProtect
#define LJ_WIN_LOADLIBA(path)	LoadLibraryExA((path), NULL, 0)
#endif
#endif

#if defined(LUAJIT_NO_UNWIND) || __GNU_COMPACT_EH__ || defined(__symbian__) || LJ_TARGET_IOS || LJ_TARGET_PS3 || LJ_TARGET_PS4 || LJ_TARGET_PS5
#define LJ_NO_UNWIND		1
#endif

#if !LJ_NO_UNWIND && !defined(LUAJIT_UNWIND_INTERNAL) && (LJ_ABI_WIN || (defined(LUAJIT_UNWIND_EXTERNAL) && (defined(__GNUC__) || defined(__clang__))))
#define LJ_UNWIND_EXT		1
#else
#define LJ_UNWIND_EXT		0
#endif

#if LJ_UNWIND_EXT && LJ_HASJIT && !LJ_TARGET_ARM && !(LJ_ABI_WIN && LJ_TARGET_X86)
#define LJ_UNWIND_JIT		1
#else
#define LJ_UNWIND_JIT		0
#endif

/* Compatibility with Lua 5.1 vs. 5.2. */
#ifdef LUAJIT_ENABLE_LUA52COMPAT
#define LJ_52			1
#else
#define LJ_52			0
#endif

/* -- VM security --------------------------------------------------------- */

/* Don't make any changes here. Instead build with:
**   make "XCFLAGS=-DLUAJIT_SECURITY_flag=value"
**
** Important note to distro maintainers: DO NOT change the defaults for a
** regular distro build -- neither upwards, nor downwards!
** These build-time configurable security flags are intended for embedders
** who may have specific needs wrt. security vs. performance.
*/

/* Security defaults. */
#ifndef LUAJIT_SECURITY_PRNG
/* PRNG init: 0 = fixed/insecure, 1 = secure from OS. */
#define LUAJIT_SECURITY_PRNG	1
#endif

#ifndef LUAJIT_SECURITY_STRHASH
/* String hash: 0 = sparse only, 1 = sparse + dense. */
#define LUAJIT_SECURITY_STRHASH	1
#endif

#ifndef LUAJIT_SECURITY_STRID
/* String IDs: 0 = linear, 1 = reseed < 255, 2 = reseed < 15, 3 = random. */
#define LUAJIT_SECURITY_STRID	1
#endif

#ifndef LUAJIT_SECURITY_MCODE
/* Machine code page protection: 0 = insecure RWX, 1 = secure RW^X. */
#define LUAJIT_SECURITY_MCODE	1
#endif

#define LJ_SECURITY_MODE \
  ( 0u \
  | ((LUAJIT_SECURITY_PRNG & 3) << 0) \
  | ((LUAJIT_SECURITY_STRHASH & 3) << 2) \
  | ((LUAJIT_SECURITY_STRID & 3) << 4) \
  | ((LUAJIT_SECURITY_MCODE & 3) << 6) \
  )
#define LJ_SECURITY_MODESTRING \
  "\004prng\007strhash\005strid\005mcode"

#endif
