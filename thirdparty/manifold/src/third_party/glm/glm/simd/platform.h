#pragma once

///////////////////////////////////////////////////////////////////////////////////
// Platform

#define GLM_PLATFORM_UNKNOWN		0x00000000
#define GLM_PLATFORM_WINDOWS		0x00010000
#define GLM_PLATFORM_LINUX			0x00020000
#define GLM_PLATFORM_APPLE			0x00040000
//#define GLM_PLATFORM_IOS			0x00080000
#define GLM_PLATFORM_ANDROID		0x00100000
#define GLM_PLATFORM_CHROME_NACL	0x00200000
#define GLM_PLATFORM_UNIX			0x00400000
#define GLM_PLATFORM_QNXNTO			0x00800000
#define GLM_PLATFORM_WINCE			0x01000000
#define GLM_PLATFORM_CYGWIN			0x02000000

#ifdef GLM_FORCE_PLATFORM_UNKNOWN
#	define GLM_PLATFORM GLM_PLATFORM_UNKNOWN
#elif defined(__CYGWIN__)
#	define GLM_PLATFORM GLM_PLATFORM_CYGWIN
#elif defined(__QNXNTO__)
#	define GLM_PLATFORM GLM_PLATFORM_QNXNTO
#elif defined(__APPLE__)
#	define GLM_PLATFORM GLM_PLATFORM_APPLE
#elif defined(WINCE)
#	define GLM_PLATFORM GLM_PLATFORM_WINCE
#elif defined(_WIN32)
#	define GLM_PLATFORM GLM_PLATFORM_WINDOWS
#elif defined(__native_client__)
#	define GLM_PLATFORM GLM_PLATFORM_CHROME_NACL
#elif defined(__ANDROID__)
#	define GLM_PLATFORM GLM_PLATFORM_ANDROID
#elif defined(__linux)
#	define GLM_PLATFORM GLM_PLATFORM_LINUX
#elif defined(__unix)
#	define GLM_PLATFORM GLM_PLATFORM_UNIX
#else
#	define GLM_PLATFORM GLM_PLATFORM_UNKNOWN
#endif//

///////////////////////////////////////////////////////////////////////////////////
// Compiler

#define GLM_COMPILER_UNKNOWN		0x00000000

// Intel
#define GLM_COMPILER_INTEL			0x00100000
#define GLM_COMPILER_INTEL14		0x00100040
#define GLM_COMPILER_INTEL15		0x00100050
#define GLM_COMPILER_INTEL16		0x00100060
#define GLM_COMPILER_INTEL17		0x00100070

// Visual C++ defines
#define GLM_COMPILER_VC				0x01000000
#define GLM_COMPILER_VC12			0x01000001
#define GLM_COMPILER_VC14			0x01000002
#define GLM_COMPILER_VC15			0x01000003
#define GLM_COMPILER_VC15_3			0x01000004
#define GLM_COMPILER_VC15_5			0x01000005
#define GLM_COMPILER_VC15_6			0x01000006
#define GLM_COMPILER_VC15_7			0x01000007
#define GLM_COMPILER_VC15_8			0x01000008
#define GLM_COMPILER_VC15_9			0x01000009
#define GLM_COMPILER_VC16			0x0100000A

// GCC defines
#define GLM_COMPILER_GCC			0x02000000
#define GLM_COMPILER_GCC46			0x020000D0
#define GLM_COMPILER_GCC47			0x020000E0
#define GLM_COMPILER_GCC48			0x020000F0
#define GLM_COMPILER_GCC49			0x02000100
#define GLM_COMPILER_GCC5			0x02000200
#define GLM_COMPILER_GCC6			0x02000300
#define GLM_COMPILER_GCC7			0x02000400
#define GLM_COMPILER_GCC8			0x02000500

// CUDA
#define GLM_COMPILER_CUDA			0x10000000
#define GLM_COMPILER_CUDA75			0x10000001
#define GLM_COMPILER_CUDA80			0x10000002
#define GLM_COMPILER_CUDA90			0x10000004
#define GLM_COMPILER_CUDA_RTC			0x10000100

// SYCL
#define GLM_COMPILER_SYCL			0x00300000

// Clang
#define GLM_COMPILER_CLANG			0x20000000
#define GLM_COMPILER_CLANG34		0x20000050
#define GLM_COMPILER_CLANG35		0x20000060
#define GLM_COMPILER_CLANG36		0x20000070
#define GLM_COMPILER_CLANG37		0x20000080
#define GLM_COMPILER_CLANG38		0x20000090
#define GLM_COMPILER_CLANG39		0x200000A0
#define GLM_COMPILER_CLANG40		0x200000B0
#define GLM_COMPILER_CLANG41		0x200000C0
#define GLM_COMPILER_CLANG42		0x200000D0

// HIP
#define GLM_COMPILER_HIP			0x40000000

// Build model
#define GLM_MODEL_32				0x00000010
#define GLM_MODEL_64				0x00000020

// Force generic C++ compiler
#ifdef GLM_FORCE_COMPILER_UNKNOWN
#	define GLM_COMPILER GLM_COMPILER_UNKNOWN

#elif defined(__INTEL_COMPILER)
#	if __INTEL_COMPILER >= 1700
#		define GLM_COMPILER GLM_COMPILER_INTEL17
#	elif __INTEL_COMPILER >= 1600
#		define GLM_COMPILER GLM_COMPILER_INTEL16
#	elif __INTEL_COMPILER >= 1500
#		define GLM_COMPILER GLM_COMPILER_INTEL15
#	elif __INTEL_COMPILER >= 1400
#		define GLM_COMPILER GLM_COMPILER_INTEL14
#	elif __INTEL_COMPILER < 1400
#		error "GLM requires ICC 2013 SP1 or newer"
#	endif

// CUDA
#elif defined(__CUDACC__)
#	if !defined(CUDA_VERSION) && !defined(GLM_FORCE_CUDA)
#		include <cuda.h>  // make sure version is defined since nvcc does not define it itself!
#	endif
#	if defined(__CUDACC_RTC__)
#		define GLM_COMPILER GLM_COMPILER_CUDA_RTC
#	elif CUDA_VERSION >= 8000
#		define GLM_COMPILER GLM_COMPILER_CUDA80
#	elif CUDA_VERSION >= 7500
#		define GLM_COMPILER GLM_COMPILER_CUDA75
#	elif CUDA_VERSION >= 7000
#		define GLM_COMPILER GLM_COMPILER_CUDA70
#	elif CUDA_VERSION < 7000
#		error "GLM requires CUDA 7.0 or higher"
#	endif

// HIP
#elif defined(__HIP__)
#	define GLM_COMPILER GLM_COMPILER_HIP

// SYCL
#elif defined(__SYCL_DEVICE_ONLY__)
#	define GLM_COMPILER GLM_COMPILER_SYCL

// Clang
#elif defined(__clang__)
#	if defined(__apple_build_version__)
#		if (__clang_major__ < 6)
#			error "GLM requires Clang 3.4 / Apple Clang 6.0 or higher"
#		elif __clang_major__ == 6 && __clang_minor__ == 0
#			define GLM_COMPILER GLM_COMPILER_CLANG35
#		elif __clang_major__ == 6 && __clang_minor__ >= 1
#			define GLM_COMPILER GLM_COMPILER_CLANG36
#		elif __clang_major__ >= 7
#			define GLM_COMPILER GLM_COMPILER_CLANG37
#		endif
#	else
#		if ((__clang_major__ == 3) && (__clang_minor__ < 4)) || (__clang_major__ < 3)
#			error "GLM requires Clang 3.4 or higher"
#		elif __clang_major__ == 3 && __clang_minor__ == 4
#			define GLM_COMPILER GLM_COMPILER_CLANG34
#		elif __clang_major__ == 3 && __clang_minor__ == 5
#			define GLM_COMPILER GLM_COMPILER_CLANG35
#		elif __clang_major__ == 3 && __clang_minor__ == 6
#			define GLM_COMPILER GLM_COMPILER_CLANG36
#		elif __clang_major__ == 3 && __clang_minor__ == 7
#			define GLM_COMPILER GLM_COMPILER_CLANG37
#		elif __clang_major__ == 3 && __clang_minor__ == 8
#			define GLM_COMPILER GLM_COMPILER_CLANG38
#		elif __clang_major__ == 3 && __clang_minor__ >= 9
#			define GLM_COMPILER GLM_COMPILER_CLANG39
#		elif __clang_major__ == 4 && __clang_minor__ == 0
#			define GLM_COMPILER GLM_COMPILER_CLANG40
#		elif __clang_major__ == 4 && __clang_minor__ == 1
#			define GLM_COMPILER GLM_COMPILER_CLANG41
#		elif __clang_major__ == 4 && __clang_minor__ >= 2
#			define GLM_COMPILER GLM_COMPILER_CLANG42
#		elif __clang_major__ >= 4
#			define GLM_COMPILER GLM_COMPILER_CLANG42
#		endif
#	endif

// Visual C++
#elif defined(_MSC_VER)
#	if _MSC_VER >= 1920
#		define GLM_COMPILER GLM_COMPILER_VC16
#	elif _MSC_VER >= 1916
#		define GLM_COMPILER GLM_COMPILER_VC15_9
#	elif _MSC_VER >= 1915
#		define GLM_COMPILER GLM_COMPILER_VC15_8
#	elif _MSC_VER >= 1914
#		define GLM_COMPILER GLM_COMPILER_VC15_7
#	elif _MSC_VER >= 1913
#		define GLM_COMPILER GLM_COMPILER_VC15_6
#	elif _MSC_VER >= 1912
#		define GLM_COMPILER GLM_COMPILER_VC15_5
#	elif _MSC_VER >= 1911
#		define GLM_COMPILER GLM_COMPILER_VC15_3
#	elif _MSC_VER >= 1910
#		define GLM_COMPILER GLM_COMPILER_VC15
#	elif _MSC_VER >= 1900
#		define GLM_COMPILER GLM_COMPILER_VC14
#	elif _MSC_VER >= 1800
#		define GLM_COMPILER GLM_COMPILER_VC12
#	elif _MSC_VER < 1800
#		error "GLM requires Visual C++ 12 - 2013 or higher"
#	endif//_MSC_VER

// G++
#elif defined(__GNUC__) || defined(__MINGW32__)
#	if __GNUC__ >= 8
#		define GLM_COMPILER GLM_COMPILER_GCC8
#	elif __GNUC__ >= 7
#		define GLM_COMPILER GLM_COMPILER_GCC7
#	elif __GNUC__ >= 6
#		define GLM_COMPILER GLM_COMPILER_GCC6
#	elif __GNUC__ >= 5
#		define GLM_COMPILER GLM_COMPILER_GCC5
#	elif __GNUC__ == 4 && __GNUC_MINOR__ >= 9
#		define GLM_COMPILER GLM_COMPILER_GCC49
#	elif __GNUC__ == 4 && __GNUC_MINOR__ >= 8
#		define GLM_COMPILER GLM_COMPILER_GCC48
#	elif __GNUC__ == 4 && __GNUC_MINOR__ >= 7
#		define GLM_COMPILER GLM_COMPILER_GCC47
#	elif __GNUC__ == 4 && __GNUC_MINOR__ >= 6
#		define GLM_COMPILER GLM_COMPILER_GCC46
#	elif ((__GNUC__ == 4) && (__GNUC_MINOR__ < 6)) || (__GNUC__ < 4)
#		error "GLM requires GCC 4.6 or higher"
#	endif

#else
#	define GLM_COMPILER GLM_COMPILER_UNKNOWN
#endif

#ifndef GLM_COMPILER
#	error "GLM_COMPILER undefined, your compiler may not be supported by GLM. Add #define GLM_COMPILER 0 to ignore this message."
#endif//GLM_COMPILER

///////////////////////////////////////////////////////////////////////////////////
// Instruction sets

// User defines: GLM_FORCE_PURE GLM_FORCE_INTRINSICS GLM_FORCE_SSE2 GLM_FORCE_SSE3 GLM_FORCE_AVX GLM_FORCE_AVX2 GLM_FORCE_AVX2

#define GLM_ARCH_MIPS_BIT	  (0x10000000)
#define GLM_ARCH_PPC_BIT	  (0x20000000)
#define GLM_ARCH_ARM_BIT	  (0x40000000)
#define GLM_ARCH_ARMV8_BIT  (0x01000000)
#define GLM_ARCH_X86_BIT	  (0x80000000)

#define GLM_ARCH_SIMD_BIT	(0x00001000)

#define GLM_ARCH_NEON_BIT	(0x00000001)
#define GLM_ARCH_SSE_BIT	(0x00000002)
#define GLM_ARCH_SSE2_BIT	(0x00000004)
#define GLM_ARCH_SSE3_BIT	(0x00000008)
#define GLM_ARCH_SSSE3_BIT	(0x00000010)
#define GLM_ARCH_SSE41_BIT	(0x00000020)
#define GLM_ARCH_SSE42_BIT	(0x00000040)
#define GLM_ARCH_AVX_BIT	(0x00000080)
#define GLM_ARCH_AVX2_BIT	(0x00000100)

#define GLM_ARCH_UNKNOWN	(0)
#define GLM_ARCH_X86		(GLM_ARCH_X86_BIT)
#define GLM_ARCH_SSE		(GLM_ARCH_SSE_BIT | GLM_ARCH_SIMD_BIT | GLM_ARCH_X86)
#define GLM_ARCH_SSE2		(GLM_ARCH_SSE2_BIT | GLM_ARCH_SSE)
#define GLM_ARCH_SSE3		(GLM_ARCH_SSE3_BIT | GLM_ARCH_SSE2)
#define GLM_ARCH_SSSE3		(GLM_ARCH_SSSE3_BIT | GLM_ARCH_SSE3)
#define GLM_ARCH_SSE41		(GLM_ARCH_SSE41_BIT | GLM_ARCH_SSSE3)
#define GLM_ARCH_SSE42		(GLM_ARCH_SSE42_BIT | GLM_ARCH_SSE41)
#define GLM_ARCH_AVX		(GLM_ARCH_AVX_BIT | GLM_ARCH_SSE42)
#define GLM_ARCH_AVX2		(GLM_ARCH_AVX2_BIT | GLM_ARCH_AVX)
#define GLM_ARCH_ARM		(GLM_ARCH_ARM_BIT)
#define GLM_ARCH_ARMV8		(GLM_ARCH_NEON_BIT | GLM_ARCH_SIMD_BIT | GLM_ARCH_ARM | GLM_ARCH_ARMV8_BIT)
#define GLM_ARCH_NEON		(GLM_ARCH_NEON_BIT | GLM_ARCH_SIMD_BIT | GLM_ARCH_ARM)
#define GLM_ARCH_MIPS		(GLM_ARCH_MIPS_BIT)
#define GLM_ARCH_PPC		(GLM_ARCH_PPC_BIT)

#if defined(GLM_FORCE_ARCH_UNKNOWN) || defined(GLM_FORCE_PURE)
#	define GLM_ARCH GLM_ARCH_UNKNOWN
#elif defined(GLM_FORCE_NEON)
#	if __ARM_ARCH >= 8
#		define GLM_ARCH (GLM_ARCH_ARMV8)
#	else
#		define GLM_ARCH (GLM_ARCH_NEON)
#	endif
#	define GLM_FORCE_INTRINSICS
#elif defined(GLM_FORCE_AVX2)
#	define GLM_ARCH (GLM_ARCH_AVX2)
#	define GLM_FORCE_INTRINSICS
#elif defined(GLM_FORCE_AVX)
#	define GLM_ARCH (GLM_ARCH_AVX)
#	define GLM_FORCE_INTRINSICS
#elif defined(GLM_FORCE_SSE42)
#	define GLM_ARCH (GLM_ARCH_SSE42)
#	define GLM_FORCE_INTRINSICS
#elif defined(GLM_FORCE_SSE41)
#	define GLM_ARCH (GLM_ARCH_SSE41)
#	define GLM_FORCE_INTRINSICS
#elif defined(GLM_FORCE_SSSE3)
#	define GLM_ARCH (GLM_ARCH_SSSE3)
#	define GLM_FORCE_INTRINSICS
#elif defined(GLM_FORCE_SSE3)
#	define GLM_ARCH (GLM_ARCH_SSE3)
#	define GLM_FORCE_INTRINSICS
#elif defined(GLM_FORCE_SSE2)
#	define GLM_ARCH (GLM_ARCH_SSE2)
#	define GLM_FORCE_INTRINSICS
#elif defined(GLM_FORCE_SSE)
#	define GLM_ARCH (GLM_ARCH_SSE)
#	define GLM_FORCE_INTRINSICS
#elif defined(GLM_FORCE_INTRINSICS) && !defined(GLM_FORCE_XYZW_ONLY)
#	if defined(__AVX2__)
#		define GLM_ARCH (GLM_ARCH_AVX2)
#	elif defined(__AVX__)
#		define GLM_ARCH (GLM_ARCH_AVX)
#	elif defined(__SSE4_2__)
#		define GLM_ARCH (GLM_ARCH_SSE42)
#	elif defined(__SSE4_1__)
#		define GLM_ARCH (GLM_ARCH_SSE41)
#	elif defined(__SSSE3__)
#		define GLM_ARCH (GLM_ARCH_SSSE3)
#	elif defined(__SSE3__)
#		define GLM_ARCH (GLM_ARCH_SSE3)
#	elif defined(__SSE2__) || defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86_FP)
#		define GLM_ARCH (GLM_ARCH_SSE2)
#	elif defined(__i386__)
#		define GLM_ARCH (GLM_ARCH_X86)
#	elif defined(__ARM_ARCH) && (__ARM_ARCH >= 8)
#		define GLM_ARCH (GLM_ARCH_ARMV8)
#	elif defined(__ARM_NEON)
#		define GLM_ARCH (GLM_ARCH_ARM | GLM_ARCH_NEON)
#	elif defined(__arm__ ) || defined(_M_ARM)
#		define GLM_ARCH (GLM_ARCH_ARM)
#	elif defined(__mips__ )
#		define GLM_ARCH (GLM_ARCH_MIPS)
#	elif defined(__powerpc__ ) || defined(_M_PPC)
#		define GLM_ARCH (GLM_ARCH_PPC)
#	else
#		define GLM_ARCH (GLM_ARCH_UNKNOWN)
#	endif
#else
#	if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86) || defined(__i386__)
#		define GLM_ARCH (GLM_ARCH_X86)
#	elif defined(__arm__) || defined(_M_ARM)
#		define GLM_ARCH (GLM_ARCH_ARM)
#	elif defined(__powerpc__) || defined(_M_PPC)
#		define GLM_ARCH (GLM_ARCH_PPC)
#	elif defined(__mips__)
#		define GLM_ARCH (GLM_ARCH_MIPS)
#	else
#		define GLM_ARCH (GLM_ARCH_UNKNOWN)
#	endif
#endif

#if GLM_ARCH & GLM_ARCH_AVX2_BIT
#	include <immintrin.h>
#elif GLM_ARCH & GLM_ARCH_AVX_BIT
#	include <immintrin.h>
#elif GLM_ARCH & GLM_ARCH_SSE42_BIT
#	if GLM_COMPILER & GLM_COMPILER_CLANG
#		include <popcntintrin.h>
#	endif
#	include <nmmintrin.h>
#elif GLM_ARCH & GLM_ARCH_SSE41_BIT
#	include <smmintrin.h>
#elif GLM_ARCH & GLM_ARCH_SSSE3_BIT
#	include <tmmintrin.h>
#elif GLM_ARCH & GLM_ARCH_SSE3_BIT
#	include <pmmintrin.h>
#elif GLM_ARCH & GLM_ARCH_SSE2_BIT
#	include <emmintrin.h>
#elif GLM_ARCH & GLM_ARCH_NEON_BIT
#	include "neon.h"
#endif//GLM_ARCH

#if GLM_ARCH & GLM_ARCH_SSE2_BIT
	typedef __m128			glm_f32vec4;
	typedef __m128i			glm_i32vec4;
	typedef __m128i			glm_u32vec4;
	typedef __m128d			glm_f64vec2;
	typedef __m128i			glm_i64vec2;
	typedef __m128i			glm_u64vec2;

	typedef glm_f32vec4		glm_vec4;
	typedef glm_i32vec4		glm_ivec4;
	typedef glm_u32vec4		glm_uvec4;
	typedef glm_f64vec2		glm_dvec2;
#endif

#if GLM_ARCH & GLM_ARCH_AVX_BIT
	typedef __m256d			glm_f64vec4;
	typedef glm_f64vec4		glm_dvec4;
#endif

#if GLM_ARCH & GLM_ARCH_AVX2_BIT
	typedef __m256i			glm_i64vec4;
	typedef __m256i			glm_u64vec4;
#endif

#if GLM_ARCH & GLM_ARCH_NEON_BIT
	typedef float32x4_t			glm_f32vec4;
	typedef int32x4_t			glm_i32vec4;
	typedef uint32x4_t			glm_u32vec4;
#endif
