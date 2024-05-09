#ifndef GLM_SETUP_INCLUDED

#include <cassert>
#include <cstddef>

#define GLM_VERSION_MAJOR 0
#define GLM_VERSION_MINOR 9
#define GLM_VERSION_PATCH 9
#define GLM_VERSION_REVISION 9
#define GLM_VERSION 999
#define GLM_VERSION_MESSAGE "GLM: version 0.9.9.9"

#define GLM_SETUP_INCLUDED GLM_VERSION

///////////////////////////////////////////////////////////////////////////////////
// Active states

#define GLM_DISABLE		0
#define GLM_ENABLE		1

///////////////////////////////////////////////////////////////////////////////////
// Messages

#if defined(GLM_FORCE_MESSAGES)
#	define GLM_MESSAGES GLM_ENABLE
#else
#	define GLM_MESSAGES GLM_DISABLE
#endif

///////////////////////////////////////////////////////////////////////////////////
// Detect the platform

#include "../simd/platform.h"

///////////////////////////////////////////////////////////////////////////////////
// Build model

#if defined(_M_ARM64) || defined(__LP64__) || defined(_M_X64) || defined(__ppc64__) || defined(__x86_64__)
#	define GLM_MODEL	GLM_MODEL_64
#elif defined(__i386__) || defined(__ppc__) || defined(__ILP32__) || defined(_M_ARM)
#	define GLM_MODEL	GLM_MODEL_32
#else
#	define GLM_MODEL	GLM_MODEL_32
#endif//

#if !defined(GLM_MODEL) && GLM_COMPILER != 0
#	error "GLM_MODEL undefined, your compiler may not be supported by GLM. Add #define GLM_MODEL 0 to ignore this message."
#endif//GLM_MODEL

///////////////////////////////////////////////////////////////////////////////////
// C++ Version

// User defines: GLM_FORCE_CXX98, GLM_FORCE_CXX03, GLM_FORCE_CXX11, GLM_FORCE_CXX14, GLM_FORCE_CXX17, GLM_FORCE_CXX2A

#define GLM_LANG_CXX98_FLAG			(1 << 1)
#define GLM_LANG_CXX03_FLAG			(1 << 2)
#define GLM_LANG_CXX0X_FLAG			(1 << 3)
#define GLM_LANG_CXX11_FLAG			(1 << 4)
#define GLM_LANG_CXX14_FLAG			(1 << 5)
#define GLM_LANG_CXX17_FLAG			(1 << 6)
#define GLM_LANG_CXX20_FLAG			(1 << 7)
#define GLM_LANG_CXXMS_FLAG			(1 << 8)
#define GLM_LANG_CXXGNU_FLAG		(1 << 9)

#define GLM_LANG_CXX98			GLM_LANG_CXX98_FLAG
#define GLM_LANG_CXX03			(GLM_LANG_CXX98 | GLM_LANG_CXX03_FLAG)
#define GLM_LANG_CXX0X			(GLM_LANG_CXX03 | GLM_LANG_CXX0X_FLAG)
#define GLM_LANG_CXX11			(GLM_LANG_CXX0X | GLM_LANG_CXX11_FLAG)
#define GLM_LANG_CXX14			(GLM_LANG_CXX11 | GLM_LANG_CXX14_FLAG)
#define GLM_LANG_CXX17			(GLM_LANG_CXX14 | GLM_LANG_CXX17_FLAG)
#define GLM_LANG_CXX20			(GLM_LANG_CXX17 | GLM_LANG_CXX20_FLAG)
#define GLM_LANG_CXXMS			GLM_LANG_CXXMS_FLAG
#define GLM_LANG_CXXGNU			GLM_LANG_CXXGNU_FLAG

#if (defined(_MSC_EXTENSIONS))
#	define GLM_LANG_EXT GLM_LANG_CXXMS_FLAG
#elif ((GLM_COMPILER & (GLM_COMPILER_CLANG | GLM_COMPILER_GCC)) && (GLM_ARCH & GLM_ARCH_SIMD_BIT))
#	define GLM_LANG_EXT GLM_LANG_CXXMS_FLAG
#else
#	define GLM_LANG_EXT 0
#endif

#if (defined(GLM_FORCE_CXX_UNKNOWN))
#	define GLM_LANG 0
#elif defined(GLM_FORCE_CXX20)
#	define GLM_LANG (GLM_LANG_CXX20 | GLM_LANG_EXT)
#	define GLM_LANG_STL11_FORCED
#elif defined(GLM_FORCE_CXX17)
#	define GLM_LANG (GLM_LANG_CXX17 | GLM_LANG_EXT)
#	define GLM_LANG_STL11_FORCED
#elif defined(GLM_FORCE_CXX14)
#	define GLM_LANG (GLM_LANG_CXX14 | GLM_LANG_EXT)
#	define GLM_LANG_STL11_FORCED
#elif defined(GLM_FORCE_CXX11)
#	define GLM_LANG (GLM_LANG_CXX11 | GLM_LANG_EXT)
#	define GLM_LANG_STL11_FORCED
#elif defined(GLM_FORCE_CXX03)
#	define GLM_LANG (GLM_LANG_CXX03 | GLM_LANG_EXT)
#elif defined(GLM_FORCE_CXX98)
#	define GLM_LANG (GLM_LANG_CXX98 | GLM_LANG_EXT)
#else
#	if GLM_COMPILER & GLM_COMPILER_VC && defined(_MSVC_LANG)
#		if GLM_COMPILER >= GLM_COMPILER_VC15_7
#			define GLM_LANG_PLATFORM _MSVC_LANG
#		elif GLM_COMPILER >= GLM_COMPILER_VC15
#			if _MSVC_LANG > 201402L
#				define GLM_LANG_PLATFORM 201402L
#			else
#				define GLM_LANG_PLATFORM _MSVC_LANG
#			endif
#		else
#			define GLM_LANG_PLATFORM 0
#		endif
#	else
#		define GLM_LANG_PLATFORM 0
#	endif

#	if __cplusplus > 201703L || GLM_LANG_PLATFORM > 201703L
#		define GLM_LANG (GLM_LANG_CXX20 | GLM_LANG_EXT)
#	elif __cplusplus == 201703L || GLM_LANG_PLATFORM == 201703L
#		define GLM_LANG (GLM_LANG_CXX17 | GLM_LANG_EXT)
#	elif __cplusplus == 201402L || __cplusplus == 201406L || __cplusplus == 201500L || GLM_LANG_PLATFORM == 201402L
#		define GLM_LANG (GLM_LANG_CXX14 | GLM_LANG_EXT)
#	elif __cplusplus == 201103L || GLM_LANG_PLATFORM == 201103L
#		define GLM_LANG (GLM_LANG_CXX11 | GLM_LANG_EXT)
#	elif defined(__INTEL_CXX11_MODE__) || defined(_MSC_VER) || defined(__GXX_EXPERIMENTAL_CXX0X__)
#		define GLM_LANG (GLM_LANG_CXX0X | GLM_LANG_EXT)
#	elif __cplusplus == 199711L
#		define GLM_LANG (GLM_LANG_CXX98 | GLM_LANG_EXT)
#	else
#		define GLM_LANG (0 | GLM_LANG_EXT)
#	endif
#endif

///////////////////////////////////////////////////////////////////////////////////
// Has of C++ features

// http://clang.llvm.org/cxx_status.html
// http://gcc.gnu.org/projects/cxx0x.html
// http://msdn.microsoft.com/en-us/library/vstudio/hh567368(v=vs.120).aspx

// Android has multiple STLs but C++11 STL detection doesn't always work #284 #564
#if GLM_PLATFORM == GLM_PLATFORM_ANDROID && !defined(GLM_LANG_STL11_FORCED)
#	define GLM_HAS_CXX11_STL 0
#elif (GLM_COMPILER & GLM_COMPILER_CUDA_RTC) == GLM_COMPILER_CUDA_RTC
#	define GLM_HAS_CXX11_STL 0
#elif (GLM_COMPILER & GLM_COMPILER_HIP)
#	define GLM_HAS_CXX11_STL 0
#elif GLM_COMPILER & GLM_COMPILER_CLANG
#	if (defined(_LIBCPP_VERSION) || (GLM_LANG & GLM_LANG_CXX11_FLAG) || defined(GLM_LANG_STL11_FORCED))
#		define GLM_HAS_CXX11_STL 1
#	else
#		define GLM_HAS_CXX11_STL 0
#	endif
#elif GLM_LANG & GLM_LANG_CXX11_FLAG
#	define GLM_HAS_CXX11_STL 1
#else
#	define GLM_HAS_CXX11_STL ((GLM_LANG & GLM_LANG_CXX0X_FLAG) && (\
		((GLM_COMPILER & GLM_COMPILER_GCC) && (GLM_COMPILER >= GLM_COMPILER_GCC48)) || \
		((GLM_COMPILER & GLM_COMPILER_VC) && (GLM_COMPILER >= GLM_COMPILER_VC12)) || \
		((GLM_PLATFORM != GLM_PLATFORM_WINDOWS) && (GLM_COMPILER & GLM_COMPILER_INTEL) && (GLM_COMPILER >= GLM_COMPILER_INTEL15))))
#endif

// N1720
#if GLM_COMPILER & GLM_COMPILER_CLANG
#	define GLM_HAS_STATIC_ASSERT __has_feature(cxx_static_assert)
#elif GLM_LANG & GLM_LANG_CXX11_FLAG
#	define GLM_HAS_STATIC_ASSERT 1
#else
#	define GLM_HAS_STATIC_ASSERT ((GLM_LANG & GLM_LANG_CXX0X_FLAG) && (\
		((GLM_COMPILER & GLM_COMPILER_CUDA)) || \
		((GLM_COMPILER & GLM_COMPILER_VC)) || \
		((GLM_COMPILER & GLM_COMPILER_HIP))))
#endif

// N1988
#if GLM_LANG & GLM_LANG_CXX11_FLAG
#	define GLM_HAS_EXTENDED_INTEGER_TYPE 1
#else
#	define GLM_HAS_EXTENDED_INTEGER_TYPE (\
		((GLM_LANG & GLM_LANG_CXX0X_FLAG) && (GLM_COMPILER & GLM_COMPILER_VC)) || \
		((GLM_LANG & GLM_LANG_CXX0X_FLAG) && (GLM_COMPILER & GLM_COMPILER_CUDA)) || \
		((GLM_LANG & GLM_LANG_CXX0X_FLAG) && (GLM_COMPILER & GLM_COMPILER_CLANG)) || \
		((GLM_COMPILER & GLM_COMPILER_HIP)))
#endif

// N2672 Initializer lists http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2672.htm
#if GLM_COMPILER & GLM_COMPILER_CLANG
#	define GLM_HAS_INITIALIZER_LISTS __has_feature(cxx_generalized_initializers)
#elif GLM_LANG & GLM_LANG_CXX11_FLAG
#	define GLM_HAS_INITIALIZER_LISTS 1
#else
#	define GLM_HAS_INITIALIZER_LISTS ((GLM_LANG & GLM_LANG_CXX0X_FLAG) && (\
		((GLM_COMPILER & GLM_COMPILER_VC) && (GLM_COMPILER >= GLM_COMPILER_VC15)) || \
		((GLM_COMPILER & GLM_COMPILER_INTEL) && (GLM_COMPILER >= GLM_COMPILER_INTEL14)) || \
		((GLM_COMPILER & GLM_COMPILER_CUDA)) || \
		((GLM_COMPILER & GLM_COMPILER_HIP))))
#endif

// N2544 Unrestricted unions http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2544.pdf
#if GLM_COMPILER & GLM_COMPILER_CLANG
#	define GLM_HAS_UNRESTRICTED_UNIONS __has_feature(cxx_unrestricted_unions)
#elif GLM_LANG & GLM_LANG_CXX11_FLAG
#	define GLM_HAS_UNRESTRICTED_UNIONS 1
#else
#	define GLM_HAS_UNRESTRICTED_UNIONS (GLM_LANG & GLM_LANG_CXX0X_FLAG) && (\
		(GLM_COMPILER & GLM_COMPILER_VC) || \
		((GLM_COMPILER & GLM_COMPILER_CUDA)) || \
		((GLM_COMPILER & GLM_COMPILER_HIP)))
#endif

// N2346
#if GLM_COMPILER & GLM_COMPILER_CLANG
#	define GLM_HAS_DEFAULTED_FUNCTIONS __has_feature(cxx_defaulted_functions)
#elif GLM_LANG & GLM_LANG_CXX11_FLAG
#	define GLM_HAS_DEFAULTED_FUNCTIONS 1
#else
#	define GLM_HAS_DEFAULTED_FUNCTIONS ((GLM_LANG & GLM_LANG_CXX0X_FLAG) && (\
		((GLM_COMPILER & GLM_COMPILER_VC) && (GLM_COMPILER >= GLM_COMPILER_VC12)) || \
		((GLM_COMPILER & GLM_COMPILER_INTEL)) || \
		(GLM_COMPILER & GLM_COMPILER_CUDA)) || \
		((GLM_COMPILER & GLM_COMPILER_HIP)))
#endif

// N2118
#if GLM_COMPILER & GLM_COMPILER_CLANG
#	define GLM_HAS_RVALUE_REFERENCES __has_feature(cxx_rvalue_references)
#elif GLM_LANG & GLM_LANG_CXX11_FLAG
#	define GLM_HAS_RVALUE_REFERENCES 1
#else
#	define GLM_HAS_RVALUE_REFERENCES ((GLM_LANG & GLM_LANG_CXX0X_FLAG) && (\
		((GLM_COMPILER & GLM_COMPILER_VC)) || \
		((GLM_COMPILER & GLM_COMPILER_CUDA)) || \
		((GLM_COMPILER & GLM_COMPILER_HIP))))
#endif

// N2437 http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2437.pdf
#if GLM_COMPILER & GLM_COMPILER_CLANG
#	define GLM_HAS_EXPLICIT_CONVERSION_OPERATORS __has_feature(cxx_explicit_conversions)
#elif GLM_LANG & GLM_LANG_CXX11_FLAG
#	define GLM_HAS_EXPLICIT_CONVERSION_OPERATORS 1
#else
#	define GLM_HAS_EXPLICIT_CONVERSION_OPERATORS ((GLM_LANG & GLM_LANG_CXX0X_FLAG) && (\
		((GLM_COMPILER & GLM_COMPILER_INTEL) && (GLM_COMPILER >= GLM_COMPILER_INTEL14)) || \
		((GLM_COMPILER & GLM_COMPILER_VC) && (GLM_COMPILER >= GLM_COMPILER_VC12)) || \
		((GLM_COMPILER & GLM_COMPILER_CUDA)) || \
		((GLM_COMPILER & GLM_COMPILER_HIP))))
#endif

// N2258 http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2258.pdf
#if GLM_COMPILER & GLM_COMPILER_CLANG
#	define GLM_HAS_TEMPLATE_ALIASES __has_feature(cxx_alias_templates)
#elif GLM_LANG & GLM_LANG_CXX11_FLAG
#	define GLM_HAS_TEMPLATE_ALIASES 1
#else
#	define GLM_HAS_TEMPLATE_ALIASES ((GLM_LANG & GLM_LANG_CXX0X_FLAG) && (\
		((GLM_COMPILER & GLM_COMPILER_INTEL)) || \
		((GLM_COMPILER & GLM_COMPILER_VC) && (GLM_COMPILER >= GLM_COMPILER_VC12)) || \
		((GLM_COMPILER & GLM_COMPILER_CUDA)) || \
		((GLM_COMPILER & GLM_COMPILER_HIP))))
#endif

// N2930 http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2009/n2930.html
#if GLM_COMPILER & GLM_COMPILER_CLANG
#	define GLM_HAS_RANGE_FOR __has_feature(cxx_range_for)
#elif GLM_LANG & GLM_LANG_CXX11_FLAG
#	define GLM_HAS_RANGE_FOR 1
#else
#	define GLM_HAS_RANGE_FOR ((GLM_LANG & GLM_LANG_CXX0X_FLAG) && (\
		((GLM_COMPILER & GLM_COMPILER_INTEL)) || \
		((GLM_COMPILER & GLM_COMPILER_VC)) || \
		((GLM_COMPILER & GLM_COMPILER_CUDA)) || \
		((GLM_COMPILER & GLM_COMPILER_HIP))))
#endif

// N2341 http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2341.pdf
#if GLM_COMPILER & GLM_COMPILER_CLANG
#	define GLM_HAS_ALIGNOF __has_feature(cxx_alignas)
#elif GLM_LANG & GLM_LANG_CXX11_FLAG
#	define GLM_HAS_ALIGNOF 1
#else
#	define GLM_HAS_ALIGNOF ((GLM_LANG & GLM_LANG_CXX0X_FLAG) && (\
		((GLM_COMPILER & GLM_COMPILER_INTEL) && (GLM_COMPILER >= GLM_COMPILER_INTEL15)) || \
		((GLM_COMPILER & GLM_COMPILER_VC) && (GLM_COMPILER >= GLM_COMPILER_VC14)) || \
		((GLM_COMPILER & GLM_COMPILER_CUDA)) || \
		((GLM_COMPILER & GLM_COMPILER_HIP))))
#endif

// N2235 Generalized Constant Expressions http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2235.pdf
// N3652 Extended Constant Expressions http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3652.html
#if (GLM_ARCH & GLM_ARCH_SIMD_BIT) // Compiler SIMD intrinsics don't support constexpr...
#	define GLM_HAS_CONSTEXPR 0
#elif (GLM_COMPILER & GLM_COMPILER_CLANG)
#	define GLM_HAS_CONSTEXPR __has_feature(cxx_relaxed_constexpr)
#elif (GLM_LANG & GLM_LANG_CXX14_FLAG)
#	define GLM_HAS_CONSTEXPR 1
#else
#	define GLM_HAS_CONSTEXPR ((GLM_LANG & GLM_LANG_CXX0X_FLAG) && GLM_HAS_INITIALIZER_LISTS && (\
		((GLM_COMPILER & GLM_COMPILER_INTEL) && (GLM_COMPILER >= GLM_COMPILER_INTEL17)) || \
		((GLM_COMPILER & GLM_COMPILER_VC) && (GLM_COMPILER >= GLM_COMPILER_VC15))))
#endif

#if GLM_HAS_CONSTEXPR
#	define GLM_CONSTEXPR constexpr
#else
#	define GLM_CONSTEXPR
#endif

//
#if GLM_HAS_CONSTEXPR
# if (GLM_COMPILER & GLM_COMPILER_CLANG)
#	if __has_feature(cxx_if_constexpr)
#		define GLM_HAS_IF_CONSTEXPR 1
#	else
# 		define GLM_HAS_IF_CONSTEXPR 0
#	endif
# elif (GLM_LANG & GLM_LANG_CXX17_FLAG)
# 	define GLM_HAS_IF_CONSTEXPR 1
# else
# 	define GLM_HAS_IF_CONSTEXPR 0
# endif
#else
#	define GLM_HAS_IF_CONSTEXPR 0
#endif

#if GLM_HAS_IF_CONSTEXPR
# 	define GLM_IF_CONSTEXPR if constexpr
#else
#	define GLM_IF_CONSTEXPR if
#endif

//
#if GLM_LANG & GLM_LANG_CXX11_FLAG
#	define GLM_HAS_ASSIGNABLE 1
#else
#	define GLM_HAS_ASSIGNABLE ((GLM_LANG & GLM_LANG_CXX0X_FLAG) && (\
		((GLM_COMPILER & GLM_COMPILER_VC) && (GLM_COMPILER >= GLM_COMPILER_VC15)) || \
		((GLM_COMPILER & GLM_COMPILER_GCC) && (GLM_COMPILER >= GLM_COMPILER_GCC49))))
#endif

//
#define GLM_HAS_TRIVIAL_QUERIES 0

//
#if GLM_LANG & GLM_LANG_CXX11_FLAG
#	define GLM_HAS_MAKE_SIGNED 1
#else
#	define GLM_HAS_MAKE_SIGNED ((GLM_LANG & GLM_LANG_CXX0X_FLAG) && (\
		((GLM_COMPILER & GLM_COMPILER_VC) && (GLM_COMPILER >= GLM_COMPILER_VC12)) || \
		((GLM_COMPILER & GLM_COMPILER_CUDA)) || \
		((GLM_COMPILER & GLM_COMPILER_HIP))))
#endif

//
#if defined(GLM_FORCE_INTRINSICS)
#	define GLM_HAS_BITSCAN_WINDOWS ((GLM_PLATFORM & GLM_PLATFORM_WINDOWS) && (\
		((GLM_COMPILER & GLM_COMPILER_INTEL)) || \
		((GLM_COMPILER & GLM_COMPILER_VC) && (GLM_COMPILER >= GLM_COMPILER_VC14) && (GLM_ARCH & GLM_ARCH_X86_BIT))))
#else
#	define GLM_HAS_BITSCAN_WINDOWS 0
#endif

#if GLM_LANG & GLM_LANG_CXX11_FLAG
#	define GLM_HAS_NOEXCEPT 1
#else
#	define GLM_HAS_NOEXCEPT 0
#endif

#if GLM_HAS_NOEXCEPT
#	define GLM_NOEXCEPT noexcept
#else
#	define GLM_NOEXCEPT
#endif

///////////////////////////////////////////////////////////////////////////////////
// OpenMP
#ifdef _OPENMP
#	if GLM_COMPILER & GLM_COMPILER_GCC
#		if GLM_COMPILER >= GLM_COMPILER_GCC61
#			define GLM_HAS_OPENMP 45
#		elif GLM_COMPILER >= GLM_COMPILER_GCC49
#			define GLM_HAS_OPENMP 40
#		elif GLM_COMPILER >= GLM_COMPILER_GCC47
#			define GLM_HAS_OPENMP 31
#		else
#			define GLM_HAS_OPENMP 0
#		endif
#	elif GLM_COMPILER & GLM_COMPILER_CLANG
#		if GLM_COMPILER >= GLM_COMPILER_CLANG38
#			define GLM_HAS_OPENMP 31
#		else
#			define GLM_HAS_OPENMP 0
#		endif
#	elif GLM_COMPILER & GLM_COMPILER_VC
#		define GLM_HAS_OPENMP 20
#	elif GLM_COMPILER & GLM_COMPILER_INTEL
#		if GLM_COMPILER >= GLM_COMPILER_INTEL16
#			define GLM_HAS_OPENMP 40
#		else
#			define GLM_HAS_OPENMP 0
#		endif
#	else
#		define GLM_HAS_OPENMP 0
#	endif
#else
#	define GLM_HAS_OPENMP 0
#endif

///////////////////////////////////////////////////////////////////////////////////
// nullptr

#if GLM_LANG & GLM_LANG_CXX0X_FLAG
#	define GLM_CONFIG_NULLPTR GLM_ENABLE
#else
#	define GLM_CONFIG_NULLPTR GLM_DISABLE
#endif

#if GLM_CONFIG_NULLPTR == GLM_ENABLE
#	define GLM_NULLPTR nullptr
#else
#	define GLM_NULLPTR 0
#endif

///////////////////////////////////////////////////////////////////////////////////
// Static assert

#if GLM_HAS_STATIC_ASSERT
#	define GLM_STATIC_ASSERT(x, message) static_assert(x, message)
#elif GLM_COMPILER & GLM_COMPILER_VC
#	define GLM_STATIC_ASSERT(x, message) typedef char __CASSERT__##__LINE__[(x) ? 1 : -1]
#else
#	define GLM_STATIC_ASSERT(x, message) assert(x)
#endif//GLM_LANG

///////////////////////////////////////////////////////////////////////////////////
// Qualifiers

// User defines: GLM_CUDA_FORCE_DEVICE_FUNC, GLM_CUDA_FORCE_HOST_FUNC

#if (GLM_COMPILER & GLM_COMPILER_CUDA) || (GLM_COMPILER & GLM_COMPILER_HIP)
#	if defined(GLM_CUDA_FORCE_DEVICE_FUNC) && defined(GLM_CUDA_FORCE_HOST_FUNC)
#		error "GLM error: GLM_CUDA_FORCE_DEVICE_FUNC and GLM_CUDA_FORCE_HOST_FUNC should not be defined at the same time, GLM by default generates both device and host code for CUDA compiler."
#	endif//defined(GLM_CUDA_FORCE_DEVICE_FUNC) && defined(GLM_CUDA_FORCE_HOST_FUNC)

#	if defined(GLM_CUDA_FORCE_DEVICE_FUNC)
#		define GLM_CUDA_FUNC_DEF __device__
#		define GLM_CUDA_FUNC_DECL __device__
#	elif defined(GLM_CUDA_FORCE_HOST_FUNC)
#		define GLM_CUDA_FUNC_DEF __host__
#		define GLM_CUDA_FUNC_DECL __host__
#	else
#		define GLM_CUDA_FUNC_DEF __device__ __host__
#		define GLM_CUDA_FUNC_DECL __device__ __host__
#	endif//defined(GLM_CUDA_FORCE_XXXX_FUNC)
#else
#	define GLM_CUDA_FUNC_DEF
#	define GLM_CUDA_FUNC_DECL
#endif

#if defined(GLM_FORCE_INLINE)
#	if GLM_COMPILER & GLM_COMPILER_VC
#		define GLM_INLINE __forceinline
#		define GLM_NEVER_INLINE __declspec(noinline)
#	elif GLM_COMPILER & (GLM_COMPILER_GCC | GLM_COMPILER_CLANG)
#		define GLM_INLINE inline __attribute__((__always_inline__))
#		define GLM_NEVER_INLINE __attribute__((__noinline__))
#	elif (GLM_COMPILER & GLM_COMPILER_CUDA) || (GLM_COMPILER & GLM_COMPILER_HIP)
#		define GLM_INLINE __forceinline__
#		define GLM_NEVER_INLINE __noinline__
#	else
#		define GLM_INLINE inline
#		define GLM_NEVER_INLINE
#	endif//GLM_COMPILER
#else
#	define GLM_INLINE inline
#	define GLM_NEVER_INLINE
#endif//defined(GLM_FORCE_INLINE)

#define GLM_FUNC_DECL GLM_CUDA_FUNC_DECL
#define GLM_FUNC_QUALIFIER GLM_CUDA_FUNC_DEF GLM_INLINE

// Do not use CUDA function qualifiers on CUDA compiler when functions are made default
#if GLM_HAS_DEFAULTED_FUNCTIONS
#	define GLM_DEFAULTED_FUNC_DECL
#	define GLM_DEFAULTED_FUNC_QUALIFIER GLM_INLINE
#else
#	define GLM_DEFAULTED_FUNC_DECL GLM_FUNC_DECL
#	define GLM_DEFAULTED_FUNC_QUALIFIER GLM_FUNC_QUALIFIER
#endif//GLM_HAS_DEFAULTED_FUNCTIONS
#if !defined(GLM_FORCE_CTOR_INIT)
#	define GLM_DEFAULTED_DEFAULT_CTOR_DECL GLM_DEFAULTED_FUNC_DECL
#	define GLM_DEFAULTED_DEFAULT_CTOR_QUALIFIER GLM_DEFAULTED_FUNC_QUALIFIER
#else
#	define GLM_DEFAULTED_DEFAULT_CTOR_DECL GLM_FUNC_DECL
#	define GLM_DEFAULTED_DEFAULT_CTOR_QUALIFIER GLM_FUNC_QUALIFIER
#endif//GLM_FORCE_CTOR_INIT

///////////////////////////////////////////////////////////////////////////////////
// Swizzle operators

// User defines: GLM_FORCE_SWIZZLE

#define GLM_SWIZZLE_DISABLED		0
#define GLM_SWIZZLE_OPERATOR		1
#define GLM_SWIZZLE_FUNCTION		2

#if defined(GLM_SWIZZLE)
#	pragma message("GLM: GLM_SWIZZLE is deprecated, use GLM_FORCE_SWIZZLE instead.")
#	define GLM_FORCE_SWIZZLE
#endif

#if defined(GLM_FORCE_SWIZZLE) && (GLM_LANG & GLM_LANG_CXXMS_FLAG) && !defined(GLM_FORCE_XYZW_ONLY)
#	define GLM_CONFIG_SWIZZLE GLM_SWIZZLE_OPERATOR
#elif defined(GLM_FORCE_SWIZZLE)
#	define GLM_CONFIG_SWIZZLE GLM_SWIZZLE_FUNCTION
#else
#	define GLM_CONFIG_SWIZZLE GLM_SWIZZLE_DISABLED
#endif

///////////////////////////////////////////////////////////////////////////////////
// Allows using not basic types as genType

// #define GLM_FORCE_UNRESTRICTED_GENTYPE

#ifdef GLM_FORCE_UNRESTRICTED_GENTYPE
#	define GLM_CONFIG_UNRESTRICTED_GENTYPE GLM_ENABLE
#else
#	define GLM_CONFIG_UNRESTRICTED_GENTYPE GLM_DISABLE
#endif

///////////////////////////////////////////////////////////////////////////////////
// Allows using any scaler as float

// #define GLM_FORCE_UNRESTRICTED_FLOAT

#ifdef GLM_FORCE_UNRESTRICTED_FLOAT
#	define GLM_CONFIG_UNRESTRICTED_FLOAT GLM_ENABLE
#else
#	define GLM_CONFIG_UNRESTRICTED_FLOAT GLM_DISABLE
#endif

///////////////////////////////////////////////////////////////////////////////////
// Clip control, define GLM_FORCE_DEPTH_ZERO_TO_ONE before including GLM
// to use a clip space between 0 to 1.
// Coordinate system, define GLM_FORCE_LEFT_HANDED before including GLM
// to use left handed coordinate system by default.

#define GLM_CLIP_CONTROL_ZO_BIT		(1 << 0) // ZERO_TO_ONE
#define GLM_CLIP_CONTROL_NO_BIT		(1 << 1) // NEGATIVE_ONE_TO_ONE
#define GLM_CLIP_CONTROL_LH_BIT		(1 << 2) // LEFT_HANDED, For DirectX, Metal, Vulkan
#define GLM_CLIP_CONTROL_RH_BIT		(1 << 3) // RIGHT_HANDED, For OpenGL, default in GLM

#define GLM_CLIP_CONTROL_LH_ZO (GLM_CLIP_CONTROL_LH_BIT | GLM_CLIP_CONTROL_ZO_BIT)
#define GLM_CLIP_CONTROL_LH_NO (GLM_CLIP_CONTROL_LH_BIT | GLM_CLIP_CONTROL_NO_BIT)
#define GLM_CLIP_CONTROL_RH_ZO (GLM_CLIP_CONTROL_RH_BIT | GLM_CLIP_CONTROL_ZO_BIT)
#define GLM_CLIP_CONTROL_RH_NO (GLM_CLIP_CONTROL_RH_BIT | GLM_CLIP_CONTROL_NO_BIT)

#ifdef GLM_FORCE_DEPTH_ZERO_TO_ONE
#	ifdef GLM_FORCE_LEFT_HANDED
#		define GLM_CONFIG_CLIP_CONTROL GLM_CLIP_CONTROL_LH_ZO
#	else
#		define GLM_CONFIG_CLIP_CONTROL GLM_CLIP_CONTROL_RH_ZO
#	endif
#else
#	ifdef GLM_FORCE_LEFT_HANDED
#		define GLM_CONFIG_CLIP_CONTROL GLM_CLIP_CONTROL_LH_NO
#	else
#		define GLM_CONFIG_CLIP_CONTROL GLM_CLIP_CONTROL_RH_NO
#	endif
#endif

///////////////////////////////////////////////////////////////////////////////////
// Qualifiers

#if (GLM_COMPILER & GLM_COMPILER_VC) || ((GLM_COMPILER & GLM_COMPILER_INTEL) && (GLM_PLATFORM & GLM_PLATFORM_WINDOWS))
#	define GLM_DEPRECATED __declspec(deprecated)
#	define GLM_ALIGNED_TYPEDEF(type, name, alignment) typedef __declspec(align(alignment)) type name
#elif GLM_COMPILER & (GLM_COMPILER_GCC | GLM_COMPILER_CLANG | GLM_COMPILER_INTEL)
#	define GLM_DEPRECATED __attribute__((__deprecated__))
#	define GLM_ALIGNED_TYPEDEF(type, name, alignment) typedef type name __attribute__((aligned(alignment)))
#elif (GLM_COMPILER & GLM_COMPILER_CUDA) || (GLM_COMPILER & GLM_COMPILER_HIP)
#	define GLM_DEPRECATED
#	define GLM_ALIGNED_TYPEDEF(type, name, alignment) typedef type name __align__(x)
#else
#	define GLM_DEPRECATED
#	define GLM_ALIGNED_TYPEDEF(type, name, alignment) typedef type name
#endif

///////////////////////////////////////////////////////////////////////////////////

#ifdef GLM_FORCE_EXPLICIT_CTOR
#	define GLM_EXPLICIT explicit
#else
#	define GLM_EXPLICIT
#endif

///////////////////////////////////////////////////////////////////////////////////
// Length type: all length functions returns a length_t type.
// When GLM_FORCE_SIZE_T_LENGTH is defined, length_t is a typedef of size_t otherwise
// length_t is a typedef of int like GLSL defines it.

#define GLM_LENGTH_INT		1
#define GLM_LENGTH_SIZE_T	2

#ifdef GLM_FORCE_SIZE_T_LENGTH
#	define GLM_CONFIG_LENGTH_TYPE		GLM_LENGTH_SIZE_T
#else
#	define GLM_CONFIG_LENGTH_TYPE		GLM_LENGTH_INT
#endif

namespace glm
{
	using std::size_t;
#	if GLM_CONFIG_LENGTH_TYPE == GLM_LENGTH_SIZE_T
		typedef size_t length_t;
#	else
		typedef int length_t;
#	endif
}//namespace glm

///////////////////////////////////////////////////////////////////////////////////
// constexpr

#if GLM_HAS_CONSTEXPR
#	define GLM_CONFIG_CONSTEXP GLM_ENABLE

	namespace glm
	{
		template<typename T, std::size_t N>
		constexpr std::size_t countof(T const (&)[N])
		{
			return N;
		}
	}//namespace glm
#	define GLM_COUNTOF(arr) glm::countof(arr)
#elif defined(_MSC_VER)
#	define GLM_CONFIG_CONSTEXP GLM_DISABLE

#	define GLM_COUNTOF(arr) _countof(arr)
#else
#	define GLM_CONFIG_CONSTEXP GLM_DISABLE

#	define GLM_COUNTOF(arr) sizeof(arr) / sizeof(arr[0])
#endif

///////////////////////////////////////////////////////////////////////////////////
// uint

namespace glm{
namespace detail
{
	template<typename T>
	struct is_int
	{
		enum test {value = 0};
	};

	template<>
	struct is_int<unsigned int>
	{
		enum test {value = ~0};
	};

	template<>
	struct is_int<signed int>
	{
		enum test {value = ~0};
	};
}//namespace detail

	typedef unsigned int	uint;
}//namespace glm

///////////////////////////////////////////////////////////////////////////////////
// 64-bit int

#if GLM_HAS_EXTENDED_INTEGER_TYPE
#	include <cstdint>
#endif

namespace glm{
namespace detail
{
#	if GLM_HAS_EXTENDED_INTEGER_TYPE
		typedef std::uint64_t						uint64;
		typedef std::int64_t						int64;
#	elif (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)) // C99 detected, 64 bit types available
		typedef uint64_t							uint64;
		typedef int64_t								int64;
#	elif GLM_COMPILER & GLM_COMPILER_VC
		typedef unsigned __int64					uint64;
		typedef signed __int64						int64;
#	elif GLM_COMPILER & GLM_COMPILER_GCC
#		pragma GCC diagnostic ignored "-Wlong-long"
		__extension__ typedef unsigned long long	uint64;
		__extension__ typedef signed long long		int64;
#	elif (GLM_COMPILER & GLM_COMPILER_CLANG)
#		pragma clang diagnostic ignored "-Wc++11-long-long"
		typedef unsigned long long					uint64;
		typedef signed long long					int64;
#	else//unknown compiler
		typedef unsigned long long					uint64;
		typedef signed long long					int64;
#	endif
}//namespace detail
}//namespace glm

///////////////////////////////////////////////////////////////////////////////////
// make_unsigned

#if GLM_HAS_MAKE_SIGNED
#	include <type_traits>

namespace glm{
namespace detail
{
	using std::make_unsigned;
}//namespace detail
}//namespace glm

#else

namespace glm{
namespace detail
{
	template<typename genType>
	struct make_unsigned
	{};

	template<>
	struct make_unsigned<char>
	{
		typedef unsigned char type;
	};

	template<>
	struct make_unsigned<signed char>
	{
		typedef unsigned char type;
	};

	template<>
	struct make_unsigned<short>
	{
		typedef unsigned short type;
	};

	template<>
	struct make_unsigned<int>
	{
		typedef unsigned int type;
	};

	template<>
	struct make_unsigned<long>
	{
		typedef unsigned long type;
	};

	template<>
	struct make_unsigned<int64>
	{
		typedef uint64 type;
	};

	template<>
	struct make_unsigned<unsigned char>
	{
		typedef unsigned char type;
	};

	template<>
	struct make_unsigned<unsigned short>
	{
		typedef unsigned short type;
	};

	template<>
	struct make_unsigned<unsigned int>
	{
		typedef unsigned int type;
	};

	template<>
	struct make_unsigned<unsigned long>
	{
		typedef unsigned long type;
	};

	template<>
	struct make_unsigned<uint64>
	{
		typedef uint64 type;
	};
}//namespace detail
}//namespace glm
#endif

///////////////////////////////////////////////////////////////////////////////////
// Only use x, y, z, w as vector type components

#ifdef GLM_FORCE_XYZW_ONLY
#	define GLM_CONFIG_XYZW_ONLY GLM_ENABLE
#else
#	define GLM_CONFIG_XYZW_ONLY GLM_DISABLE
#endif

///////////////////////////////////////////////////////////////////////////////////
// Configure the use of defaulted initialized types

#define GLM_CTOR_INIT_DISABLE		0
#define GLM_CTOR_INITIALIZER_LIST	1
#define GLM_CTOR_INITIALISATION		2

#if defined(GLM_FORCE_CTOR_INIT) && GLM_HAS_INITIALIZER_LISTS
#	define GLM_CONFIG_CTOR_INIT GLM_CTOR_INITIALIZER_LIST
#elif defined(GLM_FORCE_CTOR_INIT) && !GLM_HAS_INITIALIZER_LISTS
#	define GLM_CONFIG_CTOR_INIT GLM_CTOR_INITIALISATION
#else
#	define GLM_CONFIG_CTOR_INIT GLM_CTOR_INIT_DISABLE
#endif

///////////////////////////////////////////////////////////////////////////////////
// Use SIMD instruction sets

#if GLM_HAS_ALIGNOF && (GLM_LANG & GLM_LANG_CXXMS_FLAG) && (GLM_ARCH & GLM_ARCH_SIMD_BIT)
#	define GLM_CONFIG_SIMD GLM_ENABLE
#else
#	define GLM_CONFIG_SIMD GLM_DISABLE
#endif

///////////////////////////////////////////////////////////////////////////////////
// Configure the use of defaulted function

#if GLM_HAS_DEFAULTED_FUNCTIONS
#	define GLM_CONFIG_DEFAULTED_FUNCTIONS GLM_ENABLE
#	define GLM_DEFAULT = default
#else
#	define GLM_CONFIG_DEFAULTED_FUNCTIONS GLM_DISABLE
#	define GLM_DEFAULT
#endif

#if GLM_CONFIG_CTOR_INIT == GLM_CTOR_INIT_DISABLE && GLM_CONFIG_DEFAULTED_FUNCTIONS == GLM_ENABLE
#	define GLM_CONFIG_DEFAULTED_DEFAULT_CTOR GLM_ENABLE
#	define GLM_DEFAULT_CTOR GLM_DEFAULT
#else
#	define GLM_CONFIG_DEFAULTED_DEFAULT_CTOR GLM_DISABLE
#	define GLM_DEFAULT_CTOR
#endif

///////////////////////////////////////////////////////////////////////////////////
// Configure the use of aligned gentypes

#ifdef GLM_FORCE_ALIGNED // Legacy define
#	define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#endif

#ifdef GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#	define GLM_FORCE_ALIGNED_GENTYPES
#endif

#if GLM_HAS_ALIGNOF && (GLM_LANG & GLM_LANG_CXXMS_FLAG) && (defined(GLM_FORCE_ALIGNED_GENTYPES) || (GLM_CONFIG_SIMD == GLM_ENABLE))
#	define GLM_CONFIG_ALIGNED_GENTYPES GLM_ENABLE
#else
#	define GLM_CONFIG_ALIGNED_GENTYPES GLM_DISABLE
#endif

///////////////////////////////////////////////////////////////////////////////////
// Configure the use of anonymous structure as implementation detail

#if ((GLM_CONFIG_SIMD == GLM_ENABLE) || (GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR) || (GLM_CONFIG_ALIGNED_GENTYPES == GLM_ENABLE))
#	define GLM_CONFIG_ANONYMOUS_STRUCT GLM_ENABLE
#else
#	define GLM_CONFIG_ANONYMOUS_STRUCT GLM_DISABLE
#endif

///////////////////////////////////////////////////////////////////////////////////
// Silent warnings

#ifdef GLM_FORCE_SILENT_WARNINGS
#	define GLM_SILENT_WARNINGS GLM_ENABLE
#else
#	define GLM_SILENT_WARNINGS GLM_DISABLE
#endif

///////////////////////////////////////////////////////////////////////////////////
// Precision

#define GLM_HIGHP		1
#define GLM_MEDIUMP		2
#define GLM_LOWP		3

#if defined(GLM_FORCE_PRECISION_HIGHP_BOOL) || defined(GLM_PRECISION_HIGHP_BOOL)
#	define GLM_CONFIG_PRECISION_BOOL		GLM_HIGHP
#elif defined(GLM_FORCE_PRECISION_MEDIUMP_BOOL) || defined(GLM_PRECISION_MEDIUMP_BOOL)
#	define GLM_CONFIG_PRECISION_BOOL		GLM_MEDIUMP
#elif defined(GLM_FORCE_PRECISION_LOWP_BOOL) || defined(GLM_PRECISION_LOWP_BOOL)
#	define GLM_CONFIG_PRECISION_BOOL		GLM_LOWP
#else
#	define GLM_CONFIG_PRECISION_BOOL		GLM_HIGHP
#endif

#if defined(GLM_FORCE_PRECISION_HIGHP_INT) || defined(GLM_PRECISION_HIGHP_INT)
#	define GLM_CONFIG_PRECISION_INT			GLM_HIGHP
#elif defined(GLM_FORCE_PRECISION_MEDIUMP_INT) || defined(GLM_PRECISION_MEDIUMP_INT)
#	define GLM_CONFIG_PRECISION_INT			GLM_MEDIUMP
#elif defined(GLM_FORCE_PRECISION_LOWP_INT) || defined(GLM_PRECISION_LOWP_INT)
#	define GLM_CONFIG_PRECISION_INT			GLM_LOWP
#else
#	define GLM_CONFIG_PRECISION_INT			GLM_HIGHP
#endif

#if defined(GLM_FORCE_PRECISION_HIGHP_UINT) || defined(GLM_PRECISION_HIGHP_UINT)
#	define GLM_CONFIG_PRECISION_UINT		GLM_HIGHP
#elif defined(GLM_FORCE_PRECISION_MEDIUMP_UINT) || defined(GLM_PRECISION_MEDIUMP_UINT)
#	define GLM_CONFIG_PRECISION_UINT		GLM_MEDIUMP
#elif defined(GLM_FORCE_PRECISION_LOWP_UINT) || defined(GLM_PRECISION_LOWP_UINT)
#	define GLM_CONFIG_PRECISION_UINT		GLM_LOWP
#else
#	define GLM_CONFIG_PRECISION_UINT		GLM_HIGHP
#endif

#if defined(GLM_FORCE_PRECISION_HIGHP_FLOAT) || defined(GLM_PRECISION_HIGHP_FLOAT)
#	define GLM_CONFIG_PRECISION_FLOAT		GLM_HIGHP
#elif defined(GLM_FORCE_PRECISION_MEDIUMP_FLOAT) || defined(GLM_PRECISION_MEDIUMP_FLOAT)
#	define GLM_CONFIG_PRECISION_FLOAT		GLM_MEDIUMP
#elif defined(GLM_FORCE_PRECISION_LOWP_FLOAT) || defined(GLM_PRECISION_LOWP_FLOAT)
#	define GLM_CONFIG_PRECISION_FLOAT		GLM_LOWP
#else
#	define GLM_CONFIG_PRECISION_FLOAT		GLM_HIGHP
#endif

#if defined(GLM_FORCE_PRECISION_HIGHP_DOUBLE) || defined(GLM_PRECISION_HIGHP_DOUBLE)
#	define GLM_CONFIG_PRECISION_DOUBLE		GLM_HIGHP
#elif defined(GLM_FORCE_PRECISION_MEDIUMP_DOUBLE) || defined(GLM_PRECISION_MEDIUMP_DOUBLE)
#	define GLM_CONFIG_PRECISION_DOUBLE		GLM_MEDIUMP
#elif defined(GLM_FORCE_PRECISION_LOWP_DOUBLE) || defined(GLM_PRECISION_LOWP_DOUBLE)
#	define GLM_CONFIG_PRECISION_DOUBLE		GLM_LOWP
#else
#	define GLM_CONFIG_PRECISION_DOUBLE		GLM_HIGHP
#endif

///////////////////////////////////////////////////////////////////////////////////
// Check inclusions of different versions of GLM

#elif ((GLM_SETUP_INCLUDED != GLM_VERSION) && !defined(GLM_FORCE_IGNORE_VERSION))
#	error "GLM error: A different version of GLM is already included. Define GLM_FORCE_IGNORE_VERSION before including GLM headers to ignore this error."
#elif GLM_SETUP_INCLUDED == GLM_VERSION

///////////////////////////////////////////////////////////////////////////////////
// Messages

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_MESSAGE_DISPLAYED)
#	define GLM_MESSAGE_DISPLAYED
#		define GLM_STR_HELPER(x) #x
#		define GLM_STR(x) GLM_STR_HELPER(x)

	// Report GLM version
#		pragma message (GLM_STR(GLM_VERSION_MESSAGE))

	// Report C++ language
#	if (GLM_LANG & GLM_LANG_CXX20_FLAG) && (GLM_LANG & GLM_LANG_EXT)
#		pragma message("GLM: C++ 20 with extensions")
#	elif (GLM_LANG & GLM_LANG_CXX20_FLAG)
#		pragma message("GLM: C++ 2A")
#	elif (GLM_LANG & GLM_LANG_CXX17_FLAG) && (GLM_LANG & GLM_LANG_EXT)
#		pragma message("GLM: C++ 17 with extensions")
#	elif (GLM_LANG & GLM_LANG_CXX17_FLAG)
#		pragma message("GLM: C++ 17")
#	elif (GLM_LANG & GLM_LANG_CXX14_FLAG) && (GLM_LANG & GLM_LANG_EXT)
#		pragma message("GLM: C++ 14 with extensions")
#	elif (GLM_LANG & GLM_LANG_CXX14_FLAG)
#		pragma message("GLM: C++ 14")
#	elif (GLM_LANG & GLM_LANG_CXX11_FLAG) && (GLM_LANG & GLM_LANG_EXT)
#		pragma message("GLM: C++ 11 with extensions")
#	elif (GLM_LANG & GLM_LANG_CXX11_FLAG)
#		pragma message("GLM: C++ 11")
#	elif (GLM_LANG & GLM_LANG_CXX0X_FLAG) && (GLM_LANG & GLM_LANG_EXT)
#		pragma message("GLM: C++ 0x with extensions")
#	elif (GLM_LANG & GLM_LANG_CXX0X_FLAG)
#		pragma message("GLM: C++ 0x")
#	elif (GLM_LANG & GLM_LANG_CXX03_FLAG) && (GLM_LANG & GLM_LANG_EXT)
#		pragma message("GLM: C++ 03 with extensions")
#	elif (GLM_LANG & GLM_LANG_CXX03_FLAG)
#		pragma message("GLM: C++ 03")
#	elif (GLM_LANG & GLM_LANG_CXX98_FLAG) && (GLM_LANG & GLM_LANG_EXT)
#		pragma message("GLM: C++ 98 with extensions")
#	elif (GLM_LANG & GLM_LANG_CXX98_FLAG)
#		pragma message("GLM: C++ 98")
#	else
#		pragma message("GLM: C++ language undetected")
#	endif//GLM_LANG

	// Report compiler detection
#	if GLM_COMPILER & GLM_COMPILER_CUDA
#		pragma message("GLM: CUDA compiler detected")
#	elif GLM_COMPILER & GLM_COMPILER_HIP
#		pragma message("GLM: HIP compiler detected")
#	elif GLM_COMPILER & GLM_COMPILER_VC
#		pragma message("GLM: Visual C++ compiler detected")
#	elif GLM_COMPILER & GLM_COMPILER_CLANG
#		pragma message("GLM: Clang compiler detected")
#	elif GLM_COMPILER & GLM_COMPILER_INTEL
#		pragma message("GLM: Intel Compiler detected")
#	elif GLM_COMPILER & GLM_COMPILER_GCC
#		pragma message("GLM: GCC compiler detected")
#	else
#		pragma message("GLM: Compiler not detected")
#	endif

	// Report build target
#	if (GLM_ARCH & GLM_ARCH_AVX2_BIT) && (GLM_MODEL == GLM_MODEL_64)
#		pragma message("GLM: x86 64 bits with AVX2 instruction set build target")
#	elif (GLM_ARCH & GLM_ARCH_AVX2_BIT) && (GLM_MODEL == GLM_MODEL_32)
#		pragma message("GLM: x86 32 bits with AVX2 instruction set build target")

#	elif (GLM_ARCH & GLM_ARCH_AVX_BIT) && (GLM_MODEL == GLM_MODEL_64)
#		pragma message("GLM: x86 64 bits with AVX instruction set build target")
#	elif (GLM_ARCH & GLM_ARCH_AVX_BIT) && (GLM_MODEL == GLM_MODEL_32)
#		pragma message("GLM: x86 32 bits with AVX instruction set build target")

#	elif (GLM_ARCH & GLM_ARCH_SSE42_BIT) && (GLM_MODEL == GLM_MODEL_64)
#		pragma message("GLM: x86 64 bits with SSE4.2 instruction set build target")
#	elif (GLM_ARCH & GLM_ARCH_SSE42_BIT) && (GLM_MODEL == GLM_MODEL_32)
#		pragma message("GLM: x86 32 bits with SSE4.2 instruction set build target")

#	elif (GLM_ARCH & GLM_ARCH_SSE41_BIT) && (GLM_MODEL == GLM_MODEL_64)
#		pragma message("GLM: x86 64 bits with SSE4.1 instruction set build target")
#	elif (GLM_ARCH & GLM_ARCH_SSE41_BIT) && (GLM_MODEL == GLM_MODEL_32)
#		pragma message("GLM: x86 32 bits with SSE4.1 instruction set build target")

#	elif (GLM_ARCH & GLM_ARCH_SSSE3_BIT) && (GLM_MODEL == GLM_MODEL_64)
#		pragma message("GLM: x86 64 bits with SSSE3 instruction set build target")
#	elif (GLM_ARCH & GLM_ARCH_SSSE3_BIT) && (GLM_MODEL == GLM_MODEL_32)
#		pragma message("GLM: x86 32 bits with SSSE3 instruction set build target")

#	elif (GLM_ARCH & GLM_ARCH_SSE3_BIT) && (GLM_MODEL == GLM_MODEL_64)
#		pragma message("GLM: x86 64 bits with SSE3 instruction set build target")
#	elif (GLM_ARCH & GLM_ARCH_SSE3_BIT) && (GLM_MODEL == GLM_MODEL_32)
#		pragma message("GLM: x86 32 bits with SSE3 instruction set build target")

#	elif (GLM_ARCH & GLM_ARCH_SSE2_BIT) && (GLM_MODEL == GLM_MODEL_64)
#		pragma message("GLM: x86 64 bits with SSE2 instruction set build target")
#	elif (GLM_ARCH & GLM_ARCH_SSE2_BIT) && (GLM_MODEL == GLM_MODEL_32)
#		pragma message("GLM: x86 32 bits with SSE2 instruction set build target")

#	elif (GLM_ARCH & GLM_ARCH_X86_BIT) && (GLM_MODEL == GLM_MODEL_64)
#		pragma message("GLM: x86 64 bits build target")
#	elif (GLM_ARCH & GLM_ARCH_X86_BIT) && (GLM_MODEL == GLM_MODEL_32)
#		pragma message("GLM: x86 32 bits build target")

#	elif (GLM_ARCH & GLM_ARCH_NEON_BIT) && (GLM_MODEL == GLM_MODEL_64)
#		pragma message("GLM: ARM 64 bits with Neon instruction set build target")
#	elif (GLM_ARCH & GLM_ARCH_NEON_BIT) && (GLM_MODEL == GLM_MODEL_32)
#		pragma message("GLM: ARM 32 bits with Neon instruction set build target")

#	elif (GLM_ARCH & GLM_ARCH_ARM_BIT) && (GLM_MODEL == GLM_MODEL_64)
#		pragma message("GLM: ARM 64 bits build target")
#	elif (GLM_ARCH & GLM_ARCH_ARM_BIT) && (GLM_MODEL == GLM_MODEL_32)
#		pragma message("GLM: ARM 32 bits build target")

#	elif (GLM_ARCH & GLM_ARCH_MIPS_BIT) && (GLM_MODEL == GLM_MODEL_64)
#		pragma message("GLM: MIPS 64 bits build target")
#	elif (GLM_ARCH & GLM_ARCH_MIPS_BIT) && (GLM_MODEL == GLM_MODEL_32)
#		pragma message("GLM: MIPS 32 bits build target")

#	elif (GLM_ARCH & GLM_ARCH_PPC_BIT) && (GLM_MODEL == GLM_MODEL_64)
#		pragma message("GLM: PowerPC 64 bits build target")
#	elif (GLM_ARCH & GLM_ARCH_PPC_BIT) && (GLM_MODEL == GLM_MODEL_32)
#		pragma message("GLM: PowerPC 32 bits build target")
#	else
#		pragma message("GLM: Unknown build target")
#	endif//GLM_ARCH

	// Report platform name
#	if(GLM_PLATFORM & GLM_PLATFORM_QNXNTO)
#		pragma message("GLM: QNX platform detected")
//#	elif(GLM_PLATFORM & GLM_PLATFORM_IOS)
//#		pragma message("GLM: iOS platform detected")
#	elif(GLM_PLATFORM & GLM_PLATFORM_APPLE)
#		pragma message("GLM: Apple platform detected")
#	elif(GLM_PLATFORM & GLM_PLATFORM_WINCE)
#		pragma message("GLM: WinCE platform detected")
#	elif(GLM_PLATFORM & GLM_PLATFORM_WINDOWS)
#		pragma message("GLM: Windows platform detected")
#	elif(GLM_PLATFORM & GLM_PLATFORM_CHROME_NACL)
#		pragma message("GLM: Native Client detected")
#	elif(GLM_PLATFORM & GLM_PLATFORM_ANDROID)
#		pragma message("GLM: Android platform detected")
#	elif(GLM_PLATFORM & GLM_PLATFORM_LINUX)
#		pragma message("GLM: Linux platform detected")
#	elif(GLM_PLATFORM & GLM_PLATFORM_UNIX)
#		pragma message("GLM: UNIX platform detected")
#	elif(GLM_PLATFORM & GLM_PLATFORM_UNKNOWN)
#		pragma message("GLM: platform unknown")
#	else
#		pragma message("GLM: platform not detected")
#	endif

	// Report whether only xyzw component are used
#	if defined GLM_FORCE_XYZW_ONLY
#		pragma message("GLM: GLM_FORCE_XYZW_ONLY is defined. Only x, y, z and w component are available in vector type. This define disables swizzle operators and SIMD instruction sets.")
#	endif

	// Report swizzle operator support
#	if GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_OPERATOR
#		pragma message("GLM: GLM_FORCE_SWIZZLE is defined, swizzling operators enabled.")
#	elif GLM_CONFIG_SWIZZLE == GLM_SWIZZLE_FUNCTION
#		pragma message("GLM: GLM_FORCE_SWIZZLE is defined, swizzling functions enabled. Enable compiler C++ language extensions to enable swizzle operators.")
#	else
#		pragma message("GLM: GLM_FORCE_SWIZZLE is undefined. swizzling functions or operators are disabled.")
#	endif

	// Report .length() type
#	if GLM_CONFIG_LENGTH_TYPE == GLM_LENGTH_SIZE_T
#		pragma message("GLM: GLM_FORCE_SIZE_T_LENGTH is defined. .length() returns a glm::length_t, a typedef of std::size_t.")
#	else
#		pragma message("GLM: GLM_FORCE_SIZE_T_LENGTH is undefined. .length() returns a glm::length_t, a typedef of int following GLSL.")
#	endif

#	if GLM_CONFIG_UNRESTRICTED_GENTYPE == GLM_ENABLE
#		pragma message("GLM: GLM_FORCE_UNRESTRICTED_GENTYPE is defined. Removes GLSL restrictions on valid function genTypes.")
#	else
#		pragma message("GLM: GLM_FORCE_UNRESTRICTED_GENTYPE is undefined. Follows strictly GLSL on valid function genTypes.")
#	endif

#	if GLM_SILENT_WARNINGS == GLM_ENABLE
#		pragma message("GLM: GLM_FORCE_SILENT_WARNINGS is defined. Ignores C++ warnings from using C++ language extensions.")
#	else
#		pragma message("GLM: GLM_FORCE_SILENT_WARNINGS is undefined. Shows C++ warnings from using C++ language extensions.")
#	endif

#	ifdef GLM_FORCE_SINGLE_ONLY
#		pragma message("GLM: GLM_FORCE_SINGLE_ONLY is defined. Using only single precision floating-point types.")
#	endif

#	if defined(GLM_FORCE_ALIGNED_GENTYPES) && (GLM_CONFIG_ALIGNED_GENTYPES == GLM_ENABLE)
#		undef GLM_FORCE_ALIGNED_GENTYPES
#		pragma message("GLM: GLM_FORCE_ALIGNED_GENTYPES is defined, allowing aligned types. This prevents the use of C++ constexpr.")
#	elif defined(GLM_FORCE_ALIGNED_GENTYPES) && (GLM_CONFIG_ALIGNED_GENTYPES == GLM_DISABLE)
#		undef GLM_FORCE_ALIGNED_GENTYPES
#		pragma message("GLM: GLM_FORCE_ALIGNED_GENTYPES is defined but is disabled. It requires C++11 and language extensions.")
#	endif

#	if defined(GLM_FORCE_DEFAULT_ALIGNED_GENTYPES)
#		if GLM_CONFIG_ALIGNED_GENTYPES == GLM_DISABLE
#			undef GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#			pragma message("GLM: GLM_FORCE_DEFAULT_ALIGNED_GENTYPES is defined but is disabled. It requires C++11 and language extensions.")
#		elif GLM_CONFIG_ALIGNED_GENTYPES == GLM_ENABLE
#			pragma message("GLM: GLM_FORCE_DEFAULT_ALIGNED_GENTYPES is defined. All gentypes (e.g. vec3) will be aligned and padded by default.")
#		endif
#	endif

#	if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_ZO_BIT
#		pragma message("GLM: GLM_FORCE_DEPTH_ZERO_TO_ONE is defined. Using zero to one depth clip space.")
#	else
#		pragma message("GLM: GLM_FORCE_DEPTH_ZERO_TO_ONE is undefined. Using negative one to one depth clip space.")
#	endif

#	if GLM_CONFIG_CLIP_CONTROL & GLM_CLIP_CONTROL_LH_BIT
#		pragma message("GLM: GLM_FORCE_LEFT_HANDED is defined. Using left handed coordinate system.")
#	else
#		pragma message("GLM: GLM_FORCE_LEFT_HANDED is undefined. Using right handed coordinate system.")
#	endif
#endif//GLM_MESSAGES

#endif//GLM_SETUP_INCLUDED
