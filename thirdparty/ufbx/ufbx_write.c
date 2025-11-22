#ifndef UFBXW_UFBX_WRITE_C_INCLUDED
#define UFBXW_UFBX_WRITE_C_INCLUDED

#if defined(UFBX_HEADER_PATH)
	#include UFBX_HEADER_PATH
#else
	#include "ufbx_write.h"
#endif

#include <stdlib.h>
#include <string.h>

#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// -- Math
// TODO: External
#include <math.h>

// -- Malloc

#if defined(ufbxw_malloc) || defined(ufbxw_realloc) || defined(ufbxw_free)
	// User provided allocators
	#if !defined(ufbxw_malloc) || !defined(ufbxw_realloc) || !defined(ufbxw_free)
		#error Inconsistent custom global allocator
	#endif
#else
	#define ufbxw_malloc(size) malloc((size))
	#define ufbxw_realloc(ptr, old_size, new_size) realloc((ptr), (new_size))
	#define ufbxw_free(ptr, old_size) free((ptr))
#endif

// -- Platform

#if !defined(UFBXW_STANDARD_C) && defined(_MSC_VER)
	#define ufbxwi_noinline __declspec(noinline)
	#define ufbxwi_forceinline __forceinline
	#define ufbxwi_restrict __restrict
	#if defined(_Check_return_)
		#define ufbxwi_nodiscard _Check_return_
	#else
		#define ufbxwi_nodiscard
	#endif
	#define ufbxwi_unused
	#define ufbxwi_unlikely(cond) (cond)
#elif !defined(UFBXW_STANDARD_C) && (defined(__GNUC__) || defined(__clang__))
	#define ufbxwi_noinline __attribute__((noinline))
	#define ufbxwi_forceinline inline __attribute__((always_inline))
	#define ufbxwi_restrict __restrict
	#define ufbxwi_nodiscard __attribute__((warn_unused_result))
	#define ufbxwi_unused __attribute__((unused))
	#define ufbxwi_unlikely(cond) __builtin_expect((cond), 0)
#else
	#define ufbxwi_noinline
	#define ufbxwi_forceinline
	#define ufbxwi_nodiscard
	#define ufbxwi_restrict
	#define ufbxwi_unused
	#define ufbxwi_unlikely(cond) (cond)
#endif

#if !defined(UFBXW_STANDARD_C) && defined(__clang__)
	#define ufbxwi_nounroll _Pragma("clang loop unroll(disable)") _Pragma("clang loop vectorize(disable)")
#elif !defined(UFBXW_STANDARD_C) && UFBXWI_GNUC >= 8
	#define ufbxwi_nounroll _Pragma("GCC unroll 0")
#elif !defined(UFBXW_STANDARD_C) && defined(_MSC_VER)
	#define ufbxwi_nounroll __pragma(loop(no_vector))
#else
	#define ufbxwi_nounroll
#endif

// Should be standard in C99 and C++11
#define ufbxwi_func __func__

#if defined(UFBXW_STATIC_ANALYSIS)
	bool ufbxwi_analysis_opaque;
	#define ufbxwi_maybe_null(ptr) (ufbxwi_analysis_opaque ? (ptr) : NULL)
	#define ufbxwi_analysis_assert(cond) ufbxw_assert(cond)
#else
	#define ufbxwi_maybe_null(ptr) (ptr)
	#define ufbxwi_analysis_assert(cond) (void)0
#endif

#if defined(UFBXW_REGRESSION) || defined(UFBXW_DEV) || defined(UFBXW_UBSAN)
	#define ufbxwi_dev_assert(cond) ufbxw_assert(cond)
#else
	#define ufbxwi_dev_assert(cond) (void)0
#endif

#define ufbxwi_unreachable(reason) do { ufbxw_assert(0 && reason); } while (0)

#if defined(__GNUC__) && !defined(__clang__)
	#define ufbxwi_ignore(cond) (void)!(cond)
#else
	#define ufbxwi_ignore(cond) (void)(cond)
#endif

#if defined(UFBXW_UBSAN)
	static void ufbxwi_assert_zero(size_t offset) { ufbxw_assert(offset == 0); }
	#define ufbxwi_add_ptr(ptr, offset) ((ptr) ? (ptr) + (offset) : (ufbxwi_assert_zero((size_t)(offset)), (ptr)))
	#define ufbxwi_sub_ptr(ptr, offset) ((ptr) ? (ptr) - (offset) : (ufbxwi_assert_zero((size_t)(offset)), (ptr)))
#else
	#define ufbxwi_add_ptr(ptr, offset) ((ptr) + (offset))
	#define ufbxwi_sub_ptr(ptr, offset) ((ptr) - (offset))
#endif

#if defined(UFBXW_REGRESSION)
	static size_t ufbxwi_to_size(ptrdiff_t delta) {
		ufbxw_assert(delta >= 0);
		return (size_t)delta;
	}
#else
	#define ufbxwi_to_size(delta) ((size_t)(delta))
#endif

#if !defined(ufbxw_static_assert)
	#if defined(__cplusplus) && __cplusplus >= 201103
		#define ufbxw_static_assert(desc, cond) static_assert(cond, #desc ": " #cond)
	#else
		#define ufbxw_static_assert(desc, cond) typedef char ufbxwi_static_assert_##desc[(cond)?1:-1]
	#endif
#endif

#if defined(UFBXW_REGRESSION)
	#define UFBXWI_DEFLATE_WINDOW_SIZE 128
#else
	#define UFBXWI_DEFLATE_WINDOW_SIZE 65536
#endif

// -- Pointer alignment

#if !defined(UFBXW_STANDARD_C) && defined(__GNUC__) && defined(__has_builtin)
	#if __has_builtin(__builtin_is_aligned)
		#define ufbxwi_is_aligned(m_ptr, m_align) __builtin_is_aligned((m_ptr), (m_align))
		#define ufbxwi_is_aligned_mask(m_ptr, m_align) __builtin_is_aligned((m_ptr), (m_align) + 1)
	#endif
#endif
#ifndef ufbxwi_is_aligned
	#define ufbxwi_is_aligned(m_ptr, m_align) (((uintptr_t)(m_ptr) & ((m_align) - 1)) == 0)
	#define ufbxwi_is_aligned_mask(m_ptr, m_align) (((uintptr_t)(m_ptr) & (m_align)) == 0)
#endif

// -- Bit manipulation

#if defined(__cplusplus)
	#define ufbxwi_extern_c extern "C"
#else
	#define ufbxwi_extern_c
#endif

#if !defined(UFBXW_STANDARD_C) && defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
	ufbxwi_extern_c unsigned char _BitScanReverse(unsigned long * _Index, unsigned long _Mask);
	ufbxwi_extern_c unsigned char _BitScanReverse64(unsigned long * _Index, unsigned __int64 _Mask);
	static ufbxwi_forceinline ufbxwi_unused uint32_t ufbxwi_lzcnt32(uint32_t v) {
		unsigned long index;
		_BitScanReverse(&index, (unsigned long)v);
		return 31 - (uint32_t)index;
	}
	static ufbxwi_forceinline ufbxwi_unused uint32_t ufbxwi_lzcnt64(uint64_t v) {
		unsigned long index;
		#if defined(_M_X64)
			_BitScanReverse64(&index, (unsigned __int64)v);
		#else
			uint32_t hi = (uint32_t)(v >> 32u);
			uint32_t hi_nonzero = hi != 0 ? 1 : 0;
			uint32_t part = hi_nonzero ? hi : (uint32_t)v;
			_BitScanReverse(&index, (unsigned long)part);
			index += hi_nonzero * 32u;
		#endif
		return 63 - (uint32_t)index;
	}
#elif !defined(UFBXW_STANDARD_C) && (defined(__GNUC__) || defined(__clang__))
	#define ufbxwi_lzcnt32(v) ((uint32_t)__builtin_clz((unsigned)(v)))
	#define ufbxwi_lzcnt64(v) ((uint32_t)__builtin_clzll((unsigned long long)(v)))
#else
	// DeBrujin table lookup
	static const uint8_t ufbxwi_lzcnt32_table[] =  {
		31, 22, 30, 21, 18, 10, 29, 2, 20, 17, 15, 13, 9, 6, 28, 1, 23, 19, 11, 3, 16, 14, 7, 24, 12, 4, 8, 25, 5, 26, 27, 0,
	};
	static const uint8_t ufbxwi_lzcnt64_table[] = {
		63, 16, 62, 7, 15, 36, 61, 3, 6, 14, 22, 26, 35, 47, 60, 2, 9, 5, 28, 11, 13, 21, 42,
		19, 25, 31, 34, 40, 46, 52, 59, 1, 17, 8, 37, 4, 23, 27, 48, 10, 29, 12, 43, 20, 32, 41,
		53, 18, 38, 24, 49, 30, 44, 33, 54, 39, 50, 45, 55, 51, 56, 57, 58, 0,
	};
	static ufbxwi_noinline ufbxwi_unused uint32_t ufbxwi_lzcnt32(uint32_t v) {
		v |= v >> 1;
		v |= v >> 2;
		v |= v >> 4;
		v |= v >> 8;
		v |= v >> 16;
		return ufbxwi_lzcnt32_table[(v * 0x07c4acddu) >> 27];
	}
	static ufbxwi_noinline ufbxwi_unused uint32_t ufbxwi_lzcnt64(uint64_t v) {
		v |= v >> 1;
		v |= v >> 2;
		v |= v >> 4;
		v |= v >> 8;
		v |= v >> 16;
		v |= v >> 32;
		return ufbxwi_lzcnt64_table[(v * UINT64_C(0x03f79d71b4cb0a89)) >> 58];
	}
#endif

#if defined(UFBXWI_HAS_ATTRIBUTE_ALIGNED)
	#define UFBXWI_HAS_UNALIGNED 1
	#define UFBXWI_HAS_ALIASING 1
	#define ufbxwi_unaligned
	typedef uint16_t __attribute__((aligned(1))) ufbxwi_unaligned_u16;
	typedef uint32_t __attribute__((aligned(1))) ufbxwi_unaligned_u32;
	typedef uint64_t __attribute__((aligned(1))) ufbxwi_unaligned_u64;
	typedef float __attribute__((aligned(1))) ufbxwi_unaligned_f32;
	typedef double __attribute__((aligned(1))) ufbxwi_unaligned_f64;
	typedef uint32_t __attribute__((may_alias)) ufbxwi_aliasing_u32;
#elif !defined(UFBXW_STANDARD_C) && defined(_MSC_VER)
	#define UFBXWI_HAS_UNALIGNED 1
	#if defined(_M_IX86)
		// MSVC seems to assume all pointers are unaligned for x86
		#define ufbxwi_unaligned
	#else
		#define ufbxwi_unaligned __unaligned
	#endif
	typedef uint16_t ufbxwi_unaligned_u16;
	typedef uint32_t ufbxwi_unaligned_u32;
	typedef uint64_t ufbxwi_unaligned_u64;
	typedef float ufbxwi_unaligned_f32;
	typedef double ufbxwi_unaligned_f64;
	// MSVC doesn't have aliasing types in theory, but it works in practice..
	#define UFBXWI_HAS_ALIASING 1
	typedef uint32_t ufbxwi_aliasing_u32;
#endif

#if (defined(UFBXWI_HAS_UNALIGNED) && UFBXW_LITTLE_ENDIAN && !defined(UFBXW_NO_UNALIGNED_LOADS)) || defined(UFBXW_USE_UNALIGNED_LOADS)
	#define ufbxwi_read_u16(ptr) (*(const ufbxwi_unaligned ufbxwi_unaligned_u16*)(ptr))
	#define ufbxwi_read_u32(ptr) (*(const ufbxwi_unaligned ufbxwi_unaligned_u32*)(ptr))
	#define ufbxwi_read_u64(ptr) (*(const ufbxwi_unaligned ufbxwi_unaligned_u64*)(ptr))
	#define ufbxwi_read_f32(ptr) (*(const ufbxwi_unaligned ufbxwi_unaligned_f32*)(ptr))
	#define ufbxwi_read_f64(ptr) (*(const ufbxwi_unaligned ufbxwi_unaligned_f64*)(ptr))
#else
	static ufbxwi_forceinline uint16_t ufbxwi_read_u16(const void *ptr) {
		const char *p = (const char*)ptr;
		return (uint16_t)(
			(unsigned)(uint8_t)p[0] << 0u |
			(unsigned)(uint8_t)p[1] << 8u );
	}
	static ufbxwi_forceinline uint32_t ufbxwi_read_u32(const void *ptr) {
		const char *p = (const char*)ptr;
		return (uint32_t)(
			(unsigned)(uint8_t)p[0] <<  0u |
			(unsigned)(uint8_t)p[1] <<  8u |
			(unsigned)(uint8_t)p[2] << 16u |
			(unsigned)(uint8_t)p[3] << 24u );
	}
	static ufbxwi_forceinline uint64_t ufbxwi_read_u64(const void *ptr) {
		const char *p = (const char*)ptr;
		return (uint64_t)(
			(uint64_t)(uint8_t)p[0] <<  0u |
			(uint64_t)(uint8_t)p[1] <<  8u |
			(uint64_t)(uint8_t)p[2] << 16u |
			(uint64_t)(uint8_t)p[3] << 24u |
			(uint64_t)(uint8_t)p[4] << 32u |
			(uint64_t)(uint8_t)p[5] << 40u |
			(uint64_t)(uint8_t)p[6] << 48u |
			(uint64_t)(uint8_t)p[7] << 56u );
	}
	static ufbxwi_forceinline float ufbxwi_read_f32(const void *ptr) {
		uint32_t u = ufbxwi_read_u32(ptr);
		float f;
		memcpy(&f, &u, 4);
		return f;
	}
	static ufbxwi_forceinline double ufbxwi_read_f64(const void *ptr) {
		uint64_t u = ufbxwi_read_u64(ptr);
		double f;
		memcpy(&f, &u, 8);
		return f;
	}
#endif

#define UFBXWI_PI ((ufbxw_real)3.14159265358979323846)
#define UFBXWI_DPI (3.14159265358979323846)
#define UFBXWI_DEG_TO_RAD_DOUBLE (UFBXWI_DPI / 180.0)
#define UFBXWI_RAD_TO_DEG_DOUBLE (180.0 / UFBXWI_DPI)

// -- Math

#if !defined(UFBXW_EXTERNAL_MATH)
	#if !defined(UFBXW_MATH_PREFIX)
		#define UFBXW_MATH_PREFIX
	#endif
#endif

#define ufbxwi_pre_cat2(a, b) a##b
#define ufbxwi_pre_cat(a, b) ufbxwi_pre_cat2(a, b)

// -- External functions

#ifndef ufbxw_extern_abi
	#if defined(UFBXW_STATIC)
		#define ufbxw_extern_abi static
	#else
		#define ufbxw_extern_abi
	#endif
#endif

#if defined(UFBXW_MATH_PREFIX)
	#define ufbxwi_math_fn(name) ufbxwi_pre_cat(UFBXW_MATH_PREFIX, name)
	#define ufbxw_sqrt ufbxwi_math_fn(sqrt)
	#define ufbxw_fabs ufbxwi_math_fn(fabs)
	#define ufbxw_pow ufbxwi_math_fn(pow)
	#define ufbxw_sin ufbxwi_math_fn(sin)
	#define ufbxw_cos ufbxwi_math_fn(cos)
	#define ufbxw_tan ufbxwi_math_fn(tan)
	#define ufbxw_asin ufbxwi_math_fn(asin)
	#define ufbxw_acos ufbxwi_math_fn(acos)
	#define ufbxw_atan ufbxwi_math_fn(atan)
	#define ufbxw_atan2 ufbxwi_math_fn(atan2)
	#define ufbxw_copysign ufbxwi_math_fn(copysign)
	#define ufbxw_fmin ufbxwi_math_fn(fmin)
	#define ufbxw_fmax ufbxwi_math_fn(fmax)
	#define ufbxw_nextafter ufbxwi_math_fn(nextafter)
	#define ufbxw_rint ufbxwi_math_fn(rint)
	#define ufbxw_floor ufbxwi_math_fn(floor)
	#define ufbxw_ceil ufbxwi_math_fn(ceil)
#endif

// TODO: External

// -- Features

#ifndef UFBXW_UNIT_TEST
	#define UFBXWI_FEATURE_ATOMICS 1
	#define UFBXWI_FEATURE_THREAD_POOL 1
	#define UFBXWI_FEATURE_ERROR 1
	#define UFBXWI_FEATURE_ALLOCATOR 1
	#define UFBXWI_FEATURE_LIST 1
	#define UFBXWI_FEATURE_TASK_QUEUE 1
	#define UFBXWI_FEATURE_STRING_POOL 1
	#define UFBXWI_FEATURE_BUFFER 1
	#define UFBXWI_FEATURE_SCENE 1
	#define UFBXWI_FEATURE_WRITE_QUEUE 1
	#define UFBXWI_FEATURE_SAVE 1
	#define UFBXWI_FEATURE_API 1
#endif

// -- Version

#define UFBXW_SOURCE_VERSION ufbxw_pack_version(0, 1, 0)
ufbxw_abi_data_def const uint32_t ufbxw_source_version = UFBXW_SOURCE_VERSION;

ufbxw_static_assert(source_header_version, UFBXW_SOURCE_VERSION/1000U == UFBXW_HEADER_VERSION/1000U);

// -- Utility

static ufbxwi_forceinline uint32_t ufbxwi_min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }
static ufbxwi_forceinline uint32_t ufbxwi_max_u32(uint32_t a, uint32_t b) { return a < b ? b : a; }
static ufbxwi_forceinline size_t ufbxwi_min_sz(size_t a, size_t b) { return a < b ? a : b; }
static ufbxwi_forceinline size_t ufbxwi_max_sz(size_t a, size_t b) { return a < b ? b : a; }
static ufbxwi_forceinline int64_t ufbxwi_min_i64(int64_t a, int64_t b) { return a < b ? a : b; }
static ufbxwi_forceinline int64_t ufbxwi_max_i64(int64_t a, int64_t b) { return a < b ? b : a; }

#define ufbxwi_arraycount(arr) (sizeof(arr) / sizeof(*(arr)))
#define ufbxwi_for(m_type, m_name, m_begin, m_num) for (m_type *m_name = m_begin, *m_name##_end = ufbxwi_add_ptr(m_name, m_num); m_name != m_name##_end; m_name++)
#define ufbxwi_for_ptr(m_type, m_name, m_begin, m_num) for (m_type **m_name = m_begin, **m_name##_end = ufbxwi_add_ptr(m_name, m_num); m_name != m_name##_end; m_name++)

typedef bool ufbxwi_less_fn(void *user, const void *a, const void *b);

static ufbxwi_forceinline void ufbxwi_swap(void *a, void *b, size_t size)
{
#if UFBXWI_HAS_ALIASING && !defined(__CHERI__) // CHERI needs to copy pointer metadata tag bits..
	ufbxwi_dev_assert(size % 4 == 0 && (uintptr_t)a % 4 == 0 && (uintptr_t)b % 4 == 0);
	char *ca = (char*)a, *cb = (char*)b;
	for (size_t i = 0; i < size; i += 4, ca += 4, cb += 4) {
		ufbxwi_aliasing_u32 *ua = (ufbxwi_aliasing_u32*)ca, *ub = (ufbxwi_aliasing_u32*)cb;
		uint32_t va = *ua, vb = *ub;
		*ua = vb;
		*ub = va;
	}
#else
	union {
		void *align_ptr;
		uintptr_t align_uptr;
		uint64_t align_u64;
		char data[256];
	} tmp;
	ufbxwi_dev_assert(size <= sizeof(tmp));
	memcpy(tmp.data, a, size);
	memcpy(a, b, size);
	memcpy(b, tmp.data, size);
#endif
}

// Stable sort array `m_type m_data[m_size]` using the predicate `m_cmp_lambda(a, b)`
// `m_linear_size` is a hint for how large blocks handle initially do with insertion sort
// `m_tmp` must be a memory buffer with at least the same size and alignment as `m_data`
#define ufbxwwi_macro_stable_sort(m_type, m_linear_size, m_data, m_tmp, m_size, m_cmp_lambda) do { \
	typedef m_type mi_type; \
	mi_type *mi_src = (mi_type*)(m_tmp); \
	mi_type *mi_data = m_data, *mi_dst = mi_data; \
	size_t mi_block_size = m_linear_size, mi_size = m_size; \
	/* Insertion sort in `m_linear_size` blocks */ \
	for (size_t mi_base = 0; mi_base < mi_size; mi_base += mi_block_size) { \
		size_t mi_i_end = mi_base + mi_block_size; \
		if (mi_i_end > mi_size) mi_i_end = mi_size; \
		for (size_t mi_i = mi_base + 1; mi_i < mi_i_end; mi_i++) { \
			size_t mi_j = mi_i; \
			mi_src[0] = mi_dst[mi_i]; \
			for (; mi_j != mi_base; --mi_j) { \
				mi_type *a = &mi_src[0], *b = &mi_dst[mi_j - 1]; \
				if (!( m_cmp_lambda )) break; \
				mi_dst[mi_j] = mi_dst[mi_j - 1]; \
			} \
			mi_dst[mi_j] = mi_src[0]; \
		} \
	} \
	/* Merge sort ping-ponging between `m_data` and `m_tmp` */ \
	for (; mi_block_size < mi_size; mi_block_size *= 2) { \
		mi_type *mi_swap = mi_dst; mi_dst = mi_src; mi_src = mi_swap; \
		for (size_t mi_base = 0; mi_base < mi_size; mi_base += mi_block_size * 2) { \
			size_t mi_i = mi_base, mi_i_end = mi_base + mi_block_size; \
			size_t mi_j = mi_i_end, mi_j_end = mi_j + mi_block_size; \
			size_t mi_k = mi_base; \
			if (mi_i_end > mi_size) mi_i_end = mi_size; \
			if (mi_j_end > mi_size) mi_j_end = mi_size; \
			while ((mi_i < mi_i_end) & (mi_j < mi_j_end)) { \
				mi_type *a = &mi_src[mi_j], *b = &mi_src[mi_i]; \
				if ( m_cmp_lambda ) { \
					mi_dst[mi_k] = *a; mi_j++; \
				} else { \
					mi_dst[mi_k] = *b; mi_i++; \
				} \
				mi_k++; \
			} \
			while (mi_i < mi_i_end) mi_dst[mi_k++] = mi_src[mi_i++]; \
			while (mi_j < mi_j_end) mi_dst[mi_k++] = mi_src[mi_j++]; \
		} \
	} \
	/* Copy the result to `m_data` if we ended up in `m_tmp` */ \
	if (mi_dst != mi_data) memcpy((void*)mi_data, mi_dst, sizeof(mi_type) * mi_size); \
	} while (0)

static ufbxwi_noinline void ufbxwi_unstable_sort(void *in_data, size_t size, size_t stride, ufbxwi_less_fn *less_fn, void *less_user)
{
	if (size <= 1) return;

	char *data = (char*)in_data;
	size_t start = (size - 1) >> 1;
	size_t end = size - 1;
	for (;;) {
		size_t root = start;
		size_t child;
		while ((child = root*2 + 1) <= end) {
			size_t next = less_fn(less_user, data + child * stride, data + root * stride) ? root : child;
			if (child + 1 <= end && less_fn(less_user, data + next * stride, data + (child + 1) * stride)) {
				next = child + 1;
			}
			if (next == root) break;
			ufbxwi_swap(data + root * stride, data + next * stride, stride);
			root = next;
		}

		if (start > 0) {
			start--;
		} else if (end > 0) {
			ufbxwi_swap(data + end * stride, data, stride);
			end--;
		} else {
			break;
		}
	}
}

// WARNING: Evaluates `m_list` twice!
#define ufbxwi_for_list(m_type, m_name, m_list) for (m_type *m_name = (m_list).data, *m_name##_end = ufbxwi_add_ptr(m_name, (m_list).count); m_name != m_name##_end; m_name++)
#define ufbxwi_for_ptr_list(m_type, m_name, m_list) for (m_type **m_name = (m_list).data, **m_name##_end = ufbxwi_add_ptr(m_name, (m_list).count); m_name != m_name##_end; m_name++)
#define ufbxwi_for_id_list(m_type, m_name, m_list) for (m_type *m_name##_it = (m_list).data, *m_name##_end = ufbxwi_add_ptr(m_name##_it, (m_list).count), m_name; (m_name##_it != m_name##_end) && (m_name = *m_name##_it, true); m_name##_it++)

#define ufbxwi_array_list(arr) { arr, ufbxwi_arraycount(arr) }

typedef struct {
	const void *data;
	size_t count;
} ufbxwi_void_span;

typedef struct {
	void *data;
	size_t count;
} ufbxwi_mutable_void_span;

typedef uint32_t ufbxwi_task_id;

// -- Atomics

#ifdef UFBXWI_FEATURE_ATOMICS

#define UFBXWI_THREAD_SAFE 1

#if defined(__cplusplus)
	#define ufbxwi_extern_c extern "C"
#else
	#define ufbxwi_extern_c
#endif

#if !defined(UFBXW_STANDARD_C) && (defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER))
	// TODO(threads): ARM/x86/etc
	#if defined(__x86_64__)
		#include <emmintrin.h>
		#define UFBXWI_HAS_MM_PAUSE
	#endif

	typedef struct { uint32_t value; } ufbxwi_atomic_u32;

	static ufbxwi_forceinline bool ufbxwi_atomic_cas(ufbxwi_atomic_u32 *atomic, uint32_t expected, uint32_t value)
	{
		return __sync_bool_compare_and_swap(&atomic->value, expected, value);
	}

	static ufbxwi_forceinline uint32_t ufbxwi_atomic_add(ufbxwi_atomic_u32 *atomic, uint32_t value)
	{
		return __sync_fetch_and_add(&atomic->value, value);
	}

	static ufbxwi_forceinline uint32_t ufbxwi_atomic_load_relaxed(ufbxwi_atomic_u32 *atomic)
	{
		return atomic->value;
	}

	static void ufbxwi_atomic_pause(void)
	{
		#ifdef UFBXWI_HAS_MM_PAUSE
			_mm_pause();
		#endif
	}
#elif !defined(UFBXW_STANDARD_C) && defined(_MSC_VER)
	#if defined(_M_X64)
		#include <emmintrin.h>
		#define UFBXWI_HAS_MM_PAUSE
	#endif

	typedef struct { uint32_t value; } ufbxwi_atomic_u32;
	ufbxwi_extern_c long _InterlockedCompareExchange(long volatile *dst, long value, long expected);
	ufbxwi_extern_c long _InterlockedExchangeAdd(long volatile *dst, long value);

	static ufbxwi_forceinline bool ufbxwi_atomic_cas(ufbxwi_atomic_u32 *atomic, uint32_t expected, uint32_t value)
	{
		long ref = _InterlockedCompareExchange((volatile long*)&atomic->value, (long)value, (long)expected);
		return ref == expected;
	}

	static ufbxwi_forceinline uint32_t ufbxwi_atomic_add(ufbxwi_atomic_u32 *atomic, uint32_t value)
	{
		return _InterlockedExchangeAdd((volatile long*)&atomic->value, (long)value);
	}

	static ufbxwi_forceinline uint32_t ufbxwi_atomic_load_relaxed(ufbxwi_atomic_u32 *atomic)
	{
		return atomic->value;
	}

	static void ufbxwi_atomic_pause(void)
	{
		#ifdef UFBXWI_HAS_MM_PAUSE
			_mm_pause();
		#endif
	}
#else
	#error TODO: Atomic fallbacks
#endif

// Some primitives built with the above

static ufbxwi_forceinline uint32_t ufbxwi_atomic_load_acquire(ufbxwi_atomic_u32 *atomic)
{
	// TODO: This could be optimized
	return ufbxwi_atomic_add(atomic, 0);
}

static ufbxwi_forceinline uint32_t ufbxwi_atomic_sub(ufbxwi_atomic_u32 *atomic, uint32_t value)
{
	return ufbxwi_atomic_add(atomic, (uint32_t)-(int32_t)value);
}

static ufbxwi_forceinline void ufbxwi_atomic_store(ufbxwi_atomic_u32 *atomic, uint32_t value)
{
	uint32_t prev = ufbxwi_atomic_load_relaxed(atomic);
	bool ok = ufbxwi_atomic_cas(atomic, prev, value);
	ufbxw_assert(ok);
}

#endif

// -- Synchronization

// TODO(threads): Rename this
#ifdef UFBXWI_FEATURE_THREAD_POOL

typedef struct {
	bool enabled;

	ufbxw_thread_sync_wait_fn *wait_fn;
	ufbxw_thread_sync_notify_fn *notify_fn;
	ufbxw_thread_sync_free_fn *free_fn;
	void *user;
	void *ctx;

} ufbxwi_thread_pool;

static bool ufbxwi_thread_pool_init(ufbxwi_thread_pool *tp, const ufbxw_thread_sync *sync)
{
	void *ctx = sync->init_fn(sync->user);
	if (!ctx) {
		return false;
	}

	tp->enabled = true;
	tp->wait_fn = sync->wait_fn;
	tp->notify_fn = sync->notify_fn;
	tp->free_fn = sync->free_fn;
	tp->user = sync->user;
	tp->ctx = ctx;
	return true;
}

static void ufbxwi_thread_pool_free(ufbxwi_thread_pool *tp)
{
	if (tp->free_fn) {
		tp->free_fn(tp->user, tp->ctx);
	}
}

static void ufbxwi_atomic_wait(ufbxwi_thread_pool *tp, ufbxwi_atomic_u32 *p_value, uint32_t ref_value)
{
	if (ufbxwi_atomic_load_acquire(p_value) == ref_value) {
		tp->wait_fn(tp->user, tp->ctx, &p_value->value, ref_value);
	}
}

static void ufbxwi_atomic_notify(ufbxwi_thread_pool *tp, ufbxwi_atomic_u32 *p_value, uint32_t wake_count)
{
	tp->notify_fn(tp->user, tp->ctx, &p_value->value, wake_count);
}

typedef struct {
	// [0:1]   locked
	// [1:32]  waiters
	ufbxwi_atomic_u32 lockers;
} ufbxwi_mutex;

static bool ufbxwi_mutex_try_lock(ufbxwi_thread_pool *tp, ufbxwi_mutex *mutex)
{
	ufbxwi_dev_assert(tp->enabled);

	// Happy fast path
	if (ufbxwi_atomic_cas(&mutex->lockers, 0, 1)) return true;

	for (uint32_t spin = 0; ; spin++) {
		const uint32_t state = ufbxwi_atomic_load_relaxed(&mutex->lockers);
		const uint32_t locked = state & 0x1;
		const uint32_t waiters = state >> 1u;

		if (!locked) {
			// Unlocked -> locked
			const uint32_t new_state = 0x1 | waiters << 1u;
			if (ufbxwi_atomic_cas(&mutex->lockers, state, new_state)) {
				return true;
			}
		} else {
			return false;
		}
	}
}

static void ufbxwi_mutex_lock(ufbxwi_thread_pool *tp, ufbxwi_mutex *mutex)
{
	ufbxwi_dev_assert(tp->enabled);

	// Happy fast path
	if (ufbxwi_atomic_cas(&mutex->lockers, 0, 1)) return;

	for (uint32_t spin = 0; ; spin++) {
		const uint32_t state = ufbxwi_atomic_load_relaxed(&mutex->lockers);
		const uint32_t locked = state & 0x1;
		const uint32_t waiters = state >> 1u;

		if (!locked) {
			// Unlocked -> locked
			const uint32_t new_state = 0x1 | waiters << 1u;
			if (ufbxwi_atomic_cas(&mutex->lockers, state, new_state)) {
				return;
			}
		} else if (spin < 100) {
			ufbxwi_atomic_pause();
		} else {
			// Add waiter
			const uint32_t new_waiters = waiters + 1u;
			const uint32_t new_state = 0x1 | new_waiters << 1u;
			if (ufbxwi_atomic_cas(&mutex->lockers, state, new_state)) {
				ufbxwi_atomic_wait(tp, &mutex->lockers, new_state);
				ufbxwi_atomic_sub(&mutex->lockers, 1u << 1u);
			}
		}
	}
}

static void ufbxwi_mutex_unlock(ufbxwi_thread_pool *tp, ufbxwi_mutex *mutex)
{
	ufbxwi_dev_assert(tp->enabled);

	// Happy fast path
	if (ufbxwi_atomic_cas(&mutex->lockers, 1, 0)) return;

	for (;;) {
		const uint32_t state = ufbxwi_atomic_load_relaxed(&mutex->lockers);
		const uint32_t locked = state & 0x1;
		const uint32_t waiters = state >> 1u;

		ufbxwi_dev_assert(locked);

		const uint32_t new_state = 0x0 | waiters << 1u;
		if (ufbxwi_atomic_cas(&mutex->lockers, state, new_state)) {
			if (waiters > 0) {
				ufbxwi_atomic_notify(tp, &mutex->lockers, 1u);
			}
			return;
		}
	}
}

static ufbxwi_forceinline void ufbxwi_mutex_lock_if_enabled(ufbxwi_thread_pool *tp, ufbxwi_mutex *mutex)
{
	if (tp) {
		ufbxwi_mutex_lock(tp, mutex);
	}
}

static ufbxwi_forceinline void ufbxwi_mutex_unlock_if_enabled(ufbxwi_thread_pool *tp, ufbxwi_mutex *mutex)
{
	if (tp) {
		ufbxwi_mutex_unlock(tp, mutex);
	}
}

typedef struct {
	ufbxwi_atomic_u32 count;
} ufbxwi_semaphore;

static bool ufbxwi_semaphore_try_wait(ufbxwi_thread_pool *tp, ufbxwi_semaphore *sema)
{
	for (;;) {
		const uint32_t count = ufbxwi_atomic_load_relaxed(&sema->count);
		if (count == 0) return false;

		if (ufbxwi_atomic_cas(&sema->count, count, count - 1)) {
			return true;
		}
	}
}

static void ufbxwi_semaphore_wait(ufbxwi_thread_pool *tp, ufbxwi_semaphore *sema)
{
	for (uint32_t spin = 0;; spin++) {
		const uint32_t count = ufbxwi_atomic_load_relaxed(&sema->count);
		if (count > 0) {
			if (ufbxwi_atomic_cas(&sema->count, count, count - 1)) {
				return;
			}
		} else if (spin < 100) {
			ufbxwi_atomic_pause();
			continue;
		}

		ufbxwi_atomic_wait(tp, &sema->count, 0);
	}
}

static void ufbxwi_semaphore_notify(ufbxwi_thread_pool *tp, ufbxwi_semaphore *sema, uint32_t count)
{
	if (count == 0) return;

	// TODO: This could be optimized
	ufbxwi_atomic_add(&sema->count, count);
	ufbxwi_atomic_notify(tp, &sema->count, count);
}
#endif

// -- Error

#ifdef UFBXWI_FEATURE_ERROR

static const char ufbxwi_empty_char[] = "";
static const ufbxw_string ufbxwi_empty_string = { ufbxwi_empty_char, 0 };

static ufbxw_string ufbxwi_c_str(const char *str)
{
	if (str == NULL) str = ufbxwi_empty_char;
	ufbxw_string s = { str, strlen(str) };
	return s;
}

typedef struct ufbxwi_error ufbxwi_error;

typedef void ufbxwi_fatal_fn(void *user, ufbxwi_error *error);

struct ufbxwi_error {
	ufbxw_error error;

	ufbxwi_thread_pool *thread_pool;
	ufbxwi_mutex mutex;

	ufbxw_error_fn *error_fn;
	void *error_user;

	ufbxwi_fatal_fn *fatal_fn;
	void *fatal_user;
};

static ufbxwi_noinline void ufbxwi_failf_imp(ufbxwi_error *error, ufbxw_error_type type, const char *func, const char *fmt, ...)
{
	// Don't report errors if we have already failed fatally
	if (error->error.type >= UFBXW_ERROR_FATAL) return;

	if (type < UFBXW_ERROR_FATAL) {
		if (error->error_fn) {
			va_list args;
			va_start(args, fmt);

			ufbxw_error err;
			err.type = type;
			err.function = ufbxwi_c_str(func);

			int len = vsnprintf(err.description, sizeof(err.description), fmt, args);
			err.description_length = (size_t)len;

			va_end(args);

			error->error_fn(error->error_user, &err);
		}
		return;
	}

	ufbxwi_mutex_lock_if_enabled(error->thread_pool, &error->mutex);
	if (error->error.type >= UFBXW_ERROR_FATAL) {
		ufbxwi_mutex_unlock_if_enabled(error->thread_pool, &error->mutex);
		return;
	}

	error->error.type = type;
	error->error.function = ufbxwi_c_str(func);

	va_list args;
	va_start(args, fmt);
	int desc_len = vsnprintf(error->error.description, sizeof(error->error.description), fmt, args);
	va_end(args);
	error->error.description_length = (size_t)desc_len;

	if (error->error_fn) {
		error->error_fn(error->error_user, &error->error);
	}

	if (error->fatal_fn) {
		error->fatal_fn(error->fatal_user, error);
	}

	ufbxwi_mutex_unlock_if_enabled(error->thread_pool, &error->mutex);
}

static ufbxwi_noinline void ufbxwi_fail_imp(ufbxwi_error *error, ufbxw_error_type type, const char *func, const char *desc)
{
	ufbxwi_failf_imp(error, type, func, "%s", desc);
}

#define ufbxwi_failf(error, type, ...) ufbxwi_failf_imp((error), (type), ufbxwi_func, __VA_ARGS__)
#define ufbxwi_fail(error, type, desc) ufbxwi_fail_imp((error), (type), ufbxwi_func, (desc))

static ufbxwi_forceinline bool ufbxwi_is_fatal(ufbxwi_error *error)
{
	return error->error.type != UFBXW_ERROR_NONE;
}

#define ufbxwi_check(cond, ...) do { if (!(cond)) return __VA_ARGS__; } while (0)

#define ufbxwi_check_index(error, index, count, ...) do { \
		size_t mi_index = (index), mi_count = (count); \
		if (mi_index >= mi_count) { \
			ufbxwi_failf((error), UFBXW_ERROR_INDEX_OUT_OF_BOUNDS, "index (%zu) out of bounds (%zu)", mi_index, mi_count); \
			return __VA_ARGS__; \
		} \
	} while (0)

#endif

// -- Allocator

#ifdef UFBXWI_FEATURE_ALLOCATOR

#if defined(UFBXW_REGRESSION)
static const char ufbxwi_zero_size_buffer[4096] = { 0 };
#else
static const char ufbxwi_zero_size_buffer[64] = { 0 };
#endif

#ifndef UFBXW_MAXIMUM_ALIGNMENT
enum { UFBXW_MAXIMUM_ALIGNMENT = sizeof(void*) > 8 ? sizeof(void*) : 8 };
#endif

static ufbxwi_forceinline size_t ufbxwi_align_to_mask(size_t value, size_t align_mask)
{
	return value + (((size_t)0 - value) & align_mask);
}

static ufbxwi_forceinline size_t ufbxwi_size_align_mask(size_t size)
{
	// Align to the all bits below the lowest set one in `size` up to the maximum alignment.
	return ((size ^ (size - 1)) >> 1) & (UFBXW_MAXIMUM_ALIGNMENT - 1);
}

static ufbxwi_forceinline size_t ufbxwi_align(size_t value, size_t align)
{
	return value + (((size_t)0 - value) & (align - 1));
}

enum {
	UFBXWI_MIN_SIZE_CLASS_LOG2 = 6,
	UFBXWI_MIN_SIZE_CLASS_SIZE = 1 << UFBXWI_MIN_SIZE_CLASS_LOG2,
	UFBXWI_MAX_SIZE_CLASS_LOG2 = 12,
	UFBXWI_MAX_SIZE_CLASS_SIZE = 1 << UFBXWI_MAX_SIZE_CLASS_LOG2,
	UFBXWI_SIZE_CLASS_COUNT = UFBXWI_MAX_SIZE_CLASS_LOG2 - UFBXWI_MIN_SIZE_CLASS_LOG2 + 1,
};

static uint32_t ufbxwi_get_size_class(size_t size)
{
	if (size > UFBXWI_MAX_SIZE_CLASS_SIZE) {
		return UINT32_MAX;
	} else if (size <= UFBXWI_MIN_SIZE_CLASS_SIZE) {
		return 0;
	} else {
		return (32 - ufbxwi_lzcnt32((uint32_t)size - 1)) - UFBXWI_MIN_SIZE_CLASS_LOG2;
	}
}

#define UFBXWI_HUGE_MAGIC  0x68617775
#define UFBXWI_BLOCK_MAGIC 0x62617775u
#define UFBXWI_ALLOC_MAGIC 0x61617775u
#define UFBXWI_FREED_MAGIC 0x66617775u

typedef struct ufbxwi_alloc ufbxwi_alloc;
struct ufbxwi_alloc {
	ufbxwi_alloc *prev;
	ufbxwi_alloc *next;
	size_t size;
	size_t magic;
};

typedef struct {
	ufbxwi_error *error;
	ufbxw_allocator ator;

	ufbxwi_thread_pool *thread_pool;
	ufbxwi_mutex mutex;

	size_t num_allocs;
	size_t max_allocs;
	size_t total_size;
	size_t max_size;
	size_t num_block_allocs;

	ufbxwi_alloc *free_root[UFBXWI_SIZE_CLASS_COUNT];
	ufbxwi_alloc block_root;

	void *current_block;
	size_t current_pos;
	size_t current_size;

	size_t next_block_size;
} ufbxwi_allocator;

static void ufbxwi_mark_allocator_failed(ufbxwi_allocator *ator)
{
	// Do not allow any further allocations
	ator->max_allocs = 0;
}

static ufbxwi_forceinline bool ufbxwi_does_overflow(size_t total, size_t a, size_t b)
{
	// If `a` and `b` have at most 4 bits per `size_t` byte, the product can't overflow.
	if (((a | b) >> sizeof(size_t)*4) != 0) {
		if (a != 0 && total / a != b) return true;
	}
	return false;
}

static ufbxwi_noinline ufbxwi_alloc *ufbxwi_alloc_block(ufbxwi_allocator *ator, size_t size)
{
	size_t alloc_size = sizeof(ufbxwi_alloc) + size;
	if (ator->max_size - ator->total_size < alloc_size) {
		ufbxwi_failf(ator->error, UFBXW_ERROR_MEMORY_LIMIT, "Memory limit exceeded (%zu bytes)", ator->max_size);
		return NULL;
	}

	ufbxwi_alloc *block = NULL;
	if (ator->ator.alloc_fn) {
		block = (ufbxwi_alloc*)ator->ator.alloc_fn(ator->ator.user, size);
	} else {
		block = (ufbxwi_alloc*)ufbxw_malloc(alloc_size);
	}
	if (!block) {
		ufbxwi_failf(ator->error, UFBXW_ERROR_ALLOCATION_FAILURE, "Failed to allocate %zu bytes", alloc_size);
		return NULL;
	}

	ator->total_size += alloc_size;
	ator->num_block_allocs++;

	block->size = size;
	block->prev = &ator->block_root;
	block->next = ator->block_root.next;
	if (block->next) block->next->prev = block;
	ator->block_root.next = block;

	return block;
}

static ufbxwi_noinline void ufbxwi_free_block(ufbxwi_allocator *ator, ufbxwi_alloc *block)
{
	if (block->prev) block->prev->next = block->next;
	if (block->next) block->next->prev = block->prev;

	size_t alloc_size = block->size + sizeof(ufbxwi_alloc);
	ator->total_size -= alloc_size;

	block->magic = UFBXWI_FREED_MAGIC;
	if (ator->ator.alloc_fn) {
		if (ator->ator.free_fn) {
			ator->ator.free_fn(ator->ator.user, block, alloc_size);
		}
	} else {
		ufbxw_free(block, alloc_size);
	}
}

static ufbxwi_noinline void *ufbxwi_alloc_size(ufbxwi_allocator *ator, size_t size, size_t n, size_t *p_alloc_size)
{
	// Always succeed with an empty non-NULL buffer for empty allocations
	ufbxw_assert(size > 0);
	if (n == 0) {
		if (p_alloc_size) *p_alloc_size = 0;
		return (void*)ufbxwi_zero_size_buffer;
	}

	size_t total = size * n;
	if (ufbxwi_does_overflow(total, size, n)) {
		ufbxwi_fail(ator->error, UFBXW_ERROR_ALLOCATION_FAILURE, "Allocation size overflow");
		return NULL;
	}
	if (ator->num_allocs >= ator->max_allocs) {
		ufbxwi_failf(ator->error, UFBXW_ERROR_ALLOCATION_LIMIT, "Allocation limit exceeded (%zu)", ator->max_allocs);
		return NULL;
	}

	ufbxwi_mutex_lock_if_enabled(ator->thread_pool, &ator->mutex);

	ator->num_allocs++;
	ator->total_size += total;

	uint32_t size_class = ufbxwi_get_size_class(total);
	if (size_class == ~0u) {
		ufbxwi_alloc *block = ufbxwi_alloc_block(ator, total);
		ufbxwi_check(block, NULL);
		block->magic = UFBXWI_HUGE_MAGIC;

		if (p_alloc_size) {
			*p_alloc_size = total;
		}

		ufbxwi_mutex_unlock_if_enabled(ator->thread_pool, &ator->mutex);
		return block + 1;
	}

	size_t size_class_size = (size_t)UFBXWI_MIN_SIZE_CLASS_SIZE << size_class;
	ufbxwi_alloc *freed = ator->free_root[size_class];
	if (freed) {
		freed->magic = UFBXWI_ALLOC_MAGIC;
		ator->free_root[size_class] = freed->next;
		if (p_alloc_size) {
			*p_alloc_size = size_class_size;
		}

		ufbxwi_mutex_unlock_if_enabled(ator->thread_pool, &ator->mutex);
		return freed + 1;
	}

	size_t alloc_size = size_class_size + sizeof(ufbxwi_alloc);
	if (ufbxwi_unlikely(ator->current_size - ator->current_pos < alloc_size)) {
		size_t block_alloc_size = ufbxwi_min_sz(ufbxwi_max_sz(ator->next_block_size * 2, 0x10000), 0x100000);

		ufbxwi_alloc *block = ufbxwi_alloc_block(ator, block_alloc_size - sizeof(ufbxwi_alloc));
		ufbxwi_check(block, NULL);
		block->magic = UFBXWI_BLOCK_MAGIC;

		ator->current_block = block;
		ator->current_pos = sizeof(ufbxwi_alloc);
		ator->current_size = block_alloc_size;

		ator->next_block_size = block_alloc_size;
	}

	size_t offset = ator->current_pos;
	ufbxwi_alloc *alloc = (ufbxwi_alloc*)((char*)ator->current_block + offset);
	ator->current_pos = offset + alloc_size;
	alloc->next = NULL;
	alloc->prev = NULL;
	alloc->size = size_class_size;
	alloc->magic = UFBXWI_ALLOC_MAGIC;
	if (p_alloc_size) {
		*p_alloc_size = size_class_size;
	}

	ufbxwi_mutex_unlock_if_enabled(ator->thread_pool, &ator->mutex);
	return alloc + 1;
}

static ufbxwi_noinline void ufbxwi_free(ufbxwi_allocator *ator, void *ptr)
{
	if (!ptr || ptr == ufbxwi_zero_size_buffer) return;

	ufbxwi_mutex_lock_if_enabled(ator->thread_pool, &ator->mutex);

	ufbxwi_alloc *alloc = (ufbxwi_alloc*)ptr - 1;
	ufbxw_assert(alloc->magic == UFBXWI_ALLOC_MAGIC || alloc->magic == UFBXWI_HUGE_MAGIC);
	uint32_t size_class = ufbxwi_get_size_class(alloc->size);

	if (size_class == ~0u) {
		ufbxwi_free_block(ator, alloc);
	} else {
		alloc->magic = UFBXWI_FREED_MAGIC;
		alloc->next = ator->free_root[size_class];
		ator->free_root[size_class] = alloc;
	}

	ufbxwi_mutex_unlock_if_enabled(ator->thread_pool, &ator->mutex);
}

static ufbxwi_noinline void ufbxwi_move_allocator(ufbxwi_allocator *dst, ufbxwi_allocator *src)
{
	*dst = *src;
	if (dst->block_root.next) {
		dst->block_root.next->prev = &dst->block_root;
	}
	memset(src, 0, sizeof(ufbxwi_allocator));
}

static ufbxwi_noinline void ufbxwi_free_allocator(ufbxwi_allocator *ator)
{
	ufbxwi_alloc *block = ator->block_root.next;
	while (block) {
		ufbxw_assert(block->magic == UFBXWI_BLOCK_MAGIC || block->magic == UFBXWI_HUGE_MAGIC);

		ufbxwi_alloc *next = block->next;
		ufbxwi_free_block(ator, block);
		block = next;
	}

	if (ator->ator.free_allocator_fn) {
		ator->ator.free_allocator_fn(ator->ator.user);
	}
}

static void *ufbxwi_alloc_zero_size(ufbxwi_allocator *ator, size_t size, size_t n)
{
	void *ptr = ufbxwi_alloc_size(ator, size, n, NULL);
	if (ptr) {
		memset(ptr, 0, size * n);
	}
	return ptr;
}

#define ufbxwi_alloc(ator, type, n) ufbxwi_maybe_null((type*)ufbxwi_alloc_size((ator), sizeof(type), (n), NULL))
#define ufbxwi_alloc_zero(ator, type, n) ufbxwi_maybe_null((type*)ufbxwi_alloc_zero_size((ator), sizeof(type), (n)))

#endif

// -- Dynamic List

#ifdef UFBXWI_FEATURE_LIST

typedef struct {
	void *data;
	size_t count;
	size_t capacity;
} ufbxwi_list;

#define UFBXWI_LIST_TYPE(p_name, p_type) typedef struct p_name { p_type *data; size_t count, capacity; } p_name

static ufbxwi_noinline void *ufbxwi_list_push_size_slow(ufbxwi_allocator *ator, ufbxwi_list *list, size_t size, size_t n)
{
	size_t count = list->count;
	if (list->capacity - count >= n) {
		if (list->data == NULL) list->data = (void*)ufbxwi_zero_size_buffer;
		list->count = count + n;
		return (char*)list->data + size * count;
	}

	size_t new_capacity = ufbxwi_max_sz(count + n, list->capacity * 2);

	size_t alloc_size = 0;
	char *new_data = (char*)ufbxwi_alloc_size(ator, size, new_capacity, &alloc_size);
	ufbxwi_check(new_data, NULL);

	memcpy(new_data, list->data, count * size);
	ufbxwi_free(ator, list->data);

	list->data = new_data;
	list->capacity = alloc_size / size;
	list->count += n;
	return (char*)list->data + size * count;
}

static ufbxwi_forceinline void *ufbxwi_list_push_size(ufbxwi_allocator *ator, void *p_list, size_t size, size_t n)
{
	ufbxwi_list *list = (ufbxwi_list*)p_list;
	size_t count = list->count;
	// TODO: Something better here.. always taking two slow paths in a row to patch non-NULL data pointer
	if (list->capacity - count > n) {
		list->count = count + n;
		return (char*)list->data + size * count;
	} else {
		return ufbxwi_list_push_size_slow(ator, list, size, n);
	}
}

static ufbxwi_noinline bool ufbxwi_list_resize_size_slow(ufbxwi_allocator *ator, ufbxwi_list *list, size_t size, size_t n)
{
	size_t new_capacity = ufbxwi_max_sz(n, list->capacity * 2);

	size_t alloc_size = 0;
	char *new_data = (char*)ufbxwi_alloc_size(ator, size, new_capacity, &alloc_size);
	ufbxwi_check(new_data, false);

	memcpy(new_data, list->data, list->count * size);

	list->data = new_data;
	list->capacity = alloc_size / size;
	list->count = n;
	return true;
}

static ufbxwi_forceinline bool ufbxwi_list_resize_size(ufbxwi_allocator *ator, void *p_list, size_t size, size_t n)
{
	ufbxwi_list *list = (ufbxwi_list*)p_list;
	if (list->capacity >= n) {
		list->count = n;
		return true;
	} else {
		return ufbxwi_list_resize_size_slow(ator, list, size, n);
	}
}

static ufbxwi_forceinline void *ufbxwi_list_push_zero_size(ufbxwi_allocator *ator, void *p_list, size_t size, size_t n)
{
	void *data = ufbxwi_list_push_size(ator, p_list, size, n);
	if (!data) return NULL;
	memset(data, 0, size * n);
	return data;
}

static ufbxwi_forceinline void *ufbxwi_list_push_copy_size(ufbxwi_allocator *ator, void *p_list, size_t size, size_t n, const void *src)
{
	void *data = ufbxwi_list_push_size(ator, p_list, size, n);
	if (!data) return NULL;
	memcpy(data, src, size * n);
	return data;
}

static ufbxwi_forceinline void ufbxwi_list_free_size(ufbxwi_allocator *ator, void *p_list, size_t size)
{
	ufbxwi_list *list = (ufbxwi_list*)p_list;
	ufbxwi_free(ator, list->data);
	list->data = NULL;
	list->count = list->capacity = 0;
}

#if UFBXW_DEV
	#define ufbxwi_check_ptr_type(type, expr) ((void)(sizeof((type*)NULL - (expr))))
#else
	#define ufbxwi_check_ptr_type(type, expr) ((void)0)
#endif

#define ufbxwi_list_push_uninit(ator, list, type) ufbxwi_maybe_null((ufbxwi_check_ptr_type(type, (list)->data), (type*)ufbxwi_list_push_size((ator), (list), sizeof(type), 1)))
#define ufbxwi_list_push_uninit_n(ator, list, type, n) ufbxwi_maybe_null((ufbxwi_check_ptr_type(type, (list)->data), (type*)ufbxwi_list_push_size((ator), (list), sizeof(type), (n))))
#define ufbxwi_list_push_zero(ator, list, type) ufbxwi_maybe_null((ufbxwi_check_ptr_type(type, (list)->data), (type*)ufbxwi_list_push_zero_size((ator), (list), sizeof(type), 1)))
#define ufbxwi_list_push_zero_n(ator, list, type, n) ufbxwi_maybe_null((ufbxwi_check_ptr_type(type, (list)->data), (type*)ufbxwi_list_push_zero_size((ator), (list), sizeof(type), (n))))
#define ufbxwi_list_push_copy(ator, list, type, src) ufbxwi_maybe_null((ufbxwi_check_ptr_type(type, (list)->data), ufbxwi_check_ptr_type(type, src), (type*)ufbxwi_list_push_copy_size((ator), (list), sizeof(type), 1, (src))))
#define ufbxwi_list_push_copy_n(ator, list, type, n, src) ufbxwi_maybe_null((ufbxwi_check_ptr_type(type, (list)->data), ufbxwi_check_ptr_type(type, src), (type*)ufbxwi_list_push_copy_size((ator), (list), sizeof(type), (n), (src))))
#define ufbxwi_list_resize_uninit(ator, list, type, n) (ufbxwi_check_ptr_type(type, (list)->data), (type*)ufbxwi_list_resize_size((ator), (list), sizeof(type), (n)))
#define ufbxwi_list_free(ator, list) ufbxwi_list_free_size((ator), (list), sizeof(*(list)->data))


// -- Special list

UFBXWI_LIST_TYPE(ufbxwi_id_list, ufbxw_id);
UFBXW_LIST_TYPE(ufbxwi_id_span, ufbxw_id);

static bool ufbxwi_id_list_add(ufbxwi_allocator *ator, void *p_list, ufbxw_id id)
{
	ufbxwi_id_list *list = (ufbxwi_id_list*)p_list;
	ufbxw_id *dst = ufbxwi_list_push_uninit(ator, list, ufbxw_id);
	if (dst) {
		*dst = id;
		return true;
	} else {
		return false;
	}
}

static bool ufbxwi_id_list_remove_one(void *p_list, ufbxw_id id)
{
	ufbxwi_id_list *list = (ufbxwi_id_list*)p_list;
	ufbxw_id *begin = list->data, *dst = begin, *end = begin + list->count;
	for (; dst != end; dst++) {
		if (*dst == id) break;
	}
	if (dst == end) return false;

	--list->count;
	--end;
	for (; dst != end; dst++) {
		dst[0] = dst[1];
	}
	return true;
}

// Some common list types
UFBXWI_LIST_TYPE(ufbxwi_uint32_list, uint32_t);
UFBXWI_LIST_TYPE(ufbxwi_ktime_list, ufbxw_ktime);
UFBXWI_LIST_TYPE(ufbxwi_real_list, ufbxw_real);
UFBXWI_LIST_TYPE(ufbxwi_float_list, float);
UFBXWI_LIST_TYPE(ufbxwi_byte_list, char);

#endif

// -- Task queue

#ifdef UFBXWI_FEATURE_TASK_QUEUE

typedef enum {
	UFBXWI_RUN_TASK_TRY,
	UFBXWI_RUN_TASK_BLOCKING,
} ufbxwi_run_task_mode;

typedef bool ufbxwi_task_fn(void *user, void *thread_ctx);

typedef void *ufbxwi_create_thread_ctx_fn(void *user);
typedef void ufbxwi_free_thread_ctx_fn(void *user, void *thread_ctx);

typedef struct {
	ufbxwi_create_thread_ctx_fn *create_thread_ctx_fn;
	ufbxwi_free_thread_ctx_fn *free_thread_ctx_fn;
	void *thread_ctx_user;

	uint32_t max_tasks;
	uint32_t num_threads;
} ufbxwi_task_queue_opts;

typedef struct {
	ufbxwi_task_fn *fn;
	void *user;
} ufbxwi_task;

typedef struct {
	ufbxwi_mutex mutex;
	ufbxwi_atomic_u32 generation;
	ufbxwi_task task;
} ufbxwi_task_slot;

typedef struct {
	ufbxwi_atomic_u32 thread_id;
	void *thread_ctx;
} ufbxwi_thread_context;

typedef struct {
	ufbxwi_thread_pool *thread_pool;

	// The naming is really nasty here
	ufbxw_thread_pool user_pool;

	ufbxwi_create_thread_ctx_fn *create_thread_ctx_fn;
	ufbxwi_free_thread_ctx_fn *free_thread_ctx_fn;
	void *thread_ctx_user;

	ufbxwi_task_slot *slots;
	size_t num_slots;

	uint32_t write_index;

	ufbxwi_semaphore task_sema;
	ufbxwi_atomic_u32 run_index;
	bool completed;
	bool enabled;

	ufbxwi_mutex fail_mutex;
	bool failed;

	void *user_ptr;

	uint32_t num_thread_contexts;
	ufbxwi_thread_context *thread_contexts;

	size_t num_threads;
} ufbxwi_task_queue;

static ufbxwi_thread_context *ufbxwi_get_thread_context(ufbxwi_task_queue *tq, uint32_t thread_id_hint)
{
	// Remap so that zero is a sentinel value
	uint32_t id = thread_id_hint;
	if (id < UINT32_MAX) {
		id += 1;
	}

	// TODO: Make this better for non-contiguous IDs
	// TODO: Attempt to reuse previous IDs if possible
	uint32_t num_contexts = tq->num_thread_contexts;
	for (uint32_t scan = 0; scan <= num_contexts; scan++) {
		ufbxwi_thread_context *tc = &tq->thread_contexts[(id + scan) % num_contexts];
		if (ufbxwi_atomic_cas(&tc->thread_id, 0, id)) {
			if (!tc->thread_ctx) {
				tc->thread_ctx = tq->create_thread_ctx_fn(tq->thread_ctx_user);
			}
			return tc;
		}
	}

	return NULL;
}

static void ufbxwi_return_thread_context(ufbxwi_task_queue *tq, ufbxwi_thread_context *tc)
{
	ufbxwi_atomic_store(&tc->thread_id, 0);
}

static bool ufbxwi_task_complete(ufbxwi_task_queue *tq, ufbxwi_task_id task_id, void *thread_ctx, bool blocking)
{
	ufbxwi_dev_assert(tq->enabled);

    const uint32_t slot_ix = task_id % tq->num_slots;
    const uint32_t generation = task_id / tq->num_slots;
    ufbxwi_task_slot *slot = &tq->slots[slot_ix];

    if (ufbxwi_atomic_load_acquire(&slot->generation) > generation) {
        return false;
    }

	if (blocking) {
		ufbxwi_mutex_lock(tq->thread_pool, &slot->mutex);
	} else {
		if (!ufbxwi_mutex_try_lock(tq->thread_pool, &slot->mutex)) {
			return false;
		}
	}

	bool completed = false;
    if (ufbxwi_atomic_load_relaxed(&slot->generation) == generation) {
        if (tq->failed) {
            // Skip task
        } else if (slot->task.fn(slot->task.user, thread_ctx)) {
			completed = true;
		} else {
            // TODO: More descriptive failing
            ufbxwi_mutex_lock(tq->thread_pool, &tq->fail_mutex);
            tq->failed = true;
            ufbxwi_mutex_unlock(tq->thread_pool, &tq->fail_mutex);
        }
		ufbxwi_atomic_store(&slot->generation, generation + 1);
    }
    ufbxwi_mutex_unlock(tq->thread_pool, &slot->mutex);
    return completed;
}

static bool ufbxwi_task_get_completed(ufbxwi_task_queue *tq, ufbxwi_task_id task_id)
{
	ufbxwi_dev_assert(tq->enabled);

    const uint32_t slot_ix = task_id % tq->num_slots;
    const uint32_t generation = task_id / tq->num_slots;
    ufbxwi_task_slot *slot = &tq->slots[slot_ix];
    return ufbxwi_atomic_load_acquire(&slot->generation) > generation;
}

static ufbxw_task_run_result ufbxwi_task_queue_run_task_imp(ufbxwi_task_queue *tq, void *thread_ctx, ufbxwi_run_task_mode mode, size_t max_count)
{
	for (size_t i = 0; i < max_count; i++) {
		if (mode == UFBXWI_RUN_TASK_TRY) {
			if (!ufbxwi_semaphore_try_wait(tq->thread_pool, &tq->task_sema)) {
				return UFBXW_TASK_RUN_RESULT_NO_TASKS;
			}
		} else {
			ufbxwi_semaphore_wait(tq->thread_pool, &tq->task_sema);
		}

		if (tq->completed) {
			return UFBXW_TASK_RUN_RESULT_ALL_FINISHED;
		}

		uint32_t task_id = ufbxwi_atomic_add(&tq->run_index, 1);
		ufbxwi_task_complete(tq, task_id, thread_ctx, false);
	}
	return UFBXW_TASK_RUN_RESULT_COMPLETED;
}

static ufbxw_task_run_result ufbxwi_task_queue_run_task(ufbxwi_task_queue *tq, ufbxwi_run_task_mode mode, uint32_t thread_id_hint, size_t max_count)
{
	if (tq->completed) {
		return UFBXW_TASK_RUN_RESULT_ALL_FINISHED;
	}

	ufbxwi_thread_context *tc = ufbxwi_get_thread_context(tq, thread_id_hint);
	if (!tc) {
		return UFBXW_TASK_RUN_RESULT_FAILED;
	}

	ufbxw_task_run_result result = ufbxwi_task_queue_run_task_imp(tq, tc->thread_ctx, mode, max_count);
	ufbxwi_return_thread_context(tq, tc);

	return result;
}

static bool ufbxwi_task_queue_init(ufbxwi_task_queue *tq, ufbxwi_thread_pool *tp, ufbxwi_allocator *ator, const ufbxwi_task_queue_opts *opts, const ufbxw_thread_pool *pool)
{
	tq->enabled = true;

	tq->user_pool = *pool;
	tq->thread_pool = tp;
	tq->create_thread_ctx_fn = opts->create_thread_ctx_fn;
	tq->free_thread_ctx_fn = opts->free_thread_ctx_fn;
	tq->thread_ctx_user = opts->thread_ctx_user;

	size_t num_slots = 1;
	const size_t max_tasks = opts->max_tasks ? opts->max_tasks : 0x1000;
	while (num_slots < max_tasks) {
		num_slots *= 2;
	}

	tq->slots = ufbxwi_alloc_zero(ator, ufbxwi_task_slot, num_slots);
	if (!tq->slots) return false;

	tq->num_slots = num_slots;

	// Reserve task ID 0 for uninitialized
	tq->write_index = 1;
	ufbxwi_atomic_store(&tq->run_index, 1);
	ufbxwi_atomic_store(&tq->slots[0].generation, 1);

	tq->write_index = 1;

	const uint32_t num_threads = opts->num_threads;
	const uint32_t num_thread_contexts = num_threads * 2;
	tq->thread_contexts = ufbxwi_alloc_zero(ator, ufbxwi_thread_context, num_thread_contexts);
	if (!tq->thread_contexts) return false;

	tq->num_thread_contexts = num_thread_contexts;

	ufbxw_thread_pool_context ctx = (ufbxw_thread_pool_context)tq;
	if (!tq->user_pool.init_fn(tq->user_pool.user, ctx, num_threads)) {
		return false;
	}

	return true;
}

static ufbxwi_task_id ufbxwi_task_push(ufbxwi_task_queue *tq, ufbxwi_task_fn *fn, void *user, void *context)
{
	ufbxwi_dev_assert(tq->enabled);

	ufbxwi_task_id task_id = tq->write_index++;

	// Wait until the previous task in this slot is completed.
	if (task_id >= tq->num_slots) {
		ufbxwi_task_complete(tq, task_id - tq->num_slots, context, true);
	}

	const uint32_t slot_ix = task_id % tq->num_slots;
	ufbxwi_task_slot *slot = &tq->slots[slot_ix];
	slot->task.fn = fn;
	slot->task.user = user;

	ufbxwi_semaphore_notify(tq->thread_pool, &tq->task_sema, 1);

	// TODO: Batch these?
	if (tq->user_pool.run_fn) {
		tq->user_pool.run_fn(tq->user_pool.user, (ufbxw_thread_pool_context)tq, 1);
	}

	return task_id;
}

static void ufbxwi_task_queue_free(ufbxwi_task_queue *tq, void *context)
{
	if (!tq->enabled) return;

	// Wait that all tasks are completed
	// TODO: This could be optimized
	// TODO: We might not even need to do this
	uint32_t task_start = tq->write_index >= tq->num_slots ? tq->write_index - tq->num_slots : 0;
	if (task_start == 0) {
		task_start = 1;
	}
	for (uint32_t task_id = task_start; task_id < tq->write_index; task_id++) {
		ufbxwi_task_complete(tq, task_id, context, true);
	}

	if (!tq->completed) {
		tq->completed = true;
		ufbxwi_semaphore_notify(tq->thread_pool, &tq->task_sema, UINT32_MAX / 2);
	}

	if (tq->user_pool.free_fn) {
		tq->user_pool.free_fn(tq->user_pool.user, (ufbxw_thread_pool_context)tq);
	}

	ufbxwi_for(ufbxwi_thread_context, tc, tq->thread_contexts, tq->num_thread_contexts) {
		if (tc->thread_ctx) {
			tq->free_thread_ctx_fn(tq->thread_ctx_user, tc->thread_ctx);
			tc->thread_ctx = NULL;
		}
	}
}

#endif

// -- Hash functions

static ufbxwi_noinline uint32_t ufbxwi_hash_string(const char *str, size_t length)
{
	uint32_t hash = (uint32_t)length;
	uint32_t seed = UINT32_C(0x9e3779b9);
	if (length >= 4) {
		do {
			uint32_t word = ufbxwi_read_u32(str);
			hash = ((hash << 5u | hash >> 27u) ^ word) * seed;
			str += 4;
			length -= 4;
		} while (length >= 4);

		uint32_t word = ufbxwi_read_u32(str + length - 4);
		hash = ((hash << 5u | hash >> 27u) ^ word) * seed;
	} else {
		uint32_t word = 0;
		if (length >= 1) word |= (uint32_t)(uint8_t)str[0] << 0;
		if (length >= 2) word |= (uint32_t)(uint8_t)str[1] << 8;
		if (length >= 3) word |= (uint32_t)(uint8_t)str[2] << 16;
		hash = ((hash << 5u | hash >> 27u) ^ word) * seed;
	}
	hash ^= hash >> 16;
	hash *= UINT32_C(0x7feb352d);
	hash ^= hash >> 15;

	// Reserve 0 and 1 for empty/tombstone
	return hash >= 2 ? hash : 2;
}

// -- String

#ifdef UFBXWI_FEATURE_STRING_POOL

typedef struct ufbxwi_string_entry {
	uint32_t hash;
	uint32_t length;
	uint32_t token;
	const char *data;
} ufbxwi_string_entry;

UFBXWI_LIST_TYPE(ufbxw_string_list, ufbxw_string);

typedef struct ufbxwi_string_pool {
	ufbxwi_allocator *ator;
	ufbxwi_error *error;
	bool failed;

	ufbxwi_string_entry *entries;
	uint32_t entry_count;
	uint32_t entry_capacity;

	ufbxw_string_list tokens;

	char *block_pos;
	char *block_end;
} ufbxwi_string_pool;

static void ufbxwi_mark_string_pool_failed(ufbxwi_string_pool *pool)
{
	pool->tokens.data = NULL;
	pool->tokens.count = 0;
	pool->tokens.capacity = 0;
	pool->entries = NULL;
	pool->entry_count = 0;
	pool->entry_capacity = 0;
}

bool ufbxwi_string_pool_rehash(ufbxwi_string_pool *pool)
{
	size_t capacity = ufbxwi_max_sz(pool->entry_capacity * 2, 512);
	ufbxwi_string_entry *new_entries = ufbxwi_alloc(pool->ator, ufbxwi_string_entry, capacity);
	ufbxwi_check(new_entries, false);

	memset(new_entries, 0, capacity * sizeof(ufbxwi_string_entry));
	ufbxwi_for(ufbxwi_string_entry, entry, pool->entries, pool->entry_capacity) {
		// TODO: Better hashing
		uint32_t index = entry->hash;
		for (;;) {
			uint32_t slot = index & (capacity - 1);
			if (new_entries[slot].hash == 0) {
				new_entries[slot] = *entry;
				break;
			}
			index++;
		}
	}

	ufbxwi_free(pool->ator, pool->entries);

	pool->entries = new_entries;
	pool->entry_capacity = (uint32_t)capacity;
	return true;
}

static char *ufbxwi_copy_string(ufbxwi_string_pool *pool, const char *str, size_t length)
{
	char *copy = NULL;
	if (length >= 256) {
		copy = ufbxwi_alloc(pool->ator, char, length + 1);
		ufbxwi_check(copy, NULL);
	} else {
		if (ufbxwi_to_size(pool->block_end - pool->block_pos) < length + 1) {
			const size_t block_size = 4096;
			char *block = ufbxwi_alloc(pool->ator, char, block_size);
			ufbxwi_check(block, NULL);

			pool->block_pos = block;
			pool->block_end = block + block_size;
		}

		copy = pool->block_pos;
		pool->block_pos = copy + (length + 1);
	}

	ufbxw_assert(copy);
	memcpy(copy, str, length);
	copy[length] = '\0';

	return copy;
}

static bool ufbxwi_intern_string(ufbxwi_string_pool *pool, ufbxw_string *dst, const char *str, size_t length)
{
	if (length == 0) {
		dst->data = ufbxwi_empty_char;
		dst->length = 0;
		return true;
	} else if (length == SIZE_MAX) {
		length = strlen(str);
	}

	size_t max_length = UINT32_MAX / 2;
	if (length > max_length) {
		ufbxwi_failf(pool->error, UFBXW_ERROR_STRING_TOO_LONG, "String is too long (%zu bytes, max %zu bytes)", length, max_length);
		return false;
	}

	uint32_t hash = ufbxwi_hash_string(str, length);

	if (pool->entry_count * 2 >= pool->entry_capacity) {
		ufbxwi_check(ufbxwi_string_pool_rehash(pool), false);
	}

	uint32_t capacity = pool->entry_capacity;
	ufbxwi_string_entry *entries = pool->entries;

	uint32_t index = hash;
	for (;;) {
		uint32_t slot = index & (capacity - 1);
		if (entries[slot].hash == hash && entries[slot].length == length && !memcmp(entries[slot].data, str, length)) {
			dst->data = entries[slot].data;
			dst->length = length;
			return true;
		} else if (entries[slot].hash == 0) {
			break;
		}
		index++;
	}

	char *copy = ufbxwi_copy_string(pool, str, length);
	ufbxwi_check(copy, false);

	pool->entry_count++;

	ufbxwi_string_entry *entry = &entries[index & (capacity - 1)];
	entry->data = copy;
	entry->hash = hash;
	entry->token = 0;
	entry->length = (uint32_t)length;

	dst->data = copy;
	dst->length = length;
	return true;
}

static uint32_t ufbxwi_intern_token(ufbxwi_string_pool *pool, const char *str, size_t length)
{
	if (length == 0) return 1; // UFBXWI_TOKEN_EMPTY

	size_t max_length = UINT32_MAX / 2;
	if (length > max_length) {
		ufbxwi_failf(pool->error, UFBXW_ERROR_STRING_TOO_LONG, "String is too long (%zu bytes, max %zu bytes)", length, max_length);
		return false;
	}

	uint32_t hash = ufbxwi_hash_string(str, length);

	if (pool->entry_count * 2 >= pool->entry_capacity) {
		ufbxwi_check(ufbxwi_string_pool_rehash(pool), 0);
	}

	uint32_t capacity = pool->entry_capacity;
	ufbxwi_string_entry *entries = pool->entries;

	uint32_t index = hash;
	for (;;) {
		uint32_t slot = index & (capacity - 1);
		if (entries[slot].hash == hash && entries[slot].length == length && !memcmp(entries[slot].data, str, length)) {
			if (entries[slot].token > 0) {
				return entries[slot].token;
			} else {
				uint32_t token = (uint32_t)pool->tokens.count;
				ufbxw_string *dst = ufbxwi_list_push_uninit(pool->ator, &pool->tokens, ufbxw_string);
				ufbxwi_check(dst, 0);
				dst->data = entries[slot].data;
				dst->length = length;
				entries[slot].token = token;
				return token;
			}
		} else if (entries[slot].hash == 0) {
			break;
		}
		index++;
	}

	char *copy = ufbxwi_copy_string(pool, str, length);
	ufbxwi_check(copy, 0);

	pool->entry_count++;

	uint32_t token = (uint32_t)pool->tokens.count;
	ufbxw_string *dst = ufbxwi_list_push_uninit(pool->ator, &pool->tokens, ufbxw_string);
	ufbxwi_check(dst, 0);
	dst->data = copy;
	dst->length = length;

	ufbxwi_string_entry *entry = &entries[index & (capacity - 1)];
	entry->data = copy;
	entry->hash = hash;
	entry->token = token;
	entry->length = (uint32_t)length;
	return token;
}

static uint32_t ufbxwi_get_token(const ufbxwi_string_pool *pool, const char *str, size_t length)
{
	if (length == 0) return 1; // UFBXWI_TOKEN_EMPTY
	if (length > UINT32_MAX / 2) return 0;
	if (pool->entry_count == 0) return 0;

	uint32_t hash = ufbxwi_hash_string(str, length);
	uint32_t capacity = pool->entry_capacity;
	const ufbxwi_string_entry *entries = pool->entries;

	uint32_t index = hash;
	for (;;) {
		uint32_t slot = index & (capacity - 1);
		if (entries[slot].hash == hash && entries[slot].length == length && !memcmp(entries[slot].data, str, length)) {
			return entries[slot].token;
		} else if (entries[slot].hash == 0) {
			return 0;
		}
		index++;
	}
}

static bool ufbxwi_intern_string_str(ufbxwi_string_pool *pool, ufbxw_string *dst, ufbxw_string str)
{
	return ufbxwi_intern_string(pool, dst, str.data, str.length);
}

static bool ufbxwi_intern_string_str_or_default(ufbxwi_string_pool *pool, ufbxw_string *dst, ufbxw_string str, ufbxw_string default_str)
{
	if (str.length > 0) {
		return ufbxwi_intern_string(pool, dst, str.data, str.length);
	} else {
		return ufbxwi_intern_string(pool, dst, default_str.data, default_str.length);
	}
}

typedef enum ufbxwi_token {
	UFBXWI_TOKEN_NONE,
	UFBXWI_TOKEN_EMPTY,
	UFBXWI_2D_Magnifier_X,
	UFBXWI_2D_Magnifier_Y,
	UFBXWI_2D_Magnifier_Zoom,
	UFBXWI_ActiveAnimStackName,
	UFBXWI_AmbientColor,
	UFBXWI_AmbientFactor,
	UFBXWI_AnimCurve,
	UFBXWI_AnimCurveNode,
	UFBXWI_AnimLayer,
	UFBXWI_AnimStack,
	UFBXWI_AnimationCurve,
	UFBXWI_AnimationCurveNode,
	UFBXWI_AnimationLayer,
	UFBXWI_AnimationStack,
	UFBXWI_AntialiasingIntensity,
	UFBXWI_AntialiasingMethod,
	UFBXWI_ApertureMode,
	UFBXWI_AreaLightShape,
	UFBXWI_AspectHeight,
	UFBXWI_AspectRatioMode,
	UFBXWI_AspectWidth,
	UFBXWI_AudioColor,
	UFBXWI_AutoComputeClipPanes,
	UFBXWI_AxisLen,
	UFBXWI_BBoxMax,
	UFBXWI_BBoxMin,
	UFBXWI_BackPlaneDistance,
	UFBXWI_BackPlaneDistanceMode,
	UFBXWI_BackPlaneOffsetX,
	UFBXWI_BackPlaneOffsetY,
	UFBXWI_BackPlaneRotation,
	UFBXWI_BackPlaneScaleX,
	UFBXWI_BackPlaneScaleY,
	UFBXWI_BackPlateCenter,
	UFBXWI_BackPlateCrop,
	UFBXWI_BackPlateFitImage,
	UFBXWI_BackPlateKeepRatio,
	UFBXWI_Background_Texture,
	UFBXWI_BackgroundAlphaTreshold,
	UFBXWI_BackgroundColor,
	UFBXWI_BindPose,
	UFBXWI_BlendMode,
	UFBXWI_BlendModeBypass,
	UFBXWI_BlendShape,
	UFBXWI_BlendShapeChannel,
	UFBXWI_BottomBarnDoor,
	UFBXWI_Bump,
	UFBXWI_BumpFactor,
	UFBXWI_Camera,
	UFBXWI_CameraFormat,
	UFBXWI_CameraProjectionType,
	UFBXWI_CastLightOnObject,
	UFBXWI_CastShadows,
	UFBXWI_Casts_Shadows,
	UFBXWI_Cluster,
	UFBXWI_Color,
	UFBXWI_CoordAxis,
	UFBXWI_CoordAxisSign,
	UFBXWI_CurrentMappingType,
	UFBXWI_CurrentTextureBlendMode,
	UFBXWI_CurrentTimeMarker,
	UFBXWI_CustomFrameRate,
	UFBXWI_DecayStart,
	UFBXWI_DecayType,
	UFBXWI_DefaultAttributeIndex,
	UFBXWI_DefaultCamera,
	UFBXWI_DeformPercent,
	UFBXWI_Deformer,
	UFBXWI_Description,
	UFBXWI_DiffuseColor,
	UFBXWI_DiffuseFactor,
	UFBXWI_DisplacementColor,
	UFBXWI_DisplacementFactor,
	UFBXWI_DisplaySafeArea,
	UFBXWI_DisplaySafeAreaOnRender,
	UFBXWI_DisplayTurnTableIcon,
	UFBXWI_Document,
	UFBXWI_DocumentUrl,
	UFBXWI_DrawFrontFacingVolumetricLight,
	UFBXWI_DrawGroundProjection,
	UFBXWI_DrawVolumetricLight,
	UFBXWI_EmissiveColor,
	UFBXWI_EmissiveFactor,
	UFBXWI_EnableBarnDoor,
	UFBXWI_EnableFarAttenuation,
	UFBXWI_EnableNearAttenuation,
	UFBXWI_FarAttenuationEnd,
	UFBXWI_FarAttenuationStart,
	UFBXWI_FarPlane,
	UFBXWI_FbxAnimCurveNode,
	UFBXWI_FbxAnimLayer,
	UFBXWI_FbxAnimStack,
	UFBXWI_FbxCamera,
	UFBXWI_FbxFileTexture,
	UFBXWI_FbxLight,
	UFBXWI_FbxMesh,
	UFBXWI_FbxNode,
	UFBXWI_FbxSkeleton,
	UFBXWI_FbxSurfaceLambert,
	UFBXWI_FieldOfView,
	UFBXWI_FieldOfViewX,
	UFBXWI_FieldOfViewY,
	UFBXWI_FileName,
	UFBXWI_FilmAspectRatio,
	UFBXWI_FilmFormatIndex,
	UFBXWI_FilmHeight,
	UFBXWI_FilmOffsetX,
	UFBXWI_FilmOffsetY,
	UFBXWI_FilmRollOrder,
	UFBXWI_FilmRollPivotX,
	UFBXWI_FilmRollPivotY,
	UFBXWI_FilmRollValue,
	UFBXWI_FilmSqueezeRatio,
	UFBXWI_FilmTranslateX,
	UFBXWI_FilmTranslateY,
	UFBXWI_FilmWidth,
	UFBXWI_FocalLength,
	UFBXWI_FocusAngle,
	UFBXWI_FocusDistance,
	UFBXWI_FocusSource,
	UFBXWI_Fog,
	UFBXWI_Foreground_Opacity,
	UFBXWI_Foreground_Texture,
	UFBXWI_FrameColor,
	UFBXWI_FrameSamplingCount,
	UFBXWI_FrameSamplingType,
	UFBXWI_Freeze,
	UFBXWI_FrontAxis,
	UFBXWI_FrontAxisSign,
	UFBXWI_FrontPlaneDistance,
	UFBXWI_FrontPlaneDistanceMode,
	UFBXWI_FrontPlaneOffsetX,
	UFBXWI_FrontPlaneOffsetY,
	UFBXWI_FrontPlaneRotation,
	UFBXWI_FrontPlaneScaleX,
	UFBXWI_FrontPlaneScaleY,
	UFBXWI_FrontPlateCenter,
	UFBXWI_FrontPlateCrop,
	UFBXWI_FrontPlateFitImage,
	UFBXWI_FrontPlateKeepRatio,
	UFBXWI_GateFit,
	UFBXWI_GeometricRotation,
	UFBXWI_GeometricScaling,
	UFBXWI_GeometricTranslation,
	UFBXWI_Geometry,
	UFBXWI_GlobalSettings,
	UFBXWI_InheritType,
	UFBXWI_InnerAngle,
	UFBXWI_Intensity,
	UFBXWI_InterestPosition,
	UFBXWI_LODBox,
	UFBXWI_LastSaved,
	UFBXWI_LastSaved_ApplicationName,
	UFBXWI_LastSaved_ApplicationVendor,
	UFBXWI_LastSaved_ApplicationVersion,
	UFBXWI_LastSaved_DateTime_GMT,
	UFBXWI_Lcl_Rotation,
	UFBXWI_Lcl_Scaling,
	UFBXWI_Lcl_Translation,
	UFBXWI_LeftBarnDoor,
	UFBXWI_Light,
	UFBXWI_LightType,
	UFBXWI_LimbLength,
	UFBXWI_LimbNode,
	UFBXWI_LocalStart,
	UFBXWI_LocalStop,
	UFBXWI_Lock,
	UFBXWI_LockInterestNavigation,
	UFBXWI_LockMode,
	UFBXWI_LookAtProperty,
	UFBXWI_Material,
	UFBXWI_MaxDampRangeX,
	UFBXWI_MaxDampRangeY,
	UFBXWI_MaxDampRangeZ,
	UFBXWI_MaxDampStrengthX,
	UFBXWI_MaxDampStrengthY,
	UFBXWI_MaxDampStrengthZ,
	UFBXWI_Mesh,
	UFBXWI_MinDampRangeX,
	UFBXWI_MinDampRangeY,
	UFBXWI_MinDampRangeZ,
	UFBXWI_MinDampStrengthX,
	UFBXWI_MinDampStrengthY,
	UFBXWI_MinDampStrengthZ,
	UFBXWI_Model,
	UFBXWI_Motion_Blur_Intensity,
	UFBXWI_MultiLayer,
	UFBXWI_Mute,
	UFBXWI_NearAttenuationEnd,
	UFBXWI_NearAttenuationStart,
	UFBXWI_NearPlane,
	UFBXWI_NegativePercentShapeSupport,
	UFBXWI_NodeAttribute,
	UFBXWI_NormalMap,
	UFBXWI_OpticalCenterX,
	UFBXWI_OpticalCenterY,
	UFBXWI_Original,
	UFBXWI_OriginalUnitScaleFactor,
	UFBXWI_OriginalUpAxis,
	UFBXWI_OriginalUpAxisSign,
	UFBXWI_Original_ApplicationName,
	UFBXWI_Original_ApplicationVendor,
	UFBXWI_Original_ApplicationVersion,
	UFBXWI_Original_DateTime_GMT,
	UFBXWI_Original_FileName,
	UFBXWI_OrthoZoom,
	UFBXWI_OuterAngle,
	UFBXWI_Path,
	UFBXWI_PixelAspectRatio,
	UFBXWI_Pose,
	UFBXWI_Position,
	UFBXWI_PostRotation,
	UFBXWI_PreRotation,
	UFBXWI_PreScale,
	UFBXWI_PreferedAngleX,
	UFBXWI_PreferedAngleY,
	UFBXWI_PreferedAngleZ,
	UFBXWI_PremultiplyAlpha,
	UFBXWI_Primary_Visibility,
	UFBXWI_QuaternionInterpolate,
	UFBXWI_Receive_Shadows,
	UFBXWI_ReferenceStart,
	UFBXWI_ReferenceStop,
	UFBXWI_RelPath,
	UFBXWI_RightBarnDoor,
	UFBXWI_Roll,
	UFBXWI_Rotation,
	UFBXWI_RotationAccumulationMode,
	UFBXWI_RotationActive,
	UFBXWI_RotationMax,
	UFBXWI_RotationMaxX,
	UFBXWI_RotationMaxY,
	UFBXWI_RotationMaxZ,
	UFBXWI_RotationMin,
	UFBXWI_RotationMinX,
	UFBXWI_RotationMinY,
	UFBXWI_RotationMinZ,
	UFBXWI_RotationOffset,
	UFBXWI_RotationOrder,
	UFBXWI_RotationPivot,
	UFBXWI_RotationSpaceForLimitOnly,
	UFBXWI_RotationStiffnessX,
	UFBXWI_RotationStiffnessY,
	UFBXWI_RotationStiffnessZ,
	UFBXWI_SafeAreaAspectRatio,
	UFBXWI_SafeAreaDisplayStyle,
	UFBXWI_ScaleAccumulationMode,
	UFBXWI_Scaling,
	UFBXWI_ScalingActive,
	UFBXWI_ScalingMax,
	UFBXWI_ScalingMaxX,
	UFBXWI_ScalingMaxY,
	UFBXWI_ScalingMaxZ,
	UFBXWI_ScalingMin,
	UFBXWI_ScalingMinX,
	UFBXWI_ScalingMinY,
	UFBXWI_ScalingMinZ,
	UFBXWI_ScalingOffset,
	UFBXWI_ScalingPivot,
	UFBXWI_SceneInfo,
	UFBXWI_ShadingModel,
	UFBXWI_ShadowColor,
	UFBXWI_Shape,
	UFBXWI_Show,
	UFBXWI_ShowAudio,
	UFBXWI_ShowAzimut,
	UFBXWI_ShowBackplate,
	UFBXWI_ShowFrontplate,
	UFBXWI_ShowGrid,
	UFBXWI_ShowInfoOnMoving,
	UFBXWI_ShowName,
	UFBXWI_ShowOpticalCenter,
	UFBXWI_ShowTimeCode,
	UFBXWI_Size,
	UFBXWI_Skin,
	UFBXWI_SnapOnFrameMode,
	UFBXWI_Solo,
	UFBXWI_SourceObject,
	UFBXWI_SrcDocumentUrl,
	UFBXWI_SubDeformer,
	UFBXWI_Texture,
	UFBXWI_Texture_alpha,
	UFBXWI_TextureRotationPivot,
	UFBXWI_TextureScalingPivot,
	UFBXWI_TextureTypeUse,
	UFBXWI_TimeMarker,
	UFBXWI_TimeMode,
	UFBXWI_TimeProtocol,
	UFBXWI_TimeSpanStart,
	UFBXWI_TimeSpanStop,
	UFBXWI_TopBarnDoor,
	UFBXWI_Translation,
	UFBXWI_TranslationActive,
	UFBXWI_TranslationMax,
	UFBXWI_TranslationMaxX,
	UFBXWI_TranslationMaxY,
	UFBXWI_TranslationMaxZ,
	UFBXWI_TranslationMin,
	UFBXWI_TranslationMinX,
	UFBXWI_TranslationMinY,
	UFBXWI_TranslationMinZ,
	UFBXWI_TransparencyFactor,
	UFBXWI_TransparentColor,
	UFBXWI_TurnTable,
	UFBXWI_UVSet,
	UFBXWI_UVSwap,
	UFBXWI_UnitScaleFactor,
	UFBXWI_UpAxis,
	UFBXWI_UpAxisSign,
	UFBXWI_UpVector,
	UFBXWI_UpVectorProperty,
	UFBXWI_Use2DMagnifierZoom,
	UFBXWI_UseAccumulationBuffer,
	UFBXWI_UseAntialiasing,
	UFBXWI_UseDepthOfField,
	UFBXWI_UseFrameColor,
	UFBXWI_UseMaterial,
	UFBXWI_UseMipMap,
	UFBXWI_UseMotionBlur,
	UFBXWI_UseRealTimeDOFAndAA,
	UFBXWI_UseRealTimeMotionBlur,
	UFBXWI_UserData,
	UFBXWI_VectorDisplacementColor,
	UFBXWI_VectorDisplacementFactor,
	UFBXWI_ViewCameraToLookAt,
	UFBXWI_ViewFrustumBackPlaneMode,
	UFBXWI_ViewFrustumFrontPlaneMode,
	UFBXWI_ViewFrustumNearFarPlane,
	UFBXWI_Visibility,
	UFBXWI_Visibility_Inheritance,
	UFBXWI_Weight,
	UFBXWI_WrapModeU,
	UFBXWI_WrapModeV,
	UFBXWI_d,
	UFBXWI_d_W,
	UFBXWI_d_X,
	UFBXWI_d_Y,
	UFBXWI_d_Z,
	UFBXWI_TOKEN_COUNT,
	UFBXWI_TOKEN_FORCE_32BIT = 0x7fffffff,
} ufbxwi_token;

static const ufbxw_string ufbxwi_tokens[] = {
	{ "", 0 },
	{ "2D Magnifier X", 14 },
	{ "2D Magnifier Y", 14 },
	{ "2D Magnifier Zoom", 17 },
	{ "ActiveAnimStackName", 19 },
	{ "AmbientColor", 12 },
	{ "AmbientFactor", 13 },
	{ "AnimCurve", 9 },
	{ "AnimCurveNode", 13 },
	{ "AnimLayer", 9 },
	{ "AnimStack", 9 },
	{ "AnimationCurve", 14 },
	{ "AnimationCurveNode", 18 },
	{ "AnimationLayer", 14 },
	{ "AnimationStack", 14 },
	{ "AntialiasingIntensity", 21 },
	{ "AntialiasingMethod", 18 },
	{ "ApertureMode", 12 },
	{ "AreaLightShape", 14 },
	{ "AspectHeight", 12 },
	{ "AspectRatioMode", 15 },
	{ "AspectWidth", 11 },
	{ "AudioColor", 10 },
	{ "AutoComputeClipPanes", 20 },
	{ "AxisLen", 7 },
	{ "BBoxMax", 7 },
	{ "BBoxMin", 7 },
	{ "BackPlaneDistance", 17 },
	{ "BackPlaneDistanceMode", 21 },
	{ "BackPlaneOffsetX", 16 },
	{ "BackPlaneOffsetY", 16 },
	{ "BackPlaneRotation", 17 },
	{ "BackPlaneScaleX", 15 },
	{ "BackPlaneScaleY", 15 },
	{ "BackPlateCenter", 15 },
	{ "BackPlateCrop", 13 },
	{ "BackPlateFitImage", 17 },
	{ "BackPlateKeepRatio", 18 },
	{ "Background Texture", 18 },
	{ "BackgroundAlphaTreshold", 23 },
	{ "BackgroundColor", 15 },
	{ "BindPose", 8 },
	{ "BlendMode", 9 },
	{ "BlendModeBypass", 15 },
	{ "BlendShape", 10 },
	{ "BlendShapeChannel", 17 },
	{ "BottomBarnDoor", 14 },
	{ "Bump", 4 },
	{ "BumpFactor", 10 },
	{ "Camera", 6 },
	{ "CameraFormat", 12 },
	{ "CameraProjectionType", 20 },
	{ "CastLightOnObject", 17 },
	{ "CastShadows", 11 },
	{ "Casts Shadows", 13 },
	{ "Cluster", 7 },
	{ "Color", 5 },
	{ "CoordAxis", 9 },
	{ "CoordAxisSign", 13 },
	{ "CurrentMappingType", 18 },
	{ "CurrentTextureBlendMode", 23 },
	{ "CurrentTimeMarker", 17 },
	{ "CustomFrameRate", 15 },
	{ "DecayStart", 10 },
	{ "DecayType", 9 },
	{ "DefaultAttributeIndex", 21 },
	{ "DefaultCamera", 13 },
	{ "DeformPercent", 13 },
	{ "Deformer", 8 },
	{ "Description", 11 },
	{ "DiffuseColor", 12 },
	{ "DiffuseFactor", 13 },
	{ "DisplacementColor", 17 },
	{ "DisplacementFactor", 18 },
	{ "DisplaySafeArea", 15 },
	{ "DisplaySafeAreaOnRender", 23 },
	{ "DisplayTurnTableIcon", 20 },
	{ "Document", 8 },
	{ "DocumentUrl", 11 },
	{ "DrawFrontFacingVolumetricLight", 30 },
	{ "DrawGroundProjection", 20 },
	{ "DrawVolumetricLight", 19 },
	{ "EmissiveColor", 13 },
	{ "EmissiveFactor", 14 },
	{ "EnableBarnDoor", 14 },
	{ "EnableFarAttenuation", 20 },
	{ "EnableNearAttenuation", 21 },
	{ "FarAttenuationEnd", 17 },
	{ "FarAttenuationStart", 19 },
	{ "FarPlane", 8 },
	{ "FbxAnimCurveNode", 16 },
	{ "FbxAnimLayer", 12 },
	{ "FbxAnimStack", 12 },
	{ "FbxCamera", 9 },
	{ "FbxFileTexture", 14 },
	{ "FbxLight", 8 },
	{ "FbxMesh", 7 },
	{ "FbxNode", 7 },
	{ "FbxSkeleton", 11 },
	{ "FbxSurfaceLambert", 17 },
	{ "FieldOfView", 11 },
	{ "FieldOfViewX", 12 },
	{ "FieldOfViewY", 12 },
	{ "FileName", 8 },
	{ "FilmAspectRatio", 15 },
	{ "FilmFormatIndex", 15 },
	{ "FilmHeight", 10 },
	{ "FilmOffsetX", 11 },
	{ "FilmOffsetY", 11 },
	{ "FilmRollOrder", 13 },
	{ "FilmRollPivotX", 14 },
	{ "FilmRollPivotY", 14 },
	{ "FilmRollValue", 13 },
	{ "FilmSqueezeRatio", 16 },
	{ "FilmTranslateX", 14 },
	{ "FilmTranslateY", 14 },
	{ "FilmWidth", 9 },
	{ "FocalLength", 11 },
	{ "FocusAngle", 10 },
	{ "FocusDistance", 13 },
	{ "FocusSource", 11 },
	{ "Fog", 3 },
	{ "Foreground Opacity", 18 },
	{ "Foreground Texture", 18 },
	{ "FrameColor", 10 },
	{ "FrameSamplingCount", 18 },
	{ "FrameSamplingType", 17 },
	{ "Freeze", 6 },
	{ "FrontAxis", 9 },
	{ "FrontAxisSign", 13 },
	{ "FrontPlaneDistance", 18 },
	{ "FrontPlaneDistanceMode", 22 },
	{ "FrontPlaneOffsetX", 17 },
	{ "FrontPlaneOffsetY", 17 },
	{ "FrontPlaneRotation", 18 },
	{ "FrontPlaneScaleX", 16 },
	{ "FrontPlaneScaleY", 16 },
	{ "FrontPlateCenter", 16 },
	{ "FrontPlateCrop", 14 },
	{ "FrontPlateFitImage", 18 },
	{ "FrontPlateKeepRatio", 19 },
	{ "GateFit", 7 },
	{ "GeometricRotation", 17 },
	{ "GeometricScaling", 16 },
	{ "GeometricTranslation", 20 },
	{ "Geometry", 8 },
	{ "GlobalSettings", 14 },
	{ "InheritType", 11 },
	{ "InnerAngle", 10 },
	{ "Intensity", 9 },
	{ "InterestPosition", 16 },
	{ "LODBox", 6 },
	{ "LastSaved", 9 },
	{ "LastSaved|ApplicationName", 25 },
	{ "LastSaved|ApplicationVendor", 27 },
	{ "LastSaved|ApplicationVersion", 28 },
	{ "LastSaved|DateTime_GMT", 22 },
	{ "Lcl Rotation", 12 },
	{ "Lcl Scaling", 11 },
	{ "Lcl Translation", 15 },
	{ "LeftBarnDoor", 12 },
	{ "Light", 5 },
	{ "LightType", 9 },
	{ "LimbLength", 10 },
	{ "LimbNode", 8 },
	{ "LocalStart", 10 },
	{ "LocalStop", 9 },
	{ "Lock", 4 },
	{ "LockInterestNavigation", 22 },
	{ "LockMode", 8 },
	{ "LookAtProperty", 14 },
	{ "Material", 8 },
	{ "MaxDampRangeX", 13 },
	{ "MaxDampRangeY", 13 },
	{ "MaxDampRangeZ", 13 },
	{ "MaxDampStrengthX", 16 },
	{ "MaxDampStrengthY", 16 },
	{ "MaxDampStrengthZ", 16 },
	{ "Mesh", 4 },
	{ "MinDampRangeX", 13 },
	{ "MinDampRangeY", 13 },
	{ "MinDampRangeZ", 13 },
	{ "MinDampStrengthX", 16 },
	{ "MinDampStrengthY", 16 },
	{ "MinDampStrengthZ", 16 },
	{ "Model", 5 },
	{ "Motion Blur Intensity", 21 },
	{ "MultiLayer", 10 },
	{ "Mute", 4 },
	{ "NearAttenuationEnd", 18 },
	{ "NearAttenuationStart", 20 },
	{ "NearPlane", 9 },
	{ "NegativePercentShapeSupport", 27 },
	{ "NodeAttribute", 13 },
	{ "NormalMap", 9 },
	{ "OpticalCenterX", 14 },
	{ "OpticalCenterY", 14 },
	{ "Original", 8 },
	{ "OriginalUnitScaleFactor", 23 },
	{ "OriginalUpAxis", 14 },
	{ "OriginalUpAxisSign", 18 },
	{ "Original|ApplicationName", 24 },
	{ "Original|ApplicationVendor", 26 },
	{ "Original|ApplicationVersion", 27 },
	{ "Original|DateTime_GMT", 21 },
	{ "Original|FileName", 17 },
	{ "OrthoZoom", 9 },
	{ "OuterAngle", 10 },
	{ "Path", 4 },
	{ "PixelAspectRatio", 16 },
	{ "Pose", 4 },
	{ "Position", 8 },
	{ "PostRotation", 12 },
	{ "PreRotation", 11 },
	{ "PreScale", 8 },
	{ "PreferedAngleX", 14 },
	{ "PreferedAngleY", 14 },
	{ "PreferedAngleZ", 14 },
	{ "PremultiplyAlpha", 16 },
	{ "Primary Visibility", 18 },
	{ "QuaternionInterpolate", 21 },
	{ "Receive Shadows", 15 },
	{ "ReferenceStart", 14 },
	{ "ReferenceStop", 13 },
	{ "RelPath", 7 },
	{ "RightBarnDoor", 13 },
	{ "Roll", 4 },
	{ "Rotation", 8 },
	{ "RotationAccumulationMode", 24 },
	{ "RotationActive", 14 },
	{ "RotationMax", 11 },
	{ "RotationMaxX", 12 },
	{ "RotationMaxY", 12 },
	{ "RotationMaxZ", 12 },
	{ "RotationMin", 11 },
	{ "RotationMinX", 12 },
	{ "RotationMinY", 12 },
	{ "RotationMinZ", 12 },
	{ "RotationOffset", 14 },
	{ "RotationOrder", 13 },
	{ "RotationPivot", 13 },
	{ "RotationSpaceForLimitOnly", 25 },
	{ "RotationStiffnessX", 18 },
	{ "RotationStiffnessY", 18 },
	{ "RotationStiffnessZ", 18 },
	{ "SafeAreaAspectRatio", 19 },
	{ "SafeAreaDisplayStyle", 20 },
	{ "ScaleAccumulationMode", 21 },
	{ "Scaling", 7 },
	{ "ScalingActive", 13 },
	{ "ScalingMax", 10 },
	{ "ScalingMaxX", 11 },
	{ "ScalingMaxY", 11 },
	{ "ScalingMaxZ", 11 },
	{ "ScalingMin", 10 },
	{ "ScalingMinX", 11 },
	{ "ScalingMinY", 11 },
	{ "ScalingMinZ", 11 },
	{ "ScalingOffset", 13 },
	{ "ScalingPivot", 12 },
	{ "SceneInfo", 9 },
	{ "ShadingModel", 12 },
	{ "ShadowColor", 11 },
	{ "Shape", 5 },
	{ "Show", 4 },
	{ "ShowAudio", 9 },
	{ "ShowAzimut", 10 },
	{ "ShowBackplate", 13 },
	{ "ShowFrontplate", 14 },
	{ "ShowGrid", 8 },
	{ "ShowInfoOnMoving", 16 },
	{ "ShowName", 8 },
	{ "ShowOpticalCenter", 17 },
	{ "ShowTimeCode", 12 },
	{ "Size", 4 },
	{ "Skin", 4 },
	{ "SnapOnFrameMode", 15 },
	{ "Solo", 4 },
	{ "SourceObject", 12 },
	{ "SrcDocumentUrl", 14 },
	{ "SubDeformer", 11 },
	{ "Texture", 7 },
	{ "Texture alpha", 13 },
	{ "TextureRotationPivot", 20 },
	{ "TextureScalingPivot", 19 },
	{ "TextureTypeUse", 14 },
	{ "TimeMarker", 10 },
	{ "TimeMode", 8 },
	{ "TimeProtocol", 12 },
	{ "TimeSpanStart", 13 },
	{ "TimeSpanStop", 12 },
	{ "TopBarnDoor", 11 },
	{ "Translation", 11 },
	{ "TranslationActive", 17 },
	{ "TranslationMax", 14 },
	{ "TranslationMaxX", 15 },
	{ "TranslationMaxY", 15 },
	{ "TranslationMaxZ", 15 },
	{ "TranslationMin", 14 },
	{ "TranslationMinX", 15 },
	{ "TranslationMinY", 15 },
	{ "TranslationMinZ", 15 },
	{ "TransparencyFactor", 18 },
	{ "TransparentColor", 16 },
	{ "TurnTable", 9 },
	{ "UVSet", 5 },
	{ "UVSwap", 6 },
	{ "UnitScaleFactor", 15 },
	{ "UpAxis", 6 },
	{ "UpAxisSign", 10 },
	{ "UpVector", 8 },
	{ "UpVectorProperty", 16 },
	{ "Use2DMagnifierZoom", 18 },
	{ "UseAccumulationBuffer", 21 },
	{ "UseAntialiasing", 15 },
	{ "UseDepthOfField", 15 },
	{ "UseFrameColor", 13 },
	{ "UseMaterial", 11 },
	{ "UseMipMap", 9 },
	{ "UseMotionBlur", 13 },
	{ "UseRealTimeDOFAndAA", 19 },
	{ "UseRealTimeMotionBlur", 21 },
	{ "UserData", 8 },
	{ "VectorDisplacementColor", 23 },
	{ "VectorDisplacementFactor", 24 },
	{ "ViewCameraToLookAt", 18 },
	{ "ViewFrustumBackPlaneMode", 24 },
	{ "ViewFrustumFrontPlaneMode", 25 },
	{ "ViewFrustumNearFarPlane", 23 },
	{ "Visibility", 10 },
	{ "Visibility Inheritance", 22 },
	{ "Weight", 6 },
	{ "WrapModeU", 9 },
	{ "WrapModeV", 9 },
	{ "d", 1 },
	{ "d|W", 3 },
	{ "d|X", 3 },
	{ "d|Y", 3 },
	{ "d|Z", 3 },
};

// Not including none
ufbxw_static_assert(ufbxwi_tokens_count, ufbxwi_arraycount(ufbxwi_tokens) == UFBXWI_TOKEN_COUNT - 1);

uint32_t ufbxwi_hash_token(ufbxwi_token token)
{
	uint32_t x = (uint32_t)token;
	x ^= x >> 16;
	x *= 0x7feb352d;
	x ^= x >> 16;
	return x;
}

#endif

// -- Buffers

#ifdef UFBXWI_FEATURE_BUFFER

#define ufbxwi_empty_int_buffer ((ufbxw_int_buffer){NULL,0})
#define ufbxwi_empty_vec3_buffer ((ufbxw_vec3_buffer){0})

typedef enum {
	UFBXWI_BUFFER_TYPE_NONE,
	UFBXWI_BUFFER_TYPE_INT,
	UFBXWI_BUFFER_TYPE_LONG,
	UFBXWI_BUFFER_TYPE_REAL,
	UFBXWI_BUFFER_TYPE_VEC2,
	UFBXWI_BUFFER_TYPE_VEC3,
	UFBXWI_BUFFER_TYPE_VEC4,
	UFBXWI_BUFFER_TYPE_FLOAT,
	UFBXWI_BUFFER_TYPE_KEY_ATTR,
} ufbxwi_buffer_type;

typedef enum {
	UFBXWI_BUFFER_TYPE_FLAG_ASCII_INT = 0x1,
} ufbxwi_buffer_type_flags;

typedef struct {
	uint8_t size;
	uint8_t components;
	uint8_t scalar_type;
	uint8_t flags;
} ufbxwi_buffer_type_info;

static const ufbxwi_buffer_type_info ufbxwi_buffer_type_infos[] = {
	{ 0 },
	{ sizeof(int32_t), 1, UFBXWI_BUFFER_TYPE_INT },
	{ sizeof(int64_t), 1, UFBXWI_BUFFER_TYPE_LONG },
	{ sizeof(ufbxw_real), 1, UFBXWI_BUFFER_TYPE_REAL },
	{ sizeof(ufbxw_vec2), 2, UFBXWI_BUFFER_TYPE_REAL },
	{ sizeof(ufbxw_vec3), 3, UFBXWI_BUFFER_TYPE_REAL },
	{ sizeof(ufbxw_vec4), 4, UFBXWI_BUFFER_TYPE_REAL },
	{ sizeof(float), 1, UFBXWI_BUFFER_TYPE_FLOAT },
	{ sizeof(float) * 4, 4, UFBXWI_BUFFER_TYPE_FLOAT, UFBXWI_BUFFER_TYPE_FLAG_ASCII_INT },
};

static ufbxwi_forceinline ufbxw_id ufbxwi_make_buffer_id(ufbxwi_buffer_type type, uint32_t generation, size_t index)
{
	ufbxw_assert((uint64_t)index < ((uint64_t)1u << 32u));
	return (ufbxw_id)(((uint64_t)type << 48) | ((uint64_t)generation << 32) | (index));
}

#define ufbxwi_buffer_id_index(id) (uint32_t)(id)
#define ufbxwi_buffer_id_type(id) (ufbxwi_buffer_type)(((id) >> 48))
#define ufbxwi_buffer_id_generation(id) (uint32_t)(((id) >> 32) & 0xffff)

typedef enum {
	UFBXWI_BUFFER_STATE_NONE,
	UFBXWI_BUFFER_STATE_OWNED,
	UFBXWI_BUFFER_STATE_EXTERNAL,
	UFBXWI_BUFFER_STATE_STREAM,
} ufbxwi_buffer_state;

typedef enum {
	UFBXWI_BUFFER_FLAG_TEMPORARY = 0x1,
} ufbxwi_buffer_flag;

typedef struct {
	float slope_right;
	float slope_next_left;
	uint32_t packed_weight;
	uint32_t packed_velocity;
} ufbxwi_key_attr;

typedef size_t ufbxwi_float_stream_fn(void *user, float *dst, size_t dst_size, size_t offset);
typedef size_t ufbxwi_key_attr_stream_fn(void *user, ufbxwi_key_attr *dst, size_t dst_size, size_t offset);

typedef union {
	ufbxw_int_stream_fn *int_fn;
	ufbxw_long_stream_fn *long_fn;
	ufbxw_real_stream_fn *real_fn;
	ufbxw_vec2_stream_fn *vec2_fn;
	ufbxw_vec3_stream_fn *vec3_fn;
	ufbxw_vec4_stream_fn *vec4_fn;
	ufbxwi_float_stream_fn *float_fn;
	ufbxwi_key_attr_stream_fn *key_attr_fn;
} ufbxwi_stream_fn;

typedef struct {
	ufbxw_buffer_id id;
	ufbxwi_buffer_state state;
	size_t count;
	uint32_t flags;

	uint32_t refcount;
	uint32_t user_refcount;

	ufbxw_buffer_deleter_fn *deleter_fn;
	void *deleter_user;

	void *stream_data;
	uint32_t stream_count;
	uint32_t stream_capacity;
	uint32_t stream_offset;
	uint32_t stream_read_count;

	union {
		struct {
			void *data;
			size_t alloc_size;
		} owned;
		struct {
			const void *data;
			size_t data_size;
		} external;
		struct {
			ufbxwi_stream_fn fn;
			void *user;
		} stream;
	} data;
} ufbxwi_buffer;

UFBXWI_LIST_TYPE(ufbxwi_buffer_list, ufbxwi_buffer);

typedef struct {
	ufbxwi_error *error;
	ufbxwi_allocator *ator;

	ufbxwi_buffer_list buffers;
	ufbxwi_uint32_list free_buffer_ids;

} ufbxwi_buffer_pool;

static void ufbxwi_buffer_pool_init(ufbxwi_buffer_pool *pool, ufbxwi_allocator *ator, ufbxwi_error *error)
{
	pool->ator = ator;
	pool->error = error;
}

static void ufbxwi_mark_buffers_failed(ufbxwi_buffer_pool *pool)
{
	pool->buffers.data = NULL;
	pool->buffers.count = 0;
	pool->buffers.capacity = 0;
	pool->free_buffer_ids.data = NULL;
	pool->free_buffer_ids.count = 0;
	pool->free_buffer_ids.capacity = 0;
}

static ufbxwi_forceinline ufbxwi_buffer *ufbxwi_get_buffer(ufbxwi_buffer_pool *pool, ufbxw_buffer_id id)
{
	size_t index = ufbxwi_buffer_id_index(id);
	if (index >= pool->buffers.count) return NULL;
	ufbxwi_buffer *buffer = &pool->buffers.data[index];
	if (buffer->id != id) return NULL;
	return buffer;
}

static ufbxwi_forceinline size_t ufbxwi_get_buffer_size(ufbxwi_buffer_pool *pool, ufbxw_buffer_id id)
{
	ufbxwi_buffer *buffer = ufbxwi_get_buffer(pool, id);
	return buffer ? buffer->count : 0;
}

static ufbxwi_mutable_void_span ufbxwi_get_buffer_owned_data(ufbxwi_buffer_pool *pool, ufbxw_buffer_id id)
{
	ufbxwi_buffer *buffer = ufbxwi_get_buffer(pool, id);
	ufbxwi_mutable_void_span result = { NULL, 0 };
	if (buffer->state == UFBXWI_BUFFER_STATE_OWNED) {
		result.data = buffer->data.owned.data;
		result.count = buffer->count;
	}
	return result;
}

static bool ufbxwi_is_buffer_streaming(ufbxwi_buffer_pool *pool, ufbxw_buffer_id id)
{
	ufbxwi_buffer *buffer = ufbxwi_get_buffer(pool, id);
	if (!buffer) return false;
	return buffer->state == UFBXWI_BUFFER_STATE_STREAM;
}

static void ufbxwi_reset_buffer(ufbxwi_buffer_pool *pool, ufbxwi_buffer *buffer)
{
	if (buffer->deleter_fn != NULL) {
		void *deleter_data = NULL;
		if (buffer->state == UFBXWI_BUFFER_STATE_EXTERNAL) {
			deleter_data = (void*)buffer->data.external.data;
		} else if (buffer->state == UFBXWI_BUFFER_STATE_STREAM) {
			deleter_data = (void*)buffer->data.stream.user;
		}
		buffer->deleter_fn(buffer->deleter_user, deleter_data);
	}

	if (buffer->state == UFBXWI_BUFFER_STATE_OWNED) {
		ufbxwi_free(pool->ator, buffer->data.owned.data);
	} else if (buffer->state == UFBXWI_BUFFER_STATE_STREAM) {
		ufbxwi_free(pool->ator, buffer->stream_data);
	}

	buffer->deleter_fn = NULL;
	buffer->deleter_user = NULL;
	buffer->state = UFBXWI_BUFFER_STATE_NONE;

	buffer->stream_data = NULL;
	buffer->stream_count = 0;
	buffer->stream_capacity = 0;
	buffer->stream_offset = 0;
	buffer->stream_read_count = 0;

	memset(&buffer->data, 0, sizeof(buffer->data));
}

static size_t ufbxwi_buffer_stream_read(void *dst, size_t dst_count, size_t offset, ufbxwi_buffer_type type, ufbxwi_stream_fn fn, void *user)
{
	switch (type) {
	case UFBXWI_BUFFER_TYPE_NONE:
		ufbxwi_unreachable("reading form uninitialized buffer");
		break;
	case UFBXWI_BUFFER_TYPE_INT:
		return fn.int_fn(user, (int32_t*)dst, dst_count, offset);
	case UFBXWI_BUFFER_TYPE_LONG:
		return fn.long_fn(user, (int64_t*)dst, dst_count, offset);
	case UFBXWI_BUFFER_TYPE_REAL:
		return fn.real_fn(user, (ufbxw_real*)dst, dst_count, offset);
	case UFBXWI_BUFFER_TYPE_VEC2:
		return fn.vec2_fn(user, (ufbxw_vec2*)dst, dst_count, offset);
	case UFBXWI_BUFFER_TYPE_VEC3:
		return fn.vec3_fn(user, (ufbxw_vec3*)dst, dst_count, offset);
	case UFBXWI_BUFFER_TYPE_VEC4:
		return fn.vec4_fn(user, (ufbxw_vec4*)dst, dst_count, offset);
	case UFBXWI_BUFFER_TYPE_FLOAT:
		return fn.float_fn(user, (float*)dst, dst_count, offset);
	case UFBXWI_BUFFER_TYPE_KEY_ATTR:
		return fn.key_attr_fn(user, (ufbxwi_key_attr*)dst, dst_count, offset);
	}
	return SIZE_MAX;
}

static size_t ufbxwi_buffer_read_to(ufbxwi_buffer_pool *pool, ufbxw_buffer_id id, void *dst, size_t dst_count, size_t offset)
{
	ufbxwi_buffer_type type = ufbxwi_buffer_id_type(id);
	ufbxwi_buffer *buffer = ufbxwi_get_buffer(pool, id);
	size_t type_size = ufbxwi_buffer_type_infos[type].size;
	if (!buffer || dst_count == 0) return 0;

	ufbxw_assert(offset + dst_count <= buffer->count);

	switch (buffer->state) {
	case UFBXWI_BUFFER_STATE_NONE:
		return 0;
	case UFBXWI_BUFFER_STATE_OWNED:
		memcpy(dst, (char*)buffer->data.owned.data + offset * type_size, dst_count * type_size);
		return dst_count;
	case UFBXWI_BUFFER_STATE_EXTERNAL:
		memcpy(dst, (char*)buffer->data.external.data + offset * type_size, dst_count * type_size);
		return dst_count;
	case UFBXWI_BUFFER_STATE_STREAM:
		for (size_t off = 0; off < dst_count; ) {
			void *dst_off = (char*)dst + off * type_size;
			size_t off_count = dst_count - off;
			size_t num_read = ufbxwi_buffer_stream_read(dst_off, off_count, offset + off, type, buffer->data.stream.fn, buffer->data.stream.user);
			if (num_read == 0 || num_read == SIZE_MAX) return off;
			ufbxw_assert(num_read <= dst_count);
			off += num_read;
		}
		return dst_count;
	}

	return 0;
}

static ufbxwi_void_span ufbxwi_buffer_read(ufbxwi_buffer_pool *pool, ufbxw_buffer_id id, void *temp, size_t temp_size, size_t offset)
{
	ufbxwi_buffer_type type = ufbxwi_buffer_id_type(id);
	ufbxwi_buffer *buffer = ufbxwi_get_buffer(pool, id);
	size_t type_size = ufbxwi_buffer_type_infos[type].size;

	ufbxw_assert(offset <= buffer->count);

	ufbxwi_void_span result = { NULL, 0 };
	if (!buffer) return result;

	ufbxw_assert(temp_size >= type_size);
	size_t temp_count = temp_size / type_size;

	switch (buffer->state) {
	case UFBXWI_BUFFER_STATE_NONE:
		break;
	case UFBXWI_BUFFER_STATE_OWNED:
		result.data = (char*)buffer->data.owned.data + offset * type_size;
		result.count = buffer->count - offset;
		break;
	case UFBXWI_BUFFER_STATE_EXTERNAL:
		result.data = (char*)buffer->data.external.data + offset * type_size;
		result.count = buffer->count - offset;
		break;
	case UFBXWI_BUFFER_STATE_STREAM: {
		size_t off = 0;
		while (off < temp_count) {
			void *dst_off = (char*)temp + off * type_size;
			size_t dst_count = temp_count - off;
			size_t num_read = ufbxwi_buffer_stream_read(dst_off, dst_count, offset + off, type, buffer->data.stream.fn, buffer->data.stream.user);
			// TODO: Report error
			if (num_read == 0 || num_read == SIZE_MAX) break;
			ufbxw_assert(num_read <= dst_count);
			off += num_read;
		}
		result.data = temp;
		result.count = off;
	} break;
	}

	return result;
}

static ufbxw_buffer_id ufbxwi_create_buffer(ufbxwi_buffer_pool *pool, ufbxwi_buffer_type type)
{
	size_t index = 0;
	if (pool->free_buffer_ids.count > 0) {
		index = pool->free_buffer_ids.data[--pool->free_buffer_ids.count];
	} else {
		index = pool->buffers.count;
		ufbxwi_check(ufbxwi_list_push_zero(pool->ator, &pool->buffers, ufbxwi_buffer), 0);
	}

	ufbxwi_buffer *buffer = &pool->buffers.data[index];
	uint32_t generation = ufbxwi_buffer_id_generation(buffer->id) + 1;
	ufbxw_buffer_id id = ufbxwi_make_buffer_id(type, generation, index);

	buffer->id = id;
	buffer->refcount = 1;

	return id;
}

static ufbxw_buffer_id ufbxwi_create_owned_buffer(ufbxwi_buffer_pool *pool, ufbxwi_buffer_type type, size_t count)
{
	ufbxw_buffer_id id = ufbxwi_create_buffer(pool, type);
	ufbxwi_buffer *buffer = ufbxwi_get_buffer(pool, id);
	if (!buffer) return id;

	size_t alloc_size = 0;
	size_t type_size = ufbxwi_buffer_type_infos[type].size;
	void *data = ufbxwi_alloc_size(pool->ator, type_size, count, &alloc_size);
	if (!data) return 0;

	buffer->state = UFBXWI_BUFFER_STATE_OWNED;
	buffer->count = count;
	buffer->data.owned.data = data;
	buffer->data.owned.alloc_size = alloc_size;

	return id;
}

static ufbxw_buffer_id ufbxwi_create_copy_buffer(ufbxwi_buffer_pool *pool, ufbxwi_buffer_type type, const void *data, size_t count)
{
	ufbxw_buffer_id id = ufbxwi_create_owned_buffer(pool, type, count);
	ufbxwi_buffer *buffer = ufbxwi_get_buffer(pool, id);
	if (!buffer) return id;

	ufbxw_assert(buffer->state == UFBXWI_BUFFER_STATE_OWNED);
	size_t type_size = ufbxwi_buffer_type_infos[type].size;
	memcpy(buffer->data.owned.data, data, count * type_size);

	return id;
}

static ufbxw_buffer_id ufbxwi_create_external_buffer(ufbxwi_buffer_pool *pool, ufbxwi_buffer_type type, const void *data, size_t count, uint32_t flags)
{
	ufbxw_buffer_id id = ufbxwi_create_buffer(pool, type);
	ufbxwi_buffer *buffer = ufbxwi_get_buffer(pool, id);
	if (!buffer) return id;

	buffer->state = UFBXWI_BUFFER_STATE_EXTERNAL;
	buffer->flags = flags;
	buffer->data.external.data = data;
	buffer->data.external.data_size = count * ufbxwi_buffer_type_infos[type].size;
	buffer->count = count;

	return id;
}

static ufbxw_buffer_id ufbxwi_create_stream_buffer(ufbxwi_buffer_pool *pool, ufbxwi_buffer_type type, ufbxwi_stream_fn fn, void *user, size_t count)
{
	ufbxw_buffer_id id = ufbxwi_create_buffer(pool, type);
	ufbxwi_buffer *buffer = ufbxwi_get_buffer(pool, id);
	if (!buffer) return id;

	buffer->state = UFBXWI_BUFFER_STATE_STREAM;
	buffer->data.stream.fn = fn;
	buffer->data.stream.user = user;
	buffer->count = count;

	return id;
}

static void ufbxwi_delete_buffer_imp(ufbxwi_buffer_pool *pool, ufbxw_buffer_id id)
{
	ufbxwi_buffer *buffer = ufbxwi_get_buffer(pool, id);
	if (!buffer) return;

	ufbxwi_reset_buffer(pool, buffer);
	memset(buffer, 0, sizeof(ufbxwi_buffer));

	uint32_t generation = ufbxwi_buffer_id_generation(id);
	buffer->id = ufbxwi_make_buffer_id(UFBXWI_BUFFER_TYPE_NONE, generation, 0);

	if (generation < 0xffff) {
		// We can leak the index safely here if we fail to allocate
		uint32_t index = ufbxwi_buffer_id_index(id);
		ufbxwi_ignore(ufbxwi_list_push_copy(pool->ator, &pool->free_buffer_ids, uint32_t, &index));
	}
}

static void ufbxwi_free_buffer(ufbxwi_buffer_pool *pool, ufbxw_buffer_id id)
{
	if (!id) return;
	ufbxwi_buffer *buffer = ufbxwi_get_buffer(pool, id);
	if (!buffer) return;

	ufbxw_assert(buffer->refcount > 0);
	if (--buffer->refcount == 0) {
		ufbxwi_delete_buffer_imp(pool, id);
	}
}

static void ufbxwi_buffer_set_deleter(ufbxwi_buffer_pool *pool, ufbxw_buffer_id id, ufbxw_buffer_deleter_fn *fn, void *user)
{
	ufbxwi_buffer *buf = ufbxwi_get_buffer(pool, id);
	if (!buf) return;

	ufbxw_assert(buf->deleter_fn == NULL);
	buf->deleter_fn = fn;
	buf->deleter_user = user;
}

static ufbxwi_forceinline ufbxw_buffer_id ufbxwi_make_user_buffer(ufbxwi_buffer_pool *pool, ufbxw_buffer_id src)
{
	ufbxwi_buffer *buffer = ufbxwi_get_buffer(pool, src);
	if (buffer) {
		buffer->user_refcount++;
		buffer->refcount++;
	}
	return src;
}

static void ufbxwi_set_buffer(ufbxwi_buffer_pool *pool, ufbxw_buffer_id *p_dst, ufbxw_buffer_id src)
{
	if (src != 0) {
		ufbxwi_buffer *buffer = ufbxwi_get_buffer(pool, src);
		ufbxw_assert(buffer && buffer->refcount > 0);
	}

	if (*p_dst == src) return;
	ufbxwi_free_buffer(pool, *p_dst);
	*p_dst = src;
}

static bool ufbxwi_make_buffer_owned(ufbxwi_buffer_pool *pool, ufbxw_buffer_id id)
{
	ufbxwi_buffer *buffer = ufbxwi_get_buffer(pool, id);
	if (buffer->state == UFBXWI_BUFFER_STATE_OWNED) return true;

	ufbxwi_buffer_type type = ufbxwi_buffer_id_type(id);
	size_t type_size = ufbxwi_buffer_type_infos[type].size;
	size_t alloc_size = 0;
	void *data = ufbxwi_alloc_size(pool->ator, type_size, buffer->count, &alloc_size);
	ufbxwi_check(data, false);

	if (ufbxwi_buffer_read_to(pool, buffer->id, data, buffer->count, 0) != buffer->count) {
		ufbxwi_free(pool->ator, data);
		ufbxwi_fail(pool->error, UFBXW_ERROR_BUFFER_STREAM, "failed to read buffer data");
		return false;
	}

	ufbxwi_reset_buffer(pool, buffer);
	buffer->flags &= ~UFBXWI_BUFFER_FLAG_TEMPORARY;
	buffer->state = UFBXWI_BUFFER_STATE_OWNED;
	buffer->data.owned.data = data;
	buffer->data.owned.alloc_size = alloc_size;

	return true;
}

static const void *ufbxwi_buffer_get_data(ufbxwi_buffer_pool *pool, ufbxw_buffer_id id)
{
	ufbxwi_buffer_type type = ufbxwi_buffer_id_type(id);
	ufbxwi_buffer *buffer = ufbxwi_get_buffer(pool, id);
	if (!buffer) return NULL;

	switch (buffer->state) {
	case UFBXWI_BUFFER_STATE_NONE:
		return NULL;
	case UFBXWI_BUFFER_STATE_OWNED:
		return buffer->data.owned.data;
	case UFBXWI_BUFFER_STATE_EXTERNAL:
		return buffer->data.external.data;
	case UFBXWI_BUFFER_STATE_STREAM:
		return NULL;
	}
	return NULL;
}

ufbxwi_nodiscard static bool ufbxwi_buffer_materialize(ufbxwi_buffer_pool *pool, ufbxw_buffer_id id)
{
	ufbxwi_buffer_type type = ufbxwi_buffer_id_type(id);
	ufbxwi_buffer *buffer = ufbxwi_get_buffer(pool, id);
	if (!buffer) return false;

	switch (buffer->state) {
	case UFBXWI_BUFFER_STATE_NONE:
		return false;
	case UFBXWI_BUFFER_STATE_OWNED:
	case UFBXWI_BUFFER_STATE_EXTERNAL:
		return true;
	case UFBXWI_BUFFER_STATE_STREAM:
		ufbxwi_check(ufbxwi_make_buffer_owned(pool, id), false);
		return true;
	}
	return false;
}

static void ufbxwi_set_buffer_from_user(ufbxwi_buffer_pool *pool, ufbxw_buffer_id *p_dst, ufbxw_buffer_id src)
{
	if (ufbxwi_is_fatal(pool->error)) return;

	if (src) {
		ufbxwi_buffer *buffer = ufbxwi_get_buffer(pool, src);
		ufbxw_assert(buffer);
		ufbxw_assert(buffer->user_refcount > 0);
		ufbxw_assert(buffer->refcount > 0);
		buffer->user_refcount--;

		if (buffer->flags & UFBXWI_BUFFER_FLAG_TEMPORARY) {
			ufbxwi_make_buffer_owned(pool, src);
		}
	}

	if (*p_dst == src) return;
	ufbxwi_free_buffer(pool, *p_dst);
	*p_dst = src;
}

static ufbxwi_forceinline ufbxw_buffer_id ufbxwi_to_user_buffer(ufbxwi_buffer_pool *pool, ufbxw_buffer_id id)
{
	ufbxwi_buffer *buffer = ufbxwi_get_buffer(pool, id);
	if (buffer) {
		buffer->user_refcount++;
	}
	return id;
}

static void ufbxwi_refer_buffers(ufbxwi_buffer_pool *dst, const ufbxwi_buffer_pool *src)
{
	size_t buffer_count = src->buffers.count;
	ufbxwi_buffer *dst_buffers = ufbxwi_list_push_zero_n(dst->ator, &dst->buffers, ufbxwi_buffer, buffer_count);
	ufbxwi_check(dst_buffers);

	for (size_t i = 0; i < buffer_count; i++) {
		const ufbxwi_buffer *sb = &src->buffers.data[i];
		if (sb->state == UFBXWI_BUFFER_STATE_NONE) continue;

		ufbxwi_buffer *db = &dst_buffers[i];
		db->id = sb->id;
		db->count = sb->count;
		db->refcount = UINT32_MAX / 2;
		db->user_refcount = UINT32_MAX / 2;

		switch (sb->state) {
		case UFBXWI_BUFFER_STATE_NONE:
			break;
		case UFBXWI_BUFFER_STATE_OWNED:
			db->state = UFBXWI_BUFFER_STATE_EXTERNAL;
			db->data.external.data = sb->data.owned.data;
			db->data.external.data_size = sb->data.owned.alloc_size;
			break;
		case UFBXWI_BUFFER_STATE_EXTERNAL:
			db->state = UFBXWI_BUFFER_STATE_EXTERNAL;
			db->data.external.data = sb->data.external.data;
			db->data.external.data_size = sb->data.external.data_size;
			break;
		case UFBXWI_BUFFER_STATE_STREAM:
			db->state = UFBXWI_BUFFER_STATE_STREAM;
			db->data.stream = sb->data.stream;
			break;
		}
	}
}

typedef struct {
	ufbxwi_buffer_type type;
	size_t count;

	const void *data;
	ufbxwi_stream_fn stream_fn;
	void *stream_user;

} ufbxwi_buffer_input;

static ufbxwi_buffer_input ufbxwi_get_buffer_input(ufbxwi_buffer_pool *pool, ufbxw_buffer_id id)
{
	ufbxwi_buffer_input result = { UFBXWI_BUFFER_TYPE_NONE };

	ufbxwi_buffer *buffer = ufbxwi_get_buffer(pool, id);
	ufbxwi_check(buffer, result);

	result.type = ufbxwi_buffer_id_type(id);
	result.count = buffer->count;
	result.data = ufbxwi_buffer_get_data(pool, id);

	if (buffer->state == UFBXWI_BUFFER_STATE_STREAM) {
		result.stream_fn = buffer->data.stream.fn;
		result.stream_user = buffer->data.stream.user;
	}
	return result;
}

static size_t ufbxwi_buffer_input_read_to(const ufbxwi_buffer_input *input, void *dst, size_t dst_count, size_t offset)
{
	ufbxwi_buffer_type type = input->type;
	size_t type_size = ufbxwi_buffer_type_infos[type].size;
	if (dst_count == 0) return 0;

	ufbxw_assert(offset + dst_count <= input->count);

	if (input->data) {
		memcpy(dst, (char*)input->data + offset * type_size, dst_count * type_size);
		return dst_count;
	}

	for (size_t off = 0; off < dst_count; ) {
		void *dst_off = (char*)dst + off * type_size;
		size_t off_count = dst_count - off;
		size_t num_read = ufbxwi_buffer_stream_read(dst_off, off_count, offset + off, type, input->stream_fn, input->stream_user);
		if (num_read == 0 || num_read == SIZE_MAX) return off;
		ufbxw_assert(num_read <= dst_count);
		off += num_read;
	}

	return dst_count;
}

static ufbxwi_void_span ufbxwi_buffer_input_read(const ufbxwi_buffer_input *input, void *temp, size_t temp_size, size_t offset)
{
	ufbxwi_buffer_type type = input->type;
	size_t type_size = ufbxwi_buffer_type_infos[type].size;

	ufbxw_assert(offset <= input->count);

	ufbxwi_void_span result = { NULL, 0 };

	ufbxw_assert(temp_size >= type_size);
	size_t temp_count = temp_size / type_size;

	if (input->data) {
		result.data = (char*)input->data + offset * type_size;
		result.count = input->count - offset;
	} else {
		size_t off = 0;
		while (off < temp_count) {
			void *dst_off = (char*)temp + off * type_size;
			size_t dst_count = temp_count - off;
			size_t num_read = ufbxwi_buffer_stream_read(dst_off, dst_count, offset + off, type, input->stream_fn, input->stream_user);
			// TODO: Report error
			if (num_read == 0 || num_read == SIZE_MAX) break;
			ufbxw_assert(num_read <= dst_count);
			off += num_read;
		}
		result.data = temp;
		result.count = off;
	}

	return result;
}

static ufbxwi_forceinline ufbxw_int_buffer ufbxwi_to_user_int_buffer(ufbxwi_buffer_pool *pool, ufbxw_buffer_id id) { ufbxw_int_buffer b = { ufbxwi_to_user_buffer(pool, id) }; return b; }
static ufbxwi_forceinline ufbxw_long_buffer ufbxwi_to_user_long_buffer(ufbxwi_buffer_pool *pool, ufbxw_buffer_id id) { ufbxw_long_buffer b = { ufbxwi_to_user_buffer(pool, id) }; return b; }
static ufbxwi_forceinline ufbxw_real_buffer ufbxwi_to_user_real_buffer(ufbxwi_buffer_pool *pool, ufbxw_buffer_id id) { ufbxw_real_buffer b = { ufbxwi_to_user_buffer(pool, id) }; return b; }
static ufbxwi_forceinline ufbxw_vec2_buffer ufbxwi_to_user_vec2_buffer(ufbxwi_buffer_pool *pool, ufbxw_buffer_id id) { ufbxw_vec2_buffer b = { ufbxwi_to_user_buffer(pool, id) }; return b; }
static ufbxwi_forceinline ufbxw_vec3_buffer ufbxwi_to_user_vec3_buffer(ufbxwi_buffer_pool *pool, ufbxw_buffer_id id) { ufbxw_vec3_buffer b = { ufbxwi_to_user_buffer(pool, id) }; return b; }
static ufbxwi_forceinline ufbxw_vec4_buffer ufbxwi_to_user_vec4_buffer(ufbxwi_buffer_pool* pool, ufbxw_buffer_id id) { ufbxw_vec4_buffer b = { ufbxwi_to_user_buffer(pool, id) }; return b; }
static ufbxwi_forceinline ufbxw_float_buffer ufbxwi_to_user_float_buffer(ufbxwi_buffer_pool* pool, ufbxw_buffer_id id) { ufbxw_float_buffer b = { ufbxwi_to_user_buffer(pool, id) }; return b; }

typedef struct {
	const void *pos, *end;
} ufbxwi_void_iterator;

#endif

// -- Prop types

static const ufbxw_vec3 ufbxwi_one_vec3 = { 1.0f, 1.0f, 1.0f };

// -- Scene

#ifdef UFBXWI_FEATURE_SCENE

#define UFBXWI_ELEMENT_TYPE_NONE ((ufbxw_element_type)0)

static ufbxwi_forceinline ufbxw_id ufbxwi_make_id(ufbxw_element_type type, uint32_t generation, size_t index)
{
	ufbxw_assert((uint64_t)index < ((uint64_t)1u << 32u));
	return (ufbxw_id)(((uint64_t)type << 48) | ((uint64_t)generation << 32) | (index));
}

#define ufbxwi_id_index(id) (uint32_t)(id)
#define ufbxwi_id_type(id) (ufbxw_element_type)(((id) >> 48))
#define ufbxwi_id_generation(id) (uint32_t)(((id) >> 32) & 0xffff)

typedef struct {
	const char *name;
	ufbxw_prop_type type;
	uint32_t flags;
} ufbxwi_prop_info;

typedef struct {
	ufbxw_string type;
	ufbxw_string sub_type;
	ufbxw_prop_data_type data_type;
} ufbxwi_prop_type;

UFBXWI_LIST_TYPE(ufbxwi_prop_type_list, ufbxwi_prop_type);

enum {
	UFBXWI_PROP_FLAG_TEMP_VALUE = 0x0 << 24,
	UFBXWI_PROP_FLAG_TEMP_DEFAULT = 0x1 << 24,
	UFBXWI_PROP_FLAG_TEMP_TEMPLATE = 0x2 << 24,
	UFBXWI_PROP_FLAG_TEMP_MASK = 0x3 << 24,
};

// Location where property data is stored.
typedef enum {
	UFBXWI_PROP_VALUE_DEFAULT = 0x0, // < Relative to `ufbxwi_prop_defaults`
	UFBXWI_PROP_VALUE_FIELD = 0x1,   // < Relative to `ufbxwi_element.data`
	UFBXWI_PROP_VALUE_DATA = 0x2,  // < Relative to `ufbxwi_element.prop_data`
} ufbxwi_prop_value_base;

// Packed `ufbxwi_prop_value_base` and offset for property data.
// `0` decodes to `{base=UFBXWI_PROP_VALUE_DEFAULT, offset=0}` which is the zero value for any type.
typedef uint32_t ufbxwi_prop_value;

#define ufbxwi_make_prop_value(m_type, m_offset) (ufbxwi_prop_value)((m_offset) << 2 | (m_type))
#define ufbxwi_prop_value_type(m_value) ((m_value) & 0x3)
#define ufbxwi_prop_value_offset(m_value) ((m_value) >> 2)

#define ufbxwi_prop_value_default(m_offset) ufbxwi_make_prop_value(UFBXWI_PROP_VALUE_DEFAULT, (uint32_t)(m_offset))
#define ufbxwi_prop_value_field(m_offset) ufbxwi_make_prop_value(UFBXWI_PROP_VALUE_FIELD, (uint32_t)(m_offset))
#define ufbxwi_prop_value_data(m_offset) ufbxwi_make_prop_value(UFBXWI_PROP_VALUE_DATA, (uint32_t)(m_offset));

typedef struct {
	ufbxwi_token token;
	uint16_t type;
	uint16_t order;
	ufbxwi_prop_value value;
	uint32_t flags;
} ufbxwi_prop;

UFBXWI_LIST_TYPE(ufbxwi_prop_list, ufbxwi_prop);

typedef struct {
	ufbxwi_prop *props;
	uint32_t capacity;
	uint32_t count;
	uint32_t order_counter;
} ufbxwi_props;

typedef struct {
	ufbxwi_token type;
} ufbxwi_object_type;

UFBXWI_LIST_TYPE(ufbxwi_object_type_list, ufbxwi_object_type);

typedef bool ufbxwi_element_init_data_fn(ufbxw_scene *scene, void *data);
typedef struct ufbxwi_element_type_desc ufbxwi_element_type_desc;

typedef struct {
	ufbxw_element_type element_type;

	ufbxwi_token class_type;
	ufbxwi_token sub_type;
	ufbxwi_token object_type;
	ufbxwi_token fbx_type;

	uint32_t object_type_id;
	ufbxwi_props props;
	ufbxw_id template_id;
	uint32_t flags;
	ufbxwi_element_init_data_fn *init_fn;

	bool initialized;
	const ufbxwi_element_type_desc *desc;
} ufbxwi_element_type;

UFBXWI_LIST_TYPE(ufbxwi_element_type_list, ufbxwi_element_type);

enum {
	UFBXWI_ELEMENT_FLAG_HAS_DEFAULT_PROPS = 0x1,
	UFBXWI_ELEMENT_FLAG_ALLOW_NO_OBJECT_ID = 0x2,
};

typedef struct {
	ufbxw_id id;
	ufbxwi_token src_prop;
	ufbxwi_token dst_prop;
} ufbxwi_conn;

UFBXWI_LIST_TYPE(ufbxwi_conn_list, ufbxwi_conn);

UFBXWI_LIST_TYPE(ufbxw_node_list, ufbxw_node);
UFBXWI_LIST_TYPE(ufbxw_material_list, ufbxw_material);

typedef struct {
	ufbxw_id id;

	void *prop_data;
	size_t prop_data_size;
	size_t prop_data_capacity;

	ufbxwi_props props;

	ufbxw_string name;

	ufbxwi_conn_list anim_props;
	ufbxwi_conn_list user_conns_src;
	ufbxwi_conn_list user_conns_dst;

	uint64_t conn_bits;
	uint32_t type_id;
	uint32_t flags;
	uint32_t element_data_size;
} ufbxwi_element;

typedef struct {
	ufbxw_id id;
	ufbxwi_element *element;
} ufbxwi_element_slot;

UFBXWI_LIST_TYPE(ufbxwi_element_slot_list, ufbxwi_element_slot);

typedef struct {
	ufbxwi_element element;

	ufbxw_node parent;
	ufbxw_node_list children;
	ufbxw_id attribute;
	ufbxw_material_list materials;
	ufbxwi_id_list skin_clusters;

	ufbxw_vec3 lcl_translation;
	ufbxw_vec3 lcl_rotation;
	ufbxw_vec3 lcl_scaling;
	ufbxw_vec3 pre_rotation;
	ufbxw_vec3 post_rotation;
	ufbxw_vec3 rotation_offset;
	ufbxw_vec3 rotation_pivot;
	ufbxw_vec3 scaling_offset;
	ufbxw_vec3 scaling_pivot;

	ufbxw_vec3 geometric_translation;
	ufbxw_vec3 geometric_rotation;
	ufbxw_vec3 geometric_scaling;

	int32_t rotation_order;
	int32_t inherit_type;
	ufbxw_real visibility;
	bool visibility_inheritance;

	int32_t default_attribute_index;
} ufbxwi_node;

typedef struct {
	ufbxwi_element element;
	ufbxw_node_list instances;
} ufbxwi_node_attribute;

typedef struct {
	ufbxw_mesh_attribute attribute;
	int32_t set;

	ufbxw_attribute_mapping mapping;
	ufbxw_string name;

	ufbxw_buffer_id values;
	ufbxw_buffer_id indices;

	ufbxw_buffer_id values_w;
} ufbxwi_mesh_attribute;

UFBXWI_LIST_TYPE(ufbxwi_mesh_attribute_list, ufbxwi_mesh_attribute);

typedef struct {
	union {
		ufbxwi_element element;
		ufbxwi_node_attribute attrib;
	};

	ufbxwi_id_list deformers;

	ufbxw_vec3_buffer vertices;

	ufbxw_int_buffer vertex_indices;
	ufbxw_int_buffer face_offsets;
	ufbxw_int_buffer polygon_vertex_index;
	ufbxw_int_buffer edges;

	ufbxwi_mesh_attribute_list attributes;

} ufbxwi_mesh;

typedef struct {
	union {
		ufbxwi_element element;
	};

	ufbxwi_id_list geometries;

} ufbxwi_deformer;

typedef struct {
	union {
		ufbxwi_element element;
		ufbxwi_deformer deformer;
	};

	ufbxwi_id_list clusters;
	ufbxw_skinning_type skinning_type;

	bool has_mesh_bind_transform;
	ufbxw_matrix mesh_bind_transform;

	// TODO: This is kind of cheesy..
	ufbxw_bind_pose bind_pose;

} ufbxwi_skin_deformer;

typedef struct {
	union {
		ufbxwi_element element;
	};

	ufbxw_id deformer;
	ufbxw_node node;

	ufbxw_int_buffer indices;
	ufbxw_real_buffer weights;

	ufbxw_matrix transform;
	ufbxw_matrix transform_link;

} ufbxwi_skin_cluster;

typedef struct {
	union {
		ufbxwi_element element;
	};

	ufbxwi_id_list channels;

} ufbxwi_blend_deformer;

typedef struct {
	ufbxw_blend_shape shape;
	ufbxw_real target_weight;
} ufbxwi_blend_shape_conn;

UFBXWI_LIST_TYPE(ufbxwi_blend_shape_conn_list, ufbxwi_blend_shape_conn);

typedef struct {
	union {
		ufbxwi_element element;
	};

	ufbxw_blend_deformer deformer;
	ufbxwi_blend_shape_conn_list blend_shapes;
	ufbxw_real deform_percent;

} ufbxwi_blend_channel;

typedef struct {
	union {
		ufbxwi_element element;
	};

	ufbxwi_id_list blend_channels;

	ufbxw_int_buffer indices;
	ufbxw_vec3_buffer vertices;
	ufbxw_vec3_buffer normals;

} ufbxwi_blend_shape;

typedef struct {
	union {
		ufbxwi_element element;
		ufbxwi_node_attribute attrib;
	};

	ufbxw_real intensity;
	ufbxw_vec3 color;
	int32_t decay_type;
	int32_t light_type;
	ufbxw_real inner_angle;
	ufbxw_real outer_angle;
} ufbxwi_light;

typedef struct {
	union {
		ufbxwi_element element;
		ufbxwi_node_attribute attrib;
	};

	// TODO: Camera properties
} ufbxwi_camera;

typedef struct {
	union {
		ufbxwi_element element;
		ufbxwi_node_attribute attrib;
	};

	ufbxw_real limb_length;
	ufbxw_real size;
} ufbxwi_skeleton;

typedef struct {
	ufbxw_node node;
	ufbxw_matrix matrix;
} ufbxwi_pose_node;

UFBXWI_LIST_TYPE(ufbxwi_pose_node_list, ufbxwi_pose_node);

typedef struct {
	union {
		ufbxwi_element element;
	};

	ufbxwi_pose_node_list pose_nodes;
} ufbxwi_bind_pose;

typedef struct {
	ufbxwi_element element;
	ufbxwi_conn_list textures;
	ufbxwi_id_list conn_nodes;
	ufbxw_string shading_model;
	bool multi_layer;
} ufbxwi_material;

typedef struct {
	ufbxwi_element element;
	ufbxwi_conn_list conn_materials;
	ufbxw_string type;
	ufbxw_string filename;
	ufbxw_string relative_filename;
} ufbxwi_texture;

typedef struct {
	uint32_t flags;
	float weight_left;
	float weight_right;
	float slope_left;
	float slope_right;
} ufbxwi_anim_key_attr;

UFBXWI_LIST_TYPE(ufbxwi_anim_key_attr_list, ufbxwi_anim_key_attr);

typedef struct {
	ufbxwi_element element;
	ufbxwi_conn prop;

	ufbxwi_ktime_list key_times;
	ufbxwi_float_list key_values;
	ufbxwi_uint32_list key_attr_indices;
	ufbxwi_anim_key_attr_list key_attr_data;

	ufbxw_long_buffer buffer_key_values;
	ufbxw_float_buffer buffer_key_times;
	ufbxw_int_buffer buffer_attr_refcounts;
	ufbxw_float_buffer buffer_attr_flags;
	ufbxw_float_buffer buffer_attr_data;

	bool keys_out_of_order;
	bool data_in_buffers;

} ufbxwi_anim_curve;

typedef struct {
	ufbxwi_element element;
	ufbxwi_conn_list curves;
	ufbxw_anim_layer layer;
	ufbxwi_conn prop;
	ufbxw_real defaults[4];
} ufbxwi_anim_prop;

typedef struct {
	ufbxwi_element element;
	ufbxwi_id_list anim_props;
	ufbxw_anim_stack stack;
	ufbxw_real weight;
} ufbxwi_anim_layer;

typedef struct {
	ufbxwi_element element;
	ufbxwi_id_list layers;

	ufbxw_ktime local_start;
	ufbxw_ktime local_stop;
	ufbxw_ktime reference_start;
	ufbxw_ktime reference_stop;
} ufbxwi_anim_stack;

typedef struct {
	ufbxwi_element element;
	ufbxwi_token type;
} ufbxwi_template;

typedef struct {
	ufbxwi_element element;
	ufbxw_string document_url;
	ufbxw_string src_document_url;
	ufbxw_string original_application_vendor;
	ufbxw_string original_application_name;
	ufbxw_string original_application_version;
	ufbxw_string original_date_time;
	ufbxw_string original_filename;
	ufbxw_string last_application_vendor;
	ufbxw_string last_application_name;
	ufbxw_string last_application_version;
	ufbxw_string last_date_time;
} ufbxwi_scene_info;

typedef struct {
	ufbxwi_element element;
	int32_t up_axis;
	int32_t up_axis_sign;
	int32_t front_axis;
	int32_t front_axis_sign;
	int32_t coord_axis;
	int32_t coord_axis_sign;

	int32_t original_up_axis;
	int32_t original_up_axis_sign;

	ufbxw_real unit_scale_factor;
	ufbxw_real original_unit_scale_factor;

	int32_t time_mode;
	int32_t time_protocol;
	int32_t snap_on_frame_mode;

	ufbxw_ktime time_span_start;
	ufbxw_ktime time_span_stop;
	ufbxw_real custom_frame_rate;

	int32_t current_time_marker;
} ufbxwi_global_settings;

typedef struct {
	ufbxwi_element element;
	ufbxw_id root_node;
} ufbxwi_document;

struct ufbxw_scene {
	ufbxwi_allocator ator;
	ufbxwi_error error;
	ufbxw_scene_opts opts;
	ufbxwi_string_pool string_pool;
	ufbxwi_buffer_pool buffers;

	ufbxwi_element_slot_list elements;
	ufbxwi_uint32_list free_element_ids;
	size_t num_elements;

	ufbxwi_object_type_list object_types;
	ufbxwi_element_type_list element_types;
	ufbxwi_prop_type_list prop_types;

	ufbxw_anim_stack default_anim_stack;
	ufbxw_anim_layer default_anim_layer;

	ufbxw_anim_stack active_anim_stack;

	ufbxwi_byte_list tmp_list;

	// TODO: Something better, hash set
	ufbxwi_id_list tmp_ids;

	bool override_creator;
	ufbxw_string creator;
};

#define ufbxwi_max_sizeof(a, b) (sizeof(a) > sizeof(b) ? sizeof(a) : sizeof(b))

typedef struct {
	uint8_t size;
	uint8_t alignment;
} ufbxwi_prop_data_info;

typedef struct {
	const char *type;
	const char *sub_type;
	ufbxw_prop_data_type data_type;
} ufbxwi_prop_type_desc;

static const ufbxwi_prop_data_info ufbxwi_prop_data_infos[] = {
	{ 0, 0 },
	{ sizeof(bool), sizeof(bool) },
	{ sizeof(int32_t), sizeof(int32_t) },
	{ sizeof(int64_t), sizeof(int64_t) },
	{ sizeof(ufbxw_real), sizeof(ufbxw_real) },
	{ sizeof(ufbxw_vec2), sizeof(ufbxw_real) },
	{ sizeof(ufbxw_vec3), sizeof(ufbxw_real) },
	{ sizeof(ufbxw_vec4), sizeof(ufbxw_real) },
	{ sizeof(ufbxw_string), ufbxwi_max_sizeof(char*, size_t) },
	{ sizeof(ufbxw_id), sizeof(ufbxw_id) },
	{ sizeof(ufbxw_real_string), ufbxwi_max_sizeof(ufbxw_real, char*) },
	{ sizeof(ufbxw_blob), ufbxwi_max_sizeof(void*, size_t) },
	{ sizeof(ufbxw_user_int), sizeof(int32_t) },
	{ sizeof(ufbxw_user_real), sizeof(ufbxw_real) },
	{ sizeof(ufbxw_user_enum), sizeof(int32_t) },
};

static const char *ufbxwi_prop_data_names[] = {
	"",
	"bool",
	"int32",
	"int64",
	"real",
	"vec2",
	"vec3",
	"vec4",
	"string",
	"id",
	"real_string",
	"blob",
	"user_int",
	"user_real",
	"user_enum",
};

ufbxw_static_assert(ufbxwi_prop_data_type_count, ufbxwi_arraycount(ufbxwi_prop_data_infos) == UFBXW_PROP_DATA_TYPE_COUNT);
ufbxw_static_assert(ufbxwi_prop_data_name_count, ufbxwi_arraycount(ufbxwi_prop_data_names) == UFBXW_PROP_DATA_TYPE_COUNT);

static const ufbxwi_prop_type_desc ufbxwi_prop_types[] = {
	{ "", "", UFBXW_PROP_DATA_NONE },
	{ "Compound", "", UFBXW_PROP_DATA_NONE },
	{ "bool", "", UFBXW_PROP_DATA_BOOL },
	{ "enum", "", UFBXW_PROP_DATA_INT32 },
	{ "int", "Integer", UFBXW_PROP_DATA_INT32 },
	{ "KTime", "Time", UFBXW_PROP_DATA_INT64 },
	{ "UByte", "", UFBXW_PROP_DATA_INT64 },
	{ "ULongLong", "", UFBXW_PROP_DATA_INT64 },
	{ "Float", "", UFBXW_PROP_DATA_REAL },
	{ "double", "Number", UFBXW_PROP_DATA_REAL },
	{ "Number", "", UFBXW_PROP_DATA_REAL },
	{ "Vector2D", "Vector2", UFBXW_PROP_DATA_VEC2 },
	{ "Vector", "", UFBXW_PROP_DATA_VEC3 },
	{ "Vector3D", "Vector", UFBXW_PROP_DATA_VEC3 },
	{ "Color", "", UFBXW_PROP_DATA_VEC3 },
	{ "ColorRGB", "Color", UFBXW_PROP_DATA_VEC3 },
	{ "ColorAndAlpha", "", UFBXW_PROP_DATA_VEC4 },
	{ "KString", "", UFBXW_PROP_DATA_STRING },
	{ "KString", "Url", UFBXW_PROP_DATA_STRING },
	{ "KString", "XRefUrl", UFBXW_PROP_DATA_STRING },
	{ "DateTime", "", UFBXW_PROP_DATA_STRING },
	{ "object", "", UFBXW_PROP_DATA_ID },
	{ "Distance", "", UFBXW_PROP_DATA_REAL_STRING },
	{ "Blob", "", UFBXW_PROP_DATA_BLOB },
	{ "Bool", "", UFBXW_PROP_DATA_BOOL },
	{ "Vector", "", UFBXW_PROP_DATA_VEC3 },
	{ "Integer", "", UFBXW_PROP_DATA_USER_INT },
	{ "Number", "", UFBXW_PROP_DATA_USER_REAL },
	{ "KString", "", UFBXW_PROP_DATA_STRING },
	{ "Enum", "", UFBXW_PROP_DATA_USER_ENUM },
	{ "Visibility", "", UFBXW_PROP_DATA_REAL },
	{ "Visibility Inheritance", "", UFBXW_PROP_DATA_BOOL },
	{ "Roll", "", UFBXW_PROP_DATA_REAL },
	{ "OpticalCenterX", "", UFBXW_PROP_DATA_REAL },
	{ "OpticalCenterY", "", UFBXW_PROP_DATA_REAL },
	{ "FieldOfViewX", "", UFBXW_PROP_DATA_REAL },
	{ "FieldOfViewX", "", UFBXW_PROP_DATA_REAL },
	{ "FieldOfViewY", "", UFBXW_PROP_DATA_REAL },
	{ "Lcl Translation", "", UFBXW_PROP_DATA_VEC3 },
	{ "Lcl Rotation", "", UFBXW_PROP_DATA_VEC3 },
	{ "Lcl Scaling", "", UFBXW_PROP_DATA_VEC3 },
};

ufbxw_static_assert(ufbxwi_prop_type_count, ufbxwi_arraycount(ufbxwi_prop_types) == UFBXW_PROP_TYPE_FIRST_CUSTOM);

typedef struct {
	ufbxwi_token type;
} ufbxwi_object_desc;

static const ufbxwi_object_desc ufbxwi_object_types[] = {
	{ UFBXWI_GlobalSettings },
	{ UFBXWI_AnimationStack },
	{ UFBXWI_AnimationLayer },
	{ UFBXWI_NodeAttribute },
	{ UFBXWI_Geometry },
	{ UFBXWI_Material },
	{ UFBXWI_Texture },
	{ UFBXWI_Model },
	{ UFBXWI_AnimationCurveNode },
	{ UFBXWI_AnimationCurve },
	{ UFBXWI_Deformer },
	{ UFBXWI_Pose },
};

typedef struct {
	union {
		bool bool_;
		int32_t int32_t;
		int64_t int64_t;
		ufbxw_real real;
		ufbxw_vec2 vec2;
		ufbxw_vec3 vec3;
		ufbxw_vec4 vec4;
		ufbxw_string string;
		ufbxw_id id;
		ufbxw_real_string real_string;
		ufbxw_blob blob;
		ufbxw_user_int user_int;
		ufbxw_user_real user_real;
		ufbxw_user_enum user_enum;
	} zero;
	bool bool_true;
	int32_t int_1;
	int32_t int_2;
	int32_t int_7;
	int32_t int_neg1;
	ufbxw_real double_1;
	ufbxw_real double_10;
	ufbxw_real double_20;
	ufbxw_real double_45;
	ufbxw_real double_50;
	ufbxw_real double_100;
	ufbxw_real double_200;
	ufbxw_real double_320;
	ufbxw_real double_4000;
	ufbxw_real double_0_5;
	ufbxw_real double_0_816;
	ufbxw_real double_0_612;
	ufbxw_real double_0_77777;
	ufbxw_real double_1_333d;
	ufbxw_real double_3_5;
	ufbxw_vec3 vec3_1;
	ufbxw_vec3 vec3_color;
	ufbxw_vec3 vec3_color30;
	ufbxw_vec3 vec3_color63;
	ufbxw_vec3 vec3_y;
	ufbxw_string string_empty;
	ufbxw_string string_default;
	ufbxw_string string_lambert;
	ufbxw_string string_producer_perspective;
} ufbxwi_prop_defaults;

static const ufbxwi_prop_defaults ufbxwi_prop_default_data = {
	{ 0 }, // zero
	true,
	1,
	2,
	7,
	-1,
	(ufbxw_real)1.0,
	(ufbxw_real)10.0,
	(ufbxw_real)20.0,
	(ufbxw_real)45.0,
	(ufbxw_real)50.0,
	(ufbxw_real)100.0,
	(ufbxw_real)200.0,
	(ufbxw_real)320.0,
	(ufbxw_real)4000.0,
	(ufbxw_real)0.5,
	(ufbxw_real)0.816,
	(ufbxw_real)0.612,
	(ufbxw_real)0.77777,
	(ufbxw_real)1.33333333333333,
	(ufbxw_real)3.5,
	{ 1.0f, 1.0f, 1.0f },
	{ (ufbxw_real)0.8, (ufbxw_real)0.8, (ufbxw_real)0.8 },
	{ (ufbxw_real)0.3, (ufbxw_real)0.3, (ufbxw_real)0.3 },
	{ (ufbxw_real)0.63, (ufbxw_real)0.63, (ufbxw_real)0.63 },
	{ 0.0f, 1.0f, 0.0f },
	{ ufbxwi_empty_char, 0 },
	{ "default", 7 },
	{ "lambert", 7 },
	{ "Producer Perspective", 20 },
};

enum {
	UFBXWI_PROP_FLAG_EXCLUDE_FROM_TEMPLATE = 0x1000,
};

// Static default/field helpers, the results always fit to 16 bits.
#define ufbxwi_default(m_field) (uint16_t)ufbxwi_prop_value_default(offsetof(ufbxwi_prop_defaults, m_field))
#define ufbxwi_field(m_type, m_field) (uint16_t)ufbxwi_prop_value_field(offsetof(m_type, m_field))

typedef struct {
	uint16_t name;
	uint8_t type;
	uint16_t value_offset;
	uint16_t default_offset;
	uint16_t flags;
} ufbxwi_prop_desc;

static const ufbxwi_prop_desc ufbxwi_node_props[] = {
	{ UFBXWI_QuaternionInterpolate, UFBXW_PROP_TYPE_ENUM, },
	{ UFBXWI_RotationOffset, UFBXW_PROP_TYPE_VECTOR3D, ufbxwi_field(ufbxwi_node, rotation_offset) },
	{ UFBXWI_RotationPivot, UFBXW_PROP_TYPE_VECTOR3D, ufbxwi_field(ufbxwi_node, rotation_pivot) },
	{ UFBXWI_ScalingOffset, UFBXW_PROP_TYPE_VECTOR3D, ufbxwi_field(ufbxwi_node, scaling_offset) },
	{ UFBXWI_ScalingPivot, UFBXW_PROP_TYPE_VECTOR3D, ufbxwi_field(ufbxwi_node, scaling_pivot) },
	{ UFBXWI_TranslationActive, UFBXW_PROP_TYPE_BOOL, },
	{ UFBXWI_TranslationMin, UFBXW_PROP_TYPE_VECTOR3D, },
	{ UFBXWI_TranslationMax, UFBXW_PROP_TYPE_VECTOR3D, },
	{ UFBXWI_TranslationMinX, UFBXW_PROP_TYPE_BOOL, },
	{ UFBXWI_TranslationMinY, UFBXW_PROP_TYPE_BOOL, },
	{ UFBXWI_TranslationMinZ, UFBXW_PROP_TYPE_BOOL, },
	{ UFBXWI_TranslationMaxX, UFBXW_PROP_TYPE_BOOL, },
	{ UFBXWI_TranslationMaxY, UFBXW_PROP_TYPE_BOOL, },
	{ UFBXWI_TranslationMaxZ, UFBXW_PROP_TYPE_BOOL, },
	{ UFBXWI_RotationOrder, UFBXW_PROP_TYPE_ENUM, ufbxwi_field(ufbxwi_node, rotation_order) },
	{ UFBXWI_RotationSpaceForLimitOnly, UFBXW_PROP_TYPE_BOOL, },
	{ UFBXWI_RotationStiffnessX, UFBXW_PROP_TYPE_DOUBLE, },
	{ UFBXWI_RotationStiffnessY, UFBXW_PROP_TYPE_DOUBLE, },
	{ UFBXWI_RotationStiffnessZ, UFBXW_PROP_TYPE_DOUBLE, },
	{ UFBXWI_AxisLen, UFBXW_PROP_TYPE_DOUBLE, ufbxwi_default(double_10) },
	{ UFBXWI_PreRotation, UFBXW_PROP_TYPE_VECTOR3D, ufbxwi_field(ufbxwi_node, pre_rotation) },
	{ UFBXWI_PostRotation, UFBXW_PROP_TYPE_VECTOR3D, ufbxwi_field(ufbxwi_node, post_rotation) },
	{ UFBXWI_RotationActive, UFBXW_PROP_TYPE_BOOL, },
	{ UFBXWI_RotationMin, UFBXW_PROP_TYPE_VECTOR3D, },
	{ UFBXWI_RotationMax, UFBXW_PROP_TYPE_VECTOR3D, },
	{ UFBXWI_RotationMinX, UFBXW_PROP_TYPE_BOOL, },
	{ UFBXWI_RotationMinY, UFBXW_PROP_TYPE_BOOL, },
	{ UFBXWI_RotationMinZ, UFBXW_PROP_TYPE_BOOL, },
	{ UFBXWI_RotationMaxX, UFBXW_PROP_TYPE_BOOL, },
	{ UFBXWI_RotationMaxY, UFBXW_PROP_TYPE_BOOL, },
	{ UFBXWI_RotationMaxZ, UFBXW_PROP_TYPE_BOOL, },
	{ UFBXWI_InheritType, UFBXW_PROP_TYPE_ENUM, ufbxwi_field(ufbxwi_node, inherit_type) },
	{ UFBXWI_ScalingActive, UFBXW_PROP_TYPE_BOOL, },
	{ UFBXWI_ScalingMin, UFBXW_PROP_TYPE_VECTOR3D, },
	{ UFBXWI_ScalingMax, UFBXW_PROP_TYPE_VECTOR3D, ufbxwi_default(vec3_1) },
	{ UFBXWI_ScalingMinX, UFBXW_PROP_TYPE_BOOL, },
	{ UFBXWI_ScalingMinY, UFBXW_PROP_TYPE_BOOL, },
	{ UFBXWI_ScalingMinZ, UFBXW_PROP_TYPE_BOOL, },
	{ UFBXWI_ScalingMaxX, UFBXW_PROP_TYPE_BOOL, },
	{ UFBXWI_ScalingMaxY, UFBXW_PROP_TYPE_BOOL, },
	{ UFBXWI_ScalingMaxZ, UFBXW_PROP_TYPE_BOOL, },
	{ UFBXWI_GeometricTranslation, UFBXW_PROP_TYPE_VECTOR3D, ufbxwi_field(ufbxwi_node, geometric_translation) },
	{ UFBXWI_GeometricRotation, UFBXW_PROP_TYPE_VECTOR3D, ufbxwi_field(ufbxwi_node, geometric_rotation) },
	{ UFBXWI_GeometricScaling, UFBXW_PROP_TYPE_VECTOR3D, ufbxwi_field(ufbxwi_node, geometric_scaling), ufbxwi_default(vec3_1) },
	{ UFBXWI_MinDampRangeX, UFBXW_PROP_TYPE_DOUBLE, },
	{ UFBXWI_MinDampRangeY, UFBXW_PROP_TYPE_DOUBLE, },
	{ UFBXWI_MinDampRangeZ, UFBXW_PROP_TYPE_DOUBLE, },
	{ UFBXWI_MaxDampRangeX, UFBXW_PROP_TYPE_DOUBLE, },
	{ UFBXWI_MaxDampRangeY, UFBXW_PROP_TYPE_DOUBLE, },
	{ UFBXWI_MaxDampRangeZ, UFBXW_PROP_TYPE_DOUBLE, },
	{ UFBXWI_MinDampStrengthX, UFBXW_PROP_TYPE_DOUBLE, },
	{ UFBXWI_MinDampStrengthY, UFBXW_PROP_TYPE_DOUBLE, },
	{ UFBXWI_MinDampStrengthZ, UFBXW_PROP_TYPE_DOUBLE, },
	{ UFBXWI_MaxDampStrengthX, UFBXW_PROP_TYPE_DOUBLE, },
	{ UFBXWI_MaxDampStrengthY, UFBXW_PROP_TYPE_DOUBLE, },
	{ UFBXWI_MaxDampStrengthZ, UFBXW_PROP_TYPE_DOUBLE, },
	{ UFBXWI_PreferedAngleX, UFBXW_PROP_TYPE_DOUBLE, },
	{ UFBXWI_PreferedAngleY, UFBXW_PROP_TYPE_DOUBLE, },
	{ UFBXWI_PreferedAngleZ, UFBXW_PROP_TYPE_DOUBLE, },
	{ UFBXWI_LookAtProperty, UFBXW_PROP_TYPE_OBJECT, },
	{ UFBXWI_UpVectorProperty, UFBXW_PROP_TYPE_OBJECT, },
	{ UFBXWI_Show, UFBXW_PROP_TYPE_BOOL, ufbxwi_default(bool_true) },
	{ UFBXWI_NegativePercentShapeSupport, UFBXW_PROP_TYPE_BOOL, ufbxwi_default(bool_true) },
	{ UFBXWI_DefaultAttributeIndex, UFBXW_PROP_TYPE_INT, ufbxwi_field(ufbxwi_node, default_attribute_index), ufbxwi_default(int_neg1) },
	{ UFBXWI_Freeze, UFBXW_PROP_TYPE_BOOL, },
	{ UFBXWI_LODBox, UFBXW_PROP_TYPE_BOOL, },
	{ UFBXWI_Lcl_Translation, UFBXW_PROP_TYPE_LCL_TRANSLATION, ufbxwi_field(ufbxwi_node, lcl_translation), 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_Lcl_Rotation, UFBXW_PROP_TYPE_LCL_ROTATION, ufbxwi_field(ufbxwi_node, lcl_rotation), 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_Lcl_Scaling, UFBXW_PROP_TYPE_LCL_SCALING, ufbxwi_field(ufbxwi_node, lcl_scaling), ufbxwi_default(vec3_1), UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_Visibility, UFBXW_PROP_TYPE_VISIBILITY, ufbxwi_field(ufbxwi_node, visibility), ufbxwi_default(double_1), UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_Visibility_Inheritance, UFBXW_PROP_TYPE_VISIBILITY_INHERITANCE, ufbxwi_field(ufbxwi_node, visibility_inheritance), ufbxwi_default(bool_true) },
};

static const ufbxwi_prop_desc ufbxwi_mesh_props[] = {
	{ UFBXWI_Color, UFBXW_PROP_TYPE_COLOR_RGB, ufbxwi_default(vec3_color) },
	{ UFBXWI_BBoxMin, UFBXW_PROP_TYPE_VECTOR3D },
	{ UFBXWI_BBoxMax, UFBXW_PROP_TYPE_VECTOR3D },
	{ UFBXWI_Primary_Visibility, UFBXW_PROP_TYPE_BOOL, ufbxwi_default(bool_true) },
	{ UFBXWI_Casts_Shadows, UFBXW_PROP_TYPE_BOOL, ufbxwi_default(bool_true) },
	{ UFBXWI_Receive_Shadows, UFBXW_PROP_TYPE_BOOL, ufbxwi_default(bool_true) },
};

static const ufbxwi_prop_desc ufbxwi_blend_channel_props[] = {
	{ UFBXWI_DeformPercent, UFBXW_PROP_TYPE_NUMBER, ufbxwi_field(ufbxwi_blend_channel, deform_percent), 0, UFBXW_PROP_FLAG_ANIMATABLE },
};

static const ufbxwi_prop_desc ufbxwi_light_props[] = {
	{ UFBXWI_Color, UFBXW_PROP_TYPE_COLOR, ufbxwi_field(ufbxwi_light, color), ufbxwi_default(vec3_1), UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_LightType, UFBXW_PROP_TYPE_ENUM, ufbxwi_field(ufbxwi_light, light_type) },
	{ UFBXWI_CastLightOnObject, UFBXW_PROP_TYPE_BOOL, ufbxwi_default(bool_true) },
	{ UFBXWI_DrawVolumetricLight, UFBXW_PROP_TYPE_BOOL, ufbxwi_default(bool_true) },
	{ UFBXWI_DrawGroundProjection, UFBXW_PROP_TYPE_BOOL, ufbxwi_default(bool_true) },
	{ UFBXWI_DrawFrontFacingVolumetricLight, UFBXW_PROP_TYPE_BOOL, },
	{ UFBXWI_Intensity, UFBXW_PROP_TYPE_NUMBER, ufbxwi_field(ufbxwi_light, intensity), ufbxwi_default(double_100), UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_InnerAngle, UFBXW_PROP_TYPE_NUMBER, ufbxwi_field(ufbxwi_light, inner_angle), 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_OuterAngle, UFBXW_PROP_TYPE_NUMBER, ufbxwi_field(ufbxwi_light, outer_angle), ufbxwi_default(double_45), UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_Fog, UFBXW_PROP_TYPE_NUMBER, ufbxwi_default(double_50), 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_DecayType, UFBXW_PROP_TYPE_ENUM, ufbxwi_field(ufbxwi_light, decay_type) },
	{ UFBXWI_DecayStart, UFBXW_PROP_TYPE_NUMBER, 0, 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_FileName, UFBXW_PROP_TYPE_STRING, ufbxwi_default(string_empty) },
	{ UFBXWI_EnableNearAttenuation, UFBXW_PROP_TYPE_BOOL, },
	{ UFBXWI_NearAttenuationStart, UFBXW_PROP_TYPE_NUMBER, 0, 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_NearAttenuationEnd, UFBXW_PROP_TYPE_NUMBER, 0, 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_EnableFarAttenuation, UFBXW_PROP_TYPE_BOOL, },
	{ UFBXWI_FarAttenuationStart, UFBXW_PROP_TYPE_NUMBER, 0, 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_FarAttenuationEnd, UFBXW_PROP_TYPE_NUMBER, 0, 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_CastShadows, UFBXW_PROP_TYPE_BOOL, ufbxwi_default(bool_true) },
	{ UFBXWI_ShadowColor, UFBXW_PROP_TYPE_COLOR, 0, 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_AreaLightShape, UFBXW_PROP_TYPE_ENUM, },
	{ UFBXWI_LeftBarnDoor, UFBXW_PROP_TYPE_FLOAT, ufbxwi_default(double_20), 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_RightBarnDoor, UFBXW_PROP_TYPE_FLOAT, ufbxwi_default(double_20), 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_TopBarnDoor, UFBXW_PROP_TYPE_FLOAT, ufbxwi_default(double_20), 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_BottomBarnDoor, UFBXW_PROP_TYPE_FLOAT, ufbxwi_default(double_20), 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_EnableBarnDoor, UFBXW_PROP_TYPE_USER_BOOL, 0, 0, UFBXW_PROP_FLAG_ANIMATABLE },
};

static const ufbxwi_prop_desc ufbxwi_camera_props[] = {
	{ UFBXWI_Color, UFBXW_PROP_TYPE_COLOR_RGB, ufbxwi_default(vec3_color) },
	{ UFBXWI_Position, UFBXW_PROP_TYPE_VECTOR, 0, 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_UpVector, UFBXW_PROP_TYPE_VECTOR, ufbxwi_default(vec3_y), 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_InterestPosition, UFBXW_PROP_TYPE_VECTOR, 0, 0, UFBXW_PROP_FLAG_ANIMATABLE  },
	{ UFBXWI_Roll, UFBXW_PROP_TYPE_ROLL, 0, 0, UFBXW_PROP_FLAG_ANIMATABLE  },
	{ UFBXWI_OpticalCenterX, UFBXW_PROP_TYPE_OPTICAL_CENTER_X, 0, 0, UFBXW_PROP_FLAG_ANIMATABLE  },
	{ UFBXWI_OpticalCenterY, UFBXW_PROP_TYPE_OPTICAL_CENTER_Y, 0, 0, UFBXW_PROP_FLAG_ANIMATABLE  },
	{ UFBXWI_BackgroundColor, UFBXW_PROP_TYPE_COLOR, ufbxwi_default(vec3_color63), 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_TurnTable, UFBXW_PROP_TYPE_NUMBER, 0, 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_DisplayTurnTableIcon, UFBXW_PROP_TYPE_BOOL },
	{ UFBXWI_UseMotionBlur, UFBXW_PROP_TYPE_BOOL },
	{ UFBXWI_UseRealTimeMotionBlur, UFBXW_PROP_TYPE_BOOL, ufbxwi_default(bool_true) },
	{ UFBXWI_Motion_Blur_Intensity, UFBXW_PROP_TYPE_NUMBER, ufbxwi_default(double_1), 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_AspectRatioMode, UFBXW_PROP_TYPE_ENUM },
	{ UFBXWI_AspectWidth, UFBXW_PROP_TYPE_DOUBLE, ufbxwi_default(double_320) },
	{ UFBXWI_AspectHeight, UFBXW_PROP_TYPE_DOUBLE, ufbxwi_default(double_200) },
	{ UFBXWI_PixelAspectRatio, UFBXW_PROP_TYPE_DOUBLE, ufbxwi_default(double_1) },
	{ UFBXWI_FilmOffsetX, UFBXW_PROP_TYPE_NUMBER, 0, 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_FilmOffsetY, UFBXW_PROP_TYPE_NUMBER, 0, 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_FilmWidth, UFBXW_PROP_TYPE_DOUBLE, ufbxwi_default(double_0_816) },
	{ UFBXWI_FilmHeight, UFBXW_PROP_TYPE_DOUBLE, ufbxwi_default(double_0_612) },
	{ UFBXWI_FilmAspectRatio, UFBXW_PROP_TYPE_DOUBLE, ufbxwi_default(double_1_333d) },
	{ UFBXWI_FilmSqueezeRatio, UFBXW_PROP_TYPE_DOUBLE, ufbxwi_default(double_1) },
	{ UFBXWI_FilmFormatIndex, UFBXW_PROP_TYPE_ENUM },
	{ UFBXWI_PreScale, UFBXW_PROP_TYPE_NUMBER, ufbxwi_default(double_1), 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_FilmTranslateX, UFBXW_PROP_TYPE_NUMBER, 0, 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_FilmTranslateY, UFBXW_PROP_TYPE_NUMBER, 0, 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_FilmRollPivotX, UFBXW_PROP_TYPE_NUMBER, 0, 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_FilmRollPivotY, UFBXW_PROP_TYPE_NUMBER, 0, 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_FilmRollValue, UFBXW_PROP_TYPE_NUMBER, 0, 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_FilmRollOrder, UFBXW_PROP_TYPE_ENUM },
	{ UFBXWI_ApertureMode, UFBXW_PROP_TYPE_ENUM, ufbxwi_default(int_2) },
	{ UFBXWI_GateFit, UFBXW_PROP_TYPE_ENUM },
	{ UFBXWI_FieldOfView, UFBXW_PROP_TYPE_FIELD_OF_VIEW, ufbxwi_default(double_45) },
	{ UFBXWI_FieldOfViewX, UFBXW_PROP_TYPE_FIELD_OF_VIEW_X, ufbxwi_default(double_45) },
	{ UFBXWI_FieldOfViewY, UFBXW_PROP_TYPE_FIELD_OF_VIEW_Y, ufbxwi_default(double_45) },
	{ UFBXWI_FocalLength, UFBXW_PROP_TYPE_NUMBER, ufbxwi_default(double_45), 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_CameraFormat, UFBXW_PROP_TYPE_ENUM },
	{ UFBXWI_UseFrameColor, UFBXW_PROP_TYPE_BOOL },
	{ UFBXWI_FrameColor, UFBXW_PROP_TYPE_COLOR_RGB, ufbxwi_default(vec3_color30) },
	{ UFBXWI_ShowName, UFBXW_PROP_TYPE_BOOL, ufbxwi_default(bool_true) },
	{ UFBXWI_ShowInfoOnMoving, UFBXW_PROP_TYPE_BOOL, ufbxwi_default(bool_true) },
	{ UFBXWI_ShowGrid, UFBXW_PROP_TYPE_BOOL, ufbxwi_default(bool_true) },
	{ UFBXWI_ShowOpticalCenter, UFBXW_PROP_TYPE_BOOL, ufbxwi_default(bool_true) },
	{ UFBXWI_ShowAzimut, UFBXW_PROP_TYPE_BOOL, ufbxwi_default(bool_true) },
	{ UFBXWI_ShowTimeCode, UFBXW_PROP_TYPE_BOOL },
	{ UFBXWI_ShowAudio, UFBXW_PROP_TYPE_BOOL },
	{ UFBXWI_AudioColor, UFBXW_PROP_TYPE_VECTOR, ufbxwi_default(vec3_y) },
	{ UFBXWI_NearPlane, UFBXW_PROP_TYPE_DOUBLE, ufbxwi_default(double_10) },
	{ UFBXWI_FarPlane, UFBXW_PROP_TYPE_DOUBLE, ufbxwi_default(double_4000) },
	{ UFBXWI_AutoComputeClipPanes, UFBXW_PROP_TYPE_BOOL },
	{ UFBXWI_ViewCameraToLookAt, UFBXW_PROP_TYPE_BOOL, ufbxwi_default(bool_true) },
	{ UFBXWI_ViewFrustumNearFarPlane, UFBXW_PROP_TYPE_BOOL },
	{ UFBXWI_ViewFrustumBackPlaneMode, UFBXW_PROP_TYPE_ENUM, ufbxwi_default(int_2) },
	{ UFBXWI_BackPlaneDistance, UFBXW_PROP_TYPE_NUMBER, ufbxwi_default(double_4000), 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_BackPlaneDistanceMode, UFBXW_PROP_TYPE_ENUM, ufbxwi_default(int_1) },
	{ UFBXWI_ViewFrustumFrontPlaneMode, UFBXW_PROP_TYPE_ENUM, ufbxwi_default(int_2) },
	{ UFBXWI_FrontPlaneDistance, UFBXW_PROP_TYPE_NUMBER, ufbxwi_default(double_10), 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_FrontPlaneDistanceMode, UFBXW_PROP_TYPE_ENUM, ufbxwi_default(int_1) },
	{ UFBXWI_LockMode, UFBXW_PROP_TYPE_BOOL },
	{ UFBXWI_LockInterestNavigation, UFBXW_PROP_TYPE_BOOL },
	{ UFBXWI_BackPlateFitImage, UFBXW_PROP_TYPE_BOOL },
	{ UFBXWI_BackPlateCrop, UFBXW_PROP_TYPE_BOOL },
	{ UFBXWI_BackPlateCenter, UFBXW_PROP_TYPE_BOOL, ufbxwi_default(bool_true) },
	{ UFBXWI_BackPlateKeepRatio, UFBXW_PROP_TYPE_BOOL, ufbxwi_default(bool_true) },
	{ UFBXWI_BackgroundAlphaTreshold, UFBXW_PROP_TYPE_DOUBLE, ufbxwi_default(double_0_5) },
	{ UFBXWI_ShowBackplate, UFBXW_PROP_TYPE_BOOL, ufbxwi_default(bool_true) },
	{ UFBXWI_BackPlaneOffsetX, UFBXW_PROP_TYPE_NUMBER, 0, 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_BackPlaneOffsetY, UFBXW_PROP_TYPE_NUMBER, 0, 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_BackPlaneRotation, UFBXW_PROP_TYPE_NUMBER, 0, 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_BackPlaneScaleX, UFBXW_PROP_TYPE_NUMBER, ufbxwi_default(double_1), 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_BackPlaneScaleY, UFBXW_PROP_TYPE_NUMBER, ufbxwi_default(double_1), 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_Background_Texture, UFBXW_PROP_TYPE_OBJECT },
	{ UFBXWI_FrontPlateFitImage, UFBXW_PROP_TYPE_BOOL, ufbxwi_default(bool_true) },
	{ UFBXWI_FrontPlateCrop, UFBXW_PROP_TYPE_BOOL, },
	{ UFBXWI_FrontPlateCenter, UFBXW_PROP_TYPE_BOOL, ufbxwi_default(bool_true) },
	{ UFBXWI_FrontPlateKeepRatio, UFBXW_PROP_TYPE_BOOL, ufbxwi_default(bool_true) },
	{ UFBXWI_Foreground_Opacity, UFBXW_PROP_TYPE_DOUBLE, },
	{ UFBXWI_ShowFrontplate, UFBXW_PROP_TYPE_BOOL, ufbxwi_default(bool_true) },
	{ UFBXWI_FrontPlaneOffsetX, UFBXW_PROP_TYPE_NUMBER, 0, 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_FrontPlaneOffsetY, UFBXW_PROP_TYPE_NUMBER, 0, 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_FrontPlaneRotation, UFBXW_PROP_TYPE_NUMBER, 0, 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_FrontPlaneScaleX, UFBXW_PROP_TYPE_NUMBER, ufbxwi_default(double_1), 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_FrontPlaneScaleY, UFBXW_PROP_TYPE_NUMBER, ufbxwi_default(double_1), 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_Foreground_Texture, UFBXW_PROP_TYPE_OBJECT },
	{ UFBXWI_DisplaySafeArea, UFBXW_PROP_TYPE_BOOL },
	{ UFBXWI_DisplaySafeAreaOnRender, UFBXW_PROP_TYPE_BOOL },
	{ UFBXWI_SafeAreaDisplayStyle, UFBXW_PROP_TYPE_ENUM, ufbxwi_default(int_1) },
	{ UFBXWI_SafeAreaAspectRatio, UFBXW_PROP_TYPE_DOUBLE, ufbxwi_default(double_1_333d) },
	{ UFBXWI_Use2DMagnifierZoom, UFBXW_PROP_TYPE_BOOL },
	{ UFBXWI_2D_Magnifier_Zoom, UFBXW_PROP_TYPE_NUMBER, ufbxwi_default(double_100), 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_2D_Magnifier_X, UFBXW_PROP_TYPE_NUMBER, ufbxwi_default(double_50), 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_2D_Magnifier_Y, UFBXW_PROP_TYPE_NUMBER, ufbxwi_default(double_50), 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_CameraProjectionType, UFBXW_PROP_TYPE_ENUM },
	{ UFBXWI_OrthoZoom, UFBXW_PROP_TYPE_DOUBLE, ufbxwi_default(double_1) },
	{ UFBXWI_UseRealTimeDOFAndAA, UFBXW_PROP_TYPE_BOOL },
	{ UFBXWI_UseDepthOfField, UFBXW_PROP_TYPE_BOOL },
	{ UFBXWI_FocusSource, UFBXW_PROP_TYPE_ENUM },
	{ UFBXWI_FocusAngle, UFBXW_PROP_TYPE_DOUBLE, ufbxwi_default(double_3_5) },
	{ UFBXWI_FocusDistance, UFBXW_PROP_TYPE_DOUBLE, ufbxwi_default(double_200) },
	{ UFBXWI_UseAntialiasing, UFBXW_PROP_TYPE_BOOL },
	{ UFBXWI_AntialiasingIntensity, UFBXW_PROP_TYPE_DOUBLE, ufbxwi_default(double_0_77777) },
	{ UFBXWI_AntialiasingMethod, UFBXW_PROP_TYPE_ENUM },
	{ UFBXWI_UseAccumulationBuffer, UFBXW_PROP_TYPE_BOOL },
	{ UFBXWI_FrameSamplingCount, UFBXW_PROP_TYPE_INT, ufbxwi_default(int_7) },
	{ UFBXWI_FrameSamplingType, UFBXW_PROP_TYPE_ENUM, ufbxwi_default(int_1) },
};

static const ufbxwi_prop_desc ufbxwi_skeleton_props[] = {
	{ UFBXWI_Color, UFBXW_PROP_TYPE_COLOR, ufbxwi_default(vec3_color) },
	{ UFBXWI_Size, UFBXW_PROP_TYPE_DOUBLE, ufbxwi_field(ufbxwi_skeleton, size), ufbxwi_default(double_100) },
	{ UFBXWI_LimbLength, UFBXW_PROP_TYPE_DOUBLE, ufbxwi_field(ufbxwi_skeleton, limb_length), ufbxwi_default(double_1) },
};

static const ufbxwi_prop_desc ufbxwi_scene_info_props[] = {
	{ UFBXWI_DocumentUrl, UFBXW_PROP_TYPE_URL, ufbxwi_field(ufbxwi_scene_info, document_url) },
	{ UFBXWI_SrcDocumentUrl, UFBXW_PROP_TYPE_URL, ufbxwi_field(ufbxwi_scene_info, src_document_url) },
	{ UFBXWI_Original, UFBXW_PROP_TYPE_COMPOUND },
	{ UFBXWI_Original_ApplicationVendor, UFBXW_PROP_TYPE_STRING, ufbxwi_field(ufbxwi_scene_info, original_application_vendor) },
	{ UFBXWI_Original_ApplicationName, UFBXW_PROP_TYPE_STRING, ufbxwi_field(ufbxwi_scene_info, original_application_name) },
	{ UFBXWI_Original_ApplicationVersion, UFBXW_PROP_TYPE_STRING, ufbxwi_field(ufbxwi_scene_info, original_application_version) },
	{ UFBXWI_Original_DateTime_GMT, UFBXW_PROP_TYPE_DATE_TIME, ufbxwi_field(ufbxwi_scene_info, original_date_time) },
	{ UFBXWI_Original_FileName, UFBXW_PROP_TYPE_STRING, ufbxwi_field(ufbxwi_scene_info, original_filename) },
	{ UFBXWI_LastSaved, UFBXW_PROP_TYPE_COMPOUND },
	{ UFBXWI_LastSaved_ApplicationVendor, UFBXW_PROP_TYPE_STRING, ufbxwi_field(ufbxwi_scene_info, last_application_vendor) },
	{ UFBXWI_LastSaved_ApplicationName, UFBXW_PROP_TYPE_STRING, ufbxwi_field(ufbxwi_scene_info, last_application_name) },
	{ UFBXWI_LastSaved_ApplicationVersion, UFBXW_PROP_TYPE_STRING, ufbxwi_field(ufbxwi_scene_info, last_application_version) },
	{ UFBXWI_LastSaved_DateTime_GMT, UFBXW_PROP_TYPE_DATE_TIME, ufbxwi_field(ufbxwi_scene_info, last_date_time) },
};

static const ufbxwi_prop_desc ufbxwi_global_settings_props[] = {
	{ UFBXWI_UpAxis, UFBXW_PROP_TYPE_INT, ufbxwi_field(ufbxwi_global_settings, up_axis) },
	{ UFBXWI_UpAxisSign, UFBXW_PROP_TYPE_INT, ufbxwi_field(ufbxwi_global_settings, up_axis_sign) },
	{ UFBXWI_FrontAxis, UFBXW_PROP_TYPE_INT, ufbxwi_field(ufbxwi_global_settings, front_axis) },
	{ UFBXWI_FrontAxisSign, UFBXW_PROP_TYPE_INT, ufbxwi_field(ufbxwi_global_settings, front_axis_sign) },
	{ UFBXWI_CoordAxis, UFBXW_PROP_TYPE_INT, ufbxwi_field(ufbxwi_global_settings, coord_axis) },
	{ UFBXWI_CoordAxisSign, UFBXW_PROP_TYPE_INT, ufbxwi_field(ufbxwi_global_settings, coord_axis_sign) },
	{ UFBXWI_OriginalUpAxis, UFBXW_PROP_TYPE_INT, ufbxwi_field(ufbxwi_global_settings, original_up_axis)},
	{ UFBXWI_OriginalUpAxisSign, UFBXW_PROP_TYPE_INT, ufbxwi_field(ufbxwi_global_settings, original_up_axis_sign) },
	{ UFBXWI_UnitScaleFactor, UFBXW_PROP_TYPE_DOUBLE, ufbxwi_field(ufbxwi_global_settings, unit_scale_factor) },
	{ UFBXWI_OriginalUnitScaleFactor, UFBXW_PROP_TYPE_DOUBLE, ufbxwi_field(ufbxwi_global_settings, original_unit_scale_factor) },
	{ UFBXWI_AmbientColor, UFBXW_PROP_TYPE_COLOR_RGB },
	{ UFBXWI_DefaultCamera, UFBXW_PROP_TYPE_STRING, ufbxwi_default(string_producer_perspective) },
	{ UFBXWI_TimeMode, UFBXW_PROP_TYPE_ENUM, ufbxwi_field(ufbxwi_global_settings, time_mode) },
	{ UFBXWI_TimeProtocol, UFBXW_PROP_TYPE_ENUM, ufbxwi_field(ufbxwi_global_settings, time_protocol) },
	{ UFBXWI_SnapOnFrameMode, UFBXW_PROP_TYPE_ENUM, ufbxwi_field(ufbxwi_global_settings, snap_on_frame_mode) },
	{ UFBXWI_TimeSpanStart, UFBXW_PROP_TYPE_TIME, ufbxwi_field(ufbxwi_global_settings, time_span_start) },
	{ UFBXWI_TimeSpanStop, UFBXW_PROP_TYPE_TIME, ufbxwi_field(ufbxwi_global_settings, time_span_stop) },
	{ UFBXWI_CustomFrameRate, UFBXW_PROP_TYPE_DOUBLE, ufbxwi_field(ufbxwi_global_settings, custom_frame_rate) },
	{ UFBXWI_TimeMarker, UFBXW_PROP_TYPE_COMPOUND, },
	{ UFBXWI_CurrentTimeMarker, UFBXW_PROP_TYPE_INT, ufbxwi_field(ufbxwi_global_settings, current_time_marker) },
};

static const ufbxwi_prop_desc ufbxwi_material_lambert_props[] = {
	{ UFBXWI_ShadingModel, UFBXW_PROP_TYPE_STRING, ufbxwi_default(string_lambert) },
	{ UFBXWI_MultiLayer, UFBXW_PROP_TYPE_BOOL, },
	{ UFBXWI_EmissiveColor, UFBXW_PROP_TYPE_COLOR, },
	{ UFBXWI_EmissiveFactor, UFBXW_PROP_TYPE_NUMBER, },
	{ UFBXWI_AmbientColor, UFBXW_PROP_TYPE_COLOR, },
	{ UFBXWI_AmbientFactor, UFBXW_PROP_TYPE_NUMBER, },
	{ UFBXWI_DiffuseColor, UFBXW_PROP_TYPE_COLOR, ufbxwi_default(vec3_color) },
	{ UFBXWI_DiffuseFactor, UFBXW_PROP_TYPE_NUMBER, ufbxwi_default(double_1) },
	{ UFBXWI_Bump, UFBXW_PROP_TYPE_VECTOR3D, },
	{ UFBXWI_NormalMap, UFBXW_PROP_TYPE_VECTOR3D, },
	{ UFBXWI_BumpFactor, UFBXW_PROP_TYPE_DOUBLE, },
	{ UFBXWI_TransparentColor, UFBXW_PROP_TYPE_COLOR, },
	{ UFBXWI_TransparencyFactor, UFBXW_PROP_TYPE_NUMBER, },
	{ UFBXWI_DisplacementColor, UFBXW_PROP_TYPE_COLOR_RGB, },
	{ UFBXWI_DisplacementFactor, UFBXW_PROP_TYPE_DOUBLE, },
	{ UFBXWI_VectorDisplacementColor, UFBXW_PROP_TYPE_COLOR_RGB, },
	{ UFBXWI_VectorDisplacementFactor, UFBXW_PROP_TYPE_DOUBLE, },
};

static const ufbxwi_prop_desc ufbxwi_file_texture_props[] = {
	{ UFBXWI_TextureTypeUse, UFBXW_PROP_TYPE_ENUM, },
	{ UFBXWI_Texture_alpha, UFBXW_PROP_TYPE_NUMBER, ufbxwi_default(double_1), 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_CurrentMappingType, UFBXW_PROP_TYPE_ENUM, },
	{ UFBXWI_WrapModeU, UFBXW_PROP_TYPE_ENUM, },
	{ UFBXWI_WrapModeV, UFBXW_PROP_TYPE_ENUM, },
	{ UFBXWI_UVSwap, UFBXW_PROP_TYPE_BOOL, },
	{ UFBXWI_PremultiplyAlpha, UFBXW_PROP_TYPE_BOOL, ufbxwi_default(bool_true) },
	{ UFBXWI_Translation, UFBXW_PROP_TYPE_VECTOR, 0, 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_Rotation, UFBXW_PROP_TYPE_VECTOR, 0, 0, UFBXW_PROP_FLAG_ANIMATABLE  },
	{ UFBXWI_Scaling, UFBXW_PROP_TYPE_VECTOR, ufbxwi_default(vec3_1), 0, UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_TextureRotationPivot, UFBXW_PROP_TYPE_VECTOR3D, },
	{ UFBXWI_TextureScalingPivot, UFBXW_PROP_TYPE_VECTOR3D, },
	{ UFBXWI_CurrentTextureBlendMode, UFBXW_PROP_TYPE_ENUM, ufbxwi_default(int_1) },
	{ UFBXWI_UVSet, UFBXW_PROP_TYPE_STRING, ufbxwi_default(string_default) },
	{ UFBXWI_UseMaterial, UFBXW_PROP_TYPE_BOOL },
	{ UFBXWI_UseMipMap, UFBXW_PROP_TYPE_BOOL },
};

static const ufbxwi_prop_desc ufbxwi_anim_prop_props[] = {
	{ UFBXWI_d, UFBXW_PROP_TYPE_COMPOUND },
};

static const ufbxwi_prop_desc ufbxwi_anim_layer_props[] = {
	{ UFBXWI_Weight, UFBXW_PROP_TYPE_NUMBER, ufbxwi_field(ufbxwi_anim_layer, weight), ufbxwi_default(double_100), UFBXW_PROP_FLAG_ANIMATABLE },
	{ UFBXWI_Mute, UFBXW_PROP_TYPE_BOOL },
	{ UFBXWI_Solo, UFBXW_PROP_TYPE_BOOL },
	{ UFBXWI_Lock, UFBXW_PROP_TYPE_BOOL },
	{ UFBXWI_Color, UFBXW_PROP_TYPE_COLOR_RGB, ufbxwi_default(vec3_color), },
	{ UFBXWI_BlendMode, UFBXW_PROP_TYPE_ENUM, },
	{ UFBXWI_RotationAccumulationMode, UFBXW_PROP_TYPE_ENUM, },
	{ UFBXWI_ScaleAccumulationMode, UFBXW_PROP_TYPE_ENUM, },
	{ UFBXWI_BlendModeBypass, UFBXW_PROP_TYPE_ULONGLONG, },
};

static const ufbxwi_prop_desc ufbxwi_anim_stack_props[] = {
	{ UFBXWI_Description, UFBXW_PROP_TYPE_STRING, ufbxwi_default(string_empty) },
	{ UFBXWI_LocalStart, UFBXW_PROP_TYPE_TIME, ufbxwi_field(ufbxwi_anim_stack, local_start) },
	{ UFBXWI_LocalStop, UFBXW_PROP_TYPE_TIME, ufbxwi_field(ufbxwi_anim_stack, local_stop) },
	{ UFBXWI_ReferenceStart, UFBXW_PROP_TYPE_TIME, ufbxwi_field(ufbxwi_anim_stack, reference_start) },
	{ UFBXWI_ReferenceStop, UFBXW_PROP_TYPE_TIME, ufbxwi_field(ufbxwi_anim_stack, reference_stop) },
};

static const ufbxwi_prop_desc ufbxwi_document_props[] = {
	{ UFBXWI_SourceObject, UFBXW_PROP_TYPE_OBJECT },
	{ UFBXWI_ActiveAnimStackName, UFBXW_PROP_TYPE_STRING, ufbxwi_default(string_empty) },
};

#define UFBXWI_CONN_BIT_ANYTHING UINT64_C(0x0)
#define UFBXWI_CONN_BIT_ELEMENT UINT64_C(0x1)
#define UFBXWI_CONN_BIT_PROPERTY UINT64_C(0x2)

#define UFBXWI_CONN_BIT_TYPE_ANY UINT64_C(0x0)
#define UFBXWI_CONN_BIT_TYPE_NODE UINT64_C(0x10)
#define UFBXWI_CONN_BIT_TYPE_NODE_ATTRIBUTE UINT64_C(0x20)
#define UFBXWI_CONN_BIT_TYPE_MATERIAL UINT64_C(0x40)
#define UFBXWI_CONN_BIT_TYPE_TEXTURE UINT64_C(0x80)
#define UFBXWI_CONN_BIT_TYPE_ANIM_CURVE UINT64_C(0x100)
#define UFBXWI_CONN_BIT_TYPE_ANIM_PROP UINT64_C(0x200)
#define UFBXWI_CONN_BIT_TYPE_ANIM_LAYER UINT64_C(0x400)
#define UFBXWI_CONN_BIT_TYPE_ANIM_STACK UINT64_C(0x800)
#define UFBXWI_CONN_BIT_TYPE_MESH UINT64_C(0x1000)
#define UFBXWI_CONN_BIT_TYPE_DEFORMER UINT64_C(0x2000)
#define UFBXWI_CONN_BIT_TYPE_SKIN_DEFORMER UINT64_C(0x4000)
#define UFBXWI_CONN_BIT_TYPE_SKIN_CLUSTER UINT64_C(0x8000)
#define UFBXWI_CONN_BIT_TYPE_BLEND_DEFORMER UINT64_C(0x10000)
#define UFBXWI_CONN_BIT_TYPE_BLEND_CHANNEL UINT64_C(0x20000)
#define UFBXWI_CONN_BIT_TYPE_BLEND_SHAPE UINT64_C(0x40000)

#define UFBXWI_CONN_BIT_ELEMENT_NODE (UFBXWI_CONN_BIT_ELEMENT | UFBXWI_CONN_BIT_TYPE_NODE)
#define UFBXWI_CONN_BIT_ELEMENT_NODE_ATTRIBUTE (UFBXWI_CONN_BIT_ELEMENT | UFBXWI_CONN_BIT_TYPE_NODE_ATTRIBUTE)
#define UFBXWI_CONN_BIT_ELEMENT_MATERIAL (UFBXWI_CONN_BIT_ELEMENT | UFBXWI_CONN_BIT_TYPE_MATERIAL)
#define UFBXWI_CONN_BIT_ELEMENT_TEXTURE (UFBXWI_CONN_BIT_ELEMENT | UFBXWI_CONN_BIT_TYPE_TEXTURE)
#define UFBXWI_CONN_BIT_ELEMENT_MESH (UFBXWI_CONN_BIT_ELEMENT | UFBXWI_CONN_BIT_TYPE_MESH)
#define UFBXWI_CONN_BIT_ELEMENT_DEFORMER (UFBXWI_CONN_BIT_ELEMENT | UFBXWI_CONN_BIT_TYPE_DEFORMER)
#define UFBXWI_CONN_BIT_ELEMENT_SKIN_DEFORMER (UFBXWI_CONN_BIT_ELEMENT | UFBXWI_CONN_BIT_TYPE_SKIN_DEFORMER)
#define UFBXWI_CONN_BIT_ELEMENT_SKIN_CLUSTER (UFBXWI_CONN_BIT_ELEMENT | UFBXWI_CONN_BIT_TYPE_SKIN_CLUSTER)
#define UFBXWI_CONN_BIT_ELEMENT_BLEND_DEFORMER (UFBXWI_CONN_BIT_ELEMENT | UFBXWI_CONN_BIT_TYPE_BLEND_DEFORMER)
#define UFBXWI_CONN_BIT_ELEMENT_BLEND_CHANNEL (UFBXWI_CONN_BIT_ELEMENT | UFBXWI_CONN_BIT_TYPE_BLEND_CHANNEL)
#define UFBXWI_CONN_BIT_ELEMENT_BLEND_SHAPE (UFBXWI_CONN_BIT_ELEMENT | UFBXWI_CONN_BIT_TYPE_BLEND_SHAPE)
#define UFBXWI_CONN_BIT_ELEMENT_ANIM_CURVE (UFBXWI_CONN_BIT_ELEMENT | UFBXWI_CONN_BIT_TYPE_ANIM_CURVE)
#define UFBXWI_CONN_BIT_ELEMENT_ANIM_PROP (UFBXWI_CONN_BIT_ELEMENT | UFBXWI_CONN_BIT_TYPE_ANIM_PROP)
#define UFBXWI_CONN_BIT_ELEMENT_ANIM_LAYER (UFBXWI_CONN_BIT_ELEMENT | UFBXWI_CONN_BIT_TYPE_ANIM_LAYER)
#define UFBXWI_CONN_BIT_ELEMENT_ANIM_STACK (UFBXWI_CONN_BIT_ELEMENT | UFBXWI_CONN_BIT_TYPE_ANIM_STACK)

#define UFBXWI_CONN_BIT_PROPERTY_ANY (UFBXWI_CONN_BIT_PROPERTY | UFBXWI_CONN_BIT_TYPE_ANY)
#define UFBXWI_CONN_BIT_PROPERTY_MATERIAL (UFBXWI_CONN_BIT_PROPERTY | UFBXWI_CONN_BIT_TYPE_MATERIAL)
#define UFBXWI_CONN_BIT_PROPERTY_ANIM_PROP (UFBXWI_CONN_BIT_PROPERTY | UFBXWI_CONN_BIT_TYPE_ANIM_PROP)

typedef struct {
	uint16_t data_size;
	uint64_t conn_bits;
} ufbxwi_element_type_info;

static const ufbxwi_element_type_info ufbxwi_element_type_infos[] = {
	{ 0 }, // 0
	{ 0 }, // CUSTOM
	{ sizeof(ufbxwi_node), UFBXWI_CONN_BIT_TYPE_NODE },
	{ sizeof(ufbxwi_node_attribute), UFBXWI_CONN_BIT_TYPE_NODE_ATTRIBUTE },
	{ sizeof(ufbxwi_mesh), UFBXWI_CONN_BIT_TYPE_MESH | UFBXWI_CONN_BIT_TYPE_NODE_ATTRIBUTE },
	{ sizeof(ufbxwi_skin_deformer), UFBXWI_CONN_BIT_TYPE_SKIN_DEFORMER | UFBXWI_CONN_BIT_TYPE_DEFORMER },
	{ sizeof(ufbxwi_skin_cluster), UFBXWI_CONN_BIT_TYPE_SKIN_CLUSTER },
	{ sizeof(ufbxwi_blend_deformer), UFBXWI_CONN_BIT_TYPE_BLEND_DEFORMER | UFBXWI_CONN_BIT_TYPE_DEFORMER },
	{ sizeof(ufbxwi_blend_channel), UFBXWI_CONN_BIT_TYPE_BLEND_CHANNEL },
	{ sizeof(ufbxwi_blend_shape), UFBXWI_CONN_BIT_TYPE_BLEND_SHAPE },
	{ sizeof(ufbxwi_light), UFBXWI_CONN_BIT_TYPE_NODE_ATTRIBUTE },
	{ sizeof(ufbxwi_camera), UFBXWI_CONN_BIT_TYPE_NODE_ATTRIBUTE },
	{ sizeof(ufbxwi_skeleton), UFBXWI_CONN_BIT_TYPE_NODE_ATTRIBUTE },
	{ sizeof(ufbxwi_bind_pose) },
	{ sizeof(ufbxwi_material), UFBXWI_CONN_BIT_TYPE_MATERIAL },
	{ sizeof(ufbxwi_texture), UFBXWI_CONN_BIT_TYPE_TEXTURE },
	{ sizeof(ufbxwi_anim_curve), UFBXWI_CONN_BIT_TYPE_ANIM_CURVE },
	{ sizeof(ufbxwi_anim_prop), UFBXWI_CONN_BIT_TYPE_ANIM_PROP },
	{ sizeof(ufbxwi_anim_layer), UFBXWI_CONN_BIT_TYPE_ANIM_LAYER },
	{ sizeof(ufbxwi_anim_stack), UFBXWI_CONN_BIT_TYPE_ANIM_STACK },
	{ sizeof(ufbxwi_template) },
	{ sizeof(ufbxwi_scene_info) },
	{ sizeof(ufbxwi_global_settings) },
	{ sizeof(ufbxwi_document) },
};

static const char *ufbxwi_element_type_names[] = {
	"",
	"custom",
	"node",
	"node_attribute",
	"mesh",
	"skin_deformer",
	"skin_cluster",
	"blend_deformer",
	"blend_channel",
	"blend_shape",
	"light",
	"camera",
	"skeleton",
	"bind_pose",
	"material",
	"texture",
	"anim_curve",
	"anim_prop",
	"anim_layer",
	"anim_stack",
	"template",
	"scene_info",
	"global_settings",
	"document",
};

ufbxw_static_assert(ufbxwi_element_type_infos_count, ufbxwi_arraycount(ufbxwi_element_type_infos) == UFBXW_ELEMENT_TYPE_COUNT);
ufbxw_static_assert(ufbxwi_element_type_names_count, ufbxwi_arraycount(ufbxwi_element_type_names) == UFBXW_ELEMENT_TYPE_COUNT);

typedef enum {
	UFBXWI_CONN_TYPE_ID = 0x1,
	UFBXWI_CONN_TYPE_CONN = 0x2,
	UFBXWI_CONN_TYPE_ID_LIST = 0x3,
	UFBXWI_CONN_TYPE_CONN_LIST = 0x4,
	UFBXWI_CONN_TYPE_BLEND_SHAPE_LIST = 0x5,
	UFBXWI_CONN_TYPE_DATA_MASK = 0xf,

	UFBXWI_CONN_TYPE_UNORDERED = 0x10,
	UFBXWI_CONN_TYPE_SPARSE = 0x20,
} ufbxwi_conn_type;

#define ufbxwi_conn_make(conn_type, type, field) (uint32_t)((conn_type) << 24 | (offsetof(type, field)))
#define ufbxwi_conn_offset(conn) (uint32_t)((conn) & 0xffffff)
#define ufbxwi_conn_type(conn) (ufbxwi_conn_type)((conn) >> 24)

#define ufbxwi_conn_id(type, field) ufbxwi_conn_make(UFBXWI_CONN_TYPE_ID, type, field)
#define ufbxwi_conn_conn(type, field) ufbxwi_conn_make(UFBXWI_CONN_TYPE_CONN, type, field)
#define ufbxwi_conn_id_list(type, field) ufbxwi_conn_make(UFBXWI_CONN_TYPE_ID_LIST, type, field)
#define ufbxwi_conn_id_list_ex(type, field, flags) ufbxwi_conn_make(UFBXWI_CONN_TYPE_ID_LIST | (flags), type, field)
#define ufbxwi_conn_conn_list(type, field) ufbxwi_conn_make(UFBXWI_CONN_TYPE_CONN_LIST, type, field)
#define ufbxwi_conn_conn_list_ex(type, field, flags) ufbxwi_conn_make(UFBXWI_CONN_TYPE_CONN_LIST | (flags), type, field)
#define ufbxwi_conn_blend_shape_list(type, field) ufbxwi_conn_make(UFBXWI_CONN_TYPE_BLEND_SHAPE_LIST, type, field)

typedef struct {
	const char *debug_name;
	uint64_t src_mask;
	uint64_t dst_mask;
	uint32_t src_conn;
	uint32_t dst_conn;
} ufbxwi_connection_info;

static const ufbxwi_connection_info ufbxwi_connection_infos[] = {
	{ 0 }, // Invalid
	{ "Node Parent", UFBXWI_CONN_BIT_ELEMENT_NODE, UFBXWI_CONN_BIT_ELEMENT_NODE, ufbxwi_conn_id(ufbxwi_node, parent), ufbxwi_conn_id_list(ufbxwi_node, children) },
	{ "Node Attribute", UFBXWI_CONN_BIT_ELEMENT_NODE_ATTRIBUTE, UFBXWI_CONN_BIT_ELEMENT_NODE, ufbxwi_conn_id_list(ufbxwi_node_attribute, instances), ufbxwi_conn_id(ufbxwi_node, attribute) },
	{ "Node Material", UFBXWI_CONN_BIT_ELEMENT_MATERIAL, UFBXWI_CONN_BIT_ELEMENT_NODE, ufbxwi_conn_id_list_ex(ufbxwi_material, conn_nodes, UFBXWI_CONN_TYPE_UNORDERED), ufbxwi_conn_id_list_ex(ufbxwi_node, materials, UFBXWI_CONN_TYPE_SPARSE) },
	{ "Material Texture", UFBXWI_CONN_BIT_ELEMENT_TEXTURE, UFBXWI_CONN_BIT_PROPERTY_MATERIAL, ufbxwi_conn_conn_list_ex(ufbxwi_texture, conn_materials, UFBXWI_CONN_TYPE_UNORDERED), ufbxwi_conn_conn_list(ufbxwi_material, textures) },
	{ "Mesh Deformer", UFBXWI_CONN_BIT_ELEMENT_DEFORMER, UFBXWI_CONN_BIT_ELEMENT_MESH, ufbxwi_conn_id_list(ufbxwi_deformer, geometries), ufbxwi_conn_id_list(ufbxwi_mesh, deformers) },
	{ "Skin Cluster", UFBXWI_CONN_BIT_ELEMENT_SKIN_CLUSTER, UFBXWI_CONN_BIT_ELEMENT_SKIN_DEFORMER, ufbxwi_conn_id(ufbxwi_skin_cluster, deformer), ufbxwi_conn_id_list(ufbxwi_skin_deformer, clusters) },
	{ "Skin Cluster Node", UFBXWI_CONN_BIT_ELEMENT_NODE, UFBXWI_CONN_BIT_ELEMENT_SKIN_CLUSTER, ufbxwi_conn_id_list_ex(ufbxwi_node, skin_clusters, UFBXWI_CONN_TYPE_UNORDERED), ufbxwi_conn_id(ufbxwi_skin_cluster, node) },
	{ "Blend Channel", UFBXWI_CONN_BIT_ELEMENT_BLEND_CHANNEL, UFBXWI_CONN_BIT_ELEMENT_BLEND_DEFORMER, ufbxwi_conn_id(ufbxwi_blend_channel, deformer), ufbxwi_conn_id_list(ufbxwi_blend_deformer, channels) },
	{ "Blend Shape", UFBXWI_CONN_BIT_ELEMENT_BLEND_SHAPE, UFBXWI_CONN_BIT_ELEMENT_BLEND_CHANNEL, ufbxwi_conn_id(ufbxwi_blend_shape, blend_channels), ufbxwi_conn_blend_shape_list(ufbxwi_blend_channel, blend_shapes) },
	{ "Animated Property", UFBXWI_CONN_BIT_ELEMENT_ANIM_PROP, UFBXWI_CONN_BIT_PROPERTY_ANY, ufbxwi_conn_conn(ufbxwi_anim_prop, prop), ufbxwi_conn_conn_list_ex(ufbxwi_element, anim_props, UFBXWI_CONN_TYPE_UNORDERED) },
	{ "Animation Curve Property", UFBXWI_CONN_BIT_ELEMENT_ANIM_CURVE, UFBXWI_CONN_BIT_PROPERTY_ANIM_PROP, ufbxwi_conn_conn(ufbxwi_anim_curve, prop), ufbxwi_conn_conn_list(ufbxwi_anim_prop, curves) },
	{ "Animation Property Layer", UFBXWI_CONN_BIT_ELEMENT_ANIM_PROP, UFBXWI_CONN_BIT_ELEMENT_ANIM_LAYER, ufbxwi_conn_id(ufbxwi_anim_prop, layer), ufbxwi_conn_id_list_ex(ufbxwi_anim_layer, anim_props, UFBXWI_CONN_TYPE_UNORDERED) },
	{ "Animation Layer Stack", UFBXWI_CONN_BIT_ELEMENT_ANIM_LAYER, UFBXWI_CONN_BIT_ELEMENT_ANIM_STACK, ufbxwi_conn_id(ufbxwi_anim_layer, stack), ufbxwi_conn_id_list(ufbxwi_anim_stack, layers) },
	{ "User", UFBXWI_CONN_BIT_ANYTHING, UFBXWI_CONN_BIT_ANYTHING, ufbxwi_conn_conn_list(ufbxwi_element, user_conns_src), ufbxwi_conn_conn_list(ufbxwi_element, user_conns_dst) },
};

ufbxw_static_assert(ufbxwi_connection_infos_counst, ufbxwi_arraycount(ufbxwi_connection_infos) == UFBXW_CONNECTION_TYPE_COUNT);

enum {
	UFBXWI_CONNECT_FLAG_DISCONNECT_SRC = 0x1,
	UFBXWI_CONNECT_FLAG_DISCONNECT_DST = 0x2,
};

static bool ufbxwi_conn_add(ufbxw_scene *scene, ufbxwi_conn_type type, void *data, ufbxw_id id, ufbxwi_token src_prop, ufbxwi_token dst_prop)
{
	switch (type & UFBXWI_CONN_TYPE_DATA_MASK) {
	case UFBXWI_CONN_TYPE_ID: {
		ufbxw_id *d = (ufbxw_id*)data;
		if (*d == ufbxw_null_id) {
			*d = id;
			return true;
		}
	} break;
	case UFBXWI_CONN_TYPE_CONN: {
		ufbxwi_conn *d = (ufbxwi_conn*)data;
		if (d->id == ufbxw_null_id) {
			d->id = id;
			d->src_prop = src_prop;
			d->dst_prop = dst_prop;
			return true;
		}
	} break;
	case UFBXWI_CONN_TYPE_ID_LIST: {
		ufbxwi_id_list *d = (ufbxwi_id_list*)data;
		ufbxwi_check(ufbxwi_id_list_add(&scene->ator, d, id), false);
		return true;
	} break;
	case UFBXWI_CONN_TYPE_CONN_LIST: {
		ufbxwi_conn_list *d = (ufbxwi_conn_list*)data;
		ufbxwi_conn *conn = ufbxwi_list_push_uninit(&scene->ator, d, ufbxwi_conn);
		ufbxwi_check(conn, false);
		conn->id = id;
		conn->src_prop = src_prop;
		conn->dst_prop = dst_prop;
		return true;
	} break;
	case UFBXWI_CONN_TYPE_BLEND_SHAPE_LIST: {
		ufbxwi_blend_shape_conn_list *d = (ufbxwi_blend_shape_conn_list*)data;
		ufbxwi_blend_shape_conn *conn = ufbxwi_list_push_uninit(&scene->ator, d, ufbxwi_blend_shape_conn);
		ufbxwi_check(conn, false);
		conn->shape.id = id;
		conn->target_weight = (ufbxw_real)100.0;
		return true;
	} break;
	}

	return false;
}

static bool ufbxwi_conn_remove_one(ufbxw_scene *scene, ufbxwi_conn_type type, void *data, ufbxw_id id)
{
	switch (type & UFBXWI_CONN_TYPE_DATA_MASK) {
	case UFBXWI_CONN_TYPE_ID: {
		ufbxw_id *d = (ufbxw_id*)data;
		if (*d == id) {
			*d = ufbxw_null_id;
			return true;
		}
	} break;
	case UFBXWI_CONN_TYPE_CONN: {
		ufbxwi_conn *d = (ufbxwi_conn*)data;
		if (d->id == id) {
			d->id = ufbxw_null_id;
			d->src_prop = d->dst_prop = UFBXWI_TOKEN_NONE;
			return true;
		}
	} break;
	case UFBXWI_CONN_TYPE_ID_LIST: {
		ufbxwi_id_list *d = (ufbxwi_id_list*)data;
		// TODO: Unordered/sparse
		return ufbxwi_id_list_remove_one(d, id);
	} break;
	case UFBXWI_CONN_TYPE_CONN_LIST: {
		ufbxwi_conn_list *d = (ufbxwi_conn_list*)data;
		// TODO: Unordered/sparse
		ufbxwi_for_list(ufbxwi_conn, conn, *d) {
			if (conn->id == id) {
				ufbxwi_conn *last = d->data + d->count - 1;
				if (conn != last) {
					memmove(conn, conn + 1, ufbxwi_to_size(last - conn) * sizeof(ufbxwi_conn));
				}
				return true;
			}
		}
	} break;
	case UFBXWI_CONN_TYPE_BLEND_SHAPE_LIST: {
		ufbxwi_blend_shape_conn_list *d = (ufbxwi_blend_shape_conn_list*)data;
		// TODO: Unordered/sparse
		ufbxwi_for_list(ufbxwi_blend_shape_conn, conn, *d) {
			if (conn->shape.id == id) {
				ufbxwi_blend_shape_conn *last = d->data + d->count - 1;
				if (conn != last) {
					memmove(conn, conn + 1, ufbxwi_to_size(last - conn) * sizeof(ufbxwi_blend_shape_conn));
				}
				return true;
			}
		}
	} break;
	}

	return false;
}

static void ufbxwi_conn_remove_all(ufbxw_scene *scene, ufbxwi_conn_type type, void *data, ufbxw_id id)
{
	switch (type & UFBXWI_CONN_TYPE_DATA_MASK) {
	case UFBXWI_CONN_TYPE_ID: {
		ufbxw_id *d = (ufbxw_id*)data;
		if (*d == id) {
			*d = ufbxw_null_id;
		}
	} break;
	case UFBXWI_CONN_TYPE_CONN: {
		ufbxwi_conn *d = (ufbxwi_conn*)data;
		if (d->id == id) {
			d->id = ufbxw_null_id;
			d->src_prop = d->dst_prop = UFBXWI_TOKEN_NONE;
		}
	} break;
	case UFBXWI_CONN_TYPE_ID_LIST: {
		ufbxwi_id_list *d = (ufbxwi_id_list*)data;
		// TODO: Unordered/sparse
		ufbxw_id *ids = d->data;
		size_t dst = 0, count = d->count;
		for (size_t src = 0; src < count; src++) {
			if (ids[src] != id) {
				if (dst != src) ids[dst] = ids[src];
				dst++;
			}
		}
		d->count = dst;
	} break;
	case UFBXWI_CONN_TYPE_CONN_LIST: {
		ufbxwi_conn_list *d = (ufbxwi_conn_list*)data;
		// TODO: Unordered/sparse
		ufbxwi_conn *conns = d->data;
		size_t dst = 0, count = d->count;
		for (size_t src = 0; src < count; src++) {
			if (conns[src].id != id) {
				if (dst != src) conns[dst] = conns[src];
				dst++;
			}
		}
		d->count = dst;
	} break;
	case UFBXWI_CONN_TYPE_BLEND_SHAPE_LIST: {
		ufbxwi_blend_shape_conn_list *d = (ufbxwi_blend_shape_conn_list*)data;
		// TODO: Unordered/sparse
		ufbxwi_blend_shape_conn *conns = d->data;
		size_t dst = 0, count = d->count;
		for (size_t src = 0; src < count; src++) {
			if (conns[src].shape.id != id) {
				if (dst != src) conns[dst] = conns[src];
				dst++;
			}
		}
		d->count = dst;
	} break;
	}
}

static int ufbxwi_cmp_id(const void *va, const void *vb)
{
	ufbxw_id a = *(const ufbxw_id*)va, b = *(const ufbxw_id*)vb;
	if (a != b) return a < b ? -1 : +1;
	return 0;
}

static bool ufbxwi_conn_collect_ids(ufbxw_scene *scene, ufbxwi_id_list *ids, ufbxwi_conn_type type, const void *data)
{
	switch (type & UFBXWI_CONN_TYPE_DATA_MASK) {
	case UFBXWI_CONN_TYPE_ID: {
		const ufbxw_id *d = (const ufbxw_id*)data;
		if (*d != ufbxw_null_id) {
			ufbxwi_check(ufbxwi_list_push_copy(&scene->ator, ids, ufbxw_id, d), false);
			return true;
		}
	} break;
	case UFBXWI_CONN_TYPE_CONN: {
		const ufbxwi_conn *d = (const ufbxwi_conn*)data;
		if (d->id != ufbxw_null_id) {
			ufbxwi_check(ufbxwi_list_push_copy(&scene->ator, ids, ufbxw_id, &d->id), false);
			return true;
		}
	} break;
	case UFBXWI_CONN_TYPE_ID_LIST: {
		ufbxwi_id_list d = *(const ufbxwi_id_list*)data;
		ufbxwi_check(ufbxwi_list_push_copy_n(&scene->ator, ids, ufbxw_id, d.count, d.data), false);
	} break;
	case UFBXWI_CONN_TYPE_CONN_LIST: {
		ufbxwi_conn_list d = *(const ufbxwi_conn_list*)data;
		ufbxw_id *dst = ufbxwi_list_push_uninit_n(&scene->ator, ids, ufbxw_id, d.count);
		ufbxwi_check(dst, false);
		for (size_t i = 0; i < d.count; i++) {
			dst[i] = d.data[i].id;
		}
	} break;
	case UFBXWI_CONN_TYPE_BLEND_SHAPE_LIST: {
		ufbxwi_blend_shape_conn_list d = *(const ufbxwi_blend_shape_conn_list*)data;
		ufbxw_id *dst = ufbxwi_list_push_uninit_n(&scene->ator, ids, ufbxw_id, d.count);
		ufbxwi_check(dst, false);
		for (size_t i = 0; i < d.count; i++) {
			dst[i] = d.data[i].shape.id;
		}
	} break;
	}

	// Deduplicate found IDs
	if (ids->count > 1) {
		// TODO: Better sort, especially for small values
		qsort(ids->data, ids->count, sizeof(ufbxw_id), &ufbxwi_cmp_id);
		ufbxw_id prev = ufbxw_null_id;
		ufbxw_id *dst = ids->data;
		ufbxwi_for_list(ufbxw_id, p_id, *ids) {
			ufbxw_id id = *p_id;
			if (id != prev) {
				*dst++ = id;
				prev = id;
			}
		}
		ids->count = dst - ids->data;
	}

	return true;
}

static bool ufbxwi_conn_collect_conns(const ufbxw_scene *scene, ufbxwi_allocator *ator, ufbxwi_conn_list *conns, ufbxwi_conn_type type, const void *data)
{
	switch (type & UFBXWI_CONN_TYPE_DATA_MASK) {
	case UFBXWI_CONN_TYPE_ID: {
		const ufbxw_id *d = (const ufbxw_id*)data;
		if (*d != ufbxw_null_id) {
			ufbxwi_conn *conn = ufbxwi_list_push_uninit(ator, conns, ufbxwi_conn);
			ufbxwi_check(conn, false);
			conn->id = *d;
			conn->src_prop = UFBXWI_TOKEN_NONE;
			conn->dst_prop = UFBXWI_TOKEN_NONE;
			return true;
		}
	} break;
	case UFBXWI_CONN_TYPE_CONN: {
		const ufbxwi_conn *d = (const ufbxwi_conn*)data;
		if (d->id != ufbxw_null_id) {
			ufbxwi_check(ufbxwi_list_push_copy(ator, conns, ufbxwi_conn, d), false);
		}
	} break;
	case UFBXWI_CONN_TYPE_ID_LIST: {
		ufbxwi_id_list d = *(const ufbxwi_id_list*)data;
		ufbxwi_conn *dst = ufbxwi_list_push_uninit_n(ator, conns, ufbxwi_conn, d.count);
		ufbxwi_check(dst, false);
		for (size_t i = 0; i < d.count; i++) {
			dst[i].id = d.data[i];
			dst[i].src_prop = UFBXWI_TOKEN_NONE;
			dst[i].dst_prop = UFBXWI_TOKEN_NONE;
		}
	} break;
	case UFBXWI_CONN_TYPE_CONN_LIST: {
		ufbxwi_conn_list d = *(const ufbxwi_conn_list*)data;
		ufbxwi_check(ufbxwi_list_push_copy_n(ator, conns, ufbxwi_conn, d.count, d.data), false);
	} break;
	case UFBXWI_CONN_TYPE_BLEND_SHAPE_LIST: {
		ufbxwi_blend_shape_conn_list d = *(const ufbxwi_blend_shape_conn_list*)data;
		ufbxwi_conn *dst = ufbxwi_list_push_uninit_n(ator, conns, ufbxwi_conn, d.count);
		ufbxwi_check(dst, false);
		for (size_t i = 0; i < d.count; i++) {
			dst[i].id = d.data[i].shape.id;
			dst[i].src_prop = UFBXWI_TOKEN_NONE;
			dst[i].dst_prop = UFBXWI_TOKEN_NONE;
		}
	} break;
	}

	return true;
}

static void ufbxwi_conn_clear(ufbxw_scene *scene, ufbxwi_conn_type type, const void *data)
{
	switch (type & UFBXWI_CONN_TYPE_DATA_MASK) {
	case UFBXWI_CONN_TYPE_ID: {
		ufbxw_id *d = (ufbxw_id*)data;
		*d = ufbxw_null_id;
	} break;
	case UFBXWI_CONN_TYPE_CONN: {
		ufbxwi_conn *d = (ufbxwi_conn*)data;
		d->id = ufbxw_null_id;
		d->src_prop = d->dst_prop = UFBXWI_TOKEN_NONE;
	} break;
	case UFBXWI_CONN_TYPE_ID_LIST: {
		ufbxwi_id_list *d = (ufbxwi_id_list*)data;
		d->count = 0;
	} break;
	case UFBXWI_CONN_TYPE_CONN_LIST: {
		ufbxwi_conn_list *d = (ufbxwi_conn_list*)data;
		d->count = 0;
	} break;
	case UFBXWI_CONN_TYPE_BLEND_SHAPE_LIST: {
		ufbxwi_blend_shape_conn_list *d = (ufbxwi_blend_shape_conn_list*)data;
		d->count = 0;
	} break;
	}
}

static bool ufbxwi_init_node(ufbxw_scene *scene, void *data)
{
	ufbxwi_node *node = (ufbxwi_node*)data;
	node->lcl_scaling = ufbxwi_one_vec3;
	node->geometric_scaling = ufbxwi_one_vec3;
	node->visibility = 1.0;
	node->visibility_inheritance = true;
	return true;
}

static bool ufbxwi_init_light(ufbxw_scene *scene, void *data)
{
	ufbxwi_light *light = (ufbxwi_light*)data;
	light->color = ufbxwi_one_vec3;
	light->intensity = (ufbxw_real)100.0;
	light->outer_angle = (ufbxw_real)45.0;
	return true;
}

static bool ufbxwi_init_camera(ufbxw_scene *scene, void *data)
{
	ufbxwi_camera *camera = (ufbxwi_camera*)data;
	return true;
}

static bool ufbxwi_init_skeleton(ufbxw_scene *scene, void *data)
{
	ufbxwi_skeleton *skeleton = (ufbxwi_skeleton*)data;
	skeleton->size = (ufbxw_real)100.0;
	skeleton->limb_length = (ufbxw_real)1.0;
	return true;
}

static bool ufbxwi_init_scene_info(ufbxw_scene *scene, void *data)
{
	ufbxwi_scene_info *info = (ufbxwi_scene_info*)data;
	info->document_url.data = ufbxwi_empty_char;
	info->src_document_url.data = ufbxwi_empty_char;
	info->original_application_vendor.data = ufbxwi_empty_char;
	info->original_application_name.data = ufbxwi_empty_char;
	info->original_application_version.data = ufbxwi_empty_char;
	info->original_date_time.data = ufbxwi_empty_char;
	info->original_filename.data = ufbxwi_empty_char;
	info->last_application_vendor.data = ufbxwi_empty_char;
	info->last_application_name.data = ufbxwi_empty_char;
	info->last_application_version.data = ufbxwi_empty_char;
	info->last_date_time.data = ufbxwi_empty_char;
	return true;
}

static bool ufbxwi_init_global_settings(ufbxw_scene *scene, void *data)
{
	ufbxwi_global_settings *settings = (ufbxwi_global_settings*)data;
	settings->up_axis = 1;
	settings->up_axis_sign = 1;
	settings->front_axis = 2;
	settings->front_axis_sign = 1;
	settings->coord_axis = 0;
	settings->coord_axis_sign = 1;
	settings->original_up_axis = 1;
	settings->original_up_axis_sign = 1;
	settings->unit_scale_factor = 1.0f;
	settings->original_unit_scale_factor = 1.0f;
	settings->time_mode = UFBXW_TIME_MODE_24_FPS;
	settings->time_protocol = UFBXW_TIME_PROTOCOL_DEFAULT;
	settings->snap_on_frame_mode = UFBXW_SNAP_MODE_NONE;
	settings->time_span_start = 0;
	settings->time_span_stop = 0;
	settings->custom_frame_rate = -1.0f;
	settings->current_time_marker = -1;
	return true;
}

static bool ufbxwi_init_skin_deformer(ufbxw_scene *scene, void *data)
{
	ufbxwi_skin_deformer *deformer = (ufbxwi_skin_deformer*)data;
	deformer->skinning_type = UFBXW_SKINNING_TYPE_LINEAR;
	return true;
}

static bool ufbxwi_init_skin_cluster(ufbxw_scene *scene, void *data)
{
	ufbxwi_skin_cluster *cluster = (ufbxwi_skin_cluster*)data;
	cluster->transform = ufbxw_identity_matrix;
	cluster->transform_link = ufbxw_identity_matrix;
	return true;
}

static bool ufbxwi_init_material_lambert(ufbxw_scene *scene, void *data)
{
	ufbxwi_material *material = (ufbxwi_material*)data;
	material->shading_model = ufbxwi_c_str("lambert");
	return true;
}

static bool ufbxwi_init_file_texture(ufbxw_scene *scene, void *data)
{
	ufbxwi_texture *texture = (ufbxwi_texture*)data;
	texture->type = ufbxwi_c_str("TextureVideoClip");
	texture->filename.data = ufbxwi_empty_char;
	texture->relative_filename.data = ufbxwi_empty_char;
	return true;
}

static bool ufbxwi_init_anim_layer(ufbxw_scene *scene, void *data)
{
	ufbxwi_anim_layer *layer = (ufbxwi_anim_layer*)data;
	layer->weight = (ufbxw_real)100.0;
	return true;
}

enum {
	UFBXWI_ELEMENT_TYPE_FLAG_EAGER_PROPS = 0x1,
};

struct ufbxwi_element_type_desc {
	ufbxw_element_type element_type;
	ufbxwi_token class_type;
	ufbxwi_token sub_type;
	ufbxwi_token object_type;
	ufbxwi_token fbx_type;
	ufbxwi_token tmpl_type;
	const ufbxwi_prop_desc *props;
	size_t num_props;
	ufbxwi_element_init_data_fn *init_fn;
	uint32_t flags;
};

static const ufbxwi_element_type_desc ufbxwi_element_types[] = {
	{
		UFBXW_ELEMENT_TEMPLATE, UFBXWI_TOKEN_NONE, UFBXWI_TOKEN_NONE, UFBXWI_TOKEN_NONE, UFBXWI_TOKEN_NONE, UFBXWI_TOKEN_NONE,
		NULL, 0, NULL,
		UFBXWI_ELEMENT_FLAG_ALLOW_NO_OBJECT_ID,
	},
	{
		UFBXW_ELEMENT_NODE, UFBXWI_TOKEN_NONE, UFBXWI_TOKEN_NONE, UFBXWI_Model, UFBXWI_Model, UFBXWI_FbxNode,
		ufbxwi_node_props, ufbxwi_arraycount(ufbxwi_node_props), &ufbxwi_init_node,
		0,
	},
	{
		UFBXW_ELEMENT_MESH, UFBXWI_TOKEN_NONE, UFBXWI_Mesh, UFBXWI_Geometry, UFBXWI_Geometry, UFBXWI_FbxMesh,
		ufbxwi_mesh_props, ufbxwi_arraycount(ufbxwi_mesh_props), NULL,
		0,
	},
	{
		UFBXW_ELEMENT_SKIN_DEFORMER, UFBXWI_TOKEN_NONE, UFBXWI_Skin, UFBXWI_Deformer, UFBXWI_Deformer, UFBXWI_TOKEN_NONE,
		NULL, 0, &ufbxwi_init_skin_deformer,
		0,
	},
	{
		UFBXW_ELEMENT_SKIN_CLUSTER, UFBXWI_TOKEN_NONE, UFBXWI_Cluster, UFBXWI_Deformer, UFBXWI_SubDeformer, UFBXWI_TOKEN_NONE,
		NULL, 0, &ufbxwi_init_skin_cluster,
		0,
	},
	{
		UFBXW_ELEMENT_BLEND_DEFORMER, UFBXWI_TOKEN_NONE, UFBXWI_BlendShape, UFBXWI_Deformer, UFBXWI_Deformer, UFBXWI_TOKEN_NONE,
		NULL, 0, NULL,
		0,
	},
	{
		UFBXW_ELEMENT_BLEND_CHANNEL, UFBXWI_TOKEN_NONE, UFBXWI_BlendShapeChannel, UFBXWI_Deformer, UFBXWI_SubDeformer, UFBXWI_TOKEN_NONE,
		ufbxwi_blend_channel_props, ufbxwi_arraycount(ufbxwi_blend_channel_props), NULL,
		0,
	},
	{
		UFBXW_ELEMENT_BLEND_SHAPE, UFBXWI_TOKEN_NONE, UFBXWI_Shape, UFBXWI_Geometry, UFBXWI_Geometry, UFBXWI_TOKEN_NONE,
		NULL, 0, NULL,
		0,
	},
	{
		UFBXW_ELEMENT_LIGHT, UFBXWI_TOKEN_NONE, UFBXWI_Light, UFBXWI_NodeAttribute, UFBXWI_NodeAttribute, UFBXWI_FbxLight,
		ufbxwi_light_props, ufbxwi_arraycount(ufbxwi_light_props), &ufbxwi_init_light,
		0,
	},
	{
		UFBXW_ELEMENT_CAMERA, UFBXWI_TOKEN_NONE, UFBXWI_Camera, UFBXWI_NodeAttribute, UFBXWI_NodeAttribute, UFBXWI_FbxCamera,
		ufbxwi_camera_props, ufbxwi_arraycount(ufbxwi_camera_props), &ufbxwi_init_camera,
		0,
	},
	{
		UFBXW_ELEMENT_SKELETON, UFBXWI_TOKEN_NONE, UFBXWI_LimbNode, UFBXWI_NodeAttribute, UFBXWI_NodeAttribute, UFBXWI_FbxSkeleton,
		ufbxwi_skeleton_props, ufbxwi_arraycount(ufbxwi_skeleton_props), &ufbxwi_init_skeleton,
		0,
	},
	{
		UFBXW_ELEMENT_BIND_POSE, UFBXWI_TOKEN_NONE, UFBXWI_BindPose, UFBXWI_Pose, UFBXWI_Pose, UFBXWI_TOKEN_NONE,
		NULL, 0, NULL,
		0,
	},
	{
		UFBXW_ELEMENT_SCENE_INFO, UFBXWI_TOKEN_NONE, UFBXWI_UserData, UFBXWI_SceneInfo, UFBXWI_SceneInfo, UFBXWI_TOKEN_NONE,
		ufbxwi_scene_info_props, ufbxwi_arraycount(ufbxwi_scene_info_props), &ufbxwi_init_scene_info,
		UFBXWI_ELEMENT_FLAG_ALLOW_NO_OBJECT_ID,
	},
	{
		UFBXW_ELEMENT_GLOBAL_SETTINGS, UFBXWI_TOKEN_NONE, UFBXWI_TOKEN_NONE, UFBXWI_GlobalSettings, UFBXWI_TOKEN_NONE, UFBXWI_TOKEN_NONE,
		ufbxwi_global_settings_props, ufbxwi_arraycount(ufbxwi_global_settings_props), &ufbxwi_init_global_settings,
		0,
	},
	{
		UFBXW_ELEMENT_DOCUMENT, UFBXWI_TOKEN_NONE, UFBXWI_Document, UFBXWI_TOKEN_NONE, UFBXWI_TOKEN_NONE, UFBXWI_TOKEN_NONE,
		ufbxwi_document_props, ufbxwi_arraycount(ufbxwi_document_props), NULL,
		UFBXWI_ELEMENT_FLAG_ALLOW_NO_OBJECT_ID,
	},
	{
		UFBXW_ELEMENT_ANIM_CURVE, UFBXWI_TOKEN_NONE, UFBXWI_TOKEN_EMPTY, UFBXWI_AnimationCurve, UFBXWI_AnimCurve, UFBXWI_TOKEN_NONE,
		NULL, 0, NULL,
		0,
	},
	{
		UFBXW_ELEMENT_ANIM_PROP, UFBXWI_TOKEN_NONE, UFBXWI_TOKEN_EMPTY, UFBXWI_AnimationCurveNode, UFBXWI_AnimCurveNode, UFBXWI_FbxAnimCurveNode,
		ufbxwi_anim_prop_props, ufbxwi_arraycount(ufbxwi_anim_prop_props), NULL,
		UFBXWI_ELEMENT_TYPE_FLAG_EAGER_PROPS,
	},
	{
		UFBXW_ELEMENT_ANIM_LAYER, UFBXWI_TOKEN_NONE, UFBXWI_TOKEN_EMPTY, UFBXWI_AnimationLayer, UFBXWI_AnimLayer, UFBXWI_FbxAnimLayer,
		ufbxwi_anim_layer_props, ufbxwi_arraycount(ufbxwi_anim_layer_props), &ufbxwi_init_anim_layer,
		0,
	},
	{
		UFBXW_ELEMENT_ANIM_STACK, UFBXWI_TOKEN_NONE, UFBXWI_TOKEN_EMPTY, UFBXWI_AnimationStack, UFBXWI_AnimStack, UFBXWI_FbxAnimStack,
		ufbxwi_anim_stack_props, ufbxwi_arraycount(ufbxwi_anim_stack_props), NULL,
		0,
	},
	{
		UFBXW_ELEMENT_MATERIAL, UFBXWI_FbxSurfaceLambert, UFBXWI_TOKEN_EMPTY, UFBXWI_Material, UFBXWI_Material, UFBXWI_FbxSurfaceLambert,
		ufbxwi_material_lambert_props, ufbxwi_arraycount(ufbxwi_material_lambert_props), &ufbxwi_init_material_lambert,
		UFBXWI_ELEMENT_TYPE_FLAG_EAGER_PROPS,
	},
	{
		UFBXW_ELEMENT_TEXTURE, UFBXWI_FbxFileTexture, UFBXWI_TOKEN_EMPTY, UFBXWI_Texture, UFBXWI_Texture, UFBXWI_FbxFileTexture,
		ufbxwi_file_texture_props, ufbxwi_arraycount(ufbxwi_file_texture_props), &ufbxwi_init_file_texture,
		0,
	},
};

static bool ufbxwi_props_rehash(ufbxw_scene *scene, ufbxwi_props *props, size_t min_capacity)
{
	uint32_t old_capacity = props->capacity;
	uint32_t new_capacity = ufbxwi_max_u32(old_capacity * 2, 1);
	while (new_capacity < min_capacity * 2) {
		new_capacity *= 2;
	}

	ufbxwi_prop *new_slots = ufbxwi_alloc(&scene->ator, ufbxwi_prop, new_capacity);
	ufbxwi_check(new_slots, false);

	memset(new_slots, 0, new_capacity * sizeof(ufbxwi_prop));

	ufbxwi_prop *old_slots = props->props;

	uint32_t mask = new_capacity - 1;
	for (size_t i = 0; i < old_capacity; i++) {
		uint32_t token = old_slots[i].token;
		if (token == 0 || token == ~0u) continue;

		uint32_t hash = ufbxwi_hash_token(token);
		for (uint32_t scan = 0; ; scan++) {
			uint32_t slot = (hash + scan) & mask;
			if (new_slots[slot].token == 0) {
				new_slots[slot] = old_slots[i];
				break;
			}
		}
	}

	ufbxwi_free(&scene->ator, old_slots);

	props->props = new_slots;
	props->capacity = new_capacity;
	return true;
}

static ufbxwi_prop *ufbxwi_props_find_prop(ufbxwi_props *props, ufbxwi_token token)
{
	if (props->count == 0) return NULL;

	ufbxwi_prop *prop = NULL;

	uint32_t mask = props->capacity - 1;
	uint32_t hash = ufbxwi_hash_token(token);
	ufbxwi_prop *slots = props->props;
	for (uint32_t scan = 0; ; scan++) {
		uint32_t slot = (hash + scan) & mask;
		if (slots[slot].token == token) {
			return &slots[slot];
		} else if (slots[slot].token == 0) {
			return NULL;
		}
	}
}

static ufbxwi_prop *ufbxwi_props_add_prop(ufbxw_scene *scene, ufbxwi_props *props, ufbxwi_token token)
{
	ufbxw_assert(token != UFBXWI_TOKEN_NONE && token != UFBXWI_TOKEN_EMPTY);
	if (props->count * 2 >= props->capacity) {
		ufbxwi_check(ufbxwi_props_rehash(scene, props, 4), NULL);
	}

	ufbxwi_prop *prop = NULL;

	uint32_t mask = props->capacity - 1;
	uint32_t hash = ufbxwi_hash_token(token);
	ufbxwi_prop *slots = props->props;
	for (uint32_t scan = 0; ; scan++) {
		uint32_t slot = (hash + scan) & mask;
		if (slots[slot].token == token) {
			prop = &slots[slot];
			break;
		} else if (slots[slot].token == 0) {
			prop = &slots[slot];
			break;
		}
	}

	if (prop->token == 0) {
		prop->token = (uint32_t)token;
		prop->order = ++props->order_counter;
		props->count++;
	}

	return prop;
}

static bool ufbxwi_props_copy(ufbxw_scene *scene, ufbxwi_props *dst, const ufbxwi_props *src)
{
	ufbxwi_prop *copy = ufbxwi_alloc(&scene->ator, ufbxwi_prop, src->capacity);
	ufbxwi_check(copy, false);
	memcpy(copy, src->props, src->capacity * sizeof(ufbxwi_prop));
	*dst = *src;
	dst->props = copy;
	return true;
}

static uint32_t ufbxwi_find_element_type_id(ufbxw_scene *scene, ufbxw_element_type type, ufbxwi_token class_type)
{
	ufbxwi_element_type_list types = scene->element_types;
	for (uint32_t i = 0; i < types.count; i++) {
		ufbxwi_element_type *et = &types.data[i];
		if (et->element_type != type) continue;
		if (et->class_type == class_type) return i;
	}
	return ~0u;
}

static ufbxwi_forceinline const void *ufbxwi_resolve_prop_value(const ufbxwi_element *elem, ufbxwi_prop_value value)
{
	const void *base = NULL;
	uint32_t offset = ufbxwi_prop_value_offset(value);
	switch (ufbxwi_prop_value_type(value)) {
	case UFBXWI_PROP_VALUE_DEFAULT:
		ufbxwi_dev_assert(offset < sizeof(ufbxwi_prop_default_data));
		base = &ufbxwi_prop_default_data;
		break;
	case UFBXWI_PROP_VALUE_FIELD:
		ufbxwi_dev_assert(offset < elem->element_data_size);
		base = elem;
		break;
	case UFBXWI_PROP_VALUE_DATA:
		ufbxwi_dev_assert(offset < elem->prop_data_size);
		base = elem->prop_data;
		break;
	}
	return (const char *)base + offset;
}

static ufbxwi_forceinline void *ufbxwi_edit_prop_value(ufbxwi_element *elem, ufbxwi_prop_value value)
{
	const void *base = NULL;
	uint32_t offset = ufbxwi_prop_value_offset(value);
	switch (ufbxwi_prop_value_type(value)) {
	case UFBXWI_PROP_VALUE_DEFAULT:
		ufbxw_assert(0 && "Attempting to edit default value");
		break;
	case UFBXWI_PROP_VALUE_FIELD:
		ufbxwi_dev_assert(offset < elem->element_data_size);
		base = elem;
		break;
	case UFBXWI_PROP_VALUE_DATA:
		ufbxwi_dev_assert(offset < elem->prop_data_size);
		base = elem->prop_data;
		break;
	}
	return (char *)base + offset;
}

static ufbxwi_prop_value ufbxwi_element_add_prop_data(ufbxw_scene *scene, ufbxwi_element *element, ufbxw_prop_data_type data_type)
{
	ufbxwi_prop_data_info data_info = ufbxwi_prop_data_infos[data_type];
	size_t data_offset = ufbxwi_align(element->prop_data_size, data_info.alignment);
	size_t data_end = data_offset + data_info.size;

	if (data_end > element->prop_data_capacity) {
		size_t new_capacity = ufbxwi_max_sz(data_end, element->prop_data_capacity * 2);
		void *new_data = ufbxwi_alloc_size(&scene->ator, 1, new_capacity, &new_capacity);
		ufbxwi_check(new_data, 0);

		memcpy(new_data, element->prop_data, element->prop_data_size);
		memset((char*)new_data + element->prop_data_size, 0, new_capacity - element->prop_data_size);

		ufbxwi_free(&scene->ator, element->prop_data);
		element->prop_data = new_data;
		element->prop_data_capacity = new_capacity;
	}

	element->prop_data_size = data_end;
	return ufbxwi_prop_value_data(data_offset);
}

static ufbxwi_prop *ufbxwi_element_find_prop(ufbxw_scene *scene, ufbxwi_element *element, ufbxwi_token token)
{
	ufbxwi_prop *prop = ufbxwi_props_find_prop(&element->props, token);
	if (!prop && element->flags & UFBXWI_ELEMENT_FLAG_HAS_DEFAULT_PROPS) {
		ufbxwi_props *default_props = &scene->element_types.data[element->type_id].props;
		prop = ufbxwi_props_find_prop(default_props, token);
	}
	return prop;
}

static ufbxwi_prop *ufbxwi_element_edit_prop(ufbxw_scene *scene, ufbxwi_element *element, ufbxwi_token token)
{
	if (element->flags & UFBXWI_ELEMENT_FLAG_HAS_DEFAULT_PROPS) {
		ufbxwi_props *default_props = &scene->element_types.data[element->type_id].props;

		ufbxwi_prop *prop = ufbxwi_props_find_prop(&element->props, token);
		if (!prop) {
			prop = ufbxwi_props_find_prop(default_props, token);
			if (!prop) return NULL;

			ufbxwi_prop *new_prop = ufbxwi_props_add_prop(scene, &element->props, token);
			ufbxwi_check(new_prop, NULL);

			*new_prop = *prop;
			prop = new_prop;
		}
		if (ufbxwi_prop_value_type(prop->value) == UFBXWI_PROP_VALUE_DEFAULT) {
			ufbxw_prop_data_type data_type = scene->prop_types.data[prop->type].data_type;
			ufbxwi_prop_value value = ufbxwi_element_add_prop_data(scene, element, data_type);
			ufbxwi_check(value, NULL);

			const void *src = ufbxwi_resolve_prop_value(element, prop->value);
			void *dst = ufbxwi_edit_prop_value(element, value);
			memcpy(dst, src, ufbxwi_prop_data_infos[data_type].size);

			prop->value = value;
		}
		return prop;
	} else {
		ufbxwi_prop *prop = ufbxwi_props_find_prop(&element->props, token);
		if (prop && ufbxwi_prop_value_type(prop->value) == UFBXWI_PROP_VALUE_DEFAULT) {
			ufbxw_prop_data_type data_type = scene->prop_types.data[prop->type].data_type;
			ufbxwi_prop_value value = ufbxwi_element_add_prop_data(scene, element, data_type);
			ufbxwi_check(value, NULL);
			prop->value = value;
		}

		return prop;
	}
}

static ufbxwi_prop *ufbxwi_element_add_prop(ufbxw_scene *scene, ufbxwi_element *element, ufbxwi_token token, ufbxw_prop_type type)
{
	ufbxw_prop_data_type data_type = scene->prop_types.data[type].data_type;
	if (element->flags & UFBXWI_ELEMENT_FLAG_HAS_DEFAULT_PROPS) {
		ufbxwi_props *default_props = &scene->element_types.data[element->type_id].props;

		ufbxwi_prop *prop = ufbxwi_props_find_prop(&element->props, token);
		if (!prop) {
			prop = ufbxwi_props_find_prop(default_props, token);
		}
		if (prop) {
			ufbxw_prop_data_type existing_type = scene->prop_types.data[prop->type].data_type;
			if (existing_type != data_type) {
				// TODO: Is this even a good check?
				return NULL;
			}
		}
		if (!prop || ufbxwi_prop_value_type(prop->value) == UFBXWI_PROP_VALUE_DEFAULT) {
			ufbxwi_prop *new_prop = ufbxwi_props_add_prop(scene, &element->props, token);
			ufbxwi_check(new_prop, NULL);

			ufbxwi_prop_value value = ufbxwi_element_add_prop_data(scene, element, data_type);
			ufbxwi_check(value, NULL);
			if (prop) {
				const void *src = ufbxwi_resolve_prop_value(element, prop->value);
				void *dst = ufbxwi_edit_prop_value(element, value);
				memcpy(dst, src, ufbxwi_prop_data_infos[data_type].size);
			}

			if (prop) {
				*new_prop = *prop;
			} else {
				new_prop->type = type;
			}
			new_prop->value = value;
			prop = new_prop;
		}
		return prop;
	} else {
		ufbxwi_prop *prop = ufbxwi_props_add_prop(scene, &element->props, token);
		ufbxwi_check(prop, NULL);

		if (ufbxwi_prop_value_type(prop->value) == UFBXWI_PROP_VALUE_DEFAULT) {
			ufbxwi_prop_value value = ufbxwi_element_add_prop_data(scene, element, data_type);
			ufbxwi_check(value, NULL);
			prop->value = value;
			prop->type = type;
		} else {
			// TODO: Is this even a good check?
			return NULL;
		}

		return prop;
	}
}

static ufbxwi_noinline bool ufbxwi_cast_value(void *dst, const void *src, ufbxw_prop_data_type dst_type, ufbxw_prop_data_type src_type)
{
	if (dst_type == src_type) {
		memcpy(dst, src, ufbxwi_prop_data_infos[dst_type].size);
		return true;
	}

	if (dst_type == UFBXW_PROP_DATA_BOOL) {
		bool *d = (bool*)dst;
		switch (src_type) {
		case UFBXW_PROP_DATA_INT32:
			*d = *(const int32_t*)src != 0;
			return true;
		case UFBXW_PROP_DATA_INT64:
			*d = *(const int64_t*)src != 0;
			return true;
		default:
			break;
		}
	} else if (dst_type == UFBXW_PROP_DATA_INT32) {
		int32_t *d = (int32_t*)dst;
		switch (src_type) {
		case UFBXW_PROP_DATA_BOOL:
			*d = *(const bool*)src ? 1 : 0;
			return true;
		case UFBXW_PROP_DATA_INT64:
			// TODO: Handle out of bounds
			*d = (int32_t)*(const int64_t*)src;
			return true;
		case UFBXW_PROP_DATA_USER_INT:
			*d = ((const ufbxw_user_int*)src)->value;
			return true;
		case UFBXW_PROP_DATA_USER_ENUM:
			*d = ((const ufbxw_user_enum*)src)->value;
			return true;
		default:
			break;
		}
	} else if (dst_type == UFBXW_PROP_DATA_INT64) {
		uint64_t *d = (uint64_t*)dst;
		switch (src_type) {
		case UFBXW_PROP_DATA_BOOL:
			*d = *(const bool*)src ? 1 : 0;
			return true;
		case UFBXW_PROP_DATA_INT32:
			*d = *(const int32_t*)src;
			return true;
		case UFBXW_PROP_DATA_USER_INT:
			*d = ((const ufbxw_user_int*)src)->value;
			return true;
		case UFBXW_PROP_DATA_USER_ENUM:
			*d = ((const ufbxw_user_enum*)src)->value;
			return true;
		default:
			break;
		}
	} else if (dst_type == UFBXW_PROP_DATA_REAL) {
		ufbxw_real *d = (ufbxw_real*)dst;
		switch (src_type) {
		case UFBXW_PROP_DATA_USER_REAL:
			*d = ((const ufbxw_user_real*)src)->value;
			return true;
		default:
			break;
		}
	} else if (dst_type == UFBXW_PROP_DATA_USER_INT) {
		ufbxw_user_int *d = (ufbxw_user_int*)dst;
		switch (src_type) {
		case UFBXW_PROP_DATA_BOOL:
			d->value = *(const bool*)src ? 1 : 0;
			return true;
		case UFBXW_PROP_DATA_INT32:
			d->value = *(const int32_t*)src;
			return true;
		case UFBXW_PROP_DATA_INT64:
			// TODO: Handle out of bounds
			d->value = (int32_t)*(const int64_t*)src;
			return true;
		default:
			break;
		}
	} else if (dst_type == UFBXW_PROP_DATA_USER_REAL) {
		ufbxw_user_real *d = (ufbxw_user_real*)dst;
		switch (src_type) {
		case UFBXW_PROP_DATA_REAL:
			d->value = *(const int32_t*)src;
			return true;
		default:
			break;
		}
	} else if (dst_type == UFBXW_PROP_DATA_USER_ENUM) {
		ufbxw_user_enum *d = (ufbxw_user_enum*)dst;
		switch (src_type) {
		case UFBXW_PROP_DATA_BOOL:
			d->value = *(const bool*)src ? 1 : 0;
			return true;
		case UFBXW_PROP_DATA_INT32:
			d->value = *(const int32_t*)src;
			return true;
		case UFBXW_PROP_DATA_INT64:
			// TODO: Handle out of bounds
			d->value = (int32_t)*(const int64_t*)src;
			return true;
		default:
			break;
		}
	}

	return false;
}

static ufbxwi_forceinline ufbxwi_element_slot *ufbxwi_get_element_slot(ufbxw_scene *scene, ufbxw_id id)
{
	size_t index = ufbxwi_id_index(id);
	if (index >= scene->elements.count) return NULL;
	ufbxwi_element_slot *slot = &scene->elements.data[index];
	if (slot->id != id) return NULL;
	return slot;
}

static ufbxwi_forceinline ufbxwi_element *ufbxwi_get_element(ufbxw_scene *scene, ufbxw_id id)
{
	size_t index = ufbxwi_id_index(id);
	if (index >= scene->elements.count) return NULL;
	ufbxwi_element_slot *slot = &scene->elements.data[index];
	if (slot->id != id) return NULL;
	return slot->element;
}

static ufbxwi_forceinline ufbxwi_element *ufbxwi_get_typed_element(ufbxw_scene *scene, ufbxw_id id, ufbxw_element_type type)
{
	size_t index = ufbxwi_id_index(id);
	if (ufbxwi_id_type(id) != type) return NULL;
	if (index >= scene->elements.count) return NULL;
	ufbxwi_element_slot *slot = &scene->elements.data[index];
	if (slot->id != id) return NULL;
	return slot->element;
}

static ufbxwi_forceinline bool ufbxwi_init_element_type_imp(ufbxw_scene *scene, ufbxwi_element_type *et)
{
	const ufbxwi_element_type_desc *desc = et->desc;
	et->initialized = true;

	if (desc->num_props > 0) {
		ufbxwi_check(ufbxwi_props_rehash(scene, &et->props, desc->num_props), false);

		ufbxwi_for(const ufbxwi_prop_desc, pd, desc->props, desc->num_props) {
			ufbxwi_prop *prop = ufbxwi_props_add_prop(scene, &et->props, pd->name);
			if (!prop) return false;
			prop->type = pd->type;
			prop->value = pd->value_offset;
			prop->flags = pd->flags & 0xff; // TODO: Better mapping, compact flags
		}
	}

	if (desc->tmpl_type != UFBXWI_TOKEN_NONE) {
		ufbxw_id template_id = ufbxw_create_element(scene, UFBXW_ELEMENT_TEMPLATE);
		ufbxwi_check(template_id, false);

		ufbxwi_element *tmpl_elem = ufbxwi_get_element(scene, template_id);
		ufbxw_assert(tmpl_elem);

		tmpl_elem->name = scene->string_pool.tokens.data[desc->tmpl_type];

		ufbxwi_template *tmpl_data = (ufbxwi_template*)tmpl_elem;
		tmpl_data->type = desc->tmpl_type;

		ufbxwi_check(ufbxwi_props_rehash(scene, &tmpl_elem->props, desc->num_props), false);

		ufbxwi_for(const ufbxwi_prop_desc, pd, desc->props, desc->num_props) {
			if (pd->flags & UFBXWI_PROP_FLAG_EXCLUDE_FROM_TEMPLATE) continue;

			ufbxwi_prop *prop = ufbxwi_props_add_prop(scene, &tmpl_elem->props, pd->name);
			ufbxwi_check(prop, false);
			prop->type = pd->type;
			prop->value = ufbxwi_prop_value_type(pd->value_offset) == UFBXWI_PROP_VALUE_DEFAULT ? pd->value_offset : pd->default_offset;
			prop->flags = pd->flags & 0xff; // TODO: Better mapping, compact flags
		}
		et->template_id = template_id;
	}

	return true;
}

static void ufbxwi_fail_element(ufbxw_scene *scene, ufbxw_id id, const char *func)
{
	ufbxwi_element *elem = ufbxwi_get_element(scene, id);
	if (!elem) {
		ufbxwi_fail_imp(&scene->error, UFBXW_ERROR_ELEMENT_NOT_FOUND, func, "element not found");
	} else {
		ufbxwi_failf_imp(&scene->error, UFBXW_ERROR_ELEMENT_WRONG_TYPE, func, "wrong type: %s", ufbxwi_element_type_names[ufbxwi_id_type(id)]);
	}
}


#define ufbxwi_check_element(scene, id, cond, ...) do { \
		if (!(cond)) { \
			ufbxwi_fail_element((scene), id, __func__); \
			return __VA_ARGS__; \
		} \
	} while (0)

static ufbxwi_forceinline bool ufbxwi_init_element_type(ufbxw_scene *scene, ufbxwi_element_type *et)
{
	if (et->initialized) return true;
	return ufbxwi_init_element_type_imp(scene, et);
}

static ufbxw_id ufbxwi_create_element(ufbxw_scene *scene, ufbxw_element_type type, ufbxwi_token class_type)
{
	uint32_t type_id = ufbxwi_find_element_type_id(scene, type, class_type);
	if (type_id == ~0u) {
		if (class_type > UFBXWI_TOKEN_EMPTY) {
			const char *class_name = scene->string_pool.tokens.data[class_type].data;
			ufbxwi_failf(&scene->error, UFBXW_ERROR_ELEMENT_TYPE_NOT_FOUND, "Element type not found: %s (%s)",
				ufbxwi_element_type_names[type], class_name);
		} else {
			ufbxwi_failf(&scene->error, UFBXW_ERROR_ELEMENT_TYPE_NOT_FOUND, "Element type not found: %s",
				ufbxwi_element_type_names[type]);
		}
		return 0;
	}

	const ufbxwi_element_type_info *type_info = &ufbxwi_element_type_infos[type];
	ufbxwi_element_type *element_type = &scene->element_types.data[type_id];
	ufbxwi_check(ufbxwi_init_element_type(scene, element_type), 0);

	uint32_t data_size = type_info->data_size;
	if (data_size < sizeof(ufbxwi_element)) {
		data_size = sizeof(ufbxwi_element);
	}

	void *data = ufbxwi_alloc_size(&scene->ator, 1, data_size, NULL);
	ufbxwi_check(data, 0);
	memset(data, 0, data_size);

	ufbxwi_element *element = (ufbxwi_element*)data;

	size_t index = 0;
	if (scene->free_element_ids.count > 0) {
		index = scene->free_element_ids.data[--scene->free_element_ids.count];
	} else {
		index = scene->elements.count;
		ufbxwi_check(ufbxwi_list_push_zero(&scene->ator, &scene->elements, ufbxwi_element_slot), 0);
	}

	ufbxwi_element_slot *slot = &scene->elements.data[index];
	uint32_t generation = ufbxwi_id_generation(slot->id) + 1;

	ufbxw_id id = ufbxwi_make_id(type, generation, index);
	slot->id = id;
	slot->element = element;

	element->id = id;
	element->name = ufbxwi_empty_string;
	element->element_data_size = data_size;
	element->conn_bits = type_info->conn_bits;

	element->type_id = type_id;

	if (element_type->props.count == 0) {
		// No properties
	} else if ((element_type->flags & UFBXWI_ELEMENT_TYPE_FLAG_EAGER_PROPS) != 0) {
		ufbxwi_check(ufbxwi_props_copy(scene, &element->props, &element_type->props), 0);
	} else {
		element->flags |= UFBXWI_ELEMENT_FLAG_HAS_DEFAULT_PROPS;
		element->props.order_counter = element_type->props.order_counter;
	}

	if (element_type->init_fn) {
		ufbxwi_check(element_type->init_fn(scene, data), 0);
	}

	scene->num_elements++;

	return ufbxwi_make_id(type, generation, index);
}

static ufbxwi_noinline bool ufbxwi_set_prop(ufbxw_scene *scene, ufbxw_id id, const char *prop, size_t prop_len, const void *src, ufbxw_prop_data_type src_type)
{
	ufbxwi_token token = ufbxwi_get_token(&scene->string_pool, prop, prop_len);
	ufbxwi_element *element = ufbxwi_get_element(scene, id);
	if (!token || !element) return false;

	ufbxwi_prop *p = ufbxwi_element_edit_prop(scene, element, token);
	if (!p) {
		ufbxwi_failf(&scene->error, UFBXW_ERROR_PROP_NOT_FOUND, "Property not found: %.*s", (int)prop_len, prop);
		return false;
	}

	ufbxw_prop_data_type data_type = scene->prop_types.data[p->type].data_type;

	void *dst = ufbxwi_edit_prop_value(element, p->value);
	if (data_type == src_type) {
		memcpy(dst, src, ufbxwi_prop_data_infos[src_type].size);
		return true;
	} else {
		if (!ufbxwi_cast_value(dst, src, data_type, src_type)) {
			ufbxwi_failf(&scene->error, UFBXW_ERROR_WRONG_DATA_TYPE, "Could not convert value from %s to %s",
				ufbxwi_prop_data_names[src_type], ufbxwi_prop_data_names[data_type]);
			return false;
		}
		return true;
	}
}

static ufbxwi_noinline bool ufbxwi_add_prop(ufbxw_scene *scene, ufbxw_id id, const char *prop, size_t prop_len, ufbxw_prop_type type, const void *src, ufbxw_prop_data_type src_type)
{
	ufbxwi_token token = ufbxwi_intern_token(&scene->string_pool, prop, prop_len);
	ufbxwi_element *element = ufbxwi_get_element(scene, id);
	if (!token || !element) return false;

	ufbxwi_prop *p = ufbxwi_element_add_prop(scene, element, token, type);
	ufbxwi_check(p, false);

	ufbxw_prop_data_type data_type = scene->prop_types.data[type].data_type;
	void *dst = ufbxwi_edit_prop_value(element, p->value);
	if (data_type == src_type) {
		memcpy(dst, src, ufbxwi_prop_data_infos[src_type].size);
		return true;
	} else {
		if (!ufbxwi_cast_value(dst, src, data_type, src_type)) {
			ufbxwi_failf(&scene->error, UFBXW_ERROR_WRONG_DATA_TYPE, "Could not convert value from %s to %s",
				ufbxwi_prop_data_names[src_type], ufbxwi_prop_data_names[data_type]);
			return false;
		}
		return true;
	}
}

static ufbxwi_noinline bool ufbxwi_get_prop(ufbxw_scene *scene, ufbxw_id id, const char *prop, size_t prop_len, void *dst, ufbxw_prop_data_type dst_type)
{
	ufbxwi_token token = ufbxwi_get_token(&scene->string_pool, prop, prop_len);
	ufbxwi_element *element = ufbxwi_get_element(scene, id);
	if (!token || !element) return false;

	ufbxwi_prop *p = ufbxwi_element_find_prop(scene, element, token);
	if (!p) return false;

	ufbxw_prop_data_type data_type = scene->prop_types.data[p->type].data_type;
	const void *src = ufbxwi_resolve_prop_value(element, p->value);
	if (data_type == dst_type) {
		memcpy(dst, src, ufbxwi_prop_data_infos[dst_type].size);
		return true;
	} else {
		return ufbxwi_cast_value(dst, src, dst_type, data_type);
	}
}

static bool ufbxwi_connect_imp(ufbxw_scene *scene, ufbxw_connection_type type, ufbxw_id src_id, ufbxw_id dst_id, ufbxwi_token src_prop, ufbxwi_token dst_prop, uint32_t flags)
{
	ufbxwi_element *src_elem = ufbxwi_get_element(scene, src_id);
	ufbxwi_element *dst_elem = ufbxwi_get_element(scene, dst_id);
	if (!src_elem || !dst_elem) return false;

	ufbxwi_connection_info info = ufbxwi_connection_infos[type];
	uint64_t src_bits = src_elem->conn_bits | (src_prop != UFBXWI_TOKEN_NONE ? UFBXWI_CONN_BIT_PROPERTY : UFBXWI_CONN_BIT_ELEMENT);
	uint64_t dst_bits = dst_elem->conn_bits | (dst_prop != UFBXWI_TOKEN_NONE ? UFBXWI_CONN_BIT_PROPERTY : UFBXWI_CONN_BIT_ELEMENT);
	if ((src_bits & info.src_mask) != info.src_mask) return false;
	if ((dst_bits & info.dst_mask) != info.dst_mask) return false;

	ufbxwi_conn_type src_type = ufbxwi_conn_type(info.src_conn);
	ufbxwi_conn_type dst_type = ufbxwi_conn_type(info.dst_conn);
	void *src_data = (char*)src_elem + ufbxwi_conn_offset(info.src_conn);
	void *dst_data = (char*)dst_elem + ufbxwi_conn_offset(info.dst_conn);

	bool disconnect = false;
	if (src_type == UFBXWI_CONN_TYPE_ID && *(ufbxw_id*)src_data != ufbxw_null_id) {
		if ((flags & UFBXWI_CONNECT_FLAG_DISCONNECT_SRC) == 0) return false;
		disconnect = true;
	}
	if (dst_type == UFBXWI_CONN_TYPE_ID && *(ufbxw_id*)dst_data != ufbxw_null_id) {
		if ((flags & UFBXWI_CONNECT_FLAG_DISCONNECT_DST) == 0) return false;
		disconnect = true;
	}

	if (disconnect) {
		ufbxwi_conn_remove_one(scene, src_type, src_data, dst_id);
		ufbxwi_conn_remove_one(scene, dst_type, dst_data, src_id);
	}

	ufbxwi_conn_add(scene, src_type, src_data, dst_id, src_prop, dst_prop);
	ufbxwi_conn_add(scene, dst_type, dst_data, src_id, src_prop, dst_prop);
	return true;
}

static ufbxwi_forceinline bool ufbxwi_connect(ufbxw_scene *scene, ufbxw_connection_type type, ufbxw_id src_id, ufbxw_id dst_id, uint32_t flags)
{
	return ufbxwi_connect_imp(scene, type, src_id, dst_id, UFBXWI_TOKEN_NONE, UFBXWI_TOKEN_NONE, flags);
}

static void ufbxwi_disconnect_all_dst(ufbxw_scene *scene, ufbxw_connection_type type, ufbxw_id src_id)
{
	ufbxwi_element *src_elem = ufbxwi_get_element(scene, src_id);
	if (!src_elem) return;

	ufbxwi_connection_info info = ufbxwi_connection_infos[type];
	uint64_t src_bits = src_elem->conn_bits | (UFBXWI_CONN_BIT_PROPERTY | UFBXWI_CONN_BIT_ELEMENT);
	if ((src_bits & info.src_mask) != info.src_mask) return;

	ufbxwi_conn_type src_type = ufbxwi_conn_type(info.src_conn);
	void *src_data = (char*)src_elem + ufbxwi_conn_offset(info.src_conn);

	scene->tmp_ids.count = 0;
	ufbxwi_conn_collect_ids(scene, &scene->tmp_ids, src_type, src_data);
	ufbxwi_conn_clear(scene, src_type, src_data);

	ufbxwi_for_id_list(ufbxw_id, dst_id, scene->tmp_ids) {
		ufbxwi_element *dst_elem = ufbxwi_get_element(scene, dst_id);
		if (!dst_elem) continue;

		ufbxwi_conn_type dst_type = ufbxwi_conn_type(info.dst_conn);
		void *dst_data = (char*)dst_elem + ufbxwi_conn_offset(info.dst_conn);

		uint64_t dst_bits = dst_elem->conn_bits | (UFBXWI_CONN_BIT_PROPERTY | UFBXWI_CONN_BIT_ELEMENT);
		ufbxw_assert((dst_bits & info.dst_mask) == info.dst_mask);

		ufbxwi_conn_remove_all(scene, dst_type, dst_data, src_id);
	}
}

static void ufbxwi_disconnect_all_src(ufbxw_scene *scene, ufbxw_connection_type type, ufbxw_id dst_id)
{
	ufbxwi_element *dst_elem = ufbxwi_get_element(scene, dst_id);
	if (!dst_elem) return;

	ufbxwi_connection_info info = ufbxwi_connection_infos[type];
	uint64_t dst_bits = dst_elem->conn_bits | (UFBXWI_CONN_BIT_PROPERTY | UFBXWI_CONN_BIT_ELEMENT);
	if ((dst_bits & info.dst_mask) != info.dst_mask) return;

	ufbxwi_conn_type dst_type = ufbxwi_conn_type(info.dst_conn);
	void *dst_data = (char*)dst_elem + ufbxwi_conn_offset(info.dst_conn);

	scene->tmp_ids.count = 0;
	ufbxwi_conn_collect_ids(scene, &scene->tmp_ids, dst_type, dst_data);
	ufbxwi_conn_clear(scene, dst_type, dst_data);

	ufbxwi_for_id_list(ufbxw_id, src_id, scene->tmp_ids) {
		ufbxwi_element *src_elem = ufbxwi_get_element(scene, src_id);
		if (!src_elem) continue;

		ufbxwi_conn_type src_type = ufbxwi_conn_type(info.src_conn);
		void *src_data = (char*)src_elem + ufbxwi_conn_offset(info.src_conn);

		uint64_t src_bits = src_elem->conn_bits | (UFBXWI_CONN_BIT_PROPERTY | UFBXWI_CONN_BIT_ELEMENT);
		ufbxw_assert((src_bits & info.src_mask) != info.src_mask);

		ufbxwi_conn_remove_all(scene, src_type, src_data, dst_id);
	}
}

static void ufbxwi_collect_src_connections(const ufbxw_scene *scene, ufbxwi_allocator *ator, ufbxwi_conn_list *conns, ufbxw_connection_type type, ufbxwi_element *dst_elem)
{
	ufbxwi_connection_info info = ufbxwi_connection_infos[type];
	uint64_t dst_bits = dst_elem->conn_bits | (UFBXWI_CONN_BIT_PROPERTY | UFBXWI_CONN_BIT_ELEMENT);
	if ((dst_bits & info.dst_mask) != info.dst_mask) return;

	ufbxwi_conn_type dst_type = ufbxwi_conn_type(info.dst_conn);
	void *dst_data = (char*)dst_elem + ufbxwi_conn_offset(info.dst_conn);

	ufbxwi_conn_collect_conns(scene, ator, conns, dst_type, dst_data);
}

static ufbxw_anim_prop ufbxwi_animate_prop(ufbxw_scene *scene, ufbxw_id id, ufbxwi_token prop, ufbxw_anim_layer layer)
{
	ufbxwi_element *element = ufbxwi_get_element(scene, id);
	if (!element) return ufbxw_null_anim_prop;

	ufbxwi_prop *p = ufbxwi_element_find_prop(scene, element, prop);
	if (!p) return ufbxw_null_anim_prop;

	size_t curve_count = 0;

	ufbxwi_prop_type prop_type = scene->prop_types.data[p->type];
	switch (prop_type.data_type) {
	case UFBXW_PROP_DATA_BOOL:
	case UFBXW_PROP_DATA_INT32:
	case UFBXW_PROP_DATA_INT64:
	case UFBXW_PROP_DATA_REAL:
	case UFBXW_PROP_DATA_REAL_STRING:
	case UFBXW_PROP_DATA_USER_REAL:
	case UFBXW_PROP_DATA_USER_INT:
	case UFBXW_PROP_DATA_USER_ENUM:
		curve_count = 1;
		break;

	case UFBXW_PROP_DATA_VEC2:
		curve_count = 2;
		break;

	case UFBXW_PROP_DATA_VEC3:
		curve_count = 3;
		break;

	case UFBXW_PROP_DATA_VEC4:
		curve_count = 4;
		break;

	default:
		// Not supported for animation.
		break;
	}

	if (curve_count == 0) return ufbxw_null_anim_prop;

	const char *anim_name = "";
	switch (p->token) {
	case UFBXWI_Lcl_Translation:
		anim_name = "T";
		break;
	case UFBXWI_Lcl_Rotation:
		anim_name = "R";
		break;
	case UFBXWI_Lcl_Scaling:
		anim_name = "S";
		break;
	default:
		anim_name = scene->string_pool.tokens.data[p->token].data;
		break;
	}

	ufbxw_anim_prop anim = { ufbxw_create_element(scene, UFBXW_ELEMENT_ANIM_PROP) };
	ufbxw_set_name(scene, anim.id, anim_name);

	ufbxwi_connect_imp(scene, UFBXW_CONNECTION_ANIM_PROPERTY, anim.id, id, 0, prop, 0);

	if (layer.id != 0) {
		ufbxwi_connect(scene, UFBXW_CONNECTION_ANIM_PROP_LAYER, anim.id, layer.id, 0);
	}

	static const ufbxwi_token curve_props[] = {
		UFBXWI_d_X, UFBXWI_d_Y, UFBXWI_d_Z, UFBXWI_d_W,
	};

	ufbxwi_token first_curve_prop = curve_props[0];

	// For single channel propertes, use the property name
	// TODO: Flexible buffer
	char name_buf[256];
	if (curve_count == 1) {
		ufbxw_string prop_name = scene->string_pool.tokens.data[prop];
		int name_len = snprintf(name_buf, sizeof(name_buf), "d|%s", prop_name.data);
		first_curve_prop = ufbxwi_intern_token(&scene->string_pool, name_buf, (size_t)name_len);
	}

	for (size_t i = 0; i < curve_count; i++) {
		ufbxw_anim_curve curve = { ufbxw_create_element(scene, UFBXW_ELEMENT_ANIM_CURVE) };

		ufbxwi_token prop_name = i == 0 ? first_curve_prop : curve_props[i];

		// Manually add the properties so we can point them to the known default values
		ufbxwi_element *anim_elem = ufbxwi_get_element(scene, anim.id);
		ufbxwi_check(anim_elem, ufbxw_null_anim_prop);

		ufbxwi_prop *p = ufbxwi_props_add_prop(scene, &anim_elem->props, prop_name);
		ufbxwi_check(p, ufbxw_null_anim_prop);

		p->type = UFBXW_PROP_TYPE_NUMBER;
		p->flags = UFBXW_PROP_FLAG_ANIMATABLE;
		p->value = ufbxwi_prop_value_field(offsetof(ufbxwi_anim_prop, defaults) + i * sizeof(ufbxw_real));

		ufbxwi_connect_imp(scene, UFBXW_CONNECTION_ANIM_CURVE_PROP, curve.id, anim.id, 0, prop_name, 0);
	}

	return anim;
}

static ufbxwi_forceinline ufbxwi_node *ufbxwi_get_node(ufbxw_scene *scene, ufbxw_node id) { return (ufbxwi_node*)ufbxwi_get_typed_element(scene, id.id, UFBXW_ELEMENT_NODE); }
static ufbxwi_forceinline ufbxwi_mesh *ufbxwi_get_mesh(ufbxw_scene *scene, ufbxw_mesh id) { return (ufbxwi_mesh*)ufbxwi_get_typed_element(scene, id.id, UFBXW_ELEMENT_MESH); }
static ufbxwi_forceinline ufbxwi_skin_deformer *ufbxwi_get_skin_deformer(ufbxw_scene *scene, ufbxw_skin_deformer id) { return (ufbxwi_skin_deformer*)ufbxwi_get_typed_element(scene, id.id, UFBXW_ELEMENT_SKIN_DEFORMER); }
static ufbxwi_forceinline ufbxwi_skin_cluster *ufbxwi_get_skin_cluster(ufbxw_scene *scene, ufbxw_skin_cluster id) { return (ufbxwi_skin_cluster*)ufbxwi_get_typed_element(scene, id.id, UFBXW_ELEMENT_SKIN_CLUSTER); }
static ufbxwi_forceinline ufbxwi_blend_deformer *ufbxwi_get_blend_deformer(ufbxw_scene *scene, ufbxw_blend_deformer id) { return (ufbxwi_blend_deformer*)ufbxwi_get_typed_element(scene, id.id, UFBXW_ELEMENT_BLEND_DEFORMER); }
static ufbxwi_forceinline ufbxwi_blend_channel *ufbxwi_get_blend_channel(ufbxw_scene *scene, ufbxw_blend_channel id) { return (ufbxwi_blend_channel*)ufbxwi_get_typed_element(scene, id.id, UFBXW_ELEMENT_BLEND_CHANNEL); }
static ufbxwi_forceinline ufbxwi_blend_shape *ufbxwi_get_blend_shape(ufbxw_scene *scene, ufbxw_blend_shape id) { return (ufbxwi_blend_shape*)ufbxwi_get_typed_element(scene, id.id, UFBXW_ELEMENT_BLEND_SHAPE); }
static ufbxwi_forceinline ufbxwi_light *ufbxwi_get_light(ufbxw_scene *scene, ufbxw_light id) { return (ufbxwi_light*)ufbxwi_get_typed_element(scene, id.id, UFBXW_ELEMENT_LIGHT); }
static ufbxwi_forceinline ufbxwi_camera *ufbxwi_get_camera(ufbxw_scene *scene, ufbxw_camera id) { return (ufbxwi_camera*)ufbxwi_get_typed_element(scene, id.id, UFBXW_ELEMENT_CAMERA); }
static ufbxwi_forceinline ufbxwi_bind_pose *ufbxwi_get_bind_pose(ufbxw_scene *scene, ufbxw_bind_pose id) { return (ufbxwi_bind_pose*)ufbxwi_get_typed_element(scene, id.id, UFBXW_ELEMENT_BIND_POSE); }
static ufbxwi_forceinline ufbxwi_anim_curve *ufbxwi_get_anim_curve(ufbxw_scene *scene, ufbxw_anim_curve id) { return (ufbxwi_anim_curve*)ufbxwi_get_typed_element(scene, id.id, UFBXW_ELEMENT_ANIM_CURVE); }
static ufbxwi_forceinline ufbxwi_anim_prop *ufbxwi_get_anim_prop(ufbxw_scene *scene, ufbxw_anim_prop id) { return (ufbxwi_anim_prop*)ufbxwi_get_typed_element(scene, id.id, UFBXW_ELEMENT_ANIM_PROP); }
static ufbxwi_forceinline ufbxwi_anim_layer *ufbxwi_get_anim_layer(ufbxw_scene *scene, ufbxw_anim_layer id) { return (ufbxwi_anim_layer*)ufbxwi_get_typed_element(scene, id.id, UFBXW_ELEMENT_ANIM_LAYER); }
static ufbxwi_forceinline ufbxwi_anim_stack *ufbxwi_get_anim_stack(ufbxw_scene *scene, ufbxw_anim_stack id) { return (ufbxwi_anim_stack*)ufbxwi_get_typed_element(scene, id.id, UFBXW_ELEMENT_ANIM_STACK); }

static ufbxwi_forceinline ufbxwi_node *ufbxwi_get_node_by_id(ufbxw_scene *scene, ufbxw_id id) { return (ufbxwi_node*)ufbxwi_get_typed_element(scene, id, UFBXW_ELEMENT_NODE); }
static ufbxwi_forceinline ufbxwi_mesh *ufbxwi_get_mesh_by_id(ufbxw_scene *scene, ufbxw_id id) { return (ufbxwi_mesh*)ufbxwi_get_typed_element(scene, id, UFBXW_ELEMENT_MESH); }
static ufbxwi_forceinline ufbxwi_skin_deformer *ufbxwi_get_skin_deformer_by_id(ufbxw_scene *scene, ufbxw_id id) { return (ufbxwi_skin_deformer*)ufbxwi_get_typed_element(scene, id, UFBXW_ELEMENT_SKIN_DEFORMER); }
static ufbxwi_forceinline ufbxwi_skin_cluster *ufbxwi_get_skin_cluster_by_id(ufbxw_scene *scene, ufbxw_id id) { return (ufbxwi_skin_cluster*)ufbxwi_get_typed_element(scene, id, UFBXW_ELEMENT_SKIN_CLUSTER); }
static ufbxwi_forceinline ufbxwi_blend_deformer *ufbxwi_get_blend_deformer_by_id(ufbxw_scene *scene, ufbxw_id id) { return (ufbxwi_blend_deformer*)ufbxwi_get_typed_element(scene, id, UFBXW_ELEMENT_BLEND_DEFORMER); }
static ufbxwi_forceinline ufbxwi_blend_channel *ufbxwi_get_blend_channel_by_id(ufbxw_scene *scene, ufbxw_id id) { return (ufbxwi_blend_channel*)ufbxwi_get_typed_element(scene, id, UFBXW_ELEMENT_BLEND_CHANNEL); }
static ufbxwi_forceinline ufbxwi_blend_shape *ufbxwi_get_blend_shape_by_id(ufbxw_scene *scene, ufbxw_id id) { return (ufbxwi_blend_shape*)ufbxwi_get_typed_element(scene, id, UFBXW_ELEMENT_BLEND_SHAPE); }
static ufbxwi_forceinline ufbxwi_light *ufbxwi_get_light_by_id(ufbxw_scene *scene, ufbxw_id id) { return (ufbxwi_light*)ufbxwi_get_typed_element(scene, id, UFBXW_ELEMENT_LIGHT); }
static ufbxwi_forceinline ufbxwi_camera *ufbxwi_get_camera_by_id(ufbxw_scene *scene, ufbxw_id id) { return (ufbxwi_camera*)ufbxwi_get_typed_element(scene, id, UFBXW_ELEMENT_CAMERA); }
static ufbxwi_forceinline ufbxwi_bind_pose *ufbxwi_get_bind_pose_by_id(ufbxw_scene *scene, ufbxw_id id) { return (ufbxwi_bind_pose*)ufbxwi_get_typed_element(scene, id, UFBXW_ELEMENT_BIND_POSE); }
static ufbxwi_forceinline ufbxwi_anim_curve *ufbxwi_get_anim_curve_by_id(ufbxw_scene *scene, ufbxw_id id) { return (ufbxwi_anim_curve*)ufbxwi_get_typed_element(scene, id, UFBXW_ELEMENT_ANIM_CURVE); }
static ufbxwi_forceinline ufbxwi_anim_prop *ufbxwi_get_anim_prop_by_id(ufbxw_scene *scene, ufbxw_id id) { return (ufbxwi_anim_prop*)ufbxwi_get_typed_element(scene, id, UFBXW_ELEMENT_ANIM_PROP); }
static ufbxwi_forceinline ufbxwi_anim_layer *ufbxwi_get_anim_layer_by_id(ufbxw_scene *scene, ufbxw_id id) { return (ufbxwi_anim_layer*)ufbxwi_get_typed_element(scene, id, UFBXW_ELEMENT_ANIM_LAYER); }
static ufbxwi_forceinline ufbxwi_anim_stack *ufbxwi_get_anim_stack_by_id(ufbxw_scene *scene, ufbxw_id id) { return (ufbxwi_anim_stack*)ufbxwi_get_typed_element(scene, id, UFBXW_ELEMENT_ANIM_STACK); }

static ufbxwi_forceinline ufbxw_node ufbxwi_assert_node(ufbxw_id id) { ufbxw_assert(ufbxwi_id_type(id) == UFBXW_ELEMENT_NODE); ufbxw_node v = { id }; return v; }
static ufbxwi_forceinline ufbxw_anim_curve ufbxwi_assert_anim_curve(ufbxw_id id) { ufbxw_assert(ufbxwi_id_type(id) == UFBXW_ELEMENT_ANIM_CURVE); ufbxw_anim_curve v = { id }; return v; }
static ufbxwi_forceinline ufbxw_anim_layer ufbxwi_assert_anim_layer(ufbxw_id id) { ufbxw_assert(ufbxwi_id_type(id) == UFBXW_ELEMENT_ANIM_LAYER); ufbxw_anim_layer v = { id }; return v; }
static ufbxwi_forceinline ufbxw_anim_stack ufbxwi_assert_anim_stack(ufbxw_id id) { ufbxw_assert(ufbxwi_id_type(id) == UFBXW_ELEMENT_ANIM_STACK); ufbxw_anim_stack v = { id }; return v; }

static bool ufbwi_init_tokens(ufbxw_scene *scene)
{
	// Reserve the none and empty tokens
	{
		ufbxw_string *null_tokens = ufbxwi_list_push_zero_n(&scene->ator, &scene->string_pool.tokens, ufbxw_string, 2);
		ufbxwi_check(null_tokens, false);
		null_tokens[0].data = ufbxwi_empty_char;
		null_tokens[1].data = ufbxwi_empty_char;
	}

	ufbxwi_for(const ufbxw_string, str, ufbxwi_tokens, ufbxwi_arraycount(ufbxwi_tokens)) {
		ufbxwi_check(ufbxwi_intern_token(&scene->string_pool, str->data, str->length), false);
	}

	// Reserve the count token
	{
		ufbxw_string *null_tokens = ufbxwi_list_push_zero_n(&scene->ator, &scene->string_pool.tokens, ufbxw_string, 1);
		ufbxwi_check(null_tokens, false);
		null_tokens[0].data = ufbxwi_empty_char;
	}

	return true;
}

static bool ufbxwi_init_prop_types(ufbxw_scene *scene)
{
	ufbxwi_prop_type *dst_type = ufbxwi_list_push_zero_n(&scene->ator, &scene->prop_types, ufbxwi_prop_type, ufbxwi_arraycount(ufbxwi_prop_types));
	ufbxwi_check(dst_type, false);

	ufbxwi_for(const ufbxwi_prop_type_desc, desc, ufbxwi_prop_types, ufbxwi_arraycount(ufbxwi_prop_types)) {
		ufbxwi_check(ufbxwi_intern_string(&scene->string_pool, &dst_type->type, desc->type, strlen(desc->type)), false);
		ufbxwi_check(ufbxwi_intern_string(&scene->string_pool, &dst_type->sub_type, desc->sub_type, strlen(desc->sub_type)), false);
		dst_type->data_type = desc->data_type;

		dst_type++;
	}

	return true;
}

static bool ufbxwi_create_object_types(ufbxw_scene *scene)
{
	ufbxwi_object_type *object_types = ufbxwi_list_push_zero_n(&scene->ator, &scene->object_types, ufbxwi_object_type, ufbxwi_arraycount(ufbxwi_object_types));
	ufbxwi_check(object_types, false);

	ufbxwi_object_type *dst = object_types;
	ufbxwi_for(const ufbxwi_object_desc, desc, ufbxwi_object_types, ufbxwi_arraycount(ufbxwi_object_types)) {
		dst->type = desc->type;
		dst++;
	}

	return true;
}

static bool ufbxwi_create_element_types(ufbxw_scene *scene)
{
	for (size_t type_ix = 0; type_ix < ufbxwi_arraycount(ufbxwi_element_types); type_ix++) {
		const ufbxwi_element_type_desc *desc = &ufbxwi_element_types[type_ix];
		ufbxwi_element_type *et = ufbxwi_list_push_zero(&scene->ator, &scene->element_types, ufbxwi_element_type);
		ufbxwi_check(et, false);

		et->element_type = desc->element_type;
		et->class_type = desc->class_type;
		et->sub_type = desc->sub_type;
		et->object_type = desc->object_type;
		et->fbx_type = desc->fbx_type;
		et->init_fn = desc->init_fn;
		et->flags = desc->flags;
		et->desc = desc;

		et->object_type_id = ~0u;
		for (uint32_t i = 0; i < scene->object_types.count; i++) {
			if (scene->object_types.data[i].type == desc->object_type) {
				et->object_type_id = i;
				break;
			}
		}
		ufbxw_assert(et->object_type_id != ~0u || (desc->flags & UFBXWI_ELEMENT_FLAG_ALLOW_NO_OBJECT_ID) != 0);
	}

	return true;
}

static void ufbxwi_create_defaults(ufbxw_scene *scene)
{
	if (!scene->opts.no_default_scene_info) {
		ufbxw_id id = ufbxw_create_element(scene, UFBXW_ELEMENT_SCENE_INFO);
		ufbxw_set_name(scene, id, "GlobalInfo");
	}

	if (!scene->opts.no_default_global_settings) {
		ufbxw_id id = ufbxw_create_element(scene, UFBXW_ELEMENT_GLOBAL_SETTINGS);

		// TODO: Make these (and the rest) into actual fast access fields
		ufbxw_add_int(scene, id, "UpAxis", UFBXW_PROP_TYPE_INT, 1);
		ufbxw_add_int(scene, id, "UpAxisSign", UFBXW_PROP_TYPE_INT, 1);
		ufbxw_add_int(scene, id, "FrontAxis", UFBXW_PROP_TYPE_INT, 2);
		ufbxw_add_int(scene, id, "FrontAxisSign", UFBXW_PROP_TYPE_INT, 1);
		ufbxw_add_int(scene, id, "CoordAxis", UFBXW_PROP_TYPE_INT, 0);
		ufbxw_add_int(scene, id, "CoordAxisSign", UFBXW_PROP_TYPE_INT, 1);
		ufbxw_add_vec3(scene, id, "AmbientColor", UFBXW_PROP_TYPE_COLOR_RGB, ufbxw_zero_vec3);
		ufbxw_add_string(scene, id, "DefaultCamera", UFBXW_PROP_TYPE_STRING, "Producer Perspective");
	}

	if (!scene->opts.no_default_document) {
		ufbxw_id id = ufbxw_create_element(scene, UFBXW_ELEMENT_DOCUMENT);
	}

	if (!scene->opts.no_default_anim_stack) {
		ufbxw_anim_stack default_stack = ufbxw_create_anim_stack(scene);
		ufbxw_set_name(scene, default_stack.id, "Take 001");
		scene->default_anim_stack = default_stack;
	}

	if (!scene->opts.no_default_anim_layer) {
		ufbxw_anim_layer default_layer = ufbxw_create_anim_layer(scene, scene->default_anim_stack);
		ufbxw_set_name(scene, default_layer.id, "BaseLayer");
		scene->default_anim_layer = default_layer;
	}
}

static bool ufbxwi_get_local_date(ufbxw_datetime *dst)
{
	time_t stamp = time(NULL);
	if (stamp == (time_t)-1) return false;

	struct tm t;
#if defined(_WIN32)
	if (localtime_s(&t, &stamp) != 0) return false;
#elif defined(__unix__) || defined(__apple__)
	if (localtime_r(&stamp, &t) == NULL) return false;
#else
	struct tm *tm_p = localtime(&stamp);
	if (tm_p == NULL) return false;
	t = *tm_p;
#endif

	dst->year = 1900 + t.tm_year;
	dst->month = 1 + t.tm_mon;
	dst->day = t.tm_mday;
	dst->hour = t.tm_hour;
	dst->minute = t.tm_min;
	dst->second = t.tm_sec;
	dst->millisecond = 0;
	return true;
}

static bool ufbxwi_get_utc_date(ufbxw_datetime *dst)
{
	time_t stamp = time(NULL);
	if (stamp == (time_t)-1) return false;

	struct tm t;
#if defined(_WIN32)
	if (gmtime_s(&t, &stamp) != 0) return false;
#elif defined(__unix__) || defined(__apple__)
	if (gmtime_r(&stamp, &t) == NULL) return false;
#else
	struct tm *tm_p = gmtime(&stamp);
	if (tm_p == NULL) return false;
	t = *tm_p;
#endif

	dst->year = 1900 + t.tm_year;
	dst->month = 1 + t.tm_mon;
	dst->day = t.tm_mday;
	dst->hour = t.tm_hour;
	dst->minute = t.tm_min;
	dst->second = t.tm_sec;
	dst->millisecond = 0;
	return true;
}

static bool ufbxwi_is_zero_date(const ufbxw_datetime *dt)
{
	return dt->year == 0 && dt->month == 0 && dt->day == 0 && dt->hour == 0 && dt->minute == 0 && dt->second == 0 && dt->millisecond == 0;
}

static bool ufbxwi_is_valid_date(const ufbxw_datetime *dt)
{
	 if (dt->month < 1 || dt->month > 12) return false;
	 if (dt->day < 1 || dt->day > 31) return false;
	 if (dt->hour < 0 || dt->hour > 23) return false;
	 if (dt->minute < 0 || dt->minute > 59) return false;
	 if (dt->second < 0 || dt->second > 60) return false;
	 if (dt->millisecond < 0 || dt->millisecond > 1000) return false;
	 return true;
}

static ufbxw_string ufbxwi_format_date(char *dst, size_t dst_size, const ufbxw_datetime *dt)
{
	if (dt->year == 0) return ufbxw_empty_string;
	if (!ufbxwi_is_valid_date(dt)) return ufbxw_empty_string;

	int len = snprintf(dst, dst_size, "%02d/%02d/%04d %02d:%02d:%02d.%03d", dt->day, dt->month, dt->year, dt->hour, dt->minute, dt->second, dt->millisecond);
	if (len > 0) {
		ufbxw_string result = { dst, (size_t)len };
		return result;
	} else {
		return ufbxw_empty_string;
	}
}

static void ufbxwi_mark_scene_failed(ufbxw_scene *scene)
{
	ufbxwi_mark_allocator_failed(&scene->ator);
	ufbxwi_mark_buffers_failed(&scene->buffers);
	scene->elements.data = NULL;
	scene->elements.count = 0;
	scene->elements.capacity = 0;
}

static void ufbxwi_scene_fatal(void *user, ufbxwi_error *error)
{
	ufbxw_scene *scene = (ufbxw_scene*)user;

	ufbxwi_mark_scene_failed(scene);
}

static void ufbxwi_init_scene(ufbxw_scene *scene)
{
	scene->error.fatal_fn = &ufbxwi_scene_fatal;
	scene->error.fatal_user = scene;

	scene->ator.error = &scene->error;

	scene->string_pool.ator = &scene->ator;
	scene->string_pool.error = &scene->error;

	ufbxwi_buffer_pool_init(&scene->buffers, &scene->ator, &scene->error);

	if (scene->opts.no_default_elements) {
		scene->opts.no_default_scene_info = true;
		scene->opts.no_default_document = true;
	}

	// size_t num_prop_types = (size_t)UFBXW_PROP_FIRST_USER;
	// ufbxwi_check(ufbxwi_list_push_zero_n(&scene->ator, &scene->prop_types, ufbxwi_prop_type, num_prop_types));

	ufbwi_init_tokens(scene);
	ufbxwi_init_prop_types(scene);
	ufbxwi_create_object_types(scene);
	ufbxwi_create_element_types(scene);
	ufbxwi_create_defaults(scene);
}

// -- Node

static ufbxwi_forceinline bool ufbxwi_is_vec3_zero(ufbxw_vec3 v)
{
	return (v.x == 0.0) & (v.y == 0.0) & (v.z == 0.0);
}

static ufbxwi_forceinline bool ufbxwi_is_quat_identity(ufbxw_quat v)
{
	return (v.x == 0.0) & (v.y == 0.0) & (v.z == 0.0) & (v.w == 1.0);
}

static ufbxwi_forceinline void ufbxwi_add_translate(ufbxw_transform *t, ufbxw_vec3 v)
{
	t->translation.x += v.x;
	t->translation.y += v.y;
	t->translation.z += v.z;
}

static ufbxwi_forceinline void ufbxwi_sub_translate(ufbxw_transform *t, ufbxw_vec3 v)
{
	t->translation.x -= v.x;
	t->translation.y -= v.y;
	t->translation.z -= v.z;
}

static ufbxwi_forceinline void ufbxwi_mul_scale(ufbxw_transform *t, ufbxw_vec3 v)
{
	t->translation.x *= v.x;
	t->translation.y *= v.y;
	t->translation.z *= v.z;
	t->scale.x *= v.x;
	t->scale.y *= v.y;
	t->scale.z *= v.z;
}

static ufbxwi_noinline ufbxw_quat ufbxwi_mul_quat(ufbxw_quat a, ufbxw_quat b)
{
	ufbxw_quat r;
	r.x = a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y;
	r.y = a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x;
	r.z = a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w;
	r.w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z;
	return r;
}

static void ufbxwi_mul_rotate(ufbxw_transform *t, ufbxw_vec3 v, ufbxw_rotation_order order)
{
	if (ufbxwi_is_vec3_zero(v)) return;

	ufbxw_quat q = ufbxw_euler_to_quat(v, order);
	if (t->rotation.w != 1.0) {
		t->rotation = ufbxwi_mul_quat(q, t->rotation);
	} else {
		t->rotation = q;
	}

	if (!ufbxwi_is_vec3_zero(t->translation)) {
		t->translation = ufbxw_quat_rotate_vec3(q, t->translation);
	}
}

static void ufbxwi_mul_rotate_quat(ufbxw_transform *t, ufbxw_quat q)
{
	if (ufbxwi_is_quat_identity(q)) return;

	if (t->rotation.w != 1.0) {
		t->rotation = ufbxwi_mul_quat(q, t->rotation);
	} else {
		t->rotation = q;
	}

	if (!ufbxwi_is_vec3_zero(t->translation)) {
		t->translation = ufbxw_quat_rotate_vec3(q, t->translation);
	}
}

static void ufbxwi_mul_inv_rotate(ufbxw_transform *t, ufbxw_vec3 v, ufbxw_rotation_order order)
{
	if (ufbxwi_is_vec3_zero(v)) return;

	ufbxw_quat q = ufbxw_euler_to_quat(v, order);
	q.x = -q.x; q.y = -q.y; q.z = -q.z;
	if (t->rotation.w != 1.0) {
		t->rotation = ufbxwi_mul_quat(q, t->rotation);
	} else {
		t->rotation = q;
	}

	if (!ufbxwi_is_vec3_zero(t->translation)) {
		t->translation = ufbxw_quat_rotate_vec3(q, t->translation);
	}
}

static ufbxwi_noinline ufbxw_matrix ufbxwi_matrix_mul_affine(const ufbxw_matrix *a, const ufbxw_matrix *b)
{
	ufbxw_assert(a && b);
	if (!a || !b) return ufbxw_identity_matrix;

	ufbxw_matrix dst;

	dst.m00 = a->m00*b->m00 + a->m01*b->m10 + a->m02*b->m20;
	dst.m10 = a->m10*b->m00 + a->m11*b->m10 + a->m12*b->m20;
	dst.m20 = a->m20*b->m00 + a->m21*b->m10 + a->m22*b->m20;
	dst.m30 = 0.0f;

	dst.m01 = a->m00*b->m01 + a->m01*b->m11 + a->m02*b->m21;
	dst.m11 = a->m10*b->m01 + a->m11*b->m11 + a->m12*b->m21;
	dst.m21 = a->m20*b->m01 + a->m21*b->m11 + a->m22*b->m21;
	dst.m31 = 0.0f;

	dst.m02 = a->m00*b->m02 + a->m01*b->m12 + a->m02*b->m22;
	dst.m12 = a->m10*b->m02 + a->m11*b->m12 + a->m12*b->m22;
	dst.m22 = a->m20*b->m02 + a->m21*b->m12 + a->m22*b->m22;
	dst.m32 = 0.0f;

	dst.m03 = a->m00*b->m03 + a->m01*b->m13 + a->m02*b->m23 + a->m03;
	dst.m13 = a->m10*b->m03 + a->m11*b->m13 + a->m12*b->m23 + a->m13;
	dst.m23 = a->m20*b->m03 + a->m21*b->m13 + a->m22*b->m23 + a->m23;
	dst.m33 = 1.0f;

	return dst;
}

// -- Mesh

typedef enum {
	UFBXWI_MESH_ATTRIBUTE_TYPE_NONE = 0x01,
	UFBXWI_MESH_ATTRIBUTE_TYPE_INT = 0x02,
	UFBXWI_MESH_ATTRIBUTE_TYPE_REAL = 0x04,
	UFBXWI_MESH_ATTRIBUTE_TYPE_VEC2 = 0x08,
	UFBXWI_MESH_ATTRIBUTE_TYPE_VEC3 = 0x10,
	UFBXWI_MESH_ATTRIBUTE_TYPE_VEC4 = 0x20,

	UFBXWI_MESH_ATTRIBUTE_FLAG_REQUIRE_INDICES = 0x100,
	UFBXWI_MESH_ATTRIBUTE_FLAG_FORBID_INDICES = 0x200,

	UFBXWI_MESH_ATTRIBUTE_FLAG_HAS_VALUES_W = 0x1000,
} ufbxwi_mesh_attribute_flags;

typedef struct {
	const char *layer_element_name;
	const char *values_name;
	const char *indices_name;
	int32_t version;
	int32_t order;
	uint32_t flags;
} ufbxwi_mesh_attribute_info;

typedef struct {
	const char *name;
} ufbxwi_attribute_mapping_info;

static const ufbxwi_mesh_attribute_info ufbxwi_mesh_attribute_infos[] = {
	{ NULL }, // Invalid
	{ "LayerElementNormal", "Normals", NULL, 102, 1, UFBXWI_MESH_ATTRIBUTE_TYPE_VEC3 | UFBXWI_MESH_ATTRIBUTE_FLAG_FORBID_INDICES | UFBXWI_MESH_ATTRIBUTE_FLAG_HAS_VALUES_W },
	{ "LayerElementUV", "UV", "UVIndex", 101, 5, UFBXWI_MESH_ATTRIBUTE_TYPE_VEC2 | UFBXWI_MESH_ATTRIBUTE_FLAG_REQUIRE_INDICES },
	{ "LayerElementTangent", "Tangents", NULL, 102, 3, UFBXWI_MESH_ATTRIBUTE_TYPE_VEC3 | UFBXWI_MESH_ATTRIBUTE_FLAG_FORBID_INDICES | UFBXWI_MESH_ATTRIBUTE_FLAG_HAS_VALUES_W },
	{ "LayerElementBinormal", "Binormals", NULL, 102, 2, UFBXWI_MESH_ATTRIBUTE_TYPE_VEC3 | UFBXWI_MESH_ATTRIBUTE_FLAG_FORBID_INDICES | UFBXWI_MESH_ATTRIBUTE_FLAG_HAS_VALUES_W },
	{ "LayerElementColor", "Colors", "ColorIndex", 101, 4, UFBXWI_MESH_ATTRIBUTE_TYPE_VEC4 | UFBXWI_MESH_ATTRIBUTE_FLAG_REQUIRE_INDICES },
	{ "LayerElementSmoothing", "Smoothing", "", 102, 6, UFBXWI_MESH_ATTRIBUTE_TYPE_INT | UFBXWI_MESH_ATTRIBUTE_FLAG_FORBID_INDICES },
	{ "LayerElementMaterial", NULL, "Materials", 101, 7, UFBXWI_MESH_ATTRIBUTE_TYPE_NONE | UFBXWI_MESH_ATTRIBUTE_FLAG_REQUIRE_INDICES },
};

static const ufbxwi_attribute_mapping_info ufbxwi_attribute_mapping_infos[] = {
	{ NULL }, // Invalid
	{ "ByPolygonVertex" },
	{ "ByVertice" },
	{ "ByEdge" },
	{ "ByPolygon" },
	{ "AllSame" },
};

static size_t ufbxwi_stream_triangle_faces(void *user, int32_t *dst, size_t dst_size, size_t offset)
{
	for (size_t i = 0; i < dst_size; i++) {
		dst[i] = (int32_t)((offset + i) * 3);
	}
	return dst_size;
}

static size_t ufbxwi_stream_real_one(void *user, ufbxw_real *dst, size_t dst_size, size_t offset)
{
	for (size_t i = 0; i < dst_size; i++) {
		dst[i] = 1.0f;
	}
	return dst_size;
}

static ufbxwi_mesh_attribute *ufbxwi_edit_mesh_attribute(ufbxw_scene *scene, ufbxw_mesh mesh, ufbxw_mesh_attribute attribute, int32_t set)
{
	ufbxwi_mesh *md = ufbxwi_get_mesh(scene, mesh);
	if (!md) return NULL;

	ufbxwi_for_list(ufbxwi_mesh_attribute, attrib, md->attributes) {
		if (attrib->attribute == attribute && attrib->set == set) {
			return attrib;
		}
	}

	ufbxwi_mesh_attribute *attrib = ufbxwi_list_push_zero(&scene->ator, &md->attributes, ufbxwi_mesh_attribute);
	ufbxwi_check(attrib, NULL);

	attrib->attribute = attribute;
	attrib->set = set;
	return attrib;
}

typedef struct {
	uint32_t hash;
	int32_t index;
} ufbxwi_index_hash_entry;

static void ufbxwi_generate_indices(ufbxw_scene *scene, ufbxw_mesh_attribute_desc *desc)
{
	// Currently not handles, practically not supported
	ufbxw_assert(!desc->values_w);

	ufbxwi_buffer_type value_type = ufbxwi_buffer_id_type(desc->values);
	size_t value_size = ufbxwi_buffer_type_infos[value_type].size;

	ufbxwi_make_buffer_owned(&scene->buffers, desc->values);
	ufbxwi_mutable_void_span values = ufbxwi_get_buffer_owned_data(&scene->buffers, desc->values);
	ufbxw_int_buffer index_buffer = ufbxw_create_int_buffer(scene, values.count);
	ufbxw_int_list indices = ufbxw_edit_int_buffer(scene, index_buffer);
	if (ufbxwi_is_fatal(&scene->error)) return;

	size_t hash_count = 1;
	while (hash_count < values.count * 2) {
		hash_count *= 2;
	}

	size_t value_count = 0;

	ufbxwi_index_hash_entry *hashes = ufbxwi_alloc(&scene->ator, ufbxwi_index_hash_entry, hash_count);
	if (!hashes) return;
	memset(hashes, 0, hash_count * sizeof(ufbxwi_index_hash_entry));

	for (size_t i = 0; i < values.count; i++) {
		const void *value = (const char*)values.data + i * value_size;

		// TODO: Better aligned hash
		uint32_t hash = ufbxwi_hash_string((const char*)value, value_size);

		// TODO: Better hashing
		uint32_t slot = hash;
		for (;;) {
			slot = slot & (hash_count - 1);
			if (hashes[slot].hash == hash) {
				int32_t index = hashes[slot].index;
				if (!memcmp((const char*)values.data + index * value_size, value, value_size)) {
					indices.data[i] = index;
					break;
				}
			} else if (hashes[slot].hash == 0) {
				int32_t index = (int32_t)value_count++;
				memcpy((char*)values.data + index * value_size, value, value_size);
				indices.data[i] = index;
				hashes[slot].hash = hash;
				hashes[slot].index = index;
				break;
			}
			slot++;
		}
	}


	ufbxwi_free(&scene->ator, hashes);

	// Currently assuming these are always user buffers
	ufbxw_free_buffer(scene, desc->values);

	ufbxw_buffer_id value_buffer = ufbxwi_create_owned_buffer(&scene->buffers, value_type, value_count);
	ufbxwi_mutable_void_span result_values = ufbxwi_get_buffer_owned_data(&scene->buffers, value_buffer);
	if (ufbxwi_is_fatal(&scene->error)) return;

	memcpy(result_values.data, values.data, value_count * value_size);

	desc->values = ufbxwi_to_user_buffer(&scene->buffers, value_buffer);
	desc->indices = index_buffer.id;
}

// -- Animation

typedef struct {
	ufbxw_ktime time;
	float value;
	uint32_t attr_index;
} ufbxwi_sort_keyframe;


static void ufbxwi_sort_anim_kefyrames(ufbxw_scene *scene, ufbxwi_anim_curve *c)
{
	size_t key_count = c->key_values.count;
	size_t capacity = key_count * 2 * sizeof(ufbxwi_sort_keyframe);
	ufbxwi_list_resize_uninit(&scene->ator, &scene->tmp_list, char, capacity);
	if (ufbxwi_is_fatal(&scene->error)) return;

	ufbxwi_sort_keyframe *keys = (ufbxwi_sort_keyframe*)scene->tmp_list.data;
	ufbxwi_sort_keyframe *tmp_keys = keys + key_count;
	for (size_t i = 0; i < key_count; i++) {
		keys[i].time = c->key_times.data[i];
		keys[i].value = c->key_values.data[i];
		keys[i].attr_index = c->key_attr_indices.data[i];
	}

	ufbxwwi_macro_stable_sort(ufbxwi_sort_keyframe, 16, keys, tmp_keys, key_count, ( a->time < b->time ));

	// Store back and remove duplicate keys
	size_t dst_count = 0;
	for (size_t src = 0; src < key_count; src++) {
		ufbxwi_sort_keyframe key = keys[src];
		if (dst_count > 0 && c->key_times.data[dst_count - 1] == key.time) {
			// Overwrite the previous key if it has identical time
			dst_count--;
		}
		c->key_times.data[dst_count] = key.time;
		c->key_values.data[dst_count] = key.value;
		c->key_attr_indices.data[dst_count] = key.attr_index;
		dst_count++;
	}
	c->key_times.count = dst_count;
	c->key_values.count = dst_count;
	c->key_attr_indices.count = dst_count;
}

// -- Pre-saving

static bool ufbxwi_less_id(void *user, const void *va, const void *vb)
{
	ufbxw_id a = *(const ufbxw_id*)va, b = *(const ufbxw_id*)vb;
	return a < b;
}

ufbxw_abi void ufbxw_set_save_info(ufbxw_scene *scene, const ufbxw_save_info *info)
{
	ufbxw_assert(info);
	ufbxw_assert(info->_begin_zero == 0 && info->_end_zero == 0);

	ufbxw_id scene_info_id = ufbxw_get_scene_info_id(scene);
	ufbxwi_scene_info *scene_info = (ufbxwi_scene_info*)ufbxwi_get_element(scene, scene_info_id);

	ufbxwi_intern_string_str(&scene->string_pool, &scene_info->document_url, info->document_url);

	ufbxwi_intern_string_str_or_default(&scene->string_pool, &scene_info->src_document_url, info->src_document_url, info->document_url);

	ufbxwi_intern_string_str(&scene->string_pool, &scene_info->last_application_name, info->application_name);
	ufbxwi_intern_string_str(&scene->string_pool, &scene_info->last_application_vendor, info->application_vendor);
	ufbxwi_intern_string_str(&scene->string_pool, &scene_info->last_application_version, info->application_version);

	ufbxwi_intern_string_str_or_default(&scene->string_pool, &scene_info->original_application_name, info->original_application_name, info->application_name);
	ufbxwi_intern_string_str_or_default(&scene->string_pool, &scene_info->original_application_vendor, info->original_application_vendor, info->application_vendor);
	ufbxwi_intern_string_str_or_default(&scene->string_pool, &scene_info->original_application_version, info->original_application_version, info->application_version);
	ufbxwi_intern_string_str_or_default(&scene->string_pool, &scene_info->original_filename, info->original_filename, info->document_url);

	char date_buffer[128];
	ufbxw_datetime last_time_utc = info->date_time_utc;
	if (ufbxwi_is_zero_date(&last_time_utc) && !info->no_default_date_time) {
		ufbxwi_get_utc_date(&last_time_utc);
	}

	ufbxw_string last_time = ufbxwi_format_date(date_buffer, sizeof(date_buffer), &last_time_utc);
	ufbxwi_intern_string_str(&scene->string_pool, &scene_info->last_date_time, last_time);

	ufbxw_datetime original_time_utc = last_time_utc;
	if (!ufbxwi_is_zero_date(&info->original_date_time_utc)) {
		original_time_utc = info->original_date_time_utc;
	}

	ufbxw_string original_time = ufbxwi_format_date(date_buffer, sizeof(date_buffer), &original_time_utc);
	ufbxwi_intern_string_str(&scene->string_pool, &scene_info->original_date_time, original_time);
}

static void ufbxwi_prepare_scene(ufbxw_scene *scene, const ufbxw_prepare_opts *opts)
{
	size_t element_count = scene->num_elements;

	ufbxwi_id_span elements;
	elements.count = scene->num_elements;
	elements.data = ufbxwi_alloc(&scene->ator, ufbxw_id, elements.count);
	ufbxwi_check(elements.data);

	size_t real_count = ufbxw_get_elements(scene, elements.data, elements.count);
	ufbxw_assert(real_count == element_count);

	ufbxwi_id_span elements_by_type[UFBXW_ELEMENT_TYPE_COUNT] = { 0 };
	for (size_t begin = 0; begin < elements.count; ) {
		ufbxw_element_type type = ufbxwi_id_type(elements.data[begin]);
		size_t end = begin + 1;
		while (end < elements.count && ufbxwi_id_type(elements.data[end]) == type) {
			end++;
		}

		elements_by_type[type].data = elements.data + begin;
		elements_by_type[type].count = end - begin;
		begin = end;
	}

	if (opts->finish_keyframes) {
		ufbxwi_for_id_list(ufbxw_id, curve_id, elements_by_type[UFBXW_ELEMENT_ANIM_CURVE]) {
			ufbxw_anim_curve_finish_keyframes(scene, ufbxwi_assert_anim_curve(curve_id));
		}
	}

	if (opts->patch_anim_stack_times) {
		ufbxwi_for_id_list(ufbxw_id, stack_id, elements_by_type[UFBXW_ELEMENT_ANIM_STACK]) {
			ufbxwi_anim_stack *stack = ufbxwi_get_anim_stack_by_id(scene, stack_id);
			ufbxw_assert(stack);

			if (stack->local_start != 0 || stack->local_stop != 0) continue;

			ufbxw_ktime time_min = INT64_MAX;
			ufbxw_ktime time_max = INT64_MIN;

			ufbxwi_for_id_list(ufbxw_id, layer_id, stack->layers) {
				ufbxwi_anim_layer *layer = ufbxwi_get_anim_layer_by_id(scene, layer_id);
				ufbxw_assert(layer);

				ufbxwi_for_id_list(ufbxw_id, prop_id, layer->anim_props) {
					ufbxwi_anim_prop *prop = ufbxwi_get_anim_prop_by_id(scene, prop_id);
					ufbxw_assert(prop);

					for (size_t i = 0; i < prop->curves.count; i++) {
						ufbxw_id curve_id = prop->curves.data[i].id;
						if (!curve_id) continue;
						ufbxwi_anim_curve *curve = ufbxwi_get_anim_curve_by_id(scene, curve_id);
						ufbxw_assert(curve);

						if (curve->key_times.count > 0) {
							time_min = ufbxwi_min_i64(time_min, curve->key_times.data[0]);
							time_max = ufbxwi_max_i64(time_max, curve->key_times.data[curve->key_times.count - 1]);
						}
					}
				}
			}

			if (time_min <= time_max) {
				stack->local_start = time_min;
				stack->local_stop = time_max;
			}
		}
	}

	if (opts->patch_anim_stack_reference_times) {
		ufbxwi_for_id_list(ufbxw_id, stack_id, elements_by_type[UFBXW_ELEMENT_ANIM_STACK]) {
			ufbxwi_anim_stack *stack = ufbxwi_get_anim_stack_by_id(scene, stack_id);
			ufbxw_assert(stack);

			if (stack->reference_start != 0 || stack->reference_stop != 0) continue;
			stack->reference_start = stack->local_start;
			stack->reference_stop = stack->local_stop;
		}
	}

	if (opts->patch_global_settings_times) {
		ufbxw_id global_settings_id = ufbxw_get_global_settings_id(scene);
		ufbxwi_global_settings *global_settings = (ufbxwi_global_settings*)ufbxwi_get_element(scene, global_settings_id);
		if (global_settings && global_settings->time_span_start == 0 && global_settings->time_span_stop == 0) {
			ufbxw_ktime time_min = INT64_MAX;
			ufbxw_ktime time_max = INT64_MIN;
			ufbxwi_for_id_list(ufbxw_id, stack_id, elements_by_type[UFBXW_ELEMENT_ANIM_STACK]) {
				ufbxwi_anim_stack *stack = ufbxwi_get_anim_stack_by_id(scene, stack_id);
				ufbxw_assert(stack);

				time_min = ufbxwi_min_i64(time_min, stack->local_start);
				time_max = ufbxwi_max_i64(time_max, stack->local_stop);
			}
			if (time_min <= time_max) {
				global_settings->time_span_start = time_min;
				global_settings->time_span_stop = time_max;
			}
		}
	}

	if (opts->add_missing_skeletons) {
		ufbxwi_for_id_list(ufbxw_id, cluster_id, elements_by_type[UFBXW_ELEMENT_SKIN_CLUSTER]) {
			ufbxwi_skin_cluster *cluster = ufbxwi_get_skin_cluster_by_id(scene, cluster_id);
			if (!cluster->node.id) continue;

			ufbxwi_node *node = ufbxwi_get_node(scene, cluster->node);
			if (!node) continue;
			if (node->attribute != 0) continue;

			ufbxw_id skeleton = ufbxw_create_element(scene, UFBXW_ELEMENT_SKELETON);
			ufbxwi_connect(scene, UFBXW_CONNECTION_NODE_ATTRIBUTE, skeleton, node->element.id, 0);
			ufbxwi_check(!ufbxwi_is_fatal(&scene->error));
		}
	}

	if (opts->add_missing_bind_poses) {
		ufbxwi_for_id_list(ufbxw_id, skin_id, elements_by_type[UFBXW_ELEMENT_SKIN_DEFORMER]) {
			ufbxwi_skin_deformer *skin = ufbxwi_get_skin_deformer_by_id(scene, skin_id);
			if (skin->bind_pose.id) continue;

			ufbxw_bind_pose pose = ufbxw_create_bind_pose(scene);
			skin->bind_pose = pose;

			ufbxw_node mesh_node = { 0 };

			// HACK
			ufbxwi_for_id_list(ufbxw_id, geom_id, skin->deformer.geometries) {
				ufbxwi_node_attribute *geom = (ufbxwi_node_attribute*)ufbxwi_get_element(scene, geom_id);
				if (!geom) continue;

				ufbxwi_for_id_list(ufbxw_node, node_id, geom->instances) {
					ufbxwi_node *node = ufbxwi_get_node(scene, node_id);
					if (!node) continue;

					mesh_node = node_id;
					break;
				}
			}

			if (mesh_node.id) {
				ufbxw_matrix bind_matrix;
				if (skin->has_mesh_bind_transform) {
					bind_matrix = skin->mesh_bind_transform;
				} else {
					bind_matrix = ufbxw_node_get_global_transform(scene, mesh_node);
				}
				ufbxw_bind_pose_add_node(scene, pose, mesh_node, bind_matrix);
			}

			ufbxwi_for_id_list(ufbxw_id, cluster_id, skin->clusters) {
				ufbxwi_skin_cluster *cluster = ufbxwi_get_skin_cluster_by_id(scene, cluster_id);
				if (!cluster) continue;

				ufbxwi_node *node = ufbxwi_get_node(scene, cluster->node);
				if (!node) continue;

				ufbxw_bind_pose_add_node(scene, pose, cluster->node, cluster->transform);
			}

			ufbxwi_check(!ufbxwi_is_fatal(&scene->error));
		}
	}

	ufbxwi_free(&scene->ator, elements.data);
}

#endif

// -- Write queue

#if UFBXWI_FEATURE_WRITE_QUEUE

// We cannot write the output file consecutively for two reasons:
// - Binary files have forward offsets that we may not know yet
// - Threaded writing may compute parts of the file in parallel
//
// Some terminology:
// - Chunk: Logical chunk of the end-result file. May be backed by a buffer or a task.
// - Buffer: Reusable memory buffer, which may be shared between multiple chunks.
// - Task: Threaded function that will output the chunk data.
// - Reloc: An offset value within a chunk that is patched later (see below).

// Relocs:
// Binary FBX files contain multiple absolute and relative file offsets.
// For example, the common node header in version 7500+ binary FBX file looks like this:
//
//   uint64_t end_offset;  // < Absolute byte offset past the final child
//   uint64_t num_values;  // < Number of values in the node
//   uint64_t values_size; // < Size of the written values
//
// As we stream the output and potentially have threads writing the chunks asynchronously,
// we might now know the exact file offsets for the chunks until later. For example:
//
//   [chunk 0: <data>] { ... end_offset = chunk2 + 0x1234 }
//   [chunk 1: <task 0>]
//   [chunk 2: <data>]
//
// We may not know the exact file offset of `chunk2 + 0x1234` until `<task 0>` has completed,
// so we need to buffer these "relocations"/"relocs" (borrowed from linker terminology) until
// we know the actual file offsets for both the patch and target chunk.

// Reusable memory buffer.
// Does not represent anything more than a blob of memory.
typedef struct {
	char *begin;
	char *pos;
	char *end;

	// Number of pending chunks that refer to this memory buffer
	uint32_t refcount;

} ufbxwi_write_buffer;

typedef struct ufbxwi_write_chunk_part ufbxwi_write_chunk_part;

struct ufbxwi_write_chunk_part {
	// Actual data for this part, suballocated from `buffer`.
	void *data;
	uint32_t size;

	// Reusable memory buffer that this part uses.
	ufbxwi_write_buffer *buffer;

	// Pointer to the next part in a linked list
	ufbxwi_write_chunk_part *next;
};

// Logical chunk of the output file.
// Either backed by data or a deferred task.
typedef struct {

	// Potential deferred task
	ufbxwi_task_id task;

	// Actual file offset of this chunk
	bool has_file_offset;
	uint64_t file_offset;
	size_t total_size;

	// Number of unresolved relocs within this chunk
	uint32_t num_unresolved_relocs;

	// Relocs waiting for this chunk to get a file offset
	ufbxwi_uint32_list pending_relocs;

	// One or multiple parts that make up this chunk
	ufbxwi_write_chunk_part data;

	// Pointer to the last part
	ufbxwi_write_chunk_part *last_part;

	// Potential buffer associated with this chunk.
	ufbxw_buffer_id buffer_id;

} ufbxwi_write_chunk;

typedef enum {
	// Absolute file position offset
	UFBXWI_WRITE_RELOC_ABSOLUTE_U32,
	UFBXWI_WRITE_RELOC_ABSOLUTE_U64,
	// Relative file position offset
	UFBXWI_WRITE_RELOC_RELATIVE_U32,
	UFBXWI_WRITE_RELOC_RELATIVE_U64,
} ufbxwi_write_reloc_type;

// Patched offset value within a chunk
typedef struct {
	ufbxwi_write_chunk *patch_chunk;
	ufbxwi_write_chunk *target_chunk;

	// Offsets relative to the `patch_chunk`/`target_chunk`
	uint32_t patch_offset;     // < Offset to write the relocation to
	uint32_t reference_offset; // < Reference offset for relative relocs
	uint32_t target_offset;    // < Offset to write to the relocation

	ufbxwi_write_reloc_type type;
} ufbxwi_write_reloc;

UFBXWI_LIST_TYPE(ufbxwi_write_buffer_ptr_list, ufbxwi_write_buffer*);
UFBXWI_LIST_TYPE(ufbxwi_write_chunk_ptr_list, ufbxwi_write_chunk*);
UFBXWI_LIST_TYPE(ufbxwi_write_reloc_list, ufbxwi_write_reloc);

typedef struct {
	ufbxwi_allocator *ator;
	ufbxwi_error *error;
	ufbxwi_task_queue *task_queue;
	ufbxwi_thread_pool *thread_pool;
	ufbxwi_buffer_pool *buffer_pool;
	void *main_ctx;

	// Fast access pointers for the current buffer
	char *buffer_begin;
	char *buffer_pos;
	char *buffer_end;
	size_t direct_write_size;

	ufbxwi_write_chunk *current_chunk;
	ufbxwi_write_chunk_ptr_list chunks;
	ufbxwi_write_chunk_ptr_list chunks_to_flush;

	size_t preferred_buffer_size;

	ufbxw_write_stream stream;

	// Chunk flushing
	size_t chunk_layout_index;
	uint64_t chunk_layout_file_offset;

	ufbxwi_write_reloc_list relocs;
	ufbxwi_uint32_list free_reloc_ids;

	ufbxwi_mutex buffer_mutex;
	ufbxwi_write_buffer_ptr_list free_buffers;

} ufbxwi_write_queue;

static ufbxwi_noinline ufbxwi_write_buffer *ufbxwi_write_queue_alloc_buffer(ufbxwi_write_queue *wq, size_t min_size)
{
	ufbxwi_mutex_lock_if_enabled(wq->thread_pool, &wq->buffer_mutex);

	// TODO(wq): Something not stupid
	const size_t num_free_buffers = wq->free_buffers.count;
	for (size_t i = 0; i < num_free_buffers; i++) {
		ufbxwi_write_buffer *buffer = wq->free_buffers.data[i];
		size_t left = ufbxwi_to_size(buffer->end - buffer->pos);
		if (left >= min_size) {
			buffer->refcount++;

			wq->free_buffers.data[i] = wq->free_buffers.data[num_free_buffers - 1];
			wq->free_buffers.count = num_free_buffers - 1;

			ufbxwi_mutex_unlock_if_enabled(wq->thread_pool, &wq->buffer_mutex);
			return buffer;
		}
	}

	ufbxwi_write_buffer *buf = ufbxwi_alloc(wq->ator, ufbxwi_write_buffer, 1);
	ufbxwi_check(buf, NULL);

	const size_t size = ufbxwi_max_sz(wq->preferred_buffer_size, min_size);
	buf->begin = ufbxwi_alloc(wq->ator, char, size);
	ufbxwi_check(buf->begin, NULL);

	buf->pos = buf->begin;
	buf->end = buf->begin + size;
	buf->refcount = 1;

	ufbxwi_mutex_unlock_if_enabled(wq->thread_pool, &wq->buffer_mutex);
	return buf;
}

static ufbxwi_noinline void ufbxwi_write_queue_return_buffer(ufbxwi_write_queue *wq, ufbxwi_write_buffer *buffer)
{
	ufbxwi_mutex_lock_if_enabled(wq->thread_pool, &wq->buffer_mutex);

	// TODO(wq): Have something better than a linear list
	ufbxwi_check(ufbxwi_list_push_copy(wq->ator, &wq->free_buffers, ufbxwi_write_buffer*, &buffer));

	ufbxwi_mutex_unlock_if_enabled(wq->thread_pool, &wq->buffer_mutex);
}

static ufbxwi_noinline void ufbxwi_write_queue_free_buffer(ufbxwi_write_queue *wq, ufbxwi_write_buffer *buffer)
{
	if (!buffer) return;

	ufbxwi_mutex_lock_if_enabled(wq->thread_pool, &wq->buffer_mutex);

	if (--buffer->refcount == 0) {
		buffer->pos = buffer->begin;
	}

	ufbxwi_mutex_unlock_if_enabled(wq->thread_pool, &wq->buffer_mutex);
}

static ufbxwi_noinline void ufbxwi_write_queue_init(ufbxwi_write_queue *wq, ufbxwi_allocator *ator, ufbxwi_error *error, ufbxwi_task_queue *tq, void *main_ctx, ufbxw_write_stream stream, size_t buffer_size)
{
	wq->ator = ator;
	wq->error = error;
	wq->stream = stream;
	wq->task_queue = tq;
	if (tq) {
		wq->thread_pool = tq->thread_pool;
	}
	wq->main_ctx = main_ctx;

	wq->preferred_buffer_size = buffer_size;
	wq->direct_write_size = buffer_size / 2;

	ufbxwi_write_buffer *initial_buffer = ufbxwi_write_queue_alloc_buffer(wq, buffer_size);
	ufbxwi_check(initial_buffer);

	wq->buffer_begin = initial_buffer->pos;
	wq->buffer_pos = initial_buffer->pos;
	wq->buffer_end = initial_buffer->end;

	ufbxwi_write_chunk *initial_chunk = ufbxwi_alloc_zero(wq->ator, ufbxwi_write_chunk, 1);
	ufbxwi_check(initial_chunk);

	initial_chunk->data.data = wq->buffer_pos;
	initial_chunk->data.size = 0;
	initial_chunk->data.buffer = initial_buffer;

	initial_chunk->has_file_offset = true;
	initial_chunk->file_offset = 0;

	wq->current_chunk = initial_chunk;
}

static ufbxwi_noinline void ufbxwi_write_queue_free(ufbxwi_write_queue *wq)
{
	if (wq->stream.close_fn) {
		wq->stream.close_fn(wq->stream.user);
	}
}

static void ufbxwi_write_queue_resolve_reloc(ufbxwi_write_queue *wq, uint32_t reloc_id)
{
	ufbxw_assert(reloc_id != ~0u);

	ufbxwi_write_reloc *reloc = &wq->relocs.data[reloc_id];
	ufbxwi_write_chunk *patch_chunk = reloc->patch_chunk;
	ufbxwi_write_chunk *target_chunk = reloc->target_chunk;

	ufbxw_assert(patch_chunk->has_file_offset);
	ufbxw_assert(target_chunk->has_file_offset);

	void *dst = (char*)patch_chunk->data.data + reloc->patch_offset;
	const uint64_t target_offset = target_chunk->file_offset + reloc->target_offset;
	const uint64_t reference_offset = patch_chunk->file_offset + reloc->reference_offset;
	const uint64_t relative_offset = target_offset - reference_offset;

	switch (reloc->type) {
	case UFBXWI_WRITE_RELOC_ABSOLUTE_U32:
		*(uint32_t*)dst = (uint32_t)target_offset;
		break;
	case UFBXWI_WRITE_RELOC_ABSOLUTE_U64:
		*(uint64_t*)dst = target_offset;
		break;
	case UFBXWI_WRITE_RELOC_RELATIVE_U32:
		*(uint32_t*)dst = (uint32_t)relative_offset;
		break;
	case UFBXWI_WRITE_RELOC_RELATIVE_U64:
		*(uint64_t*)dst = relative_offset;
		break;
	}

	ufbxwi_dev_assert(patch_chunk->num_unresolved_relocs > 0);
	patch_chunk->num_unresolved_relocs--;

	ufbxwi_check(ufbxwi_list_push_copy(wq->ator, &wq->free_reloc_ids, uint32_t, &reloc_id));
}

static ufbxwi_noinline bool ufbxwi_write_queue_flush_chunks_imp(ufbxwi_write_queue *wq)
{
	const size_t num_prev_chunks = wq->chunks_to_flush.count;
	size_t num_new_chunks = 0;

	for (size_t i = 0; i < num_prev_chunks; i++) {
		ufbxwi_write_chunk *chunk = wq->chunks_to_flush.data[i];

		if (chunk->num_unresolved_relocs == 0 && chunk->has_file_offset) {
			ufbxwi_write_chunk_part *part = &chunk->data;

			uint64_t file_offset = chunk->file_offset;
			do {
				bool write_ok = wq->stream.write_fn(wq->stream.user, file_offset, part->data, part->size);
				if (!write_ok) {
					ufbxwi_fail(wq->error, UFBXW_ERROR_WRITE_FAILED, "failed to write to output stream");
					return false;
				}

				ufbxwi_write_queue_free_buffer(wq, part->buffer);

				// TODO(wq): Free the part (or pool it)

				file_offset += part->size;
				part = part->next;
			} while (part);

			// TODO(wq): Free chunk (or pool it)

		} else {
			wq->chunks_to_flush.data[num_new_chunks++] = chunk;
		}
	}
	wq->chunks_to_flush.count = num_new_chunks;

	return true;
}

static ufbxwi_noinline bool ufbxwi_write_queue_layout_chunks(ufbxwi_write_queue *wq, bool wait)
{
	while (wq->chunk_layout_index < wq->chunks.count) {
		ufbxwi_write_chunk *chunk = wq->chunks.data[wq->chunk_layout_index];

		// Complete a pending task
		if (chunk->task) {
			if (!ufbxwi_task_get_completed(wq->task_queue, chunk->task)) {
				if (wait) {
					// If we are blocking flush the pending chunks before waiting on the task.
					ufbxwi_check(ufbxwi_write_queue_flush_chunks_imp(wq), false);
					ufbxwi_task_complete(wq->task_queue, chunk->task, wq->main_ctx, true);
				} else {
					break;
				}
			}

			if (chunk->buffer_id) {
				ufbxwi_free_buffer(wq->buffer_pool, chunk->buffer_id);
			}

			chunk->task = 0;
		}

		// Layout the chunk into a file offset
		if (!chunk->has_file_offset) {
			chunk->file_offset = wq->chunk_layout_file_offset;
			chunk->has_file_offset = true;
		} else {
			ufbxw_assert(chunk->file_offset == wq->chunk_layout_file_offset);
		}
		wq->chunk_layout_file_offset += chunk->total_size;

		// Apply any relocations
		if (chunk->pending_relocs.count > 0) {
			ufbxwi_for_list(uint32_t, p_reloc_id, chunk->pending_relocs) {
				ufbxwi_write_queue_resolve_reloc(wq, *p_reloc_id);
			}
			ufbxwi_list_free(wq->ator, &chunk->pending_relocs);
		}

		// Queue the chunk for flushing
		ufbxwi_check(ufbxwi_list_push_copy(wq->ator, &wq->chunks_to_flush, ufbxwi_write_chunk*, &chunk), false);

		wq->chunk_layout_index++;
	}

	return true;
}

static ufbxwi_noinline bool ufbxwi_write_queue_flush_chunks(ufbxwi_write_queue *wq, bool wait)
{
	ufbxwi_check(ufbxwi_write_queue_layout_chunks(wq, wait), false);
	ufbxwi_check(ufbxwi_write_queue_flush_chunks_imp(wq), false);
	return true;
}

static ufbxwi_noinline bool ufbxwi_write_queue_flush(ufbxwi_write_queue *wq, size_t min_chunk_size)
{
	if (ufbxwi_is_fatal(wq->error)) return false;

	ufbxwi_write_chunk *prev_chunk = wq->current_chunk;
	ufbxwi_write_buffer *buffer = prev_chunk->data.buffer;

	size_t size = ufbxwi_to_size(wq->buffer_pos - wq->buffer_begin);
	prev_chunk->data.size = size;
	prev_chunk->total_size = size;

	buffer->pos = wq->buffer_pos;

	uint64_t file_offset = UINT64_MAX;
	if (prev_chunk->has_file_offset) {
		file_offset = prev_chunk->file_offset + size;
	}

	ufbxwi_check(ufbxwi_list_push_copy(wq->ator, &wq->chunks, ufbxwi_write_chunk*, &prev_chunk), false);

	// TODO(wq): Don't flush so eagerly
	ufbxwi_check(ufbxwi_write_queue_flush_chunks(wq, false), false);

	// Check if we can reuse the buffer of the current chunk
	size_t buffer_left = ufbxwi_to_size(buffer->end - buffer->pos);
	if (buffer_left >= min_chunk_size) {
		buffer->refcount++;
	} else {
		ufbxwi_write_queue_return_buffer(wq, buffer);
		buffer = ufbxwi_write_queue_alloc_buffer(wq, min_chunk_size);
		ufbxwi_check(buffer, false);
	}

	// TODO(wq): Pool these?
	ufbxwi_write_chunk *new_chunk = ufbxwi_alloc_zero(wq->ator, ufbxwi_write_chunk, 1);
	ufbxwi_check(new_chunk, false);

	new_chunk->data.buffer = buffer;
	new_chunk->data.data = buffer->pos;

	if (file_offset != UINT64_MAX) {
		new_chunk->file_offset = file_offset;
		new_chunk->has_file_offset = true;
	}

	wq->current_chunk = new_chunk;
	wq->buffer_begin = buffer->pos;
	wq->buffer_pos = buffer->pos;
	wq->buffer_end = buffer->end;

	return true;
}

static ufbxwi_noinline ufbxwi_write_chunk *ufbxwi_write_queue_reserve_chunk(ufbxwi_write_queue *wq)
{
	// Flush the current chunk
	ufbxwi_check(ufbxwi_write_queue_flush(wq, 0), NULL);

	// Reset the current file offset
	ufbxwi_write_chunk *cur_chunk = wq->current_chunk;
	cur_chunk->has_file_offset = false;
	cur_chunk->file_offset = 0;

	// Create and push an empty chunk
	ufbxwi_write_chunk *new_chunk = ufbxwi_alloc_zero(wq->ator, ufbxwi_write_chunk, 1);
	ufbxwi_check(new_chunk, NULL);

	ufbxwi_check(ufbxwi_list_push_copy(wq->ator, &wq->chunks, ufbxwi_write_chunk*, &new_chunk), NULL);

	return new_chunk;
}

static ufbxwi_noinline bool ufbxwi_write_queue_finish(ufbxwi_write_queue *wq)
{
	ufbxwi_check(ufbxwi_write_queue_flush(wq, 0), false);
	ufbxwi_check(ufbxwi_write_queue_flush_chunks(wq, true), false);
	return true;
}

static ufbxwi_noinline char *ufbxwi_queue_write_reserve_slow(ufbxwi_write_queue *wq, size_t length)
{
	ufbxwi_check(ufbxwi_write_queue_flush(wq, length), NULL);

	return wq->buffer_pos;
}

static ufbxwi_forceinline char *ufbxwi_queue_write_reserve_small(ufbxwi_write_queue *wq, size_t length)
{
	ufbxwi_dev_assert(length <= 256);

	char *dst = wq->buffer_pos;
	size_t left = ufbxwi_to_size(wq->buffer_end - dst);
	if (left >= length) {
		return wq->buffer_pos;
	} else {
		return ufbxwi_queue_write_reserve_slow(wq, length);
	}
}

static ufbxwi_mutable_void_span ufbxwi_queue_write_reserve_at_least(ufbxwi_write_queue *wq, size_t length)
{
	char *dst = wq->buffer_pos;
	size_t left = ufbxwi_to_size(wq->buffer_end - dst);
	ufbxwi_mutable_void_span span;
	if (left >= length) {
		span.data = wq->buffer_pos;
		span.count = left;
	} else {
		span.data = ufbxwi_queue_write_reserve_slow(wq, length);
		span.count = ufbxwi_to_size(wq->buffer_end - (char*)span.data);
	}
	return span;
}

static ufbxwi_forceinline void ufbxwi_queue_write_commit(ufbxwi_write_queue *wq, size_t length)
{
	ufbxw_assert(length <= ufbxwi_to_size(wq->buffer_end - wq->buffer_pos));
	wq->buffer_pos += length;
}

static ufbxwi_mutable_void_span ufbxwi_queue_write_reserve_at_least_in_chunk(ufbxwi_write_queue *wq, ufbxwi_write_chunk *chunk, size_t length)
{
	if (chunk == NULL) {
		return ufbxwi_queue_write_reserve_at_least(wq, length);
	} else {
		ufbxwi_write_chunk_part *const last_part = chunk->last_part;
		ufbxwi_write_chunk_part *part;
		ufbxwi_mutable_void_span result = { 0 };
		if (last_part == NULL) {
			part = &chunk->data;
		} else {
			// TODO(wq): Pool these?
			part = ufbxwi_alloc_zero(wq->ator, ufbxwi_write_chunk_part, 1);
			ufbxwi_check(part, result);
			last_part->next = part;
		}

		ufbxwi_write_buffer *buffer = ufbxwi_write_queue_alloc_buffer(wq, length);
		ufbxwi_check(buffer, result);
		part->buffer = buffer;
		part->data = buffer->pos;
		part->size = 0;
		chunk->last_part = part;

		result.data = part->data;
		result.count = ufbxwi_to_size(buffer->end - buffer->pos);
		return result;
	}
}

static ufbxwi_forceinline void ufbxwi_queue_write_commit_in_chunk(ufbxwi_write_queue *wq, ufbxwi_write_chunk *chunk, size_t length)
{
	if (chunk == NULL) {
		ufbxwi_queue_write_commit(wq, length);
	} else {
		ufbxwi_write_chunk_part *const part = chunk->last_part;
		part->buffer->pos += length;
		part->size = length;
		chunk->total_size += length;
		ufbxwi_write_queue_return_buffer(wq, part->buffer);
	}
}

static ufbxwi_noinline bool ufbxwi_queue_write_slow(ufbxwi_write_queue *wq, const void *data, size_t length)
{
	if (ufbxwi_is_fatal(wq->error)) return false;

	if (length >= wq->direct_write_size && wq->current_chunk->has_file_offset) {
		// If we are doing a large write and know where we are writing, just write it directly to the file.
		ufbxwi_check(ufbxwi_write_queue_flush(wq, 0), false);

		uint64_t file_offset = wq->current_chunk->file_offset;
		bool write_ok = wq->stream.write_fn(wq->stream.user, file_offset, data, length);
		if (!write_ok) {
			ufbxwi_fail(wq->error, UFBXW_ERROR_WRITE_FAILED, "failed to write to output stream");
			return false;
		}

		wq->chunk_layout_file_offset += length;
		wq->current_chunk->file_offset += length;
	} else {
		// Generic buffered write path
		ufbxwi_mutable_void_span dst = ufbxwi_queue_write_reserve_at_least(wq, length);
		ufbxwi_check(dst.count > 0, false);

		memcpy(dst.data, data, length);
		ufbxwi_queue_write_commit(wq, length);
	}

	return true;
}

static ufbxwi_forceinline void ufbxwi_queue_write(ufbxwi_write_queue *wq, const void *data, size_t length)
{
	char *dst = wq->buffer_pos;
	size_t left = ufbxwi_to_size(wq->buffer_end - dst);
	if (left >= length) {
		memcpy(dst, data, length);
		wq->buffer_pos = dst + length;
	} else {
		ufbxwi_queue_write_slow(wq, data, length);
	}
}

static uint32_t ufbxwi_write_queue_add_reloc(ufbxwi_write_queue *wq, uint32_t patch_offset, uint32_t reference_offset, ufbxwi_write_reloc_type type)
{
	uint32_t id;
	if (wq->free_reloc_ids.count > 0) {
		id = wq->free_reloc_ids.data[--wq->free_reloc_ids.count];
	} else {
		id = (uint32_t)wq->relocs.count;
		ufbxwi_check(ufbxwi_list_push_zero(wq->ator, &wq->relocs, ufbxwi_write_reloc), ~0u);
	}

	const uint32_t offset_in_chunk = (uint32_t)(wq->buffer_pos - wq->buffer_begin);

	ufbxwi_write_chunk *patch_chunk = wq->current_chunk;
	patch_chunk->num_unresolved_relocs++;

	ufbxwi_write_reloc *reloc = &wq->relocs.data[id];
	reloc->patch_chunk = patch_chunk;
	reloc->patch_offset = offset_in_chunk + patch_offset;
	reloc->reference_offset = offset_in_chunk + reference_offset;
	reloc->target_chunk = NULL;
	reloc->target_offset = 0;
	reloc->type = type;
	return id;
}

static void ufbxwi_write_queue_finish_reloc(ufbxwi_write_queue *wq, uint32_t reloc_id, uint32_t target_offset)
{
	if (reloc_id == ~0u) return;

	const uint32_t offset_in_chunk = (uint32_t)(wq->buffer_pos - wq->buffer_begin);

	ufbxwi_write_reloc *reloc = &wq->relocs.data[reloc_id];
	ufbxwi_write_chunk *target_chunk = wq->current_chunk;

	reloc->target_chunk = target_chunk;
	reloc->target_offset = offset_in_chunk + target_offset;

	if (target_chunk->has_file_offset) {
		// Previous chunks must be resolved by now
		ufbxwi_write_queue_resolve_reloc(wq, reloc_id);
	} else {
		ufbxwi_check(ufbxwi_list_push_copy(wq->ator, &target_chunk->pending_relocs, uint32_t, &reloc_id));
	}
}

#endif

// -- Saving

#ifdef UFBXWI_FEATURE_SAVE

typedef struct {
	uint32_t reference_count;
	ufbxw_id template_id;
} ufbxwi_save_object_type;

UFBXWI_LIST_TYPE(ufbxwi_save_object_type_list, ufbxwi_save_object_type);
UFBXWI_LIST_TYPE(ufbxwi_mesh_attribute_ptr_list, ufbxwi_mesh_attribute*);

// TODO(wq): Rename this to something more descriptive
typedef struct {
	uint32_t reloc_end_offset;
} ufbxwi_binary_node_header;

UFBXWI_LIST_TYPE(ufbxwi_binary_node_header_list, ufbxwi_binary_node_header);

typedef struct {
	ufbxwi_allocator *ator;
	ufbxwi_error *error;
	ufbxwi_write_queue *write_queue;
	const ufbxw_save_opts *opts;

	ufbxwi_byte_list tmp_input_buffer;

	ufbxw_deflate_compressor deflate;
	bool has_deflate_compressor;
	bool tried_deflate_compressor;
} ufbxwi_save_thread_context;

typedef struct {
	ufbxwi_allocator ator;

	ufbxwi_error error;
	bool ascii;
	ufbxw_save_opts opts;

	ufbxw_scene *scene;

	ufbxwi_buffer_pool buffers;

	uint32_t depth;

	char *stream_buffer;
	size_t stream_buffer_size;

	ufbxwi_prop_list tmp_prop_list;
	ufbxwi_save_object_type_list object_types;
	ufbxwi_conn_list tmp_conns;

	ufbxwi_mesh_attribute_ptr_list tmp_attributes;
	ufbxwi_binary_node_header_list binary_headers;

	ufbxwi_thread_pool thread_pool;
	ufbxwi_task_queue task_queue;
	ufbxwi_write_queue write_queue;

	ufbxwi_save_thread_context main_thread_ctx;

	// Thread-safe error and alloocator
	ufbxwi_error thread_error;
	ufbxwi_allocator thread_ator;

} ufbxwi_save_context;

// -- Convenience API for writing

#define ufbxwi_write(sc, data, length) ufbxwi_queue_write(&(sc)->write_queue, (data), (length))
#define ufbxwi_write_reserve_small(sc, length) ufbxwi_queue_write_reserve_small(&(sc)->write_queue, (length))
#define ufbxwi_write_reserve_at_least(sc, length) ufbxwi_queue_write_reserve_at_least(&(sc)->write_queue, (length))
#define ufbxwi_write_commit(sc, length) ufbxwi_queue_write_commit(&(sc)->write_queue, (length))

// -- Saving thread context

static void ufbxwi_init_save_thread_context(ufbxwi_save_context *sc, ufbxwi_save_thread_context *tc)
{
	if (sc->task_queue.enabled) {
		tc->ator = &sc->thread_ator;
		tc->error = &sc->thread_error;
	} else {
		tc->ator = &sc->ator;
		tc->error = &sc->error;
	}

	tc->write_queue = &sc->write_queue;
	tc->opts = &sc->opts;
}

static void ufbxwi_destroy_save_thread_context(ufbxwi_save_context *sc, ufbxwi_save_thread_context *tc)
{
	if (tc->has_deflate_compressor) {
		if (tc->deflate.free_fn) {
			tc->deflate.free_fn(tc->deflate.user);
		}
	}
}

static void *ufbxwi_create_save_thread_context(void *user)
{
	ufbxwi_save_context *sc = (ufbxwi_save_context*)user;
	ufbxwi_save_thread_context *tc = ufbxwi_alloc_zero(&sc->thread_ator, ufbxwi_save_thread_context, 1);
	ufbxwi_check(tc, NULL);

	ufbxwi_init_save_thread_context(sc, tc);
	return tc;
}

static void ufbxwi_free_save_thread_context(void *user, void *thread_ctx)
{
	ufbxwi_save_context *sc = (ufbxwi_save_context*)user;
	ufbxwi_save_thread_context *tc = (ufbxwi_save_thread_context*)thread_ctx;
	ufbxwi_destroy_save_thread_context(sc, tc);
}

// -- ASCII

static size_t ufbxwi_default_ascii_format_int(void *user, char *dst, size_t dst_size, const int32_t *src, size_t src_count)
{
	char *d = dst, *end = dst + dst_size;
	for (size_t i = 0; i < src_count; i++) {
		d += snprintf(d, ufbxwi_to_size(end - d), "%d,", (int)src[i]);
	}
	return ufbxwi_to_size(d - dst);
}

static size_t ufbxwi_default_ascii_format_long(void *user, char *dst, size_t dst_size, const int64_t *src, size_t src_count)
{
	char *d = dst, *end = dst + dst_size;
	for (size_t i = 0; i < src_count; i++) {
		d += snprintf(d, ufbxwi_to_size(end - d), "%lld,", (long long)src[i]);
	}
	return ufbxwi_to_size(d - dst);
}

static size_t ufbxwi_default_ascii_format_float(void *user, char *dst, size_t dst_size, const float *src, size_t src_count, ufbxw_ascii_float_format format)
{
	const char *fmt = "";
	switch (format) {
	case UFBXW_ASCII_FLOAT_FORMAT_FIXED_PRECISION: fmt = "%.7g,"; break;
	case UFBXW_ASCII_FLOAT_FORMAT_ROUND_TRIP: fmt = "%.9g,"; break;
	default: return 0;
	}

	char *d = dst, *end = dst + dst_size;
	for (size_t i = 0; i < src_count; i++) {
		d += snprintf(d, ufbxwi_to_size(end - d), fmt, src[i]);
	}
	return ufbxwi_to_size(d - dst);
}

static size_t ufbxwi_default_ascii_format_double(void *user, char *dst, size_t dst_size, const double *src, size_t src_count, ufbxw_ascii_float_format format)
{
	const char *fmt = "";
	switch (format) {
	case UFBXW_ASCII_FLOAT_FORMAT_FIXED_PRECISION: fmt = "%.15g,"; break;
	case UFBXW_ASCII_FLOAT_FORMAT_ROUND_TRIP: fmt = "%.17g,"; break;
	default: return 0;
	}

	char *d = dst, *end = dst + dst_size;
	for (size_t i = 0; i < src_count; i++) {
		d += snprintf(d, ufbxwi_to_size(end - d), fmt, src[i]);
	}
	return ufbxwi_to_size(d - dst);
}

static void ufbxwi_ascii_indent(ufbxwi_save_context *sc)
{
	size_t indent = ufbxwi_min_sz(sc->depth, 64);
	char *dst = ufbxwi_write_reserve_small(sc, indent);
	for (size_t i = 0; i < indent; i++) {
		dst[i] = '\t';
	}
	ufbxwi_write_commit(sc, indent);
}

static void ufbxwi_ascii_comment(ufbxwi_save_context *sc, const char *fmt, va_list args)
{
	// TODO: More rigorous
	// TODO: Sanitize \n's and other special characters
	size_t max_length = 256;
	char *dst = ufbxwi_write_reserve_small(sc, max_length);

	int len = vsnprintf(dst, max_length, fmt, args);
	ufbxw_assert(len >= 0);
	ufbxwi_write_commit(sc, (size_t)len);
}

static void ufbxwi_ascii_dom_string(ufbxwi_save_context *sc, const char *str, size_t length)
{
	ufbxwi_write(sc, "\"", 1);

	const char *end = str + length;
	for (;;) {
		// TODO: Line break escaping, other special characters
		const char *quote = memchr(str, '"', ufbxwi_to_size(end - str));
		if (!quote) {
			ufbxwi_write(sc, str, ufbxwi_to_size(end - str));
			break;
		} else {
			ufbxwi_write(sc, str, ufbxwi_to_size(quote - str));
			ufbxwi_write(sc, "&quot;", 6);
			str = quote + 1;
		}
	}

	ufbxwi_write(sc, "\"", 1);
}

static void ufbxwi_ascii_dom_write(ufbxwi_save_context *sc, const char *tag, const char *fmt, va_list args, bool open)
{
	ufbxwi_ascii_indent(sc);

	ufbxwi_write(sc, tag, strlen(tag));
	ufbxwi_write(sc, ": ", 2);

	for (const char *pf = fmt; *pf; ++pf) {
		char f = *pf;
		if (pf != fmt) {
			if (f == 'C' || f == 'S' || f == 'T') {
				ufbxwi_write(sc, ", ", 2);
			} else {
				ufbxwi_write(sc, ",", 1);
			}
		}

		switch (f) {
		case 'I': {
			char *dst = ufbxwi_write_reserve_small(sc, 128);
			int32_t src[4];
			size_t src_count = 1;
			src[0] = va_arg(args, int32_t);
			for (; pf[1] == 'I' && src_count < 4; src_count++, pf++) {
				src[src_count] = va_arg(args, int32_t);
			}
			size_t len = sc->opts.ascii_formatter.format_int_fn(sc->opts.ascii_formatter.user, dst, 128, src, src_count);
			ufbxwi_write_commit(sc, (size_t)len - 1);
		} break;
		case 'L': {
			char *dst = ufbxwi_write_reserve_small(sc, 128);
			int64_t src[4];
			size_t src_count = 1;
			src[0] = va_arg(args, int64_t);
			for (; pf[1] == 'L' && src_count < 4; src_count++, pf++) {
				src[src_count] = va_arg(args, int64_t);
			}
			size_t len = sc->opts.ascii_formatter.format_long_fn(sc->opts.ascii_formatter.user, dst, 128, src, src_count);
			ufbxwi_write_commit(sc, (size_t)len - 1);
		} break;
		case 'F': {
			char *dst = ufbxwi_write_reserve_small(sc, 128);
			float src[4];
			size_t src_count = 1;
			src[0] = (float)va_arg(args, double);
			for (; pf[1] == 'F' && src_count < 4; src_count++, pf++) {
				src[src_count] = (float)va_arg(args, double);
			}
			size_t len = sc->opts.ascii_formatter.format_float_fn(sc->opts.ascii_formatter.user, dst, 128, src, src_count, sc->opts.ascii_float_format);
			ufbxwi_write_commit(sc, (size_t)len - 1);
		} break;
		case 'D': {
			char *dst = ufbxwi_write_reserve_small(sc, 128);
			double src[4];
			size_t src_count = 1;
			src[0] = va_arg(args, double);
			for (; pf[1] == 'D' && src_count < 4; src_count++, pf++) {
				src[src_count] = va_arg(args, double);
			}
			size_t len = sc->opts.ascii_formatter.format_double_fn(sc->opts.ascii_formatter.user, dst, 128, src, src_count, sc->opts.ascii_float_format);
			ufbxwi_write_commit(sc, (size_t)len - 1);
		} break;
		case 'C': {
			const char *str = va_arg(args, const char*);
			ufbxwi_ascii_dom_string(sc, str, strlen(str));
		} break;
		case 'c': {
			char str = (char)va_arg(args, int);
			ufbxwi_write(sc, &str, 1);
		} break;
		case 'S': {
			ufbxw_string str = va_arg(args, ufbxw_string);
			ufbxwi_ascii_dom_string(sc, str.data, str.length);
		} break;
		case 'T': {
			ufbxwi_token token = va_arg(args, ufbxwi_token);
			ufbxw_string str = sc->scene->string_pool.tokens.data[token];
			ufbxwi_ascii_dom_string(sc, str.data, str.length);
		} break;
		default:
			ufbxwi_unreachable("bad format specifier");
		}
	}

	if (open) {
		ufbxwi_write(sc, " {", 2);
	}
	ufbxwi_write(sc, "\n", 1);
}

#define UFBXWI_ASCII_LINE_MAX_SCALARS 128

typedef union {
	int32_t data_int[UFBXWI_ASCII_LINE_MAX_SCALARS];
	int64_t data_long[UFBXWI_ASCII_LINE_MAX_SCALARS];
	float data_float[UFBXWI_ASCII_LINE_MAX_SCALARS];
	double data_double[UFBXWI_ASCII_LINE_MAX_SCALARS];
} ufbxwi_line_buffer;

static ufbxwi_noinline bool ufbxwi_ascii_write_array_data(ufbxwi_save_thread_context *tc, ufbxwi_write_chunk *chunk, const ufbxwi_buffer_input *input, ufbxwi_buffer_type scalar_type)
{
	ufbxwi_buffer_type type = input->type;
	ufbxwi_buffer_type_info type_info = ufbxwi_buffer_type_infos[type];

	ufbxwi_line_buffer line_temp_buffer;

	size_t line_elems = UFBXWI_ASCII_LINE_MAX_SCALARS / type_info.components;
	size_t line_scalars = line_elems * type_info.components;

	size_t scalar_size = ufbxwi_buffer_type_infos[scalar_type].size;

	size_t scalars_left = input->count * type_info.components;

	ufbxw_ascii_formatter formatter = tc->opts->ascii_formatter;
	ufbxw_ascii_float_format float_format = tc->opts->ascii_float_format;

	char *dst_start = NULL;
	char *dst_pos = NULL;
	char *dst_end = NULL;

	size_t offset = 0;
	while (offset < input->count) {
		size_t max_read_size = (input->count - offset) * type_info.size;
		size_t read_size = ufbxwi_min_sz(max_read_size, line_elems * type_info.size);
		ufbxwi_void_span span = ufbxwi_buffer_input_read(input, &line_temp_buffer, read_size, offset);
		size_t span_scalars = span.count * type_info.components;

		for (size_t begin = 0; begin < span_scalars; ) {
			size_t src_count = ufbxwi_min_sz(span_scalars - begin, line_scalars);
			const void *src = (const char*)span.data + begin * scalar_size;

			size_t dst_size = src_count * 28 + 1;

			if (ufbxwi_to_size(dst_end - dst_pos) < dst_size) {
				if (dst_pos > dst_start) {
					ufbxwi_queue_write_commit_in_chunk(tc->write_queue, chunk, ufbxwi_to_size(dst_pos - dst_start));
				}

				ufbxwi_mutable_void_span dst_span = ufbxwi_queue_write_reserve_at_least_in_chunk(tc->write_queue, chunk, dst_size);
				ufbxwi_check(dst_span.data, false);

				dst_pos = dst_start = (char*)dst_span.data;
				dst_end = dst_start + dst_span.count;
			}

			size_t result_length = 0;
			switch (scalar_type) {
			case UFBXWI_BUFFER_TYPE_INT:
				result_length = formatter.format_int_fn(formatter.user, dst_pos, dst_size, (const int32_t*)src, src_count);
				break;
			case UFBXWI_BUFFER_TYPE_LONG:
				result_length = formatter.format_long_fn(formatter.user, dst_pos, dst_size, (const int64_t*)src, src_count);
				break;
			case UFBXWI_BUFFER_TYPE_FLOAT:
				result_length = formatter.format_float_fn(formatter.user, dst_pos, dst_size, (const float*)src, src_count, float_format);
				break;
			case UFBXWI_BUFFER_TYPE_REAL: // TODO: real=float
				result_length = formatter.format_double_fn(formatter.user, dst_pos, dst_size, (const double*)src, src_count, float_format);
				break;
			default:
				ufbxwi_unreachable("bad scalar type");
			}

			if (result_length == 0 || result_length == SIZE_MAX) {
				ufbxwi_fail(tc->error, UFBXW_ERROR_ASCII_FORMAT, "failed to format ASCII numbers");
				return false;
			}

			scalars_left -= src_count;
			begin += src_count;

			if (scalars_left == 0) {
				dst_pos[result_length - 1] = '\n';
			} else {
				dst_pos[result_length++] = '\n';
			}
			dst_pos += result_length;
		}

		offset += span.count;
	}

	if (dst_pos > dst_start) {
		ufbxwi_queue_write_commit_in_chunk(tc->write_queue, chunk, ufbxwi_to_size(dst_pos - dst_start));
	}

	return true;
}

typedef struct {
	ufbxwi_write_chunk *chunk;
	ufbxwi_buffer_input input;
	ufbxwi_buffer_type scalar_type;
} ufbxwi_ascii_array_task;

static bool ufbxwi_ascii_array_task_fn(void *user, void *thread_ctx)
{
	ufbxwi_save_thread_context *tc = (ufbxwi_save_thread_context*)thread_ctx;
	ufbxwi_ascii_array_task *task = (ufbxwi_ascii_array_task*)user;
	ufbxwi_check(ufbxwi_ascii_write_array_data(tc, task->chunk, &task->input, task->scalar_type), false);
	ufbxwi_free(tc->ator, task);
	return true;
}

static void ufbxwi_ascii_dom_write_array(ufbxwi_save_context *sc, const char *tag, ufbxw_buffer_id buffer_id)
{
	ufbxwi_buffer *buffer = ufbxwi_get_buffer(&sc->buffers, buffer_id);
	if (!buffer) return;

	ufbxwi_ascii_indent(sc);

	ufbxwi_write(sc, tag, strlen(tag));
	ufbxwi_write(sc, ": ", 2);

	ufbxwi_buffer_type type = ufbxwi_buffer_id_type(buffer_id);
	ufbxwi_buffer_type_info type_info = ufbxwi_buffer_type_infos[type];
	size_t scalar_count = buffer->count * type_info.components;

	char prefix[64];
	int prefix_len = snprintf(prefix, sizeof(prefix), "*%zu {\n", scalar_count);
	ufbxwi_write(sc, prefix, (size_t)prefix_len);

	sc->depth++;

	ufbxwi_ascii_indent(sc);
	ufbxwi_write(sc, "a: ", 3);

	ufbxwi_buffer_type scalar_type = type_info.scalar_type;
	if ((type_info.flags & UFBXWI_BUFFER_TYPE_FLAG_ASCII_INT) != 0 && sc->opts.version >= 7200) {
		scalar_type = UFBXWI_BUFFER_TYPE_INT;
	}

	ufbxwi_buffer_input input = ufbxwi_get_buffer_input(&sc->buffers, buffer_id);

	size_t thread_threshold = 0;
	switch (type_info.scalar_type) {
	case UFBXWI_BUFFER_TYPE_INT:
	case UFBXWI_BUFFER_TYPE_LONG:
		thread_threshold = sc->opts.threaded_min_ascii_ints;
		break;
	case UFBXWI_BUFFER_TYPE_REAL:
	case UFBXWI_BUFFER_TYPE_FLOAT:
		thread_threshold = sc->opts.threaded_min_ascii_floats;
		break;
	default:
		ufbxwi_unreachable("bad scalar type");
	}

	if (sc->task_queue.enabled && input.count >= thread_threshold) {
		ufbxwi_write_chunk *chunk = ufbxwi_write_queue_reserve_chunk(&sc->write_queue);
		ufbxwi_check(chunk);

		ufbxwi_ascii_array_task *task = ufbxwi_alloc_zero(&sc->thread_ator, ufbxwi_ascii_array_task, 1);
		ufbxwi_check(task);

		task->chunk = chunk;
		task->input = input;
		task->scalar_type = scalar_type;

		chunk->buffer_id = buffer_id;
		chunk->task = ufbxwi_task_push(&sc->task_queue, &ufbxwi_ascii_array_task_fn, task, &sc->main_thread_ctx);
	} else {
		ufbxwi_ascii_write_array_data(&sc->main_thread_ctx, NULL, &input, scalar_type);
		ufbxwi_free_buffer(&sc->buffers, buffer_id);
	}

	sc->depth--;

	ufbxwi_ascii_indent(sc);
	ufbxwi_write(sc, "}\n", 2);
}

static void ufbxwi_ascii_dom_close(ufbxwi_save_context *sc)
{
	ufbxwi_ascii_indent(sc);
	ufbxwi_write(sc, "}\n", 2);
}

// -- Binary

static void ufbxwi_binary_finish_node(ufbxwi_save_context *sc)
{
	if (sc->binary_headers.count == 0) return;
	ufbxwi_binary_node_header header = sc->binary_headers.data[--sc->binary_headers.count];

	ufbxwi_write_queue_finish_reloc(&sc->write_queue, header.reloc_end_offset, 0);
}

typedef struct {
	uint32_t reloc_end_offset;
	uint32_t reloc_value_size;
} ufbxwi_binary_header_relocs;

static ufbxwi_binary_header_relocs ufbxwi_binary_dom_write_header(ufbxwi_save_context *sc, const char *tag, size_t num_values)
{
	ufbxwi_binary_header_relocs relocs;

	// TODO(endian): Should just write directly here. Currently just to make sure the header fits.
	ufbxwi_ignore(ufbxwi_write_reserve_small(sc, 32));

	uint32_t tag_len = (uint32_t)strlen(tag);
	ufbxw_assert(tag_len <= 255);

	if (sc->opts.version >= 7500) {
		const uint32_t header_end = 24 + 1 + tag_len;
		relocs.reloc_end_offset = ufbxwi_write_queue_add_reloc(&sc->write_queue, 0, 0, UFBXWI_WRITE_RELOC_ABSOLUTE_U64);
		relocs.reloc_value_size = ufbxwi_write_queue_add_reloc(&sc->write_queue, 16, header_end, UFBXWI_WRITE_RELOC_RELATIVE_U64);

		const uint64_t header[] = { 0, (uint64_t)num_values, 0 };
		ufbxwi_write(sc, header, sizeof(header));
	} else {
		const uint32_t header_end = 12 + 1 + tag_len;
		relocs.reloc_end_offset = ufbxwi_write_queue_add_reloc(&sc->write_queue, 0, 0, UFBXWI_WRITE_RELOC_ABSOLUTE_U32);
		relocs.reloc_value_size = ufbxwi_write_queue_add_reloc(&sc->write_queue, 8, header_end, UFBXWI_WRITE_RELOC_RELATIVE_U32);

		const uint32_t header[] = { 0, (uint32_t)num_values, 0 };
		ufbxwi_write(sc, header, sizeof(header));
	}

	uint8_t tag_len8[] = { (uint8_t)tag_len };
	ufbxwi_write(sc, tag_len8, 1);
	ufbxwi_write(sc, tag, tag_len);

	return relocs;
}

static void ufbxwi_binary_dom_write(ufbxwi_save_context *sc, const char *tag, const char *fmt, va_list args, bool open)
{
	ufbxwi_binary_node_header *header = ufbxwi_list_push_zero(&sc->ator, &sc->binary_headers, ufbxwi_binary_node_header);
	ufbxwi_check(header);

	// TODO: Make sure this matches
	size_t num_values = strlen(fmt);
	ufbxwi_binary_header_relocs relocs = ufbxwi_binary_dom_write_header(sc, tag, num_values);
	header->reloc_end_offset = relocs.reloc_end_offset;

	for (const char *pf = fmt; *pf; ++pf) {
		char f = *pf;

		// TODO: Endianness
		switch (f) {
		case 'I': {
			int32_t value = va_arg(args, int32_t);
			ufbxwi_write(sc, "I", 1);
			ufbxwi_write(sc, &value, 4);
		} break;
		case 'L': {
			int64_t value = va_arg(args, int64_t);
			ufbxwi_write(sc, "L", 1);
			ufbxwi_write(sc, &value, 8);
		} break;
		case 'F': {
			float value = (float)va_arg(args, double);
			ufbxwi_write(sc, "F", 1);
			ufbxwi_write(sc, &value, 4);
		} break;
		case 'D': {
			double value = va_arg(args, double);
			ufbxwi_write(sc, "D", 1);
			ufbxwi_write(sc, &value, 8);
		} break;
		case 'C': {
			const char *value = va_arg(args, const char*);
			size_t len = strlen(value);
			uint32_t len_u32 = (uint32_t)len;
			ufbxwi_write(sc, "S", 1);
			ufbxwi_write(sc, &len_u32, 4);
			ufbxwi_write(sc, value, len);
		} break;
		case 'c': {
			char value = (char)va_arg(args, int);
			ufbxwi_write(sc, "C", 1);
			ufbxwi_write(sc, &value, 1);
		} break;
		case 'S': {
			ufbxw_string value = va_arg(args, ufbxw_string);
			uint32_t len_u32 = (uint32_t)value.length;
			ufbxwi_write(sc, "S", 1);
			ufbxwi_write(sc, &len_u32, 4);
			ufbxwi_write(sc, value.data, value.length);
		} break;
		case 'T': {
			ufbxwi_token token = va_arg(args, ufbxwi_token);
			ufbxw_string value = sc->scene->string_pool.tokens.data[token];
			uint32_t len_u32 = (uint32_t)value.length;
			ufbxwi_write(sc, "S", 1);
			ufbxwi_write(sc, &len_u32, 4);
			ufbxwi_write(sc, value.data, value.length);
		} break;
		case 'R': {
			ufbxw_blob value = va_arg(args, ufbxw_blob);
			uint32_t len_u32 = (uint32_t)value.size;
			ufbxwi_write(sc, "R", 1);
			ufbxwi_write(sc, &len_u32, 4);
			ufbxwi_write(sc, value.data, value.size);
		} break;
		default:
			ufbxwi_unreachable("bad format specifier");
		}
	}

	ufbxwi_write_queue_finish_reloc(&sc->write_queue, relocs.reloc_value_size, 0);

	if (!open) {
		ufbxwi_binary_finish_node(sc);
	}
}

static ufbxw_deflate_advance_result ufbxwi_deflate_advance(ufbxwi_save_thread_context *tc, ufbxw_deflate_advance_status *status, void *dst, size_t dst_size, const void *src, size_t src_size, bool is_final)
{
	uint32_t flags = 0;
	if (is_final) {
		flags |= UFBXW_DEFLATE_ADVANCE_FLAG_FINISH;
	}

	ufbxw_deflate_advance_result result = tc->deflate.advance_fn(tc->deflate.user, status, dst, dst_size, src, src_size, flags);
	if (result == UFBXW_DEFLATE_ADVANCE_RESULT_INCOMPLETE && status->bytes_read == 0 && status->bytes_written == 0) {
		flags |= UFBXW_DEFLATE_ADVANCE_FLAG_FLUSH;
		result = tc->deflate.advance_fn(tc->deflate.user, status, dst, dst_size, src, src_size, flags);
	}

	if (result == UFBXW_DEFLATE_ADVANCE_RESULT_ERROR) {
		ufbxwi_fail(tc->error, UFBXW_ERROR_DEFLATE_FAILED, "internal deflate error");
		return UFBXW_DEFLATE_ADVANCE_RESULT_ERROR;
	}

	if (status->bytes_written == 0 && status->bytes_read == 0) {
		ufbxwi_fail(tc->error, UFBXW_ERROR_DEFLATE_FAILED, "streaming deflate failed to make progress");
		return UFBXW_DEFLATE_ADVANCE_RESULT_ERROR;
	}

	if (result == UFBXW_DEFLATE_ADVANCE_RESULT_COMPLETED) {
		// Advance should only return completed status if the stream is finished, and it should have consumed all input.
		ufbxw_assert(is_final);
		ufbxw_assert(status->bytes_read == src_size);
	}

	return result;
}

static bool ufbxwi_deflate_init(ufbxwi_save_thread_context *tc)
{
	if (!tc->tried_deflate_compressor && tc->opts->deflate.create_cb.fn) {
		tc->tried_deflate_compressor = true;

		if (tc->opts->deflate.create_cb.fn(tc->opts->deflate.create_cb.user, &tc->deflate, tc->opts->compression_level)) {
			tc->has_deflate_compressor = true;
		} else {
			ufbxwi_fail(tc->error, UFBXW_ERROR_DEFLATE_FAILED, "failed to initialize deflate");
			return false;
		}
	}
	return true;
}

static bool ufbxwi_deflate_buffer(ufbxwi_save_thread_context *tc, ufbxwi_write_chunk *chunk, const ufbxwi_buffer_input *input)
{
	ufbxwi_check(ufbxwi_deflate_init(tc), false);

	ufbxwi_buffer_type type = input->type;
	ufbxwi_buffer_type_info type_info = ufbxwi_buffer_type_infos[type];

	size_t buffer_count = input->count;
	size_t scalar_count = buffer_count * type_info.components;
	size_t data_size = buffer_count * type_info.size;

	const size_t window_size = tc->opts->deflate_window_size;

	ufbxwi_void_iterator src = { 0 };
	size_t total_read = 0;
	size_t total_written = 0;

	size_t bound_size = tc->deflate.begin_fn(tc->deflate.user, data_size);

	const char *buffer_data = (const char*)input->data;

	size_t dst_size = window_size;
	if (!tc->opts->deflate.streaming_input && !buffer_data) {
		ufbxwi_check(ufbxwi_list_resize_uninit(tc->ator, &tc->tmp_input_buffer, char, data_size), false);
		size_t read_count = ufbxwi_buffer_input_read_to(input, tc->tmp_input_buffer.data, buffer_count, 0);
		if (read_count != buffer_count) {
			ufbxwi_fail(tc->error, UFBXW_ERROR_BUFFER_STREAM, "failed to read buffer data");
			return false;
		}
		buffer_data = tc->tmp_input_buffer.data;
	}
	if (!tc->opts->deflate.streaming_output) {
		dst_size = bound_size;
	}

	if (buffer_data) {
		const char *src = buffer_data;
		const char *src_end = src + data_size;
		for (;;) {
			size_t src_len = ufbxwi_to_size(src_end - src);

			ufbxw_deflate_advance_status status = { 0, 0 };
			ufbxwi_mutable_void_span dst = ufbxwi_queue_write_reserve_at_least_in_chunk(tc->write_queue, chunk, dst_size);
			ufbxw_deflate_advance_result result = ufbxwi_deflate_advance(tc, &status, dst.data, dst.count, src, src_len, true);
			ufbxwi_check(result != UFBXW_DEFLATE_ADVANCE_RESULT_ERROR, false);

			src += status.bytes_read;
			total_written += status.bytes_written;
			ufbxwi_queue_write_commit_in_chunk(tc->write_queue, chunk, status.bytes_written);

			if (result == UFBXW_DEFLATE_ADVANCE_RESULT_COMPLETED) break;
		}
	} else {
		size_t read_offset = 0;
		size_t src_buffer_size = window_size * 2;
		ufbxwi_check(ufbxwi_list_resize_uninit(tc->ator, &tc->tmp_input_buffer, char, src_buffer_size), false);

		char *src = tc->tmp_input_buffer.data;
		size_t src_pos = 0;
		size_t src_len = 0;
		bool at_end = false;
		size_t input_index = 0;

		for (;;) {
			if (!at_end && src_len - src_pos < window_size) {
				size_t elems_read = src_pos / type_info.size;
				size_t bytes_read = elems_read * type_info.size;

				memmove(src + src_pos - bytes_read, src + src_pos, src_len - src_pos);
				src_pos -= bytes_read;
				src_len -= bytes_read;
				input_index += elems_read;

				size_t input_left = buffer_count - input_index;
				size_t input_buffer_left = (src_buffer_size - src_len) / type_info.size;
				size_t to_read = ufbxwi_min_sz(input_left, input_buffer_left);
				size_t num_read = ufbxwi_buffer_input_read_to(input, src + src_len, to_read, input_index);
				if (num_read == 0) {
					ufbxwi_fail(tc->error, UFBXW_ERROR_BUFFER_STREAM, "failed to read buffer data");
					return false;
				}

				src_len += num_read * type_info.size;
				if (input_index + num_read == buffer_count) {
					at_end = true;
				}
			}

			ufbxw_deflate_advance_status status = { 0, 0 };
			ufbxwi_mutable_void_span dst = ufbxwi_queue_write_reserve_at_least_in_chunk(tc->write_queue, chunk, dst_size);
			ufbxw_deflate_advance_result result = ufbxwi_deflate_advance(tc, &status, dst.data, dst.count, src + src_pos, src_len - src_pos, at_end);
			ufbxwi_check(result != UFBXW_DEFLATE_ADVANCE_RESULT_ERROR, false);
			src_pos += status.bytes_read;
			total_written += status.bytes_written;
			ufbxwi_queue_write_commit_in_chunk(tc->write_queue, chunk, status.bytes_written);

			if (result == UFBXW_DEFLATE_ADVANCE_RESULT_COMPLETED) break;
		}
	}

	if (tc->deflate.end_fn) {
		tc->deflate.end_fn(&tc->deflate.user);
	}

	if (total_written > UINT32_MAX) {
		ufbxwi_failf(tc->error, UFBXW_ERROR_ARRAY_TOO_BIG, "compressed array is too big for FBX (%zu bytes, max 2^32 bytes)", total_written);
	}

	return true;
}

typedef struct {
	ufbxwi_write_chunk *chunk;
	ufbxwi_buffer_input input;
} ufbxwi_deflate_task;

static bool ufbxwi_deflate_task_fn(void *user, void *thread_ctx)
{
	ufbxwi_save_thread_context *tc = (ufbxwi_save_thread_context*)thread_ctx;
	ufbxwi_deflate_task *task = (ufbxwi_deflate_task*)user;
	ufbxwi_check(ufbxwi_deflate_buffer(tc, task->chunk, &task->input), false);
	ufbxwi_free(tc->ator, task);
	return true;
}

static void ufbxwi_binary_dom_write_array(ufbxwi_save_context *sc, const char *tag, ufbxw_buffer_id buffer_id)
{
	ufbxwi_buffer *buffer = ufbxwi_get_buffer(&sc->buffers, buffer_id);
	if (!buffer) return;

	const uint32_t num_values = 1;
	ufbxwi_binary_header_relocs relocs = ufbxwi_binary_dom_write_header(sc, tag, num_values);

	ufbxwi_buffer_type type = ufbxwi_buffer_id_type(buffer_id);
	ufbxwi_buffer_type_info type_info = ufbxwi_buffer_type_infos[type];
	
	size_t buffer_count = buffer->count;
	size_t scalar_count = buffer_count * type_info.components;
	size_t data_size = buffer_count * type_info.size;

	char type_char = ' ';
	switch (type_info.scalar_type) {
	case UFBXWI_BUFFER_TYPE_INT: type_char = 'i'; break;
	case UFBXWI_BUFFER_TYPE_LONG: type_char = 'l'; break;
	case UFBXWI_BUFFER_TYPE_REAL: type_char = 'd'; break; // TODO: real=float case?
	case UFBXWI_BUFFER_TYPE_FLOAT: type_char = 'f'; break;
	default:
		ufbxwi_unreachable("bad scalar type");
	}
	ufbxwi_write(sc, &type_char, 1);

	// TODO(endian): Should write directly here. Currently just to make sure it fits in the same chunk.
	ufbxwi_check(ufbxwi_write_reserve_small(sc, 16));

	uint32_t encoding = 0;

	// Don't deflate tiny buffers
	// TODO: Determine a better cutoff point
	if (data_size >= 16 && sc->opts.deflate.create_cb.fn) {
		encoding = 1;
	}

	uint32_t reloc_array_encoded_size = ufbxwi_write_queue_add_reloc(&sc->write_queue, 8, 12, UFBXWI_WRITE_RELOC_RELATIVE_U32);
	const uint32_t array_header[] = { (uint32_t)scalar_count, encoding, 0 };
	ufbxwi_write(sc, array_header, sizeof(array_header));

	if (encoding == 1) {
		ufbxwi_buffer_input input = ufbxwi_get_buffer_input(&sc->buffers, buffer_id);

		if (sc->task_queue.enabled && data_size >= sc->opts.threaded_min_deflate_bytes) {
			ufbxwi_write_chunk *chunk = ufbxwi_write_queue_reserve_chunk(&sc->write_queue);
			ufbxwi_check(chunk);

			ufbxwi_deflate_task *task = ufbxwi_alloc_zero(&sc->thread_ator, ufbxwi_deflate_task, 1);
			ufbxwi_check(task);

			task->chunk = chunk;
			task->input = input;

			chunk->buffer_id = buffer_id;
			chunk->task = ufbxwi_task_push(&sc->task_queue, &ufbxwi_deflate_task_fn, task, &sc->main_thread_ctx);
		} else {
			ufbxwi_deflate_buffer(&sc->main_thread_ctx, NULL, &input);
			ufbxwi_free_buffer(&sc->buffers, buffer_id);
		}
	} else {
		if (data_size > UINT32_MAX) {
			ufbxwi_failf(&sc->error, UFBXW_ERROR_ARRAY_TOO_BIG, "array is too big for FBX (%zu bytes, max 2^32 bytes)", data_size);
		}

		switch (buffer->state) {
		case UFBXWI_BUFFER_STATE_NONE:
			ufbxwi_unreachable("bad buffer");
			break;
		case UFBXWI_BUFFER_STATE_OWNED:
			ufbxwi_write(sc, buffer->data.owned.data, data_size);
			break;
		case UFBXWI_BUFFER_STATE_EXTERNAL:
			ufbxwi_write(sc, buffer->data.external.data, data_size);
			break;
		case UFBXWI_BUFFER_STATE_STREAM: {
			size_t offset = 0;
			// Read through a temporary buffer so that we can stream to aligned memory
			if (!sc->stream_buffer) {
				size_t stream_buffer_size = 4096;
				sc->stream_buffer = ufbxwi_alloc(&sc->ator, char, stream_buffer_size);
				ufbxwi_check(sc->stream_buffer);
				sc->stream_buffer_size = stream_buffer_size;
			}

			size_t max_elements = sc->stream_buffer_size / type_info.size;
			while (!ufbxwi_is_fatal(&sc->error) && offset < buffer_count) {
				size_t to_read = ufbxwi_min_sz(max_elements, buffer_count - offset);
				size_t num_read = ufbxwi_buffer_read_to(&sc->buffers, buffer_id, sc->stream_buffer, to_read, offset);
				ufbxwi_check(num_read == to_read);
				ufbxwi_write(sc, sc->stream_buffer, num_read * type_info.size);

				offset += num_read;
			}

		} break;
		}

		ufbxwi_free_buffer(&sc->buffers, buffer_id);
	}

	ufbxwi_write_queue_finish_reloc(&sc->write_queue, relocs.reloc_value_size, 0);
	ufbxwi_write_queue_finish_reloc(&sc->write_queue, relocs.reloc_end_offset, 0);
	ufbxwi_write_queue_finish_reloc(&sc->write_queue, reloc_array_encoded_size, 0);
}

// TODO: Allocate this dynamically?
static const char ufbxwi_binary_zero_buf[32] = { 0 };

static void ufbxwi_binary_dom_close(ufbxwi_save_context *sc)
{
	size_t null_size = sc->opts.version >= 7500 ? 25 : 13;
	ufbxwi_write(sc, ufbxwi_binary_zero_buf, null_size);

	ufbxwi_binary_finish_node(sc);
}

static void ufbxwi_binary_write_header(ufbxwi_save_context *sc)
{
	// TODO: Big endian??
	ufbxwi_write(sc, "Kaydara FBX Binary  \x00\x1a\x00", 23);

	uint32_t version = sc->opts.version;
	ufbxwi_write(sc, &version, 4);
}

// Derived from actual file hashes. See `misc/fbx_hash_solver.c` for details.
// The hash structure is based on work by @hamish-milne: https://github.com/hamish-milne/FbxWriter
static const char ufbxwi_binary_magic_order[] = "\x11\x12\x05\x06\x0b\x0c\x08\x09\x14\x15\x00\x01\x02\x03\x0e\x0f";
static const char ufbxwi_binary_fileid_key[] = "\x18\xb3\x1a\xea\x86\x24\xfc\xc3\x8e\xc9\x80\x23\x97\x25\xc2\xff";
static const char ufbxwi_binary_footer_key[] = "\xfa\xbc\x9b\x09\xd0\xc9\xe4\x67\xb1\x76\xca\x82\x1d\xff\x19\x78";
static const char ufbxwi_binary_footer_data[] = "\xf8\x5a\x8c\x6a\xde\xf5\xd9\x7e\xec\xe9\x0c\xe3\x75\x8f\x29\x0b";

static void ufbxwi_binary_fileid_magic(char magic[16], const char *creation_time)
{
	ufbxwi_nounroll for (size_t i = 0; i < 16; i++) magic[i] = creation_time[ufbxwi_binary_magic_order[i]];
	ufbxwi_nounroll for (size_t i = 1; i < 16; i++) magic[i] ^= magic[i - 1];
	ufbxwi_nounroll for (size_t i = 0; i < 16; i++) magic[i] ^= ufbxwi_binary_fileid_key[i];
}

static void ufbxwi_binary_footer_magic(char magic[16], const char *creation_time)
{
	ufbxwi_nounroll for (size_t i = 0; i < 16; i++) magic[i] = creation_time[ufbxwi_binary_magic_order[i]];
	ufbxwi_nounroll for (size_t i = 2; i < 16; i++) magic[i] ^= magic[i - 2];
	ufbxwi_nounroll for (size_t i = 0; i < 16; i++) magic[i] ^= creation_time[ufbxwi_binary_magic_order[i]];
	ufbxwi_nounroll for (size_t i = 1; i < 16; i++) magic[i] ^= magic[i - 1];
	ufbxwi_nounroll for (size_t i = 0; i < 16; i++) magic[i] ^= ufbxwi_binary_footer_key[i];
}

static void ufbxwi_binary_write_footer(ufbxwi_save_context *sc, const char *creation_time)
{
	// One more null record
	size_t null_size = sc->opts.version >= 7500 ? 25 : 13;
	ufbxwi_write(sc, ufbxwi_binary_zero_buf, null_size);

	char footer_magic[16];
	ufbxwi_binary_footer_magic(footer_magic, creation_time);
	ufbxwi_write(sc, footer_magic, sizeof(footer_magic));

	// Calculate the file offset, we must know the file offset at this point.
	ufbxwi_write_queue_layout_chunks(&sc->write_queue, true);
	const uint32_t offset_in_chunk = (uint32_t)(sc->write_queue.buffer_pos - sc->write_queue.buffer_begin);
	const uint64_t file_offset = sc->write_queue.chunk_layout_file_offset + offset_in_chunk;

	// Align to 16 bytes, always insert at least a single zero
	size_t align = (size_t)(16 - (file_offset % 16));

	ufbxwi_write(sc, ufbxwi_binary_zero_buf, align);

	uint32_t footer[36];
	memset(footer, 0, sizeof(footer));
	footer[0] = 0; // TODO: This has some unknown value in <7000 files
	footer[1] = sc->opts.version;

	ufbxwi_write(sc, footer, sizeof(footer));

	// Fixed footer
	ufbxwi_write(sc, ufbxwi_binary_footer_data, 16);
}

// -- DOM

static const char *ufbxwi_dom_section_str = "------------------------------------------------------------------";

static void ufbxwi_dom_comment(ufbxwi_save_context *sc, const char *fmt, ...)
{
	if (sc->ascii) {
		va_list args;
		va_start(args, fmt);
		ufbxwi_ascii_comment(sc, fmt, args);
		va_end(args);
	}
}

static void ufbxwi_dom_section(ufbxwi_save_context *sc, const char *name)
{
	if (sc->ascii) {
		ufbxwi_dom_comment(sc, "\n; %s\n;%.66s\n\n", name, ufbxwi_dom_section_str);
	}
}

static void ufbxwi_dom_open(ufbxwi_save_context *sc, const char *tag, const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);

	if (sc->ascii) {
		ufbxwi_ascii_dom_write(sc, tag, fmt, args, true);
	} else {
		ufbxwi_binary_dom_write(sc, tag, fmt, args, true);
	}

	sc->depth++;

	va_end(args);
}

static void ufbxwi_dom_close(ufbxwi_save_context *sc)
{
	sc->depth--;

	if (sc->ascii) {
		ufbxwi_ascii_dom_close(sc);
	} else {
		ufbxwi_binary_dom_close(sc);
	}
}

static void ufbxwi_dom_value(ufbxwi_save_context *sc, const char *tag, const char *fmt, ...)
{
	va_list args;
	va_start(args, fmt);

	if (sc->ascii) {
		ufbxwi_ascii_dom_write(sc, tag, fmt, args, false);
	} else {
		ufbxwi_binary_dom_write(sc, tag, fmt, args, false);
	}

	va_end(args);
}

static void ufbxwi_dom_array(ufbxwi_save_context *sc, const char *tag, ufbxw_buffer_id buffer)
{
	if (sc->ascii) {
		ufbxwi_ascii_dom_write_array(sc, tag, buffer);
	} else {
		ufbxwi_binary_dom_write_array(sc, tag, buffer);
	}
}

enum {
	UFBXWI_SAVE_ELEMENT_MANUAL_OPEN = 0x1,
	UFBXWI_SAVE_ELEMENT_NO_ID = 0x2,
};

static int ufbxwi_cmp_prop_identity(const void *va, const void *vb)
{
	const ufbxwi_prop *a = (const ufbxwi_prop*)va, *b = (const ufbxwi_prop*)vb;
	if (a->token != b->token) return a->token < b->token ? -1 : +1;
	uint32_t a_temp = a->flags & UFBXWI_PROP_FLAG_TEMP_MASK;
	uint32_t b_temp = b->flags & UFBXWI_PROP_FLAG_TEMP_MASK;
	if (a_temp != b_temp) return a_temp < b_temp ? -1 : +1;
	return 0;
}

static int ufbxwi_cmp_prop_order(const void *va, const void *vb)
{
	const ufbxwi_prop *a = (const ufbxwi_prop*)va, *b = (const ufbxwi_prop*)vb;
	if (a->order != b->order) return a->order < b->order ? -1 : +1;
	return 0;
}

static void ufbxwi_save_props(ufbxwi_save_context *sc, const ufbxwi_element *element, const ufbxwi_props *default_props, const ufbxwi_element *tmpl)
{
	ufbxw_scene *scene = sc->scene;

	// Merge all the properties into one sorted list
	size_t prop_count = element->props.count;
	if (default_props) prop_count += default_props->count;
	if (tmpl) prop_count += tmpl->props.count;

	ufbxwi_check(ufbxwi_list_resize_uninit(&sc->ator, &sc->tmp_prop_list, ufbxwi_prop, prop_count));

	ufbxwi_prop *props = sc->tmp_prop_list.data;

	{
		ufbxwi_prop *dst_prop = props;

		ufbxwi_for(const ufbxwi_prop, prop, element->props.props, element->props.capacity) {
			if (prop->token == 0 || prop->token == ~0u) continue;
			*dst_prop = *prop;
			dst_prop++;
		}

		if (default_props) {
			ufbxwi_for(const ufbxwi_prop, prop, default_props->props, default_props->capacity) {
				if (prop->token == 0 || prop->token == ~0u) continue;
				*dst_prop = *prop;
				dst_prop->flags |= UFBXWI_PROP_FLAG_TEMP_DEFAULT;
				dst_prop++;
			}
		}

		if (tmpl) {
			ufbxwi_for(const ufbxwi_prop, prop, tmpl->props.props, tmpl->props.capacity) {
				if (prop->token == 0 || prop->token == ~0u) continue;
				*dst_prop = *prop;
				dst_prop->flags |= UFBXWI_PROP_FLAG_TEMP_TEMPLATE;
				dst_prop++;
			}
		}

		ufbxw_assert(dst_prop - props == prop_count);
	}

	// TODO: Better sort here
	qsort(sc->tmp_prop_list.data, prop_count, sizeof(ufbxwi_prop), &ufbxwi_cmp_prop_identity);

	{
		// Resolve properties
		size_t dst_ix = 0;
		for (size_t src_ix = 0; src_ix < prop_count; src_ix++) {
			const ufbxwi_prop *base = &props[src_ix], *tmpl_prop = NULL;
			ufbxwi_token token = base->token;

			// Skip properties with the same token, preferring the first one and look for a template
			while (src_ix + 1 < prop_count && props[src_ix + 1].token == token) {
				if ((props[src_ix + 1].flags & UFBXWI_PROP_FLAG_TEMP_MASK) == UFBXWI_PROP_FLAG_TEMP_TEMPLATE) {
					tmpl_prop = &props[src_ix + 1];
				}
				src_ix++;
			}

			// Skip writing if the value matches the template
			// TODO: Force saving properties that are animated
			if (tmpl_prop != NULL && tmpl_prop->type == base->type) {
				const void *base_data = ufbxwi_resolve_prop_value(element, base->value);
				const void *tmpl_data = ufbxwi_resolve_prop_value(tmpl, tmpl_prop->value);
				size_t data_size = ufbxwi_prop_data_infos[scene->prop_types.data[base->type].data_type].size;
				if (data_size == 0 || !memcmp(base_data, tmpl_data, data_size)) {
					continue;
				}
			}

			props[dst_ix++] = *base;
		}
		prop_count = dst_ix;
	}

	if (prop_count == 0) {
		return;
	}

	qsort(sc->tmp_prop_list.data, prop_count, sizeof(ufbxwi_prop), &ufbxwi_cmp_prop_order);

	// TODO: Version 6
	ufbxwi_dom_open(sc, "Properties70", "");

	ufbxwi_token prev_prop = UFBXWI_TOKEN_NONE;
	for (size_t i = 0; i < prop_count; i++) {
		const ufbxwi_prop *p = &sc->tmp_prop_list.data[i];
		if (p->token == prev_prop) continue;
		prev_prop = p->token;

		ufbxw_string name = scene->string_pool.tokens.data[p->token];

		const ufbxwi_prop_type *type = &scene->prop_types.data[p->type];
		const void *data = ufbxwi_resolve_prop_value(element, p->value);

		// TODO: Proper flags
		char flags[16];
		char *flag = flags;
		if (p->flags & UFBXW_PROP_FLAG_ANIMATABLE) *flag++ = 'A';
		*flag++ = '\0';
		ufbxw_assert(flag <= flags + sizeof(flags));

		switch (type->data_type) {
			case UFBXW_PROP_DATA_NONE: {
				ufbxwi_dom_value(sc, "P", "SSSC", name, type->type, type->sub_type, flags);
			} break;
			case UFBXW_PROP_DATA_BOOL: {
				const bool *d = (const bool*)data;
				ufbxwi_dom_value(sc, "P", "SSSCI", name, type->type, type->sub_type, flags, *d ? 1 : 0);
			} break;
			case UFBXW_PROP_DATA_INT32: {
				const int32_t *d = (const int32_t*)data;
				ufbxwi_dom_value(sc, "P", "SSSCI", name, type->type, type->sub_type, flags, *d);
			} break;
			case UFBXW_PROP_DATA_INT64: {
				const int64_t *d = (const int64_t*)data;
				ufbxwi_dom_value(sc, "P", "SSSCL", name, type->type, type->sub_type, flags, *d);
			} break;
			case UFBXW_PROP_DATA_REAL: {
				const ufbxw_real *d = (const ufbxw_real*)data;
				ufbxwi_dom_value(sc, "P", "SSSCD", name, type->type, type->sub_type, flags, *d);
			} break;
			case UFBXW_PROP_DATA_VEC2: {
				const ufbxw_vec2 *d = (const ufbxw_vec2*)data;
				ufbxwi_dom_value(sc, "P", "SSSCDD", name, type->type, type->sub_type, flags, d->x, d->y);
			} break;
			case UFBXW_PROP_DATA_VEC3: {
				const ufbxw_vec3 *d = (const ufbxw_vec3*)data;
				ufbxwi_dom_value(sc, "P", "SSSCDDD", name, type->type, type->sub_type, flags, d->x, d->y, d->z);
			} break;
			case UFBXW_PROP_DATA_VEC4: {
				const ufbxw_vec4 *d = (const ufbxw_vec4*)data;
				ufbxwi_dom_value(sc, "P", "SSSCDDDD", name, type->type, type->sub_type, flags, d->x, d->y, d->z, d->w);
			} break;
			case UFBXW_PROP_DATA_STRING: {
				const ufbxw_string *d = (const ufbxw_string*)data;
				ufbxwi_dom_value(sc, "P", "SSSCS", name, type->type, type->sub_type, flags, *d);
			} break;
			case UFBXW_PROP_DATA_ID: {
				ufbxwi_dom_value(sc, "P", "SSSC", name, type->type, type->sub_type, flags);
			} break;
			case UFBXW_PROP_DATA_REAL_STRING: {
				const ufbxw_real_string *d = (const ufbxw_real_string*)data;
				ufbxwi_dom_value(sc, "P", "SSSCDS", name, type->type, type->sub_type, flags, d->value, d->string);
			} break;
			case UFBXW_PROP_DATA_BLOB: {
				const ufbxw_blob *d = (const ufbxw_blob*)data;
				ufbxw_assert(0 && "TODO");
			} break;
			case UFBXW_PROP_DATA_USER_INT: {
				const ufbxw_user_int *d = (const ufbxw_user_int*)data;
				ufbxwi_dom_value(sc, "P", "SSSCIII", name, type->type, type->sub_type, flags, d->value, d->min_value, d->max_value);
			} break;
			case UFBXW_PROP_DATA_USER_REAL: {
				const ufbxw_user_real *d = (const ufbxw_user_real*)data;
				ufbxwi_dom_value(sc, "P", "SSSCDDD", name, type->type, type->sub_type, flags, d->value, d->min_value, d->max_value);
			} break;
			case UFBXW_PROP_DATA_USER_ENUM: {
				const ufbxw_user_enum *d = (const ufbxw_user_enum*)data;
				ufbxwi_dom_value(sc, "P", "SSSCIS", name, type->type, type->sub_type, flags, d->value, d->options);
			} break;
			default:
				ufbxwi_unreachable("bad data type");
		}
	}

	ufbxwi_dom_close(sc);
}

static void ufbxwi_save_template(ufbxwi_save_context *sc, ufbxwi_element *element, uint32_t flags)
{
	ufbxwi_template *tmpl = (ufbxwi_template*)element;

	ufbxwi_dom_open(sc, "PropertyTemplate", "T", tmpl->type);

	ufbxwi_save_props(sc, &tmpl->element, NULL, NULL);

	ufbxwi_dom_close(sc);
}

static bool ufbxwi_less_mesh_attribute_ptr_order(void *user, const void *va, const void *vb)
{
	const ufbxwi_mesh_attribute *a = *(const ufbxwi_mesh_attribute**)va, *b = *(const ufbxwi_mesh_attribute**)vb;
	(void)user;

	int32_t a_order = ufbxwi_mesh_attribute_infos[a->attribute].order;
	int32_t b_order = ufbxwi_mesh_attribute_infos[b->attribute].order;
	if (a_order != b_order) return a_order < b_order;
	if (a->attribute != b->attribute) return a->attribute < b->attribute;
	return a->set < b->set;
}

static bool ufbxwi_less_mesh_attribute_ptr_set(void *user, const void *va, const void *vb)
{
	const ufbxwi_mesh_attribute *a = *(const ufbxwi_mesh_attribute**)va, *b = *(const ufbxwi_mesh_attribute**)vb;
	(void)user;

	if (a->set != b->set) return a->set < b->set;

	int32_t a_order = ufbxwi_mesh_attribute_infos[a->attribute].order;
	int32_t b_order = ufbxwi_mesh_attribute_infos[b->attribute].order;
	if (a_order != b_order) return a_order < b_order;
	return a->attribute < b->attribute;
}

typedef struct {
	ufbxwi_buffer_input indices;
	ufbxwi_buffer_input face_offsets;

	uint32_t face_offset_tmp_pos;
	uint32_t face_offset_tmp_length;
	uint32_t face_offset_buffer_offset;

	int32_t face_offset_tmp[64];
} ufbxwi_polygon_vertex_index_stream;

static void ufbxwi_deleter_free(void *user, void *data)
{
	ufbxwi_allocator *ator = (ufbxwi_allocator*)user;
	ufbxwi_free(ator, data);
}

static size_t ufbxwi_stream_polygon_vertex_index(void *user, int32_t *dst, size_t dst_size, size_t offset)
{
	ufbxwi_polygon_vertex_index_stream *stream = (ufbxwi_polygon_vertex_index_stream*)user;

	size_t num_read = ufbxwi_buffer_input_read_to(&stream->indices, dst, dst_size, offset);
	int32_t min_index = (int32_t)offset;
	int32_t max_index = (int32_t)(offset + num_read);

	for (;;) {
		if (stream->face_offset_tmp_pos == stream->face_offset_tmp_length) {
			const size_t left = stream->face_offsets.count - stream->face_offset_buffer_offset;
			if (left == 0) break;

			const size_t to_read = ufbxwi_min_sz(left, ufbxwi_arraycount(stream->face_offset_tmp));
			size_t read_count = ufbxwi_buffer_input_read_to(&stream->face_offsets, &stream->face_offset_tmp, to_read, stream->face_offset_buffer_offset);
			// TODO: Failure
			ufbxw_assert(read_count == to_read);

			stream->face_offset_buffer_offset += read_count;
			stream->face_offset_tmp_pos = 0;
			stream->face_offset_tmp_length = read_count;
		}

		int32_t ix = stream->face_offset_tmp[stream->face_offset_tmp_pos] - 1;
		if (ix >= max_index) break;
		stream->face_offset_tmp_pos++;

		if (ix >= 0) {
			dst[ix - min_index] = ~dst[ix - min_index];
		}
	}

	return num_read;
}

static ufbxw_buffer_id ufbxwi_create_polygon_vertex_index_stream(ufbxwi_buffer_pool *buffers, ufbxw_int_buffer indices, ufbxw_int_buffer face_offsets)
{
	ufbxwi_polygon_vertex_index_stream *stream = ufbxwi_alloc_zero(buffers->ator, ufbxwi_polygon_vertex_index_stream, 1);
	ufbxwi_check(stream, 0);

	stream->indices = ufbxwi_get_buffer_input(buffers, indices.id);
	stream->face_offsets = ufbxwi_get_buffer_input(buffers, face_offsets.id);

	size_t index_count = stream->indices.count;

	ufbxwi_stream_fn stream_fn;
	stream_fn.int_fn = &ufbxwi_stream_polygon_vertex_index;
	ufbxw_buffer_id index_buffer = ufbxwi_create_stream_buffer(buffers, UFBXWI_BUFFER_TYPE_INT, stream_fn, stream, index_count);
	ufbxwi_check(index_buffer, 0);
	ufbxwi_buffer_set_deleter(buffers, index_buffer, &ufbxwi_deleter_free, buffers->ator);

	return index_buffer;
}

static void ufbxwi_save_mesh_data(ufbxwi_save_context *sc, ufbxwi_element *element)
{
	ufbxwi_mesh *mesh = (ufbxwi_mesh*)element;

	ufbxwi_dom_array(sc, "Vertices", mesh->vertices.id);

	if (mesh->polygon_vertex_index.id != 0) {
		ufbxwi_dom_array(sc, "PolygonVertexIndex", mesh->polygon_vertex_index.id);
	} else {
		ufbxw_buffer_id index_buffer = ufbxwi_create_polygon_vertex_index_stream(&sc->buffers, mesh->vertex_indices, mesh->face_offsets);
		ufbxwi_check(index_buffer);
		ufbxwi_dom_array(sc, "PolygonVertexIndex", index_buffer);
	}

	ufbxwi_dom_array(sc, "Edges", mesh->edges.id);

	ufbxwi_dom_value(sc, "GeometryVersion", "I", 124);

	ufbxwi_check(ufbxwi_list_resize_uninit(&sc->ator, &sc->tmp_attributes, ufbxwi_mesh_attribute*, mesh->attributes.count));
	for (size_t i = 0; i < mesh->attributes.count; i++) {
		sc->tmp_attributes.data[i] = &mesh->attributes.data[i];
	}

	ufbxwi_unstable_sort(sc->tmp_attributes.data, sc->tmp_attributes.count, sizeof(ufbxwi_mesh_attribute*), &ufbxwi_less_mesh_attribute_ptr_order, NULL);

	ufbxwi_for_ptr_list(ufbxwi_mesh_attribute, p_attrib, sc->tmp_attributes) {
		ufbxwi_mesh_attribute *attrib = *p_attrib;
		const ufbxwi_mesh_attribute_info *info = &ufbxwi_mesh_attribute_infos[attrib->attribute];

		ufbxwi_dom_open(sc, info->layer_element_name, "I", attrib->set);

		ufbxwi_dom_value(sc, "Version", "I", info->version);
		ufbxwi_dom_value(sc, "Name", "S", &attrib->name);
		ufbxwi_dom_value(sc, "MappingInformationType", "C", ufbxwi_attribute_mapping_infos[attrib->mapping].name);
		ufbxwi_dom_value(sc, "ReferenceInformationType", "C", attrib->indices ? "IndexToDirect" : "Direct");

		if (attrib->values) {
			ufbxwi_dom_array(sc, info->values_name, attrib->values);
		}
		if (attrib->values_w) {
			// TODO: Generic `sc` format function
			char values_w_name[128];
			snprintf(values_w_name, sizeof(values_w_name), "%sW", info->values_name);
			ufbxwi_dom_array(sc, values_w_name, attrib->values_w);
		}
		if (attrib->indices) {
			ufbxwi_dom_array(sc, info->indices_name, attrib->indices);
		}

		ufbxwi_dom_close(sc);
	}

	ufbxwi_unstable_sort(sc->tmp_attributes.data, sc->tmp_attributes.count, sizeof(ufbxwi_mesh_attribute*), &ufbxwi_less_mesh_attribute_ptr_set, NULL);

	int32_t prev_set = INT32_MIN;
	ufbxwi_for_ptr_list(ufbxwi_mesh_attribute, p_attrib, sc->tmp_attributes) {
		ufbxwi_mesh_attribute *attrib = *p_attrib;
		const ufbxwi_mesh_attribute_info *info = &ufbxwi_mesh_attribute_infos[attrib->attribute];

		if (attrib->set != prev_set) {
			if (prev_set > INT32_MIN) ufbxwi_dom_close(sc);

			ufbxwi_dom_open(sc, "Layer", "I", attrib->set);
			ufbxwi_dom_value(sc, "Version", "I", 100);
			prev_set = attrib->set;
		}

		ufbxwi_dom_open(sc, "LayerElement", "");
		ufbxwi_dom_value(sc, "Type", "C", info->layer_element_name);
		ufbxwi_dom_value(sc, "TypedIndex", "I", prev_set);
		ufbxwi_dom_close(sc);
	}
	if (prev_set > INT32_MIN) ufbxwi_dom_close(sc);
}

static void ufbxwi_save_matrix(ufbxwi_save_context *sc, const char *tag, const ufbxw_matrix *matrix)
{
	ufbxw_buffer_id buf = ufbxwi_create_external_buffer(&sc->buffers, UFBXWI_BUFFER_TYPE_REAL, &matrix->m, 16, 0);
	ufbxwi_dom_array(sc, tag, buf);
}

static void ufbxwi_save_skin_cluster(ufbxwi_save_context *sc, ufbxwi_skin_cluster *cluster)
{
	ufbxwi_dom_value(sc, "Version", "I", 100);
	ufbxwi_dom_value(sc, "UserData", "CC", "", "");
	ufbxwi_dom_array(sc, "Indexes", cluster->indices.id);
	ufbxwi_dom_array(sc, "Weights", cluster->weights.id);
	ufbxwi_save_matrix(sc, "Transform", &cluster->transform);
	ufbxwi_save_matrix(sc, "TransformLink", &cluster->transform_link);
}

static size_t ufbxwi_stream_full_weights(void *user, ufbxw_real *dst, size_t dst_size, size_t offset)
{
	const ufbxwi_blend_shape_conn *conns = (const ufbxwi_blend_shape_conn*)user;
	for (size_t i = 0; i < dst_size; i++) {
		dst[i] = conns[offset + i].target_weight;
	}
	return dst_size;
}

static void ufbxwi_save_blend_channel(ufbxwi_save_context *sc, ufbxwi_blend_channel *channel)
{
	ufbxwi_dom_value(sc, "DeformPercent", "D", channel->deform_percent);

	ufbxwi_stream_fn full_weights_fn;
	full_weights_fn.real_fn = &ufbxwi_stream_full_weights;
	ufbxw_buffer_id full_weights_buffer = ufbxwi_create_stream_buffer(&sc->buffers, UFBXWI_BUFFER_TYPE_REAL, full_weights_fn, channel->blend_shapes.data, channel->blend_shapes.count);
	ufbxwi_dom_array(sc, "FullWeights", full_weights_buffer);
}

static void ufbxwi_save_blend_shape(ufbxwi_save_context *sc, ufbxwi_blend_shape *shape)
{
	ufbxwi_dom_value(sc, "Version", "I", 100);
	ufbxwi_dom_array(sc, "Indexes", shape->indices.id);
	ufbxwi_dom_array(sc, "Vertices", shape->vertices.id);
	ufbxwi_dom_array(sc, "Normals", shape->normals.id);
}

static void ufbxwi_save_bind_pose(ufbxwi_save_context *sc, ufbxwi_bind_pose *pose)
{
	ufbxw_scene *scene = sc->scene;

	ufbxwi_dom_value(sc, "Type", "C", "BindPose");
	ufbxwi_dom_value(sc, "Version", "I", 100);

	size_t num_nodes = 0;
	ufbxwi_for_list(ufbxwi_pose_node, pose_node, pose->pose_nodes) {
		if (!ufbxwi_get_node(scene, pose_node->node)) continue;
		num_nodes++;
	}

	ufbxwi_dom_value(sc, "NbPoseNodes", "I", (int32_t)num_nodes);

	ufbxwi_for_list(ufbxwi_pose_node, pose_node, pose->pose_nodes) {
		ufbxwi_node *node = ufbxwi_get_node(scene, pose_node->node);
		if (!node) continue;

		ufbxwi_dom_open(sc, "PoseNode", "");

		// TODO: fbx_id lookup here
		ufbxwi_dom_value(sc, "Node", "L", pose_node->node.id);
		ufbxwi_save_matrix(sc, "Matrix", &pose_node->matrix);

		ufbxwi_dom_close(sc);
	}
}

static uint32_t ufbxwi_pack_weight(float weight)
{
	if (!(weight >= 0.0f)) weight = 0.0f;
	if (weight > 1.0f) weight = 1.0f;
	return (uint32_t)(weight * 10000.0f);
}

typedef enum {
	UFBXWI_KEY_INTERPOLATION_CONSTANT = 0x2,
	UFBXWI_KEY_INTERPOLATION_LINEAR = 0x4,
	UFBXWI_KEY_INTERPOLATION_CUBIC = 0x8,
	UFBXWI_KEY_TANGENT_AUTO = 0x100,
	UFBXWI_KEY_TANGENT_TCB = 0x200,
	UFBXWI_KEY_TANGENT_USER = 0x400,
	UFBXWI_KEY_TANGENT_BROKEN = 0x800,
	UFBXWI_KEY_CONSTANT_NEXT = 0x100,
	UFBXWI_KEY_CLAMP = 0x1000,
	UFBXWI_KEY_TIME_INDEPENDENT = 0x2000,
	UFBXWI_KEY_CLAMP_PROGRESSIVE = 0x4000,
	UFBXWI_KEY_WEIGHTED_RIGHT = 0x1000000,
	UFBXWI_KEY_WEIGHTED_NEXT_LEFT = 0x2000000,
	UFBXWI_KEY_VELOCITY_RIGHT = 0x10000000,
	UFBXWI_KEY_VELOCITY_NEXT_LEFT = 0x20000000,
} ufbxwi_key_flags;

static ufbxwi_forceinline uint32_t ufbxwi_get_key_flags(uint32_t prev_flags, uint32_t next_flags)
{
	uint32_t flags = 0;
	if (prev_flags & UFBXW_KEYFRAME_INTERPOLATION_CONSTANT) {
		flags |= UFBXWI_KEY_INTERPOLATION_CONSTANT;
	} else if (prev_flags & UFBXW_KEYFRAME_INTERPOLATION_CONSTANT_NEXT) {
		flags |= UFBXWI_KEY_INTERPOLATION_CONSTANT | UFBXWI_KEY_CONSTANT_NEXT;
	} else if (prev_flags & UFBXW_KEYFRAME_INTERPOLATION_LINEAR) {
		flags |= UFBXWI_KEY_INTERPOLATION_LINEAR;
	} else if (prev_flags & UFBXW_KEYFRAME_INTERPOLATION_CUBIC) {
		flags |= UFBXWI_KEY_INTERPOLATION_CUBIC;
	}

	if (prev_flags & UFBXW_KEYFRAME_TANGENT_AUTO) {
		flags |= UFBXWI_KEY_TANGENT_AUTO | UFBXWI_KEY_TIME_INDEPENDENT | UFBXWI_KEY_CLAMP_PROGRESSIVE;
	} else if (prev_flags & UFBXW_KEYFRAME_TANGENT_AUTO_UNCLAMPED) {
		flags |= UFBXWI_KEY_TANGENT_AUTO | UFBXWI_KEY_TIME_INDEPENDENT;
	} else if (prev_flags & UFBXW_KEYFRAME_TANGENT_AUTO_LEGACY) {
		flags |= UFBXWI_KEY_TANGENT_AUTO;
	} else if (prev_flags & UFBXW_KEYFRAME_TANGENT_AUTO_LEGACY_CLAMPED) {
		flags |= UFBXWI_KEY_TANGENT_AUTO | UFBXWI_KEY_CLAMP;
	} else if (prev_flags & UFBXW_KEYFRAME_TANGENT_USER) {
		flags |= UFBXWI_KEY_TANGENT_USER;
	} else if (prev_flags & UFBXW_KEYFRAME_TANGENT_TCB) {
		flags |= UFBXWI_KEY_TANGENT_TCB;
	}

	if (prev_flags & UFBXW_KEYFRAME_TANGENT_BROKEN) flags |= UFBXWI_KEY_TANGENT_BROKEN;
	if (prev_flags & UFBXW_KEYFRAME_WEIGHTED_RIGHT) flags |= UFBXWI_KEY_WEIGHTED_RIGHT;
	if (next_flags & UFBXW_KEYFRAME_WEIGHTED_LEFT) flags |= UFBXWI_KEY_WEIGHTED_NEXT_LEFT;
	return flags;
}

static void ufbxwi_save_anim_curve_keys(ufbxwi_save_context *sc, ufbxwi_element *element)
{
	const ufbxwi_anim_curve *curve = (const ufbxwi_anim_curve*)element;

	if (curve->data_in_buffers) {
		ufbxwi_dom_array(sc, "KeyTime", curve->buffer_key_times.id);
		ufbxwi_dom_array(sc, "KeyValueFloat", curve->buffer_key_values.id);
		ufbxwi_dom_array(sc, "KeyAttrFlags", curve->buffer_attr_flags.id);
		ufbxwi_dom_array(sc, "KeyAttrDataFloat", curve->buffer_attr_data.id);
		ufbxwi_dom_array(sc, "KeyAttrRefCount", curve->buffer_attr_refcounts.id);
		return;
	}

	ufbxw_buffer_id buf_times = ufbxwi_create_external_buffer(&sc->buffers, UFBXWI_BUFFER_TYPE_LONG, curve->key_times.data, curve->key_times.count, 0);
	ufbxw_buffer_id buf_values = ufbxwi_create_external_buffer(&sc->buffers, UFBXWI_BUFFER_TYPE_FLOAT, curve->key_values.data, curve->key_values.count, 0);

	size_t max_attrs = ufbxwi_min_sz(curve->key_attr_data.count * 2, curve->key_values.count);

	// TODO: Pool these
	int32_t *key_flags = ufbxwi_alloc(&sc->ator, int32_t, max_attrs);
	int32_t *key_refcounts = ufbxwi_alloc(&sc->ator, int32_t, max_attrs);
	ufbxwi_key_attr *key_attrs = ufbxwi_alloc(&sc->ator, ufbxwi_key_attr, max_attrs);
	ufbxwi_check(key_flags && key_refcounts && key_attrs);

	size_t attr_count = 0;
	size_t key_count = curve->key_values.count;
	uint32_t prev_prev_ix = ~0u;
	for (size_t i = 0; i < key_count; i++) {
		size_t ni = ufbxwi_min_sz(i + 1, key_count - 1);
		uint32_t prev_ix = curve->key_attr_indices.data[i];
		uint32_t next_ix = curve->key_attr_indices.data[ni];
		if (prev_ix == next_ix && prev_ix == prev_prev_ix) {
			// Simple repeat, must have an attr if `prev_ix` exists.
			ufbxw_assert(attr_count > 0);
			key_refcounts[attr_count - 1]++;
			continue;
		}

		ufbxwi_anim_key_attr prev_attr = curve->key_attr_data.data[prev_ix];
		ufbxwi_anim_key_attr next_attr = curve->key_attr_data.data[next_ix];

		float weight_right = prev_attr.weight_right;
		float weight_next_left = next_attr.weight_left;

		uint32_t flags = ufbxwi_get_key_flags(prev_attr.flags, next_attr.flags);

		ufbxwi_key_attr attr;
		attr.slope_right = prev_attr.slope_right;
		attr.slope_next_left = next_attr.slope_left;
		attr.packed_weight = ufbxwi_pack_weight(weight_right) | ufbxwi_pack_weight(weight_next_left) << 16;
		attr.packed_velocity = 0;

		if (attr_count > 0 && key_flags[attr_count - 1] == flags && !memcmp(&key_attrs[attr_count - 1], &attr, sizeof(ufbxwi_key_attr))) {
			key_refcounts[attr_count - 1]++;
		} else {
			key_flags[attr_count] = flags;
			key_attrs[attr_count] = attr;
			key_refcounts[attr_count] = 1;
			attr_count++;
		}

		prev_prev_ix = prev_ix;
	}

	ufbxw_assert(attr_count <= max_attrs);
	ufbxw_buffer_id buf_flags = ufbxwi_create_external_buffer(&sc->buffers, UFBXWI_BUFFER_TYPE_INT, key_flags, attr_count, 0);
	ufbxw_buffer_id buf_attrs = ufbxwi_create_external_buffer(&sc->buffers, UFBXWI_BUFFER_TYPE_KEY_ATTR, key_attrs, attr_count, 0);
	ufbxw_buffer_id buf_refcounts = ufbxwi_create_external_buffer(&sc->buffers, UFBXWI_BUFFER_TYPE_INT, key_refcounts, attr_count, 0);
	ufbxwi_buffer_set_deleter(&sc->buffers, buf_flags, ufbxwi_deleter_free, &sc->ator);
	ufbxwi_buffer_set_deleter(&sc->buffers, buf_attrs, ufbxwi_deleter_free, &sc->ator);
	ufbxwi_buffer_set_deleter(&sc->buffers, buf_refcounts, ufbxwi_deleter_free, &sc->ator);

	ufbxwi_dom_array(sc, "KeyTime", buf_times);
	ufbxwi_dom_array(sc, "KeyValueFloat", buf_values);
	ufbxwi_dom_array(sc, "KeyAttrFlags", buf_flags);
	ufbxwi_dom_array(sc, "KeyAttrDataFloat", buf_attrs);
	ufbxwi_dom_array(sc, "KeyAttrRefCount", buf_refcounts);
}

static void ufbxwi_save_element(ufbxwi_save_context *sc, ufbxwi_element *element, uint32_t flags)
{
	ufbxw_scene *scene = sc->scene;

	ufbxw_id id = element->id;
	ufbxw_element_type type = ufbxwi_id_type(id);
	const ufbxwi_element_type *et = &scene->element_types.data[element->type_id];

	ufbxwi_token fbx_type = et->fbx_type;
	ufbxwi_token sub_type = et->sub_type;
	ufbxwi_token obj_type = et->object_type;

	if (type == UFBXW_ELEMENT_NODE) {
		ufbxwi_node *node = (ufbxwi_node*)element;

		ufbxwi_element *attrib = ufbxwi_get_element(scene, node->attribute);
		if (attrib) {
			const ufbxwi_element_type *attrib_et = &scene->element_types.data[attrib->type_id];
			sub_type = attrib_et->sub_type;
		} else {
			sub_type = UFBXWI_TOKEN_EMPTY; // TODO: What is the correct thing here?
		}
	}

	ufbxw_string obj_type_str = scene->string_pool.tokens.data[obj_type];
	ufbxw_string fbx_type_str = scene->string_pool.tokens.data[fbx_type];

	if ((flags & UFBXWI_SAVE_ELEMENT_MANUAL_OPEN) == 0) {
		// TODO: Dynamic buffer, do not use printf here
		char name_buf[256];
		ufbxw_string name;
		if (sc->ascii) {
			name.data = name_buf;
			name.length = (size_t)snprintf(name_buf, sizeof(name_buf), "%s::%s", fbx_type_str.data, element->name.data);
		} else {
			name.data = name_buf;
			memcpy(name_buf, element->name.data, element->name.length);
			memcpy(name_buf + element->name.length, "\x00\x01", 2);
			memcpy(name_buf + element->name.length + 2, fbx_type_str.data, fbx_type_str.length);
			name_buf[element->name.length + 2 + fbx_type_str.length] = '\0';
			name.length = element->name.length + 2 + fbx_type_str.length;
		}

		if ((flags & UFBXWI_SAVE_ELEMENT_NO_ID) != 0) {
			ufbxwi_dom_open(sc, obj_type_str.data, "ST", name, sub_type);
		} else {
			ufbxwi_dom_open(sc, obj_type_str.data, "LST", (int64_t)id, name, sub_type);
		}
	}

	if (type == UFBXW_ELEMENT_NODE) {
		ufbxwi_dom_value(sc, "Version", "I", 232);
	} else if (type == UFBXW_ELEMENT_MESH) {
	} else if (type == UFBXW_ELEMENT_BLEND_DEFORMER) {
		ufbxwi_dom_value(sc, "Version", "I", 100);
	} else if (type == UFBXW_ELEMENT_BLEND_CHANNEL) {
		ufbxwi_dom_value(sc, "Version", "I", 100);
	} else if (type == UFBXW_ELEMENT_SCENE_INFO) {
		ufbxwi_dom_value(sc, "Type", "C", "UserData");
		ufbxwi_dom_value(sc, "Version", "I", 100);

		ufbxwi_dom_open(sc, "MetaData", "");
		ufbxwi_dom_value(sc,"Version", "I", 100);
		ufbxwi_dom_value(sc, "Title", "C", "");
		ufbxwi_dom_value(sc, "Subject", "C", "");
		ufbxwi_dom_value(sc, "Author", "C", "");
		ufbxwi_dom_value(sc, "Keywords", "C", "");
		ufbxwi_dom_value(sc, "Revision", "C", "");
		ufbxwi_dom_value(sc, "Comment", "C", "");
		ufbxwi_dom_close(sc);
	} else if (type == UFBXW_ELEMENT_GLOBAL_SETTINGS) {
		ufbxwi_dom_value(sc, "Version", "I", 1000);
	} else if (type == UFBXW_ELEMENT_MATERIAL) {
		const ufbxwi_material *material = (const ufbxwi_material*)element;
		ufbxwi_dom_value(sc, "Version", "I", 120);
		ufbxwi_dom_value(sc, "ShadingModel", "S", material->shading_model);
		ufbxwi_dom_value(sc, "MultiLayer", "I", material->multi_layer ? 1 : 0);
	} else if (type == UFBXW_ELEMENT_TEXTURE) {
		const ufbxwi_texture *texture = (const ufbxwi_texture*)element;
		ufbxwi_dom_value(sc, "Type", "S", texture->type);
		ufbxwi_dom_value(sc, "Version", "I", 202);
	}

	const ufbxwi_element_type *elem_type = &scene->element_types.data[element->type_id];

	const ufbxwi_props *default_props = NULL;
	if (element->flags & UFBXWI_ELEMENT_FLAG_HAS_DEFAULT_PROPS) {
		default_props = &elem_type->props;
	}

	ufbxwi_element *tmpl = ufbxwi_get_element(sc->scene, elem_type->template_id);
	ufbxwi_save_props(sc, element, default_props, tmpl);

	if (type == UFBXW_ELEMENT_DOCUMENT) {
		ufbxwi_document *document = (ufbxwi_document*)element;
		ufbxwi_dom_value(sc, "RootNode", "L", document->root_node);
	}

	if (type == UFBXW_ELEMENT_NODE) {
		// TODO: Use actual values for these
		ufbxwi_dom_value(sc, "Shading", "c", 'T');
		ufbxwi_dom_value(sc, "Culling", "C", "CullingOff");
	}

	if (type == UFBXW_ELEMENT_MESH) {
		ufbxwi_save_mesh_data(sc, element);
	}

	if (type == UFBXW_ELEMENT_SKIN_DEFORMER) {
		ufbxwi_skin_deformer *skin = (ufbxwi_skin_deformer*)element;

		ufbxwi_dom_value(sc, "Version", "I", 101);

		// TODO: Should this be configurable?
		// SIC: Typo in the format
		ufbxwi_dom_value(sc, "Link_DeformAcuracy", "D", 50.0);

		const char *skinning_type = "";
		switch (skin->skinning_type) {
		case UFBXW_SKINNING_TYPE_RIGID:
			skinning_type = "Rigid";
			break;
		case UFBXW_SKINNING_TYPE_LINEAR:
			skinning_type = "Linear";
			break;
		case UFBXW_SKINNING_TYPE_DUAL_QUATERNION:
			skinning_type = "DualQuaternion";
			break;
		case UFBXW_SKINNING_TYPE_BLEND:
			skinning_type = "Blend";
			break;
		default:
			ufbxwi_unreachable("unhandled skinning type");
		}
		ufbxwi_dom_value(sc, "SkinningType", "C", skinning_type);
	}

	if (type == UFBXW_ELEMENT_SKIN_CLUSTER) {
		ufbxwi_save_skin_cluster(sc, (ufbxwi_skin_cluster*)element);
	}

	if (type == UFBXW_ELEMENT_BLEND_CHANNEL) {
		ufbxwi_save_blend_channel(sc, (ufbxwi_blend_channel*)element);
	}

	if (type == UFBXW_ELEMENT_BLEND_SHAPE) {
		ufbxwi_save_blend_shape(sc, (ufbxwi_blend_shape*)element);
	}

	// TODO: Light, camera

	if (type == UFBXW_ELEMENT_SKELETON) {
		ufbxwi_dom_value(sc, "TypeFlags", "C", "Skeleton");
	}

	if (type == UFBXW_ELEMENT_BIND_POSE) {
		ufbxwi_save_bind_pose(sc, (ufbxwi_bind_pose*)element);
	}

	if (type == UFBXW_ELEMENT_LIGHT) {
		// TODO: Use actual values for these
		ufbxwi_dom_value(sc, "TypeFlags", "C", "Light");
		ufbxwi_dom_value(sc, "GeometryVersion", "I", 124);
	}

	if (type == UFBXW_ELEMENT_TEXTURE) {
		const ufbxwi_texture *texture = (const ufbxwi_texture*)element;
		ufbxwi_dom_value(sc, "FileName", "S", texture->filename);
		ufbxwi_dom_value(sc, "RelativeFilename", "S", texture->relative_filename);
	}

	if (type == UFBXW_ELEMENT_ANIM_CURVE) {
		ufbxwi_dom_value(sc, "Default", "D", 0.0); // Type?
		ufbxwi_dom_value(sc, "KeyVer", "I", 4009);
		ufbxwi_save_anim_curve_keys(sc, element);
	}

	ufbxwi_dom_close(sc);
}

static void ufbxwi_save_timestamp(ufbxwi_save_context *sc, ufbxw_datetime timestamp)
{
	ufbxwi_dom_open(sc, "CreationTimeStamp", "");
	ufbxwi_dom_value(sc, "Version", "I", 1000);
	ufbxwi_dom_value(sc, "Year", "I", timestamp.year);
	ufbxwi_dom_value(sc, "Month", "I", timestamp.month);
	ufbxwi_dom_value(sc, "Day", "I", timestamp.day);
	ufbxwi_dom_value(sc, "Hour", "I", timestamp.hour);
	ufbxwi_dom_value(sc, "Minute", "I", timestamp.minute);
	ufbxwi_dom_value(sc, "Second", "I", timestamp.second);
	ufbxwi_dom_value(sc, "Millisecond", "I", timestamp.millisecond);
	ufbxwi_dom_close(sc);
}

static void ufbxwi_save_documents(ufbxwi_save_context *sc)
{
	ufbxw_scene *scene = sc->scene;

	ufbxwi_dom_open(sc, "Documents", "");

	size_t document_count = 0;

	ufbxwi_for_list(ufbxwi_element_slot, slot, scene->elements) {
		ufbxw_element_type type = ufbxwi_id_type(slot->id);
		if (type != UFBXW_ELEMENT_DOCUMENT) continue;
		document_count++;
	}

	ufbxwi_dom_value(sc, "Count", "I", (int32_t)document_count);

	ufbxwi_for_list(ufbxwi_element_slot, slot, scene->elements) {
		ufbxw_element_type type = ufbxwi_id_type(slot->id);
		if (type != UFBXW_ELEMENT_DOCUMENT) continue;

		ufbxwi_dom_open(sc, "Document", "LCC", slot->id, "", "Scene");
		ufbxwi_save_element(sc, slot->element, UFBXWI_SAVE_ELEMENT_MANUAL_OPEN);
	}

	ufbxwi_dom_close(sc);
}

static void ufbxwi_save_definitions(ufbxwi_save_context *sc)
{
	ufbxw_scene *scene = sc->scene;

	size_t object_type_count = 0;
	ufbxwi_for_list(ufbxwi_save_object_type, obj_type, sc->object_types) {
		if (obj_type->reference_count == 0) continue;
		object_type_count++;
	}

	ufbxwi_dom_open(sc, "Definitions", "");
	ufbxwi_dom_value(sc, "Version", "I", 100);
	ufbxwi_dom_value(sc, "Count", "I", (int32_t)object_type_count);

	for (size_t i = 0; i < sc->object_types.count; i++) {
		const ufbxwi_object_type *scene_obj_type = &scene->object_types.data[i];
		ufbxwi_save_object_type *obj_type = &sc->object_types.data[i];
		if (obj_type->reference_count == 0) continue;

		ufbxwi_dom_open(sc, "ObjectType", "T", scene_obj_type->type);
		ufbxwi_dom_value(sc, "Count", "I", (int32_t)obj_type->reference_count);

		ufbxwi_element *tmpl_elem = ufbxwi_get_element(scene, obj_type->template_id);
		if (tmpl_elem) {
			ufbxwi_template *tmpl = (ufbxwi_template*)tmpl_elem;
			ufbxwi_dom_open(sc, "PropertyTemplate", "T", tmpl->type);
			ufbxwi_save_props(sc, &tmpl->element, NULL, NULL);
			ufbxwi_dom_close(sc);
		}

		ufbxwi_dom_close(sc);
	}

	ufbxwi_dom_close(sc);
}

static void ufbxwi_save_objects(ufbxwi_save_context *sc)
{
	ufbxw_scene *scene = sc->scene;

	ufbxwi_dom_open(sc, "Objects", "");

	ufbxwi_for_list(ufbxwi_element_slot, slot, scene->elements) {
		if (ufbxwi_is_fatal(&sc->error)) break;

		ufbxw_element_type type = ufbxwi_id_type(slot->id);
		if (type == UFBXWI_ELEMENT_TYPE_NONE) continue;

		bool do_save = true;
		switch (type) {
		case UFBXW_ELEMENT_DOCUMENT:
		case UFBXW_ELEMENT_TEMPLATE:
		case UFBXW_ELEMENT_SCENE_INFO:
		case UFBXW_ELEMENT_GLOBAL_SETTINGS:
			do_save = false;
			break;
		default:
			break;
		}

		if (do_save) {
			ufbxwi_save_element(sc, slot->element, 0);
		}
	}

	ufbxwi_dom_close(sc);
}

static void ufbxwi_save_connections(ufbxwi_save_context *sc)
{
	ufbxw_scene *scene = sc->scene;

	ufbxwi_dom_open(sc, "Connections", "");

	// Connect root nodes to root

	ufbxwi_for_list(ufbxwi_element_slot, src_slot, scene->elements) {
		if (ufbxwi_is_fatal(&sc->error)) break;

		ufbxw_id src_id = src_slot->id;
		ufbxw_element_type src_type = ufbxwi_id_type(src_id);
		if (src_type != UFBXW_ELEMENT_NODE) continue;

		ufbxwi_node *node = (ufbxwi_node*)src_slot->element;
		if (node->parent.id == 0) {
			if (sc->ascii) {
				if (sc->opts.debug_comments) {
					ufbxwi_dom_comment(sc, "\n\t;Model::%s, Model::RootNode (Root Node)\n", node->element.name.data);
				} else {
					ufbxwi_dom_comment(sc, "\n\t;Model::%s, Model::RootNode\n", node->element.name.data);
				}
			}
			ufbxwi_dom_value(sc, "C", "CLL", "OO", src_id, 0);
		}
	}

	// TODO: This could be accelerated with a bit-mask of active connection types

	ufbxwi_for_list(ufbxwi_element_slot, dst_slot, scene->elements) {
		if (ufbxwi_is_fatal(&sc->error)) break;

		ufbxw_id dst_id = dst_slot->id;
		if (ufbxwi_id_type(dst_id) == UFBXWI_ELEMENT_TYPE_NONE) continue;
		ufbxwi_element *dst_element = dst_slot->element;

		for (uint32_t i = 1; i < UFBXW_CONNECTION_TYPE_COUNT; i++) {
			ufbxw_connection_type conn_type = (ufbxw_connection_type)i;

			sc->tmp_conns.count = 0;
			ufbxwi_collect_src_connections(scene, &sc->ator, &sc->tmp_conns, conn_type, dst_element);

			ufbxwi_for_list(ufbxwi_conn, p_conn, sc->tmp_conns) {
				ufbxwi_conn conn = *p_conn;
				ufbxw_id src_id = conn.id;

				ufbxwi_element *src_element = ufbxwi_get_element(scene, conn.id);
				ufbxw_assert(src_element);

				if (sc->ascii) {
					const ufbxwi_element_type *src_et = &scene->element_types.data[src_element->type_id];
					const ufbxwi_element_type *dst_et = &scene->element_types.data[dst_element->type_id];
					const char *src_type = scene->string_pool.tokens.data[src_et->fbx_type].data;
					const char *dst_type = scene->string_pool.tokens.data[dst_et->fbx_type].data;
					const char *src_name = src_element->name.data;
					const char *dst_name = dst_element->name.data;

					if (sc->opts.debug_comments) {
						const char *conn_name = ufbxwi_connection_infos[conn_type].debug_name;
						ufbxwi_dom_comment(sc, "\n\t;%s::%s, %s::%s (%s)\n", src_type, src_name, dst_type, dst_name, conn_name);
					} else {
						ufbxwi_dom_comment(sc, "\n\t;%s::%s, %s::%s\n", src_type, src_name, dst_type, dst_name);
					}
				}

				if (conn.src_prop == UFBXWI_TOKEN_NONE && conn.dst_prop == UFBXWI_TOKEN_NONE) {
					ufbxwi_dom_value(sc, "C", "CLL", "OO", src_id, dst_id);
				} else if (conn.src_prop == UFBXWI_TOKEN_NONE && conn.dst_prop != UFBXWI_TOKEN_NONE) {
					ufbxwi_dom_value(sc, "C", "CLLT", "OP", src_id, dst_id, conn.dst_prop);
				} else if (conn.src_prop != UFBXWI_TOKEN_NONE && conn.dst_prop == UFBXWI_TOKEN_NONE) {
					ufbxwi_dom_value(sc, "C", "CLTL", "PO", src_id, conn.src_prop, dst_id);
				} else if (conn.src_prop != UFBXWI_TOKEN_NONE && conn.dst_prop != UFBXWI_TOKEN_NONE) {
					ufbxwi_dom_value(sc, "C", "CLTLT", "PP", src_id, conn.src_prop, dst_id, conn.dst_prop);
				}
			}
		}
	}

	ufbxwi_dom_close(sc);
}

static void ufbxwi_save_takes(ufbxwi_save_context *sc)
{
	ufbxwi_dom_open(sc, "Takes", "");

	ufbxwi_dom_value(sc, "Current", "C", "Take 001");

	// TODO: Optimize with `elements_by_type[]`
	ufbxwi_for_list(ufbxwi_element_slot, slot, sc->scene->elements) {
		ufbxw_element_type type = ufbxwi_id_type(slot->id);
		if (type != UFBXW_ELEMENT_ANIM_STACK) continue;
		ufbxwi_element *element = slot->element;

		// TODO: Format utility
		char take_name[256];
		snprintf(take_name, sizeof(take_name), "%s.tak", element->name.data);

		ufbxwi_anim_stack *stack = (ufbxwi_anim_stack*)element;
		ufbxwi_dom_open(sc, "Take", "S", element->name);
		ufbxwi_dom_value(sc, "FileName", "C", take_name);
		ufbxwi_dom_value(sc, "LocalTime", "LL", stack->local_start, stack->local_stop);
		ufbxwi_dom_value(sc, "ReferenceTime", "LL", stack->reference_start, stack->reference_stop);
		ufbxwi_dom_close(sc);
	}

	ufbxwi_dom_close(sc);
}

static void ufbxwi_save_root(ufbxwi_save_context *sc)
{
	ufbxw_scene *scene = sc->scene;

	ufbxw_string creator;
	if (sc->opts.enable_override_creator) {
		creator = sc->opts.override_creator;
	} else {
		creator = ufbxwi_c_str("ufbx_write (version format TBD)");
	}

	ufbxw_datetime timestamp = sc->opts.local_timestamp;
	if (ufbxwi_is_zero_date(&timestamp) && !sc->opts.no_default_timestamp) {
		ufbxwi_get_local_date(&timestamp);
	}

	if (sc->ascii) {
		uint32_t major = sc->opts.version / 1000 % 10;
		uint32_t minor = sc->opts.version / 100 % 10;
		uint32_t patch = sc->opts.version % 100;
		ufbxwi_dom_comment(sc, "; FBX %u.%u.%u project file\n; %.52s\n\n", major, minor, patch, ufbxwi_dom_section_str);
	} else {
		ufbxwi_binary_write_header(sc);
	}

	// Header extension
	{
		ufbxwi_dom_open(sc, "FBXHeaderExtension", "");
		ufbxwi_dom_value(sc, "FBXHeaderVersion", "I", 1003);
		ufbxwi_dom_value(sc, "FBXVersion", "I", sc->opts.version);
		ufbxwi_save_timestamp(sc, timestamp);
		ufbxwi_dom_value(sc, "Creator", "S", creator);

		{
			ufbxw_id scene_info_id = ufbxw_get_scene_info_id(scene);
			ufbxwi_element *scene_info = ufbxwi_get_element(scene, scene_info_id);
			if (scene_info) {
				ufbxwi_save_element(sc, scene_info, UFBXWI_SAVE_ELEMENT_NO_ID);
			}
		}

		ufbxwi_dom_close(sc);
	}

	// Creation time in another format used by binary checksums
	char creation_time[32];
	snprintf(creation_time, sizeof(creation_time), "%04d-%02d-%02d %02d:%02d:%02d:%03d",
		timestamp.year, timestamp.month, timestamp.day,
		timestamp.hour, timestamp.minute, timestamp.second, timestamp.millisecond);

	// Weird binary-only section with some redundant data?
	if (!sc->ascii) {
		char file_id_magic[16];
		ufbxwi_binary_fileid_magic(file_id_magic, creation_time);

		ufbxw_blob file_id_blob = { file_id_magic, 16 };
		ufbxwi_dom_value(sc, "FileId", "R", file_id_blob);
		ufbxwi_dom_value(sc, "CreationTime", "C", creation_time);
		ufbxwi_dom_value(sc, "Creator", "S", creator);
	}

	{
		ufbxw_id global_settings_id = ufbxw_get_global_settings_id(scene);
		ufbxwi_element *global_settings = ufbxwi_get_element(scene, global_settings_id);
		if (global_settings) {
			ufbxwi_dom_open(sc, "GlobalSettings", "");
			ufbxwi_save_element(sc, global_settings, UFBXWI_SAVE_ELEMENT_MANUAL_OPEN);
		}
	}

	ufbxwi_dom_section(sc, "Documents Description");
	ufbxwi_save_documents(sc);

	ufbxwi_dom_section(sc, "Document References");
	ufbxwi_dom_open(sc, "References", "");
	ufbxwi_dom_close(sc);

	ufbxwi_dom_section(sc, "Object definitions");
	ufbxwi_save_definitions(sc);

	ufbxwi_dom_section(sc, "Object properties");
	ufbxwi_save_objects(sc);

	ufbxwi_dom_section(sc, "Object connections");
	ufbxwi_save_connections(sc);

	// Differently formatted section...
	ufbxwi_dom_comment(sc, ";Takes section\n;%.52s\n\n", ufbxwi_dom_section_str);

	ufbxwi_save_takes(sc);

	// Binary footer
	if (!sc->ascii) {
		// We need to make sure all the chunks are flushed
		if (!sc->write_queue.current_chunk->has_file_offset) {
			ufbxwi_write_queue_finish(&sc->write_queue);
		}

		ufbxwi_binary_write_footer(sc, creation_time);
	}
}

static void ufbxwi_save_init(ufbxwi_save_context *sc)
{
	ufbxw_scene *scene = sc->scene;

	ufbxwi_save_object_type *object_types = ufbxwi_list_push_zero_n(&sc->ator, &sc->object_types, ufbxwi_save_object_type, scene->object_types.count);
	ufbxwi_check(object_types);

	ufbxwi_for_list(ufbxwi_element_slot, slot, scene->elements) {
		ufbxwi_element *element = slot->element;
		if (!element) continue;

		const ufbxwi_element_type *et = &scene->element_types.data[element->type_id];
		if (et->object_type_id != ~0u) {
			ufbxwi_save_object_type *object_type = &object_types[et->object_type_id];
			object_type->reference_count++;

			// TODO: Better prioritization?
			if (object_type->template_id == 0 && et->template_id != 0) {
				object_type->template_id = et->template_id;
			}
		}
	}

}

// -- File IO

static bool ufbxwi_stdio_write(void *user, uint64_t offset, const void *data, size_t size)
{
	// TODO: Do not seek all the time, support >4GB files
	FILE *f = (FILE*)user;
	if (fseek(f, (int)offset, SEEK_SET)) return false;

	size_t num_written = fwrite(data, 1, size, f);
	if (num_written != size) return false;

	return true;
}

static void ufbxwi_stdio_close(void *user)
{
	FILE *f = (FILE*)user;
	fclose(f);
}

static bool ufbxwi_open_file_write(ufbxw_write_stream *stream, const char *path, size_t path_len, ufbxwi_error *error)
{
	if (path_len >= 1023) {
		ufbxwi_failf(error, UFBXW_ERROR_PATH_TOO_LONG, "Path too long (%zu bytes, max 1023)", path_len);
		return false;
	}

	// TODO: Do this properly
	char copy[1024];
	memcpy(copy, path, path_len);
	copy[path_len] = '\0';

	// TODO: fopen_s() etc
	FILE *f = fopen(copy, "wb");
	if (!f) {
		ufbxwi_failf(error, UFBXW_ERROR_FILE_OPEN_FAILED, "Failed to open file: %.*s", (int)path_len, path);
		return false;
	}

	stream->write_fn = ufbxwi_stdio_write;
	stream->close_fn = ufbxwi_stdio_close;
	stream->user = f;

	return true;
}

// -- Save root

static void ufbxwi_mark_save_context_failed(ufbxwi_save_context *sc)
{
	ufbxwi_mark_buffers_failed(&sc->buffers);

	// TODO(wq): Make this better
	sc->write_queue.buffer_begin = NULL;
	sc->write_queue.buffer_pos = NULL;
	sc->write_queue.buffer_end = NULL;
}

static void ufbxwi_save_fatal(void *user, ufbxwi_error *error)
{
	ufbxwi_save_context *sc = (ufbxwi_save_context*)user;
	ufbxwi_mark_save_context_failed(sc);
}

static void ufbxwi_save_imp(ufbxwi_save_context *sc, ufbxw_write_stream *stream)
{
	ufbxw_assert(sc->opts._begin_zero == 0 && sc->opts._end_zero == 0);

	if (sc->opts.version == 0) sc->opts.version = 7500;

	sc->ascii = sc->opts.format == UFBXW_SAVE_FORMAT_ASCII;

	sc->error.fatal_fn = &ufbxwi_save_fatal;
	sc->error.fatal_user = sc;
	sc->ator.error = &sc->error;

	// TODO: Options for these
	sc->ator.max_allocs = SIZE_MAX;
	sc->ator.max_size = SIZE_MAX / 4;

	// TODO: Proper hanling
	sc->buffers.ator = &sc->ator;
	sc->buffers.error = &sc->error;
	ufbxwi_refer_buffers(&sc->buffers, &sc->scene->buffers);
	if (ufbxwi_is_fatal(&sc->error)) return;

	if (sc->opts.compression_level == 0) sc->opts.compression_level = 6;
	if (sc->opts.deflate_window_size == 0) sc->opts.deflate_window_size = UFBXWI_DEFLATE_WINDOW_SIZE;

	size_t buffer_size = 0x10000;
	if (sc->opts.buffer_size > 0) {
		buffer_size = ufbxwi_max_sz(sc->opts.buffer_size, 512);
	}

	if (!sc->opts.ascii_formatter.format_int_fn) {
		sc->opts.ascii_formatter.format_int_fn = &ufbxwi_default_ascii_format_int;
	}
	if (!sc->opts.ascii_formatter.format_long_fn) {
		sc->opts.ascii_formatter.format_long_fn = &ufbxwi_default_ascii_format_long;
}
	if (!sc->opts.ascii_formatter.format_float_fn) {
		sc->opts.ascii_formatter.format_float_fn = &ufbxwi_default_ascii_format_float;
	}
	if (!sc->opts.ascii_formatter.format_double_fn) {
		sc->opts.ascii_formatter.format_double_fn = &ufbxwi_default_ascii_format_double;
	}

	// TODO: Determine these somehow
	if (sc->opts.threaded_min_deflate_bytes == 0) {
		sc->opts.threaded_min_deflate_bytes = 512;
	}
	if (sc->opts.threaded_min_ascii_floats == 0) {
		sc->opts.threaded_min_ascii_floats = 128;
	}
	if (sc->opts.threaded_min_ascii_ints == 0) {
		sc->opts.threaded_min_ascii_ints = 256;
	}

	// TODO(threads): Better checking
	if (sc->opts.thread_sync.wait_fn) {
		if (!ufbxwi_thread_pool_init(&sc->thread_pool, &sc->opts.thread_sync)) {
			ufbxwi_failf(&sc->error, UFBXW_ERROR_THREAD_SYNC_INIT, "failed to init thread sync");
			return;
		}

		sc->thread_error.thread_pool = &sc->thread_pool;

		sc->thread_ator.error = &sc->thread_error;
		sc->thread_ator.thread_pool = &sc->thread_pool;

		// TODO: Options for these
		sc->thread_ator.max_allocs = SIZE_MAX;
		sc->thread_ator.max_size = SIZE_MAX / 4;

		ufbxwi_task_queue_opts task_queue_opts = { 0 };

		// TODO(tq): Make this configurable
		task_queue_opts.max_tasks = 1024;
		task_queue_opts.num_threads = 8;

		task_queue_opts.create_thread_ctx_fn = &ufbxwi_create_save_thread_context;
		task_queue_opts.free_thread_ctx_fn = &ufbxwi_free_save_thread_context;
		task_queue_opts.thread_ctx_user = sc;

		ufbxwi_task_queue_init(&sc->task_queue, &sc->thread_pool, &sc->thread_ator, &task_queue_opts, &sc->opts.thread_pool);

		ufbxwi_init_save_thread_context(sc, &sc->main_thread_ctx);
		ufbxwi_write_queue_init(&sc->write_queue, &sc->thread_ator, &sc->thread_error, &sc->task_queue, &sc->main_thread_ctx, *stream, buffer_size);

	} else {
		ufbxwi_init_save_thread_context(sc, &sc->main_thread_ctx);
		ufbxwi_write_queue_init(&sc->write_queue, &sc->ator, &sc->error, NULL, &sc->main_thread_ctx, *stream, buffer_size);
	}

	// HACK: Hook the write queue to the buffer pool, as write chunks may temporarily own a buffer that
	// needs to be returned on the main thread.
	sc->write_queue.buffer_pool = &sc->buffers;

	ufbxwi_save_init(sc);
	ufbxwi_save_root(sc);

	ufbxwi_write_queue_finish(&sc->write_queue);
}

#endif

// -- API

#ifdef UFBXWI_FEATURE_API

#ifdef __cplusplus
extern "C" {
#endif

ufbxw_abi_data_def const ufbxw_string ufbxw_empty_string = { ufbxwi_empty_char, 0 };
ufbxw_abi_data_def const ufbxw_vec2 ufbxw_zero_vec2 = { 0.0f, 0.0f };
ufbxw_abi_data_def const ufbxw_vec3 ufbxw_zero_vec3 = { 0.0f, 0.0f, 0.0f };
ufbxw_abi_data_def const ufbxw_vec4 ufbxw_zero_vec4 = { 0.0f, 0.0f, 0.0f, 0.0f };
ufbxw_abi_data_def const ufbxw_matrix ufbxw_identity_matrix = {
	1.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 1.0f,
};

ufbxw_abi void ufbxw_retain_buffer(ufbxw_scene *scene, ufbxw_buffer_id buffer)
{
	ufbxwi_buffer *buf = ufbxwi_get_buffer(&scene->buffers, buffer);
	ufbxw_assert(buf || ufbxwi_is_fatal(&scene->error));
	if (!buf) return;

	ufbxw_assert(buf->user_refcount > 0);
	buf->user_refcount++;
	buf->refcount++;
}

ufbxw_abi void ufbxw_free_buffer(ufbxw_scene *scene, ufbxw_buffer_id buffer)
{
	ufbxwi_buffer *buf = ufbxwi_get_buffer(&scene->buffers, buffer);
	ufbxw_assert(buf || ufbxwi_is_fatal(&scene->error));
	if (!buf) return;

	ufbxw_assert(buf->user_refcount > 0);
	buf->user_refcount--;

	ufbxwi_free_buffer(&scene->buffers, buffer);
}

ufbxw_abi void ufbxw_buffer_set_deleter(ufbxw_scene *scene, ufbxw_buffer_id buffer, ufbxw_buffer_deleter_fn *fn, void *user)
{
	ufbxwi_buffer_set_deleter(&scene->buffers, buffer, fn, user);
}

ufbxw_abi ufbxw_int_buffer ufbxw_create_int_buffer(ufbxw_scene *scene, size_t count)
{
	ufbxw_buffer_id id = ufbxwi_create_owned_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_INT, count);
	return ufbxwi_to_user_int_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_int_buffer ufbxw_copy_int_array(ufbxw_scene *scene, const int32_t *data, size_t count)
{
	ufbxw_buffer_id id = ufbxwi_create_copy_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_INT, data, count);
	return ufbxwi_to_user_int_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_int_buffer ufbxw_view_int_array(ufbxw_scene *scene, const int32_t *data, size_t count)
{
	ufbxw_buffer_id id = ufbxwi_create_external_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_INT, data, count, UFBXWI_BUFFER_FLAG_TEMPORARY);
	return ufbxwi_to_user_int_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_int_buffer ufbxw_external_int_array(ufbxw_scene *scene, const int32_t *data, size_t count)
{
	ufbxw_buffer_id id = ufbxwi_create_external_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_INT, data, count, 0);
	return ufbxwi_to_user_int_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_int_buffer ufbxw_external_int_stream(ufbxw_scene *scene, ufbxw_int_stream_fn *fn, void *user, size_t count)
{
	ufbxwi_stream_fn stream_fn;
	stream_fn.int_fn = fn;
	ufbxw_buffer_id id = ufbxwi_create_stream_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_INT, stream_fn, user, count);
	return ufbxwi_to_user_int_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_long_buffer ufbxw_create_long_buffer(ufbxw_scene *scene, size_t count)
{
	ufbxw_buffer_id id = ufbxwi_create_owned_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_LONG, count);
	return ufbxwi_to_user_long_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_long_buffer ufbxw_copy_long_array(ufbxw_scene *scene, const int64_t *data, size_t count)
{
	ufbxw_buffer_id id = ufbxwi_create_copy_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_LONG, data, count);
	return ufbxwi_to_user_long_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_long_buffer ufbxw_view_long_array(ufbxw_scene *scene, const int64_t *data, size_t count)
{
	ufbxw_buffer_id id = ufbxwi_create_external_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_LONG, data, count, UFBXWI_BUFFER_FLAG_TEMPORARY);
	return ufbxwi_to_user_long_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_long_buffer ufbxw_external_long_array(ufbxw_scene *scene, const int64_t *data, size_t count)
{
	ufbxw_buffer_id id = ufbxwi_create_external_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_LONG, data, count, 0);
	return ufbxwi_to_user_long_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_long_buffer ufbxw_external_long_stream(ufbxw_scene *scene, ufbxw_long_stream_fn *fn, void *user, size_t count)
{
	ufbxwi_stream_fn stream_fn;
	stream_fn.long_fn = fn;
	ufbxw_buffer_id id = ufbxwi_create_stream_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_LONG, stream_fn, user, count);
	return ufbxwi_to_user_long_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_real_buffer ufbxw_create_real_buffer(ufbxw_scene *scene, size_t count)
{
	ufbxw_buffer_id id = ufbxwi_create_owned_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_REAL, count);
	return ufbxwi_to_user_real_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_real_buffer ufbxw_copy_real_array(ufbxw_scene *scene, const ufbxw_real *data, size_t count)
{
	ufbxw_buffer_id id = ufbxwi_create_copy_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_REAL, data, count);
	return ufbxwi_to_user_real_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_real_buffer ufbxw_view_real_array(ufbxw_scene *scene, const ufbxw_real *data, size_t count)
{
	ufbxw_buffer_id id = ufbxwi_create_external_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_REAL, data, count, UFBXWI_BUFFER_FLAG_TEMPORARY);
	return ufbxwi_to_user_real_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_real_buffer ufbxw_external_real_array(ufbxw_scene *scene, const ufbxw_real *data, size_t count)
{
	ufbxw_buffer_id id = ufbxwi_create_external_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_REAL, data, count, 0);
	return ufbxwi_to_user_real_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_real_buffer ufbxw_external_real_stream(ufbxw_scene *scene, ufbxw_real_stream_fn *fn, void *user, size_t count)
{
	ufbxwi_stream_fn stream_fn;
	stream_fn.real_fn = fn;
	ufbxw_buffer_id id = ufbxwi_create_stream_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_REAL, stream_fn, user, count);
	return ufbxwi_to_user_real_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_vec2_buffer ufbxw_create_vec2_buffer(ufbxw_scene *scene, size_t count)
{
	ufbxw_buffer_id id = ufbxwi_create_owned_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_VEC2, count);
	return ufbxwi_to_user_vec2_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_vec2_buffer ufbxw_copy_vec2_array(ufbxw_scene *scene, const ufbxw_vec2 *data, size_t count)
{
	ufbxw_buffer_id id = ufbxwi_create_copy_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_VEC2, data, count);
	return ufbxwi_to_user_vec2_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_vec2_buffer ufbxw_view_vec2_array(ufbxw_scene *scene, const ufbxw_vec2 *data, size_t count)
{
	ufbxw_buffer_id id = ufbxwi_create_external_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_VEC2, data, count, UFBXWI_BUFFER_FLAG_TEMPORARY);
	return ufbxwi_to_user_vec2_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_vec2_buffer ufbxw_external_vec2_array(ufbxw_scene *scene, const ufbxw_vec2 *data, size_t count)
{
	ufbxw_buffer_id id = ufbxwi_create_external_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_VEC2, data, count, 0);
	return ufbxwi_to_user_vec2_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_vec2_buffer ufbxw_external_vec2_stream(ufbxw_scene *scene, ufbxw_vec2_stream_fn *fn, void *user, size_t count)
{
	ufbxwi_stream_fn stream_fn;
	stream_fn.vec2_fn = fn;
	ufbxw_buffer_id id = ufbxwi_create_stream_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_VEC2, stream_fn, user, count);
	return ufbxwi_to_user_vec2_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_vec3_buffer ufbxw_create_vec3_buffer(ufbxw_scene *scene, size_t count)
{
	ufbxw_buffer_id id = ufbxwi_create_owned_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_VEC3, count);
	return ufbxwi_to_user_vec3_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_vec3_buffer ufbxw_copy_vec3_array(ufbxw_scene *scene, const ufbxw_vec3 *data, size_t count)
{
	ufbxw_buffer_id id = ufbxwi_create_copy_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_VEC3, data, count);
	return ufbxwi_to_user_vec3_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_vec3_buffer ufbxw_view_vec3_array(ufbxw_scene *scene, const ufbxw_vec3 *data, size_t count)
{
	ufbxw_buffer_id id = ufbxwi_create_external_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_VEC3, data, count, UFBXWI_BUFFER_FLAG_TEMPORARY);
	return ufbxwi_to_user_vec3_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_vec3_buffer ufbxw_external_vec3_array(ufbxw_scene *scene, const ufbxw_vec3 *data, size_t count)
{
	ufbxw_buffer_id id = ufbxwi_create_external_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_VEC3, data, count, 0);
	return ufbxwi_to_user_vec3_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_vec3_buffer ufbxw_external_vec3_stream(ufbxw_scene *scene, ufbxw_vec3_stream_fn *fn, void *user, size_t count)
{
	ufbxwi_stream_fn stream_fn;
	stream_fn.vec3_fn = fn;
	ufbxw_buffer_id id = ufbxwi_create_stream_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_VEC3, stream_fn, user, count);
	return ufbxwi_to_user_vec3_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_vec4_buffer ufbxw_create_vec4_buffer(ufbxw_scene* scene, size_t count)
{
	ufbxw_buffer_id id = ufbxwi_create_owned_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_VEC4, count);
	return ufbxwi_to_user_vec4_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_vec4_buffer ufbxw_copy_vec4_array(ufbxw_scene *scene, const ufbxw_vec4 *data, size_t count)
{
	ufbxw_buffer_id id = ufbxwi_create_copy_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_VEC4, data, count);
	return ufbxwi_to_user_vec4_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_vec4_buffer ufbxw_view_vec4_array(ufbxw_scene *scene, const ufbxw_vec4 *data, size_t count)
{
	ufbxw_buffer_id id = ufbxwi_create_external_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_VEC4, data, count, UFBXWI_BUFFER_FLAG_TEMPORARY);
	return ufbxwi_to_user_vec4_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_vec4_buffer ufbxw_external_vec4_array(ufbxw_scene *scene, const ufbxw_vec4 *data, size_t count)
{
	ufbxw_buffer_id id = ufbxwi_create_external_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_VEC4, data, count, 0);
	return ufbxwi_to_user_vec4_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_vec4_buffer ufbxw_external_vec4_stream(ufbxw_scene *scene, ufbxw_vec4_stream_fn *fn, void *user, size_t count)
{
	ufbxwi_stream_fn stream_fn;
	stream_fn.vec4_fn = fn;
	ufbxw_buffer_id id = ufbxwi_create_stream_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_VEC4, stream_fn, user, count);
	return ufbxwi_to_user_vec4_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_float_buffer ufbxw_create_float_buffer(ufbxw_scene* scene, size_t count)
{
	ufbxw_buffer_id id = ufbxwi_create_owned_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_FLOAT, count);
	return ufbxwi_to_user_float_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_float_buffer ufbxw_copy_float_array(ufbxw_scene *scene, const float *data, size_t count)
{
	ufbxw_buffer_id id = ufbxwi_create_copy_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_FLOAT, data, count);
	return ufbxwi_to_user_float_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_float_buffer ufbxw_view_float_array(ufbxw_scene *scene, const float *data, size_t count)
{
	ufbxw_buffer_id id = ufbxwi_create_external_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_FLOAT, data, count, UFBXWI_BUFFER_FLAG_TEMPORARY);
	return ufbxwi_to_user_float_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_float_buffer ufbxw_external_float_array(ufbxw_scene *scene, const float *data, size_t count)
{
	ufbxw_buffer_id id = ufbxwi_create_external_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_FLOAT, data, count, 0);
	return ufbxwi_to_user_float_buffer(&scene->buffers, id);
}

ufbxw_abi ufbxw_float_buffer ufbxw_external_float_stream(ufbxw_scene *scene, ufbxw_float_stream_fn *fn, void *user, size_t count)
{
	ufbxwi_stream_fn stream_fn;
	stream_fn.float_fn = fn;
	ufbxw_buffer_id id = ufbxwi_create_stream_buffer(&scene->buffers, UFBXWI_BUFFER_TYPE_FLOAT, stream_fn, user, count);
	return ufbxwi_to_user_float_buffer(&scene->buffers, id);
}

// TODO: Lock/unlock version for Rust
ufbxw_abi ufbxw_int_list ufbxw_edit_int_buffer(ufbxw_scene *scene, ufbxw_int_buffer buffer)
{
	ufbxw_assert(ufbxwi_buffer_id_type(buffer.id) == UFBXWI_BUFFER_TYPE_INT);
	ufbxwi_buffer *buf = ufbxwi_get_buffer(&scene->buffers, buffer.id);
	ufbxw_int_list result = { NULL, 0 };
	if (buf && buf->state == UFBXWI_BUFFER_STATE_OWNED) {
		result.data = buf->data.owned.data;
		result.count = buf->count;
	}
	return result;
}

ufbxw_abi ufbxw_long_list ufbxw_edit_long_buffer(ufbxw_scene *scene, ufbxw_long_buffer buffer)
{
	ufbxw_assert(ufbxwi_buffer_id_type(buffer.id) == UFBXWI_BUFFER_TYPE_LONG);
	ufbxwi_buffer *buf = ufbxwi_get_buffer(&scene->buffers, buffer.id);
	ufbxw_long_list result = { NULL, 0 };
	if (buf && buf->state == UFBXWI_BUFFER_STATE_OWNED) {
		result.data = buf->data.owned.data;
		result.count = buf->count;
	}
	return result;
}

ufbxw_abi ufbxw_real_list ufbxw_edit_real_buffer(ufbxw_scene *scene, ufbxw_real_buffer buffer)
{
	ufbxw_assert(ufbxwi_buffer_id_type(buffer.id) == UFBXWI_BUFFER_TYPE_REAL);
	ufbxwi_buffer *buf = ufbxwi_get_buffer(&scene->buffers, buffer.id);
	ufbxw_real_list result = { NULL, 0 };
	if (buf && buf->state == UFBXWI_BUFFER_STATE_OWNED) {
		result.data = buf->data.owned.data;
		result.count = buf->count;
	}
	return result;
}

ufbxw_abi ufbxw_vec2_list ufbxw_edit_vec2_buffer(ufbxw_scene *scene, ufbxw_vec2_buffer buffer)
{
	ufbxw_assert(ufbxwi_buffer_id_type(buffer.id) == UFBXWI_BUFFER_TYPE_VEC2);
	ufbxwi_buffer *buf = ufbxwi_get_buffer(&scene->buffers, buffer.id);
	ufbxw_vec2_list result = { NULL, 0 };
	if (buf && buf->state == UFBXWI_BUFFER_STATE_OWNED) {
		result.data = buf->data.owned.data;
		result.count = buf->count;
	}
	return result;
}

ufbxw_abi ufbxw_vec3_list ufbxw_edit_vec3_buffer(ufbxw_scene *scene, ufbxw_vec3_buffer buffer)
{
	ufbxw_assert(ufbxwi_buffer_id_type(buffer.id) == UFBXWI_BUFFER_TYPE_VEC3);
	ufbxwi_buffer *buf = ufbxwi_get_buffer(&scene->buffers, buffer.id);
	ufbxw_vec3_list result = { NULL, 0 };
	if (buf && buf->state == UFBXWI_BUFFER_STATE_OWNED) {
		result.data = buf->data.owned.data;
		result.count = buf->count;
	}
	return result;
}

ufbxw_abi ufbxw_vec4_list ufbxw_edit_vec4_buffer(ufbxw_scene* scene, ufbxw_vec4_buffer buffer)
{
	ufbxw_assert(ufbxwi_buffer_id_type(buffer.id) == UFBXWI_BUFFER_TYPE_VEC4);
	ufbxwi_buffer* buf = ufbxwi_get_buffer(&scene->buffers, buffer.id);
	ufbxw_vec4_list result = { NULL, 0 };
	if (buf && buf->state == UFBXWI_BUFFER_STATE_OWNED) {
		result.data = buf->data.owned.data;
		result.count = buf->count;
	}
	return result;
}

ufbxw_abi ufbxw_float_list ufbxw_edit_float_buffer(ufbxw_scene* scene, ufbxw_float_buffer buffer)
{
	ufbxw_assert(ufbxwi_buffer_id_type(buffer.id) == UFBXWI_BUFFER_TYPE_FLOAT);
	ufbxwi_buffer* buf = ufbxwi_get_buffer(&scene->buffers, buffer.id);
	ufbxw_float_list result = { NULL, 0 };
	if (buf && buf->state == UFBXWI_BUFFER_STATE_OWNED) {
		result.data = buf->data.owned.data;
		result.count = buf->count;
	}
	return result;
}

ufbxw_abi ufbxw_scene *ufbxw_create_scene(const ufbxw_scene_opts *opts)
{
	ufbxw_scene_opts default_opts;
	if (!opts) {
		memset(&default_opts, 0, sizeof(default_opts));
		opts = &default_opts;
	}
	ufbxw_assert(opts->_begin_zero == 0 && opts->_end_zero == 0);

	ufbxwi_error alloc_error;
	memset(&alloc_error, 0, sizeof(alloc_error));

	ufbxwi_allocator ator = { 0 };
	ator.ator = opts->allocator;
	ator.max_allocs = SIZE_MAX;
	ator.max_size = SIZE_MAX / 4;
	ator.error = &alloc_error;

	ufbxw_scene *scene = ufbxwi_alloc(&ator, ufbxw_scene, 1);
	if (!scene) return NULL;

	if (opts->max_allocations > 0) {
		ator.max_allocs = opts->max_allocations;
	}

	memset(scene, 0, sizeof(ufbxw_scene));
	scene->opts = *opts;
	ufbxwi_move_allocator(&scene->ator, &ator);

	ufbxwi_init_scene(scene);

	return scene;
}

ufbxw_abi void ufbxw_free_scene(ufbxw_scene *scene)
{
	ufbxwi_allocator ator;
	ufbxwi_move_allocator(&ator, &scene->ator);
	ufbxwi_free_allocator(&ator);
}

ufbxw_abi void ufbxw_set_error_callback(ufbxw_scene *scene, ufbxw_error_fn *fn, void *user)
{
	scene->error.error_fn = fn;
	scene->error.error_user = user;
}

ufbxw_abi bool ufbxw_get_error(ufbxw_scene *scene, ufbxw_error *error)
{
	if (scene->error.error.type != UFBXW_ERROR_NONE) {
		if (error) {
			*error = scene->error.error;
		}
		return true;
	} else {
		if (error) {
			error->type = UFBXW_ERROR_NONE;
			error->description[0] = '\0';
			error->description_length = 0;
		}
		return false;
	}
}

ufbxw_abi ufbxw_memory_stats ufbxw_get_memory_stats(ufbxw_scene *scene)
{
	ufbxw_memory_stats stats;
	stats.allocated_bytes = scene->ator.total_size;
	stats.allocation_count = scene->ator.num_allocs;
	stats.block_allocation_count = scene->ator.num_block_allocs;
	return stats;
}

ufbxw_abi ufbxw_id ufbxw_create_element(ufbxw_scene *scene, ufbxw_element_type type)
{
	return ufbxwi_create_element(scene, type, UFBXWI_TOKEN_NONE);
}

ufbxw_abi ufbxw_id ufbxw_create_element_ex(ufbxw_scene *scene, ufbxw_element_type type, const char *class_type)
{
	return ufbxw_create_element_ex_len(scene, type, class_type, strlen(class_type));
}

ufbxw_abi ufbxw_id ufbxw_create_element_ex_len(ufbxw_scene *scene, ufbxw_element_type type, const char *class_type, size_t class_type_len)
{
	ufbxwi_token class_type_token = ufbxwi_get_token(&scene->string_pool, class_type, class_type_len);
	return ufbxwi_create_element(scene, type, class_type_token);
}

ufbxw_abi void ufbxw_delete_element(ufbxw_scene *scene, ufbxw_id id)
{
	if (id == 0) return;

	ufbxwi_element_slot *slot = ufbxwi_get_element_slot(scene, id);
	ufbxwi_check_element(scene, id, slot);

	// TODO: This could be accelerated with a bit-mask of active connection types
	for (uint32_t i = 1; i < UFBXW_CONNECTION_TYPE_COUNT; i++) {
		ufbxw_connection_type conn_type = (ufbxw_connection_type)i;
		ufbxwi_disconnect_all_dst(scene, conn_type, id);
		ufbxwi_disconnect_all_src(scene, conn_type, id);
	}

	ufbxwi_free(&scene->ator, slot->element);

	uint32_t generation = ufbxwi_id_generation(slot->id);
	slot->id = ufbxwi_make_id(UFBXWI_ELEMENT_TYPE_NONE, generation, 0);
	slot->element = NULL;
	scene->num_elements--;

	if (generation < 0xffff) {
		// We can leak the index safely here if we fail to allocate
		uint32_t index = ufbxwi_id_index(id);
		ufbxwi_ignore(ufbxwi_list_push_copy(&scene->ator, &scene->free_element_ids, uint32_t, &index));
	}
}

ufbxw_abi size_t ufbxw_get_num_elements(ufbxw_scene *scene)
{
	return scene->num_elements;
}

ufbxw_abi size_t ufbxw_get_elements(const ufbxw_scene *scene, ufbxw_id *elements, size_t num_elements)
{
	size_t count = 0;
	size_t element_count = scene->elements.count;
	for (size_t index = 0; index < element_count; index++) {
		const ufbxwi_element_slot *slot = &scene->elements.data[index];
		if (ufbxwi_id_type(slot->id) != UFBXWI_ELEMENT_TYPE_NONE) {
			if (count >= num_elements) break;
			elements[count++] = slot->id;
		}
	}
	return count;
}

ufbxw_abi void ufbxw_set_name(ufbxw_scene *scene, ufbxw_id id, const char *name)
{
	ufbxw_set_name_len(scene, id, name, strlen(name));
}

ufbxw_abi void ufbxw_set_name_len(ufbxw_scene *scene, ufbxw_id id, const char *name, size_t name_len)
{
	ufbxwi_element *element = ufbxwi_get_element(scene, id);
	ufbxwi_check_element(scene, id, element);

	ufbxwi_intern_string(&scene->string_pool, &element->name, name, name_len);
}

ufbxw_abi ufbxw_string ufbxw_get_name(ufbxw_scene *scene, ufbxw_id id)
{
	ufbxwi_element *element = ufbxwi_get_element(scene, id);
	ufbxwi_check_element(scene, id, element, ufbxw_empty_string);

	return element->name;
}

ufbxw_abi void ufbxw_connect(ufbxw_scene *scene, ufbxw_id src, ufbxw_id dst)
{
	for (uint32_t i = 1; i < UFBXW_CONNECTION_TYPE_COUNT; i++) {
		ufbxw_connection_type conn_type = (ufbxw_connection_type)i;
		if (ufbxwi_connect(scene, conn_type, src, dst, 0)) break;
	}
}

ufbxw_abi void ufbxw_connect_prop(ufbxw_scene *scene, ufbxw_id src, const char *src_prop, ufbxw_id dst, const char *dst_prop)
{
	ufbxw_connect_prop_len(scene, src, src_prop, src_prop ? strlen(src_prop) : 0, dst, dst_prop, dst_prop ? strlen(dst_prop) : 0);
}

ufbxw_abi void ufbxw_connect_prop_len(ufbxw_scene *scene, ufbxw_id src, const char *src_prop, size_t src_prop_len, ufbxw_id dst, const char *dst_prop, size_t dst_prop_len)
{
	ufbxwi_token src_token = src_prop_len > 0 ? ufbxwi_intern_token(&scene->string_pool, src_prop, src_prop_len) : UFBXWI_TOKEN_NONE;
	ufbxwi_token dst_token = dst_prop_len > 0 ? ufbxwi_intern_token(&scene->string_pool, dst_prop, dst_prop_len) : UFBXWI_TOKEN_NONE;

	for (uint32_t i = 1; i < UFBXW_CONNECTION_TYPE_COUNT; i++) {
		ufbxw_connection_type conn_type = (ufbxw_connection_type)i;
		if (ufbxwi_connect_imp(scene, conn_type, src, dst, src_token, dst_token, 0)) break;
	}
}

ufbxw_abi void ufbxw_set_bool(ufbxw_scene *scene, ufbxw_id id, const char *prop, bool value)
{
	ufbxwi_set_prop(scene, id, prop, strlen(prop), &value, UFBXW_PROP_DATA_BOOL);
}

ufbxw_abi void ufbxw_set_int(ufbxw_scene *scene, ufbxw_id id, const char *prop, int32_t value)
{
	ufbxwi_set_prop(scene, id, prop, strlen(prop), &value, UFBXW_PROP_DATA_INT32);
}

ufbxw_abi void ufbxw_set_int64(ufbxw_scene *scene, ufbxw_id id, const char *prop, int64_t value)
{
	ufbxwi_set_prop(scene, id, prop, strlen(prop), &value, UFBXW_PROP_DATA_INT64);
}

ufbxw_abi void ufbxw_set_real(ufbxw_scene *scene, ufbxw_id id, const char *prop, ufbxw_real value)
{
	ufbxwi_set_prop(scene, id, prop, strlen(prop), &value, UFBXW_PROP_DATA_REAL);
}

ufbxw_abi void ufbxw_set_vec2(ufbxw_scene *scene, ufbxw_id id, const char *prop, ufbxw_vec2 value)
{
	ufbxwi_set_prop(scene, id, prop, strlen(prop), &value, UFBXW_PROP_DATA_VEC2);
}

ufbxw_abi void ufbxw_set_vec3(ufbxw_scene *scene, ufbxw_id id, const char *prop, ufbxw_vec3 value)
{
	ufbxwi_set_prop(scene, id, prop, strlen(prop), &value, UFBXW_PROP_DATA_VEC3);
}

ufbxw_abi void ufbxw_set_vec4(ufbxw_scene *scene, ufbxw_id id, const char *prop, ufbxw_vec4 value)
{
	ufbxwi_set_prop(scene, id, prop, strlen(prop), &value, UFBXW_PROP_DATA_VEC4);
}

ufbxw_abi void ufbxw_set_string(ufbxw_scene *scene, ufbxw_id id, const char *prop, const char *value)
{
	ufbxw_string str;
	if (!ufbxwi_intern_string(&scene->string_pool, &str, value, strlen(value))) return;
	ufbxwi_set_prop(scene, id, prop, strlen(prop), &str, UFBXW_PROP_DATA_STRING);
}

ufbxw_abi void ufbxw_add_bool(ufbxw_scene *scene, ufbxw_id id, const char *prop, ufbxw_prop_type type, bool value)
{
	ufbxwi_add_prop(scene, id, prop, strlen(prop), type, &value, UFBXW_PROP_DATA_BOOL);
}

ufbxw_abi void ufbxw_add_int(ufbxw_scene *scene, ufbxw_id id, const char *prop, ufbxw_prop_type type, int32_t value)
{
	ufbxwi_add_prop(scene, id, prop, strlen(prop), type, &value, UFBXW_PROP_DATA_INT32);
}

ufbxw_abi void ufbxw_add_int64(ufbxw_scene *scene, ufbxw_id id, const char *prop, ufbxw_prop_type type, int64_t value)
{
	ufbxwi_add_prop(scene, id, prop, strlen(prop), type, &value, UFBXW_PROP_DATA_INT64);
}

ufbxw_abi void ufbxw_add_real(ufbxw_scene *scene, ufbxw_id id, const char *prop, ufbxw_prop_type type, ufbxw_real value)
{
	ufbxwi_add_prop(scene, id, prop, strlen(prop), type, &value, UFBXW_PROP_DATA_REAL);
}

ufbxw_abi void ufbxw_add_vec2(ufbxw_scene *scene, ufbxw_id id, const char *prop, ufbxw_prop_type type, ufbxw_vec2 value)
{
	ufbxwi_add_prop(scene, id, prop, strlen(prop), type, &value, UFBXW_PROP_DATA_VEC2);
}

ufbxw_abi void ufbxw_add_vec3(ufbxw_scene *scene, ufbxw_id id, const char *prop, ufbxw_prop_type type, ufbxw_vec3 value)
{
	ufbxwi_add_prop(scene, id, prop, strlen(prop), type, &value, UFBXW_PROP_DATA_VEC3);
}

ufbxw_abi void ufbxw_add_vec4(ufbxw_scene *scene, ufbxw_id id, const char *prop, ufbxw_prop_type type, ufbxw_vec4 value)
{
	ufbxwi_add_prop(scene, id, prop, strlen(prop), type, &value, UFBXW_PROP_DATA_VEC4);
}

ufbxw_abi void ufbxw_add_string(ufbxw_scene *scene, ufbxw_id id, const char *prop, ufbxw_prop_type type, const char *value)
{
	ufbxw_string str;
	if (!ufbxwi_intern_string(&scene->string_pool, &str, value, strlen(value))) return;
	ufbxwi_add_prop(scene, id, prop, strlen(prop), type, &str, UFBXW_PROP_DATA_STRING);
}

ufbxw_abi bool ufbxw_get_bool(ufbxw_scene *scene, ufbxw_id id, const char *prop)
{
	bool ret;
	ufbxwi_get_prop(scene, id, prop, strlen(prop), &ret, UFBXW_PROP_DATA_BOOL);
	return ret;
}

ufbxw_abi int32_t ufbxw_get_int(ufbxw_scene *scene, ufbxw_id id, const char *prop)
{
	int32_t ret;
	ufbxwi_get_prop(scene, id, prop, strlen(prop), &ret, UFBXW_PROP_DATA_INT32);
	return ret;
}

ufbxw_abi int64_t ufbxw_get_int64(ufbxw_scene *scene, ufbxw_id id, const char *prop)
{
	int64_t ret;
	ufbxwi_get_prop(scene, id, prop, strlen(prop), &ret, UFBXW_PROP_DATA_INT64);
	return ret;
}

ufbxw_abi ufbxw_real ufbxw_get_real(ufbxw_scene *scene, ufbxw_id id, const char *prop)
{
	ufbxw_real ret;
	ufbxwi_get_prop(scene, id, prop, strlen(prop), &ret, UFBXW_PROP_DATA_REAL);
	return ret;
}

ufbxw_abi ufbxw_vec2 ufbxw_get_vec2(ufbxw_scene *scene, ufbxw_id id, const char *prop)
{
	ufbxw_vec2 ret;
	ufbxwi_get_prop(scene, id, prop, strlen(prop), &ret, UFBXW_PROP_DATA_VEC2);
	return ret;
}

ufbxw_abi ufbxw_vec3 ufbxw_get_vec3(ufbxw_scene *scene, ufbxw_id id, const char *prop)
{
	ufbxw_vec3 ret;
	ufbxwi_get_prop(scene, id, prop, strlen(prop), &ret, UFBXW_PROP_DATA_VEC3);
	return ret;
}

ufbxw_abi ufbxw_vec4 ufbxw_get_vec4(ufbxw_scene *scene, ufbxw_id id, const char *prop)
{
	ufbxw_vec4 ret;
	ufbxwi_get_prop(scene, id, prop, strlen(prop), &ret, UFBXW_PROP_DATA_VEC4);
	return ret;
}

ufbxw_abi ufbxw_string ufbxw_get_string(ufbxw_scene *scene, ufbxw_id id, const char *prop)
{
	ufbxw_string ret;
	ufbxwi_get_prop(scene, id, prop, strlen(prop), &ret, UFBXW_PROP_DATA_STRING);
	return ret;
}

ufbxw_abi ufbxw_anim_prop ufbxw_animate_prop(ufbxw_scene *scene, ufbxw_id id, const char *prop, ufbxw_anim_layer layer)
{
	return ufbxw_animate_prop_len(scene, id, prop, strlen(prop), layer);
}

ufbxw_abi ufbxw_anim_prop ufbxw_animate_prop_len(ufbxw_scene *scene, ufbxw_id id, const char *prop, size_t prop_len, ufbxw_anim_layer layer)
{
	ufbxwi_token token = ufbxwi_intern_token(&scene->string_pool, prop, prop_len);
	if (!token) return ufbxw_null_anim_prop;

	return ufbxwi_animate_prop(scene, id, token, layer);
}

ufbxw_abi ufbxw_node ufbxw_create_node(ufbxw_scene *scene)
{
	ufbxw_node node = { ufbxw_create_element(scene, UFBXW_ELEMENT_NODE) };
	return node;
}

ufbxw_abi ufbxw_node ufbxw_as_node(ufbxw_id id)
{
	ufbxw_node node = { ufbxwi_id_type(id) == UFBXW_ELEMENT_NODE ? id : 0 };
	return node;
}

ufbxw_abi size_t ufbxw_node_get_num_children(ufbxw_scene *scene, ufbxw_node node)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data, 0);
	return data->children.count;
}

ufbxw_abi ufbxw_node ufbxw_node_get_child(ufbxw_scene *scene, ufbxw_node node, size_t index)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data, ufbxw_null_node);
	ufbxwi_check_index(&scene->error, index, data->children.count, ufbxw_null_node);
	return data->children.data[index];
}

ufbxw_abi void ufbxw_node_set_translation(ufbxw_scene *scene, ufbxw_node node, ufbxw_vec3 translation)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data);
	data->lcl_translation = translation;
}

ufbxw_abi ufbxw_vec3 ufbxw_node_get_translation(ufbxw_scene *scene, ufbxw_node node)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data, ufbxw_zero_vec3);
	return data->lcl_translation;
}

ufbxw_abi void ufbxw_node_set_rotation(ufbxw_scene *scene, ufbxw_node node, ufbxw_vec3 rotation)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data);
	data->lcl_rotation = rotation;
}

ufbxw_abi ufbxw_vec3 ufbxw_node_get_rotation(ufbxw_scene *scene, ufbxw_node node)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data, ufbxw_zero_vec3);
	return data->lcl_rotation;
}

ufbxw_abi void ufbxw_node_set_scaling(ufbxw_scene *scene, ufbxw_node node, ufbxw_vec3 scaling)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data);
	data->lcl_scaling = scaling;
}

ufbxw_abi ufbxw_vec3 ufbxw_node_get_scaling(ufbxw_scene *scene, ufbxw_node node)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data, ufbxw_zero_vec3);
	return data->lcl_scaling;
}

ufbxw_abi void ufbxw_node_set_rotation_order(ufbxw_scene *scene, ufbxw_node node, ufbxw_rotation_order order)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data);
	data->rotation_order = (int32_t)order;
}

ufbxw_abi ufbxw_rotation_order ufbxw_node_get_rotation_order(ufbxw_scene *scene, ufbxw_node node)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data, UFBXW_ROTATION_ORDER_XYZ);
	return (ufbxw_rotation_order)data->rotation_order;
}

ufbxw_abi void ufbxw_node_set_rotation_quat(ufbxw_scene *scene, ufbxw_node node, ufbxw_quat rotation, ufbxw_rotation_order order)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data);
	data->lcl_rotation = ufbxw_quat_to_euler(rotation, order);
	data->rotation_order = (int32_t)order;
}

ufbxw_abi void ufbxw_node_set_inherit_type(ufbxw_scene *scene, ufbxw_node node, ufbxw_inherit_type order)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data);
	data->inherit_type = (int32_t)order;
}

ufbxw_abi ufbxw_inherit_type ufbxw_node_get_inherit_type(ufbxw_scene *scene, ufbxw_node node)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data, UFBXW_INHERIT_TYPE_NORMAL);
	return (ufbxw_inherit_type)data->inherit_type;
}

ufbxw_abi void ufbxw_node_set_pre_rotation(ufbxw_scene *scene, ufbxw_node node, ufbxw_vec3 rotation)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data);
	data->pre_rotation = rotation;
}

ufbxw_abi ufbxw_vec3 ufbxw_node_get_pre_rotation(ufbxw_scene *scene, ufbxw_node node)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data, ufbxw_zero_vec3);
	return data->pre_rotation;
}

ufbxw_abi void ufbxw_node_set_post_rotation(ufbxw_scene *scene, ufbxw_node node, ufbxw_vec3 rotation)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data);
	data->post_rotation = rotation;
}

ufbxw_abi ufbxw_vec3 ufbxw_node_get_post_rotation(ufbxw_scene *scene, ufbxw_node node)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data, ufbxw_zero_vec3);
	return data->post_rotation;
}

ufbxw_abi void ufbxw_node_set_rotation_offset(ufbxw_scene *scene, ufbxw_node node, ufbxw_vec3 rotation)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data);
	data->rotation_offset = rotation;
}

ufbxw_abi ufbxw_vec3 ufbxw_node_get_rotation_offset(ufbxw_scene *scene, ufbxw_node node)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data, ufbxw_zero_vec3);
	return data->rotation_offset;
}

ufbxw_abi void ufbxw_node_set_rotation_pivot(ufbxw_scene *scene, ufbxw_node node, ufbxw_vec3 rotation)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data);
	data->rotation_pivot = rotation;
}

ufbxw_abi ufbxw_vec3 ufbxw_node_get_rotation_pivot(ufbxw_scene *scene, ufbxw_node node)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data, ufbxw_zero_vec3);
	return data->rotation_pivot;
}

ufbxw_abi void ufbxw_node_set_scaling_offset(ufbxw_scene *scene, ufbxw_node node, ufbxw_vec3 scaling)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data);
	data->scaling_offset = scaling;
}

ufbxw_abi ufbxw_vec3 ufbxw_node_get_scaling_offset(ufbxw_scene *scene, ufbxw_node node)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data, ufbxw_zero_vec3);
	return data->scaling_offset;
}

ufbxw_abi void ufbxw_node_set_scaling_pivot(ufbxw_scene *scene, ufbxw_node node, ufbxw_vec3 scaling)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data);
	data->scaling_pivot = scaling;
}

ufbxw_abi ufbxw_vec3 ufbxw_node_get_scaling_pivot(ufbxw_scene *scene, ufbxw_node node)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data, ufbxw_zero_vec3);
	return data->scaling_pivot;
}

ufbxw_abi void ufbxw_node_set_geometric_translation(ufbxw_scene *scene, ufbxw_node node, ufbxw_vec3 geometric_translation)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data);
	data->geometric_translation = geometric_translation;
}

ufbxw_abi ufbxw_vec3 ufbxw_node_get_geometric_translation(ufbxw_scene *scene, ufbxw_node node)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data, ufbxw_zero_vec3);
	return data->geometric_translation;
}

ufbxw_abi void ufbxw_node_set_geometric_rotation(ufbxw_scene *scene, ufbxw_node node, ufbxw_vec3 geometric_rotation)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data);
	data->geometric_rotation = geometric_rotation;
}

ufbxw_abi ufbxw_vec3 ufbxw_node_get_geometric_rotation(ufbxw_scene *scene, ufbxw_node node)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data, ufbxw_zero_vec3);
	return data->geometric_rotation;
}

ufbxw_abi void ufbxw_node_set_geometric_scaling(ufbxw_scene *scene, ufbxw_node node, ufbxw_vec3 geometric_scaling)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data);
	data->geometric_scaling = geometric_scaling;
}

ufbxw_abi ufbxw_vec3 ufbxw_node_get_geometric_scaling(ufbxw_scene *scene, ufbxw_node node)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data, ufbxw_zero_vec3);
	return data->geometric_scaling;
}

ufbxw_abi void ufbxw_node_set_visibility(ufbxw_scene *scene, ufbxw_node node, bool visible)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data);
	data->visibility = visible ? 1.0f : 0.0f;
}

ufbxw_abi bool ufbxw_node_get_visibility(ufbxw_scene *scene, ufbxw_node node)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data, false);
	return data->visibility > 0.0f;
}

ufbxw_abi void ufbxw_node_set_visibility_inheritance(ufbxw_scene *scene, ufbxw_node node, bool inherit)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data);
	data->visibility_inheritance = inherit;
}

ufbxw_abi bool ufbxw_node_get_visibility_inheritance(ufbxw_scene *scene, ufbxw_node node)
{
	ufbxwi_node *data = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, data, false);
	return data->visibility_inheritance;
}

ufbxw_abi ufbxw_anim_prop ufbxw_node_animate_translation(ufbxw_scene *scene, ufbxw_node node, ufbxw_anim_layer layer)
{
	return ufbxwi_animate_prop(scene, node.id, UFBXWI_Lcl_Translation, layer);
}

ufbxw_abi ufbxw_anim_prop ufbxw_node_animate_rotation(ufbxw_scene *scene, ufbxw_node node, ufbxw_anim_layer layer)
{
	return ufbxwi_animate_prop(scene, node.id, UFBXWI_Lcl_Rotation, layer);
}

ufbxw_abi ufbxw_anim_prop ufbxw_node_animate_scaling(ufbxw_scene *scene, ufbxw_node node, ufbxw_anim_layer layer)
{
	return ufbxwi_animate_prop(scene, node.id, UFBXWI_Lcl_Scaling, layer);
}

ufbxw_abi ufbxw_transform ufbxw_node_get_local_transform(ufbxw_scene *scene, ufbxw_node node)
{
	ufbxw_transform t = { { 0,0,0 }, { 0,0,0,1 }, { 1,1,1 }};
	ufbxwi_node *nd = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, nd, t);

	// WorldTransform = ParentWorldTransform * T * Roff * Rp * Rpre * R * Rpost * Rp-1 * Soff * Sp * S * Sp-1
	// NOTE: Rpost is inverted (!) after converting from PostRotation Euler angles

	ufbxwi_sub_translate(&t, nd->scaling_pivot);
	ufbxwi_mul_scale(&t, nd->lcl_scaling);
	ufbxwi_add_translate(&t, nd->scaling_pivot);

	ufbxwi_add_translate(&t, nd->scaling_offset);

	ufbxwi_sub_translate(&t, nd->rotation_pivot);
	ufbxwi_mul_inv_rotate(&t, nd->post_rotation, UFBXW_ROTATION_ORDER_XYZ);
	ufbxwi_mul_rotate(&t, nd->lcl_rotation, (ufbxw_rotation_order)nd->rotation_order);
	ufbxwi_mul_rotate(&t, nd->pre_rotation, UFBXW_ROTATION_ORDER_XYZ);
	ufbxwi_add_translate(&t, nd->rotation_pivot);

	ufbxwi_add_translate(&t, nd->rotation_offset);

	ufbxwi_add_translate(&t, nd->lcl_translation);

	return t;
}

ufbxw_abi ufbxw_matrix ufbxw_node_get_global_transform(ufbxw_scene *scene, ufbxw_node node)
{
	size_t max_depth = scene->num_elements + 1;

	ufbxw_transform local_transform = ufbxw_node_get_local_transform(scene, node);
	ufbxw_matrix result = ufbxw_transform_to_matrix(&local_transform);

	// TODO: Handle inheritance modes
	size_t depth = 0;
	for (; depth < max_depth; depth++) {
		ufbxw_node parent = ufbxw_node_get_parent(scene, node);
		if (parent.id == 0) break;

		ufbxw_transform parent_transform = ufbxw_node_get_local_transform(scene, parent);
		ufbxw_matrix parent_matrix = ufbxw_transform_to_matrix(&parent_transform);
		result = ufbxwi_matrix_mul_affine(&parent_matrix, &result);

		node = parent;
	}
	if (depth == max_depth) {
		ufbxwi_fail(&scene->error, UFBXW_ERROR_CYCLICAL_PARENT, "cyclical parent hierarchy");
		return ufbxw_identity_matrix;
	}

	return result;
}

ufbxw_abi void ufbxw_node_set_attribute(ufbxw_scene *scene, ufbxw_node node, ufbxw_id attrib)
{
	ufbxwi_connect(scene, UFBXW_CONNECTION_NODE_ATTRIBUTE, attrib, node.id, UFBXWI_CONNECT_FLAG_DISCONNECT_DST);
}

ufbxw_abi void ufbxw_node_set_parent(ufbxw_scene *scene, ufbxw_node node, ufbxw_node parent)
{
	if (parent.id == ufbxw_null_id) {
		ufbxwi_disconnect_all_dst(scene, UFBXW_CONNECTION_NODE_PARENT, node.id);
	} else {
		ufbxwi_connect(scene, UFBXW_CONNECTION_NODE_PARENT, node.id, parent.id, UFBXWI_CONNECT_FLAG_DISCONNECT_SRC);
	}
}

ufbxw_abi ufbxw_node ufbxw_node_get_parent(ufbxw_scene *scene, ufbxw_node node)
{
	ufbxwi_node *nd = ufbxwi_get_node(scene, node);
	ufbxwi_check_element(scene, node.id, nd, ufbxw_null_node);
	return nd->parent;
}

ufbxw_abi ufbxw_mesh ufbxw_create_mesh(ufbxw_scene *scene)
{
	ufbxw_mesh mesh = { ufbxw_create_element(scene, UFBXW_ELEMENT_MESH) };
	return mesh;
}

ufbxw_abi ufbxw_mesh ufbxw_as_mesh(ufbxw_id id)
{
	ufbxw_mesh mesh = { ufbxwi_id_type(id) == UFBXW_ELEMENT_MESH ? id : 0 };
	return mesh;
}

ufbxw_abi void ufbxw_mesh_set_vertices(ufbxw_scene *scene, ufbxw_mesh mesh, ufbxw_vec3_buffer vertices)
{
	ufbxwi_mesh *md = ufbxwi_get_mesh(scene, mesh);
	ufbxwi_check_element(scene, mesh.id, md);

	ufbxwi_set_buffer_from_user(&scene->buffers, &md->vertices.id, vertices.id);
}

ufbxw_abi void ufbxw_mesh_set_triangles(ufbxw_scene *scene, ufbxw_mesh mesh, ufbxw_int_buffer indices)
{
	size_t index_count = ufbxwi_get_buffer_size(&scene->buffers, indices.id);
	ufbxw_assert(index_count % 3 == 0);

	size_t tri_count = index_count / 3;
	ufbxw_int_buffer face_offsets = ufbxw_external_int_stream(scene, &ufbxwi_stream_triangle_faces, NULL, tri_count + 1);
	ufbxwi_check(face_offsets.id);

	ufbxwi_mesh *md = ufbxwi_get_mesh(scene, mesh);
	ufbxwi_check_element(scene, mesh.id, md);

	ufbxwi_set_buffer(&scene->buffers, &md->polygon_vertex_index.id, 0);
	ufbxwi_set_buffer_from_user(&scene->buffers, &md->vertex_indices.id, indices.id);
	ufbxwi_set_buffer(&scene->buffers, &md->face_offsets.id, face_offsets.id);
}

ufbxw_abi void ufbxw_mesh_set_polygons(ufbxw_scene *scene, ufbxw_mesh mesh, ufbxw_int_buffer indices, ufbxw_int_buffer face_offsets)
{
	ufbxwi_mesh *md = ufbxwi_get_mesh(scene, mesh);
	ufbxwi_check_element(scene, mesh.id, md);

	ufbxwi_set_buffer(&scene->buffers, &md->polygon_vertex_index.id, 0);
	ufbxwi_set_buffer_from_user(&scene->buffers, &md->vertex_indices.id, indices.id);
	ufbxwi_set_buffer_from_user(&scene->buffers, &md->face_offsets.id, face_offsets.id);
}

ufbxw_abi void ufbxw_mesh_set_fbx_polygon_vertex_index(ufbxw_scene *scene, ufbxw_mesh mesh, ufbxw_int_buffer polygon_vertex_index)
{
	ufbxwi_mesh *md = ufbxwi_get_mesh(scene, mesh);
	ufbxwi_check_element(scene, mesh.id, md);

	ufbxwi_set_buffer(&scene->buffers, &md->vertex_indices.id, 0);
	ufbxwi_set_buffer(&scene->buffers, &md->face_offsets.id, 0);
	ufbxwi_set_buffer_from_user(&scene->buffers, &md->polygon_vertex_index.id, polygon_vertex_index.id);
}

ufbxw_abi void ufbxw_mesh_set_normals(ufbxw_scene *scene, ufbxw_mesh mesh, ufbxw_vec3_buffer normals, ufbxw_attribute_mapping mapping)
{
	ufbxw_mesh_attribute_desc desc = { 0 };
	desc.values = normals.id;
	desc.mapping = mapping;
	ufbxw_mesh_set_attribute(scene, mesh, UFBXW_MESH_ATTRIBUTE_NORMAL, 0, &desc);
}

ufbxw_abi ufbxw_light ufbxw_create_light(ufbxw_scene *scene, ufbxw_node node)
{
	ufbxw_light light = { ufbxw_create_element(scene, UFBXW_ELEMENT_LIGHT) };
	if (node.id) {
		ufbxwi_connect(scene, UFBXW_CONNECTION_NODE_ATTRIBUTE, light.id, node.id, 0);
	}
	return light;
}

ufbxw_abi void ufbxw_light_set_color(ufbxw_scene *scene, ufbxw_light light, ufbxw_vec3 color)
{
	ufbxwi_light *ld = ufbxwi_get_light(scene, light);
	ufbxwi_check_element(scene, light.id, ld);
	ld->color = color;
}

ufbxw_abi ufbxw_vec3 ufbxw_light_get_color(ufbxw_scene *scene, ufbxw_light light)
{
	ufbxwi_light *ld = ufbxwi_get_light(scene, light);
	ufbxwi_check_element(scene, light.id, ld, ufbxw_zero_vec3);
	return ld->color;
}

ufbxw_abi void ufbxw_light_set_intensity(ufbxw_scene *scene, ufbxw_light light, ufbxw_real intensity)
{
	ufbxwi_light *ld = ufbxwi_get_light(scene, light);
	ufbxwi_check_element(scene, light.id, ld);
	ld->intensity = intensity;
}

ufbxw_abi ufbxw_real ufbxw_light_get_intensity(ufbxw_scene *scene, ufbxw_light light)
{
	ufbxwi_light *ld = ufbxwi_get_light(scene, light);
	ufbxwi_check_element(scene, light.id, ld, 0.0f);
	return ld->intensity;
}

ufbxw_abi void ufbxw_light_set_decay(ufbxw_scene *scene, ufbxw_light light, ufbxw_light_decay decay)
{
	ufbxwi_light *ld = ufbxwi_get_light(scene, light);
	ufbxwi_check_element(scene, light.id, ld);
	ld->decay_type = (int32_t)decay;
}

ufbxw_abi ufbxw_light_decay ufbxw_light_get_decay(ufbxw_scene *scene, ufbxw_light light)
{
	ufbxwi_light *ld = ufbxwi_get_light(scene, light);
	ufbxwi_check_element(scene, light.id, ld, (ufbxw_light_decay)0);
	return (ufbxw_light_decay)ld->decay_type;
}

ufbxw_abi void ufbxw_light_set_type(ufbxw_scene *scene, ufbxw_light light, ufbxw_light_type type)
{
	ufbxwi_light *ld = ufbxwi_get_light(scene, light);
	ufbxwi_check_element(scene, light.id, ld);
	ld->light_type = (int32_t)type;
}

ufbxw_abi ufbxw_light_type ufbxw_light_get_type(ufbxw_scene *scene, ufbxw_light light)
{
	ufbxwi_light *ld = ufbxwi_get_light(scene, light);
	ufbxwi_check_element(scene, light.id, ld, (ufbxw_light_type)0);
	return (ufbxw_light_type)ld->light_type;
}

ufbxw_abi void ufbxw_light_set_inner_angle(ufbxw_scene *scene, ufbxw_light light, ufbxw_real value)
{
	ufbxwi_light *ld = ufbxwi_get_light(scene, light);
	ufbxwi_check_element(scene, light.id, ld);
	ld->inner_angle = value;
}

ufbxw_abi ufbxw_real ufbxw_light_get_inner_angle(ufbxw_scene *scene, ufbxw_light light)
{
	ufbxwi_light *ld = ufbxwi_get_light(scene, light);
	ufbxwi_check_element(scene, light.id, ld, 0.0f);
	return ld->inner_angle;
}

ufbxw_abi void ufbxw_light_set_outer_angle(ufbxw_scene *scene, ufbxw_light light, ufbxw_real value)
{
	ufbxwi_light *ld = ufbxwi_get_light(scene, light);
	ufbxwi_check_element(scene, light.id, ld);
	ld->outer_angle = value;
}

ufbxw_abi ufbxw_real ufbxw_light_get_outer_angle(ufbxw_scene *scene, ufbxw_light light)
{
	ufbxwi_light *ld = ufbxwi_get_light(scene, light);
	ufbxwi_check_element(scene, light.id, ld, 0.0f);
	return ld->outer_angle;
}

ufbxw_abi ufbxw_camera ufbxw_create_camera(ufbxw_scene *scene, ufbxw_node node)
{
	ufbxw_camera camera = { ufbxw_create_element(scene, UFBXW_ELEMENT_CAMERA) };
	if (node.id) {
		ufbxwi_connect(scene, UFBXW_CONNECTION_NODE_ATTRIBUTE, camera.id, node.id, 0);
	}
	return camera;
}

ufbxw_abi void ufbxw_mesh_set_uvs(ufbxw_scene *scene, ufbxw_mesh mesh, int32_t set, ufbxw_vec2_buffer uvs, ufbxw_attribute_mapping mapping)
{
	ufbxw_mesh_attribute_desc desc = { 0 };
	desc.values = uvs.id;
	desc.mapping = mapping;
	desc.generate_indices = true;
	ufbxw_mesh_set_attribute(scene, mesh, UFBXW_MESH_ATTRIBUTE_UV, set, &desc);
}

ufbxw_abi void ufbxw_mesh_set_uvs_indexed(ufbxw_scene *scene, ufbxw_mesh mesh, int32_t set, ufbxw_vec2_buffer uvs, ufbxw_int_buffer indices, ufbxw_attribute_mapping mapping)
{
	ufbxw_mesh_attribute_desc desc = { 0 };
	desc.values = uvs.id;
	desc.indices = indices.id;
	desc.mapping = mapping;
	ufbxw_mesh_set_attribute(scene, mesh, UFBXW_MESH_ATTRIBUTE_UV, set, &desc);
}

ufbxw_abi void ufbxw_mesh_set_tangents(ufbxw_scene *scene, ufbxw_mesh mesh, int32_t set, ufbxw_vec3_buffer tangents, ufbxw_attribute_mapping mapping)
{
	ufbxw_mesh_attribute_desc desc = { 0 };
	desc.values = tangents.id;
	desc.mapping = mapping;
	ufbxw_mesh_set_attribute(scene, mesh, UFBXW_MESH_ATTRIBUTE_TANGENT, set, &desc);
}

ufbxw_abi void ufbxw_mesh_set_binormals(ufbxw_scene *scene, ufbxw_mesh mesh, int32_t set, ufbxw_vec3_buffer binormals, ufbxw_attribute_mapping mapping)
{
	ufbxw_mesh_attribute_desc desc = { 0 };
	desc.values = binormals.id;
	desc.mapping = mapping;
	ufbxw_mesh_set_attribute(scene, mesh, UFBXW_MESH_ATTRIBUTE_BINORMAL, set, &desc);
}

ufbxw_abi void ufbxw_mesh_set_colors(ufbxw_scene *scene, ufbxw_mesh mesh, int32_t set, ufbxw_vec4_buffer colors, ufbxw_attribute_mapping mapping)
{
	ufbxw_mesh_attribute_desc desc = { 0 };
	desc.values = colors.id;
	desc.mapping = mapping;
	desc.generate_indices = true;
	ufbxw_mesh_set_attribute(scene, mesh, UFBXW_MESH_ATTRIBUTE_COLOR, set, &desc);
}

ufbxw_abi void ufbxw_mesh_set_colors_indexed(ufbxw_scene *scene, ufbxw_mesh mesh, int32_t set, ufbxw_vec4_buffer colors, ufbxw_int_buffer indices, ufbxw_attribute_mapping mapping)
{
	ufbxw_mesh_attribute_desc desc = { 0 };
	desc.values = colors.id;
	desc.indices = indices.id;
	desc.mapping = mapping;
	ufbxw_mesh_set_attribute(scene, mesh, UFBXW_MESH_ATTRIBUTE_COLOR, set, &desc);
}

ufbxw_abi void ufbxw_mesh_set_single_material(ufbxw_scene *scene, ufbxw_mesh mesh, int32_t material_index)
{
	ufbxw_mesh_attribute_desc desc = { 0 };
	if (material_index == 0) {
		desc.indices = ufbxw_external_int_array(scene, (const int32_t*)&ufbxwi_prop_default_data.zero.int32_t, 1).id;
	} else {
		desc.indices = ufbxw_copy_int_array(scene, &material_index, 1).id;
	}
	desc.mapping = UFBXW_ATTRIBUTE_MAPPING_ALL_SAME;
	ufbxw_mesh_set_attribute(scene, mesh, UFBXW_MESH_ATTRIBUTE_MATERIAL, 0, &desc);
}

ufbxw_abi void ufbxw_mesh_set_face_material(ufbxw_scene *scene, ufbxw_mesh mesh, ufbxw_int_buffer material_indices)
{
	ufbxw_mesh_attribute_desc desc = { 0 };
	desc.indices = material_indices.id;
	desc.mapping = UFBXW_ATTRIBUTE_MAPPING_POLYGON;
	ufbxw_mesh_set_attribute(scene, mesh, UFBXW_MESH_ATTRIBUTE_MATERIAL, 0, &desc);
}

ufbxw_abi void ufbxw_mesh_set_fbx_edges(ufbxw_scene *scene, ufbxw_mesh mesh, ufbxw_int_buffer edges)
{
	ufbxwi_mesh *md = ufbxwi_get_mesh(scene, mesh);
	ufbxwi_check_element(scene, mesh.id, md);

	ufbxwi_set_buffer_from_user(&scene->buffers, &md->edges.id, edges.id);
}

ufbxw_abi void ufbxw_mesh_set_attribute(ufbxw_scene *scene, ufbxw_mesh mesh, ufbxw_mesh_attribute attribute, int32_t set, const ufbxw_mesh_attribute_desc *desc)
{
	ufbxw_assert(desc);

	ufbxwi_mesh_attribute *attrib = ufbxwi_edit_mesh_attribute(scene, mesh, attribute, set);
	if (!attrib) return;

	const ufbxwi_mesh_attribute_info *attrib_info = &ufbxwi_mesh_attribute_infos[attrib->attribute];

	ufbxw_mesh_attribute_desc d = *desc;
	if (d.generate_indices && !d.indices) {
		ufbxwi_generate_indices(scene, &d);
	}

	// TODO: Check forbid/require indices, or bypass

	if ((attrib_info->flags & UFBXWI_MESH_ATTRIBUTE_FLAG_HAS_VALUES_W) != 0 && !d.values_w) {
		size_t values_count = ufbxwi_get_buffer_size(&scene->buffers, d.values);
		ufbxw_real_buffer values_w = ufbxw_external_real_stream(scene, &ufbxwi_stream_real_one, NULL, values_count);
		d.values_w = values_w.id;
	}

	attrib->mapping = desc->mapping;
	ufbxwi_intern_string_str(&scene->string_pool, &attrib->name, d.name);
	ufbxwi_set_buffer_from_user(&scene->buffers, &attrib->values, d.values);
	ufbxwi_set_buffer_from_user(&scene->buffers, &attrib->indices, d.indices);
	ufbxwi_set_buffer_from_user(&scene->buffers, &attrib->values_w, d.values_w);
}

ufbxw_abi void ufbxw_mesh_set_attribute_name(ufbxw_scene *scene, ufbxw_mesh mesh, ufbxw_mesh_attribute attribute, int32_t set, const char *name)
{
	ufbxw_mesh_set_attribute_name_len(scene, mesh, attribute, set, name, strlen(name));
}

ufbxw_abi void ufbxw_mesh_set_attribute_name_len(ufbxw_scene *scene, ufbxw_mesh mesh, ufbxw_mesh_attribute attribute, int32_t set, const char *name, size_t name_len)
{
	ufbxwi_mesh_attribute *attrib = ufbxwi_edit_mesh_attribute(scene, mesh, attribute, set);
	if (!attrib) return;
	ufbxwi_intern_string(&scene->string_pool, &attrib->name, name, name_len);
}

ufbxw_vec3_buffer ufbxw_mesh_get_vertices(ufbxw_scene *scene, ufbxw_mesh mesh)
{
	ufbxwi_mesh *md = ufbxwi_get_mesh(scene, mesh);
	ufbxwi_check_element(scene, mesh.id, md, ufbxwi_empty_vec3_buffer);
	return md->vertices;
}

ufbxw_abi void ufbxw_mesh_add_instance(ufbxw_scene *scene, ufbxw_mesh mesh, ufbxw_node node)
{
	ufbxwi_connect(scene, UFBXW_CONNECTION_NODE_ATTRIBUTE, mesh.id, node.id, 0);
}

ufbxw_abi ufbxw_abi ufbxw_skin_deformer ufbxw_create_skin_deformer(ufbxw_scene *scene, ufbxw_mesh mesh)
{
	ufbxw_skin_deformer skin = { ufbxw_create_element(scene, UFBXW_ELEMENT_SKIN_DEFORMER) };
	if (mesh.id) {
		ufbxwi_connect(scene, UFBXW_CONNECTION_MESH_DEFORMER, skin.id, mesh.id, 0);
	}

	return skin;
}

ufbxw_abi void ufbxw_skin_deformer_add_mesh(ufbxw_scene *scene, ufbxw_skin_deformer skin, ufbxw_mesh mesh)
{
	ufbxwi_connect(scene, UFBXW_CONNECTION_MESH_DEFORMER, skin.id, mesh.id, 0);
}

ufbxw_abi void ufbxw_skin_deformer_set_skinning_type(ufbxw_scene *scene, ufbxw_skin_deformer skin, ufbxw_skinning_type type)
{
	ufbxwi_skin_deformer *sd = ufbxwi_get_skin_deformer(scene, skin);
	ufbxwi_check_element(scene, skin.id, sd);
	sd->skinning_type = type;
}

ufbxw_abi ufbxw_skinning_type ufbxw_skin_deformer_get_skinning_type(ufbxw_scene *scene, ufbxw_skin_deformer skin)
{
	ufbxwi_skin_deformer *sd = ufbxwi_get_skin_deformer(scene, skin);
	ufbxwi_check_element(scene, skin.id, sd, UFBXW_SKINNING_TYPE_RIGID);
	return sd->skinning_type;
}

ufbxw_abi void ufbxw_skin_deformer_set_mesh_bind_transform(ufbxw_scene *scene, ufbxw_skin_deformer skin, ufbxw_matrix matrix)
{
	ufbxwi_skin_deformer *sd = ufbxwi_get_skin_deformer(scene, skin);
	ufbxwi_check_element(scene, skin.id, sd);

	sd->has_mesh_bind_transform = true;
	sd->mesh_bind_transform = matrix;
}

ufbxw_abi void ufbxw_skin_deformer_set_bind_pose(ufbxw_scene *scene, ufbxw_skin_deformer skin, ufbxw_bind_pose pose)
{
	ufbxwi_skin_deformer *sd = ufbxwi_get_skin_deformer(scene, skin);
	ufbxwi_check_element(scene, skin.id, sd);
	sd->bind_pose = pose;
}

ufbxw_abi ufbxw_skin_cluster ufbxw_create_skin_cluster(ufbxw_scene *scene, ufbxw_skin_deformer skin, ufbxw_node node)
{
	ufbxw_skin_cluster cluster = { ufbxw_create_element(scene, UFBXW_ELEMENT_SKIN_CLUSTER) };
	if (skin.id) {
		ufbxwi_connect(scene, UFBXW_CONNECTION_SKIN_CLUSTER, cluster.id, skin.id, 0);
	}
	if (node.id) {
		ufbxwi_connect(scene, UFBXW_CONNECTION_SKIN_CLUSTER_NODE, node.id, cluster.id, 0);
	}
	return cluster;
}

ufbxw_abi void ufbxw_skin_cluster_set_deformer(ufbxw_scene *scene, ufbxw_skin_cluster cluster, ufbxw_skin_deformer skin)
{
	ufbxwi_connect(scene, UFBXW_CONNECTION_SKIN_CLUSTER, cluster.id, skin.id, UFBXWI_CONNECT_FLAG_DISCONNECT_SRC);
}

ufbxw_abi void ufbxw_skin_cluster_set_node(ufbxw_scene *scene, ufbxw_skin_cluster cluster, ufbxw_node node)
{
	ufbxwi_connect(scene, UFBXW_CONNECTION_SKIN_CLUSTER_NODE, node.id, cluster.id, UFBXWI_CONNECT_FLAG_DISCONNECT_DST);
}

ufbxw_abi void ufbxw_skin_cluster_set_weights(ufbxw_scene *scene, ufbxw_skin_cluster cluster, ufbxw_int_buffer indices, ufbxw_real_buffer weights)
{
	ufbxwi_skin_cluster *sc = ufbxwi_get_skin_cluster(scene, cluster);
	ufbxwi_check_element(scene, cluster.id, sc);

	size_t index_count = ufbxwi_get_buffer_size(&scene->buffers, indices.id);
	size_t weights_count = ufbxwi_get_buffer_size(&scene->buffers, weights.id);
	ufbxw_assert(index_count == weights_count);

	ufbxwi_set_buffer_from_user(&scene->buffers, &sc->indices.id, indices.id);
	ufbxwi_set_buffer_from_user(&scene->buffers, &sc->weights.id, weights.id);
}

ufbxw_abi void ufbxw_skin_cluster_set_transform(ufbxw_scene *scene, ufbxw_skin_cluster cluster, ufbxw_matrix matrix)
{
	ufbxwi_skin_cluster *sc = ufbxwi_get_skin_cluster(scene, cluster);
	ufbxwi_check_element(scene, cluster.id, sc);
	sc->transform = matrix;
}

ufbxw_abi void ufbxw_skin_cluster_set_link_transform(ufbxw_scene *scene, ufbxw_skin_cluster cluster, ufbxw_matrix matrix)
{
	ufbxwi_skin_cluster *sc = ufbxwi_get_skin_cluster(scene, cluster);
	ufbxwi_check_element(scene, cluster.id, sc);
	sc->transform_link = matrix;
}

ufbxw_abi ufbxw_blend_deformer ufbxw_create_blend_deformer(ufbxw_scene *scene, ufbxw_mesh mesh)
{
	ufbxw_blend_deformer deformer = { ufbxw_create_element(scene, UFBXW_ELEMENT_BLEND_DEFORMER) };
	if (mesh.id) {
		ufbxwi_connect(scene, UFBXW_CONNECTION_MESH_DEFORMER, deformer.id, mesh.id, 0);
	}
	return deformer;
}

ufbxw_abi void ufbxw_blend_deformer_add_mesh(ufbxw_scene *scene, ufbxw_blend_deformer deformer, ufbxw_mesh mesh)
{
	ufbxwi_connect(scene, UFBXW_CONNECTION_MESH_DEFORMER, deformer.id, mesh.id, 0);
}

ufbxw_abi ufbxw_blend_channel ufbxw_create_blend_channel(ufbxw_scene *scene, ufbxw_blend_deformer deformer)
{
	ufbxw_blend_channel channel = { ufbxw_create_element(scene, UFBXW_ELEMENT_BLEND_CHANNEL) };
	if (deformer.id) {
		ufbxwi_connect(scene, UFBXW_CONNECTION_BLEND_CHANNEL, channel.id, deformer.id, 0);
	}
	return channel;
}

ufbxw_abi void ufbxw_blend_channel_set_deformer(ufbxw_scene *scene, ufbxw_blend_channel channel, ufbxw_blend_deformer deformer)
{
	// TODO: Allow these to be null to disconnect?
	ufbxwi_connect(scene, UFBXW_CONNECTION_BLEND_CHANNEL, channel.id, deformer.id, UFBXWI_CONNECT_FLAG_DISCONNECT_SRC);
}

ufbxw_abi void ufbxw_blend_channel_set_shape(ufbxw_scene *scene, ufbxw_blend_channel channel, ufbxw_blend_shape shape)
{
	// TODO: Test and hack RootGroup to prepare scene?
	ufbxwi_disconnect_all_src(scene, UFBXW_CONNECTION_BLEND_SHAPE, channel.id);
	ufbxw_blend_channel_add_shape(scene, channel, shape, 100.0);
}

ufbxw_abi void ufbxw_blend_channel_add_shape(ufbxw_scene *scene, ufbxw_blend_channel channel, ufbxw_blend_shape shape, ufbxw_real target_weight)
{
	// Manually add connections
	ufbxwi_blend_channel *bc = ufbxwi_get_blend_channel(scene, channel);
	ufbxwi_blend_shape *bs = ufbxwi_get_blend_shape(scene, shape);
	ufbxwi_check_element(scene, channel.id, bc);
	ufbxwi_check_element(scene, shape.id, bs);

	ufbxwi_check(ufbxwi_list_push_zero(&scene->ator, &bc->blend_shapes, ufbxwi_blend_shape_conn));
	ufbxwi_check(ufbxwi_list_push_copy(&scene->ator, &bs->blend_channels, ufbxw_id, &channel.id));

	ufbxwi_blend_shape_conn *shapes = bc->blend_shapes.data;
	size_t prev_count = bc->blend_shapes.count - 1;

	// Insert to the right position
	size_t pos = 0;
	while (pos < prev_count && shapes[pos].target_weight <= target_weight) {
		pos++;
	}

	memmove(shapes + pos, shapes + pos + 1, (prev_count - pos) * sizeof(ufbxwi_blend_shape_conn));
	shapes[pos].shape = shape;
	shapes[pos].target_weight = target_weight;
}

ufbxw_abi void ufbxw_blend_channel_set_weight(ufbxw_scene *scene, ufbxw_blend_channel channel, ufbxw_real weight)
{
	ufbxwi_blend_channel *bc = ufbxwi_get_blend_channel(scene, channel);
	ufbxwi_check_element(scene, channel.id, bc);
	bc->deform_percent = weight;
}

ufbxw_abi ufbxw_real ufbxw_blend_channel_get_weight(ufbxw_scene *scene, ufbxw_blend_channel channel)
{
	ufbxwi_blend_channel *bc = ufbxwi_get_blend_channel(scene, channel);
	ufbxwi_check_element(scene, channel.id, bc, 0.0f);
	return bc->deform_percent;
}

ufbxw_abi ufbxw_blend_shape ufbxw_create_blend_shape(ufbxw_scene *scene)
{
	ufbxw_blend_shape shape = { ufbxw_create_element(scene, UFBXW_ELEMENT_BLEND_SHAPE) };
	return shape;
}

ufbxw_abi void ufbxw_blend_shape_set_offsets(ufbxw_scene *scene, ufbxw_blend_shape shape, ufbxw_int_buffer indices, ufbxw_vec3_buffer offsets)
{
	ufbxwi_blend_shape *bs = ufbxwi_get_blend_shape(scene, shape);
	ufbxwi_check_element(scene, shape.id, bs);

	size_t index_count = ufbxwi_get_buffer_size(&scene->buffers, indices.id);
	size_t offset_count = ufbxwi_get_buffer_size(&scene->buffers, offsets.id);
	ufbxw_assert(index_count == offset_count);

	ufbxwi_set_buffer_from_user(&scene->buffers, &bs->indices.id, indices.id);
	ufbxwi_set_buffer_from_user(&scene->buffers, &bs->vertices.id, offsets.id);
}

ufbxw_abi void ufbxw_blend_shape_set_normals(ufbxw_scene *scene, ufbxw_blend_shape shape, ufbxw_vec3_buffer normals)
{
	ufbxwi_blend_shape *bs = ufbxwi_get_blend_shape(scene, shape);
	ufbxwi_check_element(scene, shape.id, bs);

	ufbxwi_set_buffer_from_user(&scene->buffers, &bs->normals.id, normals.id);
}

ufbxw_abi ufbxw_bind_pose ufbxw_create_bind_pose(ufbxw_scene *scene)
{
	ufbxw_bind_pose pose = { ufbxw_create_element(scene, UFBXW_ELEMENT_BIND_POSE) };
	return pose;
}

ufbxw_abi void ufbxw_bind_pose_add_node(ufbxw_scene *scene, ufbxw_bind_pose pose, ufbxw_node node, ufbxw_matrix matrix)
{
	ufbxwi_bind_pose *bp = ufbxwi_get_bind_pose(scene, pose);
	ufbxwi_check_element(scene, pose.id, bp);

	// TODO: Search existing?
	ufbxwi_pose_node *pose_node = ufbxwi_list_push_uninit(&scene->ator, &bp->pose_nodes, ufbxwi_pose_node);
	ufbxwi_check(pose_node);

	pose_node->node = node;
	pose_node->matrix = matrix;
}

ufbxw_abi ufbxw_anim_stack ufbxw_get_default_anim_stack(ufbxw_scene *scene)
{
	return scene->default_anim_stack;
}

ufbxw_abi ufbxw_anim_stack ufbxw_create_anim_stack(ufbxw_scene *scene)
{
	ufbxw_anim_stack stack = { ufbxw_create_element(scene, UFBXW_ELEMENT_ANIM_STACK) };
	return stack;
}

ufbxw_abi ufbxw_anim_layer ufbxw_anim_stack_get_layer(ufbxw_scene *scene, ufbxw_anim_stack stack, size_t index)
{
	ufbxwi_anim_stack *s = ufbxwi_get_anim_stack(scene, stack);
	if (s && index < s->layers.count) {
		ufbxw_anim_layer layer = { s->layers.data[index] };
		return layer;
	} else {
		return ufbxw_null_anim_layer;
	}
}

ufbxw_abi void ufbxw_anim_stack_set_time_range(ufbxw_scene *scene, ufbxw_anim_stack stack, ufbxw_ktime time_begin, ufbxw_ktime time_end)
{
	ufbxwi_anim_stack *s = ufbxwi_get_anim_stack(scene, stack);
	if (s) {
		s->local_start = time_begin;
		s->local_stop = time_end;
	}
}

ufbxw_abi void ufbxw_anim_stack_set_reference_time_range(ufbxw_scene *scene, ufbxw_anim_stack stack, ufbxw_ktime time_begin, ufbxw_ktime time_end)
{
	ufbxwi_anim_stack *s = ufbxwi_get_anim_stack(scene, stack);
	if (s) {
		s->reference_start = time_begin;
		s->reference_stop = time_end;
	}
}

ufbxw_abi ufbxw_ktime_range ufbxw_anim_stack_get_time_range(ufbxw_scene *scene, ufbxw_anim_stack stack)
{
	ufbxw_ktime_range range = { 0, 0 };
	ufbxwi_anim_stack *s = ufbxwi_get_anim_stack(scene, stack);
	if (s) {
		range.begin = s->local_start;
		range.end = s->local_stop;
	}
	return range;
}

ufbxw_abi ufbxw_ktime_range ufbxw_anim_stack_get_reference_time_range(ufbxw_scene *scene, ufbxw_anim_stack stack)
{
	ufbxw_ktime_range range = { 0, 0 };
	ufbxwi_anim_stack *s = ufbxwi_get_anim_stack(scene, stack);
	if (s) {
		range.begin = s->reference_start;
		range.end = s->reference_stop;
	}
	return range;
}

ufbxw_abi void ufbxw_set_active_anim_stack(ufbxw_scene *scene, ufbxw_anim_stack stack)
{
	scene->active_anim_stack = stack;
}

ufbxw_abi ufbxw_anim_stack ufbxw_get_active_anim_stack(const ufbxw_scene *scene)
{
	return scene->active_anim_stack;
}

ufbxw_abi ufbxw_anim_layer ufbxw_get_default_anim_layer(ufbxw_scene *scene)
{
	return scene->default_anim_layer;
}

ufbxw_abi ufbxw_anim_layer ufbxw_create_anim_layer(ufbxw_scene *scene, ufbxw_anim_stack stack)
{
	ufbxw_anim_layer layer = { ufbxw_create_element(scene, UFBXW_ELEMENT_ANIM_LAYER) };

	if (stack.id != 0) {
		ufbxw_anim_layer_set_stack(scene, layer, stack);
	}

	return layer;
}

ufbxw_abi void ufbxw_anim_layer_set_weight(ufbxw_scene *scene, ufbxw_anim_layer layer, ufbxw_real weight)
{
	ufbxwi_anim_layer *l = ufbxwi_get_anim_layer(scene, layer);
	l->weight = weight;
}

ufbxw_abi void ufbxw_anim_layer_set_stack(ufbxw_scene *scene, ufbxw_anim_layer layer, ufbxw_anim_stack stack)
{
	ufbxwi_connect(scene, UFBXW_CONNECTION_ANIM_LAYER_STACK, layer.id, stack.id, UFBXWI_CONNECT_FLAG_DISCONNECT_SRC);
}

ufbxw_abi ufbxw_anim_curve ufbxw_anim_get_curve(ufbxw_scene *scene, ufbxw_anim_prop anim, size_t index)
{
	ufbxwi_anim_prop *p = ufbxwi_get_anim_prop(scene, anim);
	if (p && index < p->curves.count) {
		ufbxw_anim_curve curve = { p->curves.data[index].id };
		return curve;
	} else {
		return ufbxw_null_anim_curve;
	}
}

ufbxw_abi void ufbxw_anim_set_default_value(ufbxw_scene *scene, ufbxw_anim_prop anim, size_t index, ufbxw_real value)
{
	ufbxwi_anim_prop *p = ufbxwi_get_anim_prop(scene, anim);
	if (index >= 4) {
		ufbxwi_failf(&scene->error, UFBXW_ERROR_INDEX_OUT_OF_BOUNDS, "index (%zu) out of bounds (4)", index);
		return;
	}
	p->defaults[index] = value;
}

ufbxw_abi void ufbxw_anim_add_keyframe_real(ufbxw_scene *scene, ufbxw_anim_prop anim, ufbxw_ktime time, ufbxw_real value, uint32_t type)
{
	ufbxwi_anim_prop *ap = ufbxwi_get_anim_prop(scene, anim);
	if (!ap || ap->curves.count < 1) return;

	ufbxwi_conn *curves = ap->curves.data;
	ufbxw_anim_curve_add_keyframe(scene, ufbxwi_assert_anim_curve(curves[0].id), time, value, type);
}

ufbxw_abi void ufbxw_anim_add_keyframe_vec2(ufbxw_scene *scene, ufbxw_anim_prop anim, ufbxw_ktime time, ufbxw_vec2 value, uint32_t type)
{
	ufbxwi_anim_prop *ap = ufbxwi_get_anim_prop(scene, anim);
	if (!ap || ap->curves.count < 2) return;

	ufbxwi_conn *curves = ap->curves.data;
	ufbxw_anim_curve_add_keyframe(scene, ufbxwi_assert_anim_curve(curves[0].id), time, value.x, type);
	ufbxw_anim_curve_add_keyframe(scene, ufbxwi_assert_anim_curve(curves[1].id), time, value.y, type);
}

ufbxw_abi void ufbxw_anim_add_keyframe_vec3(ufbxw_scene *scene, ufbxw_anim_prop anim, ufbxw_ktime time, ufbxw_vec3 value, uint32_t type)
{
	ufbxwi_anim_prop *ap = ufbxwi_get_anim_prop(scene, anim);
	if (!ap || ap->curves.count < 3) return;

	ufbxwi_conn *curves = ap->curves.data;
	ufbxw_anim_curve_add_keyframe(scene, ufbxwi_assert_anim_curve(curves[0].id), time, value.x, type);
	ufbxw_anim_curve_add_keyframe(scene, ufbxwi_assert_anim_curve(curves[1].id), time, value.y, type);
	ufbxw_anim_curve_add_keyframe(scene, ufbxwi_assert_anim_curve(curves[2].id), time, value.z, type);
}

ufbxw_abi void ufbxw_anim_add_keyframe_real_key(ufbxw_scene *scene, ufbxw_anim_prop anim, ufbxw_keyframe_real key)
{
	ufbxwi_anim_prop *ap = ufbxwi_get_anim_prop(scene, anim);
	if (!ap || ap->curves.count < 1) return;

	ufbxwi_conn *curves = ap->curves.data;
	ufbxw_anim_curve_add_keyframe_key(scene, ufbxwi_assert_anim_curve(curves[0].id), key);
}

ufbxw_abi void ufbxw_anim_add_keyframe_vec2_key(ufbxw_scene *scene, ufbxw_anim_prop anim, ufbxw_keyframe_vec2 key)
{
	ufbxwi_anim_prop *ap = ufbxwi_get_anim_prop(scene, anim);
	if (!ap || ap->curves.count < 2) return;

	ufbxwi_conn *curves = ap->curves.data;

	{
		ufbxw_keyframe_real k = { key.time, key.value.x, key.flags, key.weight_left, key.weight_right, key.slope_left.x, key.slope_right.x };
		ufbxw_anim_curve_add_keyframe_key(scene, ufbxwi_assert_anim_curve(curves[0].id), k);
	}

	{
		ufbxw_keyframe_real k = { key.time, key.value.y, key.flags, key.weight_left, key.weight_right, key.slope_left.y, key.slope_right.y };
		ufbxw_anim_curve_add_keyframe_key(scene, ufbxwi_assert_anim_curve(curves[1].id), k);
	}
}

ufbxw_abi void ufbxw_anim_add_keyframe_vec3_key(ufbxw_scene *scene, ufbxw_anim_prop anim, ufbxw_keyframe_vec3 key)
{
	ufbxwi_anim_prop *ap = ufbxwi_get_anim_prop(scene, anim);
	if (!ap || ap->curves.count < 2) return;

	ufbxwi_conn *curves = ap->curves.data;

	{
		ufbxw_keyframe_real k = { key.time, key.value.x, key.flags, key.weight_left, key.weight_right, key.slope_left.x, key.slope_right.x };
		ufbxw_anim_curve_add_keyframe_key(scene, ufbxwi_assert_anim_curve(curves[0].id), k);
	}

	{
		ufbxw_keyframe_real k = { key.time, key.value.y, key.flags, key.weight_left, key.weight_right, key.slope_left.y, key.slope_right.y };
		ufbxw_anim_curve_add_keyframe_key(scene, ufbxwi_assert_anim_curve(curves[1].id), k);
	}

	{
		ufbxw_keyframe_real k = { key.time, key.value.z, key.flags, key.weight_left, key.weight_right, key.slope_left.z, key.slope_right.z };
		ufbxw_anim_curve_add_keyframe_key(scene, ufbxwi_assert_anim_curve(curves[2].id), k);
	}
}

ufbxw_abi void ufbxw_anim_finish_keyframes(ufbxw_scene *scene, ufbxw_anim_prop anim)
{
	ufbxwi_anim_prop *ap = ufbxwi_get_anim_prop(scene, anim);
	if (!ap) return;
	for (size_t i = 0; i < ap->curves.count; i++) {
		ufbxw_anim_curve_finish_keyframes(scene, ufbxwi_assert_anim_curve(ap->curves.data[i].id));
	}
}

ufbxw_abi void ufbxw_anim_set_layer(ufbxw_scene *scene, ufbxw_anim_prop anim, ufbxw_anim_layer layer)
{
	ufbxwi_connect(scene, UFBXW_CONNECTION_ANIM_PROP_LAYER, anim.id, layer.id, UFBXWI_CONNECT_FLAG_DISCONNECT_SRC);
}

ufbxw_abi void ufbxw_anim_curve_add_keyframe(ufbxw_scene *scene, ufbxw_anim_curve curve, ufbxw_ktime time, ufbxw_real value, uint32_t type)
{
	ufbxwi_anim_curve *c = ufbxwi_get_anim_curve(scene, curve);
	if (!c) return;

	// TODO: Pass by pointer to some implementation function
	// TODO: Assert that type does not depend on user tangents
	ufbxw_keyframe_real key = { time, value, type };
	ufbxw_anim_curve_add_keyframe_key(scene, curve, key);
}

ufbxw_abi void ufbxw_anim_curve_add_keyframe_key(ufbxw_scene *scene, ufbxw_anim_curve curve, ufbxw_keyframe_real key)
{
	ufbxwi_anim_curve *c = ufbxwi_get_anim_curve(scene, curve);
	if (!c) return;

	if (c->key_times.count > 0 && key.time < c->key_times.data[c->key_times.count - 1]) {
		c->keys_out_of_order = true;
	}

	float value = (float)key.value;
	ufbxwi_anim_key_attr key_attr;
	key_attr.flags = key.flags;

	if (key.flags & UFBXW_KEYFRAME_WEIGHTED_LEFT) {
		key_attr.weight_left = (float)key.weight_left;
	} else {
		key_attr.weight_left = (float)(1.0 / 3.0);
	}

	if (key.flags & UFBXW_KEYFRAME_WEIGHTED_RIGHT) {
		key_attr.weight_right = (float)key.weight_right;
	} else {
		key_attr.weight_right = (float)(1.0 / 3.0);
	}

	if (key.flags & UFBXW_KEYFRAME_TANGENT_USER) {
		key_attr.slope_left = (float)key.slope_left;
		key_attr.slope_right = (float)key.slope_right;
	} else {
		key_attr.slope_left = 0.0f;
		key_attr.slope_right = 0.0f;
	}

	uint32_t key_attr_index = (uint32_t)c->key_attr_data.count;
	if (c->key_attr_data.count > 0 && !memcmp(&c->key_attr_data.data[c->key_attr_data.count - 1], &key_attr, sizeof(ufbxwi_anim_key_attr))) {
		key_attr_index -= 1;
	} else {
		ufbxwi_ignore(ufbxwi_list_push_copy(&scene->ator, &c->key_attr_data, ufbxwi_anim_key_attr, &key_attr));
	}

	ufbxwi_ignore(ufbxwi_list_push_copy(&scene->ator, &c->key_times, ufbxw_ktime, &key.time));
	ufbxwi_ignore(ufbxwi_list_push_copy(&scene->ator, &c->key_values, float, &value));
	ufbxwi_ignore(ufbxwi_list_push_copy(&scene->ator, &c->key_attr_indices, uint32_t, &key_attr_index));
}

ufbxw_abi void ufbxw_anim_curve_finish_keyframes(ufbxw_scene *scene, ufbxw_anim_curve curve)
{
	ufbxwi_anim_curve *c = ufbxwi_get_anim_curve(scene, curve);
	if (!c) return;
	if (!c->keys_out_of_order) return;

	c->keys_out_of_order = false;
	ufbxwi_sort_anim_kefyrames(scene, c);
}

ufbxw_abi void ufbxw_anim_curve_set_data(ufbxw_scene *scene, ufbxw_anim_curve curve, const ufbxw_anim_curve_data_desc *data)
{
	ufbxwi_anim_curve *c = ufbxwi_get_anim_curve(scene, curve);
	if (!c) return;

	ufbxwi_list_free(&scene->ator, &c->key_times);
	ufbxwi_list_free(&scene->ator, &c->key_values);
	ufbxwi_list_free(&scene->ator, &c->key_attr_indices);
	ufbxwi_list_free(&scene->ator, &c->key_attr_data);
	c->keys_out_of_order = false;
	c->data_in_buffers = true;

	size_t num_times = ufbxwi_get_buffer_size(&scene->buffers, data->key_times.id);
	size_t num_values = ufbxwi_get_buffer_size(&scene->buffers, data->key_times.id);

	if (data->key_times.id && data->key_values.id) {
		ufbxw_assert(num_times == num_values);
		ufbxwi_set_buffer_from_user(&scene->buffers, &c->buffer_key_times.id, data->key_times.id);
		ufbxwi_set_buffer_from_user(&scene->buffers, &c->buffer_key_values.id, data->key_values.id);
	} else {
		ufbxwi_set_buffer_from_user(&scene->buffers, &c->buffer_key_times.id, 0);
		ufbxwi_set_buffer_from_user(&scene->buffers, &c->buffer_key_values.id, 0);
	}

	if (data->fbx_attr_refcounts.id && data->fbx_attr_flags.id && data->fbx_attr_data.id) {
		ufbxwi_set_buffer_from_user(&scene->buffers, &c->buffer_attr_refcounts.id, data->fbx_attr_refcounts.id);
		ufbxwi_set_buffer_from_user(&scene->buffers, &c->buffer_attr_flags.id, data->fbx_attr_data.id);
		ufbxwi_set_buffer_from_user(&scene->buffers, &c->buffer_attr_data.id, data->fbx_attr_data.id);
	} else {
		int32_t refcount = (int32_t)num_times;
		int32_t flags = (int32_t)ufbxwi_get_key_flags(data->key_flags, data->key_flags);
		ufbxwi_key_attr attr;
		attr.slope_right = 0.0f;
		attr.slope_next_left = 0.0f;
		attr.packed_weight = ufbxwi_pack_weight((float)(1.0 / 3.0)) | ufbxwi_pack_weight((float)(1.0 / 3.0)) << 16;
		attr.packed_velocity = 0;

		ufbxwi_set_buffer_from_user(&scene->buffers, &c->buffer_attr_refcounts.id, ufbxw_copy_int_array(scene, &refcount, 1).id);
		ufbxwi_set_buffer_from_user(&scene->buffers, &c->buffer_attr_flags.id, ufbxw_copy_int_array(scene, &flags, 1).id);
		ufbxwi_set_buffer_from_user(&scene->buffers, &c->buffer_attr_data.id, ufbxw_copy_float_array(scene, (float*)&attr, 4).id);
	}
}

ufbxw_abi ufbxw_id ufbxw_get_scene_info_id(ufbxw_scene *scene)
{
	ufbxwi_for_list(ufbxwi_element_slot, slot, scene->elements) {
		if (ufbxwi_id_type(slot->id) == UFBXW_ELEMENT_SCENE_INFO) {
			return slot->id;
		}
	}
	return ufbxw_null_id;
}

ufbxw_abi ufbxw_id ufbxw_get_global_settings_id(ufbxw_scene *scene)
{
	ufbxwi_for_list(ufbxwi_element_slot, slot, scene->elements) {
		if (ufbxwi_id_type(slot->id) == UFBXW_ELEMENT_GLOBAL_SETTINGS) {
			return slot->id;
		}
	}
	return ufbxw_null_id;
}

ufbxw_abi ufbxw_id ufbxw_get_template_id(ufbxw_scene *scene, ufbxw_element_type type)
{
	uint32_t type_id = ufbxwi_find_element_type_id(scene, type, UFBXWI_TOKEN_NONE);
	if (type_id == ~0u) return ufbxw_null_id;
	ufbxwi_element_type *et = &scene->element_types.data[type_id];
	ufbxwi_check(ufbxwi_init_element_type(scene, et), 0);
	return et->template_id;
}

// -- Pre-saving

ufbxw_abi_data_def const ufbxw_prepare_opts ufbxw_default_prepare_opts = {
	true, true, true, true, true, true,
};

ufbxw_abi void ufbxw_prepare_scene(ufbxw_scene *scene, const ufbxw_prepare_opts *opts)
{
	ufbxw_assert(scene);
	if (!opts) {
		opts = &ufbxw_default_prepare_opts;
	}

	ufbxwi_prepare_scene(scene, opts);
}

ufbxw_abi void ufbxw_validate_scene(const ufbxw_scene *scene)
{
	// TODO: Check and report errors somehow
	// - Missing materials
	// - Cyclical node hierarchies
	// - Unnamed blend channels
	// - Unnamed blend shapes
}

// -- Streams

ufbxw_abi bool ufbxw_open_file_write(ufbxw_write_stream *stream, const char *path, size_t path_len, ufbxw_error *error)
{
	ufbxwi_error err = { 0 };
	if (!ufbxwi_open_file_write(stream, path, path_len, &err)) {
		if (error) {
			*error = err.error;
		}
		return false;
	}

	if (error) {
		error->type = UFBXW_ERROR_NONE;
	}

	return true;
}

// -- IO

ufbxw_abi bool ufbxw_save_file(ufbxw_scene *scene, const char *path, const ufbxw_save_opts *opts, ufbxw_error *error)
{
	return ufbxw_save_file_len(scene, path, strlen(path), opts, error);
}

ufbxw_abi bool ufbxw_save_file_len(ufbxw_scene *scene, const char *path, size_t path_len, const ufbxw_save_opts *opts, ufbxw_error *error)
{
	ufbxw_write_stream stream;
	if (!ufbxw_open_file_write(&stream, path, path_len, error)) {
		return false;
	}

	return ufbxw_save_stream(scene, &stream, opts, error);
}

ufbxw_abi bool ufbxw_save_stream(ufbxw_scene *scene, ufbxw_write_stream *stream, const ufbxw_save_opts *opts, ufbxw_error *error)
{
	ufbxwi_save_context sc = { 0 };
	sc.scene = scene;
	if (opts) {
		sc.opts = *opts;
	}

	ufbxwi_save_imp(&sc, stream);
	ufbxwi_write_queue_free(&sc.write_queue);

	ufbxwi_task_queue_free(&sc.task_queue, &sc.main_thread_ctx);
	ufbxwi_destroy_save_thread_context(&sc, &sc.main_thread_ctx);

	ufbxwi_thread_pool_free(&sc.thread_pool);

	ufbxwi_free_allocator(&sc.ator);

	if (sc.error.error.type != UFBXW_ERROR_NONE) {
		if (error) {
			*error = sc.error.error;
		}
		return false;
	}

	if (error) {
		error->type = UFBXW_ERROR_NONE;
	}

	return true;
}

ufbxw_unsafe ufbxw_abi ufbxw_task_run_result ufbxw_thread_pool_try_run_tasks(ufbxw_thread_pool_context ctx, uint32_t thread_id_hint, size_t max_count)
{
	ufbxwi_task_queue *tq = (ufbxwi_task_queue*)ctx;
	return ufbxwi_task_queue_run_task(tq, UFBXWI_RUN_TASK_TRY, thread_id_hint, max_count);
}

ufbxw_unsafe ufbxw_abi ufbxw_task_run_result ufbxw_thread_pool_blocking_run_tasks(ufbxw_thread_pool_context ctx, uint32_t thread_id_hint, size_t max_count)
{
	ufbxwi_task_queue *tq = (ufbxwi_task_queue*)ctx;
	return ufbxwi_task_queue_run_task(tq, UFBXWI_RUN_TASK_BLOCKING, thread_id_hint, max_count);
}

ufbxw_unsafe ufbxw_abi void ufbxw_thread_pool_set_user_ptr(ufbxw_thread_pool_context ctx, void *user_ptr)
{
	ufbxwi_task_queue *tq = (ufbxwi_task_queue*)ctx;
	tq->user_ptr = user_ptr;
}

ufbxw_unsafe ufbxw_abi void *ufbxw_thread_pool_get_user_ptr(ufbxw_thread_pool_context ctx)
{
	ufbxwi_task_queue *tq = (ufbxwi_task_queue*)ctx;
	return tq->user_ptr;
}

ufbxw_abi ufbxwi_noinline ufbxw_matrix ufbxw_transform_to_matrix(const ufbxw_transform *t)
{
	ufbxw_assert(t);
	if (!t) return ufbxw_identity_matrix;

	ufbxw_quat q = t->rotation;
	ufbxw_real sx = 2.0f * t->scale.x, sy = 2.0f * t->scale.y, sz = 2.0f * t->scale.z;
	ufbxw_real xx = q.x*q.x, xy = q.x*q.y, xz = q.x*q.z, xw = q.x*q.w;
	ufbxw_real yy = q.y*q.y, yz = q.y*q.z, yw = q.y*q.w;
	ufbxw_real zz = q.z*q.z, zw = q.z*q.w;
	ufbxw_matrix m;
	m.m00 = sx * (- yy - zz + 0.5f);
	m.m10 = sx * (+ xy + zw);
	m.m20 = sx * (- yw + xz);
	m.m01 = sy * (- zw + xy);
	m.m11 = sy * (- xx - zz + 0.5f);
	m.m21 = sy * (+ xw + yz);
	m.m02 = sz * (+ xz + yw);
	m.m12 = sz * (- xw + yz);
	m.m22 = sz * (- xx - yy + 0.5f);
	m.m03 = t->translation.x;
	m.m13 = t->translation.y;
	m.m23 = t->translation.z;
	return m;
}

ufbxw_abi ufbxw_vec3 ufbxw_quat_rotate_vec3(ufbxw_quat q, ufbxw_vec3 v)
{
	ufbxw_real xy = q.x*v.y - q.y*v.x;
	ufbxw_real xz = q.x*v.z - q.z*v.x;
	ufbxw_real yz = q.y*v.z - q.z*v.y;
	ufbxw_vec3 r;
	r.x = 2.0f * (+ q.w*yz + q.y*xy + q.z*xz) + v.x;
	r.y = 2.0f * (- q.x*xy - q.w*xz + q.z*yz) + v.y;
	r.z = 2.0f * (- q.x*xz - q.y*yz + q.w*xy) + v.z;
	return r;
}

ufbxw_abi ufbxwi_noinline ufbxw_quat ufbxw_euler_to_quat(ufbxw_vec3 v, ufbxw_rotation_order order)
{
	double vx = v.x * (UFBXWI_DEG_TO_RAD_DOUBLE * 0.5);
	double vy = v.y * (UFBXWI_DEG_TO_RAD_DOUBLE * 0.5);
	double vz = v.z * (UFBXWI_DEG_TO_RAD_DOUBLE * 0.5);
	double cx = ufbxw_cos(vx), sx = ufbxw_sin(vx);
	double cy = ufbxw_cos(vy), sy = ufbxw_sin(vy);
	double cz = ufbxw_cos(vz), sz = ufbxw_sin(vz);
	ufbxw_quat q;

	// Generated by `misc/gen_rotation_order.py`
	switch (order) {
	case UFBXW_ROTATION_ORDER_XYZ:
		q.x = (ufbxw_real)(-cx*sy*sz + cy*cz*sx);
		q.y = (ufbxw_real)(cx*cz*sy + cy*sx*sz);
		q.z = (ufbxw_real)(cx*cy*sz - cz*sx*sy);
		q.w = (ufbxw_real)(cx*cy*cz + sx*sy*sz);
		break;
	case UFBXW_ROTATION_ORDER_XZY:
		q.x = (ufbxw_real)(cx*sy*sz + cy*cz*sx);
		q.y = (ufbxw_real)(cx*cz*sy + cy*sx*sz);
		q.z = (ufbxw_real)(cx*cy*sz - cz*sx*sy);
		q.w = (ufbxw_real)(cx*cy*cz - sx*sy*sz);
		break;
	case UFBXW_ROTATION_ORDER_YZX:
		q.x = (ufbxw_real)(-cx*sy*sz + cy*cz*sx);
		q.y = (ufbxw_real)(cx*cz*sy - cy*sx*sz);
		q.z = (ufbxw_real)(cx*cy*sz + cz*sx*sy);
		q.w = (ufbxw_real)(cx*cy*cz + sx*sy*sz);
		break;
	case UFBXW_ROTATION_ORDER_YXZ:
		q.x = (ufbxw_real)(-cx*sy*sz + cy*cz*sx);
		q.y = (ufbxw_real)(cx*cz*sy + cy*sx*sz);
		q.z = (ufbxw_real)(cx*cy*sz + cz*sx*sy);
		q.w = (ufbxw_real)(cx*cy*cz - sx*sy*sz);
		break;
	case UFBXW_ROTATION_ORDER_ZXY:
		q.x = (ufbxw_real)(cx*sy*sz + cy*cz*sx);
		q.y = (ufbxw_real)(cx*cz*sy - cy*sx*sz);
		q.z = (ufbxw_real)(cx*cy*sz - cz*sx*sy);
		q.w = (ufbxw_real)(cx*cy*cz + sx*sy*sz);
		break;
	case UFBXW_ROTATION_ORDER_ZYX:
		q.x = (ufbxw_real)(cx*sy*sz + cy*cz*sx);
		q.y = (ufbxw_real)(cx*cz*sy - cy*sx*sz);
		q.z = (ufbxw_real)(cx*cy*sz + cz*sx*sy);
		q.w = (ufbxw_real)(cx*cy*cz - sx*sy*sz);
		break;
	default:
		q.x = q.y = q.z = 0.0f; q.w = 1.0f;
		break;
	}

	return q;
}

ufbxw_abi ufbxwi_noinline ufbxw_vec3 ufbxw_quat_to_euler(ufbxw_quat q, ufbxw_rotation_order order)
{
	// TODO: Derive these rigorously
	#if defined(UFBXW_REAL_IS_FLOAT)
		const double eps = 0.9999999;
	#else
		const double eps = 0.999999999;
	#endif

	double vx, vy, vz;
	double t;

	double qx = q.x, qy = q.y, qz = q.z, qw = q.w;

	// Generated by `misc/gen_quat_to_euler.py`
	switch (order) {
	case UFBXW_ROTATION_ORDER_XYZ:
		t = 2.0f*(qw*qy - qx*qz);
		if (ufbxw_fabs(t) < eps) {
			vy = ufbxw_asin(t);
			vz = ufbxw_atan2(2.0f*(qw*qz + qx*qy), 2.0f*(qw*qw + qx*qx) - 1.0f);
			vx = -ufbxw_atan2(-2.0f*(qw*qx + qy*qz), 2.0f*(qw*qw + qz*qz) - 1.0f);
		} else {
			vy = ufbxw_copysign(UFBXWI_DPI*0.5, t);
			vz = ufbxw_atan2(-2.0f*t*(qw*qx - qy*qz), t*(2.0f*qw*qy + 2.0f*qx*qz));
			vx = 0.0f;
		}
		break;
	case UFBXW_ROTATION_ORDER_XZY:
		t = 2.0f*(qw*qz + qx*qy);
		if (ufbxw_fabs(t) < eps) {
			vz = ufbxw_asin(t);
			vy = ufbxw_atan2(2.0f*(qw*qy - qx*qz), 2.0f*(qw*qw + qx*qx) - 1.0f);
			vx = -ufbxw_atan2(-2.0f*(qw*qx - qy*qz), 2.0f*(qw*qw + qy*qy) - 1.0f);
		} else {
			vz = ufbxw_copysign(UFBXWI_DPI*0.5, t);
			vy = ufbxw_atan2(2.0f*t*(qw*qx + qy*qz), -t*(2.0f*qx*qy - 2.0f*qw*qz));
			vx = 0.0f;
		}
		break;
	case UFBXW_ROTATION_ORDER_YZX:
		t = 2.0f*(qw*qz - qx*qy);
		if (ufbxw_fabs(t) < eps) {
			vz = ufbxw_asin(t);
			vx = ufbxw_atan2(2.0f*(qw*qx + qy*qz), 2.0f*(qw*qw + qy*qy) - 1.0f);
			vy = -ufbxw_atan2(-2.0f*(qw*qy + qx*qz), 2.0f*(qw*qw + qx*qx) - 1.0f);
		} else {
			vz = ufbxw_copysign(UFBXWI_DPI*0.5, t);
			vx = ufbxw_atan2(-2.0f*t*(qw*qy - qx*qz), t*(2.0f*qw*qz + 2.0f*qx*qy));
			vy = 0.0f;
		}
		break;
	case UFBXW_ROTATION_ORDER_YXZ:
		t = 2.0f*(qw*qx + qy*qz);
		if (ufbxw_fabs(t) < eps) {
			vx = ufbxw_asin(t);
			vz = ufbxw_atan2(2.0f*(qw*qz - qx*qy), 2.0f*(qw*qw + qy*qy) - 1.0f);
			vy = -ufbxw_atan2(-2.0f*(qw*qy - qx*qz), 2.0f*(qw*qw + qz*qz) - 1.0f);
		} else {
			vx = ufbxw_copysign(UFBXWI_DPI*0.5, t);
			vz = ufbxw_atan2(2.0f*t*(qw*qy + qx*qz), -t*(2.0f*qy*qz - 2.0f*qw*qx));
			vy = 0.0f;
		}
		break;
	case UFBXW_ROTATION_ORDER_ZXY:
		t = 2.0f*(qw*qx - qy*qz);
		if (ufbxw_fabs(t) < eps) {
			vx = ufbxw_asin(t);
			vy = ufbxw_atan2(2.0f*(qw*qy + qx*qz), 2.0f*(qw*qw + qz*qz) - 1.0f);
			vz = -ufbxw_atan2(-2.0f*(qw*qz + qx*qy), 2.0f*(qw*qw + qy*qy) - 1.0f);
		} else {
			vx = ufbxw_copysign(UFBXWI_DPI*0.5, t);
			vy = ufbxw_atan2(-2.0f*t*(qw*qz - qx*qy), t*(2.0f*qw*qx + 2.0f*qy*qz));
			vz = 0.0f;
		}
		break;
	case UFBXW_ROTATION_ORDER_ZYX:
		t = 2.0f*(qw*qy + qx*qz);
		if (ufbxw_fabs(t) < eps) {
			vy = ufbxw_asin(t);
			vx = ufbxw_atan2(2.0f*(qw*qx - qy*qz), 2.0f*(qw*qw + qz*qz) - 1.0f);
			vz = -ufbxw_atan2(-2.0f*(qw*qz - qx*qy), 2.0f*(qw*qw + qx*qx) - 1.0f);
		} else {
			vy = ufbxw_copysign(UFBXWI_DPI*0.5, t);
			vx = ufbxw_atan2(2.0f*t*(qw*qz + qx*qy), -t*(2.0f*qx*qz - 2.0f*qw*qy));
			vz = 0.0f;
		}
		break;
	default:
		vx = vy = vz = 0.0;
		break;
	}

	vx *= UFBXWI_RAD_TO_DEG_DOUBLE;
	vy *= UFBXWI_RAD_TO_DEG_DOUBLE;
	vz *= UFBXWI_RAD_TO_DEG_DOUBLE;

	ufbxw_vec3 v = { (ufbxw_real)vx, (ufbxw_real)vy, (ufbxw_real)vz };
	return v;
}

ufbxw_abi ufbxw_string ufbxw_str(const char *str)
{
	return ufbxwi_c_str(str);
}

#ifdef __cplusplus
}
#endif

#endif

#endif
