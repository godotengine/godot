/* Simple Plugin API */
/* SPDX-FileCopyrightText: Copyright © 2018 Wim Taymans */
/* SPDX-License-Identifier: MIT */

#ifndef SPA_UTILS_DEFS_H
#define SPA_UTILS_DEFS_H

#ifdef __cplusplus
extern "C" {
# if __cplusplus >= 201103L
#  define SPA_STATIC_ASSERT_IMPL(expr, msg, ...) static_assert(expr, msg)
# endif
#else
# include <stdbool.h>
# if __STDC_VERSION__ >= 201112L
#  define SPA_STATIC_ASSERT_IMPL(expr, msg, ...) _Static_assert(expr, msg)
# endif
#endif
#ifndef SPA_STATIC_ASSERT_IMPL
#define SPA_STATIC_ASSERT_IMPL(expr, ...) \
	((void)sizeof(struct { int spa_static_assertion_failed : 2 * !!(expr) - 1; }))
#endif

#define SPA_STATIC_ASSERT(expr, ...) SPA_STATIC_ASSERT_IMPL(expr, ## __VA_ARGS__, "`" #expr "` evaluated to false")

#include <inttypes.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdio.h>

/**
 * \defgroup spa_utils_defs Miscellaneous
 * Helper macros and functions
 */

/**
 * \addtogroup spa_utils_defs
 * \{
 */

/**
 * SPA_FALLTHROUGH is an annotation to suppress compiler warnings about switch
 * cases that fall through without a break or return statement. SPA_FALLTHROUGH
 * is only needed on cases that have code:
 *
 * switch (foo) {
 *   case 1: // These cases have no code. No fallthrough annotations are needed.
 *   case 2:
 *   case 3:
 *     foo = 4; // This case has code, so a fallthrough annotation is needed:
 *     SPA_FALLTHROUGH;
 *   default:
 *     return foo;
 * }
 */
#if defined(__clang__) && defined(__cplusplus) && __cplusplus >= 201103L
   /* clang's fallthrough annotations are only available starting in C++11. */
#  define SPA_FALLTHROUGH [[clang::fallthrough]];
#elif __GNUC__ >= 7 || __clang_major__ >= 10
#  define SPA_FALLTHROUGH __attribute__ ((fallthrough));
#else
#  define SPA_FALLTHROUGH /* FALLTHROUGH */
#endif

#define SPA_FLAG_MASK(field,mask,flag)	(((field) & (mask)) == (flag))
#define SPA_FLAG_IS_SET(field,flag)	SPA_FLAG_MASK(field, flag, flag)

#define SPA_FLAG_SET(field,flag)	((field) |= (flag))
#define SPA_FLAG_CLEAR(field, flag)					\
({									\
	SPA_STATIC_ASSERT(__builtin_constant_p(flag) ?			\
	              (__typeof__(flag))(__typeof__(field))(__typeof__(flag))(flag) == (flag) : \
		      sizeof(field) >= sizeof(flag),			\
			"truncation problem when masking " #field	\
			" with ~" #flag);				\
	((field) &= ~(__typeof__(field))(flag));			\
})
#define SPA_FLAG_UPDATE(field,flag,val)	((val) ? SPA_FLAG_SET((field),(flag)) : SPA_FLAG_CLEAR((field),(flag)))

enum spa_direction {
	SPA_DIRECTION_INPUT = 0,
	SPA_DIRECTION_OUTPUT = 1,
};

#define SPA_DIRECTION_REVERSE(d)	((d) ^ 1)

#define SPA_RECTANGLE(width,height) ((struct spa_rectangle){ (width), (height) })
struct spa_rectangle {
	uint32_t width;
	uint32_t height;
};

#define SPA_POINT(x,y) ((struct spa_point){ (x), (y) })
struct spa_point {
	int32_t x;
	int32_t y;
};

#define SPA_REGION(x,y,width,height) ((struct spa_region){ SPA_POINT(x,y), SPA_RECTANGLE(width,height) })
struct spa_region {
	struct spa_point position;
	struct spa_rectangle size;
};

#define SPA_FRACTION(num,denom) ((struct spa_fraction){ (num), (denom) })
struct spa_fraction {
	uint32_t num;
	uint32_t denom;
};

#define SPA_N_ELEMENTS(arr)  (sizeof(arr) / sizeof((arr)[0]))
/**
 * Array iterator macro. Usage:
 * ```c
 * struct foo array[16];
 * struct foo *f;
 * SPA_FOR_EACH_ELEMENT(array, f) {
 *	f->bar = baz;
 * }
 * ```
 */
#define SPA_FOR_EACH_ELEMENT(arr, ptr) \
	for ((ptr) = arr; (void*)(ptr) < SPA_PTROFF(arr, sizeof(arr), void); (ptr)++)

#define SPA_FOR_EACH_ELEMENT_VAR(arr, var) \
	for (__typeof__((arr)[0])* var = arr; (void*)(var) < SPA_PTROFF(arr, sizeof(arr), void); (var)++)

#define SPA_ABS(a)			\
({					\
	__typeof__(a) _a = (a);		\
	SPA_LIKELY(_a >= 0) ? _a : -_a;	\
})
#define SPA_MIN(a,b)					\
({							\
	__typeof__(a) _min_a = (a);			\
	__typeof__(b) _min_b = (b);			\
	SPA_LIKELY(_min_a <= _min_b) ? _min_a : _min_b;	\
})
#define SPA_MAX(a,b)					\
({							\
	__typeof__(a) _max_a = (a);			\
	__typeof__(b) _max_b = (b);			\
	SPA_LIKELY(_max_a >= _max_b) ? _max_a : _max_b;	\
})
#define SPA_CLAMP(v,low,high)				\
({							\
	__typeof__(v) _v = (v);				\
	__typeof__(low) _low = (low);			\
	__typeof__(high) _high = (high);		\
	SPA_MIN(SPA_MAX(_v, _low), _high);		\
})

#define SPA_CLAMPF(v,low,high)				\
({							\
	fminf(fmaxf(v, low), high);			\
})


#define SPA_SWAP(a,b)					\
({							\
	__typeof__(a) _t = (a);				\
	(a) = b; (b) = _t;				\
})

#define SPA_TYPECHECK(type,x)		\
({	type _dummy;			\
	typeof(x) _dummy2;		\
	(void)(&_dummy == &_dummy2);	\
	x;				\
})

/**
 * Return the address (buffer + offset) as pointer of \a type
 */
#define SPA_PTROFF(ptr_,offset_,type_) ((type_*)((uintptr_t)(ptr_) + (ptrdiff_t)(offset_)))
#define SPA_PTROFF_ALIGN(ptr_,offset_,alignment_,type_) \
   SPA_PTR_ALIGN(SPA_PTROFF(ptr_,offset_,type_),alignment_,type_)


/**
 * Deprecated, use SPA_PTROFF and SPA_PTROFF_ALIGN instead
 */
#define SPA_MEMBER(b,o,t) SPA_PTROFF(b,o,t)
#define SPA_MEMBER_ALIGN(b,o,a,t) SPA_PTROFF_ALIGN(b,o,a,t)

#define SPA_CONTAINER_OF(p,t,m) ((t*)((uintptr_t)(p) - offsetof(t,m)))

#define SPA_PTRDIFF(p1,p2) ((intptr_t)(p1) - (intptr_t)(p2))

#define SPA_PTR_TO_INT(p) ((int) ((intptr_t) (p)))
#define SPA_INT_TO_PTR(u) ((void*) ((intptr_t) (u)))

#define SPA_PTR_TO_UINT32(p) ((uint32_t) ((uintptr_t) (p)))
#define SPA_UINT32_TO_PTR(u) ((void*) ((uintptr_t) (u)))

#define SPA_TIME_INVALID  ((int64_t)INT64_MIN)
#define SPA_IDX_INVALID  ((unsigned int)-1)
#define SPA_ID_INVALID  ((uint32_t)0xffffffff)

#define SPA_NSEC_PER_SEC  (1000000000LL)
#define SPA_NSEC_PER_MSEC (1000000ll)
#define SPA_NSEC_PER_USEC (1000ll)
#define SPA_USEC_PER_SEC  (1000000ll)
#define SPA_USEC_PER_MSEC (1000ll)
#define SPA_MSEC_PER_SEC  (1000ll)

#define SPA_TIMESPEC_TO_NSEC(ts) ((ts)->tv_sec * SPA_NSEC_PER_SEC + (ts)->tv_nsec)
#define SPA_TIMESPEC_TO_USEC(ts) ((ts)->tv_sec * SPA_USEC_PER_SEC + (ts)->tv_nsec / SPA_NSEC_PER_USEC)
#define SPA_TIMEVAL_TO_NSEC(tv)  ((tv)->tv_sec * SPA_NSEC_PER_SEC + (tv)->tv_usec * SPA_NSEC_PER_USEC)
#define SPA_TIMEVAL_TO_USEC(tv)  ((tv)->tv_sec * SPA_USEC_PER_SEC + (tv)->tv_usec)

#ifdef __GNUC__
#define SPA_PRINTF_FUNC(fmt, arg1) __attribute__((format(printf, fmt, arg1)))
#define SPA_FORMAT_ARG_FUNC(arg1) __attribute__((format_arg(arg1)))
#define SPA_ALIGNED(align) __attribute__((aligned(align)))
#define SPA_DEPRECATED __attribute__ ((deprecated))
#define SPA_EXPORT __attribute__((visibility("default")))
#define SPA_SENTINEL __attribute__((__sentinel__))
#define SPA_UNUSED __attribute__ ((unused))
#define SPA_NORETURN __attribute__ ((noreturn))
#define SPA_WARN_UNUSED_RESULT __attribute__ ((warn_unused_result))
#else
#define SPA_PRINTF_FUNC(fmt, arg1)
#define SPA_FORMAT_ARG_FUNC(arg1)
#define SPA_ALIGNED(align)
#define SPA_DEPRECATED
#define SPA_EXPORT
#define SPA_SENTINEL
#define SPA_UNUSED
#define SPA_NORETURN
#define SPA_WARN_UNUSED_RESULT
#endif

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
#define SPA_RESTRICT restrict
#elif defined(__GNUC__) && __GNUC__ >= 4
#define SPA_RESTRICT __restrict__
#else
#define SPA_RESTRICT
#endif

#define SPA_ROUND_DOWN(num,value)		\
({						\
	__typeof__(num) _num = (num);		\
	((_num) - ((_num) % (value)));		\
})
#define SPA_ROUND_UP(num,value)			\
({						\
	__typeof__(value) _v = (value);		\
	((((num) + (_v) - 1) / (_v)) * (_v));	\
})

#define SPA_ROUND_MASK(num,mask)	((__typeof__(num))((mask)-1))

#define SPA_ROUND_DOWN_N(num,align)	((num) & ~SPA_ROUND_MASK(num, align))
#define SPA_ROUND_UP_N(num,align)	((((num)-1) | SPA_ROUND_MASK(num, align))+1)

#define SPA_SCALE32_UP(val,num,denom)				\
({								\
	uint64_t _val = (val);					\
	uint64_t _denom = (denom);				\
	(uint32_t)(((_val) * (num) + (_denom)-1) / (_denom));	\
})


#define SPA_PTR_ALIGNMENT(p,align)	((intptr_t)(p) & ((align)-1))
#define SPA_IS_ALIGNED(p,align)		(SPA_PTR_ALIGNMENT(p,align) == 0)
#define SPA_PTR_ALIGN(p,align,type)	((type*)SPA_ROUND_UP_N((intptr_t)(p), (intptr_t)(align)))

#ifndef SPA_LIKELY
#ifdef __GNUC__
#define SPA_LIKELY(x) (__builtin_expect(!!(x),1))
#define SPA_UNLIKELY(x) (__builtin_expect(!!(x),0))
#else
#define SPA_LIKELY(x) (x)
#define SPA_UNLIKELY(x) (x)
#endif
#endif

#define SPA_STRINGIFY_1(...)	#__VA_ARGS__
#define SPA_STRINGIFY(...)	SPA_STRINGIFY_1(__VA_ARGS__)

#define spa_return_if_fail(expr)					\
	do {								\
		if (SPA_UNLIKELY(!(expr))) {				\
			fprintf(stderr, "'%s' failed at %s:%u %s()\n",	\
				#expr , __FILE__, __LINE__, __func__);	\
			return;						\
		}							\
	} while(false)

#define spa_return_val_if_fail(expr, val)				\
	do {								\
		if (SPA_UNLIKELY(!(expr))) {				\
			fprintf(stderr, "'%s' failed at %s:%u %s()\n",	\
				#expr , __FILE__, __LINE__, __func__);	\
			return (val);					\
		}							\
	} while(false)

/* spa_assert_se() is an assert which guarantees side effects of x,
 * i.e. is never optimized away, regardless of NDEBUG or FASTPATH. */
#ifndef __COVERITY__
#define spa_assert_se(expr)						\
	do {								\
		if (SPA_UNLIKELY(!(expr))) {				\
			fprintf(stderr, "'%s' failed at %s:%u %s()\n",	\
				#expr , __FILE__, __LINE__, __func__);	\
			abort();					\
		}							\
	} while (false)
#else
#define spa_assert_se(expr)						\
	do {								\
		int _unique_var = (expr);				\
		if (!_unique_var)					\
			abort();					\
		} while (false)
#endif

/* Does exactly nothing */
#define spa_nop() do {} while (false)

#ifdef NDEBUG
#define spa_assert(expr) spa_nop()
#elif defined (FASTPATH)
#define spa_assert(expr) spa_assert_se(expr)
#else
#define spa_assert(expr) spa_assert_se(expr)
#endif

#ifdef NDEBUG
#define spa_assert_not_reached() abort()
#else
#define spa_assert_not_reached()						\
	do {									\
		fprintf(stderr, "Code should not be reached at %s:%u %s()\n",	\
				__FILE__, __LINE__, __func__);			\
		abort();							\
	} while (false)
#endif

#define spa_memzero(x,l) (memset((x), 0, (l)))
#define spa_zero(x) (spa_memzero(&(x), sizeof(x)))

#ifdef SPA_DEBUG_MEMCPY
#define spa_memcpy(d,s,n)						\
({									\
	fprintf(stderr, "%s:%u %s() memcpy(%p, %p, %zd)\n",		\
		__FILE__, __LINE__, __func__, (d), (s), (size_t)(n));	\
	memcpy(d,s,n);							\
})
#define spa_memmove(d,s,n)						\
({									\
	fprintf(stderr, "%s:%u %s() memmove(%p, %p, %zd)\n",		\
		__FILE__, __LINE__, __func__, (d), (s), (size_t)(n));	\
	memmove(d,s,n);							\
})
#else
#define spa_memcpy(d,s,n)	memcpy(d,s,n)
#define spa_memmove(d,s,n)	memmove(d,s,n)
#endif

#define spa_aprintf(_fmt, ...)						\
({									\
	char *_strp;							\
	if (asprintf(&(_strp), (_fmt), ## __VA_ARGS__ ) == -1)		\
		_strp = NULL;						\
	_strp;								\
})

/**
 * \}
 */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SPA_UTILS_DEFS_H */
