#ifndef _C4_MEMORY_UTIL_HPP_
#define _C4_MEMORY_UTIL_HPP_

#include "config.hpp"
#include "error.hpp"
#include "compiler.hpp"
#include "cpu.hpp"
#ifdef C4_MSVC
#include <intrin.h>
#endif
#include <string.h>

#if (defined(__GNUC__) && __GNUC__ >= 10) || defined(__has_builtin)
#define _C4_USE_LSB_INTRINSIC(which) __has_builtin(which)
#define _C4_USE_MSB_INTRINSIC(which) __has_builtin(which)
#elif defined(C4_MSVC)
#define _C4_USE_LSB_INTRINSIC(which) true
#define _C4_USE_MSB_INTRINSIC(which) true
#else
// let's try our luck
#define _C4_USE_LSB_INTRINSIC(which) true
#define _C4_USE_MSB_INTRINSIC(which) true
#endif


/** @file memory_util.hpp Some memory utilities. */

namespace c4 {

C4_SUPPRESS_WARNING_GCC_CLANG_WITH_PUSH("-Wold-style-cast")

/** set the given memory to zero */
C4_ALWAYS_INLINE void mem_zero(void* mem, size_t num_bytes)
{
    memset(mem, 0, num_bytes);
}
/** set the given memory to zero */
template<class T>
C4_ALWAYS_INLINE void mem_zero(T* mem, size_t num_elms)
{
    memset(mem, 0, sizeof(T) * num_elms);
}
/** set the given memory to zero */
template<class T>
C4_ALWAYS_INLINE void mem_zero(T* mem)
{
    memset(mem, 0, sizeof(T));
}

C4_ALWAYS_INLINE C4_CONST bool mem_overlaps(void const* a, void const* b, size_t sza, size_t szb)
{
    // thanks @timwynants
    return (((const char*)b + szb) > a && b < ((const char*)a+sza));
}

void mem_repeat(void* dest, void const* pattern, size_t pattern_size, size_t num_times);


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

template<class T>
C4_ALWAYS_INLINE C4_CONST bool is_aligned(T *ptr, uintptr_t alignment=alignof(T))
{
    return (uintptr_t(ptr) & (alignment - uintptr_t(1))) == uintptr_t(0);
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// least significant bit

/** @name msb Compute the least significant bit
 * @note the input value must be nonzero
 * @note the input type must be unsigned
 */
/** @{ */

// https://graphics.stanford.edu/~seander/bithacks.html#ZerosOnRightLinear
#define _c4_lsb_fallback                                                \
    unsigned c = 0;                                                     \
    v = (v ^ (v - 1)) >> 1; /* Set v's trailing 0s to 1s and zero rest */ \
    for(; v; ++c)                                                       \
        v >>= 1;                                                        \
    return (unsigned) c

// u8
template<class I>
C4_CONSTEXPR14
auto lsb(I v) noexcept
    -> typename std::enable_if<sizeof(I) == 1u, unsigned>::type
{
    C4_STATIC_ASSERT(std::is_unsigned<I>::value);
    C4_ASSERT(v != 0);
    #if _C4_USE_LSB_INTRINSIC(__builtin_ctz)
        // upcast to use the intrinsic, it's cheaper.
        #ifdef C4_MSVC
            #if !defined(C4_CPU_ARM64) && !defined(C4_CPU_ARM)
                unsigned long bit;
                _BitScanForward(&bit, (unsigned long)v);
                return bit;
            #else
                _c4_lsb_fallback;
            #endif
        #else
            return (unsigned)__builtin_ctz((unsigned)v);
        #endif
    #else
        _c4_lsb_fallback;
    #endif
}

// u16
template<class I>
C4_CONSTEXPR14
auto lsb(I v) noexcept
    -> typename std::enable_if<sizeof(I) == 2u, unsigned>::type
{
    C4_STATIC_ASSERT(std::is_unsigned<I>::value);
    C4_ASSERT(v != 0);
    #if _C4_USE_LSB_INTRINSIC(__builtin_ctz)
        // upcast to use the intrinsic, it's cheaper.
        // Then remember that the upcast makes it to 31bits
        #ifdef C4_MSVC
            #if !defined(C4_CPU_ARM64) && !defined(C4_CPU_ARM)
                unsigned long bit;
                _BitScanForward(&bit, (unsigned long)v);
                return bit;
            #else
                _c4_lsb_fallback;
            #endif
        #else
            return (unsigned)__builtin_ctz((unsigned)v);
        #endif
    #else
        _c4_lsb_fallback;
    #endif
}

// u32
template<class I>
C4_CONSTEXPR14
auto lsb(I v) noexcept
    -> typename std::enable_if<sizeof(I) == 4u, unsigned>::type
{
    C4_STATIC_ASSERT(std::is_unsigned<I>::value);
    C4_ASSERT(v != 0);
    #if _C4_USE_LSB_INTRINSIC(__builtin_ctz)
        #ifdef C4_MSVC
            #if !defined(C4_CPU_ARM64) && !defined(C4_CPU_ARM)
                unsigned long bit;
                _BitScanForward(&bit, v);
                return bit;
            #else
                _c4_lsb_fallback;
            #endif
        #else
            return (unsigned)__builtin_ctz((unsigned)v);
        #endif
    #else
        _c4_lsb_fallback;
    #endif
}

// u64 in 64bits
template<class I>
C4_CONSTEXPR14
auto lsb(I v) noexcept
    -> typename std::enable_if<sizeof(I) == 8u && sizeof(unsigned long) == 8u, unsigned>::type
{
    C4_STATIC_ASSERT(std::is_unsigned<I>::value);
    C4_ASSERT(v != 0);
    #if _C4_USE_LSB_INTRINSIC(__builtin_ctzl)
        #if defined(C4_MSVC)
            #if !defined(C4_CPU_ARM64) && !defined(C4_CPU_ARM)
                unsigned long bit;
                _BitScanForward64(&bit, v);
                return bit;
            #else
                _c4_lsb_fallback;
            #endif
        #else
            return (unsigned)__builtin_ctzl((unsigned long)v);
        #endif
    #else
        _c4_lsb_fallback;
    #endif
}

// u64 in 32bits
template<class I>
C4_CONSTEXPR14
auto lsb(I v) noexcept
    -> typename std::enable_if<sizeof(I) == 8u && sizeof(unsigned long long) == 8u && sizeof(unsigned long) != sizeof(unsigned long long), unsigned>::type
{
    C4_STATIC_ASSERT(std::is_unsigned<I>::value);
    C4_ASSERT(v != 0);
    #if _C4_USE_LSB_INTRINSIC(__builtin_ctzll)
        #if defined(C4_MSVC)
            #if !defined(C4_CPU_X86) && !defined(C4_CPU_ARM64) && !defined(C4_CPU_ARM)
                unsigned long bit;
                _BitScanForward64(&bit, v);
                return bit;
            #else
                _c4_lsb_fallback;
            #endif
        #else
            return (unsigned)__builtin_ctzll((unsigned long long)v);
        #endif
    #else
        _c4_lsb_fallback;
    #endif
}

#undef _c4_lsb_fallback

/** @} */


namespace detail {
template<class I, I val, unsigned num_bits, bool finished> struct _lsb11;
template<class I, I val, unsigned num_bits>
struct _lsb11<I, val, num_bits, false>
{
    enum : unsigned { num = _lsb11<I, (val>>1), num_bits+I(1), (((val>>1)&I(1))!=I(0))>::num };
};
template<class I, I val, unsigned num_bits>
struct _lsb11<I, val, num_bits, true>
{
    enum : unsigned { num = num_bits };
};
} // namespace detail


/** TMP version of lsb(); this needs to be implemented with template
 * meta-programming because C++11 cannot use a constexpr function with
 * local variables
 * @see lsb */
template<class I, I number>
struct lsb11
{
    static_assert(number != 0, "lsb: number must be nonzero");
    enum : unsigned { value = detail::_lsb11<I, number, 0, ((number&I(1))!=I(0))>::num};
};


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// most significant bit


/** @name msb Compute the most significant bit
 * @note the input value must be nonzero
 * @note the input type must be unsigned
 */
/** @{ */


#define _c4_msb8_fallback                       \
    unsigned n = 0;                             \
    if(v & I(0xf0)) v >>= 4, n |= I(4);         \
    if(v & I(0x0c)) v >>= 2, n |= I(2);         \
    if(v & I(0x02)) v >>= 1, n |= I(1);         \
    return n

#define _c4_msb16_fallback                      \
    unsigned n = 0;                             \
    if(v & I(0xff00)) v >>= 8, n |= I(8);       \
    if(v & I(0x00f0)) v >>= 4, n |= I(4);       \
    if(v & I(0x000c)) v >>= 2, n |= I(2);       \
    if(v & I(0x0002)) v >>= 1, n |= I(1);       \
    return n

#define _c4_msb32_fallback                      \
    unsigned n = 0;                             \
    if(v & I(0xffff0000)) v >>= 16, n |= 16;    \
    if(v & I(0x0000ff00)) v >>= 8, n |= 8;      \
    if(v & I(0x000000f0)) v >>= 4, n |= 4;      \
    if(v & I(0x0000000c)) v >>= 2, n |= 2;      \
    if(v & I(0x00000002)) v >>= 1, n |= 1;      \
    return n

#define _c4_msb64_fallback                              \
    unsigned n = 0;                                     \
    if(v & I(0xffffffff00000000)) v >>= 32, n |= I(32); \
    if(v & I(0x00000000ffff0000)) v >>= 16, n |= I(16); \
    if(v & I(0x000000000000ff00)) v >>= 8, n |= I(8);   \
    if(v & I(0x00000000000000f0)) v >>= 4, n |= I(4);   \
    if(v & I(0x000000000000000c)) v >>= 2, n |= I(2);   \
    if(v & I(0x0000000000000002)) v >>= 1, n |= I(1);   \
    return n


// u8
template<class I>
C4_CONSTEXPR14
auto msb(I v) noexcept
    -> typename std::enable_if<sizeof(I) == 1u, unsigned>::type
{
    C4_STATIC_ASSERT(std::is_unsigned<I>::value);
    C4_ASSERT(v != 0);
    #if _C4_USE_MSB_INTRINSIC(__builtin_clz)
        // upcast to use the intrinsic, it's cheaper.
        // Then remember that the upcast makes it to 31bits
        #ifdef C4_MSVC
            #if !defined(C4_CPU_ARM64) && !defined(C4_CPU_ARM)
                unsigned long bit;
                _BitScanReverse(&bit, (unsigned long)v);
                return bit;
            #else
                _c4_msb8_fallback;
            #endif
        #else
            return 31u - (unsigned)__builtin_clz((unsigned)v);
        #endif
    #else
        _c4_msb8_fallback;
    #endif
}

// u16
template<class I>
C4_CONSTEXPR14
auto msb(I v) noexcept
    -> typename std::enable_if<sizeof(I) == 2u, unsigned>::type
{
    C4_STATIC_ASSERT(std::is_unsigned<I>::value);
    C4_ASSERT(v != 0);
    #if _C4_USE_MSB_INTRINSIC(__builtin_clz)
        // upcast to use the intrinsic, it's cheaper.
        // Then remember that the upcast makes it to 31bits
        #ifdef C4_MSVC
            #if !defined(C4_CPU_ARM64) && !defined(C4_CPU_ARM)
                unsigned long bit;
                _BitScanReverse(&bit, (unsigned long)v);
                return bit;
            #else
                _c4_msb16_fallback;
            #endif
        #else
            return 31u - (unsigned)__builtin_clz((unsigned)v);
        #endif
    #else
        _c4_msb16_fallback;
    #endif
}

// u32
template<class I>
C4_CONSTEXPR14
auto msb(I v) noexcept
    -> typename std::enable_if<sizeof(I) == 4u, unsigned>::type
{
    C4_STATIC_ASSERT(std::is_unsigned<I>::value);
    C4_ASSERT(v != 0);
    #if _C4_USE_MSB_INTRINSIC(__builtin_clz)
        #ifdef C4_MSVC
            #if !defined(C4_CPU_ARM64) && !defined(C4_CPU_ARM)
                unsigned long bit;
                _BitScanReverse(&bit, v);
                return bit;
            #else
                _c4_msb32_fallback;
            #endif
        #else
            return 31u - (unsigned)__builtin_clz((unsigned)v);
        #endif
    #else
        _c4_msb32_fallback;
    #endif
}

// u64 in 64bits
template<class I>
C4_CONSTEXPR14
auto msb(I v) noexcept
    -> typename std::enable_if<sizeof(I) == 8u && sizeof(unsigned long) == 8u, unsigned>::type
{
    C4_STATIC_ASSERT(std::is_unsigned<I>::value);
    C4_ASSERT(v != 0);
    #if _C4_USE_MSB_INTRINSIC(__builtin_clzl)
        #ifdef C4_MSVC
            #if !defined(C4_CPU_ARM64) && !defined(C4_CPU_ARM)
                unsigned long bit;
                _BitScanReverse64(&bit, v);
                return bit;
            #else
                _c4_msb64_fallback;
            #endif
        #else
            return 63u - (unsigned)__builtin_clzl((unsigned long)v);
        #endif
    #else
        _c4_msb64_fallback;
    #endif
}

// u64 in 32bits
template<class I>
C4_CONSTEXPR14
auto msb(I v) noexcept
    -> typename std::enable_if<sizeof(I) == 8u && sizeof(unsigned long long) == 8u && sizeof(unsigned long) != sizeof(unsigned long long), unsigned>::type
{
    C4_STATIC_ASSERT(std::is_unsigned<I>::value);
    C4_ASSERT(v != 0);
    #if _C4_USE_MSB_INTRINSIC(__builtin_clzll)
        #ifdef C4_MSVC
            #if !defined(C4_CPU_X86) && !defined(C4_CPU_ARM64) && !defined(C4_CPU_ARM)
                unsigned long bit;
                _BitScanReverse64(&bit, v);
                return bit;
            #else
                _c4_msb64_fallback;
            #endif
        #else
            return 63u - (unsigned)__builtin_clzll((unsigned long long)v);
        #endif
    #else
        _c4_msb64_fallback;
    #endif
}

#undef _c4_msb8_fallback
#undef _c4_msb16_fallback
#undef _c4_msb32_fallback
#undef _c4_msb64_fallback

/** @} */


namespace detail {
template<class I, I val, I num_bits, bool finished> struct _msb11;
template<class I, I val, I num_bits>
struct _msb11< I, val, num_bits, false>
{
    enum : unsigned { num = _msb11<I, (val>>1), num_bits+I(1), ((val>>1)==I(0))>::num };
};
template<class I, I val, I num_bits>
struct _msb11<I, val, num_bits, true>
{
    static_assert(val == 0, "bad implementation");
    enum : unsigned { num = (unsigned)(num_bits-1) };
};
} // namespace detail


/** TMP version of msb(); this needs to be implemented with template
 * meta-programming because C++11 cannot use a constexpr function with
 * local variables
 * @see msb */
template<class I, I number>
struct msb11
{
    enum : unsigned { value = detail::_msb11<I, number, 0, (number==I(0))>::num };
};



#undef _C4_USE_LSB_INTRINSIC
#undef _C4_USE_MSB_INTRINSIC

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

// there is an implicit conversion below; it happens when E or B are
// narrower than int, and thus any operation will upcast the result to
// int, and then downcast to assign
C4_SUPPRESS_WARNING_GCC_CLANG_WITH_PUSH("-Wconversion")

/** integer power; this function is constexpr-14 because of the local
 * variables */
template<class B, class E>
C4_CONSTEXPR14 C4_CONST auto ipow(B base, E exponent) noexcept -> typename std::enable_if<std::is_signed<E>::value, B>::type
{
    C4_STATIC_ASSERT(std::is_integral<E>::value);
    B r = B(1);
    if(exponent >= 0)
    {
        for(E e = 0; e < exponent; ++e)
            r *= base;
    }
    else
    {
        exponent *= E(-1);
        for(E e = 0; e < exponent; ++e)
            r /= base;
    }
    return r;
}

/** integer power; this function is constexpr-14 because of the local
 * variables */
template<class B, B base, class E>
C4_CONSTEXPR14 C4_CONST auto ipow(E exponent) noexcept -> typename std::enable_if<std::is_signed<E>::value, B>::type
{
    C4_STATIC_ASSERT(std::is_integral<E>::value);
    B r = B(1);
    if(exponent >= 0)
    {
        for(E e = 0; e < exponent; ++e)
            r *= base;
    }
    else
    {
        exponent *= E(-1);
        for(E e = 0; e < exponent; ++e)
            r /= base;
    }
    return r;
}

/** integer power; this function is constexpr-14 because of the local
 * variables */
template<class B, class Base, Base base, class E>
C4_CONSTEXPR14 C4_CONST auto ipow(E exponent) noexcept -> typename std::enable_if<std::is_signed<E>::value, B>::type
{
    C4_STATIC_ASSERT(std::is_integral<E>::value);
    B r = B(1);
    B bbase = B(base);
    if(exponent >= 0)
    {
        for(E e = 0; e < exponent; ++e)
            r *= bbase;
    }
    else
    {
        exponent *= E(-1);
        for(E e = 0; e < exponent; ++e)
            r /= bbase;
    }
    return r;
}

/** integer power; this function is constexpr-14 because of the local
 * variables */
template<class B, class E>
C4_CONSTEXPR14 C4_CONST auto ipow(B base, E exponent) noexcept -> typename std::enable_if<!std::is_signed<E>::value, B>::type
{
    C4_STATIC_ASSERT(std::is_integral<E>::value);
    B r = B(1);
    for(E e = 0; e < exponent; ++e)
        r *= base;
    return r;
}

/** integer power; this function is constexpr-14 because of the local
 * variables */
template<class B, B base, class E>
C4_CONSTEXPR14 C4_CONST auto ipow(E exponent) noexcept -> typename std::enable_if<!std::is_signed<E>::value, B>::type
{
    C4_STATIC_ASSERT(std::is_integral<E>::value);
    B r = B(1);
    for(E e = 0; e < exponent; ++e)
        r *= base;
    return r;
}
/** integer power; this function is constexpr-14 because of the local
 * variables */
template<class B, class Base, Base base, class E>
C4_CONSTEXPR14 C4_CONST auto ipow(E exponent) noexcept -> typename std::enable_if<!std::is_signed<E>::value, B>::type
{
    C4_STATIC_ASSERT(std::is_integral<E>::value);
    B r = B(1);
    B bbase = B(base);
    for(E e = 0; e < exponent; ++e)
        r *= bbase;
    return r;
}

C4_SUPPRESS_WARNING_GCC_CLANG_POP


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

/** return a mask with all bits set [first_bit,last_bit[; this function
 * is constexpr-14 because of the local variables */
template<class I>
C4_CONSTEXPR14 I contiguous_mask(I first_bit, I last_bit)
{
    I r = 0;
    for(I i = first_bit; i < last_bit; ++i)
    {
        r |= (I(1) << i);
    }
    return r;
}


namespace detail {

template<class I, I val, I first, I last, bool finished>
struct _ctgmsk11;

template<class I, I val, I first, I last>
struct _ctgmsk11< I, val, first, last, true>
{
    enum : I { value = _ctgmsk11<I, val|(I(1)<<first), first+I(1), last, (first+1!=last)>::value };
};

template<class I, I val, I first, I last>
struct _ctgmsk11< I, val, first, last, false>
{
    enum : I { value = val };
};

} // namespace detail


/** TMP version of contiguous_mask(); this needs to be implemented with template
 * meta-programming because C++11 cannot use a constexpr function with
 * local variables
 * @see contiguous_mask */
template<class I, I first_bit, I last_bit>
struct contiguous_mask11
{
    enum : I { value = detail::_ctgmsk11<I, I(0), first_bit, last_bit, (first_bit!=last_bit)>::value };
};


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
/** use Empty Base Class Optimization to reduce the size of a pair of
 * potentially empty types*/

namespace detail {
typedef enum {
    tpc_same,
    tpc_same_empty,
    tpc_both_empty,
    tpc_first_empty,
    tpc_second_empty,
    tpc_general
} TightPairCase_e;

template<class First, class Second>
constexpr TightPairCase_e tpc_which_case()
{
    return std::is_same<First, Second>::value ?
               std::is_empty<First>::value ?
                   tpc_same_empty
                   :
                   tpc_same
               :
               std::is_empty<First>::value && std::is_empty<Second>::value ?
                   tpc_both_empty
                   :
                   std::is_empty<First>::value ?
                       tpc_first_empty
                       :
                       std::is_empty<Second>::value ?
                           tpc_second_empty
                           :
                           tpc_general
           ;
}

template<class First, class Second, TightPairCase_e Case>
struct tight_pair
{
private:

    First m_first;
    Second m_second;

public:

    using first_type = First;
    using second_type = Second;

    tight_pair() : m_first(), m_second() {}
    tight_pair(First const& f, Second const& s) : m_first(f), m_second(s) {}

    C4_ALWAYS_INLINE C4_CONSTEXPR14 First       & first ()       { return m_first; }
    C4_ALWAYS_INLINE C4_CONSTEXPR14 First  const& first () const { return m_first; }
    C4_ALWAYS_INLINE C4_CONSTEXPR14 Second      & second()       { return m_second; }
    C4_ALWAYS_INLINE C4_CONSTEXPR14 Second const& second() const { return m_second; }
};

template<class First, class Second>
struct tight_pair<First, Second, tpc_same_empty> : public First
{
    static_assert(std::is_same<First, Second>::value, "bad implementation");

    using first_type = First;
    using second_type = Second;

    tight_pair() : First() {}
    tight_pair(First const& f, Second const& /*s*/) : First(f) {}

    C4_ALWAYS_INLINE C4_CONSTEXPR14 First      & first ()       { return static_cast<First      &>(*this); }
    C4_ALWAYS_INLINE C4_CONSTEXPR14 First const& first () const { return static_cast<First const&>(*this); }
    C4_ALWAYS_INLINE C4_CONSTEXPR14 Second      & second()       { return reinterpret_cast<Second      &>(*this); }
    C4_ALWAYS_INLINE C4_CONSTEXPR14 Second const& second() const { return reinterpret_cast<Second const&>(*this); }
};

template<class First, class Second>
struct tight_pair<First, Second, tpc_both_empty> : public First, public Second
{
    using first_type = First;
    using second_type = Second;

    tight_pair() : First(), Second() {}
    tight_pair(First const& f, Second const& s) : First(f), Second(s) {}

    C4_ALWAYS_INLINE C4_CONSTEXPR14 First      & first ()       { return static_cast<First      &>(*this); }
    C4_ALWAYS_INLINE C4_CONSTEXPR14 First const& first () const { return static_cast<First const&>(*this); }
    C4_ALWAYS_INLINE C4_CONSTEXPR14 Second      & second()       { return static_cast<Second      &>(*this); }
    C4_ALWAYS_INLINE C4_CONSTEXPR14 Second const& second() const { return static_cast<Second const&>(*this); }
};

template<class First, class Second>
struct tight_pair<First, Second, tpc_same> : public First
{
    Second m_second;

    using first_type = First;
    using second_type = Second;

    tight_pair() : First() {}
    tight_pair(First const& f, Second const& s) : First(f), m_second(s) {}

    C4_ALWAYS_INLINE C4_CONSTEXPR14 First      & first ()       { return static_cast<First      &>(*this); }
    C4_ALWAYS_INLINE C4_CONSTEXPR14 First const& first () const { return static_cast<First const&>(*this); }
    C4_ALWAYS_INLINE C4_CONSTEXPR14 Second      & second()       { return m_second; }
    C4_ALWAYS_INLINE C4_CONSTEXPR14 Second const& second() const { return m_second; }
};

template<class First, class Second>
struct tight_pair<First, Second, tpc_first_empty> : public First
{
    Second m_second;

    using first_type = First;
    using second_type = Second;

    tight_pair() : First(), m_second() {}
    tight_pair(First const& f, Second const& s) : First(f), m_second(s) {}

    C4_ALWAYS_INLINE C4_CONSTEXPR14 First      & first ()       { return static_cast<First      &>(*this); }
    C4_ALWAYS_INLINE C4_CONSTEXPR14 First const& first () const { return static_cast<First const&>(*this); }
    C4_ALWAYS_INLINE C4_CONSTEXPR14 Second      & second()       { return m_second; }
    C4_ALWAYS_INLINE C4_CONSTEXPR14 Second const& second() const { return m_second; }
};

template<class First, class Second>
struct tight_pair<First, Second, tpc_second_empty> : public Second
{
    First m_first;

    using first_type = First;
    using second_type = Second;

    tight_pair() : Second(), m_first() {}
    tight_pair(First const& f, Second const& s) : Second(s), m_first(f) {}

    C4_ALWAYS_INLINE C4_CONSTEXPR14 First      & first ()       { return m_first; }
    C4_ALWAYS_INLINE C4_CONSTEXPR14 First const& first () const { return m_first; }
    C4_ALWAYS_INLINE C4_CONSTEXPR14 Second      & second()       { return static_cast<Second      &>(*this); }
    C4_ALWAYS_INLINE C4_CONSTEXPR14 Second const& second() const { return static_cast<Second const&>(*this); }
};

} // namespace detail

template<class First, class Second>
using tight_pair = detail::tight_pair<First, Second, detail::tpc_which_case<First,Second>()>;

C4_SUPPRESS_WARNING_GCC_CLANG_POP

} // namespace c4

#endif /* _C4_MEMORY_UTIL_HPP_ */
