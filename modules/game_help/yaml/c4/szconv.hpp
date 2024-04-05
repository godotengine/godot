#ifndef _C4_SZCONV_HPP_
#define _C4_SZCONV_HPP_

/** @file szconv.hpp utilities to deal safely with narrowing conversions */

#include "config.hpp"
#include "error.hpp"

#include <limits>

namespace c4 {

C4_SUPPRESS_WARNING_GCC_CLANG_WITH_PUSH("-Wold-style-cast")

/** @todo this would be so much easier with calls to numeric_limits::max()... */
template<class SizeOut, class SizeIn>
struct is_narrower_size : std::conditional
<
   (std::is_signed<SizeOut>::value == std::is_signed<SizeIn>::value)
   ?
   (sizeof(SizeOut) < sizeof(SizeIn))
   :
   (
       (sizeof(SizeOut) < sizeof(SizeIn))
       ||
       (
           (sizeof(SizeOut) == sizeof(SizeIn))
           &&
           (std::is_signed<SizeOut>::value && std::is_unsigned<SizeIn>::value)
       )
   ),
   std::true_type,
   std::false_type
>::type
{
    static_assert(std::is_integral<SizeIn >::value, "must be integral type");
    static_assert(std::is_integral<SizeOut>::value, "must be integral type");
};


/** when SizeOut is wider than SizeIn, assignment can occur without reservations */
template<class SizeOut, class SizeIn>
C4_ALWAYS_INLINE
typename std::enable_if< ! is_narrower_size<SizeOut, SizeIn>::value, SizeOut>::type
szconv(SizeIn sz) noexcept
{
    return static_cast<SizeOut>(sz);
}

/** when SizeOut is narrower than SizeIn, narrowing will occur, so we check
 * for overflow. Note that this check is done only if C4_XASSERT is enabled.
 * @see C4_XASSERT */
template<class SizeOut, class SizeIn>
C4_ALWAYS_INLINE
typename std::enable_if<is_narrower_size<SizeOut, SizeIn>::value, SizeOut>::type
szconv(SizeIn sz) C4_NOEXCEPT_X
{
    C4_XASSERT(sz >= 0);
    C4_XASSERT_MSG((SizeIn)sz <= (SizeIn)std::numeric_limits<SizeOut>::max(), "size conversion overflow: in=%zu", (size_t)sz);
    SizeOut szo = static_cast<SizeOut>(sz);
    return szo;
}

C4_SUPPRESS_WARNING_GCC_CLANG_POP

} // namespace c4

#endif /* _C4_SZCONV_HPP_ */
