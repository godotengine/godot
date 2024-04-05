#ifndef _C4_RESTRICT_HPP_
#define _C4_RESTRICT_HPP_

/** @file restrict.hpp macros defining shorthand symbols for restricted
 * pointers and references
 * @see unrestrict.hpp
 * @see restrict
 */

/** @defgroup restrict Restrict utilities
 * macros defining shorthand symbols for restricted
 * pointers and references
 * ```cpp
 * void sum_arrays(size_t sz, float const* C4_RESTRICT a, float const *C4_RESTRICT b, float *result);
 * float * C4_RESTRICT ptr;
 * float & C4_RESTRICT ref = *ptr;
 * float const* C4_RESTRICT cptr;
 * float const& C4_RESTRICT cref = *cptr;
 *
 * // becomes this:
 * #include <c4/restrict.hpp>
 * void sum_arrays(size_t sz, float c$ a, float c$ b, float * result);
 * float   $ ptr;
 * float  $$ ref = *ptr;
 * float  c$ cptr;
 * float c$$ cref = *cptr;
 * ```
 * @ingroup types
 * @{ */

/** @def \$ a restricted pointer */
/** @def c\$ a restricted pointer to const data */

/** @def \$\$  a restricted reference */
/** @def c\$\$  a restricted reference to const data */

#ifdef __clang__
#   pragma clang diagnostic push
#   pragma clang diagnostic ignored "-Wdollar-in-identifier-extension"
#elif defined(__GNUC__)
#endif

#define   $       * C4_RESTRICT // a restricted pointer
#define  c$  const* C4_RESTRICT // a restricted pointer to const data

#define  $$       & C4_RESTRICT // restricted reference
#define c$$  const& C4_RESTRICT // restricted reference to const data

/** @} */

#endif /* _C4_RESTRICT_HPP_ */
