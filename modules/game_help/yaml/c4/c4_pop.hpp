#ifdef _C4_PUSH_HPP_ // this must match the include guard from c4_push

/** @file c4_pop.hpp disables the macros and control directives
 * enabled in c4_push.hpp.
 * @see c4_push.hpp */

#include "unrestrict.hpp"

#ifdef C4_WIN
#   include "windows_pop.hpp"
#endif

#ifdef _MSC_VER
#   pragma warning(pop)
#endif

#undef _C4_PUSH_HPP_

#endif /* _C4_PUSH_HPP_ */
