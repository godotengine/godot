#ifndef _C4_PUSH_HPP_
#define _C4_PUSH_HPP_


/** @file c4_push.hpp enables macros and warning control directives
 * needed by c4core. This is implemented in a push/pop way.
 * @see c4_pop.hpp */


#ifndef _C4_CONFIG_HPP_
#include "config.hpp"
#endif

#include "restrict.hpp"

#ifdef C4_WIN
#   include "windows_push.hpp"
#endif

#ifdef _MSC_VER
#   pragma warning(push)
#   pragma warning(disable : 4068) // unknown pragma
#   pragma warning(disable : 4100) // unreferenced formal parameter
#   pragma warning(disable : 4127) // conditional expression is constant -- eg  do {} while(1);
#   pragma warning(disable : 4201) // nonstandard extension used : nameless struct/union
//#   pragma warning(disable : 4238) // nonstandard extension used: class rvalue used as lvalue
#   pragma warning(disable : 4244)
#   pragma warning(disable : 4503) // decorated name length exceeded, name was truncated
#   pragma warning(disable : 4702) // unreachable code
#   pragma warning(disable : 4714) // function marked as __forceinline not inlined
#   pragma warning(disable : 4996) // 'strncpy', fopen, etc: This function or variable may be unsafe
#   if C4_MSVC_VERSION != C4_MSVC_VERSION_2017
#       pragma warning(disable : 4800) // forcing value to bool 'true' or 'false' (performance warning)
#   endif
#endif

#endif /* _C4_PUSH_HPP_ */
