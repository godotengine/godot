#ifndef _C4_EXT_FAST_FLOAT_HPP_
#define _C4_EXT_FAST_FLOAT_HPP_

#if defined(_MSC_VER) && !defined(__clang__)
#   pragma warning(push)
#   pragma warning(disable: 4365) // '=': conversion from 'const _Ty' to 'fast_float::limb', signed/unsigned mismatch
#   pragma warning(disable: 4996) // snprintf/scanf: this function or variable may be unsafe
#elif defined(__clang__) || defined(__APPLE_CC__) || defined(_LIBCPP_VERSION)
#   pragma clang diagnostic push
#   if (defined(__clang_major__) && (__clang_major__ >= 9)) || defined(__APPLE_CC__)
#       pragma clang diagnostic ignored "-Wfortify-source"
#   endif
#   pragma clang diagnostic ignored "-Wshift-count-overflow"
#   pragma clang diagnostic ignored "-Wold-style-cast"
#elif defined(__GNUC__)
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wuseless-cast"
#   pragma GCC diagnostic ignored "-Wold-style-cast"
#endif

#include "fast_float_all.h"

#ifdef _MSC_VER
#   pragma warning(pop)
#elif defined(__clang__) || defined(__APPLE_CC__)
#   pragma clang diagnostic pop
#elif defined(__GNUC__)
#   pragma GCC diagnostic pop
#endif

#endif // _C4_EXT_FAST_FLOAT_HPP_
