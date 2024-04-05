#ifndef _C4_GCC_4_8_HPP_
#define _C4_GCC_4_8_HPP_

#if __GNUC__ == 4 && __GNUC_MINOR__ >= 8
/* STL polyfills for old GNU compilers */

_Pragma("GCC diagnostic ignored \"-Wshadow\"")
_Pragma("GCC diagnostic ignored \"-Wmissing-field-initializers\"")

#if __cplusplus
#include <cstdint>
#include <type_traits>

namespace std {

template<typename _Tp>
struct is_trivially_copyable : public integral_constant<bool,
    is_destructible<_Tp>::value && __has_trivial_destructor(_Tp) &&
    (__has_trivial_constructor(_Tp) || __has_trivial_copy(_Tp) || __has_trivial_assign(_Tp))>
{ };

template<typename _Tp>
using is_trivially_copy_constructible = has_trivial_copy_constructor<_Tp>;

template<typename _Tp>
using is_trivially_default_constructible = has_trivial_default_constructor<_Tp>;

template<typename _Tp>
using is_trivially_copy_assignable = has_trivial_copy_assign<_Tp>;

/* not supported */
template<typename _Tp>
struct is_trivially_move_constructible : false_type
{ };

/* not supported */
template<typename _Tp>
struct is_trivially_move_assignable : false_type
{ };

inline void *align(size_t __align, size_t __size, void*& __ptr, size_t& __space) noexcept
{
    if (__space < __size)
        return nullptr;
    const auto __intptr = reinterpret_cast<uintptr_t>(__ptr);
    const auto __aligned = (__intptr - 1u + __align) & -__align;
    const auto __diff = __aligned - __intptr;
    if (__diff > (__space - __size))
        return nullptr;
    else
    {
        __space -= __diff;
        return __ptr = reinterpret_cast<void*>(__aligned);
    }
}

#if __GNUC__ == 4 && __GNUC_MINOR__ == 8
typedef long double max_align_t ;
#endif

}
#else // __cplusplus

#include <string.h>
// see https://sourceware.org/bugzilla/show_bug.cgi?id=25399 (ubuntu gcc-4.8)
#define memset(s, c, count) __builtin_memset(s, c, count)

#endif // __cplusplus

#endif // __GNUC__ == 4 && __GNUC_MINOR__ >= 8

#endif // _C4_GCC_4_8_HPP_
