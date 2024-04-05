#ifndef _C4_TYPENAME_HPP_
#define _C4_TYPENAME_HPP_

/** @file type_name.hpp compile-time type name */

#include "span.hpp"
#include "compiler.hpp"

/// @cond dev
struct _c4t
{
    const char *str;
    size_t sz;
    template<size_t N>
    constexpr _c4t(const char (&s)[N]) : str(s), sz(N-1) {} // take off the \0
};
// this is a more abbreviated way of getting the type name
// (if we used span in the return type, the name would involve
// templates and would create longer type name strings,
// as well as larger differences between compilers)
template<class T>
C4_CONSTEXPR14 C4_ALWAYS_INLINE
_c4t _c4tn()
{
    auto p = _c4t(C4_PRETTY_FUNC);
    return p;
}
/// @endcond


namespace c4 {

/** compile-time type name
 * @see http://stackoverflow.com/a/20170989/5875572 */
template<class T>
C4_CONSTEXPR14 cspan<char> type_name()
{
    const _c4t p = _c4tn<T>();

#if (0) // enable this to debug and find the offsets
    for(size_t index = 0; index < p.sz; ++index)
        printf(" %2c", p.str[index]);
    printf("\n");
    for(size_t index = 0; index < p.sz; ++index)
        printf(" %2zu", index);
    printf("\n");
#endif

#if defined(_MSC_VER)
#   if defined(__clang__) // Visual Studio has the clang toolset
#   if (_MSC_VER >= 1930) // do not use this: defined(C4_MSVC_2022)
    // ..............................xxx.
    // _c4t __cdecl _c4tn(void) [T = int]
    enum : size_t { tstart = 30, tend = 1};
#   else
    // example:
    // ..........................xxx.
    // _c4t __cdecl _c4tn() [T = int]
    enum : size_t { tstart = 26, tend = 1};
#   endif
#   elif defined(C4_MSVC_2015) || defined(C4_MSVC_2017) || defined(C4_MSVC_2019) || defined(C4_MSVC_2022)
    // Note: subtract 7 at the end because the function terminates with ">(void)" in VS2015+
    cspan<char>::size_type tstart = 26, tend = 7;

    const char *C4_RESTRICT s = p.str + tstart; // look at the start

    // we're not using strcmp() or memcmp() to spare the #include

    // does it start with 'class '?
    if(p.sz > 6 && s[0] == 'c' && s[1] == 'l' && s[2] == 'a' && s[3] == 's' && s[4] == 's' && s[5] == ' ')
    {
        tstart += 6;
    }
    // does it start with 'struct '?
    else if(p.sz > 7 && s[0] == 's' && s[1] == 't' && s[2] == 'r' && s[3] == 'u' && s[4] == 'c' && s[5] == 't' && s[6] == ' ')
    {
        tstart += 7;
    }

#   else
    C4_NOT_IMPLEMENTED();
#   endif

#elif defined(__ICC)
    // example:
    // ........................xxx.
    // "_c4t _c4tn() [with T = int]"
    enum : size_t { tstart = 23, tend = 1};

#elif defined(__clang__)
    // example:
    // ...................xxx.
    // "_c4t _c4tn() [T = int]"
    enum : size_t { tstart = 18, tend = 1};

#elif defined(__GNUC__)
    #if __GNUC__ >= 7 && C4_CPP >= 14
        // example:
        // ..................................xxx.
        // "constexpr _c4t _c4tn() [with T = int]"
        enum : size_t { tstart = 33, tend = 1 };
    #else
        // example:
        // ........................xxx.
        // "_c4t _c4tn() [with T = int]"
        enum : size_t { tstart = 23, tend = 1 };
    #endif
#else
    C4_NOT_IMPLEMENTED();
#endif

    cspan<char> o(p.str + tstart, p.sz - tstart - tend);

    return o;
}

/** compile-time type name
 * @overload */
template<class T>
C4_CONSTEXPR14 C4_ALWAYS_INLINE cspan<char> type_name(T const&)
{
    return type_name<T>();
}

} // namespace c4

#endif //_C4_TYPENAME_HPP_
