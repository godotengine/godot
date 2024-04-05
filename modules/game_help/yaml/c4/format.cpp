#include "format.hpp"

#include <memory> // for std::align

#ifdef __clang__
#   pragma clang diagnostic push
#   pragma clang diagnostic ignored "-Wformat-nonliteral"
#   pragma clang diagnostic ignored "-Wold-style-cast"
#elif defined(__GNUC__)
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wformat-nonliteral"
#   pragma GCC diagnostic ignored "-Wold-style-cast"
#endif

namespace c4 {


size_t to_chars(substr buf, fmt::const_raw_wrapper r)
{
    void * vptr = buf.str;
    size_t space = buf.len;
    auto ptr = (decltype(buf.str)) std::align(r.alignment, r.len, vptr, space);
    if(ptr == nullptr)
    {
        // if it was not possible to align, return a conservative estimate
        // of the required space
        return r.alignment + r.len;
    }
    C4_CHECK(ptr >= buf.begin() && ptr <= buf.end());
    size_t sz = static_cast<size_t>(ptr - buf.str) + r.len;
    if(sz <= buf.len)
    {
        memcpy(ptr, r.buf, r.len);
    }
    return sz;
}


bool from_chars(csubstr buf, fmt::raw_wrapper *r)
{
    C4_SUPPRESS_WARNING_GCC_WITH_PUSH("-Wcast-qual")
    void * vptr = (void*)buf.str;
    C4_SUPPRESS_WARNING_GCC_POP
    size_t space = buf.len;
    auto ptr = (decltype(buf.str)) std::align(r->alignment, r->len, vptr, space);
    C4_CHECK(ptr != nullptr);
    C4_CHECK(ptr >= buf.begin() && ptr <= buf.end());
    C4_SUPPRESS_WARNING_GCC_PUSH
    #if defined(__GNUC__) && __GNUC__ > 9
    C4_SUPPRESS_WARNING_GCC("-Wanalyzer-null-argument")
    #endif
    memcpy(r->buf, ptr, r->len);
    C4_SUPPRESS_WARNING_GCC_POP
    return true;
}


} // namespace c4

#ifdef __clang__
#   pragma clang diagnostic pop
#elif defined(__GNUC__)
#   pragma GCC diagnostic pop
#endif
