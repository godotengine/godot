#ifndef _C4_CHAR_TRAITS_HPP_
#define _C4_CHAR_TRAITS_HPP_

#include "config.hpp"

#include <string> // needed because of std::char_traits
#include <cctype>
#include <cwctype>

namespace c4 {

C4_ALWAYS_INLINE bool isspace(char c) { return std::isspace(c) != 0; }
C4_ALWAYS_INLINE bool isspace(wchar_t c) { return std::iswspace(static_cast<wint_t>(c)) != 0; }

//-----------------------------------------------------------------------------
template<typename C>
struct char_traits;

template<>
struct char_traits<char> : public std::char_traits<char>
{
    constexpr static const char whitespace_chars[] = " \f\n\r\t\v";
    constexpr static const size_t num_whitespace_chars = sizeof(whitespace_chars) - 1;
};

template<>
struct char_traits<wchar_t> : public std::char_traits<wchar_t>
{
    constexpr static const wchar_t whitespace_chars[] = L" \f\n\r\t\v";
    constexpr static const size_t num_whitespace_chars = sizeof(whitespace_chars) - 1;
};


//-----------------------------------------------------------------------------
namespace detail {
template<typename C>
struct needed_chars;
template<>
struct needed_chars<char>
{
    template<class SizeType>
    C4_ALWAYS_INLINE constexpr static SizeType for_bytes(SizeType num_bytes)
    {
        return num_bytes;
    }
};
template<>
struct needed_chars<wchar_t>
{
    template<class SizeType>
    C4_ALWAYS_INLINE constexpr static SizeType for_bytes(SizeType num_bytes)
    {
        // wchar_t is not necessarily 2 bytes.
        return (num_bytes / static_cast<SizeType>(sizeof(wchar_t))) + ((num_bytes & static_cast<SizeType>(SizeType(sizeof(wchar_t)) - SizeType(1))) != 0);
    }
};
} // namespace detail

/** get the number of C characters needed to store a number of bytes */
template<typename C, typename SizeType>
C4_ALWAYS_INLINE constexpr SizeType num_needed_chars(SizeType num_bytes)
{
    return detail::needed_chars<C>::for_bytes(num_bytes);
}


//-----------------------------------------------------------------------------

/** get the given text string as either char or wchar_t according to the given type */
#define C4_TXTTY(txt, type) \
    /* is there a smarter way to do this? */\
    c4::detail::literal_as<type>::get(txt, C4_WIDEN(txt))

namespace detail {
template<typename C>
struct literal_as;

template<>
struct literal_as<char>
{
    C4_ALWAYS_INLINE static constexpr const char* get(const char* str, const wchar_t *)
    {
        return str;
    }
};
template<>
struct literal_as<wchar_t>
{
    C4_ALWAYS_INLINE static constexpr const wchar_t* get(const char*, const wchar_t *wstr)
    {
        return wstr;
    }
};
} // namespace detail

} // namespace c4

#endif /* _C4_CHAR_TRAITS_HPP_ */
