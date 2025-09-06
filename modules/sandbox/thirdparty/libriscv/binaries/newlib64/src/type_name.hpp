#pragma once
#include <type_traits>
#include <typeinfo>
#include <memory>
#include <string>

#include <cstddef>
#include <stdexcept>
#include <cstring>

class static_string
{
    const char* const p_;
    const std::size_t sz_;

public:
    typedef const char* const_iterator;

    template <std::size_t N>
    constexpr static_string(const char(&a)[N]) noexcept
        : p_(a)
        , sz_(N-1)
        {}

    constexpr static_string(const char* p, std::size_t N) noexcept
        : p_(p)
        , sz_(N)
        {}

    constexpr const char* data() const noexcept {return p_;}
    constexpr std::size_t size() const noexcept {return sz_;}

	std::string to_string() const { return std::string(p_, sz_); }

    constexpr const_iterator begin() const noexcept {return p_;}
    constexpr const_iterator end()   const noexcept {return p_ + sz_;}

    constexpr char operator[](std::size_t n) const
    {
        return n < sz_ ? p_[n] : throw std::out_of_range("static_string");
    }
};

template <class T>
constexpr inline static_string type_name()
{
#ifdef __clang__
    static_string p = __PRETTY_FUNCTION__;
    return static_string(p.data() + 31, p.size() - 31 - 1);
#elif defined(__GNUC__)
    static_string p = __PRETTY_FUNCTION__;
#  if __cplusplus < 201402
    return static_string(p.data() + 36, p.size() - 36 - 1);
#  else
    return static_string(p.data() + 46, p.size() - 46 - 1);
#  endif
#elif defined(_MSC_VER)
    static_string p = __FUNCSIG__;
    return static_string(p.data() + 38, p.size() - 38 - 7);
#endif
}

#define TYPE_NAME(x) type_name<decltype(x)> ()
