#ifndef _C4_STD_STRING_FWD_HPP_
#define _C4_STD_STRING_FWD_HPP_

/** @file string_fwd.hpp */

#ifndef DOXYGEN

#ifndef C4CORE_SINGLE_HEADER
#include "../substr_fwd.hpp"
#endif

#include <cstddef>

// forward declarations for std::string
#if defined(__GLIBCXX__) || defined(__GLIBCPP__)
#include <bits/stringfwd.h>  // use the fwd header in glibcxx
#elif defined(_LIBCPP_VERSION) || defined(__APPLE_CC__)
#include <iosfwd>  // use the fwd header in stdlibc++
#elif defined(_MSC_VER)
#include "../error.hpp"
//! @todo is there a fwd header in msvc?
namespace std {
C4_SUPPRESS_WARNING_MSVC_WITH_PUSH(4643) // Forward declaring 'char_traits' in namespace std is not permitted by the C++ Standard.
template<typename> struct char_traits;
template<typename> class allocator;
template<typename _CharT, typename _Traits, typename _Alloc> class basic_string;
using string = basic_string<char, char_traits<char>, allocator<char>>;
C4_SUPPRESS_WARNING_MSVC_POP
} /* namespace std */
#else
#error "unknown standard library"
#endif

namespace c4 {

c4::substr to_substr(std::string &s) noexcept;
c4::csubstr to_csubstr(std::string const& s) noexcept;

bool operator== (c4::csubstr ss, std::string const& s);
bool operator!= (c4::csubstr ss, std::string const& s);
bool operator>= (c4::csubstr ss, std::string const& s);
bool operator>  (c4::csubstr ss, std::string const& s);
bool operator<= (c4::csubstr ss, std::string const& s);
bool operator<  (c4::csubstr ss, std::string const& s);

bool operator== (std::string const& s, c4::csubstr ss);
bool operator!= (std::string const& s, c4::csubstr ss);
bool operator>= (std::string const& s, c4::csubstr ss);
bool operator>  (std::string const& s, c4::csubstr ss);
bool operator<= (std::string const& s, c4::csubstr ss);
bool operator<  (std::string const& s, c4::csubstr ss);


} // namespace c4

#endif // DOXYGEN
#endif // _C4_STD_STRING_FWD_HPP_
