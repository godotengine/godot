#ifndef _C4_STD_VECTOR_FWD_HPP_
#define _C4_STD_VECTOR_FWD_HPP_

/** @file vector_fwd.hpp */

#include <cstddef>

// forward declarations for std::vector
#if defined(__GLIBCXX__) || defined(__GLIBCPP__) || defined(_MSC_VER)
#if defined(_MSC_VER)
__pragma(warning(push))
__pragma(warning(disable : 4643))
#endif
namespace std {
template<typename> class allocator;
#ifdef _GLIBCXX_DEBUG
inline namespace __debug {
template<typename T, typename Alloc> class vector;
}
#else
template<typename T, typename Alloc> class vector;
#endif
} // namespace std
#if defined(_MSC_VER)
__pragma(warning(pop))
#endif
#elif defined(_LIBCPP_ABI_NAMESPACE)
namespace std {
inline namespace _LIBCPP_ABI_NAMESPACE {
template<typename> class allocator;
template<typename T, typename Alloc> class vector;
} // namespace _LIBCPP_ABI_NAMESPACE
} // namespace std
#else
#error "unknown standard library"
#endif

#ifndef C4CORE_SINGLE_HEADER
#include "../substr_fwd.hpp"
#endif

namespace c4 {

template<class Alloc> c4::substr to_substr(std::vector<char, Alloc> &vec);
template<class Alloc> c4::csubstr to_csubstr(std::vector<char, Alloc> const& vec);

template<class Alloc> bool operator!= (c4::csubstr ss, std::vector<char, Alloc> const& s);
template<class Alloc> bool operator== (c4::csubstr ss, std::vector<char, Alloc> const& s);
template<class Alloc> bool operator>= (c4::csubstr ss, std::vector<char, Alloc> const& s);
template<class Alloc> bool operator>  (c4::csubstr ss, std::vector<char, Alloc> const& s);
template<class Alloc> bool operator<= (c4::csubstr ss, std::vector<char, Alloc> const& s);
template<class Alloc> bool operator<  (c4::csubstr ss, std::vector<char, Alloc> const& s);

template<class Alloc> bool operator!= (std::vector<char, Alloc> const& s, c4::csubstr ss);
template<class Alloc> bool operator== (std::vector<char, Alloc> const& s, c4::csubstr ss);
template<class Alloc> bool operator>= (std::vector<char, Alloc> const& s, c4::csubstr ss);
template<class Alloc> bool operator>  (std::vector<char, Alloc> const& s, c4::csubstr ss);
template<class Alloc> bool operator<= (std::vector<char, Alloc> const& s, c4::csubstr ss);
template<class Alloc> bool operator<  (std::vector<char, Alloc> const& s, c4::csubstr ss);

template<class Alloc> size_t to_chars(c4::substr buf, std::vector<char, Alloc> const& s);
template<class Alloc> bool from_chars(c4::csubstr buf, std::vector<char, Alloc> * s);

} // namespace c4

#endif // _C4_STD_VECTOR_FWD_HPP_
