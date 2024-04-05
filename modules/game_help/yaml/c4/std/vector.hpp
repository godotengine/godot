#ifndef _C4_STD_VECTOR_HPP_
#define _C4_STD_VECTOR_HPP_

/** @file vector.hpp provides conversion and comparison facilities
 * from/between std::vector<char> to c4::substr and c4::csubstr.
 * @todo add to_span() and friends
 */

#ifndef C4CORE_SINGLE_HEADER
#include "../substr.hpp"
#endif

#include <vector>

namespace c4 {

//-----------------------------------------------------------------------------

/** get a substr (writeable string view) of an existing std::vector<char> */
template<class Alloc>
c4::substr to_substr(std::vector<char, Alloc> &vec)
{
    char *data = vec.empty() ? nullptr : vec.data(); // data() may or may not return a null pointer.
    return c4::substr(data, vec.size());
}

/** get a csubstr (read-only string) view of an existing std::vector<char> */
template<class Alloc>
c4::csubstr to_csubstr(std::vector<char, Alloc> const& vec)
{
    const char *data = vec.empty() ? nullptr : vec.data(); // data() may or may not return a null pointer.
    return c4::csubstr(data, vec.size());
}

//-----------------------------------------------------------------------------
// comparisons between substrings and std::vector<char>

template<class Alloc> C4_ALWAYS_INLINE bool operator!= (c4::csubstr ss, std::vector<char, Alloc> const& s) { return ss != to_csubstr(s); }
template<class Alloc> C4_ALWAYS_INLINE bool operator== (c4::csubstr ss, std::vector<char, Alloc> const& s) { return ss == to_csubstr(s); }
template<class Alloc> C4_ALWAYS_INLINE bool operator>= (c4::csubstr ss, std::vector<char, Alloc> const& s) { return ss >= to_csubstr(s); }
template<class Alloc> C4_ALWAYS_INLINE bool operator>  (c4::csubstr ss, std::vector<char, Alloc> const& s) { return ss >  to_csubstr(s); }
template<class Alloc> C4_ALWAYS_INLINE bool operator<= (c4::csubstr ss, std::vector<char, Alloc> const& s) { return ss <= to_csubstr(s); }
template<class Alloc> C4_ALWAYS_INLINE bool operator<  (c4::csubstr ss, std::vector<char, Alloc> const& s) { return ss <  to_csubstr(s); }

template<class Alloc> C4_ALWAYS_INLINE bool operator!= (std::vector<char, Alloc> const& s, c4::csubstr ss) { return ss != to_csubstr(s); }
template<class Alloc> C4_ALWAYS_INLINE bool operator== (std::vector<char, Alloc> const& s, c4::csubstr ss) { return ss == to_csubstr(s); }
template<class Alloc> C4_ALWAYS_INLINE bool operator>= (std::vector<char, Alloc> const& s, c4::csubstr ss) { return ss <= to_csubstr(s); }
template<class Alloc> C4_ALWAYS_INLINE bool operator>  (std::vector<char, Alloc> const& s, c4::csubstr ss) { return ss <  to_csubstr(s); }
template<class Alloc> C4_ALWAYS_INLINE bool operator<= (std::vector<char, Alloc> const& s, c4::csubstr ss) { return ss >= to_csubstr(s); }
template<class Alloc> C4_ALWAYS_INLINE bool operator<  (std::vector<char, Alloc> const& s, c4::csubstr ss) { return ss >  to_csubstr(s); }

//-----------------------------------------------------------------------------

/** copy a std::vector<char> to a writeable string view */
template<class Alloc>
inline size_t to_chars(c4::substr buf, std::vector<char, Alloc> const& s)
{
    C4_ASSERT(!buf.overlaps(to_csubstr(s)));
    size_t len = buf.len < s.size() ? buf.len : s.size();
    // calling memcpy with null strings is undefined behavior
    // and will wreak havoc in calling code's branches.
    // see https://github.com/biojppm/rapidyaml/pull/264#issuecomment-1262133637
    if(len > 0)
    {
        memcpy(buf.str, s.data(), len);
    }
    return s.size(); // return the number of needed chars
}

/** copy a string view to an existing std::vector<char> */
template<class Alloc>
inline bool from_chars(c4::csubstr buf, std::vector<char, Alloc> * s)
{
    s->resize(buf.len);
    C4_ASSERT(!buf.overlaps(to_csubstr(*s)));
    // calling memcpy with null strings is undefined behavior
    // and will wreak havoc in calling code's branches.
    // see https://github.com/biojppm/rapidyaml/pull/264#issuecomment-1262133637
    if(buf.len > 0)
    {
        memcpy(&(*s)[0], buf.str, buf.len);
    }
    return true;
}

} // namespace c4

#endif // _C4_STD_VECTOR_HPP_
