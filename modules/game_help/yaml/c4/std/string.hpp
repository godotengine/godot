#ifndef _C4_STD_STRING_HPP_
#define _C4_STD_STRING_HPP_

/** @file string.hpp */

#ifndef C4CORE_SINGLE_HEADER
#include "../substr.hpp"
#endif

#include <string>

namespace c4 {

//-----------------------------------------------------------------------------

/** get a writeable view to an existing std::string.
 * When the string is empty, the returned view will be pointing
 * at the character with value '\0', but the size will be zero.
 * @see https://en.cppreference.com/w/cpp/string/basic_string/operator_at
 */
C4_ALWAYS_INLINE c4::substr to_substr(std::string &s) noexcept
{
    #if C4_CPP < 11
    #error this function will have undefined behavior
    #endif
    // since c++11 it is legal to call s[s.size()].
    return c4::substr(&s[0], s.size());
}

/** get a readonly view to an existing std::string.
 * When the string is empty, the returned view will be pointing
 * at the character with value '\0', but the size will be zero.
 * @see https://en.cppreference.com/w/cpp/string/basic_string/operator_at
 */
C4_ALWAYS_INLINE c4::csubstr to_csubstr(std::string const& s) noexcept
{
    #if C4_CPP < 11
    #error this function will have undefined behavior
    #endif
    // since c++11 it is legal to call s[s.size()].
    return c4::csubstr(&s[0], s.size());
}

//-----------------------------------------------------------------------------

C4_ALWAYS_INLINE bool operator== (c4::csubstr ss, std::string const& s) { return ss.compare(to_csubstr(s)) == 0; }
C4_ALWAYS_INLINE bool operator!= (c4::csubstr ss, std::string const& s) { return ss.compare(to_csubstr(s)) != 0; }
C4_ALWAYS_INLINE bool operator>= (c4::csubstr ss, std::string const& s) { return ss.compare(to_csubstr(s)) >= 0; }
C4_ALWAYS_INLINE bool operator>  (c4::csubstr ss, std::string const& s) { return ss.compare(to_csubstr(s)) >  0; }
C4_ALWAYS_INLINE bool operator<= (c4::csubstr ss, std::string const& s) { return ss.compare(to_csubstr(s)) <= 0; }
C4_ALWAYS_INLINE bool operator<  (c4::csubstr ss, std::string const& s) { return ss.compare(to_csubstr(s)) <  0; }

C4_ALWAYS_INLINE bool operator== (std::string const& s, c4::csubstr ss) { return ss.compare(to_csubstr(s)) == 0; }
C4_ALWAYS_INLINE bool operator!= (std::string const& s, c4::csubstr ss) { return ss.compare(to_csubstr(s)) != 0; }
C4_ALWAYS_INLINE bool operator>= (std::string const& s, c4::csubstr ss) { return ss.compare(to_csubstr(s)) <= 0; }
C4_ALWAYS_INLINE bool operator>  (std::string const& s, c4::csubstr ss) { return ss.compare(to_csubstr(s)) <  0; }
C4_ALWAYS_INLINE bool operator<= (std::string const& s, c4::csubstr ss) { return ss.compare(to_csubstr(s)) >= 0; }
C4_ALWAYS_INLINE bool operator<  (std::string const& s, c4::csubstr ss) { return ss.compare(to_csubstr(s)) >  0; }

//-----------------------------------------------------------------------------

/** copy an std::string to a writeable string view */
inline size_t to_chars(c4::substr buf, std::string const& s)
{
    C4_ASSERT(!buf.overlaps(to_csubstr(s)));
    size_t len = buf.len < s.size() ? buf.len : s.size();
    // calling memcpy with null strings is undefined behavior
    // and will wreak havoc in calling code's branches.
    // see https://github.com/biojppm/rapidyaml/pull/264#issuecomment-1262133637
    if(len)
    {
        C4_ASSERT(s.data() != nullptr);
        C4_ASSERT(buf.str != nullptr);
        memcpy(buf.str, s.data(), len);
    }
    return s.size(); // return the number of needed chars
}

/** copy a string view to an existing std::string */
inline bool from_chars(c4::csubstr buf, std::string * s)
{
    s->resize(buf.len);
    C4_ASSERT(!buf.overlaps(to_csubstr(*s)));
    // calling memcpy with null strings is undefined behavior
    // and will wreak havoc in calling code's branches.
    // see https://github.com/biojppm/rapidyaml/pull/264#issuecomment-1262133637
    if(buf.len)
    {
        C4_ASSERT(buf.str != nullptr);
        memcpy(&(*s)[0], buf.str, buf.len);
    }
    return true;
}

} // namespace c4

#endif // _C4_STD_STRING_HPP_
