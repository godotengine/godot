#ifndef _C4_STD_STRING_VIEW_HPP_
#define _C4_STD_STRING_VIEW_HPP_

/** @file string_view.hpp */

#ifndef C4CORE_SINGLE_HEADER
#include "../language.hpp"
#endif

#if (C4_CPP >= 17 && defined(__cpp_lib_string_view))

#ifndef C4CORE_SINGLE_HEADER
#include "../substr.hpp"
#endif

#include <string_view>


namespace c4 {

//-----------------------------------------------------------------------------

/** create a csubstr from an existing std::string_view. */
C4_ALWAYS_INLINE c4::csubstr to_csubstr(std::string_view s) noexcept
{
    return c4::csubstr(s.data(), s.size());
}


//-----------------------------------------------------------------------------

C4_ALWAYS_INLINE bool operator== (c4::csubstr ss, std::string_view s) { return ss.compare(s.data(), s.size()) == 0; }
C4_ALWAYS_INLINE bool operator!= (c4::csubstr ss, std::string_view s) { return ss.compare(s.data(), s.size()) != 0; }
C4_ALWAYS_INLINE bool operator>= (c4::csubstr ss, std::string_view s) { return ss.compare(s.data(), s.size()) >= 0; }
C4_ALWAYS_INLINE bool operator>  (c4::csubstr ss, std::string_view s) { return ss.compare(s.data(), s.size()) >  0; }
C4_ALWAYS_INLINE bool operator<= (c4::csubstr ss, std::string_view s) { return ss.compare(s.data(), s.size()) <= 0; }
C4_ALWAYS_INLINE bool operator<  (c4::csubstr ss, std::string_view s) { return ss.compare(s.data(), s.size()) <  0; }

C4_ALWAYS_INLINE bool operator== (std::string_view s, c4::csubstr ss) { return ss.compare(s.data(), s.size()) == 0; }
C4_ALWAYS_INLINE bool operator!= (std::string_view s, c4::csubstr ss) { return ss.compare(s.data(), s.size()) != 0; }
C4_ALWAYS_INLINE bool operator<= (std::string_view s, c4::csubstr ss) { return ss.compare(s.data(), s.size()) >= 0; }
C4_ALWAYS_INLINE bool operator<  (std::string_view s, c4::csubstr ss) { return ss.compare(s.data(), s.size()) >  0; }
C4_ALWAYS_INLINE bool operator>= (std::string_view s, c4::csubstr ss) { return ss.compare(s.data(), s.size()) <= 0; }
C4_ALWAYS_INLINE bool operator>  (std::string_view s, c4::csubstr ss) { return ss.compare(s.data(), s.size()) <  0; }


//-----------------------------------------------------------------------------

/** copy an std::string_view to a writeable substr */
inline size_t to_chars(c4::substr buf, std::string_view s)
{
    C4_ASSERT(!buf.overlaps(to_csubstr(s)));
    size_t sz = s.size();
    size_t len = buf.len < sz ? buf.len : sz;
    // calling memcpy with null strings is undefined behavior
    // and will wreak havoc in calling code's branches.
    // see https://github.com/biojppm/rapidyaml/pull/264#issuecomment-1262133637
    if(len)
    {
        C4_ASSERT(s.data() != nullptr);
        C4_ASSERT(buf.str != nullptr);
        memcpy(buf.str, s.data(), len);
    }
    return sz; // return the number of needed chars
}

} // namespace c4

#endif // C4_STRING_VIEW_AVAILABLE

#endif // _C4_STD_STRING_VIEW_HPP_
