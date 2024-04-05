#ifndef _C4_SUBSTR_HPP_
#define _C4_SUBSTR_HPP_

/** @file substr.hpp read+write string views */

#include <string.h>
#include <ctype.h>
#include <type_traits>

#include "config.hpp"
#include "error.hpp"
#include "substr_fwd.hpp"

#ifdef __clang__
#   pragma clang diagnostic push
#   pragma clang diagnostic ignored "-Wold-style-cast"
#elif defined(__GNUC__)
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wtype-limits" // disable warnings on size_t>=0, used heavily in assertions below. These assertions are a preparation step for providing the index type as a template parameter.
#   pragma GCC diagnostic ignored "-Wuseless-cast"
#   pragma GCC diagnostic ignored "-Wold-style-cast"
#endif


namespace c4 {


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

namespace detail {

template<typename C>
static inline void _do_reverse(C *C4_RESTRICT first, C *C4_RESTRICT last)
{
    while(last > first)
    {
        C tmp = *last;
        *last-- = *first;
        *first++ = tmp;
    }
}

} // namespace detail


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

// utility macros to deuglify SFINAE code; undefined after the class.
// https://stackoverflow.com/questions/43051882/how-to-disable-a-class-member-funrtion-for-certain-template-types
#define C4_REQUIRE_RW(ret_type) \
    template <typename U=C> \
    typename std::enable_if< ! std::is_const<U>::value, ret_type>::type


/** a non-owning string-view, consisting of a character pointer
 * and a length.
 *
 * @note The pointer is explicitly restricted.
 *
 * @see to_substr()
 * @see to_csubstr()
 */
template<class C>
struct C4CORE_EXPORT basic_substring
{
public:

    /** a restricted pointer to the first character of the substring */
    C * C4_RESTRICT str;
    /** the length of the substring */
    size_t          len;

public:

    /** @name Types */
    /** @{ */

    using  CC  = typename std::add_const<C>::type;     //!< CC=const char
    using NCC_ = typename std::remove_const<C>::type; //!< NCC_=non const char

    using ro_substr = basic_substring<CC>;
    using rw_substr = basic_substring<NCC_>;

    using char_type = C;
    using size_type = size_t;

    using iterator = C*;
    using const_iterator = CC*;

    enum : size_t { npos = (size_t)-1, NONE = (size_t)-1 };

    /// convert automatically to substring of const C
    template<class U=C>
    C4_ALWAYS_INLINE operator typename std::enable_if<!std::is_const<U>::value, ro_substr const&>::type () const noexcept
    {
        return *(ro_substr const*)this; // don't call the str+len ctor because it does a check
    }

    /** @} */

public:

    /** @name Default construction and assignment */
    /** @{ */

    C4_ALWAYS_INLINE constexpr basic_substring() noexcept : str(), len() {}

    C4_ALWAYS_INLINE basic_substring(basic_substring const&) noexcept = default;
    C4_ALWAYS_INLINE basic_substring(basic_substring     &&) noexcept = default;
    C4_ALWAYS_INLINE basic_substring(std::nullptr_t) noexcept : str(nullptr), len(0) {}

    C4_ALWAYS_INLINE basic_substring& operator= (basic_substring const&) noexcept = default;
    C4_ALWAYS_INLINE basic_substring& operator= (basic_substring     &&) noexcept = default;
    C4_ALWAYS_INLINE basic_substring& operator= (std::nullptr_t) noexcept { str = nullptr; len = 0; return *this; }

    C4_ALWAYS_INLINE void clear() noexcept { str = nullptr; len = 0; }

    /** @} */

public:

    /** @name Construction and assignment from characters with the same type */
    /** @{ */

    /** Construct from an array.
     * @warning the input string need not be zero terminated, but the
     * length is taken as if the string was zero terminated */
    template<size_t N>
    C4_ALWAYS_INLINE constexpr basic_substring(C (&s_)[N]) noexcept : str(s_), len(N-1) {}
    /** Construct from a pointer and length.
     * @warning the input string need not be zero terminated. */
    C4_ALWAYS_INLINE basic_substring(C *s_, size_t len_) noexcept : str(s_), len(len_) { C4_ASSERT(str || !len_); }
    /** Construct from two pointers.
     * @warning the end pointer MUST BE larger than or equal to the begin pointer
     * @warning the input string need not be zero terminated */
    C4_ALWAYS_INLINE basic_substring(C *beg_, C *end_) noexcept : str(beg_), len(static_cast<size_t>(end_ - beg_)) { C4_ASSERT(end_ >= beg_); }
    /** Construct from a C-string (zero-terminated string)
     * @warning the input string MUST BE zero terminated.
     * @warning will call strlen()
     * @note this overload uses SFINAE to prevent it from overriding the array ctor
     * @see For a more detailed explanation on why the plain overloads cannot
     * coexist, see http://cplusplus.bordoon.com/specializeForCharacterArrays.html */
    template<class U, typename std::enable_if<std::is_same<U, C*>::value || std::is_same<U, NCC_*>::value, int>::type=0>
    C4_ALWAYS_INLINE basic_substring(U s_) noexcept : str(s_), len(s_ ? strlen(s_) : 0) {}

    /** Assign from an array.
     * @warning the input string need not be zero terminated, but the
     * length is taken as if the string was zero terminated */
    template<size_t N>
    C4_ALWAYS_INLINE void assign(C (&s_)[N]) noexcept { str = (s_); len = (N-1); }
    /** Assign from a pointer and length.
     * @warning the input string need not be zero terminated. */
    C4_ALWAYS_INLINE void assign(C *s_, size_t len_) noexcept { str = s_; len = len_; C4_ASSERT(str || !len_); }
    /** Assign from two pointers.
     * @warning the end pointer MUST BE larger than or equal to the begin pointer
     * @warning the input string need not be zero terminated. */
    C4_ALWAYS_INLINE void assign(C *beg_, C *end_) noexcept { C4_ASSERT(end_ >= beg_); str = (beg_); len = static_cast<size_t>(end_ - beg_); }
    /** Assign from a C-string (zero-terminated string)
     * @warning the input string must be zero terminated.
     * @warning will call strlen()
     * @note this overload uses SFINAE to prevent it from overriding the array ctor
     * @see For a more detailed explanation on why the plain overloads cannot
     * coexist, see http://cplusplus.bordoon.com/specializeForCharacterArrays.html */
    template<class U, typename std::enable_if<std::is_same<U, C*>::value || std::is_same<U, NCC_*>::value, int>::type=0>
    C4_ALWAYS_INLINE void assign(U s_) noexcept { str = (s_); len = (s_ ? strlen(s_) : 0); }

    /** Assign from an array.
     * @warning the input string need not be zero terminated. */
    template<size_t N>
    C4_ALWAYS_INLINE basic_substring& operator= (C (&s_)[N]) noexcept { str = (s_); len = (N-1); return *this; }
    /** Assign from a C-string (zero-terminated string)
     * @warning the input string MUST BE zero terminated.
     * @warning will call strlen()
     * @note this overload uses SFINAE to prevent it from overriding the array ctor
     * @see For a more detailed explanation on why the plain overloads cannot
     * coexist, see http://cplusplus.bordoon.com/specializeForCharacterArrays.html */
    template<class U, typename std::enable_if<std::is_same<U, C*>::value || std::is_same<U, NCC_*>::value, int>::type=0>
    C4_ALWAYS_INLINE basic_substring& operator= (U s_) noexcept { str = s_; len = s_ ? strlen(s_) : 0; return *this; }

    /** @} */

public:

    /** @name Standard accessor methods */
    /** @{ */

    C4_ALWAYS_INLINE C4_PURE bool   has_str()   const noexcept { return ! empty() && str[0] != C(0); }
    C4_ALWAYS_INLINE C4_PURE bool   empty()     const noexcept { return (len == 0 || str == nullptr); }
    C4_ALWAYS_INLINE C4_PURE bool   not_empty() const noexcept { return (len != 0 && str != nullptr); }
    C4_ALWAYS_INLINE C4_PURE size_t size()      const noexcept { return len; }

    C4_ALWAYS_INLINE C4_PURE iterator begin() noexcept { return str; }
    C4_ALWAYS_INLINE C4_PURE iterator end  () noexcept { return str + len; }

    C4_ALWAYS_INLINE C4_PURE const_iterator begin() const noexcept { return str; }
    C4_ALWAYS_INLINE C4_PURE const_iterator end  () const noexcept { return str + len; }

    C4_ALWAYS_INLINE C4_PURE C      * data()       noexcept { return str; }
    C4_ALWAYS_INLINE C4_PURE C const* data() const noexcept { return str; }

    C4_ALWAYS_INLINE C4_PURE C      & operator[] (size_t i)       noexcept { C4_ASSERT(i >= 0 && i < len); return str[i]; }
    C4_ALWAYS_INLINE C4_PURE C const& operator[] (size_t i) const noexcept { C4_ASSERT(i >= 0 && i < len); return str[i]; }

    C4_ALWAYS_INLINE C4_PURE C      & front()       noexcept { C4_ASSERT(len > 0 && str != nullptr); return *str; }
    C4_ALWAYS_INLINE C4_PURE C const& front() const noexcept { C4_ASSERT(len > 0 && str != nullptr); return *str; }

    C4_ALWAYS_INLINE C4_PURE C      & back()       noexcept { C4_ASSERT(len > 0 && str != nullptr); return *(str + len - 1); }
    C4_ALWAYS_INLINE C4_PURE C const& back() const noexcept { C4_ASSERT(len > 0 && str != nullptr); return *(str + len - 1); }

    /** @} */

public:

    /** @name Comparison methods */
    /** @{ */

    C4_PURE int compare(C const c) const noexcept
    {
        C4_XASSERT((str != nullptr) || len == 0);
        if(C4_LIKELY(str != nullptr && len > 0))
            return (*str != c) ? *str - c : (static_cast<int>(len) - 1);
        else
            return -1;
    }

    C4_PURE int compare(const char *C4_RESTRICT that, size_t sz) const noexcept
    {
        C4_XASSERT(that || sz  == 0);
        C4_XASSERT(str  || len == 0);
        if(C4_LIKELY(str && that))
        {
            {
                const size_t min = len < sz ? len : sz;
                for(size_t i = 0; i < min; ++i)
                    if(str[i] != that[i])
                        return str[i] < that[i] ? -1 : 1;
            }
            if(len < sz)
                return -1;
            else if(len == sz)
                return 0;
            else
                return 1;
        }
        else if(len == sz)
        {
            C4_XASSERT(len == 0 && sz == 0);
            return 0;
        }
        return len < sz ? -1 : 1;
    }

    C4_ALWAYS_INLINE C4_PURE int compare(ro_substr const that) const noexcept { return this->compare(that.str, that.len); }

    C4_ALWAYS_INLINE C4_PURE bool operator== (std::nullptr_t) const noexcept { return str == nullptr; }
    C4_ALWAYS_INLINE C4_PURE bool operator!= (std::nullptr_t) const noexcept { return str != nullptr; }

    C4_ALWAYS_INLINE C4_PURE bool operator== (C const c) const noexcept { return this->compare(c) == 0; }
    C4_ALWAYS_INLINE C4_PURE bool operator!= (C const c) const noexcept { return this->compare(c) != 0; }
    C4_ALWAYS_INLINE C4_PURE bool operator<  (C const c) const noexcept { return this->compare(c) <  0; }
    C4_ALWAYS_INLINE C4_PURE bool operator>  (C const c) const noexcept { return this->compare(c) >  0; }
    C4_ALWAYS_INLINE C4_PURE bool operator<= (C const c) const noexcept { return this->compare(c) <= 0; }
    C4_ALWAYS_INLINE C4_PURE bool operator>= (C const c) const noexcept { return this->compare(c) >= 0; }

    template<class U> C4_ALWAYS_INLINE C4_PURE bool operator== (basic_substring<U> const that) const noexcept { return this->compare(that) == 0; }
    template<class U> C4_ALWAYS_INLINE C4_PURE bool operator!= (basic_substring<U> const that) const noexcept { return this->compare(that) != 0; }
    template<class U> C4_ALWAYS_INLINE C4_PURE bool operator<  (basic_substring<U> const that) const noexcept { return this->compare(that) <  0; }
    template<class U> C4_ALWAYS_INLINE C4_PURE bool operator>  (basic_substring<U> const that) const noexcept { return this->compare(that) >  0; }
    template<class U> C4_ALWAYS_INLINE C4_PURE bool operator<= (basic_substring<U> const that) const noexcept { return this->compare(that) <= 0; }
    template<class U> C4_ALWAYS_INLINE C4_PURE bool operator>= (basic_substring<U> const that) const noexcept { return this->compare(that) >= 0; }

    template<size_t N> C4_ALWAYS_INLINE C4_PURE bool operator== (const char (&that)[N]) const noexcept { return this->compare(that, N-1) == 0; }
    template<size_t N> C4_ALWAYS_INLINE C4_PURE bool operator!= (const char (&that)[N]) const noexcept { return this->compare(that, N-1) != 0; }
    template<size_t N> C4_ALWAYS_INLINE C4_PURE bool operator<  (const char (&that)[N]) const noexcept { return this->compare(that, N-1) <  0; }
    template<size_t N> C4_ALWAYS_INLINE C4_PURE bool operator>  (const char (&that)[N]) const noexcept { return this->compare(that, N-1) >  0; }
    template<size_t N> C4_ALWAYS_INLINE C4_PURE bool operator<= (const char (&that)[N]) const noexcept { return this->compare(that, N-1) <= 0; }
    template<size_t N> C4_ALWAYS_INLINE C4_PURE bool operator>= (const char (&that)[N]) const noexcept { return this->compare(that, N-1) >= 0; }

    /** @} */

public:

    /** @name Sub-selection methods */
    /** @{ */

    /** true if *this is a substring of that (ie, from the same buffer) */
    C4_ALWAYS_INLINE C4_PURE bool is_sub(ro_substr const that) const noexcept
    {
        return that.is_super(*this);
    }

    /** true if that is a substring of *this (ie, from the same buffer) */
    C4_ALWAYS_INLINE C4_PURE bool is_super(ro_substr const that) const noexcept
    {
        if(C4_LIKELY(len > 0))
            return that.str >= str && that.str+that.len <= str+len;
        else
            return that.len == 0 && that.str == str && str != nullptr;
    }

    /** true if there is overlap of at least one element between that and *this */
    C4_ALWAYS_INLINE C4_PURE bool overlaps(ro_substr const that) const noexcept
    {
        // thanks @timwynants
        return that.str+that.len > str && that.str < str+len;
    }

public:

    /** return [first,len[ */
    C4_ALWAYS_INLINE C4_PURE basic_substring sub(size_t first) const noexcept
    {
        C4_ASSERT(first >= 0 && first <= len);
        return basic_substring(str + first, len - first);
    }

    /** return [first,first+num[. If num==npos, return [first,len[ */
    C4_ALWAYS_INLINE C4_PURE basic_substring sub(size_t first, size_t num) const noexcept
    {
        C4_ASSERT(first >= 0 && first <= len);
        C4_ASSERT((num >= 0 && num <= len) || (num == npos));
        size_t rnum = num != npos ? num : len - first;
        C4_ASSERT((first >= 0 && first + rnum <= len) || (num == 0));
        return basic_substring(str + first, rnum);
    }

    /** return [first,last[. If last==npos, return [first,len[ */
    C4_ALWAYS_INLINE C4_PURE basic_substring range(size_t first, size_t last=npos) const noexcept
    {
        C4_ASSERT(first >= 0 && first <= len);
        last = last != npos ? last : len;
        C4_ASSERT(first <= last);
        C4_ASSERT(last  >= 0 && last  <= len);
        return basic_substring(str + first, last - first);
    }

    /** return the first @p num elements: [0,num[*/
    C4_ALWAYS_INLINE C4_PURE basic_substring first(size_t num) const noexcept
    {
        C4_ASSERT(num <= len || num == npos);
        return basic_substring(str, num != npos ? num : len);
    }

    /** return the last @num elements: [len-num,len[*/
    C4_ALWAYS_INLINE C4_PURE basic_substring last(size_t num) const noexcept
    {
        C4_ASSERT(num <= len || num == npos);
        return num != npos ?
            basic_substring(str + len - num, num) :
            *this;
    }

    /** offset from the ends: return [left,len-right[ ; ie, trim a
        number of characters from the left and right. This is
        equivalent to python's negative list indices. */
    C4_ALWAYS_INLINE C4_PURE basic_substring offs(size_t left, size_t right) const noexcept
    {
        C4_ASSERT(left  >= 0 && left  <= len);
        C4_ASSERT(right >= 0 && right <= len);
        C4_ASSERT(left  <= len - right + 1);
        return basic_substring(str + left, len - right - left);
    }

    /** return [0, pos[ . Same as .first(pos), but provided for compatibility with .right_of() */
    C4_ALWAYS_INLINE C4_PURE basic_substring left_of(size_t pos) const noexcept
    {
        C4_ASSERT(pos <= len || pos == npos);
        return (pos != npos) ?
            basic_substring(str, pos) :
            *this;
    }

    /** return [0, pos+include_pos[ . Same as .first(pos+1), but provided for compatibility with .right_of() */
    C4_ALWAYS_INLINE C4_PURE basic_substring left_of(size_t pos, bool include_pos) const noexcept
    {
        C4_ASSERT(pos <= len || pos == npos);
        return (pos != npos) ?
            basic_substring(str, pos+include_pos) :
            *this;
    }

    /** return [pos+1, len[ */
    C4_ALWAYS_INLINE C4_PURE basic_substring right_of(size_t pos) const noexcept
    {
        C4_ASSERT(pos <= len || pos == npos);
        return (pos != npos) ?
            basic_substring(str + (pos + 1), len - (pos + 1)) :
            basic_substring(str + len, size_t(0));
    }

    /** return [pos+!include_pos, len[ */
    C4_ALWAYS_INLINE C4_PURE basic_substring right_of(size_t pos, bool include_pos) const noexcept
    {
        C4_ASSERT(pos <= len || pos == npos);
        return (pos != npos) ?
            basic_substring(str + (pos + !include_pos), len - (pos + !include_pos)) :
            basic_substring(str + len, size_t(0));
    }

public:

    /** given @p subs a substring of the current string, get the
     * portion of the current string to the left of it */
    C4_ALWAYS_INLINE C4_PURE basic_substring left_of(ro_substr const subs) const noexcept
    {
        C4_ASSERT(is_super(subs) || subs.empty());
        auto ssb = subs.begin();
        auto b = begin();
        auto e = end();
        if(ssb >= b && ssb <= e)
            return sub(0, static_cast<size_t>(ssb - b));
        else
            return sub(0, 0);
    }

    /** given @p subs a substring of the current string, get the
     * portion of the current string to the right of it */
    C4_ALWAYS_INLINE C4_PURE basic_substring right_of(ro_substr const subs) const noexcept
    {
        C4_ASSERT(is_super(subs) || subs.empty());
        auto sse = subs.end();
        auto b = begin();
        auto e = end();
        if(sse >= b && sse <= e)
            return sub(static_cast<size_t>(sse - b), static_cast<size_t>(e - sse));
        else
            return sub(0, 0);
    }

    /** @} */

public:

    /** @name Removing characters (trim()) / patterns (strip()) from the tips of the string */
    /** @{ */

    /** trim left */
    basic_substring triml(const C c) const
    {
        if( ! empty())
        {
            size_t pos = first_not_of(c);
            if(pos != npos)
                return sub(pos);
        }
        return sub(0, 0);
    }
    /** trim left ANY of the characters.
     * @see stripl() to remove a pattern from the left */
    basic_substring triml(ro_substr chars) const
    {
        if( ! empty())
        {
            size_t pos = first_not_of(chars);
            if(pos != npos)
                return sub(pos);
        }
        return sub(0, 0);
    }

    /** trim the character c from the right */
    basic_substring trimr(const C c) const
    {
        if( ! empty())
        {
            size_t pos = last_not_of(c, npos);
            if(pos != npos)
                return sub(0, pos+1);
        }
        return sub(0, 0);
    }
    /** trim right ANY of the characters
     * @see stripr() to remove a pattern from the right  */
    basic_substring trimr(ro_substr chars) const
    {
        if( ! empty())
        {
            size_t pos = last_not_of(chars, npos);
            if(pos != npos)
                return sub(0, pos+1);
        }
        return sub(0, 0);
    }

    /** trim the character c left and right */
    basic_substring trim(const C c) const
    {
        return triml(c).trimr(c);
    }
    /** trim left and right ANY of the characters
     * @see strip() to remove a pattern from the left and right */
    basic_substring trim(ro_substr const chars) const
    {
        return triml(chars).trimr(chars);
    }

    /** remove a pattern from the left
     * @see triml() to remove characters*/
    basic_substring stripl(ro_substr pattern) const
    {
        if( ! begins_with(pattern))
            return *this;
        return sub(pattern.len < len ? pattern.len : len);
    }

    /** remove a pattern from the right
     * @see trimr() to remove characters*/
    basic_substring stripr(ro_substr pattern) const
    {
        if( ! ends_with(pattern))
            return *this;
        return left_of(len - (pattern.len < len ? pattern.len : len));
    }

    /** @} */

public:

    /** @name Lookup methods */
    /** @{ */

    inline size_t find(const C c, size_t start_pos=0) const
    {
        return first_of(c, start_pos);
    }
    inline size_t find(ro_substr pattern, size_t start_pos=0) const
    {
        C4_ASSERT(start_pos == npos || (start_pos >= 0 && start_pos <= len));
        if(len < pattern.len) return npos;
        for(size_t i = start_pos, e = len - pattern.len + 1; i < e; ++i)
        {
            bool gotit = true;
            for(size_t j = 0; j < pattern.len; ++j)
            {
                C4_ASSERT(i + j < len);
                if(str[i + j] != pattern.str[j])
                {
                    gotit = false;
                    break;
                }
            }
            if(gotit)
            {
                return i;
            }
        }
        return npos;
    }

public:

    /** count the number of occurrences of c */
    inline size_t count(const C c, size_t pos=0) const
    {
        C4_ASSERT(pos >= 0 && pos <= len);
        size_t num = 0;
        pos = find(c, pos);
        while(pos != npos)
        {
            ++num;
            pos = find(c, pos + 1);
        }
        return num;
    }

    /** count the number of occurrences of s */
    inline size_t count(ro_substr c, size_t pos=0) const
    {
        C4_ASSERT(pos >= 0 && pos <= len);
        size_t num = 0;
        pos = find(c, pos);
        while(pos != npos)
        {
            ++num;
            pos = find(c, pos + c.len);
        }
        return num;
    }

    /** get the substr consisting of the first occurrence of @p c after @p pos, or an empty substr if none occurs */
    inline basic_substring select(const C c, size_t pos=0) const
    {
        pos = find(c, pos);
        return pos != npos ? sub(pos, 1) : basic_substring();
    }

    /** get the substr consisting of the first occurrence of @p pattern after @p pos, or an empty substr if none occurs */
    inline basic_substring select(ro_substr pattern, size_t pos=0) const
    {
        pos = find(pattern, pos);
        return pos != npos ? sub(pos, pattern.len) : basic_substring();
    }

public:

    struct first_of_any_result
    {
        size_t which;
        size_t pos;
        inline operator bool() const { return which != NONE && pos != npos; }
    };

    first_of_any_result first_of_any(ro_substr s0, ro_substr s1) const
    {
        ro_substr s[2] = {s0, s1};
        return first_of_any_iter(&s[0], &s[0] + 2);
    }

    first_of_any_result first_of_any(ro_substr s0, ro_substr s1, ro_substr s2) const
    {
        ro_substr s[3] = {s0, s1, s2};
        return first_of_any_iter(&s[0], &s[0] + 3);
    }

    first_of_any_result first_of_any(ro_substr s0, ro_substr s1, ro_substr s2, ro_substr s3) const
    {
        ro_substr s[4] = {s0, s1, s2, s3};
        return first_of_any_iter(&s[0], &s[0] + 4);
    }

    first_of_any_result first_of_any(ro_substr s0, ro_substr s1, ro_substr s2, ro_substr s3, ro_substr s4) const
    {
        ro_substr s[5] = {s0, s1, s2, s3, s4};
        return first_of_any_iter(&s[0], &s[0] + 5);
    }

    template<class It>
    first_of_any_result first_of_any_iter(It first_span, It last_span) const
    {
        for(size_t i = 0; i < len; ++i)
        {
            size_t curr = 0;
            for(It it = first_span; it != last_span; ++curr, ++it)
            {
                auto const& chars = *it;
                if((i + chars.len) > len) continue;
                bool gotit = true;
                for(size_t j = 0; j < chars.len; ++j)
                {
                    C4_ASSERT(i + j < len);
                    if(str[i + j] != chars[j])
                    {
                        gotit = false;
                        break;
                    }
                }
                if(gotit)
                {
                    return {curr, i};
                }
            }
        }
        return {NONE, npos};
    }

public:

    /** true if the first character of the string is @p c */
    bool begins_with(const C c) const
    {
        return len > 0 ? str[0] == c : false;
    }

    /** true if the first @p num characters of the string are @p c */
    bool begins_with(const C c, size_t num) const
    {
        if(len < num)
        {
            return false;
        }
        for(size_t i = 0; i < num; ++i)
        {
            if(str[i] != c)
            {
                return false;
            }
        }
        return true;
    }

    /** true if the string begins with the given @p pattern */
    bool begins_with(ro_substr pattern) const
    {
        if(len < pattern.len)
        {
            return false;
        }
        for(size_t i = 0; i < pattern.len; ++i)
        {
            if(str[i] != pattern[i])
            {
                return false;
            }
        }
        return true;
    }

    /** true if the first character of the string is any of the given @p chars */
    bool begins_with_any(ro_substr chars) const
    {
        if(len == 0)
        {
            return false;
        }
        for(size_t i = 0; i < chars.len; ++i)
        {
            if(str[0] == chars.str[i])
            {
                return true;
            }
        }
        return false;
    }

    /** true if the last character of the string is @p c */
    bool ends_with(const C c) const
    {
        return len > 0 ? str[len-1] == c : false;
    }

    /** true if the last @p num characters of the string are @p c */
    bool ends_with(const C c, size_t num) const
    {
        if(len < num)
        {
            return false;
        }
        for(size_t i = len - num; i < len; ++i)
        {
            if(str[i] != c)
            {
                return false;
            }
        }
        return true;
    }

    /** true if the string ends with the given @p pattern */
    bool ends_with(ro_substr pattern) const
    {
        if(len < pattern.len)
        {
            return false;
        }
        for(size_t i = 0, s = len-pattern.len; i < pattern.len; ++i)
        {
            if(str[s+i] != pattern[i])
            {
                return false;
            }
        }
        return true;
    }

    /** true if the last character of the string is any of the given @p chars */
    bool ends_with_any(ro_substr chars) const
    {
        if(len == 0)
        {
            return false;
        }
        for(size_t i = 0; i < chars.len; ++i)
        {
            if(str[len - 1] == chars[i])
            {
                return true;
            }
        }
        return false;
    }

public:

    /** @return the first position where c is found in the string, or npos if none is found */
    size_t first_of(const C c, size_t start=0) const
    {
        C4_ASSERT(start == npos || (start >= 0 && start <= len));
        for(size_t i = start; i < len; ++i)
        {
            if(str[i] == c)
                return i;
        }
        return npos;
    }

    /** @return the last position where c is found in the string, or npos if none is found */
    size_t last_of(const C c, size_t start=npos) const
    {
        C4_ASSERT(start == npos || (start >= 0 && start <= len));
        if(start == npos)
            start = len;
        for(size_t i = start-1; i != size_t(-1); --i)
        {
            if(str[i] == c)
                return i;
        }
        return npos;
    }

    /** @return the first position where ANY of the chars is found in the string, or npos if none is found */
    size_t first_of(ro_substr chars, size_t start=0) const
    {
        C4_ASSERT(start == npos || (start >= 0 && start <= len));
        for(size_t i = start; i < len; ++i)
        {
            for(size_t j = 0; j < chars.len; ++j)
            {
                if(str[i] == chars[j])
                    return i;
            }
        }
        return npos;
    }

    /** @return the last position where ANY of the chars is found in the string, or npos if none is found */
    size_t last_of(ro_substr chars, size_t start=npos) const
    {
        C4_ASSERT(start == npos || (start >= 0 && start <= len));
        if(start == npos)
            start = len;
        for(size_t i = start-1; i != size_t(-1); --i)
        {
            for(size_t j = 0; j < chars.len; ++j)
            {
                if(str[i] == chars[j])
                    return i;
            }
        }
        return npos;
    }

public:

    size_t first_not_of(const C c) const
    {
        for(size_t i = 0; i < len; ++i)
        {
            if(str[i] != c)
                return i;
        }
        return npos;
    }

    size_t first_not_of(const C c, size_t start) const
    {
        C4_ASSERT((start >= 0 && start <= len) || (start == len && len == 0));
        for(size_t i = start; i < len; ++i)
        {
            if(str[i] != c)
                return i;
        }
        return npos;
    }

    size_t last_not_of(const C c) const
    {
        for(size_t i = len-1; i != size_t(-1); --i)
        {
            if(str[i] != c)
                return i;
        }
        return npos;
    }

    size_t last_not_of(const C c, size_t start) const
    {
        C4_ASSERT(start == npos || (start >= 0 && start <= len));
        if(start == npos)
            start = len;
        for(size_t i = start-1; i != size_t(-1); --i)
        {
            if(str[i] != c)
                return i;
        }
        return npos;
    }

    size_t first_not_of(ro_substr chars) const
    {
        for(size_t i = 0; i < len; ++i)
        {
            bool gotit = true;
            for(size_t j = 0; j < chars.len; ++j)
            {
                if(str[i] == chars.str[j])
                {
                    gotit = false;
                    break;
                }
            }
            if(gotit)
            {
                return i;
            }
        }
        return npos;
    }

    size_t first_not_of(ro_substr chars, size_t start) const
    {
        C4_ASSERT((start >= 0 && start <= len) || (start == len && len == 0));
        for(size_t i = start; i < len; ++i)
        {
            bool gotit = true;
            for(size_t j = 0; j < chars.len; ++j)
            {
                if(str[i] == chars.str[j])
                {
                    gotit = false;
                    break;
                }
            }
            if(gotit)
            {
                return i;
            }
        }
        return npos;
    }

    size_t last_not_of(ro_substr chars) const
    {
        for(size_t i = len-1; i != size_t(-1); --i)
        {
            bool gotit = true;
            for(size_t j = 0; j < chars.len; ++j)
            {
                if(str[i] == chars.str[j])
                {
                    gotit = false;
                    break;
                }
            }
            if(gotit)
            {
                return i;
            }
        }
        return npos;
    }

    size_t last_not_of(ro_substr chars, size_t start) const
    {
        C4_ASSERT(start == npos || (start >= 0 && start <= len));
        if(start == npos)
            start = len;
        for(size_t i = start-1; i != size_t(-1); --i)
        {
            bool gotit = true;
            for(size_t j = 0; j < chars.len; ++j)
            {
                if(str[i] == chars.str[j])
                {
                    gotit = false;
                    break;
                }
            }
            if(gotit)
            {
                return i;
            }
        }
        return npos;
    }

    /** @} */

public:

    /** @name Range lookup methods */
    /** @{ */

    /** get the range delimited by an open-close pair of characters.
     * @note There must be no nested pairs.
     * @note No checks for escapes are performed. */
    basic_substring pair_range(CC open, CC close) const
    {
        size_t b = find(open);
        if(b == npos)
            return basic_substring();
        size_t e = find(close, b+1);
        if(e == npos)
            return basic_substring();
        basic_substring ret = range(b, e+1);
        C4_ASSERT(ret.sub(1).find(open) == npos);
        return ret;
    }

    /** get the range delimited by a single open-close character (eg, quotes).
     * @note The open-close character can be escaped. */
    basic_substring pair_range_esc(CC open_close, CC escape=CC('\\'))
    {
        size_t b = find(open_close);
        if(b == npos) return basic_substring();
        for(size_t i = b+1; i < len; ++i)
        {
            CC c = str[i];
            if(c == open_close)
            {
                if(str[i-1] != escape)
                {
                    return range(b, i+1);
                }
            }
        }
        return basic_substring();
    }

    /** get the range delimited by an open-close pair of characters,
     * with possibly nested occurrences. No checks for escapes are
     * performed. */
    basic_substring pair_range_nested(CC open, CC close) const
    {
        size_t b = find(open);
        if(b == npos) return basic_substring();
        size_t e, curr = b+1, count = 0;
        const char both[] = {open, close, '\0'};
        while((e = first_of(both, curr)) != npos)
        {
            if(str[e] == open)
            {
                ++count;
                curr = e+1;
            }
            else if(str[e] == close)
            {
                if(count == 0) return range(b, e+1);
                --count;
                curr = e+1;
            }
        }
        return basic_substring();
    }

    basic_substring unquoted() const
    {
        constexpr const C dq('"'), sq('\'');
        if(len >= 2 && (str[len - 2] != C('\\')) &&
           ((begins_with(sq) && ends_with(sq))
            ||
            (begins_with(dq) && ends_with(dq))))
        {
            return range(1, len -1);
        }
        return *this;
    }

    /** @} */

public:

    /** @name Number-matching query methods */
    /** @{ */

    /** @return true if the substring contents are a floating-point or integer number.
     * @note any leading or trailing whitespace will return false. */
    bool is_number() const
    {
        if(empty() || (first_non_empty_span().empty()))
            return false;
        if(first_uint_span() == *this)
            return true;
        if(first_int_span() == *this)
            return true;
        if(first_real_span() == *this)
            return true;
        return false;
    }

    /** @return true if the substring contents are a real number.
     * @note any leading or trailing whitespace will return false. */
    bool is_real() const
    {
        if(empty() || (first_non_empty_span().empty()))
            return false;
        if(first_real_span() == *this)
            return true;
        return false;
    }

    /** @return true if the substring contents are an integer number.
     * @note any leading or trailing whitespace will return false. */
    bool is_integer() const
    {
        if(empty() || (first_non_empty_span().empty()))
            return false;
        if(first_uint_span() == *this)
            return true;
        if(first_int_span() == *this)
            return true;
        return false;
    }

    /** @return true if the substring contents are an unsigned integer number.
     * @note any leading or trailing whitespace will return false. */
    bool is_unsigned_integer() const
    {
        if(empty() || (first_non_empty_span().empty()))
            return false;
        if(first_uint_span() == *this)
            return true;
        return false;
    }

    /** get the first span consisting exclusively of non-empty characters */
    basic_substring first_non_empty_span() const
    {
        constexpr const ro_substr empty_chars(" \n\r\t");
        size_t pos = first_not_of(empty_chars);
        if(pos == npos)
            return first(0);
        auto ret = sub(pos);
        pos = ret.first_of(empty_chars);
        return ret.first(pos);
    }

    /** get the first span which can be interpreted as an unsigned integer */
    basic_substring first_uint_span() const
    {
        basic_substring ne = first_non_empty_span();
        if(ne.empty())
            return ne;
        if(ne.str[0] == '-')
            return first(0);
        size_t skip_start = size_t(ne.str[0] == '+');
        return ne._first_integral_span(skip_start);
    }

    /** get the first span which can be interpreted as a signed integer */
    basic_substring first_int_span() const
    {
        basic_substring ne = first_non_empty_span();
        if(ne.empty())
            return ne;
        size_t skip_start = size_t(ne.str[0] == '+' || ne.str[0] == '-');
        return ne._first_integral_span(skip_start);
    }

    basic_substring _first_integral_span(size_t skip_start) const
    {
        C4_ASSERT(!empty());
        if(skip_start == len)
            return first(0);
        C4_ASSERT(skip_start < len);
        if(len >= skip_start + 3)
        {
            if(str[skip_start] != '0')
            {
                for(size_t i = skip_start; i < len; ++i)
                {
                    char c = str[i];
                    if(c < '0' || c > '9')
                        return i > skip_start && _is_delim_char(c) ? first(i) : first(0);
                }
            }
            else
            {
                char next = str[skip_start + 1];
                if(next == 'x' || next == 'X')
                {
                    skip_start += 2;
                    for(size_t i = skip_start; i < len; ++i)
                    {
                        const char c = str[i];
                        if( ! _is_hex_char(c))
                            return i > skip_start && _is_delim_char(c) ? first(i) : first(0);
                    }
                    return *this;
                }
                else if(next == 'b' || next == 'B')
                {
                    skip_start += 2;
                    for(size_t i = skip_start; i < len; ++i)
                    {
                        const char c = str[i];
                        if(c != '0' && c != '1')
                            return i > skip_start && _is_delim_char(c) ? first(i) : first(0);
                    }
                    return *this;
                }
                else if(next == 'o' || next == 'O')
                {
                    skip_start += 2;
                    for(size_t i = skip_start; i < len; ++i)
                    {
                        const char c = str[i];
                        if(c < '0' || c > '7')
                            return i > skip_start && _is_delim_char(c) ? first(i) : first(0);
                    }
                    return *this;
                }
            }
        }
        // must be a decimal, or it is not a an number
        for(size_t i = skip_start; i < len; ++i)
        {
            const char c = str[i];
            if(c < '0' || c > '9')
                return i > skip_start && _is_delim_char(c) ? first(i) : first(0);
        }
        return *this;
    }

    /** get the first span which can be interpreted as a real (floating-point) number */
    basic_substring first_real_span() const
    {
        basic_substring ne = first_non_empty_span();
        if(ne.empty())
            return ne;
        const size_t skip_start = (ne.str[0] == '+' || ne.str[0] == '-');
        C4_ASSERT(skip_start == 0 || skip_start == 1);
        // if we have at least three digits after the leading sign, it
        // can be decimal, or hex, or bin or oct. Ex:
        // non-decimal: 0x0, 0b0, 0o0
        // decimal: 1.0, 10., 1e1, 100, inf, nan, infinity
        if(ne.len >= skip_start+3)
        {
            // if it does not have leading 0, it must be decimal, or it is not a real
            if(ne.str[skip_start] != '0')
            {
                if(ne.str[skip_start] == 'i') // is it infinity or inf?
                {
                    basic_substring word = ne._word_follows(skip_start + 1, "nfinity");
                    if(word.len)
                        return word;
                    return ne._word_follows(skip_start + 1, "nf");
                }
                else if(ne.str[skip_start] == 'n') // is it nan?
                {
                    return ne._word_follows(skip_start + 1, "an");
                }
                else // must be a decimal, or it is not a real
                {
                    return ne._first_real_span_dec(skip_start);
                }
            }
            else // starts with 0. is it 0x, 0b or 0o?
            {
                const char next = ne.str[skip_start + 1];
                // hexadecimal
                if(next == 'x' || next == 'X')
                    return ne._first_real_span_hex(skip_start + 2);
                // binary
                else if(next == 'b' || next == 'B')
                    return ne._first_real_span_bin(skip_start + 2);
                // octal
                else if(next == 'o' || next == 'O')
                    return ne._first_real_span_oct(skip_start + 2);
                // none of the above. may still be a decimal.
                else
                    return ne._first_real_span_dec(skip_start); // do not skip the 0.
            }
        }
        // less than 3 chars after the leading sign. It is either a
        // decimal or it is not a real. (cannot be any of 0x0, etc).
        return ne._first_real_span_dec(skip_start);
    }

    /** true if the character is a delimiter character *at the end* */
    static constexpr C4_ALWAYS_INLINE C4_CONST bool _is_delim_char(char c) noexcept
    {
        return c == ' ' || c == '\n'
            || c == ']' || c == ')'  || c == '}'
            || c == ',' || c == ';' || c == '\r' || c == '\t' || c == '\0';
    }

    /** true if the character is in [0-9a-fA-F] */
    static constexpr C4_ALWAYS_INLINE C4_CONST bool _is_hex_char(char c) noexcept
    {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
    }

    C4_NO_INLINE C4_PURE basic_substring _word_follows(size_t pos, csubstr word) const noexcept
    {
        size_t posend = pos + word.len;
        if(len >= posend && sub(pos, word.len) == word)
            if(len == posend || _is_delim_char(str[posend]))
                return first(posend);
        return first(0);
    }

    // this function is declared inside the class to avoid a VS error with __declspec(dllimport)
    C4_NO_INLINE C4_PURE basic_substring _first_real_span_dec(size_t pos) const noexcept
    {
        bool intchars = false;
        bool fracchars = false;
        bool powchars;
        // integral part
        for( ; pos < len; ++pos)
        {
            const char c = str[pos];
            if(c >= '0' && c <= '9')
            {
                intchars = true;
            }
            else if(c == '.')
            {
                ++pos;
                goto fractional_part_dec;
            }
            else if(c == 'e' || c == 'E')
            {
                ++pos;
                goto power_part_dec;
            }
            else if(_is_delim_char(c))
            {
                return intchars ? first(pos) : first(0);
            }
            else
            {
                return first(0);
            }
        }
        // no . or p were found; this is either an integral number
        // or not a number at all
        return intchars ?
            *this :
            first(0);
    fractional_part_dec:
        C4_ASSERT(pos > 0);
        C4_ASSERT(str[pos - 1] == '.');
        for( ; pos < len; ++pos)
        {
            const char c = str[pos];
            if(c >= '0' && c <= '9')
            {
                fracchars = true;
            }
            else if(c == 'e' || c == 'E')
            {
                ++pos;
                goto power_part_dec;
            }
            else if(_is_delim_char(c))
            {
                return intchars || fracchars ? first(pos) : first(0);
            }
            else
            {
                return first(0);
            }
        }
        return intchars || fracchars ?
            *this :
            first(0);
    power_part_dec:
        C4_ASSERT(pos > 0);
        C4_ASSERT(str[pos - 1] == 'e' || str[pos - 1] == 'E');
        // either digits, or +, or - are expected here, followed by more digits.
        if((len == pos) || ((!intchars) && (!fracchars)))
            return first(0);
        if(str[pos] == '-' || str[pos] == '+')
            ++pos; // skip the sign
        powchars = false;
        for( ; pos < len; ++pos)
        {
            const char c = str[pos];
            if(c >= '0' && c <= '9')
                powchars = true;
            else if(powchars && _is_delim_char(c))
                return first(pos);
            else
                return first(0);
        }
        return powchars ? *this : first(0);
    }

    // this function is declared inside the class to avoid a VS error with __declspec(dllimport)
    C4_NO_INLINE C4_PURE basic_substring _first_real_span_hex(size_t pos) const noexcept
    {
        bool intchars = false;
        bool fracchars = false;
        bool powchars;
        // integral part
        for( ; pos < len; ++pos)
        {
            const char c = str[pos];
            if(_is_hex_char(c))
            {
                intchars = true;
            }
            else if(c == '.')
            {
                ++pos;
                goto fractional_part_hex;
            }
            else if(c == 'p' || c == 'P')
            {
                ++pos;
                goto power_part_hex;
            }
            else if(_is_delim_char(c))
            {
                return intchars ? first(pos) : first(0);
            }
            else
            {
                return first(0);
            }
        }
        // no . or p were found; this is either an integral number
        // or not a number at all
        return intchars ?
            *this :
            first(0);
    fractional_part_hex:
        C4_ASSERT(pos > 0);
        C4_ASSERT(str[pos - 1] == '.');
        for( ; pos < len; ++pos)
        {
            const char c = str[pos];
            if(_is_hex_char(c))
            {
                fracchars = true;
            }
            else if(c == 'p' || c == 'P')
            {
                ++pos;
                goto power_part_hex;
            }
            else if(_is_delim_char(c))
            {
                return intchars || fracchars ? first(pos) : first(0);
            }
            else
            {
                return first(0);
            }
        }
        return intchars || fracchars ?
            *this :
            first(0);
    power_part_hex:
        C4_ASSERT(pos > 0);
        C4_ASSERT(str[pos - 1] == 'p' || str[pos - 1] == 'P');
        // either a + or a - is expected here, followed by more chars.
        // also, using (pos+1) in this check will cause an early
        // return when no more chars follow the sign.
        if(len <= (pos+1) || (str[pos] != '+' && str[pos] != '-') || ((!intchars) && (!fracchars)))
            return first(0);
        ++pos; // this was the sign.
        // ... so the (pos+1) ensures that we enter the loop and
        // hence that there exist chars in the power part
        powchars = false;
        for( ; pos < len; ++pos)
        {
            const char c = str[pos];
            if(c >= '0' && c <= '9')
                powchars = true;
            else if(powchars && _is_delim_char(c))
                return first(pos);
            else
                return first(0);
        }
        return *this;
    }

    // this function is declared inside the class to avoid a VS error with __declspec(dllimport)
    C4_NO_INLINE C4_PURE basic_substring _first_real_span_bin(size_t pos) const noexcept
    {
        bool intchars = false;
        bool fracchars = false;
        bool powchars;
        // integral part
        for( ; pos < len; ++pos)
        {
            const char c = str[pos];
            if(c == '0' || c == '1')
            {
                intchars = true;
            }
            else if(c == '.')
            {
                ++pos;
                goto fractional_part_bin;
            }
            else if(c == 'p' || c == 'P')
            {
                ++pos;
                goto power_part_bin;
            }
            else if(_is_delim_char(c))
            {
                return intchars ? first(pos) : first(0);
            }
            else
            {
                return first(0);
            }
        }
        // no . or p were found; this is either an integral number
        // or not a number at all
        return intchars ?
            *this :
            first(0);
    fractional_part_bin:
        C4_ASSERT(pos > 0);
        C4_ASSERT(str[pos - 1] == '.');
        for( ; pos < len; ++pos)
        {
            const char c = str[pos];
            if(c == '0' || c == '1')
            {
                fracchars = true;
            }
            else if(c == 'p' || c == 'P')
            {
                ++pos;
                goto power_part_bin;
            }
            else if(_is_delim_char(c))
            {
                return intchars || fracchars ? first(pos) : first(0);
            }
            else
            {
                return first(0);
            }
        }
        return intchars || fracchars ?
            *this :
            first(0);
    power_part_bin:
        C4_ASSERT(pos > 0);
        C4_ASSERT(str[pos - 1] == 'p' || str[pos - 1] == 'P');
        // either a + or a - is expected here, followed by more chars.
        // also, using (pos+1) in this check will cause an early
        // return when no more chars follow the sign.
        if(len <= (pos+1) || (str[pos] != '+' && str[pos] != '-') || ((!intchars) && (!fracchars)))
            return first(0);
        ++pos; // this was the sign.
        // ... so the (pos+1) ensures that we enter the loop and
        // hence that there exist chars in the power part
        powchars = false;
        for( ; pos < len; ++pos)
        {
            const char c = str[pos];
            if(c >= '0' && c <= '9')
                powchars = true;
            else if(powchars && _is_delim_char(c))
                return first(pos);
            else
                return first(0);
        }
        return *this;
    }

    // this function is declared inside the class to avoid a VS error with __declspec(dllimport)
    C4_NO_INLINE C4_PURE basic_substring _first_real_span_oct(size_t pos) const noexcept
    {
        bool intchars = false;
        bool fracchars = false;
        bool powchars;
        // integral part
        for( ; pos < len; ++pos)
        {
            const char c = str[pos];
            if(c >= '0' && c <= '7')
            {
                intchars = true;
            }
            else if(c == '.')
            {
                ++pos;
                goto fractional_part_oct;
            }
            else if(c == 'p' || c == 'P')
            {
                ++pos;
                goto power_part_oct;
            }
            else if(_is_delim_char(c))
            {
                return intchars ? first(pos) : first(0);
            }
            else
            {
                return first(0);
            }
        }
        // no . or p were found; this is either an integral number
        // or not a number at all
        return intchars ?
            *this :
            first(0);
    fractional_part_oct:
        C4_ASSERT(pos > 0);
        C4_ASSERT(str[pos - 1] == '.');
        for( ; pos < len; ++pos)
        {
            const char c = str[pos];
            if(c >= '0' && c <= '7')
            {
                fracchars = true;
            }
            else if(c == 'p' || c == 'P')
            {
                ++pos;
                goto power_part_oct;
            }
            else if(_is_delim_char(c))
            {
                return intchars || fracchars ? first(pos) : first(0);
            }
            else
            {
                return first(0);
            }
        }
        return intchars || fracchars ?
            *this :
            first(0);
    power_part_oct:
        C4_ASSERT(pos > 0);
        C4_ASSERT(str[pos - 1] == 'p' || str[pos - 1] == 'P');
        // either a + or a - is expected here, followed by more chars.
        // also, using (pos+1) in this check will cause an early
        // return when no more chars follow the sign.
        if(len <= (pos+1) || (str[pos] != '+' && str[pos] != '-') || ((!intchars) && (!fracchars)))
            return first(0);
        ++pos; // this was the sign.
        // ... so the (pos+1) ensures that we enter the loop and
        // hence that there exist chars in the power part
        powchars = false;
        for( ; pos < len; ++pos)
        {
            const char c = str[pos];
            if(c >= '0' && c <= '9')
                powchars = true;
            else if(powchars && _is_delim_char(c))
                return first(pos);
            else
                return first(0);
        }
        return *this;
    }

    /** @} */

public:

    /** @name Splitting methods */
    /** @{ */

    /** returns true if the string has not been exhausted yet, meaning
     * it's ok to call next_split() again. When no instance of sep
     * exists in the string, returns the full string. When the input
     * is an empty string, the output string is the empty string. */
    bool next_split(C sep, size_t *C4_RESTRICT start_pos, basic_substring *C4_RESTRICT out) const
    {
        if(C4_LIKELY(*start_pos < len))
        {
            for(size_t i = *start_pos; i < len; i++)
            {
                if(str[i] == sep)
                {
                    out->assign(str + *start_pos, i - *start_pos);
                    *start_pos = i+1;
                    return true;
                }
            }
            out->assign(str + *start_pos, len - *start_pos);
            *start_pos = len + 1;
            return true;
        }
        else
        {
            bool valid = len > 0 && (*start_pos == len);
            if(valid && str && str[len-1] == sep)
            {
                out->assign(str + len, size_t(0)); // the cast is needed to prevent overload ambiguity
            }
            else
            {
                out->assign(str + len + 1, size_t(0)); // the cast is needed to prevent overload ambiguity
            }
            *start_pos = len + 1;
            return valid;
        }
    }

private:

    struct split_proxy_impl
    {
        struct split_iterator_impl
        {
            split_proxy_impl const* m_proxy;
            basic_substring m_str;
            size_t m_pos;
            NCC_ m_sep;

            split_iterator_impl(split_proxy_impl const* proxy, size_t pos, C sep)
                : m_proxy(proxy), m_pos(pos), m_sep(sep)
            {
                _tick();
            }

            void _tick()
            {
                m_proxy->m_str.next_split(m_sep, &m_pos, &m_str);
            }

            split_iterator_impl& operator++ () { _tick(); return *this; }
            split_iterator_impl  operator++ (int) { split_iterator_impl it = *this; _tick(); return it; }

            basic_substring& operator*  () { return  m_str; }
            basic_substring* operator-> () { return &m_str; }

            bool operator!= (split_iterator_impl const& that) const
            {
                return !(this->operator==(that));
            }
            bool operator== (split_iterator_impl const& that) const
            {
                C4_XASSERT((m_sep == that.m_sep) && "cannot compare split iterators with different separators");
                if(m_str.size() != that.m_str.size())
                    return false;
                if(m_str.data() != that.m_str.data())
                    return false;
                return m_pos == that.m_pos;
            }
        };

        basic_substring m_str;
        size_t m_start_pos;
        C m_sep;

        split_proxy_impl(basic_substring str_, size_t start_pos, C sep)
            : m_str(str_), m_start_pos(start_pos), m_sep(sep)
        {
        }

        split_iterator_impl begin() const
        {
            auto it = split_iterator_impl(this, m_start_pos, m_sep);
            return it;
        }
        split_iterator_impl end() const
        {
            size_t pos = m_str.size() + 1;
            auto it = split_iterator_impl(this, pos, m_sep);
            return it;
        }
    };

public:

    using split_proxy = split_proxy_impl;

    /** a view into the splits */
    split_proxy split(C sep, size_t start_pos=0) const
    {
        C4_XASSERT((start_pos >= 0 && start_pos < len) || empty());
        auto ss = sub(0, len);
        auto it = split_proxy(ss, start_pos, sep);
        return it;
    }

public:

    /** pop right: return the first split from the right. Use
     * gpop_left() to get the reciprocal part.
     */
    basic_substring pop_right(C sep=C('/'), bool skip_empty=false) const
    {
        if(C4_LIKELY(len > 1))
        {
            auto pos = last_of(sep);
            if(pos != npos)
            {
                if(pos + 1 < len) // does not end with sep
                {
                    return sub(pos + 1); // return from sep to end
                }
                else // the string ends with sep
                {
                    if( ! skip_empty)
                    {
                        return sub(pos + 1, 0);
                    }
                    auto ppos = last_not_of(sep); // skip repeated seps
                    if(ppos == npos) // the string is all made of seps
                    {
                        return sub(0, 0);
                    }
                    // find the previous sep
                    auto pos0 = last_of(sep, ppos);
                    if(pos0 == npos) // only the last sep exists
                    {
                        return sub(0); // return the full string (because skip_empty is true)
                    }
                    ++pos0;
                    return sub(pos0);
                }
            }
            else // no sep was found, return the full string
            {
                return *this;
            }
        }
        else if(len == 1)
        {
            if(begins_with(sep))
            {
                return sub(0, 0);
            }
            return *this;
        }
        else // an empty string
        {
            return basic_substring();
        }
    }

    /** return the first split from the left. Use gpop_right() to get
     * the reciprocal part. */
    basic_substring pop_left(C sep = C('/'), bool skip_empty=false) const
    {
        if(C4_LIKELY(len > 1))
        {
            auto pos = first_of(sep);
            if(pos != npos)
            {
                if(pos > 0)  // does not start with sep
                {
                    return sub(0, pos); //  return everything up to it
                }
                else  // the string starts with sep
                {
                    if( ! skip_empty)
                    {
                        return sub(0, 0);
                    }
                    auto ppos = first_not_of(sep); // skip repeated seps
                    if(ppos == npos) // the string is all made of seps
                    {
                        return sub(0, 0);
                    }
                    // find the next sep
                    auto pos0 = first_of(sep, ppos);
                    if(pos0 == npos) // only the first sep exists
                    {
                        return sub(0); // return the full string (because skip_empty is true)
                    }
                    C4_XASSERT(pos0 > 0);
                    // return everything up to the second sep
                    return sub(0, pos0);
                }
            }
            else // no sep was found, return the full string
            {
                return sub(0);
            }
        }
        else if(len == 1)
        {
            if(begins_with(sep))
            {
                return sub(0, 0);
            }
            return sub(0);
        }
        else // an empty string
        {
            return basic_substring();
        }
    }

public:

    /** greedy pop left. eg, csubstr("a/b/c").gpop_left('/')="c" */
    basic_substring gpop_left(C sep = C('/'), bool skip_empty=false) const
    {
        auto ss = pop_right(sep, skip_empty);
        ss = left_of(ss);
        if(ss.find(sep) != npos)
        {
            if(ss.ends_with(sep))
            {
                if(skip_empty)
                {
                    ss = ss.trimr(sep);
                }
                else
                {
                    ss = ss.sub(0, ss.len-1); // safe to subtract because ends_with(sep) is true
                }
            }
        }
        return ss;
    }

    /** greedy pop right. eg, csubstr("a/b/c").gpop_right('/')="a" */
    basic_substring gpop_right(C sep = C('/'), bool skip_empty=false) const
    {
        auto ss = pop_left(sep, skip_empty);
        ss = right_of(ss);
        if(ss.find(sep) != npos)
        {
            if(ss.begins_with(sep))
            {
                if(skip_empty)
                {
                    ss = ss.triml(sep);
                }
                else
                {
                    ss = ss.sub(1);
                }
            }
        }
        return ss;
    }

    /** @} */

public:

    /** @name Path-like manipulation methods */
    /** @{ */

    basic_substring basename(C sep=C('/')) const
    {
        auto ss = pop_right(sep, /*skip_empty*/true);
        ss = ss.trimr(sep);
        return ss;
    }

    basic_substring dirname(C sep=C('/')) const
    {
        auto ss = basename(sep);
        ss = ss.empty() ? *this : left_of(ss);
        return ss;
    }

    C4_ALWAYS_INLINE basic_substring name_wo_extshort() const
    {
        return gpop_left('.');
    }

    C4_ALWAYS_INLINE basic_substring name_wo_extlong() const
    {
        return pop_left('.');
    }

    C4_ALWAYS_INLINE basic_substring extshort() const
    {
        return pop_right('.');
    }

    C4_ALWAYS_INLINE basic_substring extlong() const
    {
        return gpop_right('.');
    }

    /** @} */

public:

    /** @name Content-modification methods (only for non-const C) */
    /** @{ */

    /** convert the string to upper-case
     * @note this method requires that the string memory is writeable and is SFINAEd out for const C */
    C4_REQUIRE_RW(void) toupper()
    {
        for(size_t i = 0; i < len; ++i)
        {
            str[i] = static_cast<C>(::toupper(str[i]));
        }
    }

    /** convert the string to lower-case
     * @note this method requires that the string memory is writeable and is SFINAEd out for const C */
    C4_REQUIRE_RW(void) tolower()
    {
        for(size_t i = 0; i < len; ++i)
        {
            str[i] = static_cast<C>(::tolower(str[i]));
        }
    }

public:

    /** fill the entire contents with the given @p val
     * @note this method requires that the string memory is writeable and is SFINAEd out for const C */
    C4_REQUIRE_RW(void) fill(C val)
    {
        for(size_t i = 0; i < len; ++i)
        {
            str[i] = val;
        }
    }

public:

    /** set the current substring to a copy of the given csubstr
     * @note this method requires that the string memory is writeable and is SFINAEd out for const C */
    C4_REQUIRE_RW(void) copy_from(ro_substr that, size_t ifirst=0, size_t num=npos)
    {
        C4_ASSERT(ifirst >= 0 && ifirst <= len);
        num = num != npos ? num : len - ifirst;
        num = num < that.len ? num : that.len;
        C4_ASSERT(ifirst + num >= 0 && ifirst + num <= len);
        // calling memcpy with null strings is undefined behavior
        // and will wreak havoc in calling code's branches.
        // see https://github.com/biojppm/rapidyaml/pull/264#issuecomment-1262133637
        if(num)
            memcpy(str + sizeof(C) * ifirst, that.str, sizeof(C) * num);
    }

public:

    /** reverse in place
     * @note this method requires that the string memory is writeable and is SFINAEd out for const C */
    C4_REQUIRE_RW(void) reverse()
    {
        if(len == 0) return;
        detail::_do_reverse(str, str + len - 1);
    }

    /** revert a subpart in place
     * @note this method requires that the string memory is writeable and is SFINAEd out for const C */
    C4_REQUIRE_RW(void) reverse_sub(size_t ifirst, size_t num)
    {
        C4_ASSERT(ifirst >= 0 && ifirst <= len);
        C4_ASSERT(ifirst + num >= 0 && ifirst + num <= len);
        if(num == 0) return;
        detail::_do_reverse(str + ifirst, str + ifirst + num - 1);
    }

    /** revert a range in place
     * @note this method requires that the string memory is writeable and is SFINAEd out for const C */
    C4_REQUIRE_RW(void) reverse_range(size_t ifirst, size_t ilast)
    {
        C4_ASSERT(ifirst >= 0 && ifirst <= len);
        C4_ASSERT(ilast  >= 0 && ilast  <= len);
        if(ifirst == ilast) return;
        detail::_do_reverse(str + ifirst, str + ilast - 1);
    }

public:

    /** erase part of the string. eg, with char s[] = "0123456789",
     * substr(s).erase(3, 2) = "01256789", and s is now "01245678989"
     * @note this method requires that the string memory is writeable and is SFINAEd out for const C */
    C4_REQUIRE_RW(basic_substring) erase(size_t pos, size_t num)
    {
        C4_ASSERT(pos >= 0 && pos+num <= len);
        size_t num_to_move = len - pos - num;
        memmove(str + pos, str + pos + num, sizeof(C) * num_to_move);
        return basic_substring{str, len - num};
    }

    /** @note this method requires that the string memory is writeable and is SFINAEd out for const C */
    C4_REQUIRE_RW(basic_substring) erase_range(size_t first, size_t last)
    {
        C4_ASSERT(first <= last);
        return erase(first, static_cast<size_t>(last-first));
    }

    /** erase a part of the string.
     * @note @p sub must be a substring of this string
     * @note this method requires that the string memory is writeable and is SFINAEd out for const C */
    C4_REQUIRE_RW(basic_substring) erase(ro_substr sub)
    {
        C4_ASSERT(is_super(sub));
        C4_ASSERT(sub.str >= str);
        return erase(static_cast<size_t>(sub.str - str), sub.len);
    }

public:

    /** replace every occurrence of character @p value with the character @p repl
     * @return the number of characters that were replaced
     * @note this method requires that the string memory is writeable and is SFINAEd out for const C */
    C4_REQUIRE_RW(size_t) replace(C value, C repl, size_t pos=0)
    {
        C4_ASSERT((pos >= 0 && pos <= len) || pos == npos);
        size_t did_it = 0;
        while((pos = find(value, pos)) != npos)
        {
            str[pos++] = repl;
            ++did_it;
        }
        return did_it;
    }

    /** replace every occurrence of each character in @p value with
     * the character @p repl.
     * @return the number of characters that were replaced
     * @note this method requires that the string memory is writeable and is SFINAEd out for const C */
    C4_REQUIRE_RW(size_t) replace(ro_substr chars, C repl, size_t pos=0)
    {
        C4_ASSERT((pos >= 0 && pos <= len) || pos == npos);
        size_t did_it = 0;
        while((pos = first_of(chars, pos)) != npos)
        {
            str[pos++] = repl;
            ++did_it;
        }
        return did_it;
    }

    /** replace @p pattern with @p repl, and write the result into
     * @dst. pattern and repl don't need equal sizes.
     *
     * @return the required size for dst. No overflow occurs if
     * dst.len is smaller than the required size; this can be used to
     * determine the required size for an existing container. */
    size_t replace_all(rw_substr dst, ro_substr pattern, ro_substr repl, size_t pos=0) const
    {
        C4_ASSERT( ! pattern.empty()); //!< @todo relax this precondition
        C4_ASSERT( ! this  ->overlaps(dst)); //!< @todo relax this precondition
        C4_ASSERT( ! pattern.overlaps(dst));
        C4_ASSERT( ! repl   .overlaps(dst));
        C4_ASSERT((pos >= 0 && pos <= len) || pos == npos);
        C4_SUPPRESS_WARNING_GCC_PUSH
        C4_SUPPRESS_WARNING_GCC("-Warray-bounds")  // gcc11 has a false positive here
        #if (!defined(__clang__)) && (defined(__GNUC__) && (__GNUC__ >= 7))
        C4_SUPPRESS_WARNING_GCC("-Wstringop-overflow")  // gcc11 has a false positive here
        #endif
        #define _c4append(first, last)                                  \
            {                                                           \
                C4_ASSERT((last) >= (first));                           \
                size_t num = static_cast<size_t>((last) - (first));     \
                if(num > 0 && sz + num <= dst.len)                      \
                {                                                       \
                    memcpy(dst.str + sz, first, num * sizeof(C));       \
                }                                                       \
                sz += num;                                              \
            }
        size_t sz = 0;
        size_t b = pos;
        _c4append(str, str + pos);
        do {
            size_t e = find(pattern, b);
            if(e == npos)
            {
                _c4append(str + b, str + len);
                break;
            }
            _c4append(str + b, str + e);
            _c4append(repl.begin(), repl.end());
            b = e + pattern.size();
        } while(b < len && b != npos);
        return sz;
        #undef _c4append
        C4_SUPPRESS_WARNING_GCC_POP
    }

    /** @} */

}; // template class basic_substring


#undef C4_REQUIRE_RW


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------


/** @name Adapter functions. to_substr() and to_csubstr() is used in
 * generic code like format(), and allow adding construction of
 * substrings from new types like containers. */
/** @{ */


/** neutral version for use in generic code */
C4_ALWAYS_INLINE substr to_substr(substr s) noexcept { return s; }
/** neutral version for use in generic code */
C4_ALWAYS_INLINE csubstr to_csubstr(substr s) noexcept { return s; }
/** neutral version for use in generic code */
C4_ALWAYS_INLINE csubstr to_csubstr(csubstr s) noexcept { return s; }


template<size_t N>
C4_ALWAYS_INLINE substr
to_substr(char (&s)[N]) noexcept { substr ss(s, N-1); return ss; }
template<size_t N>
C4_ALWAYS_INLINE csubstr
to_csubstr(const char (&s)[N]) noexcept { csubstr ss(s, N-1); return ss; }


/** @note this overload uses SFINAE to prevent it from overriding the array overload
 * @see For a more detailed explanation on why the plain overloads cannot
 * coexist, see http://cplusplus.bordoon.com/specializeForCharacterArrays.html */
template<class U>
C4_ALWAYS_INLINE typename std::enable_if<std::is_same<U, char*>::value, substr>::type
to_substr(U s) noexcept { substr ss(s); return ss; }
/** @note this overload uses SFINAE to prevent it from overriding the array overload
 * @see For a more detailed explanation on why the plain overloads cannot
 * coexist, see http://cplusplus.bordoon.com/specializeForCharacterArrays.html */
template<class U>
C4_ALWAYS_INLINE typename std::enable_if<std::is_same<U, const char*>::value || std::is_same<U, char*>::value, csubstr>::type
to_csubstr(U s) noexcept { csubstr ss(s); return ss; }


/** @} */


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

template<typename C, size_t N> inline bool operator== (const char (&s)[N], basic_substring<C> const that) noexcept { return that.compare(s, N-1) == 0; }
template<typename C, size_t N> inline bool operator!= (const char (&s)[N], basic_substring<C> const that) noexcept { return that.compare(s, N-1) != 0; }
template<typename C, size_t N> inline bool operator<  (const char (&s)[N], basic_substring<C> const that) noexcept { return that.compare(s, N-1) >  0; }
template<typename C, size_t N> inline bool operator>  (const char (&s)[N], basic_substring<C> const that) noexcept { return that.compare(s, N-1) <  0; }
template<typename C, size_t N> inline bool operator<= (const char (&s)[N], basic_substring<C> const that) noexcept { return that.compare(s, N-1) >= 0; }
template<typename C, size_t N> inline bool operator>= (const char (&s)[N], basic_substring<C> const that) noexcept { return that.compare(s, N-1) <= 0; }

template<typename C> inline bool operator== (const char c, basic_substring<C> const that) noexcept { return that.compare(c) == 0; }
template<typename C> inline bool operator!= (const char c, basic_substring<C> const that) noexcept { return that.compare(c) != 0; }
template<typename C> inline bool operator<  (const char c, basic_substring<C> const that) noexcept { return that.compare(c) >  0; }
template<typename C> inline bool operator>  (const char c, basic_substring<C> const that) noexcept { return that.compare(c) <  0; }
template<typename C> inline bool operator<= (const char c, basic_substring<C> const that) noexcept { return that.compare(c) >= 0; }
template<typename C> inline bool operator>= (const char c, basic_substring<C> const that) noexcept { return that.compare(c) <= 0; }


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

/** @define C4_SUBSTR_NO_OSTREAM_LSHIFT doctest does not deal well with
 * template operator<<
 * @see https://github.com/onqtam/doctest/pull/431 */
#ifndef C4_SUBSTR_NO_OSTREAM_LSHIFT
#ifdef __clang__
#   pragma clang diagnostic push
#   pragma clang diagnostic ignored "-Wsign-conversion"
#elif defined(__GNUC__)
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wsign-conversion"
#endif

/** output the string to a stream */
template<class OStream, class C>
inline OStream& operator<< (OStream& os, basic_substring<C> s)
{
    os.write(s.str, s.len);
    return os;
}

// this causes ambiguity
///** this is used by google test */
//template<class OStream, class C>
//inline void PrintTo(basic_substring<C> s, OStream* os)
//{
//    os->write(s.str, s.len);
//}

#ifdef __clang__
#   pragma clang diagnostic pop
#elif defined(__GNUC__)
#   pragma GCC diagnostic pop
#endif
#endif // !C4_SUBSTR_NO_OSTREAM_LSHIFT

} // namespace c4


#ifdef __clang__
#   pragma clang diagnostic pop
#elif defined(__GNUC__)
#   pragma GCC diagnostic pop
#endif

#endif /* _C4_SUBSTR_HPP_ */
