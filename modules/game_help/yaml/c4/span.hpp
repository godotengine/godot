#ifndef _C4_SPAN_HPP_
#define _C4_SPAN_HPP_

/** @file span.hpp Provides span classes. */

#include "config.hpp"
#include "error.hpp"
#include "szconv.hpp"

#include <algorithm>

namespace c4 {

C4_SUPPRESS_WARNING_GCC_CLANG_WITH_PUSH("-Wold-style-cast")

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
/** a crtp base for implementing span classes
 *
 * A span is a non-owning range of elements contiguously stored in memory.
 * Unlike STL's array_view, the span allows write-access to its members.
 *
 * To obtain subspans from a span, the following const member functions
 * are available:
 *  - subspan(first, num)
 *  - range(first, last)
 *  - first(num)
 *  - last(num)
 *
 * A span can also be resized via the following non-const member functions:
 *  - resize(sz)
 *  - ltrim(num)
 *  - rtrim(num)
 *
 * @see span
 * @see cspan
 * @see spanrs
 * @see cspanrs
 * @see spanrsl
 * @see cspanrsl
 */
template<class T, class I, class SpanImpl>
class span_crtp
{
// some utility defines, undefined at the end of this class
#define _c4this  ((SpanImpl      *)this)
#define _c4cthis ((SpanImpl const*)this)
#define _c4ptr   ((SpanImpl      *)this)->m_ptr
#define _c4cptr  ((SpanImpl const*)this)->m_ptr
#define _c4sz    ((SpanImpl      *)this)->m_size
#define _c4csz   ((SpanImpl const*)this)->m_size

public:

    _c4_DEFINE_ARRAY_TYPES(T, I);

public:

    C4_ALWAYS_INLINE constexpr I value_size() const noexcept { return sizeof(T); }
    C4_ALWAYS_INLINE constexpr I elm_size  () const noexcept { return sizeof(T); }
    C4_ALWAYS_INLINE constexpr I type_size () const noexcept { return sizeof(T); }
    C4_ALWAYS_INLINE           I byte_size () const noexcept { return _c4csz*sizeof(T); }

    C4_ALWAYS_INLINE bool empty()    const noexcept { return _c4csz == 0; }
    C4_ALWAYS_INLINE I    size()     const noexcept { return _c4csz; }
    //C4_ALWAYS_INLINE I    capacity() const noexcept { return _c4sz; } // this must be defined by impl classes

    C4_ALWAYS_INLINE void clear() noexcept { _c4sz = 0; }

    C4_ALWAYS_INLINE T      * data()       noexcept { return _c4ptr; }
    C4_ALWAYS_INLINE T const* data() const noexcept { return _c4cptr; }

    C4_ALWAYS_INLINE       iterator  begin()       noexcept { return _c4ptr; }
    C4_ALWAYS_INLINE const_iterator  begin() const noexcept { return _c4cptr; }
    C4_ALWAYS_INLINE const_iterator cbegin() const noexcept { return _c4cptr; }

    C4_ALWAYS_INLINE       iterator  end()       noexcept { return _c4ptr  + _c4sz; }
    C4_ALWAYS_INLINE const_iterator  end() const noexcept { return _c4cptr + _c4csz; }
    C4_ALWAYS_INLINE const_iterator cend() const noexcept { return _c4cptr + _c4csz; }

    C4_ALWAYS_INLINE       reverse_iterator  rbegin()       noexcept { return reverse_iterator(_c4ptr + _c4sz); }
    C4_ALWAYS_INLINE const_reverse_iterator  rbegin() const noexcept { return reverse_iterator(_c4cptr + _c4csz); }
    C4_ALWAYS_INLINE const_reverse_iterator crbegin() const noexcept { return reverse_iterator(_c4cptr + _c4csz); }

    C4_ALWAYS_INLINE       reverse_iterator  rend()       noexcept { return const_reverse_iterator(_c4ptr); }
    C4_ALWAYS_INLINE const_reverse_iterator  rend() const noexcept { return const_reverse_iterator(_c4cptr); }
    C4_ALWAYS_INLINE const_reverse_iterator crend() const noexcept { return const_reverse_iterator(_c4cptr); }

    C4_ALWAYS_INLINE T      & front()       C4_NOEXCEPT_X { C4_XASSERT(!empty()); return _c4ptr [0]; }
    C4_ALWAYS_INLINE T const& front() const C4_NOEXCEPT_X { C4_XASSERT(!empty()); return _c4cptr[0]; }

    C4_ALWAYS_INLINE T      & back()       C4_NOEXCEPT_X { C4_XASSERT(!empty()); return _c4ptr [_c4sz  - 1]; }
    C4_ALWAYS_INLINE T const& back() const C4_NOEXCEPT_X { C4_XASSERT(!empty()); return _c4cptr[_c4csz - 1]; }

    C4_ALWAYS_INLINE T      & operator[] (I i)       C4_NOEXCEPT_X { C4_XASSERT(i >= 0 && i < _c4sz ); return _c4ptr [i]; }
    C4_ALWAYS_INLINE T const& operator[] (I i) const C4_NOEXCEPT_X { C4_XASSERT(i >= 0 && i < _c4csz); return _c4cptr[i]; }

    C4_ALWAYS_INLINE SpanImpl subspan(I first, I num) const C4_NOEXCEPT_X
    {
        C4_XASSERT((first >= 0 && first < _c4csz) || (first == _c4csz && num == 0));
        C4_XASSERT((first + num >= 0) && (first + num <= _c4csz));
        return _c4cthis->_select(_c4cptr + first, num);
    }
    C4_ALWAYS_INLINE SpanImpl subspan(I first) const C4_NOEXCEPT_X ///< goes up until the end of the span
    {
        C4_XASSERT(first >= 0 && first <= _c4csz);
        return _c4cthis->_select(_c4cptr + first, _c4csz - first);
    }

    C4_ALWAYS_INLINE SpanImpl range(I first, I last) const C4_NOEXCEPT_X ///< last element is NOT included
    {
        C4_XASSERT(((first >= 0) && (first < _c4csz)) || (first == _c4csz && first == last));
        C4_XASSERT((last >= 0) && (last <= _c4csz));
        C4_XASSERT(last >= first);
        return _c4cthis->_select(_c4cptr + first, last - first);
    }
    C4_ALWAYS_INLINE SpanImpl range(I first) const C4_NOEXCEPT_X ///< goes up until the end of the span
    {
        C4_XASSERT(((first >= 0) && (first <= _c4csz)));
        return _c4cthis->_select(_c4cptr + first, _c4csz - first);
    }

    C4_ALWAYS_INLINE SpanImpl first(I num) const C4_NOEXCEPT_X ///< get the first num elements, starting at 0
    {
        C4_XASSERT((num >= 0) && (num <= _c4csz));
        return _c4cthis->_select(_c4cptr, num);
    }
    C4_ALWAYS_INLINE SpanImpl last(I num) const C4_NOEXCEPT_X ///< get the last num elements, starting at size()-num
    {
        C4_XASSERT((num >= 0) && (num <= _c4csz));
        return _c4cthis->_select(_c4cptr + _c4csz - num, num);
    }

    bool is_subspan(span_crtp const& ss) const noexcept
    {
        if(_c4cptr == nullptr) return false;
        auto *b = begin(), *e = end();
        auto *ssb = ss.begin(), *sse = ss.end();
        if(ssb >= b && sse <= e)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    /** COMPLement Left: return the complement to the left of the beginning of the given subspan.
     * If ss does not begin inside this, returns an empty substring. */
    SpanImpl compll(span_crtp const& ss) const C4_NOEXCEPT_X
    {
        auto ssb = ss.begin();
        auto b = begin();
        auto e = end();
        if(ssb >= b && ssb <= e)
        {
            return subspan(0, static_cast<size_t>(ssb - b));
        }
        else
        {
            return subspan(0, 0);
        }
    }

    /** COMPLement Right: return the complement to the right of the end of the given subspan.
     * If ss does not end inside this, returns an empty substring. */
    SpanImpl complr(span_crtp const& ss) const C4_NOEXCEPT_X
    {
        auto sse = ss.end();
        auto b = begin();
        auto e = end();
        if(sse >= b && sse <= e)
        {
            return subspan(static_cast<size_t>(sse - b), static_cast<size_t>(e - sse));
        }
        else
        {
            return subspan(0, 0);
        }
    }

    C4_ALWAYS_INLINE bool same_span(span_crtp const& that) const noexcept
    {
        return size() == that.size() && data() == that.data();
    }
    template<class I2, class Impl2>
    C4_ALWAYS_INLINE bool same_span(span_crtp<T, I2, Impl2> const& that) const C4_NOEXCEPT_X
    {
        I tsz = szconv<I>(that.size()); // x-asserts that the size does not overflow
        return size() == tsz && data() == that.data();
    }

#undef _c4this
#undef _c4cthis
#undef _c4ptr
#undef _c4cptr
#undef _c4sz
#undef _c4csz
};

//-----------------------------------------------------------------------------
template<class T, class Il, class Ir, class _Impll, class _Implr>
inline constexpr bool operator==
(
    span_crtp<T, Il, _Impll> const& l,
    span_crtp<T, Ir, _Implr> const& r
)
{
#if C4_CPP >= 14
    return std::equal(l.begin(), l.end(), r.begin(), r.end());
#else
    return l.same_span(r) || std::equal(l.begin(), l.end(), r.begin());
#endif
}

template<class T, class Il, class Ir, class _Impll, class _Implr>
inline constexpr bool operator!=
(
    span_crtp<T, Il, _Impll> const& l,
    span_crtp<T, Ir, _Implr> const& r
)
{
    return ! (l == r);
}

//-----------------------------------------------------------------------------
template<class T, class Il, class Ir, class _Impll, class _Implr>
inline constexpr bool operator<
(
    span_crtp<T, Il, _Impll> const& l,
    span_crtp<T, Ir, _Implr> const& r
)
{
    return std::lexicographical_compare(l.begin(), l.end(), r.begin(), r.end());
}

template<class T, class Il, class Ir, class _Impll, class _Implr>
inline constexpr bool operator<=
(
    span_crtp<T, Il, _Impll> const& l,
    span_crtp<T, Ir, _Implr> const& r
)
{
    return ! (l > r);
}

//-----------------------------------------------------------------------------
template<class T, class Il, class Ir, class _Impll, class _Implr>
inline constexpr bool operator>
(
    span_crtp<T, Il, _Impll> const& l,
    span_crtp<T, Ir, _Implr> const& r
)
{
    return r < l;
}

//-----------------------------------------------------------------------------
template<class T, class Il, class Ir, class _Impll, class _Implr>
inline constexpr bool operator>=
(
    span_crtp<T, Il, _Impll> const& l,
    span_crtp<T, Ir, _Implr> const& r
)
{
    return ! (l < r);
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
/** A non-owning span of elements contiguously stored in memory. */
template<class T, class I=C4_SIZE_TYPE>
class span : public span_crtp<T, I, span<T, I>>
{
    friend class span_crtp<T, I, span<T, I>>;

    T * C4_RESTRICT m_ptr;
    I   m_size;

    C4_ALWAYS_INLINE span _select(T *p, I sz) const { return span(p, sz); }

public:

    _c4_DEFINE_ARRAY_TYPES(T, I);
    using NCT = typename std::remove_const<T>::type; //!< NCT=non const type
    using CT = typename std::add_const<T>::type; //!< CT=const type
    using const_type = span<CT, I>;

    /// convert automatically to span of const T
    operator span<CT, I> () const { span<CT, I> s(m_ptr, m_size); return s; }

public:

    C4_ALWAYS_INLINE C4_CONSTEXPR14 span() noexcept : m_ptr{nullptr}, m_size{0} {}

    span(span const&) = default;
    span(span     &&) = default;

    span& operator= (span const&) = default;
    span& operator= (span     &&) = default;

public:

    /** @name Construction and assignment from same type */
    /** @{ */

    template<size_t N> C4_ALWAYS_INLINE C4_CONSTEXPR14      span  (T (&arr)[N]) noexcept : m_ptr{arr}, m_size{N} {}
    template<size_t N> C4_ALWAYS_INLINE C4_CONSTEXPR14 void assign(T (&arr)[N]) noexcept { m_ptr = arr; m_size = N; }

    C4_ALWAYS_INLINE C4_CONSTEXPR14        span(T *p, I sz) noexcept : m_ptr{p}, m_size{sz} {}
    C4_ALWAYS_INLINE C4_CONSTEXPR14 void   assign(T *p, I sz) noexcept { m_ptr = p; m_size = sz; }

    C4_ALWAYS_INLINE C4_CONSTEXPR14      span  (c4::aggregate_t, std::initializer_list<T> il) noexcept : m_ptr{&*il.begin()}, m_size{il.size()} {}
    C4_ALWAYS_INLINE C4_CONSTEXPR14 void assign(c4::aggregate_t, std::initializer_list<T> il) noexcept { m_ptr = &*il.begin(); m_size = il.size(); }

    /** @} */

public:

    C4_ALWAYS_INLINE I capacity() const noexcept { return m_size; }

    C4_ALWAYS_INLINE void resize(I sz) C4_NOEXCEPT_A { C4_ASSERT(sz <= m_size); m_size = sz; }
    C4_ALWAYS_INLINE void rtrim (I n ) C4_NOEXCEPT_A { C4_ASSERT(n >= 0 && n < m_size); m_size -= n; }
    C4_ALWAYS_INLINE void ltrim (I n ) C4_NOEXCEPT_A { C4_ASSERT(n >= 0 && n < m_size); m_size -= n; m_ptr += n; }

};
template<class T, class I=C4_SIZE_TYPE> using cspan = span<const T, I>;


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
/** A non-owning span resizeable up to a capacity. Subselection or resizing
 * will keep the original provided it starts at begin(). If subselection or
 * resizing change the pointer, then the original capacity information will
 * be lost.
 *
 * Thus, resizing via resize() and ltrim() and subselecting via first()
 * or any of subspan() or range() when starting from the beginning will keep
 * the original capacity. OTOH, using last(), or any of subspan() or range()
 * with an offset from the start will remove from capacity (shifting the
 * pointer) by the corresponding offset. If this is undesired, then consider
 * using spanrsl.
 *
 * @see spanrs for a span resizeable on the right
 * @see spanrsl for a span resizeable on the right and left
 */

template<class T, class I=C4_SIZE_TYPE>
class spanrs : public span_crtp<T, I, spanrs<T, I>>
{
    friend class span_crtp<T, I, spanrs<T, I>>;

    T * C4_RESTRICT m_ptr;
    I   m_size;
    I   m_capacity;

    C4_ALWAYS_INLINE spanrs _select(T *p, I sz) const noexcept
    {
        C4_ASSERT(p >= m_ptr);
        size_t delta = static_cast<size_t>(p - m_ptr);
        C4_ASSERT(m_capacity >= delta);
        return spanrs(p, sz, static_cast<size_t>(m_capacity - delta));
    }

public:

    _c4_DEFINE_ARRAY_TYPES(T, I);
    using NCT = typename std::remove_const<T>::type; //!< NCT=non const type
    using CT = typename std::add_const<T>::type; //!< CT=const type
    using const_type = spanrs<CT, I>;

    /// convert automatically to span of T
    C4_ALWAYS_INLINE operator span<T, I > () const noexcept { return span<T, I>(m_ptr, m_size); }
    /// convert automatically to span of const T
    //C4_ALWAYS_INLINE operator span<CT, I> () const noexcept { span<CT, I> s(m_ptr, m_size); return s; }
    /// convert automatically to spanrs of const T
    C4_ALWAYS_INLINE operator spanrs<CT, I> () const noexcept { spanrs<CT, I> s(m_ptr, m_size, m_capacity); return s; }

public:

    C4_ALWAYS_INLINE spanrs() noexcept : m_ptr{nullptr}, m_size{0}, m_capacity{0} {}

    spanrs(spanrs const&) = default;
    spanrs(spanrs     &&) = default;

    spanrs& operator= (spanrs const&) = default;
    spanrs& operator= (spanrs     &&) = default;

public:

    /** @name Construction and assignment from same type */
    /** @{ */

    C4_ALWAYS_INLINE      spanrs(T *p, I sz) noexcept : m_ptr{p}, m_size{sz}, m_capacity{sz} {}
    /** @warning will reset the capacity to sz */
    C4_ALWAYS_INLINE void assign(T *p, I sz) noexcept { m_ptr = p; m_size = sz; m_capacity = sz; }

    C4_ALWAYS_INLINE      spanrs(T *p, I sz, I cap) noexcept : m_ptr{p}, m_size{sz}, m_capacity{cap} {}
    C4_ALWAYS_INLINE void assign(T *p, I sz, I cap) noexcept { m_ptr = p; m_size = sz; m_capacity = cap; }

    template<size_t N> C4_ALWAYS_INLINE      spanrs(T (&arr)[N]) noexcept : m_ptr{arr}, m_size{N}, m_capacity{N} {}
    template<size_t N> C4_ALWAYS_INLINE void assign(T (&arr)[N]) noexcept { m_ptr = arr; m_size = N; m_capacity = N; }

    C4_ALWAYS_INLINE      spanrs(c4::aggregate_t, std::initializer_list<T> il) noexcept : m_ptr{il.begin()}, m_size{il.size()}, m_capacity{il.size()} {}
    C4_ALWAYS_INLINE void assign(c4::aggregate_t, std::initializer_list<T> il) noexcept { m_ptr = il.begin(); m_size = il.size(); m_capacity = il.size(); }

    /** @} */

public:

    C4_ALWAYS_INLINE I capacity() const noexcept { return m_capacity; }

    C4_ALWAYS_INLINE void resize(I sz) C4_NOEXCEPT_A { C4_ASSERT(sz <= m_capacity); m_size = sz; }
    C4_ALWAYS_INLINE void rtrim (I n ) C4_NOEXCEPT_A { C4_ASSERT(n >= 0 && n < m_size); m_size -= n; }
    C4_ALWAYS_INLINE void ltrim (I n ) C4_NOEXCEPT_A { C4_ASSERT(n >= 0 && n < m_size); m_size -= n; m_ptr += n; m_capacity -= n; }

};
template<class T, class I=C4_SIZE_TYPE> using cspanrs = spanrs<const T, I>;


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
/** A non-owning span which always retains the capacity of the original
 * range it was taken from (though it may loose its original size).
 * The resizing methods resize(), ltrim(), rtrim() as well
 * as the subselection methods subspan(), range(), first() and last() can be
 * used at will without loosing the original capacity; the full capacity span
 * can always be recovered by calling original().
 */
template<class T, class I=C4_SIZE_TYPE>
class spanrsl : public span_crtp<T, I, spanrsl<T, I>>
{
    friend class span_crtp<T, I, spanrsl<T, I>>;

    T *C4_RESTRICT m_ptr;      ///< the current ptr. the original ptr is (m_ptr - m_offset).
    I   m_size;     ///< the current size. the original size is unrecoverable.
    I   m_capacity; ///< the current capacity. the original capacity is (m_capacity + m_offset).
    I   m_offset;   ///< the offset of the current m_ptr to the start of the original memory block.

    C4_ALWAYS_INLINE spanrsl _select(T *p, I sz) const noexcept
    {
        C4_ASSERT(p >= m_ptr);
        I delta = static_cast<I>(p - m_ptr);
        C4_ASSERT(m_capacity >= delta);
        return spanrsl(p, sz, static_cast<I>(m_capacity - delta), m_offset + delta);
    }

public:

    _c4_DEFINE_ARRAY_TYPES(T, I);
    using NCT = typename std::remove_const<T>::type; //!< NCT=non const type
    using CT = typename std::add_const<T>::type; //!< CT=const type
    using const_type = spanrsl<CT, I>;

    C4_ALWAYS_INLINE operator span<T, I> () const noexcept { return span<T, I>(m_ptr, m_size); }
    C4_ALWAYS_INLINE operator spanrs<T, I> () const noexcept { return spanrs<T, I>(m_ptr, m_size, m_capacity); }
    C4_ALWAYS_INLINE operator spanrsl<CT, I> () const noexcept { return spanrsl<CT, I>(m_ptr, m_size, m_capacity, m_offset); }

public:

    C4_ALWAYS_INLINE spanrsl() noexcept : m_ptr{nullptr}, m_size{0}, m_capacity{0}, m_offset{0} {}

    spanrsl(spanrsl const&) = default;
    spanrsl(spanrsl     &&) = default;

    spanrsl& operator= (spanrsl const&) = default;
    spanrsl& operator= (spanrsl     &&) = default;

public:

    C4_ALWAYS_INLINE     spanrsl(T *p, I sz) noexcept : m_ptr{p}, m_size{sz}, m_capacity{sz}, m_offset{0} {}
    C4_ALWAYS_INLINE void assign(T *p, I sz) noexcept { m_ptr = p; m_size = sz; m_capacity = sz; m_offset = 0; }

    C4_ALWAYS_INLINE     spanrsl(T *p, I sz, I cap) noexcept : m_ptr{p}, m_size{sz}, m_capacity{cap}, m_offset{0} {}
    C4_ALWAYS_INLINE void assign(T *p, I sz, I cap) noexcept { m_ptr = p; m_size = sz; m_capacity = cap; m_offset = 0; }

    C4_ALWAYS_INLINE     spanrsl(T *p, I sz, I cap, I offs) noexcept : m_ptr{p}, m_size{sz}, m_capacity{cap}, m_offset{offs} {}
    C4_ALWAYS_INLINE void assign(T *p, I sz, I cap, I offs) noexcept { m_ptr = p; m_size = sz; m_capacity = cap; m_offset = offs; }

    template<size_t N> C4_ALWAYS_INLINE     spanrsl(T (&arr)[N]) noexcept : m_ptr{arr}, m_size{N}, m_capacity{N}, m_offset{0} {}
    template<size_t N> C4_ALWAYS_INLINE void assign(T (&arr)[N]) noexcept { m_ptr = arr; m_size = N; m_capacity = N; m_offset = 0; }

    C4_ALWAYS_INLINE      spanrsl(c4::aggregate_t, std::initializer_list<T> il) noexcept : m_ptr{il.begin()}, m_size{il.size()}, m_capacity{il.size()}, m_offset{0} {}
    C4_ALWAYS_INLINE void assign (c4::aggregate_t, std::initializer_list<T> il) noexcept { m_ptr = il.begin(); m_size = il.size(); m_capacity = il.size(); m_offset = 0; }

public:

    C4_ALWAYS_INLINE I offset() const noexcept { return m_offset; }
    C4_ALWAYS_INLINE I capacity() const noexcept { return m_capacity; }

    C4_ALWAYS_INLINE void resize(I sz) C4_NOEXCEPT_A { C4_ASSERT(sz <= m_capacity); m_size = sz; }
    C4_ALWAYS_INLINE void rtrim (I n ) C4_NOEXCEPT_A { C4_ASSERT(n >= 0 && n < m_size); m_size -= n; }
    C4_ALWAYS_INLINE void ltrim (I n ) C4_NOEXCEPT_A { C4_ASSERT(n >= 0 && n < m_size); m_size -= n; m_ptr += n; m_offset += n; m_capacity -= n; }

    /** recover the original span as an spanrsl */
    C4_ALWAYS_INLINE spanrsl original() const
    {
        return spanrsl(m_ptr - m_offset, m_capacity + m_offset, m_capacity + m_offset, 0);
    }
    /** recover the original span as a different span type. Example: spanrs<...> orig = s.original<spanrs>(); */
    template<template<class, class> class OtherSpanType>
    C4_ALWAYS_INLINE OtherSpanType<T, I> original()
    {
        return OtherSpanType<T, I>(m_ptr - m_offset, m_capacity + m_offset);
    }
};
template<class T, class I=C4_SIZE_TYPE> using cspanrsl = spanrsl<const T, I>;

C4_SUPPRESS_WARNING_GCC_CLANG_POP

} // namespace c4


#endif /* _C4_SPAN_HPP_ */
