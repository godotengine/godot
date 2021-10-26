/*  GRAPHITE2 LICENSING

    Copyright 2010, SIL International
    All rights reserved.

    This library is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published
    by the Free Software Foundation; either version 2.1 of License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should also have received a copy of the GNU Lesser General Public
    License along with this library in the file named "LICENSE".
    If not, write to the Free Software Foundation, 51 Franklin Street,
    Suite 500, Boston, MA 02110-1335, USA or visit their web page on the
    internet at http://www.fsf.org/licenses/lgpl.html.

Alternatively, the contents of this file may be used under the terms of the
Mozilla Public License (http://mozilla.org/MPL) or the GNU General Public
License, as published by the Free Software Foundation, either version 2
of the License or (at your option) any later version.
*/

// designed to have a limited subset of the std::vector api
#pragma once

#include <cstddef>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <new>

#include "Main.h"

namespace graphite2 {

template <typename T>
inline
ptrdiff_t distance(T* first, T* last) { return last-first; }


template <typename T>
class Vector
{
    T * m_first, *m_last, *m_end;
public:
    typedef       T &   reference;
    typedef const T &   const_reference;
    typedef       T *   iterator;
    typedef const T *   const_iterator;

    Vector() : m_first(0), m_last(0), m_end(0) {}
    Vector(size_t n, const T& value = T())      : m_first(0), m_last(0), m_end(0) { insert(begin(), n, value); }
    Vector(const Vector<T> &rhs)                : m_first(0), m_last(0), m_end(0) { insert(begin(), rhs.begin(), rhs.end()); }
    template <typename I>
    Vector(I first, const I last)               : m_first(0), m_last(0), m_end(0) { insert(begin(), first, last); }
    ~Vector() { clear(); free(m_first); }

    iterator            begin()         { return m_first; }
    const_iterator      begin() const   { return m_first; }

    iterator            end()           { return m_last; }
    const_iterator      end() const     { return m_last; }

    bool                empty() const   { return m_first == m_last; }
    size_t              size() const    { return m_last - m_first; }
    size_t              capacity() const{ return m_end - m_first; }

    void                reserve(size_t n);
    void                resize(size_t n, const T & v = T());

    reference           front()         { assert(size() > 0); return *begin(); }
    const_reference     front() const   { assert(size() > 0); return *begin(); }
    reference           back()          { assert(size() > 0); return *(end()-1); }
    const_reference     back() const    { assert(size() > 0); return *(end()-1); }

    Vector<T>         & operator = (const Vector<T> & rhs) { assign(rhs.begin(), rhs.end()); return *this; }
    reference           operator [] (size_t n)          { assert(size() > n); return m_first[n]; }
    const_reference     operator [] (size_t n) const    { assert(size() > n); return m_first[n]; }

    void                assign(size_t n, const T& u)    { clear(); insert(begin(), n, u); }
    void                assign(const_iterator first, const_iterator last)      { clear(); insert(begin(), first, last); }
    iterator            insert(iterator p, const T & x) { p = _insert_default(p, 1); new (p) T(x); return p; }
    void                insert(iterator p, size_t n, const T & x);
    void                insert(iterator p, const_iterator first, const_iterator last);
    void                pop_back()              { assert(size() > 0); --m_last; }
    void                push_back(const T &v)   { if (m_last == m_end) reserve(size()+1); new (m_last++) T(v); }

    void                clear()                 { erase(begin(), end()); }
    iterator            erase(iterator p)       { return erase(p, p+1); }
    iterator            erase(iterator first, iterator last);

private:
    iterator            _insert_default(iterator p, size_t n);
};

template <typename T>
inline
void Vector<T>::reserve(size_t n)
{
    if (n > capacity())
    {
        const ptrdiff_t sz = size();
        size_t requested;
        if (checked_mul(n,sizeof(T), requested))  std::abort();
        m_first = static_cast<T*>(realloc(m_first, requested));
        if (!m_first)   std::abort();
        m_last  = m_first + sz;
        m_end   = m_first + n;
    }
}

template <typename T>
inline
void Vector<T>::resize(size_t n, const T & v) {
    const ptrdiff_t d = n-size();
    if (d < 0)      erase(end()+d, end());
    else if (d > 0) insert(end(), d, v);
}

template<typename T>
inline
typename Vector<T>::iterator Vector<T>::_insert_default(iterator p, size_t n)
{
    assert(begin() <= p && p <= end());
    const ptrdiff_t i = p - begin();
    reserve(((size() + n + 7) >> 3) << 3);
    p = begin() + i;
    // Move tail if there is one
    if (p != end()) memmove(p + n, p, distance(p,end())*sizeof(T));
    m_last += n;
    return p;
}

template<typename T>
inline
void Vector<T>::insert(iterator p, size_t n, const T & x)
{
    p = _insert_default(p, n);
    // Copy in elements
    for (; n; --n, ++p) { new (p) T(x); }
}

template<typename T>
inline
void Vector<T>::insert(iterator p, const_iterator first, const_iterator last)
{
    p = _insert_default(p, distance(first, last));
    // Copy in elements
    for (;first != last; ++first, ++p) { new (p) T(*first); }
}

template<typename T>
inline
typename Vector<T>::iterator Vector<T>::erase(iterator first, iterator last)
{
    for (iterator e = first; e != last; ++e) e->~T();
    const size_t sz = distance(first, last);
    if (m_last != last) memmove(first, last, distance(last,end())*sizeof(T));
    m_last -= sz;
    return first;
}

} // namespace graphite2
