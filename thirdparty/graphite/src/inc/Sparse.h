/*  GRAPHITE2 LICENSING

    Copyright 2011, SIL International
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
#pragma once
#include <iterator>
#include <utility>

#include "inc/Main.h"

namespace graphite2 {


// A read-only packed fast sparse array of uint16 with uint16 keys.
// Like most container classes this has capacity and size properties and these
// refer to the number of stored entries and the number of addressable entries
// as normal. However due the sparse nature the capacity is always <= than the
// size.
class sparse
{
public:
    typedef uint16  key_type;
    typedef uint16  mapped_type;
    typedef std::pair<const key_type, mapped_type> value_type;

private:
    typedef unsigned long   mask_t;

    static const unsigned char  SIZEOF_CHUNK = (sizeof(mask_t) - sizeof(key_type))*8;

    struct chunk
    {
        mask_t          mask:SIZEOF_CHUNK;
        key_type        offset;
    };

    static const chunk  empty_chunk;
    sparse(const sparse &);
    sparse & operator = (const sparse &);

public:
    template<typename I>
    sparse(I first, const I last);
    sparse() throw();
    ~sparse() throw();

    operator bool () const throw();
    mapped_type     operator [] (const key_type k) const throw();

    size_t capacity() const throw();
    size_t size()     const throw();

    size_t _sizeof() const throw();

    CLASS_NEW_DELETE;

private:
    union {
        chunk         * map;
        mapped_type   * values;
    }           m_array;
    key_type    m_nchunks;
};


inline
sparse::sparse() throw() : m_nchunks(0)
{
    m_array.map = const_cast<graphite2::sparse::chunk *>(&empty_chunk);
}


template <typename I>
sparse::sparse(I attr, const I last)
: m_nchunks(0)
{
    m_array.map = 0;

    // Find the maximum extent of the key space.
    size_t n_values=0;
    long lastkey = -1;
    for (I i = attr; i != last; ++i, ++n_values)
    {
        const typename std::iterator_traits<I>::value_type v = *i;
        if (v.second == 0)      { --n_values; continue; }
        if (v.first <= lastkey) { m_nchunks = 0; return; }

        lastkey = v.first;
        const key_type k = v.first / SIZEOF_CHUNK;
        if (k >= m_nchunks) m_nchunks = k+1;
    }
    if (m_nchunks == 0)
    {
        m_array.map=const_cast<graphite2::sparse::chunk *>(&empty_chunk);
        return;
    }

    m_array.values = grzeroalloc<mapped_type>((m_nchunks*sizeof(chunk) + sizeof(mapped_type)-1)
                                                 / sizeof(mapped_type)
                                                 + n_values);

    if (m_array.values == 0)
        return;

    // coverity[forward_null : FALSE] Since m_array is union and m_array.values is not NULL
    chunk * ci = m_array.map;
    ci->offset = (m_nchunks*sizeof(chunk) + sizeof(mapped_type)-1)/sizeof(mapped_type);
    mapped_type * vi = m_array.values + ci->offset;
    for (; attr != last; ++attr, ++vi)
    {
        const typename std::iterator_traits<I>::value_type v = *attr;
        if (v.second == 0)  { --vi; continue; }

        chunk * const ci_ = m_array.map + v.first/SIZEOF_CHUNK;

        if (ci != ci_)
        {
            ci = ci_;
            ci->offset = key_type(vi - m_array.values);
        }

        ci->mask |= 1UL << (SIZEOF_CHUNK - 1 - (v.first % SIZEOF_CHUNK));
        *vi = v.second;
    }
}


inline
sparse::operator bool () const throw()
{
    return m_array.map != 0;
}

inline
size_t sparse::size() const throw()
{
    return m_nchunks*SIZEOF_CHUNK;
}

inline
size_t sparse::_sizeof() const throw()
{
    return sizeof(sparse) + capacity()*sizeof(mapped_type) + m_nchunks*sizeof(chunk);
}

} // namespace graphite2
