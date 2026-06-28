// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2011, SIL International, All rights reserved.

#pragma once

#include <cstdlib>
#include "inc/Main.h"

namespace graphite2 {

typedef uint32  uchar_t;

template <int N>
struct _utf_codec
{
    typedef uchar_t codeunit_t;

    static void     put(codeunit_t * cp, const uchar_t , int8 & len) throw();
    static uchar_t  get(const codeunit_t * cp, int8 & len) throw();
    static bool     validate(const codeunit_t * s, const codeunit_t * const e) throw();
};


template <>
struct _utf_codec<32>
{
private:
    static const uchar_t    limit = 0x110000;
public:
    typedef uint32  codeunit_t;

    inline
    static void put(codeunit_t * cp, const uchar_t usv, int8 & l) throw()
    {
        *cp = usv; l = 1;
    }

    inline
    static uchar_t get(const codeunit_t * cp, int8 & l) throw()
    {
        if (cp[0] < limit)  { l = 1;  return cp[0]; }
        else                { l = -1; return 0xFFFD; }
    }

    inline
    static bool validate(const codeunit_t * s, const codeunit_t * const e) throw()
    {
        return s <= e;
    }
};


template <>
struct _utf_codec<16>
{
private:
    static const int32  lead_offset      = 0xD800 - (0x10000 >> 10);
    static const int32  surrogate_offset = 0x10000 - (0xD800 << 10) - 0xDC00;
public:
    typedef uint16  codeunit_t;

    inline
    static void put(codeunit_t * cp, const uchar_t usv, int8 & l) throw()
    {
        if (usv < 0x10000)  { l = 1; cp[0] = codeunit_t(usv); }
        else
        {
            cp[0] = codeunit_t(lead_offset + (usv >> 10));
            cp[1] = codeunit_t(0xDC00 + (usv & 0x3FF));
            l = 2;
        }
    }

    inline
    static uchar_t get(const codeunit_t * cp, int8 & l) throw()
    {
        const uint32    uh = cp[0];
        l = 1;

        if (uh < 0xD800|| uh > 0xDFFF) { return uh; }
        if (uh > 0xDBFF) { l = -1; return 0xFFFD; }
        const uint32 ul = cp[1];
        if (ul < 0xDC00 || ul > 0xDFFF) { l = -1; return 0xFFFD; }
        ++l;
        return (uh<<10) + ul + surrogate_offset;
    }

    inline
    static bool validate(const codeunit_t * s, const codeunit_t * const e) throw()
    {
        const ptrdiff_t n = e-s;
        if (n <= 0) return n == 0;
        const uint32 u = *(e-1); // Get the last codepoint
        return (u < 0xD800 || u > 0xDBFF);
    }
};


template <>
struct _utf_codec<8>
{
private:
    static const int8 sz_lut[16];
    static const byte mask_lut[5];
    static const uchar_t    limit = 0x110000;

public:
    typedef uint8   codeunit_t;

    inline
    static void put(codeunit_t * cp, const uchar_t usv, int8 & l) throw()
    {
        if (usv < 0x80)     {l = 1; cp[0] = usv; return; }
        if (usv < 0x0800)   {l = 2; cp[0] = 0xC0 + (usv >> 6);  cp[1] = 0x80 + (usv & 0x3F); return; }
        if (usv < 0x10000)  {l = 3; cp[0] = 0xE0 + (usv >> 12); cp[1] = 0x80 + ((usv >> 6) & 0x3F);  cp[2] = 0x80 + (usv & 0x3F); return; }
        else                {l = 4; cp[0] = 0xF0 + (usv >> 18); cp[1] = 0x80 + ((usv >> 12) & 0x3F); cp[2] = 0x80 + ((usv >> 6) & 0x3F); cp[3] = 0x80 + (usv & 0x3F); return; }
    }

    inline
    static uchar_t get(const codeunit_t * cp, int8 & l) throw()
    {
        const int8 seq_sz = sz_lut[*cp >> 4];
        uchar_t u = *cp & mask_lut[seq_sz];
        l = 1;
        bool toolong = false;

        switch(seq_sz) {
            case 4:     u <<= 6; u |= *++cp & 0x3F; if (*cp >> 6 != 2) break; ++l; toolong  = (u < 0x10); GR_FALLTHROUGH;
                // no break
            case 3:     u <<= 6; u |= *++cp & 0x3F; if (*cp >> 6 != 2) break; ++l; toolong |= (u < 0x20); GR_FALLTHROUGH;
                // no break
            case 2:     u <<= 6; u |= *++cp & 0x3F; if (*cp >> 6 != 2) break; ++l; toolong |= (u < 0x80); GR_FALLTHROUGH;
                // no break
            case 1:     break;
            case 0:     l = -1; return 0xFFFD;
        }

        if (l != seq_sz || toolong  || u >= limit)
        {
            l = -l;
            return 0xFFFD;
        }
        return u;
    }

    inline
    static bool validate(const codeunit_t * s, const codeunit_t * const e) throw()
    {
        const ptrdiff_t n = e-s;
        if (n <= 0) return n == 0;
        s += (n-1);
        if (*s < 0x80) return true;
        if (*s >= 0xC0) return false;
        if (n == 1) return true;
        if (*--s < 0x80) return true;
        if (*s >= 0xE0) return false;
        if (n == 2 || *s >= 0xC0) return true;
        if (*--s < 0x80) return true;
        if (*s >= 0xF0) return false;
        return true;
    }

};


template <typename C>
class _utf_iterator
{
    typedef _utf_codec<sizeof(C)*8> codec;

    C             * cp;
    mutable int8    sl;

public:
    typedef C           codeunit_type;
    typedef uchar_t     value_type;
    typedef uchar_t   * pointer;

    class reference
    {
        const _utf_iterator & _i;

        reference(const _utf_iterator & i): _i(i) {}
    public:
        operator value_type () const throw ()                   { return codec::get(_i.cp, _i.sl); }
        reference & operator = (const value_type usv) throw()   { codec::put(_i.cp, usv, _i.sl); return *this; }

        friend class _utf_iterator;
    };


    _utf_iterator(const void * us=0)    : cp(reinterpret_cast<C *>(const_cast<void *>(us))), sl(1) { }

    _utf_iterator   & operator ++ ()    { cp += abs(sl); return *this; }
    _utf_iterator   operator ++ (int)   { _utf_iterator tmp(*this); operator++(); return tmp; }

    bool operator == (const _utf_iterator & rhs) const throw() { return cp >= rhs.cp; }
    bool operator != (const _utf_iterator & rhs) const throw() { return !operator==(rhs); }

    reference   operator * () const throw() { return *this; }
    pointer     operator ->() const throw() { return &operator *(); }

    operator codeunit_type * () const throw() { return cp; }

    bool error() const throw()  { return sl < 1; }
    bool validate(const _utf_iterator & e)  { return codec::validate(cp, e.cp); }
};

template <typename C>
struct utf
{
    typedef typename _utf_codec<sizeof(C)*8>::codeunit_t codeunit_t;

    typedef _utf_iterator<C>        iterator;
    typedef _utf_iterator<const C>  const_iterator;

    inline
    static bool validate(codeunit_t * s, codeunit_t * e) throw() {
        return _utf_codec<sizeof(C)*8>::validate(s,e);
    }
};


typedef utf<uint32> utf32;
typedef utf<uint16> utf16;
typedef utf<uint8>  utf8;

} // namespace graphite2
