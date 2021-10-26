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

/*
Description:
    A set of fast template based decoders for decoding values of any C integer
    type up to long int size laid out with most significant byte first or least
    significant byte first (aka big endian or little endian).  These are CPU
    byte order agnostic and will function the same regardless of the CPUs native
    byte order.

    Being template based means if the either le or be class is not used then
    template code of unused functions will not be instantiated by the compiler
    and thus shouldn't cause any overhead.
*/

#include <cstddef>

#pragma once


class be
{
    template<int S>
    inline static unsigned long int _peek(const unsigned char * p) {
        return _peek<S/2>(p) << (S/2)*8 | _peek<S/2>(p+S/2);
    }
public:
    template<typename T>
    inline static T peek(const void * p) {
        return T(_peek<sizeof(T)>(static_cast<const unsigned char *>(p)));
    }

    template<typename T>
    inline static T read(const unsigned char * &p) {
        const T r = T(_peek<sizeof(T)>(p));
        p += sizeof r;
        return r;
    }

    template<typename T>
    inline static T swap(const T x) {
        return T(_peek<sizeof(T)>(reinterpret_cast<const unsigned char *>(&x)));
    }

    template<typename T>
    inline static void skip(const unsigned char * &p, size_t n=1) {
        p += sizeof(T)*n;
    }
};

template<>
inline unsigned long int be::_peek<1>(const unsigned char * p) { return *p; }


class le
{
    template<int S>
    inline static unsigned long int _peek(const unsigned char * p) {
        return _peek<S/2>(p) | _peek<S/2>(p+S/2)  << (S/2)*8;
    }
public:
    template<typename T>
    inline static T peek(const void * p) {
        return T(_peek<sizeof(T)>(static_cast<const unsigned char *>(p)));
    }

    template<typename T>
    inline static T read(const unsigned char * &p) {
        const T r = T(_peek<sizeof(T)>(p));
        p += sizeof r;
        return r;
    }

    template<typename T>
    inline static T swap(const T x) {
        return T(_peek<sizeof(T)>(reinterpret_cast<const unsigned char *>(&x)));
    }

    template<typename T>
    inline static void skip(const unsigned char * &p, size_t n=1) {
        p += sizeof(T)*n;
    }
};

template<>
inline unsigned long int le::_peek<1>(const unsigned char * p) { return *p; }
