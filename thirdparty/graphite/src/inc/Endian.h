// SPDX-License-Identifier: MIT OR MPL-2.0 OR LGPL-2.1-or-later OR GPL-2.0-or-later
// Copyright 2011, SIL International, All rights reserved.
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
