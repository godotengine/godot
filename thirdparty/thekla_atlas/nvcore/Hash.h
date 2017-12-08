// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#pragma once
#ifndef NV_CORE_HASH_H
#define NV_CORE_HASH_H

#include "nvcore.h"

namespace nv
{
    inline uint sdbmHash(const void * data_in, uint size, uint h = 5381)
    {
        const uint8 * data = (const uint8 *) data_in;
        uint i = 0;
        while (i < size) {
            h = (h << 16) + (h << 6) - h + (uint) data[i++];
        }
        return h;
    }

    // Note that this hash does not handle NaN properly.
    inline uint sdbmFloatHash(const float * f, uint count, uint h = 5381)
    {
        for (uint i = 0; i < count; i++) {
            //nvDebugCheck(nv::isFinite(*f));
            union { float f; uint32 i; } x = { f[i] };
            if (x.i == 0x80000000) x.i = 0;
            h = sdbmHash(&x, 4, h);
        }
        return h;
    }


    template <typename T>
    inline uint hash(const T & t, uint h = 5381)
    {
        return sdbmHash(&t, sizeof(T), h);
    }

    template <>
    inline uint hash(const float & f, uint h)
    {
        return sdbmFloatHash(&f, 1, h);
    }


    // Functors for hash table:
    template <typename Key> struct Hash 
    {
        uint operator()(const Key & k) const {
            return hash(k);
        }
    };

    template <typename Key> struct Equal
    {
        bool operator()(const Key & k0, const Key & k1) const {
            return k0 == k1;
        }
    };


    // @@ Move to Utils.h?
    template <typename T1, typename T2>
    struct Pair {
        T1 first;
        T2 second;
    };

    template <typename T1, typename T2>
    bool operator==(const Pair<T1,T2> & p0, const Pair<T1,T2> & p1) {
        return p0.first == p1.first && p0.second == p1.second;
    }

    template <typename T1, typename T2>
    uint hash(const Pair<T1,T2> & p, uint h = 5381) {
        return hash(p.second, hash(p.first));
    }


} // nv namespace

#endif // NV_CORE_HASH_H
