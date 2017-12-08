// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#pragma once
#ifndef NV_MATH_KAHANSUM_H
#define NV_MATH_KAHANSUM_H

#include "nvmath.h"

namespace nv
{

    class KahanSum
    {
    public:
        KahanSum() : accum(0.0f), err(0) {};

        void add(float f)
        {
            float compensated = f + err;
            float tmp = accum + compensated;
            err = accum - tmp;
            err += compensated;
            accum = tmp;
        }

        float sum() const
        {
            return accum;
        }

    private:
        float accum;
        float err;
    };

} // nv namespace


#endif // NV_MATH_KAHANSUM_H
