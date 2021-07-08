//
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// mathutil.cpp: Math and bit manipulation functions.

#include "common/mathutil.h"

#include <math.h>
#include <algorithm>

namespace gl
{

namespace
{

struct RGB9E5Data
{
    unsigned int R : 9;
    unsigned int G : 9;
    unsigned int B : 9;
    unsigned int E : 5;
};

// B is the exponent bias (15)
constexpr int g_sharedexp_bias = 15;

// N is the number of mantissa bits per component (9)
constexpr int g_sharedexp_mantissabits = 9;

// Emax is the maximum allowed biased exponent value (31)
constexpr int g_sharedexp_maxexponent = 31;

constexpr float g_sharedexp_max =
    ((static_cast<float>(1 << g_sharedexp_mantissabits) - 1) /
     static_cast<float>(1 << g_sharedexp_mantissabits)) *
    static_cast<float>(1 << (g_sharedexp_maxexponent - g_sharedexp_bias));

}  // anonymous namespace

unsigned int convertRGBFloatsTo999E5(float red, float green, float blue)
{
    const float red_c   = std::max<float>(0, std::min(g_sharedexp_max, red));
    const float green_c = std::max<float>(0, std::min(g_sharedexp_max, green));
    const float blue_c  = std::max<float>(0, std::min(g_sharedexp_max, blue));

    const float max_c = std::max<float>(std::max<float>(red_c, green_c), blue_c);
    const float exp_p =
        std::max<float>(-g_sharedexp_bias - 1, floor(log(max_c))) + 1 + g_sharedexp_bias;
    const int max_s = static_cast<int>(
        floor((max_c / (pow(2.0f, exp_p - g_sharedexp_bias - g_sharedexp_mantissabits))) + 0.5f));
    const int exp_s =
        static_cast<int>((max_s < pow(2.0f, g_sharedexp_mantissabits)) ? exp_p : exp_p + 1);

    RGB9E5Data output;
    output.R = static_cast<unsigned int>(
        floor((red_c / (pow(2.0f, exp_s - g_sharedexp_bias - g_sharedexp_mantissabits))) + 0.5f));
    output.G = static_cast<unsigned int>(
        floor((green_c / (pow(2.0f, exp_s - g_sharedexp_bias - g_sharedexp_mantissabits))) + 0.5f));
    output.B = static_cast<unsigned int>(
        floor((blue_c / (pow(2.0f, exp_s - g_sharedexp_bias - g_sharedexp_mantissabits))) + 0.5f));
    output.E = exp_s;

    return bitCast<unsigned int>(output);
}

void convert999E5toRGBFloats(unsigned int input, float *red, float *green, float *blue)
{
    const RGB9E5Data *inputData = reinterpret_cast<const RGB9E5Data *>(&input);

    *red =
        inputData->R * pow(2.0f, (int)inputData->E - g_sharedexp_bias - g_sharedexp_mantissabits);
    *green =
        inputData->G * pow(2.0f, (int)inputData->E - g_sharedexp_bias - g_sharedexp_mantissabits);
    *blue =
        inputData->B * pow(2.0f, (int)inputData->E - g_sharedexp_bias - g_sharedexp_mantissabits);
}

int BitCountPolyfill(uint32_t bits)
{
    int ones = 0;
    while (bits)
    {
        ones += bool(bits & 1);
        bits >>= 1;
    }
    return ones;
}

}  // namespace gl
