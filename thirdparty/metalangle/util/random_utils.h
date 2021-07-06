//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// random_utils:
//   Helper functions for random number generation.
//

#ifndef UTIL_RANDOM_UTILS_H
#define UTIL_RANDOM_UTILS_H

// TODO(jmadill): Rework this if Chromium decides to ban <random>
#include <random>
#include <vector>

#include "util/util_export.h"

namespace angle
{

class ANGLE_UTIL_EXPORT RNG
{
  public:
    // Seed from clock
    RNG();
    // Seed from fixed number.
    RNG(unsigned int seed);
    ~RNG();

    void reseed(unsigned int newSeed);

    bool randomBool(float probTrue = 0.5f);
    int randomInt();
    int randomIntBetween(int min, int max);
    unsigned int randomUInt();
    float randomFloat();
    float randomFloatBetween(float min, float max);
    float randomFloatNonnegative();
    float randomNegativeOneToOne();

    template <class T>
    T &randomSelect(std::vector<T> &elements)
    {
        return elements[randomIntBetween(0, static_cast<int>(elements.size()) - 1)];
    }

    template <class T>
    const T &randomSelect(const std::vector<T> &elements)
    {
        return elements.at(randomIntBetween(0, static_cast<int>(elements.size()) - 1));
    }

  private:
    std::default_random_engine mGenerator;
};

// Implemented inline to avoid cross-module allocation issues.
inline void FillVectorWithRandomUBytes(RNG *rng, std::vector<uint8_t> *data)
{
    for (size_t i = 0; i < data->size(); ++i)
    {
        (*data)[i] = static_cast<uint8_t>(rng->randomIntBetween(0, 255));
    }
}

inline void FillVectorWithRandomUBytes(std::vector<uint8_t> *data)
{
    RNG rng;
    FillVectorWithRandomUBytes(&rng, data);
}

}  // namespace angle

#endif  // UTIL_RANDOM_UTILS_H
