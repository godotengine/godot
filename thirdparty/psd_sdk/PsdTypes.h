// Copyright 2011-2020, Molecular Matters GmbH <office@molecular-matters.com>
// See LICENSE.txt for licensing details (2-clause BSD License: https://opensource.org/licenses/BSD-2-Clause)

#pragma once


#include "Psdstdint.h"


/// \ingroup Platform
/// \brief Defines a standard 32-bit floating-point type.
typedef float float32_t;


/// \ingroup Platform
/// \brief Defines a standard 64-bit floating-point type.
typedef double float64_t;


static_assert(sizeof(uint8_t) == 1, "sizeof(uint8_t) is not 1 byte");
static_assert(sizeof(int8_t) == 1, "sizeof(int8_t) is not 1 byte");

static_assert(sizeof(uint16_t) == 2, "sizeof(uint16_t) is not 2 bytes");
static_assert(sizeof(int16_t) == 2, "sizeof(int16_t) is not 2 bytes");

static_assert(sizeof(uint32_t) == 4, "sizeof(uint32_t) is not 4 bytes");
static_assert(sizeof(int32_t) == 4, "sizeof(int32_t) is not 4 bytes");

static_assert(sizeof(uint64_t) == 8, "sizeof(uint64_t) is not 8 bytes");
static_assert(sizeof(int64_t) == 8, "sizeof(int64_t) is not 8 bytes");

static_assert(sizeof(float32_t) == 4, "sizeof(float32_t) is not 4 bytes");
static_assert(sizeof(float64_t) == 8, "sizeof(float64_t) is not 8 bytes");
