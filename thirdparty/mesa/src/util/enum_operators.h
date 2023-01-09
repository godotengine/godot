/*
 * Copyright © 2020 Red Hat Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifdef __cplusplus

#include "macros.h"
#include <type_traits>

// some enum helpers
#define MESA_DEFINE_CPP_ENUM_BINARY_OPERATOR(Enum, op) \
extern "C++" {                                         \
UNUSED static constexpr                                \
Enum operator op (Enum a, Enum b)                      \
{                                                      \
   using IntType = std::underlying_type_t<Enum>;       \
   IntType ua = static_cast<IntType>(a);               \
   IntType ub = static_cast<IntType>(b);               \
   return static_cast<Enum>(ua op ub);                 \
}                                                      \
                                                       \
UNUSED static constexpr                                \
Enum& operator op##= (Enum &a, Enum b)                 \
{                                                      \
   using IntType = std::underlying_type_t<Enum>;       \
   IntType ua = static_cast<IntType>(a);               \
   IntType ub = static_cast<IntType>(b);               \
   ua op##= ub;                                        \
   a = static_cast<Enum>(ua);                          \
   return a;                                           \
}                                                      \
}

#define MESA_DEFINE_CPP_ENUM_UNARY_OPERATOR(Enum, op) \
extern "C++" {                                        \
UNUSED static constexpr                               \
Enum operator op (Enum a)                             \
{                                                     \
   using IntType = std::underlying_type_t<Enum>;      \
   IntType ua = static_cast<IntType>(a);              \
   return static_cast<Enum>(op ua);                   \
}                                                     \
}

#define MESA_DEFINE_CPP_ENUM_BITFIELD_OPERATORS(Enum) \
MESA_DEFINE_CPP_ENUM_BINARY_OPERATOR(Enum, |)         \
MESA_DEFINE_CPP_ENUM_BINARY_OPERATOR(Enum, &)         \
MESA_DEFINE_CPP_ENUM_BINARY_OPERATOR(Enum, ^)         \
MESA_DEFINE_CPP_ENUM_UNARY_OPERATOR(Enum, ~)

#else

#define MESA_DEFINE_CPP_ENUM_BITFIELD_OPERATORS(Enum)

#endif
