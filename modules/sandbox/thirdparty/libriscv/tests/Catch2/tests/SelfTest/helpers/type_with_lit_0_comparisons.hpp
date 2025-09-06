
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#ifndef CATCH_TEST_HELPERS_TYPE_WITH_LIT_0_COMPARISONS_HPP_INCLUDED
#define CATCH_TEST_HELPERS_TYPE_WITH_LIT_0_COMPARISONS_HPP_INCLUDED

#include <type_traits>

// Should only be constructible from literal 0.
// Based on the constructor from pointer trick, used by libstdc++ and libc++
// (formerly also MSVC, but they've moved to consteval int constructor).
// Used by `TypeWithLit0Comparisons` for testing comparison
// ops that only work with literal zero, the way std::*orderings do
struct ZeroLiteralAsPointer {
    constexpr ZeroLiteralAsPointer( ZeroLiteralAsPointer* ) noexcept {}

    template <typename T,
              typename = std::enable_if_t<!std::is_same<T, int>::value>>
    constexpr ZeroLiteralAsPointer( T ) = delete;
};


struct TypeWithLit0Comparisons {
#define DEFINE_COMP_OP( op )                                          \
    constexpr friend bool operator op( TypeWithLit0Comparisons,       \
                                       ZeroLiteralAsPointer ) {       \
        return true;                                                  \
    }                                                                 \
    constexpr friend bool operator op( ZeroLiteralAsPointer,          \
                                       TypeWithLit0Comparisons ) {    \
        return false;                                                 \
    }                                                                 \
    /* std::orderings only have these for ==, but we add them for all \
       operators so we can test all overloads for decomposer */       \
    constexpr friend bool operator op( TypeWithLit0Comparisons,       \
                                       TypeWithLit0Comparisons ) {    \
        return true;                                                  \
    }

    DEFINE_COMP_OP( < )
    DEFINE_COMP_OP( <= )
    DEFINE_COMP_OP( > )
    DEFINE_COMP_OP( >= )
    DEFINE_COMP_OP( == )
    DEFINE_COMP_OP( != )

#undef DEFINE_COMP_OP
};

#endif // CATCH_TEST_HELPERS_TYPE_WITH_LIT_0_COMPARISONS_HPP_INCLUDED
