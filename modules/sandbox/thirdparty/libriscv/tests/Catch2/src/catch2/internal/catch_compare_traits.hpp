
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_COMPARE_TRAITS_HPP_INCLUDED
#define CATCH_COMPARE_TRAITS_HPP_INCLUDED

#include <catch2/internal/catch_void_type.hpp>

#include <type_traits>

namespace Catch {
    namespace Detail {

#if defined( __GNUC__ ) && !defined( __clang__ )
#    pragma GCC diagnostic push
    // GCC likes to complain about comparing bool with 0, in the decltype()
    // that defines the comparable traits below.
#    pragma GCC diagnostic ignored "-Wbool-compare"
    // "ordered comparison of pointer with integer zero" same as above,
    // but it does not have a separate warning flag to suppress
#    pragma GCC diagnostic ignored "-Wextra"
    // Did you know that comparing floats with `0` directly
    // is super-duper dangerous in unevaluated context?
#    pragma GCC diagnostic ignored "-Wfloat-equal"
#endif

#if defined( __clang__ )
#    pragma clang diagnostic push
    // Did you know that comparing floats with `0` directly
    // is super-duper dangerous in unevaluated context?
#    pragma clang diagnostic ignored "-Wfloat-equal"
#endif

#define CATCH_DEFINE_COMPARABLE_TRAIT( id, op )                               \
    template <typename, typename, typename = void>                            \
    struct is_##id##_comparable : std::false_type {};                         \
    template <typename T, typename U>                                         \
    struct is_##id##_comparable<                                              \
        T,                                                                    \
        U,                                                                    \
        void_t<decltype( std::declval<T>() op std::declval<U>() )>>           \
        : std::true_type {};                                                  \
    template <typename, typename = void>                                      \
    struct is_##id##_0_comparable : std::false_type {};                       \
    template <typename T>                                                     \
    struct is_##id##_0_comparable<T,                                          \
                                  void_t<decltype( std::declval<T>() op 0 )>> \
        : std::true_type {};

        // We need all 6 pre-spaceship comparison ops: <, <=, >, >=, ==, !=
        CATCH_DEFINE_COMPARABLE_TRAIT( lt, < )
        CATCH_DEFINE_COMPARABLE_TRAIT( le, <= )
        CATCH_DEFINE_COMPARABLE_TRAIT( gt, > )
        CATCH_DEFINE_COMPARABLE_TRAIT( ge, >= )
        CATCH_DEFINE_COMPARABLE_TRAIT( eq, == )
        CATCH_DEFINE_COMPARABLE_TRAIT( ne, != )

#undef CATCH_DEFINE_COMPARABLE_TRAIT

#if defined( __GNUC__ ) && !defined( __clang__ )
#    pragma GCC diagnostic pop
#endif
#if defined( __clang__ )
#    pragma clang diagnostic pop
#endif


    } // namespace Detail
} // namespace Catch

#endif // CATCH_COMPARE_TRAITS_HPP_INCLUDED
