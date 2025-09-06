
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_LOGICAL_TRAITS_HPP_INCLUDED
#define CATCH_LOGICAL_TRAITS_HPP_INCLUDED

#include <type_traits>

namespace Catch {
namespace Detail {

#if defined( __cpp_lib_logical_traits ) && __cpp_lib_logical_traits >= 201510

    using std::conjunction;
    using std::disjunction;
    using std::negation;

#else

    template <class...> struct conjunction : std::true_type {};
    template <class B1> struct conjunction<B1> : B1 {};
    template <class B1, class... Bn>
    struct conjunction<B1, Bn...>
        : std::conditional_t<bool( B1::value ), conjunction<Bn...>, B1> {};

    template <class...> struct disjunction : std::false_type {};
    template <class B1> struct disjunction<B1> : B1 {};
    template <class B1, class... Bn>
    struct disjunction<B1, Bn...>
        : std::conditional_t<bool( B1::value ), B1, disjunction<Bn...>> {};

    template <class B>
    struct negation : std::integral_constant<bool, !bool(B::value)> {};

#endif

} // namespace Detail
} // namespace Catch

#endif // CATCH_LOGICAL_TRAITS_HPP_INCLUDED
