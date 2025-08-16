
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_META_HPP_INCLUDED
#define CATCH_META_HPP_INCLUDED

#include <type_traits>

namespace Catch {
    template <typename>
    struct true_given : std::true_type {};

    struct is_callable_tester {
        template <typename Fun, typename... Args>
        static true_given<decltype(std::declval<Fun>()(std::declval<Args>()...))> test(int);
        template <typename...>
        static std::false_type test(...);
    };

    template <typename T>
    struct is_callable;

    template <typename Fun, typename... Args>
    struct is_callable<Fun(Args...)> : decltype(is_callable_tester::test<Fun, Args...>(0)) {};


#if defined(__cpp_lib_is_invocable) && __cpp_lib_is_invocable >= 201703
    // std::result_of is deprecated in C++17 and removed in C++20. Hence, it is
    // replaced with std::invoke_result here.
    template <typename Func, typename... U>
    using FunctionReturnType = std::remove_reference_t<std::remove_cv_t<std::invoke_result_t<Func, U...>>>;
#else
    template <typename Func, typename... U>
    using FunctionReturnType = std::remove_reference_t<std::remove_cv_t<std::result_of_t<Func(U...)>>>;
#endif

} // namespace Catch

namespace mpl_{
    struct na;
}

#endif // CATCH_META_HPP_INCLUDED
