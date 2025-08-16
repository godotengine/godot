
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
// Adapted from donated nonius code.

#ifndef CATCH_COMPLETE_INVOKE_HPP_INCLUDED
#define CATCH_COMPLETE_INVOKE_HPP_INCLUDED

#include <catch2/internal/catch_meta.hpp>
#include <catch2/internal/catch_move_and_forward.hpp>

namespace Catch {
    namespace Benchmark {
        namespace Detail {
            template <typename T>
            struct CompleteType { using type = T; };
            template <>
            struct CompleteType<void> { struct type {}; };

            template <typename T>
            using CompleteType_t = typename CompleteType<T>::type;

            template <typename Result>
            struct CompleteInvoker {
                template <typename Fun, typename... Args>
                static Result invoke(Fun&& fun, Args&&... args) {
                    return CATCH_FORWARD(fun)(CATCH_FORWARD(args)...);
                }
            };
            template <>
            struct CompleteInvoker<void> {
                template <typename Fun, typename... Args>
                static CompleteType_t<void> invoke(Fun&& fun, Args&&... args) {
                    CATCH_FORWARD(fun)(CATCH_FORWARD(args)...);
                    return {};
                }
            };

            // invoke and not return void :(
            template <typename Fun, typename... Args>
            CompleteType_t<FunctionReturnType<Fun, Args...>> complete_invoke(Fun&& fun, Args&&... args) {
                return CompleteInvoker<FunctionReturnType<Fun, Args...>>::invoke(CATCH_FORWARD(fun), CATCH_FORWARD(args)...);
            }

        } // namespace Detail

        template <typename Fun>
        Detail::CompleteType_t<FunctionReturnType<Fun>> user_code(Fun&& fun) {
            return Detail::complete_invoke(CATCH_FORWARD(fun));
        }
    } // namespace Benchmark
} // namespace Catch

#endif // CATCH_COMPLETE_INVOKE_HPP_INCLUDED
