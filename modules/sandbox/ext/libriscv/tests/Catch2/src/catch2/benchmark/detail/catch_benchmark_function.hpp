
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
// Adapted from donated nonius code.

#ifndef CATCH_BENCHMARK_FUNCTION_HPP_INCLUDED
#define CATCH_BENCHMARK_FUNCTION_HPP_INCLUDED

#include <catch2/benchmark/catch_chronometer.hpp>
#include <catch2/internal/catch_meta.hpp>
#include <catch2/internal/catch_unique_ptr.hpp>
#include <catch2/internal/catch_move_and_forward.hpp>

#include <type_traits>

namespace Catch {
    namespace Benchmark {
        namespace Detail {
            template <typename T, typename U>
            static constexpr bool is_related_v = std::is_same<std::decay_t<T>, std::decay_t<U>>::value;

            /// We need to reinvent std::function because every piece of code that might add overhead
            /// in a measurement context needs to have consistent performance characteristics so that we
            /// can account for it in the measurement.
            /// Implementations of std::function with optimizations that aren't always applicable, like
            /// small buffer optimizations, are not uncommon.
            /// This is effectively an implementation of std::function without any such optimizations;
            /// it may be slow, but it is consistently slow.
            struct BenchmarkFunction {
            private:
                struct callable {
                    virtual void call(Chronometer meter) const = 0;
                    virtual ~callable(); // = default;

                    callable() = default;
                    callable(callable&&) = default;
                    callable& operator=(callable&&) = default;
                };
                template <typename Fun>
                struct model : public callable {
                    model(Fun&& fun_) : fun(CATCH_MOVE(fun_)) {}
                    model(Fun const& fun_) : fun(fun_) {}

                    void call(Chronometer meter) const override {
                        call(meter, is_callable<Fun(Chronometer)>());
                    }
                    void call(Chronometer meter, std::true_type) const {
                        fun(meter);
                    }
                    void call(Chronometer meter, std::false_type) const {
                        meter.measure(fun);
                    }

                    Fun fun;
                };

            public:
                BenchmarkFunction();

                template <typename Fun,
                    std::enable_if_t<!is_related_v<Fun, BenchmarkFunction>, int> = 0>
                    BenchmarkFunction(Fun&& fun)
                    : f(new model<std::decay_t<Fun>>(CATCH_FORWARD(fun))) {}

                BenchmarkFunction( BenchmarkFunction&& that ) noexcept:
                    f( CATCH_MOVE( that.f ) ) {}

                BenchmarkFunction&
                operator=( BenchmarkFunction&& that ) noexcept {
                    f = CATCH_MOVE( that.f );
                    return *this;
                }

                void operator()(Chronometer meter) const { f->call(meter); }

            private:
                Catch::Detail::unique_ptr<callable> f;
            };
        } // namespace Detail
    } // namespace Benchmark
} // namespace Catch

#endif // CATCH_BENCHMARK_FUNCTION_HPP_INCLUDED
