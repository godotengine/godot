
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
// Adapted from donated nonius code.

#ifndef CATCH_CHRONOMETER_HPP_INCLUDED
#define CATCH_CHRONOMETER_HPP_INCLUDED

#include <catch2/benchmark/catch_clock.hpp>
#include <catch2/benchmark/catch_optimizer.hpp>
#include <catch2/internal/catch_meta.hpp>
#include <catch2/internal/catch_move_and_forward.hpp>

namespace Catch {
    namespace Benchmark {
        namespace Detail {
            struct ChronometerConcept {
                virtual void start() = 0;
                virtual void finish() = 0;
                virtual ~ChronometerConcept(); // = default;

                ChronometerConcept() = default;
                ChronometerConcept(ChronometerConcept const&) = default;
                ChronometerConcept& operator=(ChronometerConcept const&) = default;
            };
            template <typename Clock>
            struct ChronometerModel final : public ChronometerConcept {
                void start() override { started = Clock::now(); }
                void finish() override { finished = Clock::now(); }

                IDuration elapsed() const {
                    return std::chrono::duration_cast<std::chrono::nanoseconds>(
                        finished - started );
                }

                TimePoint<Clock> started;
                TimePoint<Clock> finished;
            };
        } // namespace Detail

        struct Chronometer {
        public:
            template <typename Fun>
            void measure(Fun&& fun) { measure(CATCH_FORWARD(fun), is_callable<Fun(int)>()); }

            int runs() const { return repeats; }

            Chronometer(Detail::ChronometerConcept& meter, int repeats_)
                : impl(&meter)
                , repeats(repeats_) {}

        private:
            template <typename Fun>
            void measure(Fun&& fun, std::false_type) {
                measure([&fun](int) { return fun(); }, std::true_type());
            }

            template <typename Fun>
            void measure(Fun&& fun, std::true_type) {
                Detail::optimizer_barrier();
                impl->start();
                for (int i = 0; i < repeats; ++i) invoke_deoptimized(fun, i);
                impl->finish();
                Detail::optimizer_barrier();
            }

            Detail::ChronometerConcept* impl;
            int repeats;
        };
    } // namespace Benchmark
} // namespace Catch

#endif // CATCH_CHRONOMETER_HPP_INCLUDED
