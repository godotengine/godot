
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_MATCHERS_PREDICATE_HPP_INCLUDED
#define CATCH_MATCHERS_PREDICATE_HPP_INCLUDED

#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/internal/catch_meta.hpp>
#include <catch2/internal/catch_move_and_forward.hpp>

#include <string>

namespace Catch {
namespace Matchers {

namespace Detail {
    std::string finalizeDescription(const std::string& desc);
} // namespace Detail

template <typename T, typename Predicate>
class PredicateMatcher final : public MatcherBase<T> {
    Predicate m_predicate;
    std::string m_description;
public:

    PredicateMatcher(Predicate&& elem, std::string const& descr)
        :m_predicate(CATCH_FORWARD(elem)),
        m_description(Detail::finalizeDescription(descr))
    {}

    bool match( T const& item ) const override {
        return m_predicate(item);
    }

    std::string describe() const override {
        return m_description;
    }
};

    /**
     * Creates a matcher that calls delegates `match` to the provided predicate.
     *
     * The user has to explicitly specify the argument type to the matcher
     */
    template<typename T, typename Pred>
    PredicateMatcher<T, Pred> Predicate(Pred&& predicate, std::string const& description = "") {
        static_assert(is_callable<Pred(T)>::value, "Predicate not callable with argument T");
        static_assert(std::is_same<bool, FunctionReturnType<Pred, T>>::value, "Predicate does not return bool");
        return PredicateMatcher<T, Pred>(CATCH_FORWARD(predicate), description);
    }

} // namespace Matchers
} // namespace Catch

#endif // CATCH_MATCHERS_PREDICATE_HPP_INCLUDED
