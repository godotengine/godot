
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_MATCHERS_EXCEPTION_HPP_INCLUDED
#define CATCH_MATCHERS_EXCEPTION_HPP_INCLUDED

#include <catch2/matchers/catch_matchers.hpp>

namespace Catch {
namespace Matchers {

class ExceptionMessageMatcher final : public MatcherBase<std::exception> {
    std::string m_message;
public:

    ExceptionMessageMatcher(std::string const& message):
        m_message(message)
    {}

    bool match(std::exception const& ex) const override;

    std::string describe() const override;
};

//! Creates a matcher that checks whether a std derived exception has the provided message
ExceptionMessageMatcher Message(std::string const& message);

template <typename StringMatcherType>
class ExceptionMessageMatchesMatcher final
    : public MatcherBase<std::exception> {
    StringMatcherType m_matcher;

public:
    ExceptionMessageMatchesMatcher( StringMatcherType matcher ):
        m_matcher( CATCH_MOVE( matcher ) ) {}

    bool match( std::exception const& ex ) const override {
        return m_matcher.match( ex.what() );
    }

    std::string describe() const override {
        return " matches \"" + m_matcher.describe() + '"';
    }
};

//! Creates a matcher that checks whether a message from an std derived
//! exception matches a provided matcher
template <typename StringMatcherType>
ExceptionMessageMatchesMatcher<StringMatcherType>
MessageMatches( StringMatcherType&& matcher ) {
    return { CATCH_FORWARD( matcher ) };
}

} // namespace Matchers
} // namespace Catch

#endif // CATCH_MATCHERS_EXCEPTION_HPP_INCLUDED
