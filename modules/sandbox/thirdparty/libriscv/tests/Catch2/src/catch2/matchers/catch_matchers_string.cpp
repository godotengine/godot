
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/internal/catch_string_manip.hpp>
#include <catch2/catch_tostring.hpp>
#include <catch2/internal/catch_move_and_forward.hpp>

#include <regex>

namespace Catch {
namespace Matchers {

    CasedString::CasedString( std::string const& str, CaseSensitive caseSensitivity )
    :   m_caseSensitivity( caseSensitivity ),
        m_str( adjustString( str ) )
    {}
    std::string CasedString::adjustString( std::string const& str ) const {
        return m_caseSensitivity == CaseSensitive::No
               ? toLower( str )
               : str;
    }
    StringRef CasedString::caseSensitivitySuffix() const {
        return m_caseSensitivity == CaseSensitive::Yes
                   ? StringRef()
                   : " (case insensitive)"_sr;
    }


    StringMatcherBase::StringMatcherBase( StringRef operation, CasedString const& comparator )
    : m_comparator( comparator ),
      m_operation( operation ) {
    }

    std::string StringMatcherBase::describe() const {
        std::string description;
        description.reserve(5 + m_operation.size() + m_comparator.m_str.size() +
                                    m_comparator.caseSensitivitySuffix().size());
        description += m_operation;
        description += ": \"";
        description += m_comparator.m_str;
        description += '"';
        description += m_comparator.caseSensitivitySuffix();
        return description;
    }

    StringEqualsMatcher::StringEqualsMatcher( CasedString const& comparator ) : StringMatcherBase( "equals"_sr, comparator ) {}

    bool StringEqualsMatcher::match( std::string const& source ) const {
        return m_comparator.adjustString( source ) == m_comparator.m_str;
    }


    StringContainsMatcher::StringContainsMatcher( CasedString const& comparator ) : StringMatcherBase( "contains"_sr, comparator ) {}

    bool StringContainsMatcher::match( std::string const& source ) const {
        return contains( m_comparator.adjustString( source ), m_comparator.m_str );
    }


    StartsWithMatcher::StartsWithMatcher( CasedString const& comparator ) : StringMatcherBase( "starts with"_sr, comparator ) {}

    bool StartsWithMatcher::match( std::string const& source ) const {
        return startsWith( m_comparator.adjustString( source ), m_comparator.m_str );
    }


    EndsWithMatcher::EndsWithMatcher( CasedString const& comparator ) : StringMatcherBase( "ends with"_sr, comparator ) {}

    bool EndsWithMatcher::match( std::string const& source ) const {
        return endsWith( m_comparator.adjustString( source ), m_comparator.m_str );
    }



    RegexMatcher::RegexMatcher(std::string regex, CaseSensitive caseSensitivity): m_regex(CATCH_MOVE(regex)), m_caseSensitivity(caseSensitivity) {}

    bool RegexMatcher::match(std::string const& matchee) const {
        auto flags = std::regex::ECMAScript; // ECMAScript is the default syntax option anyway
        if (m_caseSensitivity == CaseSensitive::No) {
            flags |= std::regex::icase;
        }
        auto reg = std::regex(m_regex, flags);
        return std::regex_match(matchee, reg);
    }

    std::string RegexMatcher::describe() const {
        return "matches " + ::Catch::Detail::stringify(m_regex) + ((m_caseSensitivity == CaseSensitive::Yes)? " case sensitively" : " case insensitively");
    }


    StringEqualsMatcher Equals( std::string const& str, CaseSensitive caseSensitivity ) {
        return StringEqualsMatcher( CasedString( str, caseSensitivity) );
    }
    StringContainsMatcher ContainsSubstring( std::string const& str, CaseSensitive caseSensitivity ) {
        return StringContainsMatcher( CasedString( str, caseSensitivity) );
    }
    EndsWithMatcher EndsWith( std::string const& str, CaseSensitive caseSensitivity ) {
        return EndsWithMatcher( CasedString( str, caseSensitivity) );
    }
    StartsWithMatcher StartsWith( std::string const& str, CaseSensitive caseSensitivity ) {
        return StartsWithMatcher( CasedString( str, caseSensitivity) );
    }

    RegexMatcher Matches(std::string const& regex, CaseSensitive caseSensitivity) {
        return RegexMatcher(regex, caseSensitivity);
    }

} // namespace Matchers
} // namespace Catch
