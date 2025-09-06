
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#include <catch2/catch_test_spec.hpp>
#include <catch2/interfaces/catch_interfaces_testcase.hpp>
#include <catch2/internal/catch_test_case_registry_impl.hpp>
#include <catch2/internal/catch_reusable_string_stream.hpp>
#include <catch2/internal/catch_string_manip.hpp>
#include <catch2/catch_test_case_info.hpp>

#include <algorithm>
#include <string>
#include <vector>
#include <ostream>

namespace Catch {

    TestSpec::Pattern::Pattern( std::string const& name )
    : m_name( name )
    {}

    TestSpec::Pattern::~Pattern() = default;

    std::string const& TestSpec::Pattern::name() const {
        return m_name;
    }


    TestSpec::NamePattern::NamePattern( std::string const& name, std::string const& filterString )
    : Pattern( filterString )
    , m_wildcardPattern( toLower( name ), CaseSensitive::No )
    {}

    bool TestSpec::NamePattern::matches( TestCaseInfo const& testCase ) const {
        return m_wildcardPattern.matches( testCase.name );
    }

    void TestSpec::NamePattern::serializeTo( std::ostream& out ) const {
        out << '"' << name() << '"';
    }


    TestSpec::TagPattern::TagPattern( std::string const& tag, std::string const& filterString )
    : Pattern( filterString )
    , m_tag( tag )
    {}

    bool TestSpec::TagPattern::matches( TestCaseInfo const& testCase ) const {
        return std::find( begin( testCase.tags ),
                          end( testCase.tags ),
                          Tag( m_tag ) ) != end( testCase.tags );
    }

    void TestSpec::TagPattern::serializeTo( std::ostream& out ) const {
        out << name();
    }

    bool TestSpec::Filter::matches( TestCaseInfo const& testCase ) const {
        bool should_use = !testCase.isHidden();
        for (auto const& pattern : m_required) {
            should_use = true;
            if (!pattern->matches(testCase)) {
                return false;
            }
        }
        for (auto const& pattern : m_forbidden) {
            if (pattern->matches(testCase)) {
                return false;
            }
        }
        return should_use;
    }

    void TestSpec::Filter::serializeTo( std::ostream& out ) const {
        bool first = true;
        for ( auto const& pattern : m_required ) {
            if ( !first ) {
                out << ' ';
            }
            out << *pattern;
            first = false;
        }
        for ( auto const& pattern : m_forbidden ) {
            if ( !first ) {
                out << ' ';
            }
            out << *pattern;
            first = false;
        }
    }


    std::string TestSpec::extractFilterName( Filter const& filter ) {
        Catch::ReusableStringStream sstr;
        sstr << filter;
        return sstr.str();
    }

    bool TestSpec::hasFilters() const {
        return !m_filters.empty();
    }

    bool TestSpec::matches( TestCaseInfo const& testCase ) const {
        return std::any_of( m_filters.begin(), m_filters.end(), [&]( Filter const& f ){ return f.matches( testCase ); } );
    }

    TestSpec::Matches TestSpec::matchesByFilter( std::vector<TestCaseHandle> const& testCases, IConfig const& config ) const {
        Matches matches;
        matches.reserve( m_filters.size() );
        for ( auto const& filter : m_filters ) {
            std::vector<TestCaseHandle const*> currentMatches;
            for ( auto const& test : testCases )
                if ( isThrowSafe( test, config ) &&
                     filter.matches( test.getTestCaseInfo() ) )
                    currentMatches.emplace_back( &test );
            matches.push_back(
                FilterMatch{ extractFilterName( filter ), currentMatches } );
        }
        return matches;
    }

    const TestSpec::vectorStrings& TestSpec::getInvalidSpecs() const {
        return m_invalidSpecs;
    }

    void TestSpec::serializeTo( std::ostream& out ) const {
        bool first = true;
        for ( auto const& filter : m_filters ) {
            if ( !first ) {
                out << ',';
            }
            out << filter;
            first = false;
        }
    }

}
