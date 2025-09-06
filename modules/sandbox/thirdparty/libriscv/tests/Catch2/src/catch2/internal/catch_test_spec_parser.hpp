
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_TEST_SPEC_PARSER_HPP_INCLUDED
#define CATCH_TEST_SPEC_PARSER_HPP_INCLUDED

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
#endif

#include <catch2/catch_test_spec.hpp>

#include <vector>
#include <string>

namespace Catch {

    class ITagAliasRegistry;

    class TestSpecParser {
        enum Mode{ None, Name, QuotedName, Tag, EscapedName };
        Mode m_mode = None;
        Mode lastMode = None;
        bool m_exclusion = false;
        std::size_t m_pos = 0;
        std::size_t m_realPatternPos = 0;
        std::string m_arg;
        std::string m_substring;
        std::string m_patternName;
        std::vector<std::size_t> m_escapeChars;
        TestSpec::Filter m_currentFilter;
        TestSpec m_testSpec;
        ITagAliasRegistry const* m_tagAliases = nullptr;

    public:
        TestSpecParser( ITagAliasRegistry const& tagAliases );

        TestSpecParser& parse( std::string const& arg );
        TestSpec testSpec();

    private:
        bool visitChar( char c );
        void startNewMode( Mode mode );
        bool processNoneChar( char c );
        void processNameChar( char c );
        bool processOtherChar( char c );
        void endMode();
        void escape();
        bool isControlChar( char c ) const;
        void saveLastMode();
        void revertBackToLastMode();
        void addFilter();
        bool separate();

        // Handles common preprocessing of the pattern for name/tag patterns
        std::string preprocessPattern();
        // Adds the current pattern as a test name
        void addNamePattern();
        // Adds the current pattern as a tag
        void addTagPattern();

        inline void addCharToPattern(char c) {
            m_substring += c;
            m_patternName += c;
            m_realPatternPos++;
        }

    };

} // namespace Catch

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif // CATCH_TEST_SPEC_PARSER_HPP_INCLUDED
