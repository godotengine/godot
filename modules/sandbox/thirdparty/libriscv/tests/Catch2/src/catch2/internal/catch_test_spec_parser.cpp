
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#include <catch2/internal/catch_test_spec_parser.hpp>

#include <catch2/internal/catch_string_manip.hpp>
#include <catch2/interfaces/catch_interfaces_tag_alias_registry.hpp>
#include <catch2/internal/catch_move_and_forward.hpp>


namespace Catch {

    TestSpecParser::TestSpecParser( ITagAliasRegistry const& tagAliases ) : m_tagAliases( &tagAliases ) {}

    TestSpecParser& TestSpecParser::parse( std::string const& arg ) {
        m_mode = None;
        m_exclusion = false;
        m_arg = m_tagAliases->expandAliases( arg );
        m_escapeChars.clear();
        m_substring.reserve(m_arg.size());
        m_patternName.reserve(m_arg.size());
        m_realPatternPos = 0;

        for( m_pos = 0; m_pos < m_arg.size(); ++m_pos )
          //if visitChar fails
           if( !visitChar( m_arg[m_pos] ) ){
               m_testSpec.m_invalidSpecs.push_back(arg);
               break;
           }
        endMode();
        return *this;
    }
    TestSpec TestSpecParser::testSpec() {
        addFilter();
        return CATCH_MOVE(m_testSpec);
    }
    bool TestSpecParser::visitChar( char c ) {
        if( (m_mode != EscapedName) && (c == '\\') ) {
            escape();
            addCharToPattern(c);
            return true;
        }else if((m_mode != EscapedName) && (c == ',') )  {
            return separate();
        }

        switch( m_mode ) {
        case None:
            if( processNoneChar( c ) )
                return true;
            break;
        case Name:
            processNameChar( c );
            break;
        case EscapedName:
            endMode();
            addCharToPattern(c);
            return true;
        default:
        case Tag:
        case QuotedName:
            if( processOtherChar( c ) )
                return true;
            break;
        }

        m_substring += c;
        if( !isControlChar( c ) ) {
            m_patternName += c;
            m_realPatternPos++;
        }
        return true;
    }
    // Two of the processing methods return true to signal the caller to return
    // without adding the given character to the current pattern strings
    bool TestSpecParser::processNoneChar( char c ) {
        switch( c ) {
        case ' ':
            return true;
        case '~':
            m_exclusion = true;
            return false;
        case '[':
            startNewMode( Tag );
            return false;
        case '"':
            startNewMode( QuotedName );
            return false;
        default:
            startNewMode( Name );
            return false;
        }
    }
    void TestSpecParser::processNameChar( char c ) {
        if( c == '[' ) {
            if( m_substring == "exclude:" )
                m_exclusion = true;
            else
                endMode();
            startNewMode( Tag );
        }
    }
    bool TestSpecParser::processOtherChar( char c ) {
        if( !isControlChar( c ) )
            return false;
        m_substring += c;
        endMode();
        return true;
    }
    void TestSpecParser::startNewMode( Mode mode ) {
        m_mode = mode;
    }
    void TestSpecParser::endMode() {
        switch( m_mode ) {
        case Name:
        case QuotedName:
            return addNamePattern();
        case Tag:
            return addTagPattern();
        case EscapedName:
            revertBackToLastMode();
            return;
        case None:
        default:
            return startNewMode( None );
        }
    }
    void TestSpecParser::escape() {
        saveLastMode();
        m_mode = EscapedName;
        m_escapeChars.push_back(m_realPatternPos);
    }
    bool TestSpecParser::isControlChar( char c ) const {
        switch( m_mode ) {
            default:
                return false;
            case None:
                return c == '~';
            case Name:
                return c == '[';
            case EscapedName:
                return true;
            case QuotedName:
                return c == '"';
            case Tag:
                return c == '[' || c == ']';
        }
    }

    void TestSpecParser::addFilter() {
        if( !m_currentFilter.m_required.empty() || !m_currentFilter.m_forbidden.empty() ) {
            m_testSpec.m_filters.push_back( CATCH_MOVE(m_currentFilter) );
            m_currentFilter = TestSpec::Filter();
        }
    }

    void TestSpecParser::saveLastMode() {
      lastMode = m_mode;
    }

    void TestSpecParser::revertBackToLastMode() {
      m_mode = lastMode;
    }

    bool TestSpecParser::separate() {
      if( (m_mode==QuotedName) || (m_mode==Tag) ){
         //invalid argument, signal failure to previous scope.
         m_mode = None;
         m_pos = m_arg.size();
         m_substring.clear();
         m_patternName.clear();
         m_realPatternPos = 0;
         return false;
      }
      endMode();
      addFilter();
      return true; //success
    }

    std::string TestSpecParser::preprocessPattern() {
        std::string token = m_patternName;
        for (std::size_t i = 0; i < m_escapeChars.size(); ++i)
            token = token.substr(0, m_escapeChars[i] - i) + token.substr(m_escapeChars[i] - i + 1);
        m_escapeChars.clear();
        if (startsWith(token, "exclude:")) {
            m_exclusion = true;
            token = token.substr(8);
        }

        m_patternName.clear();
        m_realPatternPos = 0;

        return token;
    }

    void TestSpecParser::addNamePattern() {
        auto token = preprocessPattern();

        if (!token.empty()) {
            if (m_exclusion) {
                m_currentFilter.m_forbidden.emplace_back(Detail::make_unique<TestSpec::NamePattern>(token, m_substring));
            } else {
                m_currentFilter.m_required.emplace_back(Detail::make_unique<TestSpec::NamePattern>(token, m_substring));
            }
        }
        m_substring.clear();
        m_exclusion = false;
        m_mode = None;
    }

    void TestSpecParser::addTagPattern() {
        auto token = preprocessPattern();

        if (!token.empty()) {
            // If the tag pattern is the "hide and tag" shorthand (e.g. [.foo])
            // we have to create a separate hide tag and shorten the real one
            if (token.size() > 1 && token[0] == '.') {
                token.erase(token.begin());
                if (m_exclusion) {
                    m_currentFilter.m_forbidden.emplace_back(Detail::make_unique<TestSpec::TagPattern>(".", m_substring));
                } else {
                    m_currentFilter.m_required.emplace_back(Detail::make_unique<TestSpec::TagPattern>(".", m_substring));
                }
            }
            if (m_exclusion) {
                m_currentFilter.m_forbidden.emplace_back(Detail::make_unique<TestSpec::TagPattern>(token, m_substring));
            } else {
                m_currentFilter.m_required.emplace_back(Detail::make_unique<TestSpec::TagPattern>(token, m_substring));
            }
        }
        m_substring.clear();
        m_exclusion = false;
        m_mode = None;
    }

} // namespace Catch
