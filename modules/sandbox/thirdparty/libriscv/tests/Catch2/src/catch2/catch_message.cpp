
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#include <catch2/catch_message.hpp>
#include <catch2/interfaces/catch_interfaces_capture.hpp>
#include <catch2/internal/catch_enforce.hpp>
#include <catch2/internal/catch_move_and_forward.hpp>

#include <cassert>
#include <stack>

namespace Catch {

    ////////////////////////////////////////////////////////////////////////////


    ScopedMessage::ScopedMessage( MessageBuilder&& builder ):
        m_info( CATCH_MOVE(builder.m_info) ) {
        m_info.message = builder.m_stream.str();
        getResultCapture().pushScopedMessage( m_info );
    }

    ScopedMessage::ScopedMessage( ScopedMessage&& old ) noexcept:
        m_info( CATCH_MOVE( old.m_info ) ) {
        old.m_moved = true;
    }

    ScopedMessage::~ScopedMessage() {
        if ( !m_moved ){
            getResultCapture().popScopedMessage(m_info);
        }
    }


    Capturer::Capturer( StringRef macroName,
                        SourceLineInfo const& lineInfo,
                        ResultWas::OfType resultType,
                        StringRef names ):
        m_resultCapture( getResultCapture() ) {
        auto trimmed = [&] (size_t start, size_t end) {
            while (names[start] == ',' || isspace(static_cast<unsigned char>(names[start]))) {
                ++start;
            }
            while (names[end] == ',' || isspace(static_cast<unsigned char>(names[end]))) {
                --end;
            }
            return names.substr(start, end - start + 1);
        };
        auto skipq = [&] (size_t start, char quote) {
            for (auto i = start + 1; i < names.size() ; ++i) {
                if (names[i] == quote)
                    return i;
                if (names[i] == '\\')
                    ++i;
            }
            CATCH_INTERNAL_ERROR("CAPTURE parsing encountered unmatched quote");
        };

        size_t start = 0;
        std::stack<char> openings;
        for (size_t pos = 0; pos < names.size(); ++pos) {
            char c = names[pos];
            switch (c) {
            case '[':
            case '{':
            case '(':
            // It is basically impossible to disambiguate between
            // comparison and start of template args in this context
//            case '<':
                openings.push(c);
                break;
            case ']':
            case '}':
            case ')':
//           case '>':
                openings.pop();
                break;
            case '"':
            case '\'':
                pos = skipq(pos, c);
                break;
            case ',':
                if (start != pos && openings.empty()) {
                    m_messages.emplace_back(macroName, lineInfo, resultType);
                    m_messages.back().message = static_cast<std::string>(trimmed(start, pos));
                    m_messages.back().message += " := ";
                    start = pos;
                }
                break;
            default:; // noop
            }
        }
        assert(openings.empty() && "Mismatched openings");
        m_messages.emplace_back(macroName, lineInfo, resultType);
        m_messages.back().message = static_cast<std::string>(trimmed(start, names.size() - 1));
        m_messages.back().message += " := ";
    }
    Capturer::~Capturer() {
        assert( m_captured == m_messages.size() );
        for ( size_t i = 0; i < m_captured; ++i ) {
            m_resultCapture.popScopedMessage( m_messages[i] );
        }
    }

    void Capturer::captureValue( size_t index, std::string const& value ) {
        assert( index < m_messages.size() );
        m_messages[index].message += value;
        m_resultCapture.pushScopedMessage( m_messages[index] );
        m_captured++;
    }

} // end namespace Catch
