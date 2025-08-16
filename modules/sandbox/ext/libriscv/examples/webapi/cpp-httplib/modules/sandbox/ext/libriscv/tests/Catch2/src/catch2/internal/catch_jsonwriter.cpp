
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#include <catch2/internal/catch_enforce.hpp>
#include <catch2/internal/catch_jsonwriter.hpp>

namespace Catch {
    void JsonUtils::indent( std::ostream& os, std::uint64_t level ) {
        for ( std::uint64_t i = 0; i < level; ++i ) {
            os << "  ";
        }
    }
    void JsonUtils::appendCommaNewline( std::ostream& os,
                                        bool& should_comma,
                                        std::uint64_t level ) {
        if ( should_comma ) { os << ','; }
        should_comma = true;
        os << '\n';
        indent( os, level );
    }

    JsonObjectWriter::JsonObjectWriter( std::ostream& os ):
        JsonObjectWriter{ os, 0 } {}

    JsonObjectWriter::JsonObjectWriter( std::ostream& os,
                                        std::uint64_t indent_level ):
        m_os{ os }, m_indent_level{ indent_level } {
        m_os << '{';
    }
    JsonObjectWriter::JsonObjectWriter( JsonObjectWriter&& source ) noexcept:
        m_os{ source.m_os },
        m_indent_level{ source.m_indent_level },
        m_should_comma{ source.m_should_comma },
        m_active{ source.m_active } {
        source.m_active = false;
    }

    JsonObjectWriter::~JsonObjectWriter() {
        if ( !m_active ) { return; }

        m_os << '\n';
        JsonUtils::indent( m_os, m_indent_level );
        m_os << '}';
    }

    JsonValueWriter JsonObjectWriter::write( StringRef key ) {
        JsonUtils::appendCommaNewline(
            m_os, m_should_comma, m_indent_level + 1 );

        m_os << '"' << key << "\": ";
        return JsonValueWriter{ m_os, m_indent_level + 1 };
    }

    JsonArrayWriter::JsonArrayWriter( std::ostream& os ):
        JsonArrayWriter{ os, 0 } {}
    JsonArrayWriter::JsonArrayWriter( std::ostream& os,
                                      std::uint64_t indent_level ):
        m_os{ os }, m_indent_level{ indent_level } {
        m_os << '[';
    }
    JsonArrayWriter::JsonArrayWriter( JsonArrayWriter&& source ) noexcept:
        m_os{ source.m_os },
        m_indent_level{ source.m_indent_level },
        m_should_comma{ source.m_should_comma },
        m_active{ source.m_active } {
        source.m_active = false;
    }
    JsonArrayWriter::~JsonArrayWriter() {
        if ( !m_active ) { return; }

        m_os << '\n';
        JsonUtils::indent( m_os, m_indent_level );
        m_os << ']';
    }

    JsonObjectWriter JsonArrayWriter::writeObject() {
        JsonUtils::appendCommaNewline(
            m_os, m_should_comma, m_indent_level + 1 );
        return JsonObjectWriter{ m_os, m_indent_level + 1 };
    }

    JsonArrayWriter JsonArrayWriter::writeArray() {
        JsonUtils::appendCommaNewline(
            m_os, m_should_comma, m_indent_level + 1 );
        return JsonArrayWriter{ m_os, m_indent_level + 1 };
    }

    JsonArrayWriter& JsonArrayWriter::write( bool value ) {
        return writeImpl( value );
    }

    JsonValueWriter::JsonValueWriter( std::ostream& os ):
        JsonValueWriter{ os, 0 } {}

    JsonValueWriter::JsonValueWriter( std::ostream& os,
                                      std::uint64_t indent_level ):
        m_os{ os }, m_indent_level{ indent_level } {}

    JsonObjectWriter JsonValueWriter::writeObject() && {
        return JsonObjectWriter{ m_os, m_indent_level };
    }

    JsonArrayWriter JsonValueWriter::writeArray() && {
        return JsonArrayWriter{ m_os, m_indent_level };
    }

    void JsonValueWriter::write( Catch::StringRef value ) && {
        writeImpl( value, true );
    }

    void JsonValueWriter::write( bool value ) && {
        writeImpl( value ? "true"_sr : "false"_sr, false );
    }

    void JsonValueWriter::writeImpl( Catch::StringRef value, bool quote ) {
        if ( quote ) { m_os << '"'; }
        for (char c : value) {
            // Escape list taken from https://www.json.org/json-en.html,
            // string definition.
            // Note that while forward slash _can_ be escaped, it does
            // not have to be, if JSON is not further embedded somewhere
            // where forward slash is meaningful.
            if ( c == '"' ) {
                m_os << "\\\"";
            } else if ( c == '\\' ) {
                m_os << "\\\\";
            } else if ( c == '\b' ) {
                m_os << "\\b";
            } else if ( c == '\f' ) {
                m_os << "\\f";
            } else if ( c == '\n' ) {
                m_os << "\\n";
            } else if ( c == '\r' ) {
                m_os << "\\r";
            } else if ( c == '\t' ) {
                m_os << "\\t";
            } else {
                m_os << c;
            }
        }
        if ( quote ) { m_os << '"'; }
    }

} // namespace Catch
