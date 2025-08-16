
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_JSONWRITER_HPP_INCLUDED
#define CATCH_JSONWRITER_HPP_INCLUDED

#include <catch2/internal/catch_reusable_string_stream.hpp>
#include <catch2/internal/catch_stringref.hpp>

#include <cstdint>
#include <sstream>

namespace Catch {
    class JsonObjectWriter;
    class JsonArrayWriter;

    struct JsonUtils {
        static void indent( std::ostream& os, std::uint64_t level );
        static void appendCommaNewline( std::ostream& os,
                                        bool& should_comma,
                                        std::uint64_t level );
    };

    class JsonValueWriter {
    public:
        JsonValueWriter( std::ostream& os );
        JsonValueWriter( std::ostream& os, std::uint64_t indent_level );

        JsonObjectWriter writeObject() &&;
        JsonArrayWriter writeArray() &&;

        template <typename T>
        void write( T const& value ) && {
            writeImpl( value, !std::is_arithmetic<T>::value );
        }
        void write( StringRef value ) &&;
        void write( bool value ) &&;

    private:
        void writeImpl( StringRef value, bool quote );

        // Without this SFINAE, this overload is a better match
        // for `std::string`, `char const*`, `char const[N]` args.
        // While it would still work, it would cause code bloat
        // and multiple iteration over the strings
        template <typename T,
                  typename = typename std::enable_if_t<
                      !std::is_convertible<T, StringRef>::value>>
        void writeImpl( T const& value, bool quote_value ) {
            m_sstream << value;
            writeImpl( m_sstream.str(), quote_value );
        }

        std::ostream& m_os;
        std::stringstream m_sstream;
        std::uint64_t m_indent_level;
    };

    class JsonObjectWriter {
    public:
        JsonObjectWriter( std::ostream& os );
        JsonObjectWriter( std::ostream& os, std::uint64_t indent_level );

        JsonObjectWriter( JsonObjectWriter&& source ) noexcept;
        JsonObjectWriter& operator=( JsonObjectWriter&& source ) = delete;

        ~JsonObjectWriter();

        JsonValueWriter write( StringRef key );

    private:
        std::ostream& m_os;
        std::uint64_t m_indent_level;
        bool m_should_comma = false;
        bool m_active = true;
    };

    class JsonArrayWriter {
    public:
        JsonArrayWriter( std::ostream& os );
        JsonArrayWriter( std::ostream& os, std::uint64_t indent_level );

        JsonArrayWriter( JsonArrayWriter&& source ) noexcept;
        JsonArrayWriter& operator=( JsonArrayWriter&& source ) = delete;

        ~JsonArrayWriter();

        JsonObjectWriter writeObject();
        JsonArrayWriter writeArray();

        template <typename T>
        JsonArrayWriter& write( T const& value ) {
            return writeImpl( value );
        }

        JsonArrayWriter& write( bool value );

    private:
        template <typename T>
        JsonArrayWriter& writeImpl( T const& value ) {
            JsonUtils::appendCommaNewline(
                m_os, m_should_comma, m_indent_level + 1 );
            JsonValueWriter{ m_os }.write( value );

            return *this;
        }

        std::ostream& m_os;
        std::uint64_t m_indent_level;
        bool m_should_comma = false;
        bool m_active = true;
    };

} // namespace Catch

#endif // CATCH_JSONWRITER_HPP_INCLUDED
