
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_TEXTFLOW_HPP_INCLUDED
#define CATCH_TEXTFLOW_HPP_INCLUDED

#include <catch2/internal/catch_console_width.hpp>
#include <catch2/internal/catch_move_and_forward.hpp>

#include <cassert>
#include <string>
#include <vector>

namespace Catch {
    namespace TextFlow {

        class Columns;

        /**
         * Abstraction for a string with ansi escape sequences that
         * automatically skips over escapes when iterating. Only graphical
         * escape sequences are considered.
         *
         * Internal representation:
         * An escape sequence looks like \033[39;49m
         * We need bidirectional iteration and the unbound length of escape
         * sequences poses a problem for operator-- To make this work we'll
         * replace the last `m` with a 0xff (this is a codepoint that won't have
         * any utf-8 meaning).
         */
        class AnsiSkippingString {
            std::string m_string;
            std::size_t m_size = 0;

            // perform 0xff replacement and calculate m_size
            void preprocessString();

        public:
            class const_iterator;
            using iterator = const_iterator;
            // note: must be u-suffixed or this will cause a "truncation of
            // constant value" warning on MSVC
            static constexpr char sentinel = static_cast<char>( 0xffu );

            explicit AnsiSkippingString( std::string const& text );
            explicit AnsiSkippingString( std::string&& text );

            const_iterator begin() const;
            const_iterator end() const;

            size_t size() const { return m_size; }

            std::string substring( const_iterator begin,
                                   const_iterator end ) const;
        };

        class AnsiSkippingString::const_iterator {
            friend AnsiSkippingString;
            struct EndTag {};

            const std::string* m_string;
            std::string::const_iterator m_it;

            explicit const_iterator( const std::string& string, EndTag ):
                m_string( &string ), m_it( string.end() ) {}

            void tryParseAnsiEscapes();
            void advance();
            void unadvance();

        public:
            using difference_type = std::ptrdiff_t;
            using value_type = char;
            using pointer = value_type*;
            using reference = value_type&;
            using iterator_category = std::bidirectional_iterator_tag;

            explicit const_iterator( const std::string& string ):
                m_string( &string ), m_it( string.begin() ) {
                tryParseAnsiEscapes();
            }

            char operator*() const { return *m_it; }

            const_iterator& operator++() {
                advance();
                return *this;
            }
            const_iterator operator++( int ) {
                iterator prev( *this );
                operator++();
                return prev;
            }
            const_iterator& operator--() {
                unadvance();
                return *this;
            }
            const_iterator operator--( int ) {
                iterator prev( *this );
                operator--();
                return prev;
            }

            bool operator==( const_iterator const& other ) const {
                return m_it == other.m_it;
            }
            bool operator!=( const_iterator const& other ) const {
                return !operator==( other );
            }
            bool operator<=( const_iterator const& other ) const {
                return m_it <= other.m_it;
            }

            const_iterator oneBefore() const {
                auto it = *this;
                return --it;
            }
        };

        /**
         * Represents a column of text with specific width and indentation
         *
         * When written out to a stream, it will perform linebreaking
         * of the provided text so that the written lines fit within
         * target width.
         */
        class Column {
            // String to be written out
            AnsiSkippingString m_string;
            // Width of the column for linebreaking
            size_t m_width = CATCH_CONFIG_CONSOLE_WIDTH - 1;
            // Indentation of other lines (including first if initial indent is
            // unset)
            size_t m_indent = 0;
            // Indentation of the first line
            size_t m_initialIndent = std::string::npos;

        public:
            /**
             * Iterates "lines" in `Column` and returns them
             */
            class const_iterator {
                friend Column;
                struct EndTag {};

                Column const& m_column;
                // Where does the current line start?
                AnsiSkippingString::const_iterator m_lineStart;
                // How long should the current line be?
                AnsiSkippingString::const_iterator m_lineEnd;
                // How far have we checked the string to iterate?
                AnsiSkippingString::const_iterator m_parsedTo;
                // Should a '-' be appended to the line?
                bool m_addHyphen = false;

                const_iterator( Column const& column, EndTag ):
                    m_column( column ),
                    m_lineStart( m_column.m_string.end() ),
                    m_lineEnd( column.m_string.end() ),
                    m_parsedTo( column.m_string.end() ) {}

                // Calculates the length of the current line
                void calcLength();

                // Returns current indentation width
                size_t indentSize() const;

                // Creates an indented and (optionally) suffixed string from
                // current iterator position, indentation and length.
                std::string addIndentAndSuffix(
                    AnsiSkippingString::const_iterator start,
                    AnsiSkippingString::const_iterator end ) const;

            public:
                using difference_type = std::ptrdiff_t;
                using value_type = std::string;
                using pointer = value_type*;
                using reference = value_type&;
                using iterator_category = std::forward_iterator_tag;

                explicit const_iterator( Column const& column );

                std::string operator*() const;

                const_iterator& operator++();
                const_iterator operator++( int );

                bool operator==( const_iterator const& other ) const {
                    return m_lineStart == other.m_lineStart &&
                           &m_column == &other.m_column;
                }
                bool operator!=( const_iterator const& other ) const {
                    return !operator==( other );
                }
            };
            using iterator = const_iterator;

            explicit Column( std::string const& text ): m_string( text ) {}
            explicit Column( std::string&& text ):
                m_string( CATCH_MOVE( text ) ) {}

            Column& width( size_t newWidth ) & {
                assert( newWidth > 0 );
                m_width = newWidth;
                return *this;
            }
            Column&& width( size_t newWidth ) && {
                assert( newWidth > 0 );
                m_width = newWidth;
                return CATCH_MOVE( *this );
            }
            Column& indent( size_t newIndent ) & {
                m_indent = newIndent;
                return *this;
            }
            Column&& indent( size_t newIndent ) && {
                m_indent = newIndent;
                return CATCH_MOVE( *this );
            }
            Column& initialIndent( size_t newIndent ) & {
                m_initialIndent = newIndent;
                return *this;
            }
            Column&& initialIndent( size_t newIndent ) && {
                m_initialIndent = newIndent;
                return CATCH_MOVE( *this );
            }

            size_t width() const { return m_width; }
            const_iterator begin() const { return const_iterator( *this ); }
            const_iterator end() const {
                return { *this, const_iterator::EndTag{} };
            }

            friend std::ostream& operator<<( std::ostream& os,
                                             Column const& col );

            friend Columns operator+( Column const& lhs, Column const& rhs );
            friend Columns operator+( Column&& lhs, Column&& rhs );
        };

        //! Creates a column that serves as an empty space of specific width
        Column Spacer( size_t spaceWidth );

        class Columns {
            std::vector<Column> m_columns;

        public:
            class iterator {
                friend Columns;
                struct EndTag {};

                std::vector<Column> const& m_columns;
                std::vector<Column::const_iterator> m_iterators;
                size_t m_activeIterators;

                iterator( Columns const& columns, EndTag );

            public:
                using difference_type = std::ptrdiff_t;
                using value_type = std::string;
                using pointer = value_type*;
                using reference = value_type&;
                using iterator_category = std::forward_iterator_tag;

                explicit iterator( Columns const& columns );

                auto operator==( iterator const& other ) const -> bool {
                    return m_iterators == other.m_iterators;
                }
                auto operator!=( iterator const& other ) const -> bool {
                    return m_iterators != other.m_iterators;
                }
                std::string operator*() const;
                iterator& operator++();
                iterator operator++( int );
            };
            using const_iterator = iterator;

            iterator begin() const { return iterator( *this ); }
            iterator end() const { return { *this, iterator::EndTag() }; }

            friend Columns& operator+=( Columns& lhs, Column const& rhs );
            friend Columns& operator+=( Columns& lhs, Column&& rhs );
            friend Columns operator+( Columns const& lhs, Column const& rhs );
            friend Columns operator+( Columns&& lhs, Column&& rhs );

            friend std::ostream& operator<<( std::ostream& os,
                                             Columns const& cols );
        };

    } // namespace TextFlow
} // namespace Catch
#endif // CATCH_TEXTFLOW_HPP_INCLUDED
