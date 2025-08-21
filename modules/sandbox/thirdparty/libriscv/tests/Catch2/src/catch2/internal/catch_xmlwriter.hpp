
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_XMLWRITER_HPP_INCLUDED
#define CATCH_XMLWRITER_HPP_INCLUDED

#include <catch2/internal/catch_reusable_string_stream.hpp>
#include <catch2/internal/catch_stringref.hpp>

#include <iosfwd>
#include <vector>
#include <cstdint>

namespace Catch {
    enum class XmlFormatting : std::uint8_t {
        None = 0x00,
        Indent = 0x01,
        Newline = 0x02,
    };

    constexpr XmlFormatting operator|( XmlFormatting lhs, XmlFormatting rhs ) {
        return static_cast<XmlFormatting>( static_cast<std::uint8_t>( lhs ) |
                                           static_cast<std::uint8_t>( rhs ) );
    }

    constexpr XmlFormatting operator&( XmlFormatting lhs, XmlFormatting rhs ) {
        return static_cast<XmlFormatting>( static_cast<std::uint8_t>( lhs ) &
                                           static_cast<std::uint8_t>( rhs ) );
    }


    /**
     * Helper for XML-encoding text (escaping angle brackets, quotes, etc)
     *
     * Note: doesn't take ownership of passed strings, and thus the
     *       encoded string must outlive the encoding instance.
     */
    class XmlEncode {
    public:
        enum ForWhat { ForTextNodes, ForAttributes };

        constexpr XmlEncode( StringRef str, ForWhat forWhat = ForTextNodes ):
            m_str( str ), m_forWhat( forWhat ) {}


        void encodeTo( std::ostream& os ) const;

        friend std::ostream& operator << ( std::ostream& os, XmlEncode const& xmlEncode );

    private:
        StringRef m_str;
        ForWhat m_forWhat;
    };

    class XmlWriter {
    public:

        class ScopedElement {
        public:
            ScopedElement( XmlWriter* writer, XmlFormatting fmt );

            ScopedElement( ScopedElement&& other ) noexcept;
            ScopedElement& operator=( ScopedElement&& other ) noexcept;

            ~ScopedElement();

            ScopedElement&
            writeText( StringRef text,
                       XmlFormatting fmt = XmlFormatting::Newline |
                                           XmlFormatting::Indent );

            ScopedElement& writeAttribute( StringRef name,
                                           StringRef attribute );
            template <typename T,
                      // Without this SFINAE, this overload is a better match
                      // for `std::string`, `char const*`, `char const[N]` args.
                      // While it would still work, it would cause code bloat
                      // and multiple iteration over the strings
                      typename = typename std::enable_if_t<
                          !std::is_convertible<T, StringRef>::value>>
            ScopedElement& writeAttribute( StringRef name,
                                           T const& attribute ) {
                m_writer->writeAttribute( name, attribute );
                return *this;
            }

        private:
            XmlWriter* m_writer = nullptr;
            XmlFormatting m_fmt;
        };

        XmlWriter( std::ostream& os );
        ~XmlWriter();

        XmlWriter( XmlWriter const& ) = delete;
        XmlWriter& operator=( XmlWriter const& ) = delete;

        XmlWriter& startElement( std::string const& name, XmlFormatting fmt = XmlFormatting::Newline | XmlFormatting::Indent);

        ScopedElement scopedElement( std::string const& name, XmlFormatting fmt = XmlFormatting::Newline | XmlFormatting::Indent);

        XmlWriter& endElement(XmlFormatting fmt = XmlFormatting::Newline | XmlFormatting::Indent);

        //! The attribute content is XML-encoded
        XmlWriter& writeAttribute( StringRef name, StringRef attribute );

        //! Writes the attribute as "true/false"
        XmlWriter& writeAttribute( StringRef name, bool attribute );

        //! The attribute content is XML-encoded
        XmlWriter& writeAttribute( StringRef name, char const* attribute );

        //! The attribute value must provide op<<(ostream&, T). The resulting
        //! serialization is XML-encoded
        template <typename T,
                  // Without this SFINAE, this overload is a better match
                  // for `std::string`, `char const*`, `char const[N]` args.
                  // While it would still work, it would cause code bloat
                  // and multiple iteration over the strings
                  typename = typename std::enable_if_t<
                      !std::is_convertible<T, StringRef>::value>>
        XmlWriter& writeAttribute( StringRef name, T const& attribute ) {
            ReusableStringStream rss;
            rss << attribute;
            return writeAttribute( name, rss.str() );
        }

        //! Writes escaped `text` in a element
        XmlWriter& writeText( StringRef text,
                              XmlFormatting fmt = XmlFormatting::Newline |
                                                  XmlFormatting::Indent );

        //! Writes XML comment as "<!-- text -->"
        XmlWriter& writeComment( StringRef text,
                                 XmlFormatting fmt = XmlFormatting::Newline |
                                                     XmlFormatting::Indent );

        void writeStylesheetRef( StringRef url );

        void ensureTagClosed();

    private:

        void applyFormatting(XmlFormatting fmt);

        void writeDeclaration();

        void newlineIfNecessary();

        bool m_tagIsOpen = false;
        bool m_needsNewline = false;
        std::vector<std::string> m_tags;
        std::string m_indent;
        std::ostream& m_os;
    };

}

#endif // CATCH_XMLWRITER_HPP_INCLUDED
