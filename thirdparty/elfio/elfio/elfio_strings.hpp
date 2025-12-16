/*
Copyright (C) 2001-present by Serge Lamikhov-Center

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef ELFIO_STRINGS_HPP
#define ELFIO_STRINGS_HPP

#include <cstdlib>
#include <cstring>
#include <string>
#include <limits>

namespace ELFIO {

//------------------------------------------------------------------------------
//! \class string_section_accessor_template
//! \brief Class for accessing string section data
template <class S> class string_section_accessor_template
{
  public:
    //------------------------------------------------------------------------------
    //! \brief Constructor
    //! \param section Pointer to the section
    explicit string_section_accessor_template( S* section )
        : string_section( section )
    {
    }

    //------------------------------------------------------------------------------
    //! \brief Get a string from the section
    //! \param index Index of the string
    //! \return Pointer to the string, or nullptr if not found
    const char* get_string( Elf_Word index ) const
    {
        if ( string_section ) {
            const char* data = string_section->get_data();
            size_t      section_size =
                static_cast<size_t>( string_section->get_size() );

            // Check if index is within bounds
            if ( index >= section_size || nullptr == data ) {
                return nullptr;
            }

            // Check for integer overflow in size calculation
            size_t remaining_size = section_size - index;
            if ( remaining_size > section_size ) { // Check for underflow
                return nullptr;
            }

            // Use standard C++ functions to find string length
            const char* str = data + index;
            const char* end =
                (const char*)std::memchr( str, '\0', remaining_size );
            if ( end != nullptr && end < str + remaining_size ) {
                return str;
            }
        }

        return nullptr;
    }

    //------------------------------------------------------------------------------
    //! \brief Add a string to the section
    //! \param str Pointer to the string
    //! \return Index of the added string
    Elf_Word add_string( const char* str )
    {
        if ( !str ) {
            return 0; // Return index of empty string for null input
        }

        Elf_Word current_position = 0;

        if ( string_section ) {
            // Strings are added to the end of the current section data
            current_position =
                static_cast<Elf_Word>( string_section->get_size() );

            if ( current_position == 0 ) {
                char empty_string = '\0';
                string_section->append_data( &empty_string, 1 );
                current_position++;
            }

            // Calculate string length and check for overflow
            size_t str_len = std::strlen( str );
            if ( str_len > std::numeric_limits<Elf_Word>::max() - 1 ) {
                return 0; // String too long
            }

            // Check if appending would overflow section size
            Elf_Word append_size = static_cast<Elf_Word>( str_len + 1 );
            if ( append_size >
                 std::numeric_limits<Elf_Word>::max() - current_position ) {
                return 0; // Would overflow section size
            }

            string_section->append_data( str, append_size );
        }

        return current_position;
    }

    //------------------------------------------------------------------------------
    //! \brief Add a string to the section
    //! \param str The string to add
    //! \return Index of the added string
    Elf_Word add_string( const std::string& str )
    {
        return add_string( str.c_str() );
    }

    //------------------------------------------------------------------------------
  private:
    S* string_section; //!< Pointer to the section
};

using string_section_accessor = string_section_accessor_template<section>;
using const_string_section_accessor =
    string_section_accessor_template<const section>;

} // namespace ELFIO

#endif // ELFIO_STRINGS_HPP
