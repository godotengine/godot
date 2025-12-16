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

#ifndef ELFIO_UTILS_HPP
#define ELFIO_UTILS_HPP

#include <cstdint>
#include <ostream>
#include <cstring>

#define ELFIO_GET_ACCESS_DECL( TYPE, NAME ) virtual TYPE get_##NAME() const = 0

#define ELFIO_SET_ACCESS_DECL( TYPE, NAME ) \
    virtual void set_##NAME( const TYPE& value ) = 0

#define ELFIO_GET_SET_ACCESS_DECL( TYPE, NAME )       \
    virtual TYPE get_##NAME() const              = 0; \
    virtual void set_##NAME( const TYPE& value ) = 0

#define ELFIO_GET_ACCESS( TYPE, NAME, FIELD ) \
    TYPE get_##NAME() const override { return ( *convertor )( FIELD ); }

#define ELFIO_SET_ACCESS( TYPE, NAME, FIELD )     \
    void set_##NAME( const TYPE& value ) override \
    {                                             \
        FIELD = decltype( FIELD )( value );       \
        FIELD = ( *convertor )( FIELD );          \
    }
#define ELFIO_GET_SET_ACCESS( TYPE, NAME, FIELD )                        \
    TYPE get_##NAME() const override { return ( *convertor )( FIELD ); } \
    void set_##NAME( const TYPE& value ) override                        \
    {                                                                    \
        FIELD = decltype( FIELD )( value );                              \
        FIELD = ( *convertor )( FIELD );                                 \
    }

namespace ELFIO {

//------------------------------------------------------------------------------
//! \class endianness_convertor
//! \brief Class for converting endianness of data
class endianness_convertor
{
  public:
    //------------------------------------------------------------------------------
    //! \brief Setup the endianness convertor
    //! \param elf_file_encoding The encoding of the ELF file
    void setup( unsigned char elf_file_encoding )
    {
        need_conversion = ( elf_file_encoding != get_host_encoding() );
    }

    //------------------------------------------------------------------------------
    //! \brief Convert a 64-bit unsigned integer
    //! \param value The value to convert
    //! \return The converted value
    std::uint64_t operator()( std::uint64_t value ) const
    {
        if ( !need_conversion ) {
            return value;
        }
        value = ( ( value & 0x00000000000000FFuLL ) << 56 ) |
                ( ( value & 0x000000000000FF00uLL ) << 40 ) |
                ( ( value & 0x0000000000FF0000uLL ) << 24 ) |
                ( ( value & 0x00000000FF000000uLL ) << 8 ) |
                ( ( value & 0x000000FF00000000uLL ) >> 8 ) |
                ( ( value & 0x0000FF0000000000uLL ) >> 24 ) |
                ( ( value & 0x00FF000000000000uLL ) >> 40 ) |
                ( ( value & 0xFF00000000000000uLL ) >> 56 );

        return value;
    }

    //------------------------------------------------------------------------------
    //! \brief Convert a 64-bit signed integer
    //! \param value The value to convert
    //! \return The converted value
    std::int64_t operator()( std::int64_t value ) const
    {
        if ( !need_conversion ) {
            return value;
        }
        return ( std::int64_t )( *this )( (std::uint64_t)value );
    }

    //------------------------------------------------------------------------------
    //! \brief Convert a 32-bit unsigned integer
    //! \param value The value to convert
    //! \return The converted value
    std::uint32_t operator()( std::uint32_t value ) const
    {
        if ( !need_conversion ) {
            return value;
        }
        value =
            ( ( value & 0x000000FF ) << 24 ) | ( ( value & 0x0000FF00 ) << 8 ) |
            ( ( value & 0x00FF0000 ) >> 8 ) | ( ( value & 0xFF000000 ) >> 24 );

        return value;
    }

    //------------------------------------------------------------------------------
    //! \brief Convert a 32-bit signed integer
    //! \param value The value to convert
    //! \return The converted value
    std::int32_t operator()( std::int32_t value ) const
    {
        if ( !need_conversion ) {
            return value;
        }
        return ( std::int32_t )( *this )( (std::uint32_t)value );
    }

    //------------------------------------------------------------------------------
    //! \brief Convert a 16-bit unsigned integer
    //! \param value The value to convert
    //! \return The converted value
    std::uint16_t operator()( std::uint16_t value ) const
    {
        if ( !need_conversion ) {
            return value;
        }
        value = ( std::uint16_t )( ( value & 0x00FF ) << 8 ) |
                ( ( value & 0xFF00 ) >> 8 );

        return value;
    }

    //------------------------------------------------------------------------------
    //! \brief Convert a 16-bit signed integer
    //! \param value The value to convert
    //! \return The converted value
    std::int16_t operator()( std::int16_t value ) const
    {
        if ( !need_conversion ) {
            return value;
        }
        return ( std::int16_t )( *this )( (std::uint16_t)value );
    }

    //------------------------------------------------------------------------------
    //! \brief Convert an 8-bit signed integer
    //! \param value The value to convert
    //! \return The converted value
    std::int8_t operator()( std::int8_t value ) const { return value; }

    //------------------------------------------------------------------------------
    //! \brief Convert an 8-bit unsigned integer
    //! \param value The value to convert
    //! \return The converted value
    std::uint8_t operator()( std::uint8_t value ) const { return value; }

    //------------------------------------------------------------------------------
  private:
    //------------------------------------------------------------------------------
    //! \brief Get the host encoding
    //! \return The host encoding
    unsigned char get_host_encoding() const
    {
        static const int tmp = 1;
        if ( 1 == *reinterpret_cast<const char*>( &tmp ) ) {
            return ELFDATA2LSB;
        }
        else {
            return ELFDATA2MSB;
        }
    }

    //------------------------------------------------------------------------------
    bool need_conversion = false; //!< Flag indicating if conversion is needed
};

//------------------------------------------------------------------------------
//! \struct address_translation
//! \brief Structure for address translation
struct address_translation
{
    //------------------------------------------------------------------------------
    //! \brief Constructor
    //! \param start The start address
    //! \param size The size of the address range
    //! \param mapped_to The mapped address
    address_translation( std::uint64_t start,
                         std::uint64_t size,
                         std::uint64_t mapped_to )
        : start( start ), size( size ), mapped_to( mapped_to ){};
    std::streampos start;     //!< Start address
    std::streampos size;      //!< Size of the address range
    std::streampos mapped_to; //!< Mapped address
};

//------------------------------------------------------------------------------
//! \class address_translator
//! \brief Class for translating addresses
class address_translator
{
  public:
    //------------------------------------------------------------------------------
    //! \brief Set address translation
    //! \param addr_trans Vector of address translations
    void set_address_translation( std::vector<address_translation>& addr_trans )
    {
        addr_translations = addr_trans;

        std::sort( addr_translations.begin(), addr_translations.end(),
                   []( const address_translation& a,
                       const address_translation& b ) -> bool {
                       return a.start < b.start;
                   } );
    }

    //------------------------------------------------------------------------------
    //! \brief Translate an address
    //! \param value The address to translate
    //! \return The translated address
    std::streampos operator[]( std::streampos value ) const
    {
        if ( addr_translations.empty() ) {
            return value;
        }

        for ( auto& t : addr_translations ) {
            if ( ( t.start <= value ) && ( ( value - t.start ) < t.size ) ) {
                return value - t.start + t.mapped_to;
            }
        }

        return value;
    }

    //------------------------------------------------------------------------------
    //! \brief Check if the address translator is empty
    //! \return True if empty, false otherwise
    bool empty() const { return addr_translations.empty(); }

  private:
    std::vector<address_translation>
        addr_translations; //!< Vector of address translations
};

//------------------------------------------------------------------------------
//! \brief Calculate the ELF hash of a name
//! \param name The name to hash
//! \return The ELF hash
inline std::uint32_t elf_hash( const unsigned char* name )
{
    std::uint32_t h = 0;
    std::uint32_t g = 0;
    while ( *name != '\0' ) {
        h = ( h << 4 ) + *name++;
        g = h & 0xf0000000;
        if ( g != 0 )
            h ^= g >> 24;
        h &= ~g;
    }
    return h;
}

//------------------------------------------------------------------------------
//! \brief Calculate the GNU hash of a name
//! \param s The name to hash
//! \return The GNU hash
inline std::uint32_t elf_gnu_hash( const unsigned char* s )
{
    std::uint32_t h = 0x1505;
    for ( unsigned char c = *s; c != '\0'; c = *++s )
        h = ( h << 5 ) + h + c;
    return h;
}

//------------------------------------------------------------------------------
//! \brief Convert a value to a hexadecimal string
//! \param value The value to convert
//! \return The hexadecimal string
inline std::string to_hex_string( std::uint64_t value )
{
    std::string str;

    while ( value ) {
        if ( auto digit = value & 0xF; digit < 0xA ) {
            str = char( '0' + digit ) + str;
        }
        else {
            str = char( 'A' + digit - 0xA ) + str;
        }
        value >>= 4;
    }

    return "0x" + str;
}

//------------------------------------------------------------------------------
//! \brief Adjust the size of a stream
//! \param stream The stream to adjust
//! \param offset The offset to adjust to
inline void adjust_stream_size( std::ostream& stream, std::streamsize offset )
{
    stream.seekp( 0, std::ios_base::end );
    if ( stream.tellp() < offset ) {
        std::streamsize size = offset - stream.tellp();
        stream.write( std::string( size_t( size ), '\0' ).c_str(), size );
    }
    stream.seekp( offset );
}

//------------------------------------------------------------------------------
//! \brief Get the length of a string with a maximum length
//! \param s The string
//! \param n The maximum length
//! \return The length of the string
inline static size_t strnlength( const char* s, size_t n )
{
    auto found = (const char*)std::memchr( s, '\0', n );
    return found ? (size_t)( found - s ) : n;
}

//------------------------------------------------------------------------------
//! \class compression_interface
//! \brief Interface for compression and decompression
class compression_interface
{
  public:
    virtual ~compression_interface() = default;

    //------------------------------------------------------------------------------
    //! \brief Decompress a compressed section
    //! \param data The buffer of compressed data
    //! \param convertor Pointer to an endianness convertor instance
    //! \param compressed_size The size of the compressed data buffer
    //! \param uncompressed_size Reference to a variable to store the decompressed buffer size
    //! \return A smart pointer to the decompressed data
    virtual std::unique_ptr<char[]>
    inflate( const char*                                 data,
             std::shared_ptr<const endianness_convertor> convertor,
             Elf_Xword                                   compressed_size,
             Elf_Xword& uncompressed_size ) const = 0;

    //------------------------------------------------------------------------------
    //! \brief Compress a section
    //! \param data The buffer of uncompressed data
    //! \param convertor Pointer to an endianness convertor instance
    //! \param decompressed_size The size of the uncompressed data buffer
    //! \param compressed_size Reference to a variable to store the compressed buffer size
    //! \return A smart pointer to the compressed data
    virtual std::unique_ptr<char[]>
    deflate( const char*                                 data,
             std::shared_ptr<const endianness_convertor> convertor,
             Elf_Xword                                   decompressed_size,
             Elf_Xword& compressed_size ) const = 0;
};

} // namespace ELFIO

#endif // ELFIO_UTILS_HPP
