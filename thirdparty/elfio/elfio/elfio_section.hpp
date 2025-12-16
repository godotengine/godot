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

#ifndef ELFIO_SECTION_HPP
#define ELFIO_SECTION_HPP

#include <string>
#include <iostream>
#include <new>
#include <limits>

namespace ELFIO {

/**
 * @class section
 * @brief Represents a section in an ELF file.
 */
class section
{
    friend class elfio;

  public:
    virtual ~section() = default;

    ELFIO_GET_ACCESS_DECL( Elf_Half, index );
    ELFIO_GET_SET_ACCESS_DECL( std::string, name );
    ELFIO_GET_SET_ACCESS_DECL( Elf_Word, type );
    ELFIO_GET_SET_ACCESS_DECL( Elf_Xword, flags );
    ELFIO_GET_SET_ACCESS_DECL( Elf_Word, info );
    ELFIO_GET_SET_ACCESS_DECL( Elf_Word, link );
    ELFIO_GET_SET_ACCESS_DECL( Elf_Xword, addr_align );
    ELFIO_GET_SET_ACCESS_DECL( Elf_Xword, entry_size );
    ELFIO_GET_SET_ACCESS_DECL( Elf64_Addr, address );
    ELFIO_GET_SET_ACCESS_DECL( Elf_Xword, size );
    ELFIO_GET_SET_ACCESS_DECL( Elf_Word, name_string_offset );
    ELFIO_GET_ACCESS_DECL( Elf64_Off, offset );

    /**
     * @brief Get the data of the section.
     * @return Pointer to the data.
     */
    virtual const char* get_data() const = 0;

    /**
     * @brief Free the data of the section.
     */
    virtual void free_data() const = 0;

    /**
     * @brief Set the data of the section.
     * @param raw_data Pointer to the raw data.
     * @param size Size of the data.
     */
    virtual void set_data( const char* raw_data, Elf_Xword size ) = 0;

    /**
     * @brief Set the data of the section.
     * @param data String containing the data.
     */
    virtual void set_data( const std::string& data ) = 0;

    /**
     * @brief Append data to the section.
     * @param raw_data Pointer to the raw data.
     * @param size Size of the data.
     */
    virtual void append_data( const char* raw_data, Elf_Xword size ) = 0;

    /**
     * @brief Append data to the section.
     * @param data String containing the data.
     */
    virtual void append_data( const std::string& data ) = 0;

    /**
     * @brief Insert data into the section at a specific position.
     * @param pos Position to insert the data.
     * @param raw_data Pointer to the raw data.
     * @param size Size of the data.
     */
    virtual void
    insert_data( Elf_Xword pos, const char* raw_data, Elf_Xword size ) = 0;

    /**
     * @brief Insert data into the section at a specific position.
     * @param pos Position to insert the data.
     * @param data String containing the data.
     */
    virtual void insert_data( Elf_Xword pos, const std::string& data ) = 0;

    /**
     * @brief Get the size of the stream.
     * @return Size of the stream.
     */
    virtual size_t get_stream_size() const = 0;

    /**
     * @brief Set the size of the stream.
     * @param value Size of the stream.
     */
    virtual void set_stream_size( size_t value ) = 0;

  protected:
    ELFIO_SET_ACCESS_DECL( Elf64_Off, offset );
    ELFIO_SET_ACCESS_DECL( Elf_Half, index );

    /**
     * @brief Load the section from a stream.
     * @param stream Input stream.
     * @param header_offset Offset of the header.
     * @param is_lazy Whether to load lazily.
     * @return True if successful, false otherwise.
     */
    virtual bool load( std::istream&  stream,
                       std::streampos header_offset,
                       bool           is_lazy ) = 0;

    /**
     * @brief Save the section to a stream.
     * @param stream Output stream.
     * @param header_offset Offset of the header.
     * @param data_offset Offset of the data.
     */
    virtual void save( std::ostream&  stream,
                       std::streampos header_offset,
                       std::streampos data_offset ) = 0;

    /**
     * @brief Check if the address is initialized.
     * @return True if initialized, false otherwise.
     */
    virtual bool is_address_initialized() const = 0;
};

/**
 * @class section_impl
 * @brief Implementation of the section class.
 * @tparam T Type of the section header.
 */
template <class T> class section_impl : public section
{
  public:
    /**
     * @brief Constructor.
     * @param convertor Pointer to the endianness convertor.
     * @param translator Pointer to the address translator.
     * @param compression Shared pointer to the compression interface.
     */
    section_impl( std::shared_ptr<endianness_convertor>  convertor,
                  std::shared_ptr<address_translator>    translator,
                  std::shared_ptr<compression_interface> compression )
        : convertor( convertor ), translator( translator ),
          compression( compression )
    {
    }

    // Section info functions
    ELFIO_GET_SET_ACCESS( Elf_Word, type, header.sh_type );
    ELFIO_GET_SET_ACCESS( Elf_Xword, flags, header.sh_flags );
    ELFIO_GET_SET_ACCESS( Elf_Xword, size, header.sh_size );
    ELFIO_GET_SET_ACCESS( Elf_Word, link, header.sh_link );
    ELFIO_GET_SET_ACCESS( Elf_Word, info, header.sh_info );
    ELFIO_GET_SET_ACCESS( Elf_Xword, addr_align, header.sh_addralign );
    ELFIO_GET_SET_ACCESS( Elf_Xword, entry_size, header.sh_entsize );
    ELFIO_GET_SET_ACCESS( Elf_Word, name_string_offset, header.sh_name );
    ELFIO_GET_ACCESS( Elf64_Addr, address, header.sh_addr );

    /**
     * @brief Get the index of the section.
     * @return Index of the section.
     */
    Elf_Half get_index() const override { return index; }

    /**
     * @brief Get the name of the section.
     * @return Name of the section.
     */
    std::string get_name() const override { return name; }

    /**
     * @brief Set the name of the section.
     * @param name_prm Name of the section.
     */
    void set_name( const std::string& name_prm ) override
    {
        this->name = name_prm;
    }

    /**
     * @brief Set the address of the section.
     * @param value Address of the section.
     */
    void set_address( const Elf64_Addr& value ) override
    {
        header.sh_addr = decltype( header.sh_addr )( value );
        header.sh_addr = ( *convertor )( header.sh_addr );
        is_address_set = true;
    }

    /**
     * @brief Check if the address is initialized.
     * @return True if initialized, false otherwise.
     */
    bool is_address_initialized() const override { return is_address_set; }

    /**
     * @brief Get the data of the section.
     * @return Pointer to the data.
     */
    const char* get_data() const override
    {
        // If data load failed, the stream is corrupt
        // When lazy loading, attempts to call get_data() on it after initial load are useless
        // When loading non-lazily, that load_data() will attempt to read data from
        // the stream specified on load() call, which might be freed by this point
        if ( !is_loaded && can_be_loaded ) {
            bool res = load_data();

            if ( !res ) {
                can_be_loaded = false;
            }
        }
        return data.get();
    }

    /**
     * @brief Free the data of the section.
     */
    void free_data() const override
    {
        if ( is_lazy ) {
            data.reset( nullptr );
            is_loaded = false;
        }
    }

    /**
     * @brief Set the data of the section.
     * @param raw_data Pointer to the raw data.
     * @param size Size of the data.
     */
    void set_data( const char* raw_data, Elf_Xword size ) override
    {
        if ( get_type() != SHT_NOBITS ) {
            data = std::unique_ptr<char[]>(
                new ( std::nothrow ) char[(size_t)size] );
            if ( nullptr != data.get() && nullptr != raw_data ) {
                data_size = size;
                std::copy( raw_data, raw_data + size, data.get() );
            }
            else {
                data_size = 0;
            }
        }

        set_size( data_size );
        if ( translator->empty() ) {
            set_stream_size( (size_t)data_size );
        }
    }

    /**
     * @brief Set the data of the section.
     * @param str_data String containing the data.
     */
    void set_data( const std::string& str_data ) override
    {
        return set_data( str_data.c_str(), (Elf_Word)str_data.size() );
    }

    /**
     * @brief Append data to the section.
     * @param raw_data Pointer to the raw data.
     * @param size Size of the data.
     */
    void append_data( const char* raw_data, Elf_Xword size ) override
    {
        insert_data( get_size(), raw_data, size );
    }

    /**
     * @brief Append data to the section.
     * @param str_data String containing the data.
     */
    void append_data( const std::string& str_data ) override
    {
        return append_data( str_data.c_str(), (Elf_Word)str_data.size() );
    }

    /**
     * @brief Insert data into the section at a specific position.
     * @param pos Position to insert the data.
     * @param raw_data Pointer to the raw data.
     * @param size Size of the data.
     */
    void
    insert_data( Elf_Xword pos, const char* raw_data, Elf_Xword size ) override
    {
        if ( get_type() != SHT_NOBITS ) {
            // Check for valid position
            if ( pos > get_size() ) {
                return; // Invalid position
            }

            // Check for integer overflow in size calculation
            Elf_Xword new_size = get_size();
            if ( size > std::numeric_limits<Elf_Xword>::max() - new_size ) {
                return; // Size would overflow
            }
            new_size += size;

            if ( new_size <= data_size ) {
                char* d = data.get();
                std::copy_backward( d + pos, d + get_size(),
                                    d + get_size() + size );
                std::copy( raw_data, raw_data + size, d + pos );
            }
            else {
                // Calculate new size with overflow check
                Elf_Xword new_data_size = data_size;
                if ( new_data_size >
                     std::numeric_limits<Elf_Xword>::max() / 2 ) {
                    return; // Multiplication would overflow
                }
                new_data_size *= 2;
                if ( size >
                     std::numeric_limits<Elf_Xword>::max() - new_data_size ) {
                    return; // Addition would overflow
                }
                new_data_size += size;

                // Check if the size would overflow size_t
                if ( new_data_size > std::numeric_limits<size_t>::max() ) {
                    return; // Size would overflow size_t
                }

                std::unique_ptr<char[]> new_data(
                    new ( std::nothrow ) char[(size_t)new_data_size] );

                if ( nullptr != new_data ) {
                    char* d = data.get();
                    std::copy( d, d + pos, new_data.get() );
                    std::copy( raw_data, raw_data + size,
                               new_data.get() + pos );
                    std::copy( d + pos, d + get_size(),
                               new_data.get() + pos + size );
                    data      = std::move( new_data );
                    data_size = new_data_size;
                }
                else {
                    return; // Allocation failed
                }
            }
            set_size( new_size );
            if ( translator->empty() ) {
                set_stream_size( get_stream_size() + (size_t)size );
            }
        }
    }

    /**
     * @brief Insert data into the section at a specific position.
     * @param pos Position to insert the data.
     * @param str_data String containing the data.
     */
    void insert_data( Elf_Xword pos, const std::string& str_data ) override
    {
        return insert_data( pos, str_data.c_str(), (Elf_Word)str_data.size() );
    }

    /**
     * @brief Get the size of the stream.
     * @return Size of the stream.
     */
    size_t get_stream_size() const override { return stream_size; }

    /**
     * @brief Set the size of the stream.
     * @param value Size of the stream.
     */
    void set_stream_size( size_t value ) override { stream_size = value; }

  protected:
    ELFIO_GET_SET_ACCESS( Elf64_Off, offset, header.sh_offset );

    /**
     * @brief Set the index of the section.
     * @param value Index of the section.
     */
    void set_index( const Elf_Half& value ) override { index = value; }

    /**
     * @brief Check if the section is compressed.
     * @return True if compressed, false otherwise.
     */
    bool is_compressed() const
    {
        return ( ( get_flags() & SHF_RPX_DEFLATE ) ||
                 ( get_flags() & SHF_COMPRESSED ) ) &&
               compression != nullptr;
    }

    /**
     * @brief Load the section from a stream.
     * @param stream Input stream.
     * @param header_offset Offset of the header.
     * @param is_lazy_ Whether to load lazily.
     * @return True if successful, false otherwise.
     */
    bool load( std::istream&  stream,
               std::streampos header_offset,
               bool           is_lazy_ ) override
    {
        pstream = &stream;
        is_lazy = is_lazy_;

        if ( translator->empty() ) {
            stream.seekg( 0, std::istream::end );
            set_stream_size( size_t( stream.tellg() ) );
        }
        else {
            set_stream_size( std::numeric_limits<size_t>::max() );
        }

        stream.seekg( ( *translator )[header_offset] );
        stream.read( reinterpret_cast<char*>( &header ), sizeof( header ) );

        if ( !( is_lazy || is_loaded ) ) {
            bool ret = get_data();

            if ( is_compressed() ) {
                Elf_Xword size              = get_size();
                Elf_Xword uncompressed_size = 0;
                auto      decompressed_data = compression->inflate(
                    data.get(), convertor, size, uncompressed_size );
                if ( decompressed_data != nullptr ) {
                    set_size( uncompressed_size );
                    data = std::move( decompressed_data );
                }
            }

            return ret;
        }

        return true;
    }

    /**
     * @brief Load the data of the section.
     * @return True if successful, false otherwise.
     */
    bool load_data() const
    {
        Elf_Xword sh_offset =
            ( *translator )[( *convertor )( header.sh_offset )];
        Elf_Xword size = get_size();

        // Check for integer overflow in offset calculation
        if ( sh_offset > get_stream_size() ) {
            return false;
        }

        // Check for integer overflow in size calculation
        if ( size > get_stream_size() ||
             size > ( get_stream_size() - sh_offset ) ) {
            return false;
        }

        // Check if we need to load data
        if ( nullptr == data && SHT_NULL != get_type() &&
             SHT_NOBITS != get_type() ) {
            // Check if size can be safely converted to size_t
            if ( size > std::numeric_limits<size_t>::max() - 1 ) {
                return false;
            }

            data.reset( new ( std::nothrow ) char[size_t( size ) + 1] );

            if ( ( 0 != size ) && ( nullptr != data ) ) {
                pstream->seekg( sh_offset );
                pstream->read( data.get(), size );
                if ( static_cast<Elf_Xword>( pstream->gcount() ) != size ) {
                    data.reset( nullptr );
                    data_size = 0;
                    return false;
                }

                data_size        = size;
                data.get()[size] = 0; // Safe now as we allocated size + 1
            }
            else {
                data_size = 0;
                if ( size != 0 ) {
                    return false; // Failed to allocate required memory
                }
            }

            is_loaded = true;
            return true;
        }

        // Data already loaded or doesn't need loading
        is_loaded = ( nullptr != data ) || ( SHT_NULL == get_type() ) ||
                    ( SHT_NOBITS == get_type() );
        return is_loaded;
    }

    /**
     * @brief Save the section to a stream.
     * @param stream Output stream.
     * @param header_offset Offset of the header.
     * @param data_offset Offset of the data.
     */
    void save( std::ostream&  stream,
               std::streampos header_offset,
               std::streampos data_offset ) override
    {
        if ( 0 != get_index() ) {
            header.sh_offset = decltype( header.sh_offset )( data_offset );
            header.sh_offset = ( *convertor )( header.sh_offset );
        }

        save_header( stream, header_offset );
        if ( get_type() != SHT_NOBITS && get_type() != SHT_NULL &&
             get_size() != 0 && data != nullptr ) {
            save_data( stream, data_offset );
        }
    }

  private:
    /**
     * @brief Save the header of the section to a stream.
     * @param stream Output stream.
     * @param header_offset Offset of the header.
     */
    void save_header( std::ostream& stream, std::streampos header_offset ) const
    {
        adjust_stream_size( stream, header_offset );
        stream.write( reinterpret_cast<const char*>( &header ),
                      sizeof( header ) );
    }

    /**
     * @brief Save the data of the section to a stream.
     * @param stream Output stream.
     * @param data_offset Offset of the data.
     */
    void save_data( std::ostream& stream, std::streampos data_offset )
    {
        adjust_stream_size( stream, data_offset );

        if ( ( ( get_flags() & SHF_COMPRESSED ) ||
               ( get_flags() & SHF_RPX_DEFLATE ) ) &&
             compression != nullptr ) {
            Elf_Xword decompressed_size = get_size();
            Elf_Xword compressed_size   = 0;
            auto      compressed_ptr    = compression->deflate(
                data.get(), convertor, decompressed_size, compressed_size );
            stream.write( compressed_ptr.get(), compressed_size );
        }
        else {
            stream.write( get_data(), get_size() );
        }
    }

  private:
    mutable std::istream* pstream =
        nullptr; /**< Pointer to the input stream. */
    T                               header = {};   /**< Section header. */
    Elf_Half                        index  = 0;    /**< Index of the section. */
    std::string                     name;          /**< Name of the section. */
    mutable std::unique_ptr<char[]> data;          /**< Pointer to the data. */
    mutable Elf_Xword               data_size = 0; /**< Size of the data. */
    std::shared_ptr<endianness_convertor> convertor =
        nullptr; /**< Pointer to the endianness convertor. */
    std::shared_ptr<address_translator> translator =
        nullptr; /**< Pointer to the address translator. */
    std::shared_ptr<compression_interface> compression =
        nullptr; /**< Shared pointer to the compression interface. */
    bool is_address_set = false;  /**< Flag indicating if the address is set. */
    size_t       stream_size = 0; /**< Size of the stream. */
    mutable bool is_lazy =
        false; /**< Flag indicating if lazy loading is enabled. */
    mutable bool is_loaded =
        false; /**< Flag indicating if the data is loaded. */
    mutable bool can_be_loaded =
        true; /**< Flag indicating if the data can loaded. This is not the case if the section is corrupted. */
};

} // namespace ELFIO

#endif // ELFIO_SECTION_HPP
