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

#ifndef ELFIO_SEGMENT_HPP
#define ELFIO_SEGMENT_HPP

#include <iostream>
#include <vector>
#include <new>
#include <limits>

namespace ELFIO {

//------------------------------------------------------------------------------
//! \class segment
//! \brief Class for accessing segment data
class segment
{
    friend class elfio;

  public:
    virtual ~segment() = default;

    //------------------------------------------------------------------------------
    //! \brief Get the index of the segment
    //! \return Index of the segment
    ELFIO_GET_ACCESS_DECL( Elf_Half, index );
    //------------------------------------------------------------------------------
    //! \brief Get the type of the segment
    //! \return Type of the segment
    ELFIO_GET_SET_ACCESS_DECL( Elf_Word, type );
    //------------------------------------------------------------------------------
    //! \brief Get the flags of the segment
    //! \return Flags of the segment
    ELFIO_GET_SET_ACCESS_DECL( Elf_Word, flags );
    //------------------------------------------------------------------------------
    //! \brief Get the alignment of the segment
    //! \return Alignment of the segment
    ELFIO_GET_SET_ACCESS_DECL( Elf_Xword, align );
    //------------------------------------------------------------------------------
    //! \brief Get the virtual address of the segment
    //! \return Virtual address of the segment
    ELFIO_GET_SET_ACCESS_DECL( Elf64_Addr, virtual_address );
    //------------------------------------------------------------------------------
    //! \brief Get the physical address of the segment
    //! \return Physical address of the segment
    ELFIO_GET_SET_ACCESS_DECL( Elf64_Addr, physical_address );
    //------------------------------------------------------------------------------
    //! \brief Get the file size of the segment
    //! \return File size of the segment
    ELFIO_GET_SET_ACCESS_DECL( Elf_Xword, file_size );
    //------------------------------------------------------------------------------
    //! \brief Get the memory size of the segment
    //! \return Memory size of the segment
    ELFIO_GET_SET_ACCESS_DECL( Elf_Xword, memory_size );
    //------------------------------------------------------------------------------
    //! \brief Get the offset of the segment
    //! \return Offset of the segment
    ELFIO_GET_ACCESS_DECL( Elf64_Off, offset );

    //------------------------------------------------------------------------------
    //! \brief Get the data of the segment
    //! \return Pointer to the data
    virtual const char* get_data() const = 0;
    //------------------------------------------------------------------------------
    //! \brief Free the data of the segment
    virtual void free_data() const = 0;

    //------------------------------------------------------------------------------
    //! \brief Add a section to the segment
    //! \param psec Pointer to the section
    //! \param addr_align Alignment of the section
    //! \return Index of the added section
    virtual Elf_Half add_section( section* psec, Elf_Xword addr_align ) = 0;
    //------------------------------------------------------------------------------
    //! \brief Add a section index to the segment
    //! \param index Index of the section
    //! \param addr_align Alignment of the section
    //! \return Index of the added section
    virtual Elf_Half add_section_index( Elf_Half  index,
                                        Elf_Xword addr_align ) = 0;
    //------------------------------------------------------------------------------
    //! \brief Get the number of sections in the segment
    //! \return Number of sections in the segment
    virtual Elf_Half get_sections_num() const = 0;
    //------------------------------------------------------------------------------
    //! \brief Get the index of a section at a given position
    //! \param num Position of the section
    //! \return Index of the section
    virtual Elf_Half get_section_index_at( Elf_Half num ) const = 0;
    //------------------------------------------------------------------------------
    //! \brief Check if the offset is initialized
    //! \return True if the offset is initialized, false otherwise
    virtual bool is_offset_initialized() const = 0;

  protected:
    //------------------------------------------------------------------------------
    //! \brief Set the offset of the segment
    //! \param offset Offset of the segment
    ELFIO_SET_ACCESS_DECL( Elf64_Off, offset );
    //------------------------------------------------------------------------------
    //! \brief Set the index of the segment
    //! \param index Index of the segment
    ELFIO_SET_ACCESS_DECL( Elf_Half, index );

    //------------------------------------------------------------------------------
    //! \brief Get the sections of the segment
    //! \return Vector of section indices
    virtual const std::vector<Elf_Half>& get_sections() const = 0;

    //------------------------------------------------------------------------------
    //! \brief Load the segment from a stream
    //! \param stream Input stream
    //! \param header_offset Offset of the segment header
    //! \param is_lazy Whether to load the segment lazily
    //! \return True if successful, false otherwise
    virtual bool load( std::istream&  stream,
                       std::streampos header_offset,
                       bool           is_lazy ) = 0;
    //------------------------------------------------------------------------------
    //! \brief Save the segment to a stream
    //! \param stream Output stream
    //! \param header_offset Offset of the segment header
    //! \param data_offset Offset of the segment data
    virtual void save( std::ostream&  stream,
                       std::streampos header_offset,
                       std::streampos data_offset ) = 0;
};

//------------------------------------------------------------------------------
//! \class segment_impl
//! \brief Implementation of the segment class
template <class T> class segment_impl : public segment
{
  public:
    //------------------------------------------------------------------------------
    //! \brief Constructor
    //! \param convertor Pointer to the endianness convertor
    //! \param translator Pointer to the address translator
    segment_impl( std::shared_ptr<endianness_convertor> convertor,
                  std::shared_ptr<address_translator>   translator )
        : convertor( convertor ), translator( translator )
    {
    }

    //------------------------------------------------------------------------------
    // Section info functions
    ELFIO_GET_SET_ACCESS( Elf_Word, type, ph.p_type );
    ELFIO_GET_SET_ACCESS( Elf_Word, flags, ph.p_flags );
    ELFIO_GET_SET_ACCESS( Elf_Xword, align, ph.p_align );
    ELFIO_GET_SET_ACCESS( Elf64_Addr, virtual_address, ph.p_vaddr );
    ELFIO_GET_SET_ACCESS( Elf64_Addr, physical_address, ph.p_paddr );
    ELFIO_GET_SET_ACCESS( Elf_Xword, file_size, ph.p_filesz );
    ELFIO_GET_SET_ACCESS( Elf_Xword, memory_size, ph.p_memsz );
    ELFIO_GET_ACCESS( Elf64_Off, offset, ph.p_offset );

    //------------------------------------------------------------------------------
    //! \brief Get the index of the segment
    //! \return Index of the segment
    Elf_Half get_index() const override { return index; }

    //------------------------------------------------------------------------------
    //! \brief Get the data of the segment
    //! \return Pointer to the data
    const char* get_data() const override
    {
        if ( !is_loaded ) {
            load_data();
        }
        return data.get();
    }

    //------------------------------------------------------------------------------
    //! \brief Free the data of the segment
    void free_data() const override
    {
        if ( is_lazy ) {
            data.reset( nullptr );
            is_loaded = false;
        }
    }

    //------------------------------------------------------------------------------
    //! \brief Add a section index to the segment
    //! \param sec_index Index of the section
    //! \param addr_align Alignment of the section
    //! \return Index of the added section
    Elf_Half add_section_index( Elf_Half  sec_index,
                                Elf_Xword addr_align ) override
    {
        sections.emplace_back( sec_index );
        if ( addr_align > get_align() ) {
            set_align( addr_align );
        }

        return (Elf_Half)sections.size();
    }

    //------------------------------------------------------------------------------
    //! \brief Add a section to the segment
    //! \param psec Pointer to the section
    //! \param addr_align Alignment of the section
    //! \return Index of the added section
    Elf_Half add_section( section* psec, Elf_Xword addr_align ) override
    {
        return add_section_index( psec->get_index(), addr_align );
    }

    //------------------------------------------------------------------------------
    //! \brief Get the number of sections in the segment
    //! \return Number of sections in the segment
    Elf_Half get_sections_num() const override
    {
        return (Elf_Half)sections.size();
    }

    //------------------------------------------------------------------------------
    //! \brief Get the index of a section at a given position
    //! \param num Position of the section
    //! \return Index of the section
    Elf_Half get_section_index_at( Elf_Half num ) const override
    {
        if ( num < sections.size() ) {
            return sections[num];
        }

        return Elf_Half( -1 );
    }

    //------------------------------------------------------------------------------
  protected:
    //------------------------------------------------------------------------------
    //! \brief Set the offset of the segment
    //! \param value Offset of the segment
    void set_offset( const Elf64_Off& value ) override
    {
        ph.p_offset   = decltype( ph.p_offset )( value );
        ph.p_offset   = ( *convertor )( ph.p_offset );
        is_offset_set = true;
    }

    //------------------------------------------------------------------------------
    //! \brief Check if the offset is initialized
    //! \return True if the offset is initialized, false otherwise
    bool is_offset_initialized() const override { return is_offset_set; }

    //------------------------------------------------------------------------------
    //! \brief Get the sections of the segment
    //! \return Vector of section indices
    const std::vector<Elf_Half>& get_sections() const override
    {
        return sections;
    }

    //------------------------------------------------------------------------------
    //! \brief Set the index of the segment
    //! \param value Index of the segment
    void set_index( const Elf_Half& value ) override { index = value; }

    //------------------------------------------------------------------------------
    //! \brief Load the segment from a stream
    //! \param stream Input stream
    //! \param header_offset Offset of the segment header
    //! \param is_lazy_ Whether to load the segment lazily
    //! \return True if successful, false otherwise
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
        stream.read( reinterpret_cast<char*>( &ph ), sizeof( ph ) );

        is_offset_set = true;

        if ( !( is_lazy || is_loaded ) ) {
            return load_data();
        }

        return true;
    }

    //------------------------------------------------------------------------------
    //! \brief Load the data of the segment
    //! \return True if successful, false otherwise
    bool load_data() const
    {
        if ( PT_NULL == get_type() || 0 == get_file_size() ) {
            return true;
        }

        Elf_Xword p_offset = ( *translator )[( *convertor )( ph.p_offset )];
        Elf_Xword size     = get_file_size();

        // Check for integer overflow in offset calculation
        if ( p_offset > get_stream_size() ) {
            data = nullptr;
            return false;
        }

        // Check for integer overflow in size calculation
        if ( size > get_stream_size() ||
             size > ( get_stream_size() - p_offset ) ) {
            data = nullptr;
            return false;
        }

        // Check if size can be safely converted to size_t
        if ( size > std::numeric_limits<size_t>::max() - 1 ) {
            data = nullptr;
            return false;
        }

        data.reset( new ( std::nothrow ) char[(size_t)size + 1] );

        pstream->seekg( p_offset );
        if ( nullptr != data.get() && pstream->read( data.get(), size ) ) {
            data.get()[size] = 0;
        }
        else {
            data = nullptr;
            return false;
        }

        is_loaded = true;

        return true;
    }

    //------------------------------------------------------------------------------
    //! \brief Save the segment to a stream
    //! \param stream Output stream
    //! \param header_offset Offset of the segment header
    //! \param data_offset Offset of the segment data
    void save( std::ostream&  stream,
               std::streampos header_offset,
               std::streampos data_offset ) override
    {
        ph.p_offset = decltype( ph.p_offset )( data_offset );
        ph.p_offset = ( *convertor )( ph.p_offset );
        adjust_stream_size( stream, header_offset );
        stream.write( reinterpret_cast<const char*>( &ph ), sizeof( ph ) );
    }

    //------------------------------------------------------------------------------
    //! \brief Get the stream size
    //! \return Stream size
    size_t get_stream_size() const { return stream_size; }

    //------------------------------------------------------------------------------
    //! \brief Set the stream size
    //! \param value Stream size
    void set_stream_size( size_t value ) { stream_size = value; }

    //------------------------------------------------------------------------------
  private:
    mutable std::istream* pstream = nullptr;  //!< Pointer to the input stream
    T                     ph      = {};       //!< Segment header
    Elf_Half              index   = 0;        //!< Index of the segment
    mutable std::unique_ptr<char[]> data;     //!< Pointer to the segment data
    std::vector<Elf_Half>           sections; //!< Vector of section indices
    std::shared_ptr<endianness_convertor> convertor =
        nullptr; //!< Pointer to the endianness convertor
    std::shared_ptr<address_translator> translator =
        nullptr;                  //!< Pointer to the address translator
    size_t stream_size   = 0;     //!< Stream size
    bool   is_offset_set = false; //!< Flag indicating if the offset is set
    mutable bool is_lazy =
        false; //!< Flag indicating if the segment is loaded lazily
    mutable bool is_loaded =
        false; //!< Flag indicating if the segment is loaded
};

} // namespace ELFIO

#endif // ELFIO_SEGMENT_HPP
