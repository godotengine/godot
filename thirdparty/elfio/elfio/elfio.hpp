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

#ifndef ELFIO_HPP
#define ELFIO_HPP

#include <string>
#include <iostream>
#include <fstream>
#include <functional>
#include <algorithm>
#include <array>
#include <vector>
#include <deque>
#include <memory>

#include <elfio/elf_types.hpp>
#include <elfio/elfio_version.hpp>
#include <elfio/elfio_utils.hpp>
#include <elfio/elfio_header.hpp>
#include <elfio/elfio_section.hpp>
#include <elfio/elfio_segment.hpp>
#include <elfio/elfio_strings.hpp>

#define ELFIO_HEADER_ACCESS_GET( TYPE, FNAME ) \
    TYPE get_##FNAME() const { return header ? ( header->get_##FNAME() ) : 0; }

#define ELFIO_HEADER_ACCESS_GET_SET( TYPE, FNAME )     \
    TYPE get_##FNAME() const                           \
    {                                                  \
        return header ? ( header->get_##FNAME() ) : 0; \
    }                                                  \
    void set_##FNAME( TYPE val )                       \
    {                                                  \
        if ( header ) {                                \
            header->set_##FNAME( val );                \
        }                                              \
    }

namespace ELFIO {

//------------------------------------------------------------------------------
//! \class elfio
//! \brief The elfio class represents an ELF file and provides methods to manipulate it.
class elfio
{
  public:
    //------------------------------------------------------------------------------
    //! \brief Default constructor
    elfio() noexcept : sections( this ), segments( this )
    {
        convertor       = std::make_shared<endianness_convertor>();
        addr_translator = std::make_shared<address_translator>();
        create( ELFCLASS32, ELFDATA2LSB );
    }

    //------------------------------------------------------------------------------
    //! \brief Constructor with compression interface
    //! \param compression Pointer to the compression interface
    explicit elfio( compression_interface* compression_ptr ) noexcept : elfio()
    {
        this->compression =
            std::shared_ptr<compression_interface>( compression_ptr );
    }

    //------------------------------------------------------------------------------
    //! \brief Move constructor
    //! \param other The other elfio object to move from
    elfio( elfio&& other ) noexcept
        : sections( this ), segments( this ),
          current_file_pos( other.current_file_pos )
    {
        header          = std::move( other.header );
        sections_       = std::move( other.sections_ );
        segments_       = std::move( other.segments_ );
        convertor       = std::move( other.convertor );
        addr_translator = std::move( other.addr_translator );
        compression     = std::move( other.compression );

        other.header = nullptr;
        other.sections_.clear();
        other.segments_.clear();
        other.compression = nullptr;
    }

    //------------------------------------------------------------------------------
    //! \brief Move assignment operator
    //! \param other The other elfio object to move from
    //! \return Reference to this object
    elfio& operator=( elfio&& other ) noexcept
    {
        if ( this != &other ) {
            header           = std::move( other.header );
            sections_        = std::move( other.sections_ );
            segments_        = std::move( other.segments_ );
            convertor        = std::move( other.convertor );
            addr_translator  = std::move( other.addr_translator );
            current_file_pos = other.current_file_pos;
            compression      = std::move( other.compression );

            other.current_file_pos = 0;
            other.header           = nullptr;
            other.compression      = nullptr;
            other.sections_.clear();
            other.segments_.clear();
        }
        return *this;
    }

    //------------------------------------------------------------------------------
    //! \brief Delete copy constructor and copy assignment operator
    elfio( const elfio& )            = delete;
    elfio& operator=( const elfio& ) = delete;
    ~elfio()                         = default;

    //------------------------------------------------------------------------------
    //! \brief Create a new ELF file with the specified class and encoding
    //! \param file_class The class of the ELF file (ELFCLASS32 or ELFCLASS64)
    //! \param encoding The encoding of the ELF file (ELFDATA2LSB or ELFDATA2MSB)
    void create( unsigned char file_class, unsigned char encoding )
    {
        sections_.clear();
        segments_.clear();
        ( *convertor ).setup( encoding );
        header = create_header( file_class, encoding );
        create_mandatory_sections();
    }

    //------------------------------------------------------------------------------
    //! \brief Set address translation
    //! \param addr_trans Vector of address translations
    void set_address_translation( std::vector<address_translation>& addr_trans )
    {
        ( *addr_translator ).set_address_translation( addr_trans );
    }

    //------------------------------------------------------------------------------
    //! \brief Load an ELF file from a file
    //! \param file_name The name of the file to load
    //! \param is_lazy Whether to load the file lazily
    //! \return True if successful, false otherwise
    bool load( const std::string& file_name, bool is_lazy = false )
    {
        pstream = std::make_unique<std::ifstream>();
        if ( !pstream ) {
            return false;
        }

        pstream->open( file_name.c_str(), std::ios::in | std::ios::binary );
        if ( !*pstream ) {
            return false;
        }

        bool ret = load( *pstream, is_lazy );

        if ( !is_lazy ) {
            pstream.reset();
        }

        return ret;
    }

    //------------------------------------------------------------------------------
    //! \brief Load an ELF file from a stream
    //! \param stream The input stream to load from
    //! \param is_lazy Whether to load the file lazily
    //! \return True if successful, false otherwise
    bool load( std::istream& stream, bool is_lazy = false )
    {
        sections_.clear();
        segments_.clear();

        std::array<char, EI_NIDENT> e_ident = { 0 };
        // Read ELF file signature
        stream.seekg( ( *addr_translator )[0] );
        stream.read( e_ident.data(), sizeof( e_ident ) );

        // Is it ELF file?
        if ( stream.gcount() != sizeof( e_ident ) ||
             e_ident[EI_MAG0] != ELFMAG0 || e_ident[EI_MAG1] != ELFMAG1 ||
             e_ident[EI_MAG2] != ELFMAG2 || e_ident[EI_MAG3] != ELFMAG3 ) {
            return false;
        }

        if ( ( e_ident[EI_CLASS] != ELFCLASS64 ) &&
             ( e_ident[EI_CLASS] != ELFCLASS32 ) ) {
            return false;
        }

        if ( ( e_ident[EI_DATA] != ELFDATA2LSB ) &&
             ( e_ident[EI_DATA] != ELFDATA2MSB ) ) {
            return false;
        }

        ( *convertor ).setup( e_ident[EI_DATA] );
        header = create_header( e_ident[EI_CLASS], e_ident[EI_DATA] );
        if ( nullptr == header ) {
            return false;
        }
        if ( !header->load( stream ) ) {
            return false;
        }

        load_sections( stream, is_lazy );
        bool is_still_good = load_segments( stream, is_lazy );
        return is_still_good;
    }

    //------------------------------------------------------------------------------
    //! \brief Save the ELF file to a file
    //! \param file_name The name of the file to save to
    //! \return True if successful, false otherwise
    bool save( const std::string& file_name )
    {
        std::ofstream stream;
        stream.open( file_name.c_str(), std::ios::out | std::ios::binary );
        if ( !stream ) {
            return false;
        }

        return save( stream );
    }

    //------------------------------------------------------------------------------
    //! \brief Save the ELF file to a stream
    //! \param stream The output stream to save to
    //! \return True if successful, false otherwise
    bool save( std::ostream& stream )
    {
        if ( !stream || header == nullptr ) {
            return false;
        }

        // Define layout specific header fields
        // The position of the segment table is fixed after the header.
        // The position of the section table is variable and needs to be fixed
        // before saving.
        header->set_segments_num( segments.size() );
        header->set_segments_offset(
            segments.size() > 0 ? header->get_header_size() : 0 );
        header->set_sections_num( sections.size() );
        header->set_sections_offset( 0 );

        // Layout the first section right after the segment table
        current_file_pos =
            header->get_header_size() +
            header->get_segment_entry_size() *
                static_cast<Elf_Xword>( header->get_segments_num() );

        calc_segment_alignment();

        bool is_still_good = layout_segments_and_their_sections();
        is_still_good = is_still_good && layout_sections_without_segments();
        is_still_good = is_still_good && layout_section_table();

        is_still_good = is_still_good && save_header( stream );
        is_still_good = is_still_good && save_sections( stream );
        is_still_good = is_still_good && save_segments( stream );

        return is_still_good;
    }

    //------------------------------------------------------------------------------
    // ELF header access functions
    ELFIO_HEADER_ACCESS_GET( unsigned char, class );
    ELFIO_HEADER_ACCESS_GET( unsigned char, elf_version );
    ELFIO_HEADER_ACCESS_GET( unsigned char, encoding );
    ELFIO_HEADER_ACCESS_GET( Elf_Word, version );
    ELFIO_HEADER_ACCESS_GET( Elf_Half, header_size );
    ELFIO_HEADER_ACCESS_GET( Elf_Half, section_entry_size );
    ELFIO_HEADER_ACCESS_GET( Elf_Half, segment_entry_size );

    ELFIO_HEADER_ACCESS_GET_SET( unsigned char, os_abi );
    ELFIO_HEADER_ACCESS_GET_SET( unsigned char, abi_version );
    ELFIO_HEADER_ACCESS_GET_SET( Elf_Half, type );
    ELFIO_HEADER_ACCESS_GET_SET( Elf_Half, machine );
    ELFIO_HEADER_ACCESS_GET_SET( Elf_Word, flags );
    ELFIO_HEADER_ACCESS_GET_SET( Elf64_Addr, entry );
    ELFIO_HEADER_ACCESS_GET_SET( Elf64_Off, sections_offset );
    ELFIO_HEADER_ACCESS_GET_SET( Elf64_Off, segments_offset );
    ELFIO_HEADER_ACCESS_GET_SET( Elf_Half, section_name_str_index );

    //------------------------------------------------------------------------------
    //! \brief Get the endianness convertor
    //! \return Reference to the endianness convertor
    const std::shared_ptr<endianness_convertor>& get_convertor() const
    {
        return convertor;
    }

    //------------------------------------------------------------------------------
    //! \brief Get the default entry size for a section type
    //! \param section_type The type of the section
    //! \return The default entry size for the section type
    Elf_Xword get_default_entry_size( Elf_Word section_type ) const
    {
        switch ( section_type ) {
        case SHT_RELA:
            if ( header->get_class() == ELFCLASS64 ) {
                return sizeof( Elf64_Rela );
            }
            else {
                return sizeof( Elf32_Rela );
            }
        case SHT_REL:
            if ( header->get_class() == ELFCLASS64 ) {
                return sizeof( Elf64_Rel );
            }
            else {
                return sizeof( Elf32_Rel );
            }
        case SHT_SYMTAB:
            if ( header->get_class() == ELFCLASS64 ) {
                return sizeof( Elf64_Sym );
            }
            else {
                return sizeof( Elf32_Sym );
            }
        case SHT_DYNAMIC:
            if ( header->get_class() == ELFCLASS64 ) {
                return sizeof( Elf64_Dyn );
            }
            else {
                return sizeof( Elf32_Dyn );
            }
        default:
            return 0;
        }
    }

    //------------------------------------------------------------------------------
    //! \brief Validate the ELF file
    //! \return An empty string if no problems are detected, or a string containing an error message if problems are found, with one error per line.
    std::string validate() const
    {
        // clang-format off

        std::string errors;
        // Check for overlapping sections in the file
        // This is explicitly forbidden by ELF specification
        for ( int i = 0; i < sections.size(); ++i) {
            for ( int j = i+1; j < sections.size(); ++j ) {
                const section* a = sections[i];
                const section* b = sections[j];
                if (   ( ( a->get_type() & SHT_NOBITS) == 0 )
                    && ( ( b->get_type() & SHT_NOBITS) == 0 )
                      && ( a->get_size() > 0 )
                      && ( b->get_size() > 0 )
                      && ( a->get_offset() > 0 )
                      && ( b->get_offset() > 0 )
                      && ( is_offset_in_section( a->get_offset(), b )
                        || is_offset_in_section( a->get_offset()+a->get_size()-1, b )
                        || is_offset_in_section( b->get_offset(), a )
                        || is_offset_in_section( b->get_offset()+b->get_size()-1, a ) ) ) {
                            errors += "Sections " + a->get_name() + " and " + b->get_name() + " overlap in file\n";
                }
            }
        }
        // clang-format on

        // Check for conflicting section / program header tables, where
        // the same offset has different vaddresses in section table and
        // program header table.
        // This doesn't seem to be  explicitly forbidden by ELF specification,
        // but:
        // - it doesn't make any sense
        // - ELFIO relies on this being consistent when writing ELF files,
        //   since offsets are re-calculated from vaddress
        for ( int h = 0; h < segments.size(); ++h ) {
            const segment* seg = segments[h];
            const section* sec =
                find_prog_section_for_offset( seg->get_offset() );
            if ( seg->get_type() == PT_LOAD && seg->get_file_size() > 0 &&
                 sec != nullptr ) {
                Elf64_Addr sec_addr =
                    get_virtual_addr( seg->get_offset(), sec );
                if ( sec_addr != seg->get_virtual_address() ) {
                    errors += "Virtual address of segment " +
                              std::to_string( h ) + " (" +
                              to_hex_string( seg->get_virtual_address() ) +
                              ")" + " conflicts with address of section " +
                              sec->get_name() + " (" +
                              to_hex_string( sec_addr ) + ")" + " at offset " +
                              to_hex_string( seg->get_offset() ) + "\n";
                }
            }
        }

        // more checks to be added here...

        return errors;
    }

  private:
    //------------------------------------------------------------------------------
    //! \brief Check if an offset is within a section
    //! \param offset The offset to check
    //! \param sec Pointer to the section
    //! \return True if the offset is within the section, false otherwise
    static bool is_offset_in_section( Elf64_Off offset, const section* sec )
    {
        return ( offset >= sec->get_offset() ) &&
               ( offset < ( sec->get_offset() + sec->get_size() ) );
    }

    //------------------------------------------------------------------------------
    //! \brief Get the virtual address of an offset within a section
    //! \param offset The offset within the section
    //! \param sec Pointer to the section
    //! \return The virtual address of the offset within the section
    static Elf64_Addr get_virtual_addr( Elf64_Off offset, const section* sec )
    {
        return sec->get_address() + offset - sec->get_offset();
    }

    //------------------------------------------------------------------------------
    //! \brief Find the section that contains a given offset
    //! \param offset The offset to find
    //! \return Pointer to the section that contains the offset, or nullptr if not found
    const section* find_prog_section_for_offset( Elf64_Off offset ) const
    {
        for ( const auto& sec : sections ) {
            if ( sec->get_type() == SHT_PROGBITS &&
                 is_offset_in_section( offset, sec.get() ) ) {
                return sec.get();
            }
        }
        return nullptr;
    }

    //------------------------------------------------------------------------------
    //! \brief Create an ELF header
    //! \param file_class The class of the ELF file (ELFCLASS32 or ELFCLASS64)
    //! \param encoding The encoding of the ELF file (ELFDATA2LSB or ELFDATA2MSB)
    //! \return Unique pointer to the created ELF header
    std::unique_ptr<elf_header> create_header( unsigned char file_class,
                                               unsigned char encoding )
    {
        std::unique_ptr<elf_header> new_header;

        if ( file_class == ELFCLASS64 ) {
            new_header = std::unique_ptr<elf_header>(
                new ( std::nothrow ) elf_header_impl<Elf64_Ehdr>(
                    convertor, encoding, addr_translator ) );
        }
        else if ( file_class == ELFCLASS32 ) {
            new_header = std::unique_ptr<elf_header>(
                new ( std::nothrow ) elf_header_impl<Elf32_Ehdr>(
                    convertor, encoding, addr_translator ) );
        }
        else {
            return nullptr;
        }

        return new_header;
    }

    //------------------------------------------------------------------------------
    //! \brief Create a new section
    //! \return Pointer to the created section
    section* create_section()
    {
        if ( auto file_class = get_class(); file_class == ELFCLASS64 ) {
            sections_.emplace_back(
                new ( std::nothrow ) section_impl<Elf64_Shdr>(
                    convertor, addr_translator, compression ) );
        }
        else if ( file_class == ELFCLASS32 ) {
            sections_.emplace_back(
                new ( std::nothrow ) section_impl<Elf32_Shdr>(
                    convertor, addr_translator, compression ) );
        }
        else {
            sections_.pop_back();
            return nullptr;
        }

        section* new_section = sections_.back().get();
        new_section->set_index( static_cast<Elf_Half>( sections_.size() - 1 ) );

        return new_section;
    }

    //------------------------------------------------------------------------------
    //! \brief Create a new segment
    //! \return Pointer to the created segment
    segment* create_segment()
    {
        if ( auto file_class = header->get_class(); file_class == ELFCLASS64 ) {
            segments_.emplace_back(
                new ( std::nothrow )
                    segment_impl<Elf64_Phdr>( convertor, addr_translator ) );
        }
        else if ( file_class == ELFCLASS32 ) {
            segments_.emplace_back(
                new ( std::nothrow )
                    segment_impl<Elf32_Phdr>( convertor, addr_translator ) );
        }
        else {
            segments_.pop_back();
            return nullptr;
        }

        segment* new_segment = segments_.back().get();
        new_segment->set_index( static_cast<Elf_Half>( segments_.size() - 1 ) );

        return new_segment;
    }

    //------------------------------------------------------------------------------
    //! \brief Create mandatory sections
    void create_mandatory_sections()
    {
        // Create null section without calling to 'add_section' as no string
        // section containing section names exists yet
        section* sec0 = create_section();
        sec0->set_index( 0 );
        sec0->set_name( "" );
        sec0->set_name_string_offset( 0 );

        set_section_name_str_index( 1 );
        section* shstrtab = sections.add( ".shstrtab" );
        shstrtab->set_type( SHT_STRTAB );
        shstrtab->set_addr_align( 1 );
    }

    //------------------------------------------------------------------------------
    //! \brief Load sections from a stream
    //! \param stream The input stream to load from
    //! \param is_lazy Whether to load the sections lazily
    //! \return True if successful, false otherwise
    bool load_sections( std::istream& stream, bool is_lazy )
    {
        unsigned char file_class = header->get_class();
        Elf_Half      entry_size = header->get_section_entry_size();
        Elf_Half      num        = header->get_sections_num();
        Elf64_Off     offset     = header->get_sections_offset();

        if ( ( num != 0 && file_class == ELFCLASS64 &&
               entry_size < sizeof( Elf64_Shdr ) ) ||
             ( num != 0 && file_class == ELFCLASS32 &&
               entry_size < sizeof( Elf32_Shdr ) ) ) {
            return false;
        }

        for ( Elf_Half i = 0; i < num; ++i ) {
            section* sec = create_section();

            // Load return value is ignored here
            // This allows retrieval of information from corrupted sections
            sec->load( stream,
                       static_cast<std::streamoff>( offset ) +
                           static_cast<std::streampos>( i ) * entry_size,
                       is_lazy );
            // To mark that the section is not permitted to reassign address
            // during layout calculation
            sec->set_address( sec->get_address() );
        }

        if ( Elf_Half shstrndx = get_section_name_str_index();
             SHN_UNDEF != shstrndx ) {
            string_section_accessor str_reader( sections[shstrndx] );
            for ( Elf_Half i = 0; i < num; ++i ) {
                Elf_Word section_offset = sections[i]->get_name_string_offset();
                const char* p = str_reader.get_string( section_offset );
                if ( p != nullptr ) {
                    sections[i]->set_name( p );
                }
            }
        }

        return true;
    }

    //------------------------------------------------------------------------------
    //! \brief Checks whether the addresses of the section entirely fall within the given segment.
    //! It doesn't matter if the addresses are memory addresses, or file offsets,
    //!  they just need to be in the same address space
    //! \param sect_begin The beginning address of the section
    //! \param sect_size The size of the section
    //! \param seg_begin The beginning address of the segment
    //! \param seg_end The end address of the segment
    //! \return True if the section is within the segment, false otherwise
    static bool is_sect_in_seg( Elf64_Off sect_begin,
                                Elf_Xword sect_size,
                                Elf64_Off seg_begin,
                                Elf64_Off seg_end )
    {
        return ( seg_begin <= sect_begin ) &&
               ( sect_begin + sect_size <= seg_end ) &&
               ( sect_begin <
                 seg_end ); // this is important criteria when sect_size == 0
        // Example:  seg_begin=10, seg_end=12 (-> covering the bytes 10 and 11)
        //           sect_begin=12, sect_size=0  -> shall return false!
    }

    //------------------------------------------------------------------------------
    //! \brief Load segments from a stream
    //! \param stream The input stream to load from
    //! \param is_lazy Whether to load the segments lazily
    //! \return True if successful, false otherwise
    bool load_segments( std::istream& stream, bool is_lazy )
    {
        unsigned char file_class = header->get_class();
        Elf_Half      entry_size = header->get_segment_entry_size();
        Elf_Half      num        = header->get_segments_num();
        Elf64_Off     offset     = header->get_segments_offset();

        if ( ( num != 0 && file_class == ELFCLASS64 &&
               entry_size < sizeof( Elf64_Phdr ) ) ||
             ( num != 0 && file_class == ELFCLASS32 &&
               entry_size < sizeof( Elf32_Phdr ) ) ) {
            return false;
        }

        for ( Elf_Half i = 0; i < num; ++i ) {
            if ( file_class == ELFCLASS64 ) {
                segments_.emplace_back( new ( std::nothrow )
                                            segment_impl<Elf64_Phdr>(
                                                convertor, addr_translator ) );
            }
            else if ( file_class == ELFCLASS32 ) {
                segments_.emplace_back( new ( std::nothrow )
                                            segment_impl<Elf32_Phdr>(
                                                convertor, addr_translator ) );
            }
            else {
                segments_.pop_back();
                return false;
            }

            segment* seg = segments_.back().get();

            if ( !seg->load( stream,
                             static_cast<std::streamoff>( offset ) +
                                 static_cast<std::streampos>( i ) * entry_size,
                             is_lazy ) ||
                 stream.fail() ) {
                segments_.pop_back();
                return false;
            }

            seg->set_index( i );

            // Add sections to the segments (similar to readelfs algorithm)
            Elf64_Off segBaseOffset = seg->get_offset();
            Elf64_Off segEndOffset  = segBaseOffset + seg->get_file_size();
            Elf64_Off segVBaseAddr  = seg->get_virtual_address();
            Elf64_Off segVEndAddr   = segVBaseAddr + seg->get_memory_size();
            for ( const auto& psec : sections ) {
                // SHF_ALLOC sections are matched based on the virtual address
                // otherwise the file offset is matched
                if ( ( ( psec->get_flags() & SHF_ALLOC ) == SHF_ALLOC )
                         ? is_sect_in_seg( psec->get_address(),
                                           psec->get_size(), segVBaseAddr,
                                           segVEndAddr )
                         : is_sect_in_seg( psec->get_offset(), psec->get_size(),
                                           segBaseOffset, segEndOffset ) ) {

                    // If it is a TLS segment, add TLS sections only and vice versa
                    if ( ( ( seg->get_type() == PT_TLS ) &&
                           ( ( psec->get_flags() & SHF_TLS ) != SHF_TLS ) ) ||
                         ( ( ( psec->get_flags() & SHF_TLS ) == SHF_TLS ) &&
                           ( seg->get_type() != PT_TLS ) ) )
                        continue;

                    // Alignment of segment shall not be updated, to preserve original value
                    // It will be re-calculated on saving.
                    seg->add_section_index( psec->get_index(), 0 );
                }
            }
        }

        return true;
    }

    //------------------------------------------------------------------------------
    //! \brief Save the ELF header to a stream
    //! \param stream The output stream to save to
    //! \return True if successful, false otherwise
    bool save_header( std::ostream& stream ) const
    {
        return header->save( stream );
    }

    //------------------------------------------------------------------------------
    //! \brief Save the sections to a stream
    //! \param stream The output stream to save to
    //! \return True if successful, false otherwise
    bool save_sections( std::ostream& stream ) const
    {
        for ( const auto& sec : sections_ ) {
            std::streampos headerPosition =
                static_cast<std::streamoff>( header->get_sections_offset() ) +
                static_cast<std::streampos>(
                    header->get_section_entry_size() ) *
                    sec->get_index();

            sec->save( stream, headerPosition, sec->get_offset() );
        }
        return true;
    }

    //------------------------------------------------------------------------------
    //! \brief Save the segments to a stream
    //! \param stream The output stream to save to
    //! \return True if successful, false otherwise
    bool save_segments( std::ostream& stream ) const
    {
        for ( const auto& seg : segments_ ) {
            std::streampos headerPosition =
                static_cast<std::streamoff>( header->get_segments_offset() ) +
                static_cast<std::streampos>(
                    header->get_segment_entry_size() ) *
                    seg->get_index();

            seg->save( stream, headerPosition, seg->get_offset() );
        }
        return true;
    }

    //------------------------------------------------------------------------------
    //! \brief Check if a section is without a segment
    //! \param section_index The index of the section
    //! \return True if the section is without a segment, false otherwise
    bool is_section_without_segment( unsigned int section_index ) const
    {
        bool found = false;

        for ( unsigned int j = 0; !found && ( j < segments.size() ); ++j ) {
            for ( Elf_Half k = 0;
                  !found && ( k < segments[j]->get_sections_num() ); ++k ) {
                found = segments[j]->get_section_index_at( k ) == section_index;
            }
        }

        return !found;
    }

    //------------------------------------------------------------------------------
    //! \brief Check if a segment is a subsequence of another segment
    //! \param seg1 Pointer to the first segment
    //! \param seg2 Pointer to the second segment
    //! \return True if seg1 is a subsequence of seg2, false otherwise
    static bool is_subsequence_of( const segment* seg1, const segment* seg2 )
    {
        // Return 'true' if sections of seg1 are a subset of sections in seg2
        const std::vector<Elf_Half>& sections1 = seg1->get_sections();
        const std::vector<Elf_Half>& sections2 = seg2->get_sections();

        bool found = false;
        if ( sections1.size() < sections2.size() ) {
            found = std::includes( sections2.begin(), sections2.end(),
                                   sections1.begin(), sections1.end() );
        }

        return found;
    }

    //------------------------------------------------------------------------------
    //! \brief Get ordered segments
    //! \return Vector of ordered segments
    std::vector<segment*> get_ordered_segments() const
    {
        std::vector<segment*> res;
        std::deque<segment*>  worklist;

        res.reserve( segments.size() );
        for ( const auto& seg : segments ) {
            worklist.emplace_back( seg.get() );
        }

        // Bring the segments which start at address 0 to the front
        size_t nextSlot = 0;
        for ( size_t i = 0; i < worklist.size(); ++i ) {
            if ( i != nextSlot && worklist[i]->is_offset_initialized() &&
                 worklist[i]->get_offset() == 0 ) {
                if ( worklist[nextSlot]->get_offset() == 0 ) {
                    ++nextSlot;
                }
                std::swap( worklist[i], worklist[nextSlot] );
                ++nextSlot;
            }
        }

        while ( !worklist.empty() ) {
            segment* seg = worklist.front();
            worklist.pop_front();

            size_t i = 0;
            for ( ; i < worklist.size(); ++i ) {
                if ( is_subsequence_of( seg, worklist[i] ) ) {
                    break;
                }
            }

            if ( i < worklist.size() ) {
                worklist.emplace_back( seg );
            }
            else {
                res.emplace_back( seg );
            }
        }

        return res;
    }

    //------------------------------------------------------------------------------
    //! \brief Layout sections without segments
    //! \return True if successful, false otherwise
    bool layout_sections_without_segments()
    {
        for ( unsigned int i = 0; i < sections_.size(); ++i ) {
            if ( is_section_without_segment( i ) ) {
                const auto& sec = sections_[i];

                if ( Elf_Xword section_align = sec->get_addr_align();
                     section_align > 1 &&
                     current_file_pos % section_align != 0 ) {
                    current_file_pos +=
                        section_align - current_file_pos % section_align;
                }

                if ( 0 != sec->get_index() ) {
                    sec->set_offset( current_file_pos );
                }

                if ( SHT_NOBITS != sec->get_type() &&
                     SHT_NULL != sec->get_type() ) {
                    current_file_pos += sec->get_size();
                }
            }
        }

        return true;
    }

    //------------------------------------------------------------------------------
    //! \brief Calculate segment alignment
    void calc_segment_alignment() const
    {
        for ( const auto& seg : segments_ ) {
            for ( Elf_Half i = 0; i < seg->get_sections_num(); ++i ) {
                const auto& sect = sections_[seg->get_section_index_at( i )];
                if ( sect->get_addr_align() > seg->get_align() ) {
                    seg->set_align( sect->get_addr_align() );
                }
            }
        }
    }

    //------------------------------------------------------------------------------
    //! \brief Layout segments and their sections
    //! \return True if successful, false otherwise
    bool layout_segments_and_their_sections()
    {
        std::vector<segment*> worklist;
        std::vector<bool>     section_generated( sections.size(), false );

        // Get segments in a order in where segments which contain a
        // sub sequence of other segments are located at the end
        worklist = get_ordered_segments();

        for ( auto* seg : worklist ) {
            Elf_Xword segment_memory   = 0;
            Elf_Xword segment_filesize = 0;
            Elf_Xword seg_start_pos    = current_file_pos;
            // Special case: PHDR segment
            // This segment contains the program headers but no sections
            if ( seg->get_type() == PT_PHDR && seg->get_sections_num() == 0 ) {
                seg_start_pos  = header->get_segments_offset();
                segment_memory = segment_filesize =
                    header->get_segment_entry_size() *
                    static_cast<Elf_Xword>( header->get_segments_num() );
            }
            // Special case:
            else if ( seg->is_offset_initialized() && seg->get_offset() == 0 ) {
                seg_start_pos = 0;
                if ( seg->get_sections_num() > 0 ) {
                    segment_memory = segment_filesize = current_file_pos;
                }
            }
            // New segments with not generated sections
            // have to be aligned
            else if ( seg->get_sections_num() > 0 &&
                      !section_generated[seg->get_section_index_at( 0 )] ) {
                Elf_Xword align = seg->get_align() > 0 ? seg->get_align() : 1;
                Elf64_Off cur_page_alignment = current_file_pos % align;
                Elf64_Off req_page_alignment =
                    seg->get_virtual_address() % align;
                Elf64_Off adjustment = req_page_alignment - cur_page_alignment;

                current_file_pos += ( seg->get_align() + adjustment ) % align;
                seg_start_pos = current_file_pos;
            }
            else if ( seg->get_sections_num() > 0 ) {
                seg_start_pos =
                    sections[seg->get_section_index_at( 0 )]->get_offset();
            }

            // Write segment's data
            if ( !write_segment_data( seg, section_generated, segment_memory,
                                      segment_filesize, seg_start_pos ) ) {
                return false;
            }

            seg->set_file_size( segment_filesize );

            // If we already have a memory size from loading an elf file (value > 0),
            // it must not shrink!
            // Memory size may be bigger than file size and it is the loader's job to do something
            // with the surplus bytes in memory, like initializing them with a defined value.
            if ( seg->get_memory_size() < segment_memory ) {
                seg->set_memory_size( segment_memory );
            }

            seg->set_offset( seg_start_pos );
        }

        return true;
    }

    //------------------------------------------------------------------------------
    //! \brief Layout the section table
    //! \return True if successful, false otherwise
    bool layout_section_table()
    {
        // Simply place the section table at the end for now
        Elf64_Off alignmentError = current_file_pos % 16;
        current_file_pos += 16 - alignmentError;
        header->set_sections_offset( current_file_pos );
        return true;
    }

    //------------------------------------------------------------------------------
    //! \brief Write segment data
    //! \param seg Pointer to the segment
    //! \param section_generated Vector of section generated flags
    //! \param segment_memory Reference to the segment memory size
    //! \param segment_filesize Reference to the segment file size
    //! \param seg_start_pos The start position of the segment
    //! \return True if successful, false otherwise
    bool write_segment_data( const segment*     seg,
                             std::vector<bool>& section_generated,
                             Elf_Xword&         segment_memory,
                             Elf_Xword&         segment_filesize,
                             const Elf_Xword&   seg_start_pos )
    {
        for ( Elf_Half j = 0; j < seg->get_sections_num(); ++j ) {
            Elf_Half index = seg->get_section_index_at( j );

            section* sec = sections[index];

            // The NULL section is always generated
            if ( SHT_NULL == sec->get_type() ) {
                section_generated[index] = true;
                continue;
            }

            Elf_Xword section_align = 0;
            // Fix up the alignment
            if ( !section_generated[index] && sec->is_address_initialized() &&
                 SHT_NOBITS != sec->get_type() && SHT_NULL != sec->get_type() &&
                 0 != sec->get_size() ) {
                // Align the sections based on the virtual addresses
                // when possible (this is what matters for execution)
                Elf64_Off req_offset =
                    sec->get_address() - seg->get_virtual_address();
                Elf64_Off cur_offset = current_file_pos - seg_start_pos;
                if ( req_offset < cur_offset ) {
                    // something has gone awfully wrong, abort!
                    // section_align would turn out negative, seeking backwards and overwriting previous data
                    return false;
                }
                section_align = req_offset - cur_offset;
            }
            else if ( !section_generated[index] &&
                      !sec->is_address_initialized() ) {
                // If no address has been specified then only the section
                // alignment constraint has to be matched
                Elf_Xword align = sec->get_addr_align();
                if ( align == 0 ) {
                    align = 1;
                }
                Elf64_Off error = current_file_pos % align;
                section_align   = ( align - error ) % align;
            }
            else if ( section_generated[index] ) {
                // Alignment for already generated sections
                section_align =
                    sec->get_offset() - seg_start_pos - segment_filesize;
            }

            // Determine the segment file and memory sizes
            // Special case .tbss section (NOBITS) in non TLS segment
            if ( ( ( sec->get_flags() & SHF_ALLOC ) == SHF_ALLOC ) &&
                 !( ( ( sec->get_flags() & SHF_TLS ) == SHF_TLS ) &&
                    ( seg->get_type() != PT_TLS ) &&
                    ( SHT_NOBITS == sec->get_type() ) ) ) {
                segment_memory += sec->get_size() + section_align;
            }

            if ( SHT_NOBITS != sec->get_type() ) {
                segment_filesize += sec->get_size() + section_align;
            }

            // Nothing to be done when generating nested segments
            if ( section_generated[index] ) {
                continue;
            }

            current_file_pos += section_align;

            // Set the section addresses when missing
            if ( !sec->is_address_initialized() ) {
                sec->set_address( seg->get_virtual_address() +
                                  current_file_pos - seg_start_pos );
            }

            if ( 0 != sec->get_index() ) {
                sec->set_offset( current_file_pos );
            }

            if ( SHT_NOBITS != sec->get_type() ) {
                current_file_pos += sec->get_size();
            }

            section_generated[index] = true;
        }

        return true;
    }

    //------------------------------------------------------------------------------
  public:
    //! \class Sections
    //! \brief The Sections class provides methods to manipulate sections in an ELF file.
    friend class Sections;
    class Sections
    {
      public:
        //------------------------------------------------------------------------------
        //! \brief Constructor
        //! \param parent Pointer to the parent elfio object
        explicit Sections( elfio* parent ) : parent( parent ) {}

        //------------------------------------------------------------------------------
        //! \brief Get the number of sections
        //! \return The number of sections
        Elf_Half size() const
        {
            return static_cast<Elf_Half>( parent->sections_.size() );
        }

        //------------------------------------------------------------------------------
        //! \brief Get a section by index
        //! \param index The index of the section
        //! \return Pointer to the section, or nullptr if not found
        section* operator[]( unsigned int index ) const
        {
            section* sec = nullptr;

            if ( index < parent->sections_.size() ) {
                sec = parent->sections_[index].get();
            }

            return sec;
        }

        //------------------------------------------------------------------------------
        //! \brief Get a section by name
        //! \param name The name of the section
        //! \return Pointer to the section, or nullptr if not found
        section* operator[]( const std::string_view& name ) const
        {
            section* sec = nullptr;

            for ( const auto& it : parent->sections_ ) {
                if ( it->get_name() == name ) {
                    sec = it.get();
                    break;
                }
            }

            return sec;
        }

        //------------------------------------------------------------------------------
        //! \brief Add a new section
        //! \param name The name of the section
        //! \return Pointer to the created section
        section* add( const std::string& name ) const
        {
            section* new_section = parent->create_section();
            new_section->set_name( name );

            Elf_Half str_index = parent->get_section_name_str_index();
            section* string_table( parent->sections_[str_index].get() );
            string_section_accessor str_writer( string_table );
            Elf_Word                pos = str_writer.add_string( name );
            new_section->set_name_string_offset( pos );

            return new_section;
        }

        //------------------------------------------------------------------------------
        //! \brief Get an iterator to the beginning of the sections
        //! \return Iterator to the beginning of the sections
        std::vector<std::unique_ptr<section>>::iterator begin()
        {
            return parent->sections_.begin();
        }

        //------------------------------------------------------------------------------
        //! \brief Get an iterator to the end of the sections
        //! \return Iterator to the end of the sections
        std::vector<std::unique_ptr<section>>::iterator end()
        {
            return parent->sections_.end();
        }

        //------------------------------------------------------------------------------
        //! \brief Get a const iterator to the beginning of the sections
        //! \return Const iterator to the beginning of the sections
        std::vector<std::unique_ptr<section>>::const_iterator begin() const
        {
            return parent->sections_.cbegin();
        }

        //------------------------------------------------------------------------------
        //! \brief Get a const iterator to the end of the sections
        //! \return Const iterator to the end of the sections
        std::vector<std::unique_ptr<section>>::const_iterator end() const
        {
            return parent->sections_.cend();
        }

        //------------------------------------------------------------------------------
      private:
        elfio* parent; //!< Pointer to the parent elfio object
    };
    Sections sections; //!< Sections object

    //------------------------------------------------------------------------------
    //! \class Segments
    //! \brief The Segments class provides methods to manipulate segments in an ELF file.
    friend class Segments;
    class Segments
    {
      public:
        //------------------------------------------------------------------------------
        //! \brief Constructor
        //! \param parent Pointer to the parent elfio object
        explicit Segments( elfio* parent ) : parent( parent ) {}

        //------------------------------------------------------------------------------
        //! \brief Get the number of segments
        //! \return The number of segments
        Elf_Half size() const
        {
            return static_cast<Elf_Half>( parent->segments_.size() );
        }

        //------------------------------------------------------------------------------
        //! \brief Get a segment by index
        //! \param index The index of the segment
        //! \return Pointer to the segment, or nullptr if not found
        segment* operator[]( unsigned int index ) const
        {
            return parent->segments_[index].get();
        }

        //------------------------------------------------------------------------------
        //! \brief Add a new segment
        //! \return Pointer to the created segment
        segment* add() { return parent->create_segment(); }

        //------------------------------------------------------------------------------
        //! \brief Get an iterator to the beginning of the segments
        //! \return Iterator to the beginning of the segments
        std::vector<std::unique_ptr<segment>>::iterator begin()
        {
            return parent->segments_.begin();
        }

        //------------------------------------------------------------------------------
        //! \brief Get an iterator to the end of the segments
        //! \return Iterator to the end of the segments
        std::vector<std::unique_ptr<segment>>::iterator end()
        {
            return parent->segments_.end();
        }

        //------------------------------------------------------------------------------
        //! \brief Get a const iterator to the beginning of the segments
        //! \return Const iterator to the beginning of the segments
        std::vector<std::unique_ptr<segment>>::const_iterator begin() const
        {
            return parent->segments_.cbegin();
        }

        //------------------------------------------------------------------------------
        //! \brief Get a const iterator to the end of the segments
        //! \return Const iterator to the end of the segments
        std::vector<std::unique_ptr<segment>>::const_iterator end() const
        {
            return parent->segments_.cend();
        }

        //------------------------------------------------------------------------------
      private:
        elfio* parent; //!< Pointer to the parent elfio object
    };
    Segments segments; //!< Segments object

    //------------------------------------------------------------------------------
  private:
    std::unique_ptr<std::ifstream> pstream =
        nullptr; //!< Pointer to the input stream
    std::unique_ptr<elf_header> header = nullptr; //!< Pointer to the ELF header
    std::vector<std::unique_ptr<section>> sections_; //!< Vector of sections
    std::vector<std::unique_ptr<segment>> segments_; //!< Vector of segments
    std::shared_ptr<endianness_convertor> convertor; //!< Endianness convertor
    std::shared_ptr<address_translator> addr_translator; //!< Address translator
    std::shared_ptr<compression_interface> compression =
        nullptr; //!< Pointer to the compression interface

    Elf_Xword current_file_pos = 0; //!< Current file position
};

} // namespace ELFIO

#include <elfio/elfio_symbols.hpp>
#include <elfio/elfio_note.hpp>
#include <elfio/elfio_relocation.hpp>
#include <elfio/elfio_dynamic.hpp>
#include <elfio/elfio_array.hpp>
#include <elfio/elfio_modinfo.hpp>
#include <elfio/elfio_versym.hpp>

#endif // ELFIO_HPP
