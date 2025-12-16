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

#ifndef ELFIO_RELOCATION_HPP
#define ELFIO_RELOCATION_HPP

namespace ELFIO {

template <typename T> struct get_sym_and_type;
template <> struct get_sym_and_type<Elf32_Rel>
{
    //------------------------------------------------------------------------------
    //! \brief Get the symbol from the relocation info
    //! \param info Relocation info
    //! \return Symbol
    static int get_r_sym( Elf_Xword info )
    {
        return ELF32_R_SYM( (Elf_Word)info );
    }
    //------------------------------------------------------------------------------
    //! \brief Get the type from the relocation info
    //! \param info Relocation info
    //! \return Type
    static int get_r_type( Elf_Xword info )
    {
        return ELF32_R_TYPE( (Elf_Word)info );
    }
};
template <> struct get_sym_and_type<Elf32_Rela>
{
    //------------------------------------------------------------------------------
    //! \brief Get the symbol from the relocation info
    //! \param info Relocation info
    //! \return Symbol
    static int get_r_sym( Elf_Xword info )
    {
        return ELF32_R_SYM( (Elf_Word)info );
    }
    //------------------------------------------------------------------------------
    //! \brief Get the type from the relocation info
    //! \param info Relocation info
    //! \return Type
    static int get_r_type( Elf_Xword info )
    {
        return ELF32_R_TYPE( (Elf_Word)info );
    }
};
template <> struct get_sym_and_type<Elf64_Rel>
{
    //------------------------------------------------------------------------------
    //! \brief Get the symbol from the relocation info
    //! \param info Relocation info
    //! \return Symbol
    static int get_r_sym( Elf_Xword info ) { return ELF64_R_SYM( info ); }
    //------------------------------------------------------------------------------
    //! \brief Get the type from the relocation info
    //! \param info Relocation info
    //! \return Type
    static int get_r_type( Elf_Xword info ) { return ELF64_R_TYPE( info ); }
};
template <> struct get_sym_and_type<Elf64_Rela>
{
    //------------------------------------------------------------------------------
    //! \brief Get the symbol from the relocation info
    //! \param info Relocation info
    //! \return Symbol
    static int get_r_sym( Elf_Xword info ) { return ELF64_R_SYM( info ); }
    //------------------------------------------------------------------------------
    //! \brief Get the type from the relocation info
    //! \param info Relocation info
    //! \return Type
    static int get_r_type( Elf_Xword info ) { return ELF64_R_TYPE( info ); }
};

//------------------------------------------------------------------------------
//! \class relocation_section_accessor_template
//! \brief Class for accessing relocation section data
template <class S> class relocation_section_accessor_template
{
  public:
    //------------------------------------------------------------------------------
    //! \brief Constructor
    //! \param elf_file Reference to the ELF file
    //! \param section Pointer to the section
    explicit relocation_section_accessor_template( const elfio& elf_file,
                                                   S*           section )
        : elf_file( elf_file ), relocation_section( section )
    {
    }

    //------------------------------------------------------------------------------
    //! \brief Get the number of entries
    //! \return Number of entries
    Elf_Xword get_entries_num() const
    {
        Elf_Xword nRet = 0;

        if ( 0 != relocation_section->get_entry_size() ) {
            nRet = relocation_section->get_size() /
                   relocation_section->get_entry_size();
        }

        return nRet;
    }

    //------------------------------------------------------------------------------
    //! \brief Get an entry
    //! \param index Index of the entry
    //! \param offset Offset of the entry
    //! \param symbol Symbol of the entry
    //! \param type Type of the entry
    //! \param addend Addend of the entry
    //! \return True if successful, false otherwise
    bool get_entry( Elf_Xword   index,
                    Elf64_Addr& offset,
                    Elf_Word&   symbol,
                    unsigned&   type,
                    Elf_Sxword& addend ) const
    {
        if ( index >= get_entries_num() ) { // Is index valid
            return false;
        }

        if ( elf_file.get_class() == ELFCLASS32 ) {
            if ( SHT_REL == relocation_section->get_type() ) {
                return generic_get_entry_rel<Elf32_Rel>( index, offset, symbol,
                                                         type, addend );
            }
            else if ( SHT_RELA == relocation_section->get_type() ) {
                return generic_get_entry_rela<Elf32_Rela>(
                    index, offset, symbol, type, addend );
            }
        }
        else {
            if ( SHT_REL == relocation_section->get_type() ) {
                return generic_get_entry_rel<Elf64_Rel>( index, offset, symbol,
                                                         type, addend );
            }
            else if ( SHT_RELA == relocation_section->get_type() ) {
                return generic_get_entry_rela<Elf64_Rela>(
                    index, offset, symbol, type, addend );
            }
        }
        // Unknown relocation section type.
        return false;
    }

    //------------------------------------------------------------------------------
    //! \brief Get an entry with additional information
    //! \param index Index of the entry
    //! \param offset Offset of the entry
    //! \param symbolValue Value of the symbol
    //! \param symbolName Name of the symbol
    //! \param type Type of the entry
    //! \param addend Addend of the entry
    //! \param calcValue Calculated value
    //! \return True if successful, false otherwise
    bool get_entry( Elf_Xword    index,
                    Elf64_Addr&  offset,
                    Elf64_Addr&  symbolValue,
                    std::string& symbolName,
                    unsigned&    type,
                    Elf_Sxword&  addend,
                    Elf_Sxword&  calcValue ) const
    {
        // Do regular job
        Elf_Word symbol = 0;
        bool     ret    = get_entry( index, offset, symbol, type, addend );

        // Find the symbol
        Elf_Xword     size;
        unsigned char bind;
        unsigned char symbolType;
        Elf_Half      section;
        unsigned char other;

        symbol_section_accessor symbols(
            elf_file, elf_file.sections[get_symbol_table_index()] );
        ret = ret && symbols.get_symbol( symbol, symbolName, symbolValue, size,
                                         bind, symbolType, section, other );

        if ( ret ) { // Was it successful?
            switch ( type ) {
            case R_386_NONE: // none
                calcValue = 0;
                break;
            case R_386_32: // S + A
                calcValue = symbolValue + addend;
                break;
            case R_386_PC32: // S + A - P
                calcValue = symbolValue + addend - offset;
                break;
            case R_386_GOT32: // G + A - P
                calcValue = 0;
                break;
            case R_386_PLT32: // L + A - P
                calcValue = 0;
                break;
            case R_386_COPY: // none
                calcValue = 0;
                break;
            case R_386_GLOB_DAT: // S
            case R_386_JMP_SLOT: // S
                calcValue = symbolValue;
                break;
            case R_386_RELATIVE: // B + A
                calcValue = addend;
                break;
            case R_386_GOTOFF: // S + A - GOT
                calcValue = 0;
                break;
            case R_386_GOTPC: // GOT + A - P
                calcValue = 0;
                break;
            default: // Not recognized symbol!
                calcValue = 0;
                break;
            }
        }

        return ret;
    }

    //------------------------------------------------------------------------------
    //! \brief Set an entry
    //! \param index Index of the entry
    //! \param offset Offset of the entry
    //! \param symbol Symbol of the entry
    //! \param type Type of the entry
    //! \param addend Addend of the entry
    //! \return True if successful, false otherwise
    bool set_entry( Elf_Xword  index,
                    Elf64_Addr offset,
                    Elf_Word   symbol,
                    unsigned   type,
                    Elf_Sxword addend )
    {
        if ( index >= get_entries_num() ) { // Is index valid
            return false;
        }

        if ( elf_file.get_class() == ELFCLASS32 ) {
            if ( SHT_REL == relocation_section->get_type() ) {
                generic_set_entry_rel<Elf32_Rel>( index, offset, symbol, type,
                                                  addend );
            }
            else if ( SHT_RELA == relocation_section->get_type() ) {
                generic_set_entry_rela<Elf32_Rela>( index, offset, symbol, type,
                                                    addend );
            }
        }
        else {
            if ( SHT_REL == relocation_section->get_type() ) {
                generic_set_entry_rel<Elf64_Rel>( index, offset, symbol, type,
                                                  addend );
            }
            else if ( SHT_RELA == relocation_section->get_type() ) {
                generic_set_entry_rela<Elf64_Rela>( index, offset, symbol, type,
                                                    addend );
            }
        }

        return true;
    }

    //------------------------------------------------------------------------------
    //! \brief Add an entry
    //! \param offset Offset of the entry
    //! \param info Information of the entry
    void add_entry( Elf64_Addr offset, Elf_Xword info )
    {
        if ( elf_file.get_class() == ELFCLASS32 ) {
            generic_add_entry<Elf32_Rel>( offset, info );
        }
        else {
            generic_add_entry<Elf64_Rel>( offset, info );
        }
    }

    //------------------------------------------------------------------------------
    //! \brief Add an entry
    //! \param offset Offset of the entry
    //! \param symbol Symbol of the entry
    //! \param type Type of the entry
    void add_entry( Elf64_Addr offset, Elf_Word symbol, unsigned type )
    {
        Elf_Xword info;
        if ( elf_file.get_class() == ELFCLASS32 ) {
            info = ELF32_R_INFO( (Elf_Xword)symbol, type );
        }
        else {
            info = ELF64_R_INFO( (Elf_Xword)symbol, type );
        }

        add_entry( offset, info );
    }

    //------------------------------------------------------------------------------
    //! \brief Add an entry
    //! \param offset Offset of the entry
    //! \param info Information of the entry
    //! \param addend Addend of the entry
    void add_entry( Elf64_Addr offset, Elf_Xword info, Elf_Sxword addend )
    {
        if ( elf_file.get_class() == ELFCLASS32 ) {
            generic_add_entry<Elf32_Rela>( offset, info, addend );
        }
        else {
            generic_add_entry<Elf64_Rela>( offset, info, addend );
        }
    }

    //------------------------------------------------------------------------------
    //! \brief Add an entry
    //! \param offset Offset of the entry
    //! \param symbol Symbol of the entry
    //! \param type Type of the entry
    //! \param addend Addend of the entry
    void add_entry( Elf64_Addr offset,
                    Elf_Word   symbol,
                    unsigned   type,
                    Elf_Sxword addend )
    {
        Elf_Xword info;
        if ( elf_file.get_class() == ELFCLASS32 ) {
            info = ELF32_R_INFO( (Elf_Xword)symbol, type );
        }
        else {
            info = ELF64_R_INFO( (Elf_Xword)symbol, type );
        }

        add_entry( offset, info, addend );
    }

    //------------------------------------------------------------------------------
    //! \brief Add an entry with additional information
    //! \param str_writer String section accessor
    //! \param str String
    //! \param sym_writer Symbol section accessor
    //! \param value Value of the symbol
    //! \param size Size of the symbol
    //! \param sym_info Symbol information
    //! \param other Other information
    //! \param shndx Section index
    //! \param offset Offset of the entry
    //! \param type Type of the entry
    void add_entry( string_section_accessor str_writer,
                    const char*             str,
                    symbol_section_accessor sym_writer,
                    Elf64_Addr              value,
                    Elf_Word                size,
                    unsigned char           sym_info,
                    unsigned char           other,
                    Elf_Half                shndx,
                    Elf64_Addr              offset,
                    unsigned                type )
    {
        Elf_Word str_index = str_writer.add_string( str );
        Elf_Word sym_index = sym_writer.add_symbol( str_index, value, size,
                                                    sym_info, other, shndx );
        add_entry( offset, sym_index, type );
    }

    //------------------------------------------------------------------------------
    //! \brief Swap symbols
    //! \param first First symbol
    //! \param second Second symbol
    void swap_symbols( Elf_Xword first, Elf_Xword second )
    {
        Elf64_Addr offset = 0;
        Elf_Word   symbol = 0;
        unsigned   rtype  = 0;
        Elf_Sxword addend = 0;
        for ( Elf_Word i = 0; i < get_entries_num(); i++ ) {
            get_entry( i, offset, symbol, rtype, addend );
            if ( symbol == first ) {
                set_entry( i, offset, (Elf_Word)second, rtype, addend );
            }
            if ( symbol == second ) {
                set_entry( i, offset, (Elf_Word)first, rtype, addend );
            }
        }
    }

    //------------------------------------------------------------------------------
  private:
    //------------------------------------------------------------------------------
    //! \brief Get the symbol table index
    //! \return Symbol table index
    Elf_Half get_symbol_table_index() const
    {
        return (Elf_Half)relocation_section->get_link();
    }

    //------------------------------------------------------------------------------
    //! \brief Get a generic entry for REL type
    //! \param index Index of the entry
    //! \param offset Offset of the entry
    //! \param symbol Symbol of the entry
    //! \param type Type of the entry
    //! \param addend Addend of the entry
    //! \return True if successful, false otherwise
    template <class T>
    bool generic_get_entry_rel( Elf_Xword   index,
                                Elf64_Addr& offset,
                                Elf_Word&   symbol,
                                unsigned&   type,
                                Elf_Sxword& addend ) const
    {
        const auto& convertor = elf_file.get_convertor();

        if ( relocation_section->get_entry_size() < sizeof( T ) ) {
            return false;
        }
        const T* pEntry = reinterpret_cast<const T*>(
            relocation_section->get_data() +
            index * relocation_section->get_entry_size() );
        offset        = ( *convertor )( pEntry->r_offset );
        Elf_Xword tmp = ( *convertor )( pEntry->r_info );
        symbol        = get_sym_and_type<T>::get_r_sym( tmp );
        type          = get_sym_and_type<T>::get_r_type( tmp );
        addend        = 0;
        return true;
    }

    //------------------------------------------------------------------------------
    //! \brief Get a generic entry for RELA type
    //! \param index Index of the entry
    //! \param offset Offset of the entry
    //! \param symbol Symbol of the entry
    //! \param type Type of the entry
    //! \param addend Addend of the entry
    //! \return True if successful, false otherwise
    template <class T>
    bool generic_get_entry_rela( Elf_Xword   index,
                                 Elf64_Addr& offset,
                                 Elf_Word&   symbol,
                                 unsigned&   type,
                                 Elf_Sxword& addend ) const
    {
        const auto& convertor = elf_file.get_convertor();

        if ( relocation_section->get_entry_size() < sizeof( T ) ) {
            return false;
        }

        const T* pEntry = reinterpret_cast<const T*>(
            relocation_section->get_data() +
            index * relocation_section->get_entry_size() );
        offset        = ( *convertor )( pEntry->r_offset );
        Elf_Xword tmp = ( *convertor )( pEntry->r_info );
        symbol        = get_sym_and_type<T>::get_r_sym( tmp );
        type          = get_sym_and_type<T>::get_r_type( tmp );
        addend        = ( *convertor )( pEntry->r_addend );
        return true;
    }

    //------------------------------------------------------------------------------
    //! \brief Set a generic entry for REL type
    //! \param index Index of the entry
    //! \param offset Offset of the entry
    //! \param symbol Symbol of the entry
    //! \param type Type of the entry
    //! \param addend Addend of the entry
    template <class T>
    void generic_set_entry_rel( Elf_Xword  index,
                                Elf64_Addr offset,
                                Elf_Word   symbol,
                                unsigned   type,
                                Elf_Sxword )
    {
        const auto& convertor = elf_file.get_convertor();

        T* pEntry = const_cast<T*>( reinterpret_cast<const T*>(
            relocation_section->get_data() +
            index * relocation_section->get_entry_size() ) );

        if ( elf_file.get_class() == ELFCLASS32 ) {
            pEntry->r_info = ELF32_R_INFO( (Elf_Xword)symbol, type );
        }
        else {
            pEntry->r_info = ELF64_R_INFO( (Elf_Xword)symbol, type );
        }
        pEntry->r_offset = decltype( pEntry->r_offset )( offset );
        pEntry->r_offset = ( *convertor )( pEntry->r_offset );
        pEntry->r_info   = ( *convertor )( pEntry->r_info );
    }

    //------------------------------------------------------------------------------
    //! \brief Set a generic entry for RELA type
    //! \param index Index of the entry
    //! \param offset Offset of the entry
    //! \param symbol Symbol of the entry
    //! \param type Type of the entry
    //! \param addend Addend of the entry
    template <class T>
    void generic_set_entry_rela( Elf_Xword  index,
                                 Elf64_Addr offset,
                                 Elf_Word   symbol,
                                 unsigned   type,
                                 Elf_Sxword addend )
    {
        const auto& convertor = elf_file.get_convertor();

        T* pEntry = const_cast<T*>( reinterpret_cast<const T*>(
            relocation_section->get_data() +
            index * relocation_section->get_entry_size() ) );

        if ( elf_file.get_class() == ELFCLASS32 ) {
            pEntry->r_info = ELF32_R_INFO( (Elf_Xword)symbol, type );
        }
        else {
            pEntry->r_info = ELF64_R_INFO( (Elf_Xword)symbol, type );
        }
        pEntry->r_offset = decltype( pEntry->r_offset )( offset );
        pEntry->r_addend = decltype( pEntry->r_addend )( addend );
        pEntry->r_offset = ( *convertor )( pEntry->r_offset );
        pEntry->r_info   = ( *convertor )( pEntry->r_info );
        pEntry->r_addend = ( *convertor )( pEntry->r_addend );
    }

    //------------------------------------------------------------------------------
    //! \brief Add a generic entry for REL type
    //! \param offset Offset of the entry
    //! \param info Information of the entry
    template <class T>
    void generic_add_entry( Elf64_Addr offset, Elf_Xword info )
    {
        const auto& convertor = elf_file.get_convertor();

        T entry;
        entry.r_offset = decltype( entry.r_offset )( offset );
        entry.r_info   = decltype( entry.r_info )( info );
        entry.r_offset = ( *convertor )( entry.r_offset );
        entry.r_info   = ( *convertor )( entry.r_info );

        relocation_section->append_data( reinterpret_cast<char*>( &entry ),
                                         sizeof( entry ) );
    }

    //------------------------------------------------------------------------------
    //! \brief Add a generic entry for RELA type
    //! \param offset Offset of the entry
    //! \param info Information of the entry
    //! \param addend Addend of the entry
    template <class T>
    void
    generic_add_entry( Elf64_Addr offset, Elf_Xword info, Elf_Sxword addend )
    {
        const auto& convertor = elf_file.get_convertor();

        T entry;
        entry.r_offset = offset;
        entry.r_info   = info;
        entry.r_addend = addend;
        entry.r_offset = ( *convertor )( entry.r_offset );
        entry.r_info   = ( *convertor )( entry.r_info );
        entry.r_addend = ( *convertor )( entry.r_addend );

        relocation_section->append_data( reinterpret_cast<char*>( &entry ),
                                         sizeof( entry ) );
    }

    //------------------------------------------------------------------------------
  private:
    const elfio& elf_file;
    S*           relocation_section = nullptr;
};

using relocation_section_accessor =
    relocation_section_accessor_template<section>;
using const_relocation_section_accessor =
    relocation_section_accessor_template<const section>;

} // namespace ELFIO

#endif // ELFIO_RELOCATION_HPP
