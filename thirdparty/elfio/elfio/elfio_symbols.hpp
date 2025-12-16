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

#ifndef ELFIO_SYMBOLS_HPP
#define ELFIO_SYMBOLS_HPP

namespace ELFIO {

//------------------------------------------------------------------------------
// @class symbol_section_accessor_template
// @brief A template class for accessing symbol sections in an ELF file.
//------------------------------------------------------------------------------
template <class S> class symbol_section_accessor_template
{
  public:
    //------------------------------------------------------------------------------
    // @brief Constructor
    // @param elf_file Reference to the ELF file
    // @param symbol_section Pointer to the symbol section
    //------------------------------------------------------------------------------
    explicit symbol_section_accessor_template( const elfio& elf_file,
                                               S*           symbol_section )
        : elf_file( elf_file ), symbol_section( symbol_section )
    {
        find_hash_section();
    }

    //------------------------------------------------------------------------------
    // @brief Get the number of symbols in the section
    // @return Number of symbols
    //------------------------------------------------------------------------------
    Elf_Xword get_symbols_num() const
    {
        Elf_Xword nRet = 0;

        size_t minimum_symbol_size;
        switch ( elf_file.get_class() ) {
        case ELFCLASS32:
            minimum_symbol_size = sizeof( Elf32_Sym );
            break;
        case ELFCLASS64:
            minimum_symbol_size = sizeof( Elf64_Sym );
            break;
        default:
            return nRet;
        }

        if ( symbol_section->get_entry_size() >= minimum_symbol_size &&
             symbol_section->get_size() <= symbol_section->get_stream_size() ) {
            nRet =
                symbol_section->get_size() / symbol_section->get_entry_size();
        }

        return nRet;
    }

    //------------------------------------------------------------------------------
    // @brief Get the symbol at the specified index
    // @param index Index of the symbol
    // @param name Name of the symbol
    // @param value Value of the symbol
    // @param size Size of the symbol
    // @param bind Binding of the symbol
    // @param type Type of the symbol
    // @param section_index Section index of the symbol
    // @param other Other attributes of the symbol
    // @return True if the symbol is found, false otherwise
    //------------------------------------------------------------------------------
    bool get_symbol( Elf_Xword      index,
                     std::string&   name,
                     Elf64_Addr&    value,
                     Elf_Xword&     size,
                     unsigned char& bind,
                     unsigned char& type,
                     Elf_Half&      section_index,
                     unsigned char& other ) const
    {
        bool ret = false;

        if ( elf_file.get_class() == ELFCLASS32 ) {
            ret = generic_get_symbol<Elf32_Sym>( index, name, value, size, bind,
                                                 type, section_index, other );
        }
        else {
            ret = generic_get_symbol<Elf64_Sym>( index, name, value, size, bind,
                                                 type, section_index, other );
        }

        return ret;
    }

    //------------------------------------------------------------------------------
    // @brief Get the symbol with the specified name
    // @param name Name of the symbol
    // @param value Value of the symbol
    // @param size Size of the symbol
    // @param bind Binding of the symbol
    // @param type Type of the symbol
    // @param section_index Section index of the symbol
    // @param other Other attributes of the symbol
    // @return True if the symbol is found, false otherwise
    //------------------------------------------------------------------------------
    bool get_symbol( const std::string& name,
                     Elf64_Addr&        value,
                     Elf_Xword&         size,
                     unsigned char&     bind,
                     unsigned char&     type,
                     Elf_Half&          section_index,
                     unsigned char&     other ) const
    {
        bool ret = false;

        if ( 0 != get_hash_table_index() ) {
            if ( hash_section->get_type() == SHT_HASH ) {
                ret = hash_lookup( name, value, size, bind, type, section_index,
                                   other );
            }
            if ( hash_section->get_type() == SHT_GNU_HASH ||
                 hash_section->get_type() == DT_GNU_HASH ) {
                if ( elf_file.get_class() == ELFCLASS32 ) {
                    ret = gnu_hash_lookup<std::uint32_t>(
                        name, value, size, bind, type, section_index, other );
                }
                else {
                    ret = gnu_hash_lookup<std::uint64_t>(
                        name, value, size, bind, type, section_index, other );
                }
            }
        }

        if ( !ret ) {
            for ( Elf_Xword i = 0; !ret && i < get_symbols_num(); i++ ) {
                std::string symbol_name;
                if ( get_symbol( i, symbol_name, value, size, bind, type,
                                 section_index, other ) &&
                     ( symbol_name == name ) ) {
                    ret = true;
                }
            }
        }

        return ret;
    }

    //------------------------------------------------------------------------------
    // @brief Get the symbol with the specified value
    // @param value Value of the symbol
    // @param name Name of the symbol
    // @param size Size of the symbol
    // @param bind Binding of the symbol
    // @param type Type of the symbol
    // @param section_index Section index of the symbol
    // @param other Other attributes of the symbol
    // @return True if the symbol is found, false otherwise
    //------------------------------------------------------------------------------
    bool get_symbol( const Elf64_Addr& value,
                     std::string&      name,
                     Elf_Xword&        size,
                     unsigned char&    bind,
                     unsigned char&    type,
                     Elf_Half&         section_index,
                     unsigned char&    other ) const
    {

        const auto& convertor = elf_file.get_convertor();

        Elf_Xword  idx   = 0;
        bool       match = false;
        Elf64_Addr v     = 0;

        if ( elf_file.get_class() == ELFCLASS32 ) {
            match = generic_search_symbols<Elf32_Sym>(
                [&]( const Elf32_Sym* sym ) {
                    return ( *convertor )( sym->st_value ) == value;
                },
                idx );
        }
        else {
            match = generic_search_symbols<Elf64_Sym>(
                [&]( const Elf64_Sym* sym ) {
                    return ( *convertor )( sym->st_value ) == value;
                },
                idx );
        }

        if ( match ) {
            return get_symbol( idx, name, v, size, bind, type, section_index,
                               other );
        }

        return false;
    }

    //------------------------------------------------------------------------------
    // @brief Add a symbol to the section
    // @param name Name of the symbol
    // @param value Value of the symbol
    // @param size Size of the symbol
    // @param info Info of the symbol
    // @param other Other attributes of the symbol
    // @param shndx Section index of the symbol
    // @return Index of the added symbol
    //------------------------------------------------------------------------------
    Elf_Word add_symbol( Elf_Word      name,
                         Elf64_Addr    value,
                         Elf_Xword     size,
                         unsigned char info,
                         unsigned char other,
                         Elf_Half      shndx )
    {
        Elf_Word nRet;

        if ( symbol_section->get_size() == 0 ) {
            if ( elf_file.get_class() == ELFCLASS32 ) {
                nRet = generic_add_symbol<Elf32_Sym>( 0, 0, 0, 0, 0, 0 );
            }
            else {
                nRet = generic_add_symbol<Elf64_Sym>( 0, 0, 0, 0, 0, 0 );
            }
        }

        if ( elf_file.get_class() == ELFCLASS32 ) {
            nRet = generic_add_symbol<Elf32_Sym>( name, value, size, info,
                                                  other, shndx );
        }
        else {
            nRet = generic_add_symbol<Elf64_Sym>( name, value, size, info,
                                                  other, shndx );
        }

        return nRet;
    }

    //------------------------------------------------------------------------------
    // @brief Add a symbol to the section
    // @param name Name of the symbol
    // @param value Value of the symbol
    // @param size Size of the symbol
    // @param bind Binding of the symbol
    // @param type Type of the symbol
    // @param other Other attributes of the symbol
    // @param shndx Section index of the symbol
    // @return Index of the added symbol
    //------------------------------------------------------------------------------
    Elf_Word add_symbol( Elf_Word      name,
                         Elf64_Addr    value,
                         Elf_Xword     size,
                         unsigned char bind,
                         unsigned char type,
                         unsigned char other,
                         Elf_Half      shndx )
    {
        return add_symbol( name, value, size, ELF_ST_INFO( bind, type ), other,
                           shndx );
    }

    //------------------------------------------------------------------------------
    // @brief Add a symbol to the section
    // @param pStrWriter String section accessor
    // @param str Name of the symbol
    // @param value Value of the symbol
    // @param size Size of the symbol
    // @param info Info of the symbol
    // @param other Other attributes of the symbol
    // @param shndx Section index of the symbol
    // @return Index of the added symbol
    //------------------------------------------------------------------------------
    Elf_Word add_symbol( string_section_accessor& pStrWriter,
                         const char*              str,
                         Elf64_Addr               value,
                         Elf_Xword                size,
                         unsigned char            info,
                         unsigned char            other,
                         Elf_Half                 shndx )
    {
        Elf_Word index = pStrWriter.add_string( str );
        return add_symbol( index, value, size, info, other, shndx );
    }

    //------------------------------------------------------------------------------
    // @brief Add a symbol to the section
    // @param pStrWriter String section accessor
    // @param str Name of the symbol
    // @param value Value of the symbol
    // @param size Size of the symbol
    // @param bind Binding of the symbol
    // @param type Type of the symbol
    // @param other Other attributes of the symbol
    // @param shndx Section index of the symbol
    // @return Index of the added symbol
    //------------------------------------------------------------------------------
    Elf_Word add_symbol( string_section_accessor& pStrWriter,
                         const char*              str,
                         Elf64_Addr               value,
                         Elf_Xword                size,
                         unsigned char            bind,
                         unsigned char            type,
                         unsigned char            other,
                         Elf_Half                 shndx )
    {
        return add_symbol( pStrWriter, str, value, size,
                           ELF_ST_INFO( bind, type ), other, shndx );
    }

    //------------------------------------------------------------------------------
    // @brief Arrange local symbols in the section
    // @param func Function to be called for each pair of symbols
    // @return Number of local symbols
    //------------------------------------------------------------------------------
    Elf_Xword arrange_local_symbols(
        std::function<void( Elf_Xword first, Elf_Xword second )> func =
            nullptr )
    {
        Elf_Xword nRet = 0;

        if ( elf_file.get_class() == ELFCLASS32 ) {
            nRet = generic_arrange_local_symbols<Elf32_Sym>( func );
        }
        else {
            nRet = generic_arrange_local_symbols<Elf64_Sym>( func );
        }

        return nRet;
    }

    //------------------------------------------------------------------------------
  private:
    //------------------------------------------------------------------------------
    // @brief Find the hash section
    //------------------------------------------------------------------------------
    void find_hash_section()
    {
        Elf_Half nSecNo = elf_file.sections.size();
        for ( Elf_Half i = 0; i < nSecNo; ++i ) {
            const section* sec = elf_file.sections[i];
            if ( sec->get_link() == symbol_section->get_index() &&
                 ( sec->get_type() == SHT_HASH ||
                   sec->get_type() == SHT_GNU_HASH ||
                   sec->get_type() == DT_GNU_HASH ) ) {
                hash_section       = sec;
                hash_section_index = i;
                break;
            }
        }
    }

    //------------------------------------------------------------------------------
    // @brief Get the index of the string table
    // @return Index of the string table
    //------------------------------------------------------------------------------
    Elf_Half get_string_table_index() const
    {
        return (Elf_Half)symbol_section->get_link();
    }

    //------------------------------------------------------------------------------
    // @brief Get the index of the hash table
    // @return Index of the hash table
    //------------------------------------------------------------------------------
    Elf_Half get_hash_table_index() const { return hash_section_index; }

    //------------------------------------------------------------------------------
    // @brief Lookup a symbol in the hash table
    // @param name Name of the symbol
    // @param value Value of the symbol
    // @param size Size of the symbol
    // @param bind Binding of the symbol
    // @param type Type of the symbol
    // @param section_index Section index of the symbol
    // @param other Other attributes of the symbol
    // @return True if the symbol is found, false otherwise
    //------------------------------------------------------------------------------
    bool hash_lookup( const std::string& name,
                      Elf64_Addr&        value,
                      Elf_Xword&         size,
                      unsigned char&     bind,
                      unsigned char&     type,
                      Elf_Half&          section_index,
                      unsigned char&     other ) const
    {
        bool        ret       = false;
        const auto& convertor = elf_file.get_convertor();

        Elf_Word nbucket = *(const Elf_Word*)hash_section->get_data();
        nbucket          = ( *convertor )( nbucket );
        Elf_Word nchain =
            *(const Elf_Word*)( hash_section->get_data() + sizeof( Elf_Word ) );
        nchain       = ( *convertor )( nchain );
        Elf_Word val = elf_hash( (const unsigned char*)name.c_str() );
        Elf_Word y =
            *(const Elf_Word*)( hash_section->get_data() +
                                ( 2 + val % nbucket ) * sizeof( Elf_Word ) );
        y = ( *convertor )( y );
        std::string str;
        get_symbol( y, str, value, size, bind, type, section_index, other );
        while ( str != name && STN_UNDEF != y && y < nchain ) {
            y = *(const Elf_Word*)( hash_section->get_data() +
                                    ( 2 + nbucket + y ) * sizeof( Elf_Word ) );
            y = ( *convertor )( y );
            get_symbol( y, str, value, size, bind, type, section_index, other );
        }

        if ( str == name ) {
            ret = true;
        }

        return ret;
    }

    //------------------------------------------------------------------------------
    // @brief Lookup a symbol in the GNU hash table
    // @param name Name of the symbol
    // @param value Value of the symbol
    // @param size Size of the symbol
    // @param bind Binding of the symbol
    // @param type Type of the symbol
    // @param section_index Section index of the symbol
    // @param other Other attributes of the symbol
    // @return True if the symbol is found, false otherwise
    //------------------------------------------------------------------------------
    template <class T>
    bool gnu_hash_lookup( const std::string& name,
                          Elf64_Addr&        value,
                          Elf_Xword&         size,
                          unsigned char&     bind,
                          unsigned char&     type,
                          Elf_Half&          section_index,
                          unsigned char&     other ) const
    {
        bool        ret       = false;
        const auto& convertor = elf_file.get_convertor();

        std::uint32_t nbuckets =
            *( (std::uint32_t*)hash_section->get_data() + 0 );
        std::uint32_t symoffset =
            *( (std::uint32_t*)hash_section->get_data() + 1 );
        std::uint32_t bloom_size =
            *( (std::uint32_t*)hash_section->get_data() + 2 );
        std::uint32_t bloom_shift =
            *( (std::uint32_t*)hash_section->get_data() + 3 );
        nbuckets    = ( *convertor )( nbuckets );
        symoffset   = ( *convertor )( symoffset );
        bloom_size  = ( *convertor )( bloom_size );
        bloom_shift = ( *convertor )( bloom_shift );

        auto* bloom_filter =
            (T*)( hash_section->get_data() + 4 * sizeof( std::uint32_t ) );

        std::uint32_t hash = elf_gnu_hash( (const unsigned char*)name.c_str() );
        std::uint32_t bloom_index = ( hash / ( 8 * sizeof( T ) ) ) % bloom_size;
        T             bloom_bits =
            ( (T)1 << ( hash % ( 8 * sizeof( T ) ) ) ) |
            ( (T)1 << ( ( hash >> bloom_shift ) % ( 8 * sizeof( T ) ) ) );

        if ( ( ( *convertor )( bloom_filter[bloom_index] ) & bloom_bits ) !=
             bloom_bits )
            return ret;

        std::uint32_t bucket  = hash % nbuckets;
        auto*         buckets = (std::uint32_t*)( hash_section->get_data() +
                                          4 * sizeof( std::uint32_t ) +
                                          bloom_size * sizeof( T ) );
        auto*         chains  = (std::uint32_t*)( hash_section->get_data() +
                                         4 * sizeof( std::uint32_t ) +
                                         bloom_size * sizeof( T ) +
                                         nbuckets * sizeof( std::uint32_t ) );

        if ( ( *convertor )( buckets[bucket] ) >= symoffset ) {
            std::uint32_t chain_index =
                ( *convertor )( buckets[bucket] ) - symoffset;
            std::uint32_t chain_hash = ( *convertor )( chains[chain_index] );
            std::string   symname;

            while ( true ) {
                if ( ( chain_hash >> 1 ) == ( hash >> 1 ) &&
                     get_symbol( chain_index + symoffset, symname, value, size,
                                 bind, type, section_index, other ) &&
                     ( name == symname ) ) {
                    ret = true;
                    break;
                }

                if ( chain_hash & 1 )
                    break;

                chain_hash = ( *convertor )( chains[++chain_index] );
            }
        }

        return ret;
    }

    //------------------------------------------------------------------------------
    // @brief Get the symbol at the specified index
    // @param index Index of the symbol
    // @return Pointer to the symbol
    //------------------------------------------------------------------------------
    template <class T> const T* generic_get_symbol_ptr( Elf_Xword index ) const
    {
        if ( 0 != symbol_section->get_data() && index < get_symbols_num() ) {
            if ( symbol_section->get_entry_size() < sizeof( T ) ) {
                return nullptr;
            }
            const auto* pSym = reinterpret_cast<const T*>(
                symbol_section->get_data() +
                index * symbol_section->get_entry_size() );

            return pSym;
        }

        return nullptr;
    }

    //------------------------------------------------------------------------------
    // @brief Search for a symbol in the section
    // @param match Function to be called for each symbol
    // @param idx Index of the found symbol
    // @return True if the symbol is found, false otherwise
    //------------------------------------------------------------------------------
    template <class T>
    bool generic_search_symbols( std::function<bool( const T* )> match,
                                 Elf_Xword&                      idx ) const
    {
        for ( Elf_Xword i = 0; i < get_symbols_num(); i++ ) {
            const T* symPtr = generic_get_symbol_ptr<T>( i );

            if ( symPtr == nullptr )
                return false;

            if ( match( symPtr ) ) {
                idx = i;
                return true;
            }
        }

        return false;
    }

    //------------------------------------------------------------------------------
    // @brief Get the symbol at the specified index
    // @param index Index of the symbol
    // @param name Name of the symbol
    // @param value Value of the symbol
    // @param size Size of the symbol
    // @param bind Binding of the symbol
    // @param type Type of the symbol
    // @param section_index Section index of the symbol
    // @param other Other attributes of the symbol
    // @return True if the symbol is found, false otherwise
    //------------------------------------------------------------------------------
    template <class T>
    bool generic_get_symbol( Elf_Xword      index,
                             std::string&   name,
                             Elf64_Addr&    value,
                             Elf_Xword&     size,
                             unsigned char& bind,
                             unsigned char& type,
                             Elf_Half&      section_index,
                             unsigned char& other ) const
    {
        bool ret = false;

        if ( nullptr != symbol_section->get_data() &&
             index < get_symbols_num() ) {
            const auto* pSym = reinterpret_cast<const T*>(
                symbol_section->get_data() +
                index * symbol_section->get_entry_size() );

            const auto& convertor = elf_file.get_convertor();

            section* string_section =
                elf_file.sections[get_string_table_index()];
            string_section_accessor str_reader( string_section );
            const char*             pStr =
                str_reader.get_string( ( *convertor )( pSym->st_name ) );
            if ( nullptr != pStr ) {
                name = pStr;
            }
            value         = ( *convertor )( pSym->st_value );
            size          = ( *convertor )( pSym->st_size );
            bind          = ELF_ST_BIND( pSym->st_info );
            type          = ELF_ST_TYPE( pSym->st_info );
            section_index = ( *convertor )( pSym->st_shndx );
            other         = pSym->st_other;

            ret = true;
        }

        return ret;
    }

    //------------------------------------------------------------------------------
    // @brief Add a symbol to the section
    // @param name Name of the symbol
    // @param value Value of the symbol
    // @param size Size of the symbol
    // @param info Info of the symbol
    // @param other Other attributes of the symbol
    // @param shndx Section index of the symbol
    // @return Index of the added symbol
    //------------------------------------------------------------------------------
    template <class T>
    Elf_Word generic_add_symbol( Elf_Word      name,
                                 Elf64_Addr    value,
                                 Elf_Xword     size,
                                 unsigned char info,
                                 unsigned char other,
                                 Elf_Half      shndx )
    {
        const auto& convertor = elf_file.get_convertor();

        T entry;
        entry.st_name  = ( *convertor )( name );
        entry.st_value = decltype( entry.st_value )( value );
        entry.st_value = ( *convertor )( entry.st_value );
        entry.st_size  = decltype( entry.st_size )( size );
        entry.st_size  = ( *convertor )( entry.st_size );
        entry.st_info  = ( *convertor )( info );
        entry.st_other = ( *convertor )( other );
        entry.st_shndx = ( *convertor )( shndx );

        symbol_section->append_data( reinterpret_cast<char*>( &entry ),
                                     sizeof( entry ) );

        Elf_Word nRet =
            Elf_Word( symbol_section->get_size() / sizeof( entry ) - 1 );

        return nRet;
    }

    //------------------------------------------------------------------------------
    // @brief Arrange local symbols in the section
    // @param func Function to be called for each pair of symbols
    // @return Number of local symbols
    //------------------------------------------------------------------------------
    template <class T>
    Elf_Xword generic_arrange_local_symbols(
        std::function<void( Elf_Xword first, Elf_Xword second )> func )
    {
        const auto& convertor = elf_file.get_convertor();

        Elf_Word first_not_local =
            1; // Skip the first entry. It is always NOTYPE
        Elf_Xword current = 0;
        Elf_Xword count   = get_symbols_num();

        while ( true ) {
            T* p1 = nullptr;
            T* p2 = nullptr;

            while ( first_not_local < count ) {
                p1 = const_cast<T*>(
                    generic_get_symbol_ptr<T>( first_not_local ) );
                if ( ELF_ST_BIND( ( *convertor )( p1->st_info ) ) != STB_LOCAL )
                    break;
                ++first_not_local;
            }

            current = first_not_local + 1;
            while ( current < count ) {
                p2 = const_cast<T*>( generic_get_symbol_ptr<T>( current ) );
                if ( ELF_ST_BIND( ( *convertor )( p2->st_info ) ) == STB_LOCAL )
                    break;
                ++current;
            }

            if ( first_not_local < count && current < count ) {
                if ( func )
                    func( first_not_local, current );

                std::swap( *p1, *p2 );
            }
            else {
                // Update 'info' field of the section
                symbol_section->set_info( first_not_local );
                break;
            }
        }

        return first_not_local;
    }

    //------------------------------------------------------------------------------
  private:
    const elfio&   elf_file;                ///< Reference to the ELF file
    S*             symbol_section;          ///< Pointer to the symbol section
    Elf_Half       hash_section_index{ 0 }; ///< Index of the hash section
    const section* hash_section{ nullptr }; ///< Pointer to the hash section
};

using symbol_section_accessor = symbol_section_accessor_template<section>;
using const_symbol_section_accessor =
    symbol_section_accessor_template<const section>;

} // namespace ELFIO

#endif // ELFIO_SYMBOLS_HPP
