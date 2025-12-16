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

#ifndef ELFIO_DYNAMIC_HPP
#define ELFIO_DYNAMIC_HPP

#include <algorithm>

namespace ELFIO {

//------------------------------------------------------------------------------
// Template class for accessing dynamic sections
template <class S> class dynamic_section_accessor_template
{
  public:
    //------------------------------------------------------------------------------
    // Constructor
    explicit dynamic_section_accessor_template( const elfio& elf_file,
                                                S*           section )
        : elf_file( elf_file ), dynamic_section( section ), entries_num( 0 )
    {
    }

    //------------------------------------------------------------------------------
    // Returns the number of entries in the dynamic section
    Elf_Xword get_entries_num() const
    {
        size_t needed_entry_size = -1;
        if ( elf_file.get_class() == ELFCLASS32 ) {
            needed_entry_size = sizeof( Elf32_Dyn );
        }
        else {
            needed_entry_size = sizeof( Elf64_Dyn );
        }

        if ( ( 0 == entries_num ) &&
             ( 0 != dynamic_section->get_entry_size() &&
               dynamic_section->get_entry_size() >= needed_entry_size ) ) {
            entries_num =
                dynamic_section->get_size() / dynamic_section->get_entry_size();
            Elf_Xword   i;
            Elf_Xword   tag   = DT_NULL;
            Elf_Xword   value = 0;
            std::string str;
            for ( i = 0; i < entries_num; i++ ) {
                get_entry( i, tag, value, str );
                if ( tag == DT_NULL )
                    break;
            }
            entries_num = std::min<Elf_Xword>( entries_num, i + 1 );
        }

        return entries_num;
    }

    //------------------------------------------------------------------------------
    // Retrieves an entry from the dynamic section
    bool get_entry( Elf_Xword    index,
                    Elf_Xword&   tag,
                    Elf_Xword&   value,
                    std::string& str ) const
    {
        if ( index >= get_entries_num() ) { // Is index valid
            return false;
        }

        if ( elf_file.get_class() == ELFCLASS32 ) {
            generic_get_entry_dyn<Elf32_Dyn>( index, tag, value );
        }
        else {
            generic_get_entry_dyn<Elf64_Dyn>( index, tag, value );
        }

        // If the tag has a string table reference - prepare the string
        if ( tag == DT_NEEDED || tag == DT_SONAME || tag == DT_RPATH ||
             tag == DT_RUNPATH ) {
            string_section_accessor strsec(
                elf_file.sections[get_string_table_index()] );
            const char* result = strsec.get_string( (Elf_Word)value );
            if ( nullptr == result ) {
                str.clear();
                return false;
            }
            str = result;
        }
        else {
            str.clear();
        }

        return true;
    }

    //------------------------------------------------------------------------------
    // Adds an entry to the dynamic section
    void add_entry( Elf_Xword tag, Elf_Xword value )
    {
        if ( elf_file.get_class() == ELFCLASS32 ) {
            generic_add_entry_dyn<Elf32_Dyn>( tag, value );
        }
        else {
            generic_add_entry_dyn<Elf64_Dyn>( tag, value );
        }
    }

    //------------------------------------------------------------------------------
    // Adds an entry with a string value to the dynamic section
    void add_entry( Elf_Xword tag, const std::string& str )
    {
        string_section_accessor strsec(
            elf_file.sections[get_string_table_index()] );
        Elf_Xword value = strsec.add_string( str );
        add_entry( tag, value );
    }

  private:
    //------------------------------------------------------------------------------
    // Returns the index of the string table
    Elf_Half get_string_table_index() const
    {
        return (Elf_Half)dynamic_section->get_link();
    }

    //------------------------------------------------------------------------------
    // Retrieves a generic entry from the dynamic section
    template <class T>
    void generic_get_entry_dyn( Elf_Xword  index,
                                Elf_Xword& tag,
                                Elf_Xword& value ) const
    {
        const auto& convertor = elf_file.get_convertor();

        // Check unusual case when dynamic section has no data
        if ( dynamic_section->get_data() == nullptr ||
             dynamic_section->get_entry_size() < sizeof( T ) ) {
            tag   = DT_NULL;
            value = 0;
            return;
        }

        // Check for integer overflow in size calculation
        if ( index > ( dynamic_section->get_size() /
                       dynamic_section->get_entry_size() ) -
                         1 ) {
            tag   = DT_NULL;
            value = 0;
            return;
        }

        // Check for integer overflow in pointer arithmetic
        Elf_Xword offset = index * dynamic_section->get_entry_size();
        if ( offset > dynamic_section->get_size() - sizeof( T ) ) {
            tag   = DT_NULL;
            value = 0;
            return;
        }

        const T* pEntry =
            reinterpret_cast<const T*>( dynamic_section->get_data() + offset );
        tag = ( *convertor )( pEntry->d_tag );
        switch ( tag ) {
        case DT_NULL:
        case DT_SYMBOLIC:
        case DT_TEXTREL:
        case DT_BIND_NOW:
            value = 0;
            break;
        case DT_NEEDED:
        case DT_PLTRELSZ:
        case DT_RELASZ:
        case DT_RELAENT:
        case DT_STRSZ:
        case DT_SYMENT:
        case DT_SONAME:
        case DT_RPATH:
        case DT_RELSZ:
        case DT_RELENT:
        case DT_PLTREL:
        case DT_INIT_ARRAYSZ:
        case DT_FINI_ARRAYSZ:
        case DT_RUNPATH:
        case DT_FLAGS:
        case DT_PREINIT_ARRAYSZ:
            value = ( *convertor )( pEntry->d_un.d_val );
            break;
        case DT_PLTGOT:
        case DT_HASH:
        case DT_STRTAB:
        case DT_SYMTAB:
        case DT_RELA:
        case DT_INIT:
        case DT_FINI:
        case DT_REL:
        case DT_DEBUG:
        case DT_JMPREL:
        case DT_INIT_ARRAY:
        case DT_FINI_ARRAY:
        case DT_PREINIT_ARRAY:
        default:
            value = ( *convertor )( pEntry->d_un.d_ptr );
            break;
        }
    }

    //------------------------------------------------------------------------------
    // Adds a generic entry to the dynamic section
    template <class T>
    void generic_add_entry_dyn( Elf_Xword tag, Elf_Xword value )
    {
        const auto& convertor = elf_file.get_convertor();

        T entry;

        switch ( tag ) {
        case DT_NULL:
        case DT_SYMBOLIC:
        case DT_TEXTREL:
        case DT_BIND_NOW:
            entry.d_un.d_val =
                ( *convertor )( decltype( entry.d_un.d_val )( 0 ) );
            break;
        case DT_NEEDED:
        case DT_PLTRELSZ:
        case DT_RELASZ:
        case DT_RELAENT:
        case DT_STRSZ:
        case DT_SYMENT:
        case DT_SONAME:
        case DT_RPATH:
        case DT_RELSZ:
        case DT_RELENT:
        case DT_PLTREL:
        case DT_INIT_ARRAYSZ:
        case DT_FINI_ARRAYSZ:
        case DT_RUNPATH:
        case DT_FLAGS:
        case DT_PREINIT_ARRAYSZ:
            entry.d_un.d_val =
                ( *convertor )( decltype( entry.d_un.d_val )( value ) );
            break;
        case DT_PLTGOT:
        case DT_HASH:
        case DT_STRTAB:
        case DT_SYMTAB:
        case DT_RELA:
        case DT_INIT:
        case DT_FINI:
        case DT_REL:
        case DT_DEBUG:
        case DT_JMPREL:
        case DT_INIT_ARRAY:
        case DT_FINI_ARRAY:
        case DT_PREINIT_ARRAY:
        default:
            entry.d_un.d_ptr =
                ( *convertor )( decltype( entry.d_un.d_val )( value ) );
            break;
        }

        entry.d_tag = ( *convertor )( decltype( entry.d_tag )( tag ) );

        dynamic_section->append_data( reinterpret_cast<char*>( &entry ),
                                      sizeof( entry ) );
    }

  private:
    // Reference to the ELF file
    const elfio& elf_file;
    // Pointer to the dynamic section
    S* dynamic_section;
    // Number of entries in the dynamic section
    mutable Elf_Xword entries_num;
};

// Type aliases for dynamic section accessors
using dynamic_section_accessor = dynamic_section_accessor_template<section>;
using const_dynamic_section_accessor =
    dynamic_section_accessor_template<const section>;

} // namespace ELFIO

#endif // ELFIO_DYNAMIC_HPP
