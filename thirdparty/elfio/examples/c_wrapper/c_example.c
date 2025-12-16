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

#include <stdio.h>
#include <string.h>

#include <elfio/elf_types.hpp>
#include "elfio_c_wrapper.h"

int main( int argc, char* argv[] )
{
    pelfio_t pelfio = elfio_new();
    bool     ret;

    if ( argc == 1 )
        ret = elfio_load( pelfio, argv[0] );
    else
        ret = elfio_load( pelfio, argv[1] );

    if ( !ret ) {
        printf( "Can't load ELF file\n" );
        return 1;
    }

    char msg[128];
    ret = elfio_validate( pelfio, msg, 128 );

    if ( !ret ) {
        printf( "Validation errors:\n" );
        printf( "%s\n", msg );
        return 2;
    }

    //-----------------------------------------------------------------------------
    // elfio
    //-----------------------------------------------------------------------------
    printf( "Header size   : %d\n", elfio_get_header_size( pelfio ) );
    printf( "Version       : %d\n", elfio_get_version( pelfio ) );
    printf( "Section Entry : %d\n", elfio_get_section_entry_size( pelfio ) );
    printf( "Segment Entry : %d\n", elfio_get_segment_entry_size( pelfio ) );

    /* Uncomment a printf block of the interest */

    //-----------------------------------------------------------------------------
    // section
    //-----------------------------------------------------------------------------
    int secno = elfio_get_sections_num( pelfio );
    printf( "Sections No   : %d\n", secno );

    for ( int i = 0; i < secno; i++ ) {
        psection_t psection = elfio_get_section_by_index( pelfio, i );
        char       buff[128];
        elfio_section_get_name( psection, buff, 100 );
        // printf( "    [%02d] %s\n", i, buff );
        // printf( "        %08lx : %08lx\n",
        //         elfio_section_get_address( psection ),
        //         elfio_section_get_size( psection ) );
    }

    //-----------------------------------------------------------------------------
    // segment
    //-----------------------------------------------------------------------------
    int segno = elfio_get_segments_num( pelfio );
    printf( "Segments No   : %d\n", segno );

    for ( int i = 0; i < segno; i++ ) {
        psegment_t psegment = elfio_get_segment_by_index( pelfio, i );
        elfio_segment_get_file_size( psegment );
        // printf( "    [%02d] %08lx : %08lx : %08lx\n", i,
        //         elfio_segment_get_virtual_address( psegment ),
        //         elfio_segment_get_memory_size( psegment ),
        //         elfio_segment_get_file_size( psegment ) );
    }

    //-----------------------------------------------------------------------------
    // symbol
    //-----------------------------------------------------------------------------
    psection_t psection = elfio_get_section_by_name( pelfio, ".symtab" );
    psymbol_t  psymbols = elfio_symbol_section_accessor_new( pelfio, psection );
    Elf_Xword  symno    = elfio_symbol_get_symbols_num( psymbols );
    for ( int i = 0; i < symno; i++ ) {
        char          name[128];
        Elf64_Addr    value;
        Elf_Xword     size;
        unsigned char bind;
        unsigned char type;
        Elf_Half      section_index;
        unsigned char other;
        elfio_symbol_get_symbol( psymbols, i, name, 128, &value, &size, &bind,
                                 &type, &section_index, &other );
        // printf( "[%4d] %10lu, %4lu %s\n", i, value, size, name );
    }
    elfio_symbol_section_accessor_delete( psymbols );

    //-----------------------------------------------------------------------------
    // relocation
    //-----------------------------------------------------------------------------
    psection = elfio_get_section_by_name( pelfio, ".rela.dyn" );
    prelocation_t preloc =
        elfio_relocation_section_accessor_new( pelfio, psection );
    Elf_Xword relno = elfio_relocation_get_entries_num( preloc );
    for ( int i = 0; i < relno; i++ ) {
        Elf64_Addr offset;
        Elf_Word   symbol;
        Elf_Word   type;
        Elf_Sxword addend;
        elfio_relocation_get_entry( preloc, i, &offset, &symbol, &type,
                                    &addend );
        // printf( "[%4d] %16lx, %08x %08x %16lx\n", i, offset, symbol, type, addend );
    }
    elfio_relocation_section_accessor_delete( preloc );

    //-----------------------------------------------------------------------------
    // string
    //-----------------------------------------------------------------------------
    psection            = elfio_get_section_by_name( pelfio, ".strtab" );
    pstring_t   pstring = elfio_string_section_accessor_new( psection );
    Elf_Word    pos     = 0;
    const char* str     = elfio_string_get_string( pstring, pos );
    while ( str ) {
        pos += (Elf_Word)strlen( str ) + 1;
        str = elfio_string_get_string( pstring, pos );
        // printf( "%s\n", str );
    }
    elfio_string_section_accessor_delete( pstring );

    //-----------------------------------------------------------------------------
    // note
    //-----------------------------------------------------------------------------
    psection       = elfio_get_section_by_name( pelfio, ".note.gnu.build-id" );
    pnote_t pnote  = elfio_note_section_accessor_new( pelfio, psection );
    int     noteno = elfio_note_get_notes_num( pnote );
    for ( int i = 0; i < noteno; i++ ) {
        Elf_Word type;
        char     name[128];
        int      name_len = 128;
        char*    desc;
        Elf_Word descSize = 128;
        elfio_note_get_note( pnote, i, &type, name, name_len, (void**)&desc,
                             &descSize );
        // printf( "[%4d] %s %08x\n", i, name, descSize );
    }
    elfio_note_section_accessor_delete( pnote );

    //-----------------------------------------------------------------------------
    // dynamic
    //-----------------------------------------------------------------------------
    psection = elfio_get_section_by_name( pelfio, ".dynamic" );
    pdynamic_t pdynamic =
        elfio_dynamic_section_accessor_new( pelfio, psection );
    Elf_Xword dynno = elfio_dynamic_get_entries_num( pdynamic );
    for ( int i = 0; i < dynno; i++ ) {
        Elf_Xword tag;
        Elf_Xword value;
        char      str[128];
        elfio_dynamic_get_entry( pdynamic, i, &tag, &value, str, 128 );
        // printf( "[%4d] %16lx %16lx %s\n", i, tag, value, str );
    }
    elfio_dynamic_section_accessor_delete( pdynamic );

    //-----------------------------------------------------------------------------
    // array
    //-----------------------------------------------------------------------------
    psection = elfio_get_section_by_name( pelfio, ".init_array" );
    if ( psection != 0 ) {
        parray_t  parray = elfio_array_section_accessor_new( pelfio, psection );
        Elf_Xword arrno  = elfio_array_get_entries_num( parray );
        for ( int i = 0; i < arrno; i++ ) {
            Elf64_Addr addr;
            elfio_array_get_entry( parray, i, &addr );
            // printf( "[%4d] %16lx\n", i, addr );
        }
        elfio_array_section_accessor_delete( parray );
    }

    elfio_delete( pelfio );

    return 0;
}
