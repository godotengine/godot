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

#ifndef ELFIO_C_WRAPPER_H
#define ELFIO_C_WRAPPER_H

#define ELFIO_C_HEADER_ACCESS_GET( TYPE, FNAME ) \
    TYPE elfio_get_##FNAME( pelfio_t pelfio );

#define ELFIO_C_HEADER_ACCESS_GET_SET( TYPE, FNAME ) \
    TYPE elfio_get_##FNAME( pelfio_t pelfio );       \
    void elfio_set_##FNAME( pelfio_t pelfio, TYPE val );

#define ELFIO_C_HEADER_ACCESS_GET_IMPL( TYPE, FNAME ) \
    TYPE elfio_get_##FNAME( pelfio_t pelfio ) { return pelfio->get_##FNAME(); }

#define ELFIO_C_HEADER_ACCESS_GET_SET_IMPL( TYPE, FNAME ) \
    TYPE elfio_get_##FNAME( pelfio_t pelfio )             \
    {                                                     \
        return pelfio->get_##FNAME();                     \
    }                                                     \
    void elfio_set_##FNAME( pelfio_t pelfio, TYPE val )   \
    {                                                     \
        pelfio->set_##FNAME( val );                       \
    }

#define ELFIO_C_GET_ACCESS_IMPL( CLASS, TYPE, NAME )         \
    TYPE elfio_##CLASS##_get_##NAME( p##CLASS##_t p##CLASS ) \
    {                                                        \
        return p##CLASS->get_##NAME();                       \
    }

#define ELFIO_C_SET_ACCESS_IMPL( CLASS, TYPE, NAME )                     \
    void elfio_##CLASS##_set_##NAME( p##CLASS##_t p##CLASS, TYPE value ) \
    {                                                                    \
        p##CLASS->set_##NAME( value );                                   \
    }

#define ELFIO_C_GET_SET_ACCESS_IMPL( CLASS, TYPE, NAME )                 \
    TYPE elfio_##CLASS##_get_##NAME( p##CLASS##_t p##CLASS )             \
    {                                                                    \
        return p##CLASS->get_##NAME();                                   \
    }                                                                    \
    void elfio_##CLASS##_set_##NAME( p##CLASS##_t p##CLASS, TYPE value ) \
    {                                                                    \
        p##CLASS->set_##NAME( value );                                   \
    }

#define ELFIO_C_GET_ACCESS( CLASS, TYPE, NAME ) \
    TYPE elfio_##CLASS##_get_##NAME( p##CLASS##_t p##CLASS );

#define ELFIO_C_SET_ACCESS( CLASS, TYPE, NAME ) \
    void elfio_##CLASS##_set_##NAME( p##CLASS##_t p##CLASS, TYPE value );

#define ELFIO_C_GET_SET_ACCESS( CLASS, TYPE, NAME )           \
    TYPE elfio_##CLASS##_get_##NAME( p##CLASS##_t p##CLASS ); \
    void elfio_##CLASS##_set_##NAME( p##CLASS##_t p##CLASS, TYPE value )

#ifdef __cplusplus
typedef ELFIO::elfio*                       pelfio_t;
typedef ELFIO::section*                     psection_t;
typedef ELFIO::segment*                     psegment_t;
typedef ELFIO::symbol_section_accessor*     psymbol_t;
typedef ELFIO::relocation_section_accessor* prelocation_t;
typedef ELFIO::string_section_accessor*     pstring_t;
typedef ELFIO::note_section_accessor*       pnote_t;
typedef ELFIO::modinfo_section_accessor*    pmodinfo_t;
typedef ELFIO::dynamic_section_accessor*    pdynamic_t;
typedef ELFIO::array_section_accessor<Elf64_Word>* parray_t;

extern "C"
{
#else
typedef void* pelfio_t;
typedef void* psection_t;
typedef void* psegment_t;
typedef void* psymbol_t;
typedef void* prelocation_t;
typedef void* pstring_t;
typedef void* pnote_t;
typedef void* pmodinfo_t;
typedef void* pdynamic_t;
typedef void* parray_t;
typedef int bool;
#endif

    //-----------------------------------------------------------------------------
    // elfio
    //-----------------------------------------------------------------------------
    pelfio_t elfio_new();
    void     elfio_delete( pelfio_t pelfio );
    void     elfio_create( pelfio_t      pelfio,
                           unsigned char file_class,
                           unsigned char encoding );
    bool     elfio_load( pelfio_t pelfio, const char* file_name );
    bool     elfio_save( pelfio_t pelfio, const char* file_name );
    ELFIO_C_HEADER_ACCESS_GET( unsigned char, class );
    ELFIO_C_HEADER_ACCESS_GET( unsigned char, elf_version );
    ELFIO_C_HEADER_ACCESS_GET( unsigned char, encoding );
    ELFIO_C_HEADER_ACCESS_GET( Elf_Word, version );
    ELFIO_C_HEADER_ACCESS_GET( Elf_Half, header_size );
    ELFIO_C_HEADER_ACCESS_GET( Elf_Half, section_entry_size );
    ELFIO_C_HEADER_ACCESS_GET( Elf_Half, segment_entry_size );
    ELFIO_C_HEADER_ACCESS_GET_SET( unsigned char, os_abi );
    ELFIO_C_HEADER_ACCESS_GET_SET( unsigned char, abi_version );
    ELFIO_C_HEADER_ACCESS_GET_SET( Elf_Half, type );
    ELFIO_C_HEADER_ACCESS_GET_SET( Elf_Half, machine );
    ELFIO_C_HEADER_ACCESS_GET_SET( Elf_Word, flags );
    ELFIO_C_HEADER_ACCESS_GET_SET( Elf64_Addr, entry );
    ELFIO_C_HEADER_ACCESS_GET_SET( Elf64_Off, sections_offset );
    ELFIO_C_HEADER_ACCESS_GET_SET( Elf64_Off, segments_offset );
    ELFIO_C_HEADER_ACCESS_GET_SET( Elf_Half, section_name_str_index );
    Elf_Half   elfio_get_sections_num( pelfio_t pelfio );
    psection_t elfio_get_section_by_index( pelfio_t pelfio, int index );
    psection_t elfio_get_section_by_name( pelfio_t pelfio, char* name );
    psection_t elfio_add_section( pelfio_t pelfio, char* name );
    Elf_Half   elfio_get_segments_num( pelfio_t pelfio );
    psegment_t elfio_get_segment_by_index( pelfio_t pelfio, int index );
    psegment_t elfio_add_segment( pelfio_t pelfio );
    bool       elfio_validate( pelfio_t pelfio, char* msg, int msg_len );

    //-----------------------------------------------------------------------------
    // section
    //-----------------------------------------------------------------------------
    ELFIO_C_GET_ACCESS( section, Elf_Half, index );
    ELFIO_C_GET_SET_ACCESS( section, Elf_Word, type );
    ELFIO_C_GET_SET_ACCESS( section, Elf_Xword, flags );
    ELFIO_C_GET_SET_ACCESS( section, Elf_Word, info );
    ELFIO_C_GET_SET_ACCESS( section, Elf_Word, link );
    ELFIO_C_GET_SET_ACCESS( section, Elf_Xword, addr_align );
    ELFIO_C_GET_SET_ACCESS( section, Elf_Xword, entry_size );
    ELFIO_C_GET_SET_ACCESS( section, Elf64_Addr, address );
    ELFIO_C_GET_SET_ACCESS( section, Elf_Xword, size );
    ELFIO_C_GET_SET_ACCESS( section, Elf_Word, name_string_offset );
    ELFIO_C_GET_ACCESS( section, Elf64_Off, offset );
    void  elfio_section_get_name( psection_t psection, char* buffer, int len );
    void  elfio_section_set_name( psection_t psection, char* buffer );
    char* elfio_section_get_data( psection_t psection );
    void  elfio_section_set_data( psection_t  psection,
                                  const char* pData,
                                  Elf_Word    size );
    void  elfio_section_append_data( psection_t  psection,
                                     const char* pData,
                                     Elf_Word    size );

    //-----------------------------------------------------------------------------
    // segment
    //-----------------------------------------------------------------------------
    ELFIO_C_GET_ACCESS( segment, Elf_Half, index );
    ELFIO_C_GET_SET_ACCESS( segment, Elf_Word, type );
    ELFIO_C_GET_SET_ACCESS( segment, Elf_Word, flags );
    ELFIO_C_GET_SET_ACCESS( segment, Elf_Xword, align );
    ELFIO_C_GET_SET_ACCESS( segment, Elf_Xword, memory_size );
    ELFIO_C_GET_SET_ACCESS( segment, Elf64_Addr, virtual_address );
    ELFIO_C_GET_SET_ACCESS( segment, Elf64_Addr, physical_address );
    ELFIO_C_GET_SET_ACCESS( segment, Elf_Xword, file_size );
    ELFIO_C_GET_ACCESS( segment, Elf64_Off, offset );
    char*    elfio_segment_get_data( psegment_t psegment );
    Elf_Half elfio_segment_add_section_index( psegment_t psegment,
                                              Elf_Half   index,
                                              Elf_Xword  addr_align );
    Elf_Half elfio_segment_get_sections_num( psegment_t psegment );
    Elf_Half elfio_segment_get_section_index_at( psegment_t psegment,
                                                 Elf_Half   num );
    bool     elfio_segment_is_offset_initialized( psegment_t psegment );

    //-----------------------------------------------------------------------------
    // symbol
    //-----------------------------------------------------------------------------
    psymbol_t elfio_symbol_section_accessor_new( pelfio_t   pelfio,
                                                 psection_t psection );
    void      elfio_symbol_section_accessor_delete( psymbol_t psymbol );
    Elf_Xword elfio_symbol_get_symbols_num( psymbol_t psymbol );
    bool      elfio_symbol_get_symbol( psymbol_t      psymbol,
                                       Elf_Xword      index,
                                       char*          name,
                                       int            name_len,
                                       Elf64_Addr*    value,
                                       Elf_Xword*     size,
                                       unsigned char* bind,
                                       unsigned char* type,
                                       Elf_Half*      section_index,
                                       unsigned char* other );
    Elf_Word  elfio_symbol_add_symbol( psymbol_t     psymbol,
                                       Elf_Word      name,
                                       Elf64_Addr    value,
                                       Elf_Xword     size,
                                       unsigned char info,
                                       unsigned char other,
                                       Elf_Half      shndx );
    Elf_Xword elfio_symbol_arrange_local_symbols(
        psymbol_t psymbol,
        void ( *func )( Elf_Xword first, Elf_Xword second ) );

    //-----------------------------------------------------------------------------
    // relocation
    //-----------------------------------------------------------------------------
    prelocation_t elfio_relocation_section_accessor_new( pelfio_t   pelfio,
                                                         psection_t psection );
    void elfio_relocation_section_accessor_delete( prelocation_t prelocation );
    Elf_Xword elfio_relocation_get_entries_num( prelocation_t prelocation );
    bool      elfio_relocation_get_entry( prelocation_t prelocation,
                                          Elf_Xword     index,
                                          Elf64_Addr*   offset,
                                          Elf_Word*     symbol,
                                          Elf_Word*     type,
                                          Elf_Sxword*   addend );
    bool      elfio_relocation_set_entry( prelocation_t prelocation,
                                          Elf_Xword     index,
                                          Elf64_Addr    offset,
                                          Elf_Word      symbol,
                                          Elf_Word      type,
                                          Elf_Sxword    addend );
    void      elfio_relocation_add_entry( prelocation_t prelocation,
                                          Elf64_Addr    offset,
                                          Elf_Word      symbol,
                                          unsigned char type,
                                          Elf_Sxword    addend );
    void      elfio_relocation_swap_symbols( prelocation_t prelocation,
                                             Elf_Xword     first,
                                             Elf_Xword     second );

    //-----------------------------------------------------------------------------
    // string
    //-----------------------------------------------------------------------------
    pstring_t   elfio_string_section_accessor_new( psection_t psection );
    void        elfio_string_section_accessor_delete( pstring_t pstring );
    const char* elfio_string_get_string( pstring_t pstring, Elf_Word index );
    Elf_Word    elfio_string_add_string( pstring_t pstring, char* str );

    //-----------------------------------------------------------------------------
    // note
    //-----------------------------------------------------------------------------
    pnote_t  elfio_note_section_accessor_new( pelfio_t   pelfio,
                                              psection_t psection );
    void     elfio_note_section_accessor_delete( pnote_t pstring );
    Elf_Word elfio_note_get_notes_num( pnote_t pnote );
    bool     elfio_note_get_note( pnote_t   pnote,
                                  Elf_Word  index,
                                  Elf_Word* type,
                                  char*     name,
                                  int       name_len,
                                  void**    desc,
                                  Elf_Word* descSize );
    void     elfio_note_add_note( pnote_t     pnote,
                                  Elf_Word    type,
                                  const char* name,
                                  const void* desc,
                                  Elf_Word    descSize );

    //-----------------------------------------------------------------------------
    // modinfo
    //-----------------------------------------------------------------------------
    pmodinfo_t elfio_modinfo_section_accessor_new( psection_t psection );
    void       elfio_modinfo_section_accessor_delete( pmodinfo_t pmodinfo );
    Elf_Word   elfio_modinfo_get_attribute_num( pmodinfo_t pmodinfo );
    bool       elfio_modinfo_get_attribute( pmodinfo_t pmodinfo,
                                            Elf_Word   no,
                                            char*      field,
                                            int        field_len,
                                            char*      value,
                                            int        value_len );
    bool       elfio_modinfo_get_attribute_by_name( pmodinfo_t pmodinfo,
                                                    char*      field_name,
                                                    char*      value,
                                                    int        value_len );
    Elf_Word   elfio_modinfo_add_attribute( pmodinfo_t pmodinfo,
                                            char*      field,
                                            char*      value );

    //-----------------------------------------------------------------------------
    // dynamic
    //-----------------------------------------------------------------------------
    pdynamic_t elfio_dynamic_section_accessor_new( pelfio_t   pelfio,
                                                   psection_t psection );
    void       elfio_dynamic_section_accessor_delete( pdynamic_t pdynamic );
    Elf_Xword  elfio_dynamic_get_entries_num( pdynamic_t pdynamic );
    bool       elfio_dynamic_get_entry( pdynamic_t pdynamic,
                                        Elf_Xword  index,
                                        Elf_Xword* tag,
                                        Elf_Xword* value,
                                        char*      str,
                                        int        str_len );
    void       elfio_dynamic_add_entry( pdynamic_t pdynamic,
                                        Elf_Xword  tag,
                                        Elf_Xword  value );

    //-----------------------------------------------------------------------------
    // array
    //-----------------------------------------------------------------------------
    parray_t  elfio_array_section_accessor_new( pelfio_t   pelfio,
                                                psection_t psection );
    void      elfio_array_section_accessor_delete( parray_t parray );
    Elf_Xword elfio_array_get_entries_num( parray_t parray );
    bool      elfio_array_get_entry( parray_t    parray,
                                     Elf_Xword   index,
                                     Elf64_Addr* paddress );
    void      elfio_array_add_entry( parray_t parray, Elf64_Addr address );

#ifdef __cplusplus
}
#endif

#endif // ELFIO_C_WRAPPER_H
