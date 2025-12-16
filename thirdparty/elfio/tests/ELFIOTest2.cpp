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

#ifdef _MSC_VER
#define _SCL_SECURE_NO_WARNINGS
#define ELFIO_NO_INTTYPES
#endif

#include <gtest/gtest.h>
#include <elfio/elfio.hpp>

using namespace ELFIO;

////////////////////////////////////////////////////////////////////////////////
TEST( ELFIOTest, modinfo_read )
{
    elfio reader;
    ASSERT_EQ( reader.load( "elf_examples/zavl.ko" ), true );

    section* modinfo_sec = reader.sections[".modinfo"];
    ASSERT_NE( modinfo_sec, nullptr );

    const_modinfo_section_accessor modinfo( modinfo_sec );
    ASSERT_EQ( modinfo.get_attribute_num(), (Elf_Word)9 );

    struct
    {
        std::string field;
        std::string value;
    } attributes[] = { { "version", "0.8.3-1ubuntu12.1" },
                       { "license", "CDDL" },
                       { "author", "OpenZFS on Linux" },
                       { "description", "Generic AVL tree implementation" },
                       { "srcversion", "98E85778E754CF75DEF9E8E" },
                       { "depends", "spl" },
                       { "retpoline", "Y" },
                       { "name", "zavl" },
                       { "vermagic", "5.4.0-42-generic SMP mod_unload " } };

    for ( auto i = 0; i < sizeof( attributes ) / sizeof( attributes[0] );
          i++ ) {
        std::string field;
        std::string value;
        modinfo.get_attribute( i, field, value );

        EXPECT_EQ( field, attributes[i].field );
        EXPECT_EQ( value, attributes[i].value );
    }

    for ( auto i = 0; i < sizeof( attributes ) / sizeof( attributes[0] );
          i++ ) {
        std::string field = attributes[i].field;
        std::string value;
        modinfo.get_attribute( field, value );

        EXPECT_EQ( value, attributes[i].value );
    }
}

////////////////////////////////////////////////////////////////////////////////
TEST( ELFIOTest, modinfo_write )
{
    elfio writer;
    ASSERT_EQ( writer.load( "elf_examples/zavl.ko" ), true );

    section* modinfo_sec = writer.sections[".modinfo"];
    ASSERT_NE( modinfo_sec, nullptr );

    modinfo_section_accessor modinfo( modinfo_sec );
    ASSERT_EQ( modinfo.get_attribute_num(), (Elf_Word)9 );

    modinfo.add_attribute( "test1", "value1" );
    modinfo.add_attribute( "test2", "value2" );

    ASSERT_EQ( modinfo.get_attribute_num(), (Elf_Word)11 );

    ASSERT_EQ( writer.save( "elf_examples/zavl_gen.ko" ), true );

    elfio reader;
    ASSERT_EQ( reader.load( "elf_examples/zavl_gen.ko" ), true );

    modinfo_sec = reader.sections[".modinfo"];
    ASSERT_NE( modinfo_sec, nullptr );

    const_modinfo_section_accessor modinfo1( modinfo_sec );
    ASSERT_EQ( modinfo1.get_attribute_num(), (Elf_Word)11 );

    struct
    {
        std::string field;
        std::string value;
    } attributes[] = { { "version", "0.8.3-1ubuntu12.1" },
                       { "license", "CDDL" },
                       { "author", "OpenZFS on Linux" },
                       { "description", "Generic AVL tree implementation" },
                       { "srcversion", "98E85778E754CF75DEF9E8E" },
                       { "depends", "spl" },
                       { "retpoline", "Y" },
                       { "name", "zavl" },
                       { "vermagic", "5.4.0-42-generic SMP mod_unload " },
                       { "test1", "value1" },
                       { "test2", "value2" } };

    for ( auto i = 0; i < sizeof( attributes ) / sizeof( attributes[0] );
          i++ ) {
        std::string field;
        std::string value;
        modinfo.get_attribute( i, field, value );

        EXPECT_EQ( field, attributes[i].field );
        EXPECT_EQ( value, attributes[i].value );
    }

    for ( auto i = 0; i < sizeof( attributes ) / sizeof( attributes[0] );
          i++ ) {
        std::string field = attributes[i].field;
        std::string value;
        modinfo.get_attribute( field, value );

        EXPECT_EQ( value, attributes[i].value );
    }
}

////////////////////////////////////////////////////////////////////////////////
TEST( ELFIOTest, array_read_32 )
{
    elfio reader;
    ASSERT_EQ( reader.load( "elf_examples/hello_32" ), true );

    section* array_sec = reader.sections[".ctors"];
    ASSERT_NE( array_sec, nullptr );

    const_array_section_accessor<> array( reader, array_sec );
    ASSERT_EQ( array.get_entries_num(), (Elf_Xword)2 );
    Elf64_Addr addr;
    EXPECT_EQ( array.get_entry( 0, addr ), true );
    EXPECT_EQ( addr, 0xFFFFFFFF );
    EXPECT_EQ( array.get_entry( 1, addr ), true );
    EXPECT_EQ( addr, 0x00000000 );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ELFIOTest, array_read_64 )
{
    elfio reader;
    ASSERT_EQ( reader.load( "elf_examples/hello_64" ), true );

    section* array_sec = reader.sections[".ctors"];
    ASSERT_NE( array_sec, nullptr );

    const_array_section_accessor<Elf64_Addr> array( reader, array_sec );
    ASSERT_EQ( array.get_entries_num(), (Elf_Xword)2 );
    Elf64_Addr addr;
    EXPECT_EQ( array.get_entry( 0, addr ), true );
    EXPECT_EQ( addr, 0xFFFFFFFFFFFFFFFF );
    EXPECT_EQ( array.get_entry( 1, addr ), true );
    EXPECT_EQ( addr, 0x0000000000000000 );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ELFIOTest, init_array_read_64 )
{
    elfio      reader;
    Elf64_Addr addr;
    ASSERT_EQ( reader.load( "elf_examples/ctors" ), true );

    section* array_sec = reader.sections[".init_array"];
    ASSERT_NE( array_sec, nullptr );

    const_array_section_accessor<Elf64_Addr> array( reader, array_sec );
    ASSERT_EQ( array.get_entries_num(), (Elf_Xword)2 );
    EXPECT_EQ( array.get_entry( 0, addr ), true );
    EXPECT_EQ( addr, 0x12C0 );
    EXPECT_EQ( array.get_entry( 1, addr ), true );
    EXPECT_EQ( addr, 0x149F );

    array_sec = reader.sections[".fini_array"];
    ASSERT_NE( array_sec, nullptr );

    array_section_accessor<Elf64_Addr> arrayf( reader, array_sec );
    ASSERT_EQ( arrayf.get_entries_num(), (Elf_Xword)1 );
    EXPECT_EQ( arrayf.get_entry( 0, addr ), true );
    EXPECT_EQ( addr, 0x1280 );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ELFIOTest, init_array_write_64 )
{
    elfio      reader;
    Elf64_Addr addr;
    ASSERT_EQ( reader.load( "elf_examples/ctors" ), true );

    section* array_sec = reader.sections[".init_array"];
    ASSERT_NE( array_sec, nullptr );

    array_section_accessor<Elf64_Addr> array( reader, array_sec );
    ASSERT_EQ( array.get_entries_num(), (Elf_Xword)2 );
    EXPECT_EQ( array.get_entry( 0, addr ), true );
    EXPECT_EQ( addr, 0x12C0 );
    EXPECT_EQ( array.get_entry( 1, addr ), true );
    EXPECT_EQ( addr, 0x149F );

    array.add_entry( 0x12345678 );

    ASSERT_EQ( array.get_entries_num(), (Elf_Xword)3 );
    EXPECT_EQ( array.get_entry( 0, addr ), true );
    EXPECT_EQ( addr, 0x12C0 );
    EXPECT_EQ( array.get_entry( 1, addr ), true );
    EXPECT_EQ( addr, 0x149F );
    EXPECT_EQ( array.get_entry( 2, addr ), true );
    EXPECT_EQ( addr, 0x12345678 );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ELFIOTest, test_hex )
{
    EXPECT_EQ( to_hex_string( 1 ), "0x1" );
    EXPECT_EQ( to_hex_string( 10 ), "0xA" );
    EXPECT_EQ( to_hex_string( 0x12345678 ), "0x12345678" );
    EXPECT_EQ( to_hex_string( 0xFFFFFFFF ), "0xFFFFFFFF" );
    EXPECT_EQ( to_hex_string( 0xFFFFFFFFFFFFFFFF ), "0xFFFFFFFFFFFFFFFF" );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ELFIOTest, hash32_le )
{
    elfio reader;
    // Load ELF data

    ASSERT_EQ( reader.load( "elf_examples/ARMSCII-8.so" ), true );

    std::string             name;
    Elf64_Addr              value;
    Elf_Xword               size;
    unsigned char           bind;
    unsigned char           type;
    Elf_Half                section_index;
    unsigned char           other;
    section*                symsec = reader.sections[".dynsym"];
    symbol_section_accessor syms( reader, symsec );

    for ( Elf_Xword i = 0; i < syms.get_symbols_num(); i++ ) {
        ASSERT_EQ( syms.get_symbol( i, name, value, size, bind, type,
                                    section_index, other ),
                   true );
        EXPECT_EQ( syms.get_symbol( name, value, size, bind, type,
                                    section_index, other ),
                   true );
    }
}

////////////////////////////////////////////////////////////////////////////////
TEST( ELFIOTest, hash32_be )
{
    elfio reader;
    // Load ELF data

    ASSERT_EQ( reader.load( "elf_examples/test_ppc" ), true );

    std::string             name;
    Elf64_Addr              value;
    Elf_Xword               size;
    unsigned char           bind;
    unsigned char           type;
    Elf_Half                section_index;
    unsigned char           other;
    section*                symsec = reader.sections[".dynsym"];
    symbol_section_accessor syms( reader, symsec );

    for ( Elf_Xword i = 0; i < syms.get_symbols_num(); i++ ) {
        ASSERT_EQ( syms.get_symbol( i, name, value, size, bind, type,
                                    section_index, other ),
                   true );
        EXPECT_EQ( syms.get_symbol( name, value, size, bind, type,
                                    section_index, other ),
                   true );
    }
}

////////////////////////////////////////////////////////////////////////////////
TEST( ELFIOTest, gnu_hash32_le )
{
    elfio reader;
    // Load ELF data

    ASSERT_EQ( reader.load( "elf_examples/hello_32" ), true );

    std::string             name;
    Elf64_Addr              value;
    Elf_Xword               size;
    unsigned char           bind;
    unsigned char           type;
    Elf_Half                section_index;
    unsigned char           other;
    section*                symsec = reader.sections[".dynsym"];
    symbol_section_accessor syms( reader, symsec );

    for ( Elf_Xword i = 0; i < syms.get_symbols_num(); i++ ) {
        ASSERT_EQ( syms.get_symbol( i, name, value, size, bind, type,
                                    section_index, other ),
                   true );
        EXPECT_EQ( syms.get_symbol( name, value, size, bind, type,
                                    section_index, other ),
                   true );
    }
}

////////////////////////////////////////////////////////////////////////////////
TEST( ELFIOTest, gnu_hash64_le )
{
    elfio reader;
    // Load ELF data

    ASSERT_EQ( reader.load( "elf_examples/main" ), true );

    std::string             name;
    Elf64_Addr              value;
    Elf_Xword               size;
    unsigned char           bind;
    unsigned char           type;
    Elf_Half                section_index;
    unsigned char           other;
    section*                symsec = reader.sections[".dynsym"];
    symbol_section_accessor syms( reader, symsec );

    for ( Elf_Xword i = 0; i < syms.get_symbols_num(); i++ ) {
        ASSERT_EQ( syms.get_symbol( i, name, value, size, bind, type,
                                    section_index, other ),
                   true );
        EXPECT_EQ( syms.get_symbol( name, value, size, bind, type,
                                    section_index, other ),
                   true );
    }
}

////////////////////////////////////////////////////////////////////////////////
TEST( ELFIOTest, gnu_version_64_le )
{
    elfio reader;
    // Load ELF data

    ASSERT_EQ( reader.load( "elf_examples/hello_64" ), true );

    std::string   name;
    Elf64_Addr    value;
    Elf_Xword     size;
    unsigned char bind;
    unsigned char type;
    Elf_Half      section_index;
    unsigned char other;

    section*                      dynsym = reader.sections[".dynsym"];
    const_symbol_section_accessor dynsym_acc( reader, dynsym );

    section*                      gnu_version = reader.sections[".gnu.version"];
    const_versym_section_accessor gnu_version_arr( gnu_version );

    const section* gnu_version_r = reader.sections[".gnu.version_r"];
    const_versym_r_section_accessor gnu_version_r_arr( reader, gnu_version_r );

    section* dynstr = reader.sections[".dynstr"];

    EXPECT_EQ( gnu_version->get_link(), dynsym->get_index() );
    EXPECT_EQ( gnu_version_r->get_link(), dynstr->get_index() );

    EXPECT_EQ( dynsym_acc.get_symbols_num(),
               gnu_version_arr.get_entries_num() );

    for ( Elf64_Word i = 0; i < dynsym_acc.get_symbols_num(); i++ ) {
        ASSERT_EQ( dynsym_acc.get_symbol( i, name, value, size, bind, type,
                                          section_index, other ),
                   true );

        Elf64_Half verindex = 0;
        gnu_version_arr.get_entry( i, verindex );
        if ( i < 2 )
            EXPECT_EQ( 0, verindex );
        else
            EXPECT_EQ( 2, verindex );
    }

    EXPECT_EQ( gnu_version_r_arr.get_entries_num(), 1 );

    Elf_Half    version;
    std::string file_name;
    Elf_Word    hash;
    Elf_Half    flags;
    Elf_Half    vna_other;
    std::string dep_name;
    gnu_version_r_arr.get_entry( 0, version, file_name, hash, flags, vna_other,
                                 dep_name );
    EXPECT_EQ( version, 1 );
    EXPECT_EQ( file_name, "libc.so.6" );
    EXPECT_EQ( hash, 0x09691a75 );
    EXPECT_EQ( flags, 0 );
    EXPECT_EQ( vna_other, 2 );
    EXPECT_EQ( dep_name, "GLIBC_2.2.5" );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ELFIOTest, gnu_version_d_64_le )
{
    elfio reader;
    // Load ELF data

    ASSERT_EQ( reader.load( "elf_examples/libversion_d.so" ), true );

    section*                      dynsym = reader.sections[".dynsym"];
    const_symbol_section_accessor dynsym_acc( reader, dynsym );

    section*                      gnu_version = reader.sections[".gnu.version"];
    const_versym_section_accessor gnu_version_arr( gnu_version );

    const section* gnu_version_d = reader.sections[".gnu.version_d"];
    const_versym_d_section_accessor gnu_version_d_arr( reader, gnu_version_d );

    section* dynstr = reader.sections[".dynstr"];

    EXPECT_EQ( gnu_version_d->get_link(), dynstr->get_index() );

    EXPECT_EQ( dynsym_acc.get_symbols_num(),
               gnu_version_arr.get_entries_num() );

    EXPECT_EQ( dynsym_acc.get_symbols_num(), 10 );

    EXPECT_EQ( gnu_version_d_arr.get_entries_num(), 3 );

    auto v_check = [&]( const std::string& symbol,
                        const std::string& vername ) -> void {
        std::string   name;
        Elf64_Addr    value;
        Elf_Xword     size;
        unsigned char bind;
        unsigned char type;
        Elf_Half      section_index;
        unsigned char other;
        Elf64_Half    verindex;

        for ( Elf64_Word i = 0; i < dynsym_acc.get_symbols_num(); i++ ) {
            ASSERT_EQ( dynsym_acc.get_symbol( i, name, value, size, bind, type,
                                              section_index, other ),
                       true );

            Elf64_Half vi;
            ASSERT_EQ( gnu_version_arr.get_entry( i, vi ), true );
            if ( name == symbol ) {
                verindex = vi;
            }
        }
        ASSERT_NE( verindex, 0 );

        for ( Elf64_Word i = 0; i < gnu_version_d_arr.get_entries_num(); i++ ) {
            Elf_Half    flags;
            Elf_Half    version_index;
            Elf_Word    hash;
            std::string dep_name;
            ASSERT_EQ( gnu_version_d_arr.get_entry( i, flags, version_index,
                                                    hash, dep_name ),
                       true );
            if ( version_index == verindex ) {
                EXPECT_EQ( flags, 0 );
                EXPECT_EQ( dep_name, vername );
                return;
            }
        }
        FAIL() << "version entry is not found";
    };
    v_check( "_Z20print_hello_world_v1v", "HELLO_1.0" );
    v_check( "_Z20print_hello_world_v2v", "HELLO_2.0" );
}

////////////////////////////////////////////////////////////////////////////////
// TEST( ELFIOTest, gnu_version_64_le_modify )
// {
//     elfio reader;
//     // Load ELF data

//     ASSERT_EQ( reader.load( "elf_examples/hello_64" ), true );

//     std::string   name;
//     Elf64_Addr    value;
//     Elf_Xword     size;
//     unsigned char bind;
//     unsigned char type;
//     Elf_Half      section_index;
//     unsigned char other;

//     section*                gnu_version = reader.sections[".gnu.version"];
//     versym_section_accessor gnu_version_arr( gnu_version );

//     section*                  gnu_version_r = reader.sections[".gnu.version_r"];
//     versym_r_section_accessor gnu_version_r_arr( reader, gnu_version_r );

//     auto       orig_entries_num = gnu_version_arr.get_entries_num();
//     Elf64_Word i                = 0;
//     for ( i = 0; i < orig_entries_num; i++ ) {
//         gnu_version_arr.modify_entry( i, i + 10 );
//     }
//     gnu_version_arr.add_entry( i + 10 );
//     gnu_version_arr.add_entry( i + 11 );
//     EXPECT_EQ( orig_entries_num + 2,
//                        gnu_version_arr.get_entries_num() );

//     for ( i = 0; i < gnu_version_arr.get_entries_num(); i++ ) {
//         Elf_Half value;
//         gnu_version_arr.get_entry( i, value );
//         EXPECT_EQ( i + 10, value );
//     }
// }

////////////////////////////////////////////////////////////////////////////////
TEST( ELFIOTest, move_constructor_and_assignment )
{
    elfio r1;

    // Load ELF data
    ASSERT_EQ( r1.load( "elf_examples/hello_64" ), true );
    Elf64_Addr  entry    = r1.get_entry();
    std::string sec_name = r1.sections[".text"]->get_name();
    Elf_Xword   seg_size = r1.segments[1]->get_memory_size();

    // Move to a vector element
    std::vector<elfio> v;
    v.emplace_back( std::move( r1 ) );
    EXPECT_EQ( v[0].get_entry(), entry );
    EXPECT_EQ( v[0].sections[".text"]->get_name(), sec_name );
    EXPECT_EQ( v[0].segments[1]->get_memory_size(), seg_size );

    elfio r2;
    r2 = std::move( v[0] );
    EXPECT_EQ( r2.get_entry(), entry );
    EXPECT_EQ( r2.sections[".text"]->get_name(), sec_name );
    EXPECT_EQ( r2.segments[1]->get_memory_size(), seg_size );
}

TEST( ELFIOTest, address_translation_test )
{
    std::vector<address_translation> ranges;

    ranges.emplace_back( 0, 100, 500 );
    ranges.emplace_back( 500, 1000, 1000 );
    ranges.emplace_back( 2000, 1000, 3000 );

    address_translator tr;
    tr.set_address_translation( ranges );

    EXPECT_EQ( tr[0], 500 );
    EXPECT_EQ( tr[510], 1010 );
    EXPECT_EQ( tr[1710], 1710 );
    EXPECT_EQ( tr[2710], 3710 );
    EXPECT_EQ( tr[3710], 3710 );

    ranges.clear();
    tr.set_address_translation( ranges );

    EXPECT_EQ( tr[0], 0 );
    EXPECT_EQ( tr[510], 510 );
    EXPECT_EQ( tr[1710], 1710 );
    EXPECT_EQ( tr[2710], 2710 );
    EXPECT_EQ( tr[3710], 3710 );
}
