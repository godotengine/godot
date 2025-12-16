/*
Copyright (C) 2025-present by Serge Lamikhov-Center

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
#include <ario/ario.hpp>
#include <elfio/elfio.hpp>

using namespace ELFIO;
using namespace ARIO;

////////////////////////////////////////////////////////////////////////////////
TEST( ARIOTest, wrong_file_name )
{
    ario archive;
    ASSERT_EQ( archive.load( "ario/does_not_exist.a" ).ok(), false );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ARIOTest, wrong_file_magic )
{
    ario archive;
    auto result = archive.load( "ario/invalid_magic.a" );
    ASSERT_EQ( result.ok(), false );
    ASSERT_EQ( result.what(), "Invalid archive format. Expected magic: "
                              "!<arch>\n, but got !<arkh>\n" );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ARIOTest, simple_text_load )
{
    ario archive;
    ASSERT_EQ( archive.load( "ario/simple_text.a" ).what(), "No errors" );
    ASSERT_EQ( archive.members.size(), 6 );
    EXPECT_EQ( archive.members[0].name, "hello.c" );
    EXPECT_EQ( archive.members[0].size, 45 );
    EXPECT_EQ( archive.members[1].name, "hello2.c" );
    EXPECT_EQ( archive.members[1].size, 7 );
    EXPECT_EQ( archive.members[2].name, "hello3.c" );
    EXPECT_EQ( archive.members[2].size, 8 );
    EXPECT_EQ( archive.members[3].name, "hello4.c" );
    EXPECT_EQ( archive.members[3].size, 10 );
    EXPECT_EQ( archive.members[4].name, "hello41.c" );
    EXPECT_EQ( archive.members[4].size, 11 );
    EXPECT_EQ( archive.members[5].name, "hello5.c" );
    EXPECT_EQ( archive.members[5].size, 8 );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ARIOTest, long_name_load )
{
    ario archive;
    auto result = archive.load( "ario/long_name.a" );
    ASSERT_EQ( result.ok(), true );
    ASSERT_EQ( result.what(), "No errors" );

    ASSERT_EQ( archive.members.size(), 9 );
    EXPECT_EQ( archive.members[0].name, "hello.c" );
    EXPECT_EQ( archive.members[0].size, 45 );
    EXPECT_EQ( archive.members[1].name, "hello2.c" );
    EXPECT_EQ( archive.members[1].size, 7 );
    EXPECT_EQ( archive.members[6].name, "a_file_with_very_long_name.txt" );
    EXPECT_EQ( archive.members[6].size, 6 );
    EXPECT_EQ( archive.members[7].name, "a_file_with_very_long_name2.txt" );
    EXPECT_EQ( archive.members[7].size, 6 );
    EXPECT_EQ( archive.members[8].name, "a_file_with_very_long_name3.txt" );
    EXPECT_EQ( archive.members[8].size, 6 );

    EXPECT_EQ( archive.members["a_file_with_very_long_name.txt"].name,
               archive.members[6].name );
    EXPECT_EQ( archive.members["a_file_with_very_long_name3.txt"].name,
               archive.members[8].name );
    EXPECT_EQ( archive.members["hello2.c"].name, archive.members[1].name );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ARIOTest, load_libgcov )
{
    ario archive;
    ASSERT_EQ( archive.load( "ario/libgcov.a" ).ok(), true );

    ASSERT_EQ( archive.members.size(), 29 );
    EXPECT_EQ( archive.members[0].name, "_gcov_merge_add.o" );
    EXPECT_EQ( archive.members[0].size, 1416 );
    EXPECT_EQ( archive.members[0].mode, 0644 );
    EXPECT_EQ( archive.members[0].data().substr( 0, 4 ), "\x7F"
                                                         "ELF" );
    EXPECT_EQ( archive.members[6].name, "_gcov_interval_profiler_atomic.o" );
    EXPECT_EQ( archive.members[6].size, 1264 );
    EXPECT_EQ( archive.members[6].mode, 0644 );
    EXPECT_EQ( archive.members[6].data().substr( 0, 4 ), "\x7F"
                                                         "ELF" );
    EXPECT_EQ( archive.members[17].name,
               "_gcov_indirect_call_topn_profiler.o" );
    EXPECT_EQ( archive.members[17].size, 2104 );
    EXPECT_EQ( archive.members[17].mode, 0644 );
    EXPECT_EQ( archive.members[17].data().substr( 0, 4 ), "\x7F"
                                                          "ELF" );
    EXPECT_EQ( archive.members[28].name, "_gcov.o" );
    EXPECT_EQ( archive.members[28].size, 16768 );
    EXPECT_EQ( archive.members[28].mode, 0644 );
    EXPECT_EQ( archive.members[28].data().substr( 0, 4 ), "\x7F"
                                                          "ELF" );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ARIOTest, find_symbol_libgcov )
{
    ario archive;
    ASSERT_EQ( archive.load( "ario/libgcov.a" ).ok(), true );

    std::optional<std::reference_wrapper<const ario::Member>> member =
        std::nullopt;
    auto result = archive.find_symbol( "__gcov_merge_add", member );
    ASSERT_EQ( result.ok(), true );
    ASSERT_EQ( member->get().name, "_gcov_merge_add.o" );

    result =
        archive.find_symbol( "__gcov_indirect_call_topn_profiler", member );
    ASSERT_EQ( result.ok(), true );
    ASSERT_EQ( member->get().name, "_gcov_indirect_call_topn_profiler.o" );

    result = archive.find_symbol( "__not_found", member );
    ASSERT_EQ( result.ok(), false );
    ASSERT_EQ( member.has_value(), false );

    result = archive.find_symbol( "__gcov_write_counter", member );
    ASSERT_EQ( result.ok(), true );
    ASSERT_EQ( member->get().name, "_gcov.o" );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ARIOTest, get_symbols_for_member_libgcov )
{
    ario archive;
    ASSERT_EQ( archive.load( "ario/libgcov.a" ).ok(), true );

    std::vector<std::string> symbols;

    auto result = archive.get_symbols_for_member( archive.members[0], symbols );
    ASSERT_EQ( result.ok(), true );
    ASSERT_EQ( symbols.size(), 1 );
    ASSERT_EQ( symbols[0], "__gcov_merge_add" );

    result = archive.get_symbols_for_member( archive.members[6], symbols );
    ASSERT_EQ( result.ok(), true );
    ASSERT_EQ( symbols.size(), 1 );
    ASSERT_EQ( symbols[0], "__gcov_interval_profiler_atomic" );

    result = archive.get_symbols_for_member( archive.members[28], symbols );
    ASSERT_EQ( result.ok(), true );
    ASSERT_EQ( symbols.size(), 20 );
    // We cannot garantee the order of symbols in the symbol table,
    // so we just check that some of them are present
    ASSERT_NE( std::find( symbols.begin(), symbols.end(), "__gcov_exit" ),
               symbols.end() );
    ASSERT_NE( std::find( symbols.begin(), symbols.end(), "__gcov_var" ),
               symbols.end() );
    ASSERT_NE(
        std::find( symbols.begin(), symbols.end(), "__gcov_read_counter" ),
        symbols.end() );
    ASSERT_EQ( std::find( symbols.begin(), symbols.end(), "doesn't exist" ),
               symbols.end() );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ARIOTest, get_symbols_for_ELF_files_in_archive )
{
    // Load the archive file containing ELF object files
    ario archive;
    ASSERT_EQ( archive.load( "ario/libgcov.a" ).ok(), true );

    // Iterate over each member (object file) in the archive
    for ( const auto& member : archive.members ) {
        // Extract the raw data of the member (should be an ELF file)
        std::string        content = member.data();
        std::istringstream iss( content );

        // Parse the member's data as an ELF file using ELFIO
        elfio elf_reader;
        ASSERT_EQ( elf_reader.load( iss ), true );

        std::uint32_t counter = 0;

        // Iterate over all sections in the ELF file
        for ( const auto& sec : elf_reader.sections ) {
            // Look for the symbol table section
            if ( sec->get_type() == SHT_SYMTAB ) {
                // Access the symbols in the symbol table
                symbol_section_accessor symbols( elf_reader, sec.get() );

                std::string   name;
                Elf64_Addr    value;
                Elf_Xword     size;
                unsigned char bind = 0, type = 0;
                Elf_Half      section_index;
                unsigned char other;

                std::optional<std::reference_wrapper<const ario::Member>>
                    found_member = std::nullopt;
                // Iterate over all symbols in the symbol table
                for ( Elf_Xword i = 0; i < symbols.get_symbols_num(); ++i ) {
                    // Extract symbol properties
                    ASSERT_EQ( symbols.get_symbol( i, name, value, size, bind,
                                                   type, section_index, other ),
                               true );
                    // For each global function or object symbol, check that the archive symbol table can find it
                    if ( ( type == STT_FUNC || type == STT_OBJECT ||
                           type == STT_TLS || type == STT_COMMON ) &&
                         bind == STB_GLOBAL ) {
                        ++counter;
                        ASSERT_EQ(
                            archive.find_symbol( name, found_member ).ok(),
                            true );
                    }
                }
            }
        }

        // Check that the number of global symbols found matches the expected count
        std::vector<std::string> symbols_from_the_member;
        ASSERT_EQ(
            archive.get_symbols_for_member( member, symbols_from_the_member )
                .ok(),
            true );
        ASSERT_EQ( counter, symbols_from_the_member.size() );
    }
}

////////////////////////////////////////////////////////////////////////////////
TEST( ARIOTest, basic_header_save )
{
    ario               archive;
    std::ostringstream oss;

    auto result = archive.save( oss );
    ASSERT_EQ( result.ok(), true );

    ASSERT_EQ( oss.str(), "!<arch>\n" );
}

void compare_archives( const ario& archive1, const ario& archive2 )
{
    ASSERT_EQ( archive1.members.size(), archive2.members.size() );

    for ( size_t i = 0; i < archive1.members.size(); ++i ) {
        EXPECT_EQ( archive1.members[i].name, archive2.members[i].name );
        EXPECT_EQ( archive1.members[i].date, archive2.members[i].date );
        EXPECT_EQ( archive1.members[i].uid, archive2.members[i].uid );
        EXPECT_EQ( archive1.members[i].gid, archive2.members[i].gid );
        EXPECT_EQ( archive1.members[i].mode, archive2.members[i].mode );
        EXPECT_EQ( archive1.members[i].size, archive2.members[i].size );
        EXPECT_EQ( archive1.members[i].data(), archive2.members[i].data() );
    }
}

////////////////////////////////////////////////////////////////////////////////
TEST( ARIOTest, header_save )
{
    ario archive;
    ASSERT_EQ( archive.load( "ario/simple_text.a" ).ok(), true );

    // Save the archive to a new file
    ASSERT_EQ( archive.save( "ario/simple_text_saved.a" ).ok(), true );

    // Load the saved archive and check its contents
    ario loaded_archive;
    ASSERT_EQ( loaded_archive.load( "ario/simple_text_saved.a" ).ok(), true );

    compare_archives( loaded_archive, archive );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ARIOTest, long_name_dir_save )
{
    ario archive;
    ASSERT_EQ( archive.load( "ario/long_name.a" ).ok(), true );

    // Save the archive to a new file
    auto result = archive.save( "ario/long_name_saved.a" );
    ASSERT_EQ( result.ok(), true );

    // Load the saved archive and check its contents
    ario loaded_archive;
    ASSERT_EQ( loaded_archive.load( "ario/long_name_saved.a" ).ok(), true );

    compare_archives( loaded_archive, archive );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ARIOTest, long_name_save )
{
    ario archive;
    ASSERT_EQ( archive.load( "ario/long_name.a" ).ok(), true );

    // Save the archive to a new file
    auto result = archive.save( "ario/long_name_saved.a" );
    ASSERT_EQ( result.ok(), true );

    // Load the saved archive and check its contents
    ario loaded_archive;
    ASSERT_EQ( loaded_archive.load( "ario/long_name_saved.a" ).ok(), true );
    ASSERT_EQ( loaded_archive.members.size(), archive.members.size() );
    EXPECT_EQ( loaded_archive.members[0].name, archive.members[0].name );
    EXPECT_EQ( loaded_archive.members[0].size, archive.members[0].size );
    EXPECT_EQ( loaded_archive.members[archive.members.size() - 1].name,
               archive.members[archive.members.size() - 1].name );
    EXPECT_EQ( loaded_archive.members[archive.members.size() - 1].size,
               archive.members[archive.members.size() - 1].size );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ARIOTest, libgcov_save )
{
    ario archive;
    ASSERT_EQ( archive.load( "ario/libgcov.a" ).ok(), true );
    // Save the archive to a new file
    auto result = archive.save( "ario/libgcov_saved.a" );
    ASSERT_EQ( result.ok(), true );
    // Load the saved archive and check its contents
    ario loaded_archive;
    ASSERT_EQ( loaded_archive.load( "ario/libgcov_saved.a" ).ok(), true );
    ASSERT_EQ( loaded_archive.members.size(), archive.members.size() );
    EXPECT_EQ( loaded_archive.members[0].name, archive.members[0].name );
    EXPECT_EQ( loaded_archive.members[0].size, archive.members[0].size );
    EXPECT_EQ( loaded_archive.members[loaded_archive.members.size() - 1].name,
               archive.members[archive.members.size() - 1].name );
    EXPECT_EQ( loaded_archive.members[loaded_archive.members.size() - 1].size,
               archive.members[archive.members.size() - 1].size );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ARIOTest, add_simple_member )
{
    ario archive;
    ASSERT_EQ( archive.load( "ario/simple_text.a" ).ok(), true );

    ario::Member m;
    m.name = "added_text.txt";
    m.date = 0;
    m.gid  = 1234;
    m.uid  = 5678;
    m.mode = 0644;
    archive.add_member( m, "The content\nof this\nmember" );
    m.name = "added_text1.txt";
    m.date = 0;
    m.gid  = 1234;
    m.uid  = 5678;
    m.mode = 0644;
    archive.add_member( m, "The content\nof this\nmember\n" );
    m.name = "added_text2.txt";
    m.date = 0;
    m.gid  = 1234;
    m.uid  = 5678;
    m.mode = 0644;
    archive.add_member( m, "" );
    m.name = "added_text3.txt";
    m.date = 0;
    m.gid  = 1234;
    m.uid  = 5678;
    m.mode = 0644;
    archive.add_member( m, "Hello\n" );

    // Save the archive to a new file
    auto result = archive.save( "ario/simple_text_saved.a" );
    ASSERT_EQ( result.ok(), true );

    // Load the saved archive and check its contents
    ario loaded_archive;
    ASSERT_EQ( archive.load( "ario/simple_text.a" ).ok(), true );
    ASSERT_EQ( loaded_archive.load( "ario/simple_text_saved.a" ).ok(), true );
    ASSERT_EQ( loaded_archive.members.size(), archive.members.size() + 4 );
    EXPECT_EQ( loaded_archive.members[0].name, archive.members[0].name );
    EXPECT_EQ( loaded_archive.members[0].size, archive.members[0].size );
    EXPECT_EQ( loaded_archive.members[loaded_archive.members.size() - 5].name,
               archive.members[archive.members.size() - 1].name );
    EXPECT_EQ( loaded_archive.members[loaded_archive.members.size() - 5].size,
               archive.members[archive.members.size() - 1].size );
    EXPECT_EQ( loaded_archive.members[loaded_archive.members.size() - 4].name,
               "added_text.txt" );
    EXPECT_EQ( loaded_archive.members[loaded_archive.members.size() - 4].data(),
               "The content\nof this\nmember" );
    EXPECT_EQ( loaded_archive.members[loaded_archive.members.size() - 3].name,
               "added_text1.txt" );
    EXPECT_EQ( loaded_archive.members[loaded_archive.members.size() - 3].data(),
               "The content\nof this\nmember\n" );
    EXPECT_EQ( loaded_archive.members[loaded_archive.members.size() - 2].name,
               "added_text2.txt" );
    EXPECT_EQ( loaded_archive.members[loaded_archive.members.size() - 2].data(),
               "" );
    EXPECT_EQ( loaded_archive.members[loaded_archive.members.size() - 1].name,
               "added_text3.txt" );
    EXPECT_EQ( loaded_archive.members[loaded_archive.members.size() - 1].data(),
               "Hello\n" );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ARIOTest, add_long_name_member )
{
    ario archive;
    ASSERT_EQ( archive.load( "ario/simple_text.a" ).ok(), true );

    ario::Member                                              m;
    std::optional<std::reference_wrapper<const ario::Member>> added_member =
        std::nullopt;
    m.name = "long_name_member_added_text.txt";
    m.date = 0;
    m.gid  = 1234;
    m.uid  = 5678;
    m.mode = 0644;
    archive.add_member( m, "The content\nof this\nmember" );
    m.name = "long_name_member_added_text1.txt";
    m.date = 0;
    m.gid  = 1234;
    m.uid  = 5678;
    m.mode = 0644;
    archive.add_member( m, "The content\nof this\nmember\n" );
    m.name = "long_name_member_added_text2.txt";
    m.date = 0;
    m.gid  = 1234;
    m.uid  = 5678;
    m.mode = 0644;
    archive.add_member( m, "" );
    m.name = "long_name_member_added_text333.txt";
    m.date = 0;
    m.gid  = 1234;
    m.uid  = 5678;
    m.mode = 0644;
    archive.add_member( m, "Hello\n" );

    // Save the archive to a new file
    auto result = archive.save( "ario/long_name_saved.a" );
    ASSERT_EQ( result.ok(), true );

    // Load the saved archive and check its contents
    ario loaded_archive;
    ASSERT_EQ( archive.load( "ario/simple_text.a" ).ok(), true );
    ASSERT_EQ( loaded_archive.load( "ario/long_name_saved.a" ).ok(), true );
    ASSERT_EQ( loaded_archive.members.size(), archive.members.size() + 4 );
    EXPECT_EQ( loaded_archive.members[0].name, archive.members[0].name );
    EXPECT_EQ( loaded_archive.members[0].size, archive.members[0].size );
    EXPECT_EQ( loaded_archive.members[loaded_archive.members.size() - 5].name,
               archive.members[archive.members.size() - 1].name );
    EXPECT_EQ( loaded_archive.members[loaded_archive.members.size() - 5].size,
               archive.members[archive.members.size() - 1].size );
    EXPECT_EQ( loaded_archive.members[loaded_archive.members.size() - 4].name,
               "long_name_member_added_text.txt" );
    EXPECT_EQ( loaded_archive.members[loaded_archive.members.size() - 4].data(),
               "The content\nof this\nmember" );
    EXPECT_EQ( loaded_archive.members[loaded_archive.members.size() - 3].name,
               "long_name_member_added_text1.txt" );
    EXPECT_EQ( loaded_archive.members[loaded_archive.members.size() - 3].data(),
               "The content\nof this\nmember\n" );
    EXPECT_EQ( loaded_archive.members[loaded_archive.members.size() - 2].name,
               "long_name_member_added_text2.txt" );
    EXPECT_EQ( loaded_archive.members[loaded_archive.members.size() - 2].data(),
               "" );
    EXPECT_EQ( loaded_archive.members[loaded_archive.members.size() - 1].name,
               "long_name_member_added_text333.txt" );
    EXPECT_EQ( loaded_archive.members[loaded_archive.members.size() - 1].data(),
               "Hello\n" );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ARIOTest, new_text_lib )
{
    ario archive;

    ario::Member                                              m;
    std::optional<std::reference_wrapper<const ario::Member>> added_member =
        std::nullopt;
    m.name = "123456789012345";
    m.date = 0;
    m.gid  = 1234;
    m.uid  = 5678;
    m.mode = 0644;
    archive.add_member( m, "data\n" );
    m.name = "1234567890123456";
    m.date = 0;
    m.gid  = 1234;
    m.uid  = 5678;
    m.mode = 0644;
    archive.add_member( m, "data\n" );
    m.name = "12345678901234567";
    m.date = 0;
    m.gid  = 1234;
    m.uid  = 5678;
    m.mode = 0644;
    archive.add_member( m, "data\n" );
    m.name = "12345";
    m.date = 0;
    m.gid  = 1234;
    m.uid  = 5678;
    m.mode = 0644;
    archive.add_member( m, "data\n" );
    m.name = "123456789012345678";
    m.date = 0;
    m.gid  = 1234;
    m.uid  = 5678;
    m.mode = 0644;
    archive.add_member( m, "data\n" );
    m.name = "1234567";
    m.date = 0;
    m.gid  = 1234;
    m.uid  = 5678;
    m.mode = 0644;
    archive.add_member( m, "data\n" );

    // Save the archive to a new file
    auto result = archive.save( "ario/new_text_lib.a" );
    ASSERT_EQ( result.ok(), true );

    // Load the saved archive and check its contents
    ario loaded_archive;
    ASSERT_EQ( loaded_archive.load( "ario/new_text_lib.a" ).ok(), true );
    ASSERT_EQ( loaded_archive.members.size(), 6 );
    std::vector<std::string> ref_names = {
        "123456789012345", "1234567890123456",   "12345678901234567",
        "12345",           "123456789012345678", "1234567" };
    for ( size_t i = 0; i < loaded_archive.members.size(); i++ ) {
        EXPECT_EQ( loaded_archive.members[i].name, ref_names[i] );
        EXPECT_EQ( loaded_archive.members[i].size, 5 );
        EXPECT_EQ( loaded_archive.members[i].data(), "data\n" );
    }
}

////////////////////////////////////////////////////////////////////////////////
TEST( ARIOTest, new_text_lib_with_symbols )
{
    ario archive;

    for ( auto i = 0; i < 20; i++ ) {
        ario::Member m;
        m.name = "name__________" + std::to_string( i );
        m.date = 0;
        m.gid  = 1234;
        m.uid  = 5678;
        m.mode = 0644;
        ASSERT_EQ(
            archive.add_member( m, "data" + std::to_string( i ) + "\n" ).ok(),
            true );

        std::vector<std::string> symbols = {};
        for ( auto j = 0; j < i; j++ ) {
            symbols.emplace_back( "symbol_" + std::to_string( 100 * i + j ) );
        }
        ASSERT_EQ(
            archive.add_symbols_for_member( archive.members.back(), symbols )
                .ok(),
            true );
    }
    // Save the archive to a new file
    auto result = archive.save( "ario/new_text_lib_with_symbols.a" );
    ASSERT_EQ( result.ok(), true );

    // Load the saved archive and check its contents
    ario loaded_archive;
    ASSERT_EQ( loaded_archive.load( "ario/new_text_lib_with_symbols.a" ).ok(),
               true );
    ASSERT_EQ( loaded_archive.members.size(), 20 );
    for ( const auto& m : loaded_archive.members ) {
        auto index = std::stoi( m.name.c_str() + 14 );
        ASSERT_EQ( loaded_archive.members[index].name, m.name );

        std::vector<std::string> symbols = {};
        ASSERT_EQ( loaded_archive.get_symbols_for_member( m, symbols ).ok(),
                   true );
        ASSERT_EQ( symbols.size(), index );
        for ( const auto& symbol : symbols ) {
            std::optional<std::reference_wrapper<const ario::Member>> ms =
                std::nullopt;
            ASSERT_EQ( loaded_archive.find_symbol( symbol, ms ).ok(), true );
            ASSERT_EQ( ms->get().name, m.name );
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
TEST( ARIOTest, load_empty_archive )
{
    ario archive;
    ASSERT_EQ( archive.load( "ario/empty.a" ).ok(), true );
    ASSERT_EQ( archive.members.size(), 0 );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ARIOTest, add_duplicate_member )
{
    ario archive;
    ASSERT_EQ( archive.load( "ario/simple_text.a" ).ok(), true );

    ario::Member m = archive.members[0];
    std::optional<std::reference_wrapper<const ario::Member>> added_member =
        std::nullopt;
    auto result = archive.add_member( m, "duplicate content" );
    ASSERT_EQ( result.ok(), false );
    ASSERT_NE( std::string( result.what() ).find( "already exists" ),
               std::string::npos );
}

TEST( ARIOTest, add_symbols_for_nonexistent_member )
{
    ario archive;
    ASSERT_EQ( archive.load( "ario/simple_text.a" ).ok(), true );

    ario::Member fake_member;
    fake_member.name                 = "not_in_archive.txt";
    std::vector<std::string> symbols = { "fake_symbol" };
    auto result = archive.add_symbols_for_member( fake_member, symbols );
    ASSERT_EQ( result.ok(), false );
    ASSERT_NE( std::string( result.what() ).find( "not found" ),
               std::string::npos );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ARIOTest, get_symbols_for_nonexistent_member )
{
    ario archive;
    ASSERT_EQ( archive.load( "ario/simple_text.a" ).ok(), true );

    ario::Member fake_member;
    fake_member.name = "not_in_archive.txt";
    std::vector<std::string> symbols;
    auto result = archive.get_symbols_for_member( fake_member, symbols );
    ASSERT_EQ( result.ok(), false );
    ASSERT_NE( std::string( result.what() ).find( "not found" ),
               std::string::npos );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ARIOTest, save_and_reload_empty_archive )
{
    ario               archive;
    std::ostringstream oss;
    ASSERT_EQ( archive.save( oss ).ok(), true );
    std::istringstream iss( oss.str() );
    ario               loaded_archive;
    ASSERT_EQ(
        loaded_archive.load( std::make_unique<std::istringstream>( oss.str() ) )
            .ok(),
        true );
    ASSERT_EQ( loaded_archive.members.size(), 0 );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ARIOTest, member_access_out_of_range )
{
    ario archive;
    ASSERT_EQ( archive.load( "ario/simple_text.a" ).ok(), true );
    // Index out of range
    EXPECT_THROW( archive.members[1000], std::out_of_range );
    // Name not found
    EXPECT_THROW( archive.members["not_in_archive.txt"], std::out_of_range );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ARIOTest, find_symbol_not_present )
{
    ario archive;
    ASSERT_EQ( archive.load( "ario/simple_text.a" ).ok(), true );
    std::optional<std::reference_wrapper<const ario::Member>> member =
        std::nullopt;
    auto result = archive.find_symbol( "not_a_symbol", member );
    ASSERT_EQ( result.ok(), false );
    ASSERT_EQ( member.has_value(), false );
}

////////////////////////////////////////////////////////////////////////////////
// Test: Remove all members and save
TEST( ARIOTest, remove_all_members_and_save )
{
    ario archive;
    ASSERT_EQ( archive.load( "ario/simple_text.a" ).ok(), true );
    // Remove all members by creating a new archive and not adding any
    ario empty_archive;
    ASSERT_EQ( empty_archive.save( "ario/removed_all.a" ).ok(), true );
    ario loaded_archive;
    ASSERT_EQ( loaded_archive.load( "ario/removed_all.a" ).ok(), true );
    ASSERT_EQ( loaded_archive.members.size(), 0 );
}

////////////////////////////////////////////////////////////////////////////////
// Test: Add member with empty name
TEST( ARIOTest, add_member_with_empty_name )
{
    ario         archive;
    ario::Member m;
    m.name = "";
    m.mode = 0644;
    std::optional<std::reference_wrapper<const ario::Member>> added_member =
        std::nullopt;
    auto result = archive.add_member( m, "data" );
    ASSERT_EQ( result.ok(), false );
}

////////////////////////////////////////////////////////////////////////////////
// Test: Add member with duplicate symbols
TEST( ARIOTest, add_duplicate_symbols_for_member )
{
    ario         archive;
    ario::Member m;
    m.name = "dup_symbol.o";
    m.mode = 0644;
    ASSERT_EQ( archive.add_member( m, "data" ).ok(), true );
    std::vector<std::string> symbols = { "sym1", "sym1", "sym2" };
    ASSERT_EQ(
        archive.add_symbols_for_member( archive.members.back(), symbols ).ok(),
        true );
    std::vector<std::string> out_symbols;
    ASSERT_EQ(
        archive.get_symbols_for_member( archive.members.back(), out_symbols )
            .ok(),
        true );
    // Should contain all symbols, including duplicates
    ASSERT_EQ( std::count( out_symbols.begin(), out_symbols.end(), "sym1" ),
               1 );
    ASSERT_EQ( std::count( out_symbols.begin(), out_symbols.end(), "sym2" ),
               1 );
}

////////////////////////////////////////////////////////////////////////////////
// Test: Add member with special characters in name
TEST( ARIOTest, add_member_with_special_characters )
{
    ario         archive;
    ario::Member m;
    m.name = "spécial_名.o";
    m.mode = 0644;
    ASSERT_EQ( archive.add_member( m, "data" ).ok(), true );
    ASSERT_EQ( archive.members.back().name, "spécial_名.o" );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ARIOTest, add_member_with_non_ascii_name )
{
    ario         archive;
    ario::Member m;
    m.name      = u8"тест.o";
    m.mode      = 0644;
    auto result = archive.add_member( m, "data" );
    ASSERT_EQ( result.ok(), true );
    ASSERT_EQ( archive.members.back().name, u8"тест.o" );
}

////////////////////////////////////////////////////////////////////////////////
// Test: Save and load archive with only one member
TEST( ARIOTest, save_and_load_single_member_archive )
{
    ario         archive;
    ario::Member m;
    m.name = "single.o";
    m.mode = 0644;
    std::optional<std::reference_wrapper<const ario::Member>> added_member =
        std::nullopt;
    ASSERT_EQ( archive.add_member( m, "data" ).ok(), true );
    ASSERT_EQ( archive.save( "ario/single_member.a" ).ok(), true );
    ario loaded_archive;
    ASSERT_EQ( loaded_archive.load( "ario/single_member.a" ).ok(), true );
    ASSERT_EQ( loaded_archive.members.size(), 1 );
    ASSERT_EQ( loaded_archive.members[0].name, "single.o" );
    ASSERT_EQ( loaded_archive.members[0].data(), "data" );
}

////////////////////////////////////////////////////////////////////////////////
// Test: Add member with zero size data
TEST( ARIOTest, add_member_with_zero_size_data )
{
    ario         archive;
    ario::Member m;
    m.name = "empty.o";
    m.mode = 0644;
    ASSERT_EQ( archive.add_member( m, "" ).ok(), true );
    ASSERT_EQ( archive.members.back().size, 0 );
    ASSERT_EQ( archive.members.back().data(), "" );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ARIOTest, add_member_with_large_data )
{
    ario         archive;
    ario::Member m;
    m.name = "large.o";
    m.mode = 0644;
    std::string large_data( 10 * 1024 * 1024, 'A' ); // 10 MB
    auto        result = archive.add_member( m, large_data );
    ASSERT_EQ( result.ok(), true );
    ASSERT_EQ( archive.members.back().size, large_data.size() );
    ASSERT_EQ( archive.members.back().data(), large_data );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ARIOTest, save_to_invalid_path )
{
    ario         archive;
    ario::Member m;
    m.name = "file.o";
    m.mode = 0644;
    archive.add_member( m, "data" );
    auto result = archive.save( "/invalid_path/should_fail.a" );
    ASSERT_EQ( result.ok(), false );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ARIOTest, load_corrupted_archive )
{
    ario archive;
    // Create a corrupted archive in memory
    std::string        corrupted = "!<arch>\ncorrupted data";
    std::istringstream iss( corrupted );
    auto               result =
        archive.load( std::make_unique<std::istringstream>( corrupted ) );
    ASSERT_EQ( result.ok(), false );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ARIOTest, add_member_with_max_name_length )
{
    ario         archive;
    ario::Member m;
    m.name      = std::string( 255, 'a' ); // 255 chars
    m.mode      = 0644;
    auto result = archive.add_member( m, "data" );
    ASSERT_EQ( result.ok(), true );
    ASSERT_EQ( archive.members.back().name.size(), 255 );
}
