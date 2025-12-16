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

//------------------------------------------------------------------------------
// arioso.cpp
//
// This example demonstrates how to use the ARIO library (with optional ELFIO integration)
// to manage UNIX archive (.ar) files from the command line. It provides a practical tool
// for extracting, deleting, and adding files to an archive, similar to the standard 'ar' utility.
//
// Purpose:
//   - Showcase ARIO’s API for reading, modifying, and writing UNIX archive files.
//   - Illustrate integration with ELFIO for symbol extraction from ELF object files.
//
// Abilities:
//   - Extraction: Extracts specified files from the archive to the current directory.
//   - Deletion: Removes specified files from the archive.
//   - Addition: Adds new files to the archive, collecting and storing global symbols if the file is an ELF object.
//   - Archive update: Safely writes changes to the archive using a temporary file for atomic updates.
//   - Command-line interface: Accepts commands in the form:
//         arioso <archive> [-e <files...>] [-d <files...>] [-a <files...>]
//
// This example serves as both a reference for ARIO/ELFIO usage and a foundation for building custom archive management tools.
//

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>

#include <ario/ario.hpp>
#include <elfio/elfio.hpp>

using namespace ARIO;
using namespace ELFIO;

//------------------------------------------------------------------------------
// Simple command line parser for:
// arioso -e <files...> -d <files...> -a <files...>
struct CommandLineOptions
{
    std::string              archive_name;  ///< Name of the archive file
    std::vector<std::string> extract_files; ///< List of files to extract
    std::vector<std::string> delete_files;  ///< List of files to delete
    std::vector<std::string> add_files;     ///< List of files to add
};

//------------------------------------------------------------------------------
static CommandLineOptions parse_args( int argc, char** argv )
{
    if ( argc < 2 ) {
        std::cerr
            << "Usage: " << argv[0]
            << " <archive> [-e <files...>] [-d <files...>] [-a <files...>]"
            << std::endl;
        return {};
    }

    CommandLineOptions opts;

    opts.archive_name = argv[1];

    std::vector<std::string>* current = nullptr;
    for ( int i = 2; i < argc; ++i ) {
        std::string arg = argv[i];
        if ( arg == "-e" ) {
            current = &opts.extract_files;
        }
        else if ( arg == "-d" ) {
            current = &opts.delete_files;
        }
        else if ( arg == "-a" ) {
            current = &opts.add_files;
        }
        else if ( current ) {
            current->emplace_back( arg );
        }
        else {
            std::cerr << "Unknown argument or missing option: " << arg
                      << std::endl;
        }
    }
    return opts;
}

//------------------------------------------------------------------------------
// This function would contain the logic to extract the member data to a file
static ario::Result extract_member( const ario::Member& member )
{
    std::cout << "Extracting member: " << member << ", Size: " << member.size
              << " bytes" << std::endl;

    std::filesystem::path output_path =
        std::filesystem::current_path() / member.name;
    std::ofstream output_file( output_path, std::ios::binary );
    if ( !output_file ) {
        return { " Failed to create output file : " + output_path.string() };
    }
    output_file.write( member.data().c_str(), member.size );
    if ( output_file.fail() ) {
        return { "Failed to write member data to file: " +
                 output_path.string() };
    }

    return {}; // Return success
}

//------------------------------------------------------------------------------
// Extract all members from the command line extraction list.
// The search should be done by member's name. Exact match is expected
static int extract_members( const CommandLineOptions& opts,
                            const ARIO::ario&         archive )
{
    for ( const auto& file_name : opts.extract_files ) {
        const auto& pmember = std::find( archive.members.begin(),
                                         archive.members.end(), file_name );
        if ( pmember != archive.members.end() ) {
            auto result = extract_member( *pmember );
            if ( !result.ok() ) {
                std::cerr << "Error extracting member '" << file_name
                          << "': " << result.what() << std::endl;
            }
        }
        else {
            std::cerr << "Member '" << file_name << "' not found in the library"
                      << std::endl;
            return 1;
        }
    }

    return 0;
}

//------------------------------------------------------------------------------
// Copy members from the source archive to the target archive
static int copy_members( const CommandLineOptions& opts,
                         const ARIO::ario&         archive,
                         ARIO::ario&               target_archive )
{
    for ( const auto& member : archive.members ) {
        if ( std::find( opts.delete_files.begin(), opts.delete_files.end(),
                        member.name ) != opts.delete_files.end() ) {
            std::cout << "Removing member: " << member << std::endl;
            continue; // Skip this member
        }

        auto result = target_archive.add_member( member, member.data() );
        if ( !result.ok() ) {
            std::cerr << "Error adding member '" << member.name
                      << "': " << result.what() << std::endl;
            return 3;
        }

        // Copy member symbols
        std::vector<std::string> symbols;
        archive.get_symbols_for_member( member, symbols );
        target_archive.add_symbols_for_member( target_archive.members.back(),
                                               symbols );
    }

    return 0;
}

//------------------------------------------------------------------------------
// Collect global symbols from an ELF file
static void
collect_elf_global_symbols( const ELFIO::elfio&       elf,
                            std::vector<std::string>& gathered_symbols )
{
    for ( const auto& sec : elf.sections ) {
        // Look for the symbol table section
        if ( sec->get_type() == SHT_SYMTAB ) {
            // Access the symbols in the symbol table
            symbol_section_accessor symbols( elf, sec.get() );

            std::string   name;
            Elf64_Addr    value;
            Elf_Xword     size;
            unsigned char bind = 0, type = 0;
            Elf_Half      section_index;
            unsigned char other;

            // Iterate over all sections in the ELF file and gather all symbols
            // that are functions or objects, and have global binding
            for ( Elf_Xword i = 0; i < symbols.get_symbols_num(); ++i ) {
                // Extract symbol properties
                symbols.get_symbol( i, name, value, size, bind, type,
                                    section_index, other );
                // For each global function or object symbol, check that the archive symbol table can find it
                if ( ( type == STT_FUNC || type == STT_OBJECT ||
                       type == STT_TLS || type == STT_COMMON ) &&
                     bind == STB_GLOBAL ) {
                    gathered_symbols.emplace_back( name );
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// Add a new member to the archive
static ario::Result add_new_member( ario&                   archive,
                                    const std::string_view& file_name )
{
    std::filesystem::path full_file_path = file_name;
    if ( !std::filesystem::exists( full_file_path ) ) {
        return { "File does not exist: " + full_file_path.string() };
    }
    ario::Member new_member;
    new_member.name = full_file_path.filename().string();
    new_member.uid  = 0;
    new_member.gid  = 0;
    new_member.mode = 0644;

    std::ifstream input( full_file_path, std::ios::binary );
    if ( !input ) {
        return { "Failed to open file: " + full_file_path.string() };
    }
    const std::string data( ( std::istreambuf_iterator<char>( input ) ),
                            std::istreambuf_iterator<char>() );

    const auto result = archive.add_member( new_member, data );
    if ( !result.ok() ) {
        return result;
    }

    elfio elf;
    if ( elf.load( full_file_path.string() ) ) {
        std::vector<std::string> gathered_symbols;
        collect_elf_global_symbols( elf, gathered_symbols );
        archive.add_symbols_for_member( archive.members.back(),
                                        gathered_symbols );
    }

    return {}; // Return success
}

//------------------------------------------------------------------------------
// Add members from the command line addition list
static int add_new_members( const CommandLineOptions& opts,
                            ARIO::ario&               target_archive )
{
    for ( const auto& file_name : opts.add_files ) {
        std::cout << "Adding member: " << file_name << std::endl;
        auto result = add_new_member( target_archive, file_name );
        if ( !result.ok() ) {
            std::cerr << "Error adding member '" << file_name
                      << "': " << result.what() << std::endl;
            return 3;
        }
    }

    return 0;
}

//------------------------------------------------------------------------------
// Save the new archive to a file
static int save_new_archive( ARIO::ario&        target_archive,
                             const std::string& archive_name )
{
    // Create a temporary filename for the archive
    auto        temp_dir = std::filesystem::temp_directory_path();
    std::string temp_filename;
    do {
        // Generate a random filename
        temp_filename = "ario_tmp_" + std::to_string( std::rand() ) + ".ar";
    } while ( std::filesystem::exists( temp_dir / temp_filename ) );
    std::filesystem::path temp_path = temp_dir / temp_filename;

    target_archive.save( temp_path.string() );

    // Move temporary file to the original one
    std::error_code ec;
    std::filesystem::rename( temp_path, archive_name, ec );
    if ( ec ) {
        std::cerr << "Error renaming temporary file: " << ec.message()
                  << std::endl;
        return 4;
    }
    return 0;
}

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
    auto opts = parse_args( argc, argv );

    // Open existing library or create a new one. In the last case, the library will be empty.
    ario       archive;
    const auto result = archive.load( opts.archive_name );
    if ( !result.ok() ) {
        std::cerr << "Error loading archive: " << result.what() << std::endl;
        return 1;
    }

    // Extract members from the archive
    int retVal = extract_members( opts, archive );
    if ( retVal != 0 )
        return retVal;

    // Check if there are no files to delete or add
    if ( opts.delete_files.empty() && opts.add_files.empty() ) {
        // No files to delete or add. Exiting
        return 0;
    }

    // Create a new (empty) target archive
    ario target_archive;

    // Copy members not included to the command line deletion list
    retVal = copy_members( opts, archive, target_archive );
    if ( retVal != 0 )
        return retVal;

    // Add new members from the command line addition list
    retVal = add_new_members( opts, target_archive );
    if ( retVal != 0 )
        return retVal;

    // Save the new archive
    retVal = save_new_archive( target_archive, opts.archive_name );
    if ( retVal != 0 )
        return retVal;

    return 0;
}
