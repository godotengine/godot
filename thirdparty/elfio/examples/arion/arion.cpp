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
// arion.cpp
//
// This example demonstrates how to use the ARIO library to inspect and list the
// contents of a UNIX archive (.ar) file. It provides a simple command-line tool
// that displays information about each member of the archive, including its name,
// size, file mode, and any associated symbols.
//
// Purpose:
//   - Showcase ARIO’s API for reading and iterating over archive members.
//   - Provide a minimal example for archive inspection and symbol listing.
//
// Abilities:
//   - Loads and parses a specified archive file.
//   - Lists all members with their names, sizes, and file modes.
//   - Displays symbols associated with each member, if available.
//   - Command-line interface: Accepts the archive file name as an argument.
//
// This example serves as a reference for basic ARIO usage and as a foundation
// for building custom archive inspection tools.
//------------------------------------------------------------------------------

#include <iostream>
#include <ario/ario.hpp>

using namespace ARIO;

int main( int argc, char** argv )
{
    if ( argc != 2 ) {
        std::cout << "Usage: arion <file_name>" << std::endl;
        return 1;
    }

    ario archive;

    const auto result = archive.load( argv[1] );
    if ( !result.ok() ) {
        std::cerr << "Error loading archive: " << result.what() << std::endl;
        return 1;
    }

    for ( const auto& member : archive.members ) {
        std::cout << "Member: " << std::setw( 40 ) << std::left << member.name
                  << " Size: " << std::setw( 8 ) << std::right << member.size
                  << " Mode: " << std::setw( 3 ) << std::oct << member.mode
                  << std::dec << std::endl;
        std::vector<std::string> symbols;
        if ( archive.get_symbols_for_member( member, symbols ).ok() ) {
            auto first_time = true;
            for ( const auto& symbol : symbols ) {
                if ( first_time ) {
                    std::cout << "    ";
                    first_time = false;
                }
                else {
                    std::cout << ", ";
                }
                std::cout << symbol;
            }
            if ( !first_time ) {
                std::cout << std::endl;
            }
        }
    }

    return 0;
}
