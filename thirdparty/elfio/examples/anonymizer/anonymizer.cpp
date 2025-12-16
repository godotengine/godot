/*
anonymizer.cpp - Overwrites string table for a function name.

Copyright (C) 2017 by Martin Bickel
Copyright (C) 2020 by Serge Lamikhov-Center

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

/*
To run the example, you may use the following script:

#!/usr/bin/bash

make
cp anonymizer temp.elf
readelf -a temp.elf > before.txt
./anonymizer temp.elf
readelf -a temp.elf > after.txt
diff before.txt after.txt

*/

#ifdef _MSC_VER
#define _SCL_SECURE_NO_WARNINGS
#define ELFIO_NO_INTTYPES
#endif

#include <string>
#include <iostream>
#include <fstream>
#include <elfio/elfio.hpp>

using namespace ELFIO;

void overwrite_data( const std::string& filename,
                     Elf64_Off          offset,
                     const std::string& str )
{
    std::ofstream file( filename,
                        std::ios::in | std::ios::out | std::ios::binary );
    if ( !file )
        throw "Error opening file" + filename;
    std::string data( str.length(), '-' );
    file.seekp( (std::streampos)offset );
    file.write( data.c_str(), data.length() + 1 );
}

void process_string_table( const section* s, const std::string& filename )
{
    std::cout << "Info: processing string table section" << std::endl;
    size_t index = 1;
    while ( index < s->get_size() ) {
        auto str = std::string( s->get_data() + index );
        // For the example purpose, we rename main function name only
        if ( str == "main" )
            overwrite_data( filename, s->get_offset() + index, str );
        index += str.length() + 1;
    }
}

int main( int argc, char** argv )
{
    if ( argc != 2 ) {
        std::cout << "Usage: anonymizer <file_name>\n";
        return 1;
    }

    std::string filename = argv[1];

    elfio reader;

    if ( !reader.load( filename ) ) {
        std::cerr << "File " << filename
                  << " is not found or it is not an ELF file\n";
        return 1;
    }

    for ( const auto& section : reader.sections ) {
        if ( section->get_type() == SHT_STRTAB &&
             std::string( section->get_name() ) == std::string( ".strtab" ) ) {
            process_string_table( section.get(), filename );
        }
    }
    return 0;
}
