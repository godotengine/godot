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

/*
 * This example shows how to create ELF executable file for Linux on x86-64
 *
 * Instructions:
 * 1. Compile and link this file with ELFIO library
 *    g++ writer.cpp -o writer
 * 2. Execute result file writer
 *    ./writer
 * 3. Add executable flag for the output file
 *    chmod +x hello_x86_64
 * 4. Run the result file:
 *    ./hello_x86_64
 */

#include <elfio/elfio.hpp>

using namespace ELFIO;

const Elf64_Addr CODE_ADDR = 0x00401000;
const Elf_Xword  PAGE_SIZE = 0x1000;
const Elf64_Addr DATA_ADDR = CODE_ADDR + PAGE_SIZE;

int main( void )
{
    elfio writer;

    // You can't proceed without this function call!
    writer.create( ELFCLASS64, ELFDATA2LSB );

    writer.set_os_abi( ELFOSABI_LINUX );
    writer.set_type( ET_EXEC );
    writer.set_machine( EM_X86_64 );

    // Create code section
    section* text_sec = writer.sections.add( ".text" );
    text_sec->set_type( SHT_PROGBITS );
    text_sec->set_flags( SHF_ALLOC | SHF_EXECINSTR );
    text_sec->set_addr_align( 0x10 );

    // Add data into it
    char text[] = {
        '\xB8', '\x04', '\x00', '\x00', '\x00', // mov eax, 4
        '\xBB', '\x01', '\x00', '\x00', '\x00', // mov ebx, 1
        '\xB9', '\x00', '\x00', '\x00', '\x00', // mov ecx, msg
        '\xBA', '\x0E', '\x00', '\x00', '\x00', // mov edx, 14
        '\xCD', '\x80',                         // int 0x80
        '\xB8', '\x01', '\x00', '\x00', '\x00', // mov eax, 1
        '\xCD', '\x80'                          // int 0x80
    };
    // Adjust data address for 'msg'
    *(std::uint32_t*)( text + 11 ) = DATA_ADDR;

    text_sec->set_data( text, sizeof( text ) );

    // Create a loadable segment
    segment* text_seg = writer.segments.add();
    text_seg->set_type( PT_LOAD );
    text_seg->set_virtual_address( CODE_ADDR );
    text_seg->set_physical_address( CODE_ADDR );
    text_seg->set_flags( PF_X | PF_R );
    text_seg->set_align( PAGE_SIZE );

    // Add code section into program segment
    text_seg->add_section( text_sec, text_sec->get_addr_align() );

    // Create data section
    section* data_sec = writer.sections.add( ".data" );
    data_sec->set_type( SHT_PROGBITS );
    data_sec->set_flags( SHF_ALLOC | SHF_WRITE );
    data_sec->set_addr_align( 0x4 );

    char data[] = {
        '\x48', '\x65', '\x6C', '\x6C', '\x6F', // msg: db   'Hello, World!', 10
        '\x2C', '\x20', '\x57', '\x6F', '\x72',
        '\x6C', '\x64', '\x21', '\x0A' };
    data_sec->set_data( data, sizeof( data ) );

    // Create a read/write segment
    segment* data_seg = writer.segments.add();
    data_seg->set_type( PT_LOAD );
    data_seg->set_virtual_address( DATA_ADDR );
    data_seg->set_physical_address( DATA_ADDR );
    data_seg->set_flags( PF_W | PF_R );
    data_seg->set_align( PAGE_SIZE );

    // Add code section into program segment
    data_seg->add_section( data_sec, data_sec->get_addr_align() );

    // Add optional signature for the file producer
    section* note_sec = writer.sections.add( ".note" );
    note_sec->set_type( SHT_NOTE );
    note_sec->set_addr_align( 1 );

    note_section_accessor note_writer( writer, note_sec );
    note_writer.add_note( 0x01, "Created by ELFIO", 0, 0 );
    char descr[6] = { 0x31, 0x32, 0x33, 0x34, 0x35, 0x36 };
    note_writer.add_note( 0x01, "Never easier!", descr, sizeof( descr ) );

    // Setup entry point. Usually, a linker sets this address on base of
    // ‘_start’ label.
    // In this example, the code starts at the first address of the
    // 'text_seg' segment. Therefore, the start address is set
    // to be equal to the segment location
    writer.set_entry( text_seg->get_virtual_address() );

    // Create ELF file
    writer.save( "hello_x86_64" );

    return 0;
}
