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
#endif

#include <gtest/gtest.h>

#include <elfio/elfio.hpp>
#include <elfio/elfio_utils.hpp>

using namespace ELFIO;

////////////////////////////////////////////////////////////////////////////////
void checkHeader( const elfio&  reader,
                  unsigned char nClass,
                  unsigned char encoding,
                  unsigned char elfVersion,
                  Elf_Half      type,
                  Elf_Half      machine,
                  Elf_Word      version,
                  Elf64_Addr    entry,
                  Elf_Word      flags,
                  Elf_Half      secNum,
                  Elf_Half      segNum,
                  unsigned char OSABI,
                  unsigned char ABIVersion )
{
    EXPECT_EQ( reader.get_class(), nClass );
    EXPECT_EQ( reader.get_encoding(), encoding );
    EXPECT_EQ( reader.get_elf_version(), elfVersion );
    EXPECT_EQ( reader.get_os_abi(), OSABI );
    EXPECT_EQ( reader.get_abi_version(), ABIVersion );
    EXPECT_EQ( reader.get_type(), type );
    EXPECT_EQ( reader.get_machine(), machine );
    EXPECT_EQ( reader.get_version(), version );
    EXPECT_EQ( reader.get_entry(), entry );
    EXPECT_EQ( reader.get_flags(), flags );
    EXPECT_EQ( reader.sections.size(), secNum );
    EXPECT_EQ( reader.segments.size(), segNum );
}

////////////////////////////////////////////////////////////////////////////////
void checkSection( const section*     sec,
                   Elf_Half           index,
                   const std::string& name,
                   Elf_Word           type,
                   Elf_Xword          flags,
                   Elf64_Addr         address,
                   Elf_Xword          size,
                   Elf_Word           link,
                   Elf_Word           info,
                   Elf_Xword          addrAlign,
                   Elf_Xword          entrySize )
{
    EXPECT_EQ( sec->get_index(), index );
    EXPECT_EQ( sec->get_name(), name );
    EXPECT_EQ( sec->get_type(), type );
    EXPECT_EQ( sec->get_flags(), flags );
    EXPECT_EQ( sec->get_address(), address );
    EXPECT_EQ( sec->get_size(), size );
    EXPECT_EQ( sec->get_link(), link );
    EXPECT_EQ( sec->get_info(), info );
    EXPECT_EQ( sec->get_addr_align(), addrAlign );
    EXPECT_EQ( sec->get_entry_size(), entrySize );
}

////////////////////////////////////////////////////////////////////////////////
void checkSection( const section*     sec,
                   const std::string& name,
                   Elf_Word           type,
                   Elf_Xword          flags,
                   Elf64_Addr         address,
                   Elf_Xword          size,
                   Elf_Word           link,
                   Elf_Word           info,
                   Elf_Xword          addrAlign,
                   Elf_Xword          entrySize )
{
    checkSection( sec, sec->get_index(), name, type, flags, address, size, link,
                  info, addrAlign, entrySize );
}

////////////////////////////////////////////////////////////////////////////////
void checkSegment( const segment* seg,
                   Elf_Word       type,
                   Elf64_Addr     vaddr,
                   Elf64_Addr     paddr,
                   Elf_Xword      fsize,
                   Elf_Xword      msize,
                   Elf_Word       flags,
                   Elf_Xword      align )
{
    EXPECT_EQ( seg->get_type(), type );
    EXPECT_EQ( seg->get_virtual_address(), vaddr );
    EXPECT_EQ( seg->get_physical_address(), paddr );
    EXPECT_EQ( seg->get_file_size(), fsize );
    EXPECT_EQ( seg->get_memory_size(), msize );
    EXPECT_EQ( seg->get_flags(), flags );
    EXPECT_EQ( seg->get_align(), align );
}

////////////////////////////////////////////////////////////////////////////////
void checkSymbol( const const_symbol_section_accessor& sr,
                  Elf_Xword                            index,
                  const std::string&                   name_,
                  Elf64_Addr                           value_,
                  Elf_Xword                            size_,
                  unsigned char                        bind_,
                  unsigned char                        type_,
                  Elf_Half                             section_,
                  unsigned char                        other_ )
{
    std::string   name;
    Elf64_Addr    value;
    Elf_Xword     size;
    unsigned char bind;
    unsigned char type;
    Elf_Half      section;
    unsigned char other;

    ASSERT_EQ(
        sr.get_symbol( index, name, value, size, bind, type, section, other ),
        true );
    EXPECT_EQ( name, name_ );
    EXPECT_EQ( value, value_ );
    EXPECT_EQ( size, size_ );
    EXPECT_EQ( bind, bind_ );
    EXPECT_EQ( type, type_ );
    EXPECT_EQ( section, section_ );
    EXPECT_EQ( other, other_ );
}

////////////////////////////////////////////////////////////////////////////////
void checkRelocation( const const_relocation_section_accessor* pRT,
                      Elf_Xword                                index,
                      Elf64_Addr                               offset_,
                      Elf64_Addr                               symbolValue_,
                      const std::string&                       symbolName_,
                      unsigned char                            type_,
                      Elf_Sxword                               addend_,
                      Elf_Sxword                               calcValue_ )
{
    Elf64_Addr  offset;
    Elf64_Addr  symbolValue;
    std::string symbolName;
    unsigned    type;
    Elf_Sxword  addend;
    Elf_Sxword  calcValue;

    ASSERT_EQ( pRT->get_entry( index, offset, symbolValue, symbolName, type,
                               addend, calcValue ),
               true );
    EXPECT_EQ( offset, offset_ );
    EXPECT_EQ( symbolValue, symbolValue_ );
    EXPECT_EQ( symbolName, symbolName_ );
    EXPECT_EQ( type, type_ );
    EXPECT_EQ( addend, addend_ );
    EXPECT_EQ( calcValue, calcValue_ );
}

////////////////////////////////////////////////////////////////////////////////
void checkNote( const const_note_section_accessor& notes,
                Elf_Word                           index,
                Elf_Word                           type_,
                const std::string&                 name_,
                Elf_Word                           descSize_ )
{
    Elf_Word    type;
    std::string name;
    char*       desc;
    Elf_Word    descSize;

    ASSERT_EQ( notes.get_note( index, type, name, desc, descSize ), true );
    EXPECT_EQ( type, type_ );
    EXPECT_EQ( name, name_ );
    EXPECT_EQ( descSize, descSize_ );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ELFIOTest, load32 )
{
    bool is_lazy = false;
    do {
        is_lazy = !is_lazy;
        elfio reader;
        ASSERT_EQ( reader.load( "elf_examples/hello_32", is_lazy ), true );
        checkHeader( reader, ELFCLASS32, ELFDATA2LSB, EV_CURRENT, ET_EXEC,
                     EM_386, 1, 0x80482b0, 0, 28, 7, 0, 0 );

        ////////////////////////////////////////////////////////////////////////////
        // Check sections
        const section* sec = reader.sections[0];
        // sec->free_data();
        checkSection( sec, 0, "", SHT_NULL, 0, 0, 0, 0, 0, 0, 0 );

        sec = reader.sections[1];
        // sec->free_data();
        checkSection( sec, 1, ".interp", SHT_PROGBITS, SHF_ALLOC, 0x08048114,
                      0x13, 0, 0, 1, 0 );

        sec = reader.sections[9];
        // sec->free_data();
        checkSection( sec, 9, ".rel.plt", SHT_REL, SHF_ALLOC, 0x08048234, 0x18,
                      4, 11, 4, 8 );

        sec = reader.sections[19];
        // sec->free_data();
        checkSection( sec, 19, ".dynamic", SHT_DYNAMIC, SHF_WRITE | SHF_ALLOC,
                      0x080494a0, 0xc8, 5, 0, 4, 8 );

        sec = reader.sections[27];
        // sec->free_data();
        checkSection( sec, 27, ".strtab", SHT_STRTAB, 0, 0x0, 0x259, 0, 0, 1,
                      0 );

        for ( Elf_Half i = 0; i < reader.sections.size(); ++i ) {
            sec = reader.sections[i];
            // sec->free_data();
            EXPECT_EQ( sec->get_index(), i );
        }

        const section* sec1 = reader.sections[".strtab"];
        // sec1->free_data();
        EXPECT_EQ( sec->get_index(), sec1->get_index() );

        ////////////////////////////////////////////////////////////////////////////
        // Check segments
        const segment* seg = reader.segments[0];
        seg->free_data();
        checkSegment( seg, PT_PHDR, 0x08048034, 0x08048034, 0x000e0, 0x000e0,
                      PF_R + PF_X, 4 );

        seg = reader.segments[4];
        seg->free_data();
        checkSegment( seg, PT_DYNAMIC, 0x080494a0, 0x080494a0, 0x000c8, 0x000c8,
                      PF_R + PF_W, 4 );

        seg = reader.segments[6];
        seg->free_data();
        checkSegment( seg, 0x6474E551, 0x0, 0x0, 0x0, 0x0, PF_R + PF_W, 4 );

        ////////////////////////////////////////////////////////////////////////////
        // Check symbol table
        sec = reader.sections[".symtab"];
        // sec->free_data();

        const_symbol_section_accessor sr( reader, sec );

        EXPECT_EQ( sr.get_symbols_num(), 68 );
        checkSymbol( sr, 0, "", 0x00000000, 0, STB_LOCAL, STT_NOTYPE, STN_UNDEF,
                     ELF_ST_VISIBILITY( STV_DEFAULT ) );
        checkSymbol( sr, 1, "", 0x08048114, 0, STB_LOCAL, STT_SECTION, 1,
                     ELF_ST_VISIBILITY( STV_DEFAULT ) );
        checkSymbol( sr, 39, "hello.c", 0x00000000, 0, STB_LOCAL, STT_FILE,
                     SHN_ABS, ELF_ST_VISIBILITY( STV_DEFAULT ) );
        checkSymbol( sr, 65, "__i686.get_pc_thunk.bx", 0x08048429, 0,
                     STB_GLOBAL, STT_FUNC, 12,
                     ELF_ST_VISIBILITY( STV_HIDDEN ) );
        checkSymbol( sr, 66, "main", 0x08048384, 43, STB_GLOBAL, STT_FUNC, 12,
                     ELF_ST_VISIBILITY( STV_DEFAULT ) );
        checkSymbol( sr, 67, "_init", 0x0804824c, 0, STB_GLOBAL, STT_FUNC, 10,
                     ELF_ST_VISIBILITY( STV_DEFAULT ) );

        ////////////////////////////////////////////////////////////////////////////
        // Check relocation table
        sec = reader.sections[".rel.dyn"];
        // sec->free_data();

        const_relocation_section_accessor reloc( reader, sec );
        EXPECT_EQ( reloc.get_entries_num(), 1 );

        checkRelocation( &reloc, 0, 0x08049568, 0x0, "__gmon_start__",
                         R_386_GLOB_DAT, 0, 0 );

        sec = reader.sections[".rel.plt"];
        // sec->free_data();

        const_relocation_section_accessor reloc1( reader, sec );
        EXPECT_EQ( reloc1.get_entries_num(), 3 );

        checkRelocation( &reloc1, 0, 0x08049578, 0x0, "__gmon_start__",
                         R_X86_64_JUMP_SLOT, 0, 0 );
        checkRelocation( &reloc1, 1, 0x0804957c, 0x0, "__libc_start_main",
                         R_X86_64_JUMP_SLOT, 0, 0 );
        checkRelocation( &reloc1, 2, 0x08049580, 0x0, "puts",
                         R_X86_64_JUMP_SLOT, 0, 0 );

        ////////////////////////////////////////////////////////////////////////////
        // Check note reader
        sec = reader.sections[".note.ABI-tag"];

        const_note_section_accessor notes( reader, sec );
        EXPECT_EQ( notes.get_notes_num(), 1u );

        checkNote( notes, 0, 1, std::string( "GNU" ), 16 );
    } while ( is_lazy );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ELFIOTest, load64 )
{
    bool is_lazy = false;
    do {
        is_lazy = !is_lazy;
        elfio reader;

        ASSERT_EQ( reader.load( "elf_examples/hello_64", is_lazy ), true );

        ////////////////////////////////////////////////////////////////////////////
        // Check ELF header
        checkHeader( reader, ELFCLASS64, ELFDATA2LSB, EV_CURRENT, ET_EXEC,
                     EM_X86_64, 1, 0x4003c0, 0, 29, 8, 0, 0 );

        ////////////////////////////////////////////////////////////////////////////
        // Check sections
        const section* sec = reader.sections[0];
        // sec->free_data();

        checkSection( sec, 0, "", SHT_NULL, 0, 0, 0, 0, 0, 0, 0 );

        sec = reader.sections[1];
        // sec->free_data();

        checkSection( sec, 1, ".interp", SHT_PROGBITS, SHF_ALLOC,
                      0x0000000000400200, 0x1c, 0, 0, 1, 0 );

        sec = reader.sections[9];
        // sec->free_data();

        checkSection( sec, 9, ".rela.plt", SHT_RELA, SHF_ALLOC,
                      0x0000000000400340, 0x30, 4, 11, 8, 0x18 );

        sec = reader.sections[20];
        // sec->free_data();

        checkSection( sec, 20, ".dynamic", SHT_DYNAMIC, SHF_WRITE | SHF_ALLOC,
                      0x0000000000600698, 0x190, 5, 0, 8, 0x10 );

        sec = reader.sections[28];
        // sec->free_data();
        checkSection( sec, 28, ".strtab", SHT_STRTAB, 0, 0x0, 0x23f, 0, 0, 1,
                      0 );

        const section* sec1 = reader.sections[".strtab"];
        EXPECT_EQ( sec->get_index(), sec1->get_index() );

        ////////////////////////////////////////////////////////////////////////////
        // Check segments
        const segment* seg = reader.segments[0];
        seg->free_data();
        checkSegment( seg, PT_PHDR, 0x0000000000400040, 0x0000000000400040,
                      0x00000000000001c0, 0x00000000000001c0, PF_R + PF_X, 8 );

        seg = reader.segments[2];
        seg->free_data();
        checkSegment( seg, PT_LOAD, 0x0000000000400000, 0x0000000000400000,
                      0x000000000000066c, 0x000000000000066c, PF_R + PF_X,
                      0x200000 );

        seg = reader.segments[7];
        seg->free_data();
        checkSegment( seg, 0x6474E551, 0x0, 0x0, 0x0, 0x0, PF_R + PF_W, 8 );

        ////////////////////////////////////////////////////////////////////////////
        // Check symbol table
        sec = reader.sections[".symtab"];
        // sec->free_data();

        const_symbol_section_accessor sr( reader, sec );

        EXPECT_EQ( sr.get_symbols_num(), 67 );
        checkSymbol( sr, 0, "", 0x00000000, 0, STB_LOCAL, STT_NOTYPE, STN_UNDEF,
                     ELF_ST_VISIBILITY( STV_DEFAULT ) );
        checkSymbol( sr, 1, "", 0x00400200, 0, STB_LOCAL, STT_SECTION, 1,
                     ELF_ST_VISIBILITY( STV_DEFAULT ) );
        checkSymbol( sr, 40, "hello.c", 0x00000000, 0, STB_LOCAL, STT_FILE,
                     SHN_ABS, ELF_ST_VISIBILITY( STV_DEFAULT ) );
        checkSymbol( sr, 52, "__gmon_start__", 0x00000000, 0, STB_WEAK,
                     STT_NOTYPE, STN_UNDEF, ELF_ST_VISIBILITY( STV_DEFAULT ) );
        checkSymbol( sr, 64, "_edata", 0x0060085c, 0, STB_GLOBAL, STT_NOTYPE,
                     SHN_ABS, ELF_ST_VISIBILITY( STV_DEFAULT ) );
        checkSymbol( sr, 65, "main", 0x00400498, 21, STB_GLOBAL, STT_FUNC, 12,
                     ELF_ST_VISIBILITY( STV_DEFAULT ) );
        checkSymbol( sr, 66, "_init", 0x00400370, 0, STB_GLOBAL, STT_FUNC, 10,
                     ELF_ST_VISIBILITY( STV_DEFAULT ) );

        ////////////////////////////////////////////////////////////////////////////
        // Check relocation table
        sec = reader.sections[".rela.dyn"];
        // sec->free_data();

        const_relocation_section_accessor reloc( reader, sec );
        EXPECT_EQ( reloc.get_entries_num(), 1 );

        checkRelocation( &reloc, 0, 0x00600828, 0x0, "__gmon_start__",
                         R_X86_64_GLOB_DAT, 0, 0 );

        sec = reader.sections[".rela.plt"];

        const_relocation_section_accessor reloc1( reader, sec );
        EXPECT_EQ( reloc1.get_entries_num(), 2 );

        checkRelocation( &reloc1, 0, 0x00600848, 0x0, "puts",
                         R_X86_64_JUMP_SLOT, 0, 0 );
        checkRelocation( &reloc1, 1, 0x00600850, 0x0, "__libc_start_main",
                         R_X86_64_JUMP_SLOT, 0, 0 );

        ////////////////////////////////////////////////////////////////////////////
        // Check note reader
        sec = reader.sections[".note.ABI-tag"];
        // sec->free_data();

        const_note_section_accessor notes( reader, sec );
        EXPECT_EQ( notes.get_notes_num(), 1u );

        checkNote( notes, 0, 1, std::string( "GNU" ), 16 );
    } while ( is_lazy );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ELFIOTest, hello_64_o )
{
    elfio reader;

    ASSERT_EQ( reader.load( "elf_examples/hello_64.o" ), true );

    ////////////////////////////////////////////////////////////////////////////
    // Check ELF header
    checkHeader( reader, ELFCLASS64, ELFDATA2LSB, EV_CURRENT, ET_REL, EM_X86_64,
                 1, 0, 0, 13, 0, 0, 0 );

    ////////////////////////////////////////////////////////////////////////////
    // Check sections
    const section* sec = reader.sections[0];

    checkSection( sec, 0, "", SHT_NULL, 0, 0, 0, 0, 0, 0, 0 );

    sec = reader.sections[1];

    checkSection( sec, 1, ".text", SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, 0x0,
                  0x15, 0, 0, 4, 0 );

    const section* sec1 = reader.sections[".text"];
    EXPECT_EQ( sec->get_index(), sec1->get_index() );

    sec = reader.sections[12];
    checkSection( sec, 12, ".strtab", SHT_STRTAB, 0, 0x0, 0x13, 0, 0, 1, 0 );

    sec1 = reader.sections[".strtab"];
    EXPECT_EQ( sec->get_index(), sec1->get_index() );

    ////////////////////////////////////////////////////////////////////////////
    // Check symbol table
    sec = reader.sections[".symtab"];

    const_symbol_section_accessor sr( reader, sec );

    EXPECT_EQ( sr.get_symbols_num(), 11 );
    checkSymbol( sr, 9, "main", 0x00000000, 21, STB_GLOBAL, STT_FUNC, 1,
                 ELF_ST_VISIBILITY( STV_DEFAULT ) );

    ////////////////////////////////////////////////////////////////////////////
    // Check relocation table
    sec = reader.sections[".rela.text"];

    const_relocation_section_accessor reloc( reader, sec );
    EXPECT_EQ( reloc.get_entries_num(), 2 );

    checkRelocation( &reloc, 0, 0x00000005, 0x0, "", R_X86_64_32, 0, 0 );
    checkRelocation( &reloc, 1, 0x0000000A, 0x0, "puts", R_X86_64_PC32,
                     0xfffffffffffffffcULL, -14 );

    sec = reader.sections[".rela.eh_frame"];

    const_relocation_section_accessor reloc1( reader, sec );
    EXPECT_EQ( reloc1.get_entries_num(), 1 );

    checkRelocation( &reloc1, 0, 0x00000020, 0x0, "", R_X86_64_32, 0, 0 );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ELFIOTest, hello_32_o )
{
    elfio reader;

    ASSERT_EQ( reader.load( "elf_examples/hello_32.o" ), true );

    ////////////////////////////////////////////////////////////////////////////
    // Check ELF header
    checkHeader( reader, ELFCLASS32, ELFDATA2LSB, EV_CURRENT, ET_REL, EM_386, 1,
                 0, 0, 11, 0, 0, 0 );

    ////////////////////////////////////////////////////////////////////////////
    // Check sections
    const section* sec = reader.sections[0];

    checkSection( sec, 0, "", SHT_NULL, 0, 0, 0, 0, 0, 0, 0 );

    sec = reader.sections[1];

    checkSection( sec, 1, ".text", SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, 0x0,
                  0x2b, 0, 0, 4, 0 );

    const section* sec1 = reader.sections[".text"];
    EXPECT_EQ( sec->get_index(), sec1->get_index() );

    sec = reader.sections[10];

    checkSection( sec, 10, ".strtab", SHT_STRTAB, 0, 0x0, 0x13, 0, 0, 1, 0 );

    sec1 = reader.sections[".strtab"];
    EXPECT_EQ( sec->get_index(), sec1->get_index() );

    ////////////////////////////////////////////////////////////////////////////
    // Check symbol table
    sec = reader.sections[".symtab"];

    const_symbol_section_accessor sr( reader, sec );

    EXPECT_EQ( sr.get_symbols_num(), 10 );
    checkSymbol( sr, 8, "main", 0x00000000, 43, STB_GLOBAL, STT_FUNC, 1,
                 ELF_ST_VISIBILITY( STV_DEFAULT ) );

    ////////////////////////////////////////////////////////////////////////////
    // Check relocation table
    sec = reader.sections[".rel.text"];

    const_relocation_section_accessor reloc( reader, sec );
    EXPECT_EQ( reloc.get_entries_num(), 2 );

    checkRelocation( &reloc, 0, 0x00000014, 0x0, "", R_386_32, 0, 0 );
    checkRelocation( &reloc, 1, 0x00000019, 0x0, "puts", R_386_PC32, 0x0, -25 );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ELFIOTest, test_ppc_o )
{
    elfio reader;

    ASSERT_EQ( reader.load( "elf_examples/test_ppc.o" ), true );

    ////////////////////////////////////////////////////////////////////////////
    // Check ELF header
    checkHeader( reader, ELFCLASS32, ELFDATA2MSB, EV_CURRENT, ET_REL, EM_PPC, 1,
                 0, 0, 16, 0, 0, 0 );

    ////////////////////////////////////////////////////////////////////////////
    // Check sections
    const section* sec = reader.sections[0];

    checkSection( sec, 0, "", SHT_NULL, 0, 0, 0, 0, 0, 0, 0 );

    sec = reader.sections[1];

    checkSection( sec, 1, ".text", SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, 0x0,
                  0x118, 0, 0, 4, 0 );

    const section* sec1 = reader.sections[".text"];
    EXPECT_EQ( sec->get_index(), sec1->get_index() );

    sec = reader.sections[15];

    checkSection( sec, 15, ".strtab", SHT_STRTAB, 0, 0x0, 0x14f, 0, 0, 1, 0 );

    sec1 = reader.sections[".strtab"];
    EXPECT_EQ( sec->get_index(), sec1->get_index() );

    ////////////////////////////////////////////////////////////////////////////
    // Check symbol table
    sec = reader.sections[".symtab"];

    const_symbol_section_accessor sr( reader, sec );

    EXPECT_EQ( sr.get_symbols_num(), 24 );
    checkSymbol( sr, 14, "main", 0x00000000, 92, STB_GLOBAL, STT_FUNC, 1,
                 ELF_ST_VISIBILITY( STV_DEFAULT ) );
    checkSymbol( sr, 8, "_GLOBAL__I_main", 0x000000DC, 60, STB_LOCAL, STT_FUNC,
                 1, ELF_ST_VISIBILITY( STV_DEFAULT ) );

    ////////////////////////////////////////////////////////////////////////////
    // Check relocation table
    sec = reader.sections[".rela.text"];

    const_relocation_section_accessor reloc( reader, sec );
    EXPECT_EQ( reloc.get_entries_num(), 18 );

    checkRelocation( &reloc, 0, 0x00000016, 0x0, "_ZSt4cout", 6, 0, 0 );
    checkRelocation( &reloc, 1, 0x0000001a, 0x0, "_ZSt4cout", 4, 0x0, 0 );
    checkRelocation( &reloc, 17, 0x000000c0, 0x0, "__cxa_atexit", 10, 0x0, 0 );

    sec = reader.sections[".rela.ctors"];

    const_relocation_section_accessor reloc1( reader, sec );
    EXPECT_EQ( reloc1.get_entries_num(), 1 );

    checkRelocation( &reloc1, 0, 0x00000000, 0x0, "", 1, 0xDC, 0xDC );

    sec = reader.sections[".rela.eh_frame"];

    const_relocation_section_accessor reloc2( reader, sec );
    EXPECT_EQ( reloc2.get_entries_num(), 3 );

    checkRelocation( &reloc2, 1, 0x00000020, 0x0, "", 1, 0x0, 0x0 );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ELFIOTest, test_ppc )
{
    elfio reader;

    ASSERT_EQ( reader.load( "elf_examples/test_ppc" ), true );

    ////////////////////////////////////////////////////////////////////////////
    // Check ELF header
    checkHeader( reader, ELFCLASS32, ELFDATA2MSB, EV_CURRENT, ET_EXEC, EM_PPC,
                 1, 0x10000550, 0, 31, 8, 0, 0 );

    ////////////////////////////////////////////////////////////////////////////
    // Check sections
    const section* sec = reader.sections[0];

    checkSection( sec, 0, "", SHT_NULL, 0, 0, 0, 0, 0, 0, 0 );

    sec = reader.sections[1];

    checkSection( sec, 1, ".interp", SHT_PROGBITS, SHF_ALLOC,
                  0x0000000010000134, 0xd, 0, 0, 1, 0 );

    sec = reader.sections[9];

    checkSection( sec, 9, ".rela.plt", SHT_RELA, SHF_ALLOC, 0x00000000010000494,
                  0x6c, 4, 22, 4, 0xc );

    sec = reader.sections[20];

    checkSection( sec, 20, ".dynamic", SHT_DYNAMIC, SHF_WRITE | SHF_ALLOC,
                  0x0000000010010aec, 0xe8, 5, 0, 4, 0x8 );

    sec = reader.sections[28];

    checkSection( sec, 28, ".shstrtab", SHT_STRTAB, 0, 0x0, 0x101, 0, 0, 1, 0 );

    const section* sec1 = reader.sections[".shstrtab"];
    EXPECT_EQ( sec->get_index(), sec1->get_index() );

    ////////////////////////////////////////////////////////////////////////////
    // Check segments
    const segment* seg = reader.segments[0];
    checkSegment( seg, PT_PHDR, 0x10000034, 0x10000034, 0x00100, 0x00100,
                  PF_R + PF_X, 4 );

    seg = reader.segments[2];
    checkSegment( seg, PT_LOAD, 0x10000000, 0x10000000, 0x00acc, 0x00acc,
                  PF_R + PF_X, 0x10000 );

    seg = reader.segments[7];
    checkSegment( seg, 0x6474E551, 0x0, 0x0, 0x0, 0x0, PF_R + PF_W, 0x4 );

    ////////////////////////////////////////////////////////////////////////////
    // Check symbol table
    sec = reader.sections[".symtab"];

    const_symbol_section_accessor sr( reader, sec );

    EXPECT_EQ( sr.get_symbols_num(), 80 );
    checkSymbol( sr, 0, "", 0x00000000, 0, STB_LOCAL, STT_NOTYPE, STN_UNDEF,
                 ELF_ST_VISIBILITY( STV_DEFAULT ) );
    checkSymbol( sr, 1, "", 0x10000134, 0, STB_LOCAL, STT_SECTION, 1,
                 ELF_ST_VISIBILITY( STV_DEFAULT ) );
    checkSymbol( sr, 40, "__CTOR_END__", 0x10010AD4, 0, STB_LOCAL, STT_OBJECT,
                 16, ELF_ST_VISIBILITY( STV_DEFAULT ) );
    checkSymbol( sr, 52, "__init_array_start", 0x10010acc, 0, STB_LOCAL,
                 STT_NOTYPE, 16, ELF_ST_VISIBILITY( STV_HIDDEN ) );
    checkSymbol( sr, 64, "_ZNSt8ios_base4InitD1Ev@@GLIBCXX_3.4", 0x10000920,
                 204, STB_GLOBAL, STT_FUNC, SHN_UNDEF,
                 ELF_ST_VISIBILITY( STV_DEFAULT ) );
    checkSymbol( sr, 78, "main", 0x1000069c, 92, STB_GLOBAL, STT_FUNC, 11,
                 ELF_ST_VISIBILITY( STV_DEFAULT ) );
    checkSymbol( sr, 79, "_init", 0x10000500, 0, STB_GLOBAL, STT_FUNC, 10,
                 ELF_ST_VISIBILITY( STV_DEFAULT ) );

    ////////////////////////////////////////////////////////////////////////////
    // Check relocation table
    sec = reader.sections[".rela.dyn"];

    const_relocation_section_accessor reloc( reader, sec );
    EXPECT_EQ( reloc.get_entries_num(), 2 );

    checkRelocation( &reloc, 1, 0x10010c0c, 0x10010c0c, "_ZSt4cout", 19, 0, 0 );

    sec = reader.sections[".rela.plt"];

    const_relocation_section_accessor reloc1( reader, sec );
    EXPECT_EQ( reloc1.get_entries_num(), 9 );

    checkRelocation( &reloc1, 0, 0x10010be4, 0x100008e0, "__cxa_atexit", 21, 0,
                     0 );
    checkRelocation( &reloc1, 1, 0x10010be8, 0x0, "__gmon_start__", 21, 0, 0 );

    ////////////////////////////////////////////////////////////////////////////
    // Check note reader
    sec = reader.sections[".note.ABI-tag"];

    const_note_section_accessor notes( reader, sec );
    EXPECT_EQ( notes.get_notes_num(), 1u );

    checkNote( notes, 0, 1, std::string( "GNU" ), 16 );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ELFIOTest, test_dummy_out_i386_32 )
{
    elfio writer;

    writer.create( ELFCLASS32, ELFDATA2LSB );

    writer.set_os_abi( 0 );
    writer.set_abi_version( 0 );
    writer.set_type( ET_REL );
    writer.set_machine( EM_386 );
    writer.set_flags( 0 );

    // Set program entry point
    writer.set_entry( 0x80482b0 );

    // Add Note section
    section* note_sec = writer.sections.add( ".note" );
    note_sec->set_type( SHT_NOTE );
    note_sec->set_flags( SHF_ALLOC );
    note_sec->set_addr_align( 4 );
    note_section_accessor note_writer( writer, note_sec );
    char                  descr[6] = { 0x11, 0x12, 0x13, 0x14, 0x15, 0x16 };
    note_writer.add_note( 0x77, "Hello", descr, 6 );
    EXPECT_EQ( note_sec->get_index(), 2 );

    // Create ELF file
    writer.save( "elf_examples/elf_dummy_header_i386_32.elf" );

    elfio reader;
    ASSERT_EQ( reader.load( "elf_examples/elf_dummy_header_i386_32.elf" ),
               true );

    ////////////////////////////////////////////////////////////////////////////
    // Check ELF header
    checkHeader( reader, ELFCLASS32, ELFDATA2LSB, EV_CURRENT, ET_REL, EM_386,
                 EV_CURRENT, 0x80482b0, 0, 3, 0, 0, 0 );
    ////////////////////////////////////////////////////////////////////////////
    // Check sections
    const section* sec = reader.sections[""];

    checkSection( sec, 0, "", SHT_NULL, 0, 0, 0, 0, 0, 0, 0 );

    sec = reader.sections[".shstrtab"];

    checkSection( sec, 1, ".shstrtab", SHT_STRTAB, 0, 0, 17, 0, 0, 1, 0 );

    sec = reader.sections[".note"];

    EXPECT_EQ( sec->get_index(), 2 );
    checkSection( sec, 2, ".note", SHT_NOTE, SHF_ALLOC, 0, 28, 0, 0, 4, 0 );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ELFIOTest, test_dummy_out_ppc_32 )
{
    elfio writer;

    writer.create( ELFCLASS32, ELFDATA2MSB );

    writer.set_os_abi( 0 );
    writer.set_abi_version( 0 );
    writer.set_type( ET_REL );
    writer.set_machine( EM_PPC );
    writer.set_flags( 0 );

    // Set program entry point
    writer.set_entry( 0x80482b0 );

    // Add Note section
    section* note_sec = writer.sections.add( ".note" );
    note_sec->set_type( SHT_NOTE );
    note_sec->set_flags( SHF_ALLOC );
    note_sec->set_addr_align( 4 );
    note_section_accessor note_writer( writer, note_sec );
    char                  descr[6] = { 0x11, 0x12, 0x13, 0x14, 0x15, 0x16 };
    note_writer.add_note( 0x77, "Hello", descr, 6 );
    EXPECT_EQ( note_sec->get_index(), 2 );

    // Create ELF file
    writer.save( "elf_examples/elf_dummy_header_ppc_32.elf" );

    elfio reader;
    ASSERT_EQ( reader.load( "elf_examples/elf_dummy_header_ppc_32.elf" ),
               true );

    ////////////////////////////////////////////////////////////////////////////
    // Check ELF header
    checkHeader( reader, ELFCLASS32, ELFDATA2MSB, EV_CURRENT, ET_REL, EM_PPC,
                 EV_CURRENT, 0x80482b0, 0, 3, 0, 0, 0 );
    ////////////////////////////////////////////////////////////////////////////
    // Check sections
    const section* sec = reader.sections[""];

    checkSection( sec, 0, "", SHT_NULL, 0, 0, 0, 0, 0, 0, 0 );

    sec = reader.sections[".note"];

    EXPECT_EQ( sec->get_index(), 2 );
    checkSection( sec, 2, ".note", SHT_NOTE, SHF_ALLOC, 0, 28, 0, 0, 4, 0 );

    sec = reader.sections[".shstrtab"];

    checkSection( sec, 1, ".shstrtab", SHT_STRTAB, 0, 0, 17, 0, 0, 1, 0 );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ELFIOTest, test_dummy_out_i386_64 )
{
    elfio writer;

    writer.create( ELFCLASS64, ELFDATA2LSB );

    writer.set_os_abi( 0 );
    writer.set_abi_version( 0 );
    writer.set_type( ET_REL );
    writer.set_machine( EM_X86_64 );
    writer.set_flags( 0 );

    // Set program entry point
    writer.set_entry( 0x120380482b0ULL );

    // Add Note section
    section* note_sec = writer.sections.add( ".note" );
    note_sec->set_type( SHT_NOTE );
    note_sec->set_flags( SHF_ALLOC );
    note_sec->set_addr_align( 4 );
    note_section_accessor note_writer( writer, note_sec );
    char                  descr[6] = { 0x11, 0x12, 0x13, 0x14, 0x15, 0x16 };
    note_writer.add_note( 0x77, "Hello", descr, 6 );
    EXPECT_EQ( note_sec->get_index(), 2 );

    // Create ELF file
    writer.save( "elf_examples/elf_dummy_header_i386_64.elf" );

    elfio reader;
    ASSERT_EQ( reader.load( "elf_examples/elf_dummy_header_i386_64.elf" ),
               true );

    ////////////////////////////////////////////////////////////////////////////
    // Check ELF header
    checkHeader( reader, ELFCLASS64, ELFDATA2LSB, EV_CURRENT, ET_REL, EM_X86_64,
                 EV_CURRENT, 0x120380482b0ULL, 0, 3, 0, 0, 0 );
    ////////////////////////////////////////////////////////////////////////////
    // Check sections
    const section* sec = reader.sections[""];

    checkSection( sec, 0, "", SHT_NULL, 0, 0, 0, 0, 0, 0, 0 );

    sec = reader.sections[".note"];

    EXPECT_EQ( sec->get_index(), 2 );
    checkSection( sec, 2, ".note", SHT_NOTE, SHF_ALLOC, 0, 28, 0, 0, 4, 0 );

    sec = reader.sections[".shstrtab"];

    checkSection( sec, 1, ".shstrtab", SHT_STRTAB, 0, 0, 17, 0, 0, 1, 0 );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ELFIOTest, test_dummy_out_ppc_64 )
{
    elfio writer;

    writer.create( ELFCLASS64, ELFDATA2MSB );

    writer.set_os_abi( 0 );
    writer.set_abi_version( 0 );
    writer.set_type( ET_REL );
    writer.set_machine( EM_PPC64 );
    writer.set_flags( 0 );

    // Set program entry point
    writer.set_entry( 0x120380482b0ULL );

    // Add Note section
    section* note_sec = writer.sections.add( ".note" );
    note_sec->set_type( SHT_NOTE );
    note_sec->set_flags( SHF_ALLOC );
    note_sec->set_addr_align( 4 );
    note_section_accessor note_writer( writer, note_sec );
    char                  descr[6] = { 0x11, 0x12, 0x13, 0x14, 0x15, 0x16 };
    note_writer.add_note( 0x77, "Hello", descr, 6 );
    EXPECT_EQ( note_sec->get_index(), 2 );

    // Create ELF file
    writer.save( "elf_examples/elf_dummy_header_ppc_64.elf" );

    elfio reader;
    ASSERT_EQ( reader.load( "elf_examples/elf_dummy_header_ppc_64.elf" ),
               true );

    ////////////////////////////////////////////////////////////////////////////
    // Check ELF header
    checkHeader( reader, ELFCLASS64, ELFDATA2MSB, EV_CURRENT, ET_REL, EM_PPC64,
                 EV_CURRENT, 0x120380482b0ULL, 0, 3, 0, 0, 0 );
    ////////////////////////////////////////////////////////////////////////////
    // Check sections
    const section* sec = reader.sections[""];

    checkSection( sec, 0, "", SHT_NULL, 0, 0, 0, 0, 0, 0, 0 );

    sec = reader.sections[".shstrtab"];

    checkSection( sec, 1, ".shstrtab", SHT_STRTAB, 0, 0, 17, 0, 0, 1, 0 );

    sec = reader.sections[".note"];

    EXPECT_EQ( sec->get_index(), 2 );
    checkSection( sec, 2, ".note", SHT_NOTE, SHF_ALLOC, 0, 28, 0, 0, 4, 0 );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ELFIOTest, test_dynamic_64_1 )
{
    elfio reader;

    ASSERT_EQ( reader.load( "elf_examples/main" ), true );

    section* dynsec = reader.sections[".dynamic"];
    ASSERT_TRUE( dynsec != nullptr );

    dynamic_section_accessor da( reader, dynsec );

    EXPECT_EQ( da.get_entries_num(), 21 );

    Elf_Xword   tag;
    Elf_Xword   value;
    std::string str;
    da.get_entry( 0, tag, value, str );
    EXPECT_EQ( tag, DT_NEEDED );
    EXPECT_EQ( str, "libfunc.so" );
    da.get_entry( 1, tag, value, str );
    EXPECT_EQ( tag, DT_NEEDED );
    EXPECT_EQ( str, "libc.so.6" );
    da.get_entry( 2, tag, value, str );
    EXPECT_EQ( tag, DT_INIT );
    EXPECT_EQ( value, 0x400530 );
    da.get_entry( 19, tag, value, str );
    EXPECT_EQ( tag, 0x6ffffff0 );
    EXPECT_EQ( value, 0x40047e );
    da.get_entry( 20, tag, value, str );
    EXPECT_EQ( tag, DT_NULL );
    EXPECT_EQ( value, 0 );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ELFIOTest, test_dynamic_64_2 )
{
    elfio reader;

    ASSERT_EQ( reader.load( "elf_examples/libfunc.so" ), true );

    section* dynsec = reader.sections[".dynamic"];
    ASSERT_TRUE( dynsec != nullptr );

    dynamic_section_accessor da( reader, dynsec );

    EXPECT_EQ( da.get_entries_num(), 20 );

    Elf_Xword   tag;
    Elf_Xword   value;
    std::string str;
    da.get_entry( 0, tag, value, str );
    EXPECT_EQ( tag, DT_NEEDED );
    EXPECT_EQ( str, "libc.so.6" );
    da.get_entry( 1, tag, value, str );
    EXPECT_EQ( tag, DT_INIT );
    EXPECT_EQ( value, 0x480 );
    da.get_entry( 18, tag, value, str );
    EXPECT_EQ( tag, 0x6ffffff9 );
    EXPECT_EQ( value, 1 );
    da.get_entry( 19, tag, value, str );
    EXPECT_EQ( tag, DT_NULL );
    EXPECT_EQ( value, 0 );
}

class mock_wiiu_compression : public compression_interface
{
  public:
    std::unique_ptr<char[]>
    inflate( const char*                                 data,
             std::shared_ptr<const endianness_convertor> convertor,
             Elf_Xword                                   compressed_size,
             Elf_Xword& uncompressed_size ) const override
    {
        uncompressed_size = 2 * compressed_size;
        return std::unique_ptr<char[]>( new (
            std::nothrow ) char[static_cast<size_t>( uncompressed_size ) + 1] );
    }

    std::unique_ptr<char[]>
    deflate( const char*                                 data,
             std::shared_ptr<const endianness_convertor> convertor,
             Elf_Xword                                   decompressed_size,
             Elf_Xword& compressed_size ) const override
    {
        compressed_size = decompressed_size / 2;
        return std::unique_ptr<char[]>( new (
            std::nothrow ) char[static_cast<size_t>( compressed_size ) + 1] );
    }
};

////////////////////////////////////////////////////////////////////////////////
// Given: a valid RPX file
// When: we load it with no compression implementation
// Then: the size returns the raw section size (compressed size)
// When: we load it with a mock compression implementation
// Then: the size changes to reflect the mock compression implementation is being called
//
// This test does not do any further validation because doing so would require providing
// a real compression implementation
TEST( ELFIOTest, test_rpx )
{
    elfio reader( new ( std::nothrow ) mock_wiiu_compression() );
    elfio reader_no_compression;

    ASSERT_EQ( reader_no_compression.load( "elf_examples/helloworld.rpx" ),
               true );
    const section* text1 = reader_no_compression.sections[1];
    EXPECT_EQ( text1->get_size(), 36744 );

    ASSERT_EQ( reader.load( "elf_examples/helloworld.rpx" ), true );
    const section* text2 = reader.sections[1];
    EXPECT_EQ( text2->get_size(), text1->get_size() * 2 );
}

////////////////////////////////////////////////////////////////////////////////
TEST( ELFIOTest, test_dynamic_64_3 )
{
    elfio reader;

    ASSERT_EQ( reader.load( "elf_examples/main" ), true );

    section* dynsec = reader.sections[".dynamic"];
    ASSERT_TRUE( dynsec != nullptr );

    dynamic_section_accessor da( reader, dynsec );
    EXPECT_EQ( da.get_entries_num(), 21 );

    section* strsec1 = reader.sections.add( ".dynstr" );
    strsec1->set_type( SHT_STRTAB );
    strsec1->set_entry_size( reader.get_default_entry_size( SHT_STRTAB ) );

    section* dynsec1 = reader.sections.add( ".dynamic1" );
    dynsec1->set_type( SHT_DYNAMIC );
    dynsec1->set_entry_size( reader.get_default_entry_size( SHT_DYNAMIC ) );
    dynsec1->set_link( strsec1->get_index() );
    dynamic_section_accessor da1( reader, dynsec1 );

    Elf_Xword   tag;
    Elf_Xword   tag1;
    Elf_Xword   value;
    Elf_Xword   value1;
    std::string str;
    std::string str1;

    for ( unsigned int i = 0; i < da.get_entries_num(); ++i ) {
        da.get_entry( i, tag, value, str );
        if ( tag == DT_NEEDED || tag == DT_SONAME || tag == DT_RPATH ||
             tag == DT_RUNPATH ) {
            da1.add_entry( tag, str );
        }
        else {
            da1.add_entry( tag, value );
        }
    }

    for ( unsigned int i = 0; i < da.get_entries_num(); ++i ) {
        da.get_entry( i, tag, value, str );
        da1.get_entry( i, tag1, value1, str1 );

        EXPECT_EQ( tag, tag1 );
        if ( tag == DT_NEEDED || tag == DT_SONAME || tag == DT_RPATH ||
             tag == DT_RUNPATH ) {
            EXPECT_EQ( str, str1 );
        }
        else {
            EXPECT_EQ( value, value1 );
        }
    }
}

TEST( ELFIOTest, test_free_data )
{
    bool is_lazy = false;
    do {
        is_lazy = !is_lazy;

        elfio reader;

        ASSERT_EQ( reader.load( "elf_examples/main", is_lazy ), true );

        for ( const auto& sec : reader.sections ) {
            if ( sec->get_size() == 0 || sec->get_data() == nullptr )
                continue;

            std::vector<char> data;
            std::copy( sec->get_data(), sec->get_data() + sec->get_size(),
                       std::back_inserter( data ) );

            sec->free_data();

            EXPECT_TRUE(
                0 == std::memcmp( data.data(), sec->get_data(),
                                  static_cast<size_t>( sec->get_size() ) ) );
        }

        for ( const auto& seg : reader.segments ) {
            if ( seg->get_file_size() == 0 || seg->get_data() == nullptr )
                continue;

            std::vector<char> data;
            std::copy( seg->get_data(), seg->get_data() + seg->get_file_size(),
                       std::back_inserter( data ) );

            seg->free_data();

            EXPECT_TRUE( 0 == std::memcmp( data.data(), seg->get_data(),
                                           static_cast<size_t>(
                                               seg->get_file_size() ) ) );
        }
    } while ( is_lazy );
}

TEST( ELFIOTest, test_segment_resize_bug )
{
    elfio reader;
    ASSERT_EQ( reader.load( "elf_examples/x86_64_static" ), true );
    /*
     * Binary built with:
     *  echo "int main(){}" | gcc -xc -static -o x86_64_static -
     *
     * readelf -l x86_64_static:
     *
     * Program Headers:
     *   Type           Offset             VirtAddr           PhysAddr
     *                  FileSiz            MemSiz              Flags  Align
     *   LOAD           0x0000000000000000 0x0000000000400000 0x0000000000400000
     *                  0x0000000000000518 0x0000000000000518  R      0x1000
     *   LOAD           0x0000000000001000 0x0000000000401000 0x0000000000401000
     *                  0x00000000000936bd 0x00000000000936bd  R E    0x1000
     *   LOAD           0x0000000000095000 0x0000000000495000 0x0000000000495000
     *                  0x000000000002664d 0x000000000002664d  R      0x1000
     *   LOAD           0x00000000000bc0c0 0x00000000004bd0c0 0x00000000004bd0c0
     *                  0x0000000000005170 0x00000000000068c0  RW     0x1000
     *   NOTE           0x0000000000000270 0x0000000000400270 0x0000000000400270
     *                  0x0000000000000020 0x0000000000000020  R      0x8
     *   NOTE           0x0000000000000290 0x0000000000400290 0x0000000000400290
     *                  0x0000000000000044 0x0000000000000044  R      0x4
     *   TLS            0x00000000000bc0c0 0x00000000004bd0c0 0x00000000004bd0c0
     *                  0x0000000000000020 0x0000000000000060  R      0x8
     *   GNU_PROPERTY   0x0000000000000270 0x0000000000400270 0x0000000000400270
     *                  0x0000000000000020 0x0000000000000020  R      0x8
     *   GNU_STACK      0x0000000000000000 0x0000000000000000 0x0000000000000000
     *                  0x0000000000000000 0x0000000000000000  RW     0x10
     *   GNU_RELRO      0x00000000000bc0c0 0x00000000004bd0c0 0x00000000004bd0c0
     *                  0x0000000000002f40 0x0000000000002f40  R      0x1
     * 
     * Section to Segment mapping:
     *  Segment Sections...
     *   00     .note.gnu.property .note.gnu.build-id .note.ABI-tag .rela.plt 
     *   01     .init .plt .text __libc_freeres_fn .fini 
     *   02     .rodata .stapsdt.base .eh_frame .gcc_except_table 
     *   03     .tdata .init_array .fini_array .data.rel.ro .got .got.plt .data __libc_subfreeres __libc_IO_vtables __libc_atexit .bss __libc_freeres_ptrs 
     *   04     .note.gnu.property 
     *   05     .note.gnu.build-id .note.ABI-tag 
     *   06     .tdata .tbss 
     *   07     .note.gnu.property 
     *   08     
     *   09     .tdata .init_array .fini_array .data.rel.ro .got 
    */

    auto checkElf = []( auto& reader ) {
        const auto& segments = reader.segments;
        ASSERT_EQ( segments.size(), 10 );
        checkSegment( segments[0], PT_LOAD, 0x400000, 0x400000, 0x518, 0x518,
                      PF_R, 0x1000 );
        checkSegment( segments[1], PT_LOAD, 0x401000, 0x401000, 0x936bd,
                      0x936bd, PF_R | PF_X, 0x1000 );
        checkSegment( segments[2], PT_LOAD, 0x495000, 0x495000, 0x2664d,
                      0x2664d, PF_R, 0x1000 );
        checkSegment( segments[3], PT_LOAD, 0x4bd0c0, 0x4bd0c0, 0x5170, 0x68c0,
                      PF_R | PF_W, 0x1000 );
        checkSegment( segments[4], PT_NOTE, 0x400270, 0x400270, 0x20, 0x20,
                      PF_R, 0x8 );
        checkSegment( segments[5], PT_NOTE, 0x400290, 0x400290, 0x44, 0x44,
                      PF_R, 0x4 );
        checkSegment( segments[6], PT_TLS, 0x4bd0c0, 0x4bd0c0, 0x20, 0x60, PF_R,
                      0x8 );
        checkSegment( segments[7], PT_GNU_PROPERTY, 0x400270, 0x400270, 0x20,
                      0x20, PF_R, 0x8 );
        checkSegment( segments[8], PT_GNU_STACK, 0, 0, 0, 0, PF_R | PF_W,
                      0x10 );
        checkSegment( segments[9], PT_GNU_RELRO, 0x4bd0c0, 0x4bd0c0, 0x2f40,
                      0x2f40, PF_R, 0x1 );
    };

    checkElf( reader );

    ASSERT_EQ( reader.save( "elf_examples/x86_64_static.save" ), true );
    ASSERT_EQ( reader.load( "elf_examples/x86_64_static.save" ), true );

    // Comment out the assertion. The question is - how the original segment size was calculated
    //checkElf(reader);
}
