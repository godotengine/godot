#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include "common/linux/synth_elf.h"

#include <assert.h>
#include <elf.h>
#include <stdio.h>
#include <string.h>

#include "common/linux/elf_gnu_compat.h"
#include "common/using_std_string.h"

namespace google_breakpad {
namespace synth_elf {

ELF::ELF(uint16_t machine,
         uint8_t file_class,
         Endianness endianness)
  : Section(endianness),
    addr_size_(file_class == ELFCLASS64 ? 8 : 4),
    program_count_(0),
    program_header_table_(endianness),
    section_count_(0),
    section_header_table_(endianness),
    section_header_strings_(endianness) {
  // Could add support for more machine types here if needed.
  assert(machine == EM_386 ||
         machine == EM_X86_64 ||
         machine == EM_ARM);
  assert(file_class == ELFCLASS32 || file_class == ELFCLASS64);

  start() = 0;
  // Add ELF header
  // e_ident
  // EI_MAG0...EI_MAG3
  D8(ELFMAG0);
  D8(ELFMAG1);
  D8(ELFMAG2);
  D8(ELFMAG3);
  // EI_CLASS
  D8(file_class);
  // EI_DATA
  D8(endianness == kLittleEndian ? ELFDATA2LSB : ELFDATA2MSB);
  // EI_VERSION
  D8(EV_CURRENT);
  // EI_OSABI
  D8(ELFOSABI_SYSV);
  // EI_ABIVERSION
  D8(0);
  // EI_PAD
  Append(7, 0);
  assert(Size() == EI_NIDENT);

  // e_type
  D16(ET_EXEC);  //TODO: allow passing ET_DYN?
  // e_machine
  D16(machine);
  // e_version
  D32(EV_CURRENT);
  // e_entry
  Append(endianness, addr_size_, 0);
  // e_phoff
  Append(endianness, addr_size_, program_header_label_);
  // e_shoff
  Append(endianness, addr_size_, section_header_label_);
  // e_flags
  D32(0);
  // e_ehsize
  D16(addr_size_ == 8 ? sizeof(Elf64_Ehdr) : sizeof(Elf32_Ehdr));
  // e_phentsize
  D16(addr_size_ == 8 ? sizeof(Elf64_Phdr) : sizeof(Elf32_Phdr));
  // e_phnum
  D16(program_count_label_);
  // e_shentsize
  D16(addr_size_ == 8 ? sizeof(Elf64_Shdr) : sizeof(Elf32_Shdr));
  // e_shnum
  D16(section_count_label_);
  // e_shstrndx
  D16(section_header_string_index_);

  // Add an empty section for SHN_UNDEF.
  Section shn_undef;
  AddSection("", shn_undef, SHT_NULL);
}

int ELF::AddSection(const string& name, const Section& section,
                    uint32_t type, uint32_t flags, uint64_t addr,
                    uint32_t link, uint64_t entsize, uint64_t offset) {
  Label offset_label;
  Label string_label(section_header_strings_.Add(name));
  size_t size = section.Size();

  int index = section_count_;
  ++section_count_;

  section_header_table_
    // sh_name
    .D32(string_label)
    // sh_type
    .D32(type)
    // sh_flags
    .Append(endianness(), addr_size_, flags)
    // sh_addr
    .Append(endianness(), addr_size_, addr)
    // sh_offset
    .Append(endianness(), addr_size_, offset_label)
    // sh_size
    .Append(endianness(), addr_size_, size)
    // sh_link
    .D32(link)
    // sh_info
    .D32(0)
    // sh_addralign
    .Append(endianness(), addr_size_, 0)
    // sh_entsize
    .Append(endianness(), addr_size_, entsize);

  sections_.push_back(ElfSection(section, type, addr, offset, offset_label,
                                 size));
  return index;
}

void ELF::AppendSection(ElfSection& section) {
  // NULL and NOBITS sections have no content, so they
  // don't need to be written to the file.
  if (section.type_ == SHT_NULL) {
    section.offset_label_ = 0;
  } else if (section.type_ == SHT_NOBITS) {
    section.offset_label_ = section.offset_;
  } else {
    Mark(&section.offset_label_);
    Append(section);
    Align(4);
  }
}

void ELF::AddSegment(int start, int end, uint32_t type, uint32_t flags) {
  assert(start > 0);
  assert(size_t(start) < sections_.size());
  assert(end > 0);
  assert(size_t(end) < sections_.size());
  ++program_count_;

  // p_type
  program_header_table_.D32(type);

  if (addr_size_ == 8) {
    // p_flags
    program_header_table_.D32(flags);
  }

  size_t filesz = 0;
  size_t memsz = 0;
  bool prev_was_nobits = false;
  for (int i = start; i <= end; ++i) {
    size_t size = sections_[i].size_;
    if (sections_[i].type_ != SHT_NOBITS) {
      assert(!prev_was_nobits);
      // non SHT_NOBITS sections are 4-byte aligned (see AddSection)
      size = (size + 3) & ~3;
      filesz += size;
    } else {
      prev_was_nobits = true;
    }
    memsz += size;
  }

  program_header_table_
    // p_offset
    .Append(endianness(), addr_size_, sections_[start].offset_label_)
    // p_vaddr
    .Append(endianness(), addr_size_, sections_[start].addr_)
    // p_paddr
    .Append(endianness(), addr_size_, sections_[start].addr_)
    // p_filesz
    .Append(endianness(), addr_size_, filesz)
    // p_memsz
    .Append(endianness(), addr_size_, memsz);

  if (addr_size_ == 4) {
    // p_flags
    program_header_table_.D32(flags);
  }

  // p_align
  program_header_table_.Append(endianness(), addr_size_, 0);
}

void ELF::Finish() {
  // Add the section header string table at the end.
  section_header_string_index_ = section_count_;
  //printf(".shstrtab size: %ld\n", section_header_strings_.Size());
  AddSection(".shstrtab", section_header_strings_, SHT_STRTAB);
  //printf("section_count_: %ld, sections_.size(): %ld\n",
  //     section_count_, sections_.size());
  if (program_count_) {
    Mark(&program_header_label_);
    Append(program_header_table_);
  } else {
    program_header_label_ = 0;
  }

  for (vector<ElfSection>::iterator it = sections_.begin();
       it < sections_.end(); ++it) {
    AppendSection(*it);
  }
  section_count_label_ = section_count_;
  program_count_label_ = program_count_;

  // Section header table starts here.
  Mark(&section_header_label_);
  Append(section_header_table_);
}

SymbolTable::SymbolTable(Endianness endianness,
                         size_t addr_size,
                         StringTable& table) : Section(endianness),
                                               table_(table) {
#ifndef NDEBUG
  addr_size_ = addr_size;
#endif
  assert(addr_size_ == 4 || addr_size_ == 8);
}

void SymbolTable::AddSymbol(const string& name, uint32_t value,
                            uint32_t size, unsigned info, uint16_t shndx) {
  assert(addr_size_ == 4);
  D32(table_.Add(name));
  D32(value);
  D32(size);
  D8(info);
  D8(0); // other
  D16(shndx);
}

void SymbolTable::AddSymbol(const string& name, uint64_t value,
                            uint64_t size, unsigned info, uint16_t shndx) {
  assert(addr_size_ == 8);
  D32(table_.Add(name));
  D8(info);
  D8(0); // other
  D16(shndx);
  D64(value);
  D64(size);
}

void Notes::AddNote(int type, const string& name, const uint8_t* desc_bytes,
                    size_t desc_size) {
  // Elf32_Nhdr and Elf64_Nhdr are exactly the same.
  Elf32_Nhdr note_header;
  memset(&note_header, 0, sizeof(note_header));
  note_header.n_namesz = name.length() + 1;
  note_header.n_descsz = desc_size;
  note_header.n_type = type;

  Append(reinterpret_cast<const uint8_t*>(&note_header),
         sizeof(note_header));
  AppendCString(name);
  Align(4);
  Append(desc_bytes, desc_size);
  Align(4);
}

}  // namespace synth_elf
}  // namespace google_breakpad
