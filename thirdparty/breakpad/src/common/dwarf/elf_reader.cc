// Copyright 2005 Google LLC
// Author: chatham@google.com (Andrew Chatham)
// Author: satorux@google.com (Satoru Takabayashi)
//
// Code for reading in ELF files.
//
// For information on the ELF format, see
// http://www.x86.org/ftp/manuals/tools/elf.pdf
//
// I also liked:
// http://www.caldera.com/developers/gabi/1998-04-29/contents.html
//
// A note about types: When dealing with the file format, we use types
// like Elf32_Word, but in the public interfaces we treat all
// addresses as uint64. As a result, we should be able to symbolize
// 64-bit binaries from a 32-bit process (which we don't do,
// anyway). size_t should therefore be avoided, except where required
// by things like mmap().
//
// Although most of this code can deal with arbitrary ELF files of
// either word size, the public ElfReader interface only examines
// files loaded into the current address space, which must all match
// the machine's native word size. This code cannot handle ELF files
// with a non-native byte ordering.
//
// TODO(chatham): It would be nice if we could accomplish this task
// without using malloc(), so we could use it as the process is dying.

#ifndef _GNU_SOURCE
#define _GNU_SOURCE  // needed for pread()
#endif

#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include <fcntl.h>
#include <limits.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <map>
#include <string>
#include <string_view>
#include <vector>
// TODO(saugustine): Add support for compressed debug.
// Also need to add configure tests for zlib.
//#include "zlib.h"

#include "third_party/musl/include/elf.h"
#include "elf_reader.h"
#include "common/using_std_string.h"

// EM_AARCH64 is not defined by elf.h of GRTE v3 on x86.
// TODO(dougkwan): Remove this when v17 is retired.
#if !defined(EM_AARCH64)
#define EM_AARCH64      183             /* ARM AARCH64 */
#endif

// Map Linux macros to their Apple equivalents.
#if __APPLE__
#ifndef __LITTLE_ENDIAN
#define __LITTLE_ENDIAN __ORDER_LITTLE_ENDIAN__
#endif  // __LITTLE_ENDIAN
#ifndef __BIG_ENDIAN
#define __BIG_ENDIAN __ORDER_BIG_ENDIAN__
#endif  // __BIG_ENDIAN
#ifndef __BYTE_ORDER
#define __BYTE_ORDER __BYTE_ORDER__
#endif  // __BYTE_ORDER
#endif  // __APPLE__

// TODO(dthomson): Can be removed once all Java code is using the Google3
// launcher. We need to avoid processing PLT functions as it causes memory
// fragmentation in malloc, which is fixed in tcmalloc - and if the Google3
// launcher is used the JVM will then use tcmalloc. b/13735638
//DEFINE_bool(elfreader_process_dynsyms, true,
//            "Activate PLT function processing");

using std::vector;

namespace {

// The lowest bit of an ARM symbol value is used to indicate a Thumb address.
const int kARMThumbBitOffset = 0;

// Converts an ARM Thumb symbol value to a true aligned address value.
template <typename T>
T AdjustARMThumbSymbolValue(const T& symbol_table_value) {
  return symbol_table_value & ~(1 << kARMThumbBitOffset);
}

// Names of PLT-related sections.
const char kElfPLTRelSectionName[] = ".rel.plt";      // Use Rel struct.
const char kElfPLTRelaSectionName[] = ".rela.plt";    // Use Rela struct.
const char kElfPLTSectionName[] = ".plt";
const char kElfDynSymSectionName[] = ".dynsym";

const int kX86PLTCodeSize = 0x10;  // Size of one x86 PLT function in bytes.
const int kARMPLTCodeSize = 0xc;
const int kAARCH64PLTCodeSize = 0x10;

const int kX86PLT0Size = 0x10;  // Size of the special PLT0 entry.
const int kARMPLT0Size = 0x14;
const int kAARCH64PLT0Size = 0x20;

// Suffix for PLT functions when it needs to be explicitly identified as such.
const char kPLTFunctionSuffix[] = "@plt";

// Replace callsites of this function to std::string_view::starts_with after
// adopting C++20.
bool StringViewStartsWith(std::string_view sv, std::string_view prefix) {
  return sv.compare(0, prefix.size(), prefix) == 0;
}

}  // namespace

namespace google_breakpad {

template <class ElfArch> class ElfReaderImpl;

// 32-bit and 64-bit ELF files are processed exactly the same, except
// for various field sizes. Elf32 and Elf64 encompass all of the
// differences between the two formats, and all format-specific code
// in this file is templated on one of them.
class Elf32 {
 public:
  typedef Elf32_Ehdr Ehdr;
  typedef Elf32_Shdr Shdr;
  typedef Elf32_Phdr Phdr;
  typedef Elf32_Word Word;
  typedef Elf32_Sym Sym;
  typedef Elf32_Rel Rel;
  typedef Elf32_Rela Rela;

  // What should be in the EI_CLASS header.
  static const int kElfClass = ELFCLASS32;

  // Given a symbol pointer, return the binding type (eg STB_WEAK).
  static char Bind(const Elf32_Sym* sym) {
    return ELF32_ST_BIND(sym->st_info);
  }
  // Given a symbol pointer, return the symbol type (eg STT_FUNC).
  static char Type(const Elf32_Sym* sym) {
    return ELF32_ST_TYPE(sym->st_info);
  }

  // Extract the symbol index from the r_info field of a relocation.
  static int r_sym(const Elf32_Word r_info) {
    return ELF32_R_SYM(r_info);
  }
};


class Elf64 {
 public:
  typedef Elf64_Ehdr Ehdr;
  typedef Elf64_Shdr Shdr;
  typedef Elf64_Phdr Phdr;
  typedef Elf64_Word Word;
  typedef Elf64_Sym Sym;
  typedef Elf64_Rel Rel;
  typedef Elf64_Rela Rela;

  // What should be in the EI_CLASS header.
  static const int kElfClass = ELFCLASS64;

  static char Bind(const Elf64_Sym* sym) {
    return ELF64_ST_BIND(sym->st_info);
  }
  static char Type(const Elf64_Sym* sym) {
    return ELF64_ST_TYPE(sym->st_info);
  }
  static int r_sym(const Elf64_Xword r_info) {
    return ELF64_R_SYM(r_info);
  }
};


// ElfSectionReader mmaps a section of an ELF file ("section" is ELF
// terminology). The ElfReaderImpl object providing the section header
// must exist for the lifetime of this object.
//
// The motivation for mmaping individual sections of the file is that
// many Google executables are large enough when unstripped that we
// have to worry about running out of virtual address space.
//
// For compressed sections we have no choice but to allocate memory.
template<class ElfArch>
class ElfSectionReader {
 public:
  ElfSectionReader(const char* cname, const string& path, int fd,
                   const typename ElfArch::Shdr& section_header)
      : contents_aligned_(NULL),
        contents_(NULL),
        header_(section_header) {
    // Back up to the beginning of the page we're interested in.
    const size_t additional = header_.sh_offset % getpagesize();
    const size_t offset_aligned = header_.sh_offset - additional;
    section_size_ = header_.sh_size;
    size_aligned_ = section_size_ + additional;
    // If the section has been stripped or is empty, do not attempt
    // to process its contents.
    if (header_.sh_type == SHT_NOBITS || header_.sh_size == 0)
      return;
    // extra sh_type check for string table.
    std::string_view name{cname};
    if ((name == ".strtab" || name == ".shstrtab") &&
        header_.sh_type != SHT_STRTAB) {
      fprintf(stderr,
              "Invalid sh_type for string table section: expected "
              "SHT_STRTAB or SHT_DYNSYM, but got %d\n",
              header_.sh_type);
      return;
    }

    contents_aligned_ = mmap(NULL, size_aligned_, PROT_READ, MAP_SHARED,
                             fd, offset_aligned);
    // Set where the offset really should begin.
    contents_ = reinterpret_cast<char*>(contents_aligned_) +
                (header_.sh_offset - offset_aligned);

    // Check for and handle any compressed contents.
    //if (StringViewStartsWith(name, ".zdebug_"))
    //  DecompressZlibContents();
    // TODO(saugustine): Add support for proposed elf-section flag
    // "SHF_COMPRESS".
  }

  ~ElfSectionReader() {
    if (contents_aligned_ != NULL)
      munmap(contents_aligned_, size_aligned_);
    else
      delete[] contents_;
  }

  // Return the section header for this section.
  typename ElfArch::Shdr const& header() const { return header_; }

  // Return memory at the given offset within this section.
  const char* GetOffset(typename ElfArch::Word bytes) const {
    return contents_ + bytes;
  }

  const char* contents() const { return contents_; }
  size_t section_size() const { return section_size_; }

 private:
  // page-aligned file contents
  void* contents_aligned_;
  // contents as usable by the client. For non-compressed sections,
  // pointer within contents_aligned_ to where the section data
  // begins; for compressed sections, pointer to the decompressed
  // data.
  char* contents_;
  // size of contents_aligned_
  size_t size_aligned_;
  // size of contents.
  size_t section_size_;
  const typename ElfArch::Shdr header_;
};

// An iterator over symbols in a given section. It handles walking
// through the entries in the specified section and mapping symbol
// entries to their names in the appropriate string table (in
// another section).
template<class ElfArch>
class SymbolIterator {
 public:
  SymbolIterator(ElfReaderImpl<ElfArch>* reader,
                 typename ElfArch::Word section_type)
      : symbol_section_(reader->GetSectionByType(section_type)),
        string_section_(NULL),
        num_symbols_in_section_(0),
        symbol_within_section_(0) {

    // If this section type doesn't exist, leave
    // num_symbols_in_section_ as zero, so this iterator is already
    // done().
    if (symbol_section_ != NULL) {
      num_symbols_in_section_ = symbol_section_->header().sh_size /
                                symbol_section_->header().sh_entsize;

      // Symbol sections have sh_link set to the section number of
      // the string section containing the symbol names.
      string_section_ = reader->GetSection(symbol_section_->header().sh_link);
    }
  }

  // Return true iff we have passed all symbols in this section.
  bool done() const {
    return symbol_within_section_ >= num_symbols_in_section_;
  }

  // Advance to the next symbol in this section.
  // REQUIRES: !done()
  void Next() { ++symbol_within_section_; }

  // Return a pointer to the current symbol.
  // REQUIRES: !done()
  const typename ElfArch::Sym* GetSymbol() const {
    return reinterpret_cast<const typename ElfArch::Sym*>(
        symbol_section_->GetOffset(symbol_within_section_ *
                                   symbol_section_->header().sh_entsize));
  }

  // Return the name of the current symbol, NULL if it has none.
  // REQUIRES: !done()
  const char* GetSymbolName() const {
    int name_offset = GetSymbol()->st_name;
    if (name_offset == 0)
      return NULL;
    return string_section_->GetOffset(name_offset);
  }

  int GetCurrentSymbolIndex() const {
    return symbol_within_section_;
  }

 private:
  const ElfSectionReader<ElfArch>* const symbol_section_;
  const ElfSectionReader<ElfArch>* string_section_;
  int num_symbols_in_section_;
  int symbol_within_section_;
};


// Copied from strings/strutil.h.  Per chatham,
// this library should not depend on strings.

static inline bool MyHasSuffixString(const string& str, const string& suffix) {
  int len = str.length();
  int suflen = suffix.length();
  return (suflen <= len) && (str.compare(len-suflen, suflen, suffix) == 0);
}


// ElfReader loads an ELF binary and can provide information about its
// contents. It is most useful for matching addresses to function
// names. It does not understand debugging formats (eg dwarf2), so it
// can't print line numbers. It takes a path to an elf file and a
// readable file descriptor for that file, which it does not assume
// ownership of.
template<class ElfArch>
class ElfReaderImpl {
 public:
  explicit ElfReaderImpl(const string& path, int fd)
      : path_(path),
        fd_(fd),
        section_headers_(NULL),
        program_headers_(NULL),
        opd_section_(NULL),
        base_for_text_(0),
        plts_supported_(false),
        plt_code_size_(0),
        plt0_size_(0),
        visited_relocation_entries_(false) {
    string error;
    is_dwp_ = MyHasSuffixString(path, ".dwp");
    ParseHeaders(fd, path);
    // Currently we need some extra information for PowerPC64 binaries
    // including a way to read the .opd section for function descriptors and a
    // way to find the linked base for function symbols.
    if (header_.e_machine == EM_PPC64) {
      // "opd_section_" must always be checked for NULL before use.
      opd_section_ = GetSectionInfoByName(".opd", &opd_info_);
      for (unsigned int k = 0u; k < GetNumSections(); ++k) {
        std::string_view name{GetSectionName(section_headers_[k].sh_name)};
        if (StringViewStartsWith(name, ".text")) {
          base_for_text_ =
              section_headers_[k].sh_addr - section_headers_[k].sh_offset;
          break;
        }
      }
    }
    // Turn on PLTs.
    if (header_.e_machine == EM_386 || header_.e_machine == EM_X86_64) {
      plt_code_size_ = kX86PLTCodeSize;
      plt0_size_ = kX86PLT0Size;
      plts_supported_ = true;
    } else if (header_.e_machine == EM_ARM) {
      plt_code_size_ = kARMPLTCodeSize;
      plt0_size_ = kARMPLT0Size;
      plts_supported_ = true;
    } else if (header_.e_machine == EM_AARCH64) {
      plt_code_size_ = kAARCH64PLTCodeSize;
      plt0_size_ = kAARCH64PLT0Size;
      plts_supported_ = true;
    }
  }

  ~ElfReaderImpl() {
    for (unsigned int i = 0u; i < sections_.size(); ++i)
      delete sections_[i];
    delete [] section_headers_;
    delete [] program_headers_;
  }

  // Examine the headers of the file and return whether the file looks
  // like an ELF file for this architecture. Takes an already-open
  // file descriptor for the candidate file, reading in the prologue
  // to see if the ELF file appears to match the current
  // architecture. If error is non-NULL, it will be set with a reason
  // in case of failure.
  static bool IsArchElfFile(int fd, string* error) {
    unsigned char header[EI_NIDENT];
    if (pread(fd, header, sizeof(header), 0) != sizeof(header)) {
      if (error != NULL) *error = "Could not read header";
      return false;
    }

    if (memcmp(header, ELFMAG, SELFMAG) != 0) {
      if (error != NULL) *error = "Missing ELF magic";
      return false;
    }

    if (header[EI_CLASS] != ElfArch::kElfClass) {
      if (error != NULL) *error = "Different word size";
      return false;
    }

    int endian = 0;
    if (header[EI_DATA] == ELFDATA2LSB)
      endian = __LITTLE_ENDIAN;
    else if (header[EI_DATA] == ELFDATA2MSB)
      endian = __BIG_ENDIAN;
    if (endian != __BYTE_ORDER) {
      if (error != NULL) *error = "Different byte order";
      return false;
    }

    return true;
  }

  // Return true if we can use this symbol in Address-to-Symbol map.
  bool CanUseSymbol(const char* name, const typename ElfArch::Sym* sym) {
    // For now we only save FUNC and NOTYPE symbols. For now we just
    // care about functions, but some functions written in assembler
    // don't have a proper ELF type attached to them, so we store
    // NOTYPE symbols as well. The remaining significant type is
    // OBJECT (eg global variables), which represent about 25% of
    // the symbols in a typical google3 binary.
    if (ElfArch::Type(sym) != STT_FUNC &&
        ElfArch::Type(sym) != STT_NOTYPE) {
      return false;
    }

    // Target specific filtering.
    switch (header_.e_machine) {
    case EM_AARCH64:
    case EM_ARM:
      // Filter out '$x' special local symbols used by tools
      return name[0] != '$' || ElfArch::Bind(sym) != STB_LOCAL;
    case EM_X86_64:
      // Filter out read-only constants like .LC123.
      return name[0] != '.' || ElfArch::Bind(sym) != STB_LOCAL;
    default:
      return true;
    }
  }

  // Iterate over the symbols in a section, either SHT_DYNSYM or
  // SHT_SYMTAB. Add all symbols to the given SymbolMap.
  /*
  void GetSymbolPositions(SymbolMap* symbols,
                          typename ElfArch::Word section_type,
                          uint64_t mem_offset,
                          uint64_t file_offset) {
    // This map is used to filter out "nested" functions.
    // See comment below.
    AddrToSymMap addr_to_sym_map;
    for (SymbolIterator<ElfArch> it(this, section_type);
         !it.done(); it.Next()) {
      const char* name = it.GetSymbolName();
      if (name == NULL)
        continue;
      const typename ElfArch::Sym* sym = it.GetSymbol();
      if (CanUseSymbol(name, sym)) {
        const int sec = sym->st_shndx;

        // We don't support special section indices. The most common
        // is SHN_ABS, for absolute symbols used deep in the bowels of
        // glibc. Also ignore any undefined symbols.
        if (sec == SHN_UNDEF ||
            (sec >= SHN_LORESERVE && sec <= SHN_HIRESERVE)) {
          continue;
        }

        const typename ElfArch::Shdr& hdr = section_headers_[sec];

        // Adjust for difference between where we expected to mmap
        // this section, and where it was actually mmapped.
        const int64_t expected_base = hdr.sh_addr - hdr.sh_offset;
        const int64_t real_base = mem_offset - file_offset;
        const int64_t adjust = real_base - expected_base;

        uint64_t start = sym->st_value + adjust;

        // Adjust function symbols for PowerPC64 by dereferencing and adjusting
        // the function descriptor to get the function address.
        if (header_.e_machine == EM_PPC64 && ElfArch::Type(sym) == STT_FUNC) {
          const uint64_t opd_addr =
              AdjustPPC64FunctionDescriptorSymbolValue(sym->st_value);
          // Only adjust the returned value if the function address was found.
          if (opd_addr != sym->st_value) {
            const int64_t adjust_function_symbols =
                real_base - base_for_text_;
            start = opd_addr + adjust_function_symbols;
          }
        }

        addr_to_sym_map.push_back(std::make_pair(start, sym));
      }
    }
    std::sort(addr_to_sym_map.begin(), addr_to_sym_map.end(), &AddrToSymSorter);
    addr_to_sym_map.erase(std::unique(addr_to_sym_map.begin(),
                                      addr_to_sym_map.end(), &AddrToSymEquals),
                          addr_to_sym_map.end());

    // Squeeze out any "nested functions".
    // Nested functions are not allowed in C, but libc plays tricks.
    //
    // For example, here is disassembly of /lib64/tls/libc-2.3.5.so:
    //   0x00000000000aa380 <read+0>:             cmpl   $0x0,0x2781b9(%rip)
    //   0x00000000000aa387 <read+7>:             jne    0xaa39b <read+27>
    //   0x00000000000aa389 <__read_nocancel+0>:  mov    $0x0,%rax
    //   0x00000000000aa390 <__read_nocancel+7>:  syscall
    //   0x00000000000aa392 <__read_nocancel+9>:  cmp $0xfffffffffffff001,%rax
    //   0x00000000000aa398 <__read_nocancel+15>: jae    0xaa3ef <read+111>
    //   0x00000000000aa39a <__read_nocancel+17>: retq
    //   0x00000000000aa39b <read+27>:            sub    $0x28,%rsp
    //   0x00000000000aa39f <read+31>:            mov    %rdi,0x8(%rsp)
    //   ...
    // Without removing __read_nocancel, symbolizer will return NULL
    // given e.g. 0xaa39f (because the lower bound is __read_nocancel,
    // but 0xaa39f is beyond its end.
    if (addr_to_sym_map.empty()) {
      return;
    }
    const ElfSectionReader<ElfArch>* const symbol_section =
        this->GetSectionByType(section_type);
    const ElfSectionReader<ElfArch>* const string_section =
        this->GetSection(symbol_section->header().sh_link);

    typename AddrToSymMap::iterator curr = addr_to_sym_map.begin();
    // Always insert the first symbol.
    symbols->AddSymbol(string_section->GetOffset(curr->second->st_name),
                       curr->first, curr->second->st_size);
    typename AddrToSymMap::iterator prev = curr++;
    for (; curr != addr_to_sym_map.end(); ++curr) {
      const uint64_t prev_addr = prev->first;
      const uint64_t curr_addr = curr->first;
      const typename ElfArch::Sym* const prev_sym = prev->second;
      const typename ElfArch::Sym* const curr_sym = curr->second;
      if (prev_addr + prev_sym->st_size <= curr_addr ||
          // The next condition is true if two symbols overlap like this:
          //
          //   Previous symbol  |----------------------------|
          //   Current symbol     |-------------------------------|
          //
          // These symbols are not found in google3 codebase, but in
          // jdk1.6.0_01_gg1/jre/lib/i386/server/libjvm.so.
          //
          // 0619e040 00000046 t CardTableModRefBS::write_region_work()
          // 0619e070 00000046 t CardTableModRefBS::write_ref_array_work()
          //
          // We allow overlapped symbols rather than ignore these.
          // Due to the way SymbolMap::GetSymbolAtPosition() works,
          // lookup for any address in [curr_addr, curr_addr + its size)
          // (e.g. 0619e071) will produce the current symbol,
          // which is the desired outcome.
          prev_addr + prev_sym->st_size < curr_addr + curr_sym->st_size) {
        const char* name = string_section->GetOffset(curr_sym->st_name);
        symbols->AddSymbol(name, curr_addr, curr_sym->st_size);
        prev = curr;
      } else {
        // Current symbol is "nested" inside previous one like this:
        //
        //   Previous symbol  |----------------------------|
        //   Current symbol     |---------------------|
        //
        // This happens within glibc, e.g. __read_nocancel is nested
        // "inside" __read. Ignore "inner" symbol.
        //DCHECK_LE(curr_addr + curr_sym->st_size,
        //          prev_addr + prev_sym->st_size);
        ;
      }
    }
  }
*/

  void VisitSymbols(typename ElfArch::Word section_type,
                    ElfReader::SymbolSink* sink) {
    VisitSymbols(section_type, sink, -1, -1, false);
  }

  void VisitSymbols(typename ElfArch::Word section_type,
                    ElfReader::SymbolSink* sink,
                    int symbol_binding,
                    int symbol_type,
                    bool get_raw_symbol_values) {
    for (SymbolIterator<ElfArch> it(this, section_type);
         !it.done(); it.Next()) {
      const char* name = it.GetSymbolName();
      if (!name) continue;
      const typename ElfArch::Sym* sym = it.GetSymbol();
      if ((symbol_binding < 0 || ElfArch::Bind(sym) == symbol_binding) &&
          (symbol_type < 0 || ElfArch::Type(sym) == symbol_type)) {
        typename ElfArch::Sym symbol = *sym;
        // Add a PLT symbol in addition to the main undefined symbol.
        // Only do this for SHT_DYNSYM, because PLT symbols are dynamic.
        int symbol_index = it.GetCurrentSymbolIndex();
        // TODO(dthomson): Can be removed once all Java code is using the
        // Google3 launcher.
        if (section_type == SHT_DYNSYM &&
            static_cast<unsigned int>(symbol_index) < symbols_plt_offsets_.size() &&
            symbols_plt_offsets_[symbol_index] != 0) {
          string plt_name = string(name) + kPLTFunctionSuffix;
          if (plt_function_names_[symbol_index].empty()) {
            plt_function_names_[symbol_index] = plt_name;
          } else if (plt_function_names_[symbol_index] != plt_name) {
		;
          }
          sink->AddSymbol(plt_function_names_[symbol_index].c_str(),
                          symbols_plt_offsets_[it.GetCurrentSymbolIndex()],
                          plt_code_size_);
        }
        if (!get_raw_symbol_values)
          AdjustSymbolValue(&symbol);
        sink->AddSymbol(name, symbol.st_value, symbol.st_size);
      }
    }
  }

  void VisitRelocationEntries() {
    if (visited_relocation_entries_) {
      return;
    }
    visited_relocation_entries_ = true;

    if (!plts_supported_) {
      return;
    }
    // First determine if PLTs exist. If not, then there is nothing to do.
    ElfReader::SectionInfo plt_section_info;
    const char* plt_section =
        GetSectionInfoByName(kElfPLTSectionName, &plt_section_info);
    if (!plt_section) {
      return;
    }
    if (plt_section_info.size == 0) {
      return;
    }

    // The PLTs could be referenced by either a Rel or Rela (Rel with Addend)
    // section.
    ElfReader::SectionInfo rel_section_info;
    ElfReader::SectionInfo rela_section_info;
    const char* rel_section =
        GetSectionInfoByName(kElfPLTRelSectionName, &rel_section_info);
    const char* rela_section =
        GetSectionInfoByName(kElfPLTRelaSectionName, &rela_section_info);

    const typename ElfArch::Rel* rel =
        reinterpret_cast<const typename ElfArch::Rel*>(rel_section);
    const typename ElfArch::Rela* rela =
        reinterpret_cast<const typename ElfArch::Rela*>(rela_section);

    if (!rel_section && !rela_section) {
      return;
    }

    // Use either Rel or Rela section, depending on which one exists.
    size_t section_size = rel_section ? rel_section_info.size
                                      : rela_section_info.size;
    size_t entry_size = rel_section ? sizeof(typename ElfArch::Rel)
                                    : sizeof(typename ElfArch::Rela);

    // Determine the number of entries in the dynamic symbol table.
    ElfReader::SectionInfo dynsym_section_info;
    const char* dynsym_section =
        GetSectionInfoByName(kElfDynSymSectionName, &dynsym_section_info);
    // The dynsym section might not exist, or it might be empty. In either case
    // there is nothing to be done so return.
    if (!dynsym_section || dynsym_section_info.size == 0) {
      return;
    }
    size_t num_dynamic_symbols =
        dynsym_section_info.size / dynsym_section_info.entsize;
    symbols_plt_offsets_.resize(num_dynamic_symbols, 0);

    // TODO(dthomson): Can be removed once all Java code is using the
    // Google3 launcher.
    // Make storage room for PLT function name strings.
    plt_function_names_.resize(num_dynamic_symbols);

    for (size_t i = 0; i < section_size / entry_size; ++i) {
      // Determine symbol index from the |r_info| field.
      int sym_index = ElfArch::r_sym(rel_section ? rel[i].r_info
                                                 : rela[i].r_info);
      if (static_cast<unsigned int>(sym_index) >= symbols_plt_offsets_.size()) {
        continue;
      }
      symbols_plt_offsets_[sym_index] =
          plt_section_info.addr + plt0_size_ + i * plt_code_size_;
    }
  }

  // Return an ElfSectionReader for the first section of the given
  // type by iterating through all section headers. Returns NULL if
  // the section type is not found.
  const ElfSectionReader<ElfArch>* GetSectionByType(
      typename ElfArch::Word section_type) {
    for (unsigned int k = 0u; k < GetNumSections(); ++k) {
      if (section_headers_[k].sh_type == section_type) {
        return GetSection(k);
      }
    }
    return NULL;
  }

  // Return the name of section "shndx".  Returns NULL if the section
  // is not found.
  const char* GetSectionNameByIndex(int shndx) {
    return GetSectionName(section_headers_[shndx].sh_name);
  }

  // Return a pointer to section "shndx", and store the size in
  // "size".  Returns NULL if the section is not found.
  const char* GetSectionContentsByIndex(int shndx, size_t* size) {
    const ElfSectionReader<ElfArch>* section = GetSection(shndx);
    if (section != NULL) {
      *size = section->section_size();
      return section->contents();
    }
    return NULL;
  }

  // Return a pointer to the first section of the given name by
  // iterating through all section headers, and store the size in
  // "size".  Returns NULL if the section name is not found.
  const char* GetSectionContentsByName(const string& section_name,
                                       size_t* size) {
    for (unsigned int k = 0u; k < GetNumSections(); ++k) {
      // When searching for sections in a .dwp file, the sections
      // we're looking for will always be at the end of the section
      // table, so reverse the direction of iteration.
      int shndx = is_dwp_ ? GetNumSections() - k - 1 : k;
      const char* name = GetSectionName(section_headers_[shndx].sh_name);
      if (name != NULL && ElfReader::SectionNamesMatch(section_name, name)) {
        const ElfSectionReader<ElfArch>* section = GetSection(shndx);
        if (section == NULL) {
          return NULL;
        } else {
          *size = section->section_size();
          return section->contents();
        }
      }
    }
    return NULL;
  }

  // This is like GetSectionContentsByName() but it returns a lot of extra
  // information about the section.
  const char* GetSectionInfoByName(const string& section_name,
                                   ElfReader::SectionInfo* info) {
    for (unsigned int k = 0u; k < GetNumSections(); ++k) {
      // When searching for sections in a .dwp file, the sections
      // we're looking for will always be at the end of the section
      // table, so reverse the direction of iteration.
      int shndx = is_dwp_ ? GetNumSections() - k - 1 : k;
      const char* name = GetSectionName(section_headers_[shndx].sh_name);
      if (name != NULL && ElfReader::SectionNamesMatch(section_name, name)) {
        const ElfSectionReader<ElfArch>* section = GetSection(shndx);
        if (section == NULL) {
          return NULL;
        } else {
          info->type = section->header().sh_type;
          info->flags = section->header().sh_flags;
          info->addr = section->header().sh_addr;
          info->offset = section->header().sh_offset;
          info->size = section->header().sh_size;
          info->link = section->header().sh_link;
          info->info = section->header().sh_info;
          info->addralign = section->header().sh_addralign;
          info->entsize = section->header().sh_entsize;
          return section->contents();
        }
      }
    }
    return NULL;
  }

  // p_vaddr of the first PT_LOAD segment (if any), or 0 if no PT_LOAD
  // segments are present. This is the address an ELF image was linked
  // (by static linker) to be loaded at. Usually (but not always) 0 for
  // shared libraries and position-independent executables.
  uint64_t VaddrOfFirstLoadSegment() const {
    // Relocatable objects (of type ET_REL) do not have LOAD segments.
    if (header_.e_type == ET_REL) {
      return 0;
    }
    for (int i = 0; i < GetNumProgramHeaders(); ++i) {
      if (program_headers_[i].p_type == PT_LOAD) {
        return program_headers_[i].p_vaddr;
      }
    }
    return 0;
  }

  // According to the LSB ("ELF special sections"), sections with debug
  // info are prefixed by ".debug".  The names are not specified, but they
  // look like ".debug_line", ".debug_info", etc.
  bool HasDebugSections() {
    // Debug sections are likely to be near the end, so reverse the
    // direction of iteration.
    for (int k = GetNumSections() - 1; k >= 0; --k) {
      std::string_view name{GetSectionName(section_headers_[k].sh_name)};
      if (StringViewStartsWith(name, ".debug") ||
          StringViewStartsWith(name, ".zdebug")) {
        return true;
      }
    }
    return false;
  }

  bool IsDynamicSharedObject() const {
    return header_.e_type == ET_DYN;
  }

  // Return the number of sections.
  uint64_t GetNumSections() const {
    if (HasManySections())
      return first_section_header_.sh_size;
    return header_.e_shnum;
  }

 private:
  typedef vector<pair<uint64_t, const typename ElfArch::Sym*> > AddrToSymMap;

  static bool AddrToSymSorter(const typename AddrToSymMap::value_type& lhs,
                              const typename AddrToSymMap::value_type& rhs) {
    return lhs.first < rhs.first;
  }

  static bool AddrToSymEquals(const typename AddrToSymMap::value_type& lhs,
                              const typename AddrToSymMap::value_type& rhs) {
    return lhs.first == rhs.first;
  }

  // Does this ELF file have too many sections to fit in the program header?
  bool HasManySections() const {
    return header_.e_shnum == SHN_UNDEF;
  }

  // Return the number of program headers.
  int GetNumProgramHeaders() const {
    if (HasManySections() && header_.e_phnum == 0xffff &&
        first_section_header_.sh_info != 0)
      return first_section_header_.sh_info;
    return header_.e_phnum;
  }

  // Return the index of the string table.
  int GetStringTableIndex() const {
    if (HasManySections()) {
      if (header_.e_shstrndx == 0xffff)
        return first_section_header_.sh_link;
      else if (header_.e_shstrndx >= GetNumSections())
        return 0;
    }
    return header_.e_shstrndx;
  }

  // Given an offset into the section header string table, return the
  // section name.
  const char* GetSectionName(typename ElfArch::Word sh_name) {
    const ElfSectionReader<ElfArch>* shstrtab =
        GetSection(GetStringTableIndex());
    if (shstrtab != NULL) {
      return shstrtab->GetOffset(sh_name);
    }
    return NULL;
  }

  // Return an ElfSectionReader for the given section. The reader will
  // be freed when this object is destroyed.
  const ElfSectionReader<ElfArch>* GetSection(int num) {
    const char* name;
    // Hard-coding the name for the section-name string table prevents
    // infinite recursion.
    if (num == GetStringTableIndex())
      name = ".shstrtab";
    else
      name = GetSectionNameByIndex(num);
    ElfSectionReader<ElfArch>*& reader = sections_[num];
    if (reader == NULL)
      reader = new ElfSectionReader<ElfArch>(name, path_, fd_,
                                             section_headers_[num]);
    return reader->contents() ? reader : nullptr;
  }

  // Parse out the overall header information from the file and assert
  // that it looks sane. This contains information like the magic
  // number and target architecture.
  bool ParseHeaders(int fd, const string& path) {
    // Read in the global ELF header.
    if (pread(fd, &header_, sizeof(header_), 0) != sizeof(header_)) {
      return false;
    }

    // Must be an executable, dynamic shared object or relocatable object
    if (header_.e_type != ET_EXEC &&
        header_.e_type != ET_DYN &&
        header_.e_type != ET_REL) {
      return false;
    }
    // Need a section header.
    if (header_.e_shoff == 0) {
      return false;
    }

    if (header_.e_shnum == SHN_UNDEF) {
      // The number of sections in the program header is only a 16-bit value. In
      // the event of overflow (greater than SHN_LORESERVE sections), e_shnum
      // will read SHN_UNDEF and the true number of section header table entries
      // is found in the sh_size field of the first section header.
      // See: http://www.sco.com/developers/gabi/2003-12-17/ch4.sheader.html
      if (pread(fd, &first_section_header_, sizeof(first_section_header_),
                header_.e_shoff) != sizeof(first_section_header_)) {
        return false;
      }
    }

    // Dynamically allocate enough space to store the section headers
    // and read them out of the file.
    const int section_headers_size =
        GetNumSections() * sizeof(*section_headers_);
    section_headers_ = new typename ElfArch::Shdr[section_headers_size];
    if (pread(fd, section_headers_, section_headers_size, header_.e_shoff) !=
        section_headers_size) {
      return false;
    }

    // Dynamically allocate enough space to store the program headers
    // and read them out of the file.
    //const int program_headers_size =
    //    GetNumProgramHeaders() * sizeof(*program_headers_);
    program_headers_ = new typename ElfArch::Phdr[GetNumProgramHeaders()];

    // Presize the sections array for efficiency.
    sections_.resize(GetNumSections(), NULL);
    return true;
  }

  // Given the "value" of a function descriptor return the address of the
  // function (i.e. the dereferenced value). Otherwise return "value".
  uint64_t AdjustPPC64FunctionDescriptorSymbolValue(uint64_t value) {
    if (opd_section_ != NULL &&
        opd_info_.addr <= value &&
        value < opd_info_.addr + opd_info_.size) {
      uint64_t offset = value - opd_info_.addr;
      return (*reinterpret_cast<const uint64_t*>(opd_section_ + offset));
    }
    return value;
  }

  void AdjustSymbolValue(typename ElfArch::Sym* sym) {
    switch (header_.e_machine) {
    case EM_ARM:
      // For ARM architecture, if the LSB of the function symbol offset is set,
      // it indicates a Thumb function.  This bit should not be taken literally.
      // Clear it.
      if (ElfArch::Type(sym) == STT_FUNC)
        sym->st_value = AdjustARMThumbSymbolValue(sym->st_value);
      break;
    case EM_386:
      // No adjustment needed for Intel x86 architecture.  However, explicitly
      // define this case as we use it quite often.
      break;
    case EM_PPC64:
      // PowerPC64 currently has function descriptors as part of the ABI.
      // Function symbols need to be adjusted accordingly.
      if (ElfArch::Type(sym) == STT_FUNC)
        sym->st_value = AdjustPPC64FunctionDescriptorSymbolValue(sym->st_value);
      break;
    default:
      break;
    }
  }

  friend class SymbolIterator<ElfArch>;

  // The file we're reading.
  const string path_;
  // Open file descriptor for path_. Not owned by this object.
  const int fd_;

  // The global header of the ELF file.
  typename ElfArch::Ehdr header_;

  // The header of the first section. This may be used to supplement the ELF
  // file header.
  typename ElfArch::Shdr first_section_header_;

  // Array of GetNumSections() section headers, allocated when we read
  // in the global header.
  typename ElfArch::Shdr* section_headers_;

  // Array of GetNumProgramHeaders() program headers, allocated when we read
  // in the global header.
  typename ElfArch::Phdr* program_headers_;

  // An array of pointers to ElfSectionReaders. Sections are
  // mmaped as they're needed and not released until this object is
  // destroyed.
  vector<ElfSectionReader<ElfArch>*> sections_;

  // For PowerPC64 we need to keep track of function descriptors when looking up
  // values for funtion symbols values. Function descriptors are kept in the
  // .opd section and are dereferenced to find the function address.
  ElfReader::SectionInfo opd_info_;
  const char* opd_section_;  // Must be checked for NULL before use.
  int64_t base_for_text_;

  // Read PLT-related sections for the current architecture.
  bool plts_supported_;
  // Code size of each PLT function for the current architecture.
  size_t plt_code_size_;
  // Size of the special first entry in the .plt section that calls the runtime
  // loader resolution routine, and that all other entries jump to when doing
  // lazy symbol binding.
  size_t plt0_size_;

  // Maps a dynamic symbol index to a PLT offset.
  // The vector entry index is the dynamic symbol index.
  std::vector<uint64_t> symbols_plt_offsets_;

  // Container for PLT function name strings. These strings are passed by
  // reference to SymbolSink::AddSymbol() so they need to be stored somewhere.
  std::vector<string> plt_function_names_;

  bool visited_relocation_entries_;

  // True if this is a .dwp file.
  bool is_dwp_;
};

ElfReader::ElfReader(const string& path)
    : path_(path), fd_(-1), impl32_(NULL), impl64_(NULL) {
  // linux 2.6.XX kernel can show deleted files like this:
  //   /var/run/nscd/dbYLJYaE (deleted)
  // and the kernel-supplied vdso and vsyscall mappings like this:
  //   [vdso]
  //   [vsyscall]
  if (MyHasSuffixString(path, " (deleted)"))
    return;
  if (path == "[vdso]")
    return;
  if (path == "[vsyscall]")
    return;

  fd_ = open(path.c_str(), O_RDONLY);
}

ElfReader::~ElfReader() {
  if (fd_ != -1)
    close(fd_);
  if (impl32_ != NULL)
    delete impl32_;
  if (impl64_ != NULL)
    delete impl64_;
}


// The only word-size specific part of this file is IsNativeElfFile().
#if ULONG_MAX == 0xffffffff
#define NATIVE_ELF_ARCH Elf32
#elif ULONG_MAX == 0xffffffffffffffff
#define NATIVE_ELF_ARCH Elf64
#else
#error "Invalid word size"
#endif

template <typename ElfArch>
static bool IsElfFile(const int fd, const string& path) {
  if (fd < 0)
    return false;
  if (!ElfReaderImpl<ElfArch>::IsArchElfFile(fd, NULL)) {
    // No error message here.  IsElfFile gets called many times.
    return false;
  }
  return true;
}

bool ElfReader::IsNativeElfFile() const {
  return IsElfFile<NATIVE_ELF_ARCH>(fd_, path_);
}

bool ElfReader::IsElf32File() const {
  return IsElfFile<Elf32>(fd_, path_);
}

bool ElfReader::IsElf64File() const {
  return IsElfFile<Elf64>(fd_, path_);
}

/*
void ElfReader::AddSymbols(SymbolMap* symbols,
                           uint64_t mem_offset, uint64_t file_offset,
                           uint64_t length) {
  if (fd_ < 0)
    return;
  // TODO(chatham): Actually use the information about file offset and
  // the length of the mapped section. On some machines the data
  // section gets mapped as executable, and we'll end up reading the
  // file twice and getting some of the offsets wrong.
  if (IsElf32File()) {
    GetImpl32()->GetSymbolPositions(symbols, SHT_SYMTAB,
                                    mem_offset, file_offset);
    GetImpl32()->GetSymbolPositions(symbols, SHT_DYNSYM,
                                    mem_offset, file_offset);
  } else if (IsElf64File()) {
    GetImpl64()->GetSymbolPositions(symbols, SHT_SYMTAB,
                                    mem_offset, file_offset);
    GetImpl64()->GetSymbolPositions(symbols, SHT_DYNSYM,
                                    mem_offset, file_offset);
  }
}
*/

void ElfReader::VisitSymbols(ElfReader::SymbolSink* sink) {
  VisitSymbols(sink, -1, -1);
}

void ElfReader::VisitSymbols(ElfReader::SymbolSink* sink,
                             int symbol_binding,
                             int symbol_type) {
  VisitSymbols(sink, symbol_binding, symbol_type, false);
}

void ElfReader::VisitSymbols(ElfReader::SymbolSink* sink,
                             int symbol_binding,
                             int symbol_type,
                             bool get_raw_symbol_values) {
  if (IsElf32File()) {
    GetImpl32()->VisitRelocationEntries();
    GetImpl32()->VisitSymbols(SHT_SYMTAB, sink, symbol_binding, symbol_type,
                              get_raw_symbol_values);
    GetImpl32()->VisitSymbols(SHT_DYNSYM, sink, symbol_binding, symbol_type,
                              get_raw_symbol_values);
  } else if (IsElf64File()) {
    GetImpl64()->VisitRelocationEntries();
    GetImpl64()->VisitSymbols(SHT_SYMTAB, sink, symbol_binding, symbol_type,
                              get_raw_symbol_values);
    GetImpl64()->VisitSymbols(SHT_DYNSYM, sink, symbol_binding, symbol_type,
                              get_raw_symbol_values);
  }
}

uint64_t ElfReader::VaddrOfFirstLoadSegment() {
  if (IsElf32File()) {
    return GetImpl32()->VaddrOfFirstLoadSegment();
  } else if (IsElf64File()) {
    return GetImpl64()->VaddrOfFirstLoadSegment();
  } else {
    return 0;
  }
}

const char* ElfReader::GetSectionName(int shndx) {
  if (shndx < 0 || static_cast<unsigned int>(shndx) >= GetNumSections()) return NULL;
  if (IsElf32File()) {
    return GetImpl32()->GetSectionNameByIndex(shndx);
  } else if (IsElf64File()) {
    return GetImpl64()->GetSectionNameByIndex(shndx);
  } else {
    return NULL;
  }
}

uint64_t ElfReader::GetNumSections() {
  if (IsElf32File()) {
    return GetImpl32()->GetNumSections();
  } else if (IsElf64File()) {
    return GetImpl64()->GetNumSections();
  } else {
    return 0;
  }
}

const char* ElfReader::GetSectionByIndex(int shndx, size_t* size) {
  if (IsElf32File()) {
    return GetImpl32()->GetSectionContentsByIndex(shndx, size);
  } else if (IsElf64File()) {
    return GetImpl64()->GetSectionContentsByIndex(shndx, size);
  } else {
    return NULL;
  }
}

const char* ElfReader::GetSectionByName(const string& section_name,
                                        size_t* size) {
  if (IsElf32File()) {
    return GetImpl32()->GetSectionContentsByName(section_name, size);
  } else if (IsElf64File()) {
    return GetImpl64()->GetSectionContentsByName(section_name, size);
  } else {
    return NULL;
  }
}

const char* ElfReader::GetSectionInfoByName(const string& section_name,
                                            SectionInfo* info) {
  if (IsElf32File()) {
    return GetImpl32()->GetSectionInfoByName(section_name, info);
  } else if (IsElf64File()) {
    return GetImpl64()->GetSectionInfoByName(section_name, info);
  } else {
    return NULL;
  }
}

bool ElfReader::SectionNamesMatch(std::string_view name,
                                  std::string_view sh_name) {
  std::string_view debug_prefix{".debug_"};
  std::string_view zdebug_prefix{".zdebug_"};
  if (StringViewStartsWith(name, debug_prefix) &&
      StringViewStartsWith(sh_name, zdebug_prefix)) {
    name.remove_prefix(debug_prefix.length());
    sh_name.remove_prefix(zdebug_prefix.length());
    return name == sh_name;
  }
  return name == sh_name;
}

bool ElfReader::IsDynamicSharedObject() {
  if (IsElf32File()) {
    return GetImpl32()->IsDynamicSharedObject();
  } else if (IsElf64File()) {
    return GetImpl64()->IsDynamicSharedObject();
  } else {
    return false;
  }
}

ElfReaderImpl<Elf32>* ElfReader::GetImpl32() {
  if (impl32_ == NULL) {
    impl32_ = new ElfReaderImpl<Elf32>(path_, fd_);
  }
  return impl32_;
}

ElfReaderImpl<Elf64>* ElfReader::GetImpl64() {
  if (impl64_ == NULL) {
    impl64_ = new ElfReaderImpl<Elf64>(path_, fd_);
  }
  return impl64_;
}

// Return true if file is an ELF binary of ElfArch, with unstripped
// debug info (debug_only=true) or symbol table (debug_only=false).
// Otherwise, return false.
template <typename ElfArch>
static bool IsNonStrippedELFBinaryImpl(const string& path, const int fd,
                                       bool debug_only) {
  if (!ElfReaderImpl<ElfArch>::IsArchElfFile(fd, NULL)) return false;
  ElfReaderImpl<ElfArch> elf_reader(path, fd);
  return debug_only ?
      elf_reader.HasDebugSections()
      : (elf_reader.GetSectionByType(SHT_SYMTAB) != NULL);
}

// Helper for the IsNon[Debug]StrippedELFBinary functions.
static bool IsNonStrippedELFBinaryHelper(const string& path,
                                         bool debug_only) {
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd == -1) {
    return false;
  }

  if (IsNonStrippedELFBinaryImpl<Elf32>(path, fd, debug_only) ||
      IsNonStrippedELFBinaryImpl<Elf64>(path, fd, debug_only)) {
    close(fd);
    return true;
  }
  close(fd);
  return false;
}

bool ElfReader::IsNonStrippedELFBinary(const string& path) {
  return IsNonStrippedELFBinaryHelper(path, false);
}

bool ElfReader::IsNonDebugStrippedELFBinary(const string& path) {
  return IsNonStrippedELFBinaryHelper(path, true);
}
}  // namespace google_breakpad
