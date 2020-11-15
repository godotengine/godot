// Copyright 2017 The Crashpad Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "snapshot/elf/elf_image_reader.h"

#include <stddef.h>

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include "base/logging.h"
#include "build/build_config.h"
#include "util/numeric/checked_vm_address_range.h"

namespace crashpad {

class ElfImageReader::ProgramHeaderTable {
 public:
  virtual ~ProgramHeaderTable() {}

  virtual bool VerifyLoadSegments(bool verbose) const = 0;
  virtual size_t Size() const = 0;
  virtual bool GetDynamicSegment(VMAddress* address, VMSize* size) const = 0;
  virtual bool GetPreferredElfHeaderAddress(VMAddress* address,
                                            bool verbose) const = 0;
  virtual bool GetPreferredLoadedMemoryRange(VMAddress* address,
                                             VMSize* size,
                                             bool verbose) const = 0;

  // Locate the next PT_NOTE segment starting at segment index start_index. If a
  // PT_NOTE segment is found, start_index is set to the next index after the
  // found segment.
  virtual bool GetNoteSegment(size_t* start_index,
                              VMAddress* address,
                              VMSize* size) const = 0;

 protected:
  ProgramHeaderTable() {}
};

template <typename PhdrType>
class ElfImageReader::ProgramHeaderTableSpecific
    : public ElfImageReader::ProgramHeaderTable {
 public:
  ProgramHeaderTableSpecific<PhdrType>() {}
  ~ProgramHeaderTableSpecific<PhdrType>() {}

  bool Initialize(const ProcessMemoryRange& memory,
                  VMAddress address,
                  VMSize num_segments,
                  bool verbose) {
    INITIALIZATION_STATE_SET_INITIALIZING(initialized_);
    table_.resize(num_segments);
    if (!memory.Read(address, sizeof(PhdrType) * num_segments, table_.data())) {
      return false;
    }

    if (!VerifyLoadSegments(verbose)) {
      return false;
    }

    INITIALIZATION_STATE_SET_VALID(initialized_);
    return true;
  }

  bool VerifyLoadSegments(bool verbose) const override {
    constexpr bool is_64_bit = std::is_same<PhdrType, Elf64_Phdr>::value;
    VMAddress last_vaddr;
    bool load_found = false;
    for (const auto& header : table_) {
      if (header.p_type == PT_LOAD) {
        CheckedVMAddressRange load_range(
            is_64_bit, header.p_vaddr, header.p_memsz);

        if (!load_range.IsValid()) {
          LOG_IF(ERROR, verbose) << "bad load range";
          return false;
        }

        if (load_found && header.p_vaddr <= last_vaddr) {
          LOG_IF(ERROR, verbose) << "out of order load segments";
          return false;
        }
        load_found = true;
        last_vaddr = header.p_vaddr;
      }
    }
    return true;
  }

  size_t Size() const override { return sizeof(PhdrType) * table_.size(); }

  bool GetPreferredElfHeaderAddress(VMAddress* address,
                                    bool verbose) const override {
    INITIALIZATION_STATE_DCHECK_VALID(initialized_);
    for (const auto& header : table_) {
      if (header.p_type == PT_LOAD && header.p_offset == 0) {
        *address = header.p_vaddr;
        return true;
      }
    }
    LOG_IF(ERROR, verbose) << "no preferred header address";
    return false;
  }

  bool GetPreferredLoadedMemoryRange(VMAddress* base,
                                     VMSize* size,
                                     bool verbose) const override {
    INITIALIZATION_STATE_DCHECK_VALID(initialized_);

    VMAddress preferred_base = 0;
    VMAddress preferred_end = 0;
    bool load_found = false;
    for (const auto& header : table_) {
      if (header.p_type == PT_LOAD) {
        if (!load_found) {
          preferred_base = header.p_vaddr;
          load_found = true;
        }
        preferred_end = header.p_vaddr + header.p_memsz;
      }
    }
    if (load_found) {
      *base = preferred_base;
      *size = preferred_end - preferred_base;
      return true;
    }
    LOG_IF(ERROR, verbose) << "no load segments";
    return false;
  }

  bool GetDynamicSegment(VMAddress* address, VMSize* size) const override {
    INITIALIZATION_STATE_DCHECK_VALID(initialized_);
    const PhdrType* phdr;
    if (!GetProgramHeader(PT_DYNAMIC, &phdr)) {
      return false;
    }
    *address = phdr->p_vaddr;
    *size = phdr->p_memsz;
    return true;
  }

  bool GetProgramHeader(uint32_t type, const PhdrType** header_out) const {
    INITIALIZATION_STATE_DCHECK_VALID(initialized_);
    for (const auto& header : table_) {
      if (header.p_type == type) {
        *header_out = &header;
        return true;
      }
    }
    return false;
  }

  bool GetNoteSegment(size_t* start_index,
                      VMAddress* address,
                      VMSize* size) const override {
    INITIALIZATION_STATE_DCHECK_VALID(initialized_);
    for (size_t index = *start_index; index < table_.size(); ++index) {
      if (table_[index].p_type == PT_NOTE && table_[index].p_vaddr != 0) {
        *start_index = index + 1;
        *address = table_[index].p_vaddr;
        *size = table_[index].p_memsz;
        return true;
      }
    }
    return false;
  }

 private:
  std::vector<PhdrType> table_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(ProgramHeaderTableSpecific<PhdrType>);
};

ElfImageReader::NoteReader::~NoteReader() = default;

ElfImageReader::NoteReader::Result ElfImageReader::NoteReader::NextNote(
    std::string* name,
    NoteType* type,
    std::string* desc) {
  if (!is_valid_) {
    LOG(ERROR) << "invalid note reader";
    return Result::kError;
  }

  Result result = Result::kError;
  do {
    while (current_address_ == segment_end_address_) {
      VMSize segment_size;
      if (!phdr_table_->GetNoteSegment(
              &phdr_index_, &current_address_, &segment_size)) {
        return Result::kNoMoreNotes;
      }
      current_address_ += elf_reader_->GetLoadBias();
      segment_end_address_ = current_address_ + segment_size;
      segment_range_ = std::make_unique<ProcessMemoryRange>();
      if (!segment_range_->Initialize(*range_) ||
          !segment_range_->RestrictRange(current_address_, segment_size)) {
        return Result::kError;
      }
    }

    retry_ = false;
    result = range_->Is64Bit() ? ReadNote<Elf64_Nhdr>(name, type, desc)
                               : ReadNote<Elf32_Nhdr>(name, type, desc);
  } while (retry_);

  if (result == Result::kSuccess) {
    return Result::kSuccess;
  }
  is_valid_ = false;
  return Result::kError;
}

ElfImageReader::NoteReader::NoteReader(const ElfImageReader* elf_reader,
                                       const ProcessMemoryRange* range,
                                       const ProgramHeaderTable* phdr_table,
                                       ssize_t max_note_size,
                                       const std::string& name_filter,
                                       NoteType type_filter,
                                       bool use_filter)
    : current_address_(0),
      segment_end_address_(0),
      elf_reader_(elf_reader),
      range_(range),
      phdr_table_(phdr_table),
      segment_range_(),
      phdr_index_(0),
      max_note_size_(max_note_size),
      name_filter_(name_filter),
      type_filter_(type_filter),
      use_filter_(use_filter),
      is_valid_(true),
      retry_(false) {}

template <typename NhdrType>
ElfImageReader::NoteReader::Result ElfImageReader::NoteReader::ReadNote(
    std::string* name,
    NoteType* type,
    std::string* desc) {
  static_assert(sizeof(*type) >= sizeof(NhdrType::n_namesz),
                "Note field size mismatch");
  DCHECK_LT(current_address_, segment_end_address_);

  NhdrType note_info;
  if (!segment_range_->Read(current_address_, sizeof(note_info), &note_info)) {
    return Result::kError;
  }
  current_address_ += sizeof(note_info);

  constexpr size_t align = sizeof(note_info.n_namesz);
#define PAD(x) (((x) + align - 1) & ~(align - 1))
  size_t padded_namesz = PAD(note_info.n_namesz);
  size_t padded_descsz = PAD(note_info.n_descsz);
  size_t note_size = padded_namesz + padded_descsz;

  // Notes typically have 4-byte alignment. However, .note.android.ident may
  // inadvertently use 2-byte alignment.
  // https://android-review.googlesource.com/c/platform/bionic/+/554986/
  // We can still find .note.android.ident if it appears first in a note segment
  // but there may be 4-byte aligned notes following it. If this note was
  // aligned at less than 4-bytes, expect that the next note will be aligned at
  // 4-bytes and add extra padding, if necessary.
  VMAddress end_of_note =
      std::min(PAD(current_address_ + note_size), segment_end_address_);
#undef PAD

  if (max_note_size_ >= 0 && note_size > static_cast<size_t>(max_note_size_)) {
    current_address_ = end_of_note;
    retry_ = true;
    return Result::kError;
  }

  if (use_filter_ && note_info.n_type != type_filter_) {
    current_address_ = end_of_note;
    retry_ = true;
    return Result::kError;
  }

  std::string local_name(note_info.n_namesz, '\0');
  if (!segment_range_->Read(
          current_address_, note_info.n_namesz, &local_name[0])) {
    return Result::kError;
  }
  if (!local_name.empty()) {
    if (local_name.back() != '\0') {
      LOG(ERROR) << "unterminated note name";
      return Result::kError;
    }
    local_name.pop_back();
  }

  if (use_filter_ && local_name != name_filter_) {
    current_address_ = end_of_note;
    retry_ = true;
    return Result::kError;
  }

  current_address_ += padded_namesz;

  std::string local_desc(note_info.n_descsz, '\0');
  if (!segment_range_->Read(
          current_address_, note_info.n_descsz, &local_desc[0])) {
    return Result::kError;
  }

  current_address_ = end_of_note;

  if (name) {
    name->swap(local_name);
  }
  if (type) {
    *type = note_info.n_type;
  }
  desc->swap(local_desc);
  return Result::kSuccess;
}

ElfImageReader::ElfImageReader()
    : header_64_(),
      ehdr_address_(0),
      load_bias_(0),
      memory_(),
      program_headers_(),
      dynamic_array_(),
      symbol_table_(),
      initialized_(),
      dynamic_array_initialized_(),
      symbol_table_initialized_() {}

ElfImageReader::~ElfImageReader() {}

bool ElfImageReader::Initialize(const ProcessMemoryRange& memory,
                                VMAddress address,
                                bool verbose) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);
  ehdr_address_ = address;
  if (!memory_.Initialize(memory)) {
    return false;
  }

  uint8_t e_ident[EI_NIDENT];
  if (!memory_.Read(ehdr_address_, EI_NIDENT, e_ident)) {
    return false;
  }

  if (e_ident[EI_MAG0] != ELFMAG0 || e_ident[EI_MAG1] != ELFMAG1 ||
      e_ident[EI_MAG2] != ELFMAG2 || e_ident[EI_MAG3] != ELFMAG3) {
    LOG_IF(ERROR, verbose) << "Incorrect ELF magic number";
    return false;
  }

  if (!(memory_.Is64Bit() && e_ident[EI_CLASS] == ELFCLASS64) &&
      !(!memory_.Is64Bit() && e_ident[EI_CLASS] == ELFCLASS32)) {
    LOG_IF(ERROR, verbose) << "unexpected bitness";
    return false;
  }

#if defined(ARCH_CPU_LITTLE_ENDIAN)
  constexpr uint8_t expected_encoding = ELFDATA2LSB;
#elif defined(ARCH_CPU_BIG_ENDIAN)
  constexpr uint8_t expected_encoding = ELFDATA2MSB;
#endif
  if (e_ident[EI_DATA] != expected_encoding) {
    LOG_IF(ERROR, verbose) << "unexpected encoding";
    return false;
  }

  if (e_ident[EI_VERSION] != EV_CURRENT) {
    LOG_IF(ERROR, verbose) << "unexpected version";
    return false;
  }

  if (!(memory_.Is64Bit()
            ? memory_.Read(ehdr_address_, sizeof(header_64_), &header_64_)
            : memory_.Read(ehdr_address_, sizeof(header_32_), &header_32_))) {
    return false;
  }

#define VERIFY_HEADER(header)                                  \
  do {                                                         \
    if (header.e_type != ET_EXEC && header.e_type != ET_DYN) { \
      LOG_IF(ERROR, verbose) << "unexpected image type";       \
      return false;                                            \
    }                                                          \
    if (header.e_version != EV_CURRENT) {                      \
      LOG_IF(ERROR, verbose) << "unexpected version";          \
      return false;                                            \
    }                                                          \
    if (header.e_ehsize != sizeof(header)) {                   \
      LOG_IF(ERROR, verbose) << "unexpected header size";      \
      return false;                                            \
    }                                                          \
  } while (false);

  if (memory_.Is64Bit()) {
    VERIFY_HEADER(header_64_);
  } else {
    VERIFY_HEADER(header_32_);
  }

  if (!InitializeProgramHeaders(verbose)) {
    return false;
  }

  VMAddress preferred_ehdr_address;
  if (!program_headers_.get()->GetPreferredElfHeaderAddress(
          &preferred_ehdr_address, verbose)) {
    return false;
  }
  load_bias_ = ehdr_address_ - preferred_ehdr_address;

  VMAddress base_address;
  VMSize loaded_size;
  if (!program_headers_.get()->GetPreferredLoadedMemoryRange(
          &base_address, &loaded_size, verbose)) {
    return false;
  }
  base_address += load_bias_;

  if (!memory_.RestrictRange(base_address, loaded_size)) {
    return false;
  }

  VMSize ehdr_size;
  VMAddress phdr_address;
  if (memory_.Is64Bit()) {
    ehdr_size = sizeof(header_64_);
    phdr_address = ehdr_address_ + header_64_.e_phoff;
  } else {
    ehdr_size = sizeof(header_32_);
    phdr_address = ehdr_address_ + header_32_.e_phoff;
  }

  CheckedVMAddressRange range(memory_.Is64Bit(), base_address, loaded_size);
  if (!range.ContainsRange(
          CheckedVMAddressRange(memory_.Is64Bit(), ehdr_address_, ehdr_size))) {
    LOG_IF(ERROR, verbose) << "ehdr out of range";
    return false;
  }
  if (!range.ContainsRange(CheckedVMAddressRange(
          memory.Is64Bit(), phdr_address, program_headers_->Size()))) {
    LOG_IF(ERROR, verbose) << "phdrs out of range";
    return false;
  }

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

uint16_t ElfImageReader::FileType() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return memory_.Is64Bit() ? header_64_.e_type : header_32_.e_type;
}

bool ElfImageReader::SoName(std::string* name) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  if (!InitializeDynamicArray()) {
    return false;
  }

  VMSize offset;
  if (!dynamic_array_->GetValue(DT_SONAME, true, &offset)) {
    return false;
  }

  return ReadDynamicStringTableAtOffset(offset, name);
}

bool ElfImageReader::GetDynamicSymbol(const std::string& name,
                                      VMAddress* address,
                                      VMSize* size) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  if (!InitializeDynamicSymbolTable()) {
    return false;
  }

  ElfSymbolTableReader::SymbolInformation info;
  if (!symbol_table_->GetSymbol(name, &info)) {
    return false;
  }
  if (info.shndx == SHN_UNDEF || info.shndx == SHN_COMMON) {
    return false;
  }

  switch (info.binding) {
    case STB_GLOBAL:
    case STB_WEAK:
      break;

    case STB_LOCAL:
    default:
      return false;
  }

  switch (info.type) {
    case STT_OBJECT:
    case STT_FUNC:
      break;

    case STT_COMMON:
    case STT_NOTYPE:
    case STT_SECTION:
    case STT_FILE:
    case STT_TLS:
    default:
      return false;
  }

  if (info.shndx != SHN_ABS) {
    info.address += GetLoadBias();
  }

  *address = info.address;
  *size = info.size;
  return true;
}

bool ElfImageReader::ReadDynamicStringTableAtOffset(VMSize offset,
                                                    std::string* string) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  if (!InitializeDynamicArray()) {
    return false;
  }

  VMAddress string_table_address;
  VMSize string_table_size;
  if (!GetAddressFromDynamicArray(DT_STRTAB, true, &string_table_address) ||
      !dynamic_array_->GetValue(DT_STRSZ, true, &string_table_size)) {
    LOG(ERROR) << "missing string table info";
    return false;
  }
  if (offset >= string_table_size) {
    LOG(ERROR) << "bad offset";
    return false;
  }

  if (!memory_.ReadCStringSizeLimited(
          string_table_address + offset, string_table_size - offset, string)) {
    LOG(ERROR) << "missing nul-terminator";
    return false;
  }
  return true;
}

bool ElfImageReader::GetDebugAddress(VMAddress* debug) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  if (!InitializeDynamicArray()) {
    return false;
  }
  return GetAddressFromDynamicArray(DT_DEBUG, true, debug);
}

bool ElfImageReader::GetDynamicArrayAddress(VMAddress* address) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  VMAddress dyn_segment_address;
  VMSize dyn_segment_size;
  if (!program_headers_.get()->GetDynamicSegment(&dyn_segment_address,
                                                 &dyn_segment_size)) {
    LOG(ERROR) << "no dynamic segment";
    return false;
  }
  *address = dyn_segment_address + GetLoadBias();
  return true;
}

VMAddress ElfImageReader::GetProgramHeaderTableAddress() {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return ehdr_address_ +
         (memory_.Is64Bit() ? header_64_.e_phoff : header_32_.e_phoff);
}

bool ElfImageReader::InitializeProgramHeaders(bool verbose) {
#define INITIALIZE_PROGRAM_HEADERS(PhdrType, header)         \
  do {                                                       \
    if (header.e_phentsize != sizeof(PhdrType)) {            \
      LOG_IF(ERROR, verbose) << "unexpected phdr size";      \
      return false;                                          \
    }                                                        \
    auto phdrs = new ProgramHeaderTableSpecific<PhdrType>(); \
    program_headers_.reset(phdrs);                           \
    if (!phdrs->Initialize(memory_,                          \
                           ehdr_address_ + header.e_phoff,   \
                           header.e_phnum,                   \
                           verbose)) {                       \
      return false;                                          \
    }                                                        \
  } while (false);

  if (memory_.Is64Bit()) {
    INITIALIZE_PROGRAM_HEADERS(Elf64_Phdr, header_64_);
  } else {
    INITIALIZE_PROGRAM_HEADERS(Elf32_Phdr, header_32_);
  }
  return true;
}

bool ElfImageReader::InitializeDynamicArray() {
  if (dynamic_array_initialized_.is_valid()) {
    return true;
  }
  if (!dynamic_array_initialized_.is_uninitialized()) {
    return false;
  }
  dynamic_array_initialized_.set_invalid();

  VMAddress dyn_segment_address;
  VMSize dyn_segment_size;
  if (!program_headers_.get()->GetDynamicSegment(&dyn_segment_address,
                                                 &dyn_segment_size)) {
    LOG(ERROR) << "no dynamic segment";
    return false;
  }
  dyn_segment_address += GetLoadBias();

  dynamic_array_.reset(new ElfDynamicArrayReader());
  if (!dynamic_array_->Initialize(
          memory_, dyn_segment_address, dyn_segment_size)) {
    return false;
  }
  dynamic_array_initialized_.set_valid();
  return true;
}

bool ElfImageReader::InitializeDynamicSymbolTable() {
  if (symbol_table_initialized_.is_valid()) {
    return true;
  }
  if (!symbol_table_initialized_.is_uninitialized()) {
    return false;
  }
  symbol_table_initialized_.set_invalid();

  if (!InitializeDynamicArray()) {
    return false;
  }

  VMAddress symbol_table_address;
  if (!GetAddressFromDynamicArray(DT_SYMTAB, true, &symbol_table_address)) {
    LOG(ERROR) << "no symbol table";
    return false;
  }

  // Try both DT_HASH and DT_GNU_HASH. They're completely different, but both
  // circuitously offer a way to find the number of entries in the symbol table.
  // DT_HASH is specifically checked first, because depending on the linker, the
  // count maybe be incorrect for zero-export cases. In practice, it is believed
  // that the zero-export case is probably not particularly useful, so this
  // incorrect count will only occur in constructed test cases (see
  // ElfImageReader.DtHashAndDtGnuHashMatch).
  VMSize number_of_symbol_table_entries;
  if (!GetNumberOfSymbolEntriesFromDtHash(&number_of_symbol_table_entries) &&
      !GetNumberOfSymbolEntriesFromDtGnuHash(&number_of_symbol_table_entries)) {
    LOG(ERROR) << "could not retrieve number of symbol table entries";
    return false;
  }

  symbol_table_.reset(new ElfSymbolTableReader(
      &memory_, this, symbol_table_address, number_of_symbol_table_entries));
  symbol_table_initialized_.set_valid();
  return true;
}

bool ElfImageReader::GetAddressFromDynamicArray(uint64_t tag,
                                                bool log,
                                                VMAddress* address) {
  if (!dynamic_array_->GetValue(tag, log, address)) {
    return false;
  }
#if defined(OS_ANDROID) || defined(OS_FUCHSIA)
  // The GNU loader updates the dynamic array according to the load bias.
  // The Android and Fuchsia loaders only update the debug address.
  if (tag != DT_DEBUG) {
    *address += GetLoadBias();
  }
#endif  // OS_ANDROID
  return true;
}

bool ElfImageReader::GetNumberOfSymbolEntriesFromDtHash(
    VMSize* number_of_symbol_table_entries) {
  if (!InitializeDynamicArray()) {
    return false;
  }

  VMAddress dt_hash_address;
  if (!GetAddressFromDynamicArray(DT_HASH, false, &dt_hash_address)) {
    return false;
  }

  struct {
    uint32_t nbucket;
    uint32_t nchain;
  } header;

  if (!memory_.Read(dt_hash_address, sizeof(header), &header)) {
    LOG(ERROR) << "failed to read DT_HASH header";
    return false;
  }

  *number_of_symbol_table_entries = header.nchain;
  return true;
}

bool ElfImageReader::GetNumberOfSymbolEntriesFromDtGnuHash(
    VMSize* number_of_symbol_table_entries) {
  if (!InitializeDynamicArray()) {
    return false;
  }

  VMAddress dt_gnu_hash_address;
  if (!GetAddressFromDynamicArray(DT_GNU_HASH, false, &dt_gnu_hash_address)) {
    return false;
  }

  // See https://flapenguin.me/2017/05/10/elf-lookup-dt-gnu-hash/ and
  // https://sourceware.org/ml/binutils/2006-10/msg00377.html.
  struct {
    uint32_t nbuckets;
    uint32_t symoffset;
    uint32_t bloom_size;
    uint32_t bloom_shift;
  } header;
  if (!memory_.Read(dt_gnu_hash_address, sizeof(header), &header)) {
    LOG(ERROR) << "failed to read DT_GNU_HASH header";
    return false;
  }

  std::vector<uint32_t> buckets(header.nbuckets);
  const size_t kNumBytesForBuckets = sizeof(buckets[0]) * buckets.size();
  const size_t kWordSize =
      memory_.Is64Bit() ? sizeof(uint64_t) : sizeof(uint32_t);
  const VMAddress buckets_address =
      dt_gnu_hash_address + sizeof(header) + (kWordSize * header.bloom_size);
  if (!memory_.Read(buckets_address, kNumBytesForBuckets, buckets.data())) {
    LOG(ERROR) << "read buckets";
    return false;
  }

  // Locate the chain that handles the largest index bucket.
  uint32_t last_symbol = 0;
  for (uint32_t i = 0; i < header.nbuckets; ++i) {
    last_symbol = std::max(buckets[i], last_symbol);
  }

  if (last_symbol < header.symoffset) {
    *number_of_symbol_table_entries = header.symoffset;
    return true;
  }

  // Walk the bucket's chain to add the chain length to the total.
  const VMAddress chains_base_address = buckets_address + kNumBytesForBuckets;
  for (;;) {
    uint32_t chain_entry;
    if (!memory_.Read(chains_base_address + (last_symbol - header.symoffset) *
                                                sizeof(chain_entry),
                      sizeof(chain_entry),
                      &chain_entry)) {
      LOG(ERROR) << "read chain entry";
      return false;
    }

    ++last_symbol;

    // If the low bit is set, this entry is the end of the chain.
    if (chain_entry & 1)
      break;
  }

  *number_of_symbol_table_entries = last_symbol;
  return true;
}

std::unique_ptr<ElfImageReader::NoteReader> ElfImageReader::Notes(
    ssize_t max_note_size) {
  return std::make_unique<NoteReader>(
      this, &memory_, program_headers_.get(), max_note_size);
}

std::unique_ptr<ElfImageReader::NoteReader>
ElfImageReader::NotesWithNameAndType(const std::string& name,
                                     NoteReader::NoteType type,
                                     ssize_t max_note_size) {
  return std::make_unique<NoteReader>(
      this, &memory_, program_headers_.get(), max_note_size, name, type, true);
}

const ProcessMemoryRange* ElfImageReader::Memory() const {
  return &memory_;
}

}  // namespace crashpad
