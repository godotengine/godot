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

#ifndef CRASHPAD_SNAPSHOT_ELF_ELF_IMAGE_READER_H_
#define CRASHPAD_SNAPSHOT_ELF_ELF_IMAGE_READER_H_

#include <elf.h>
#include <stdint.h>
#include <sys/types.h>

#include <memory>
#include <string>

#include "base/macros.h"
#include "snapshot/elf/elf_dynamic_array_reader.h"
#include "snapshot/elf/elf_symbol_table_reader.h"
#include "util/misc/address_types.h"
#include "util/misc/initialization_state.h"
#include "util/misc/initialization_state_dcheck.h"
#include "util/process/process_memory_range.h"

namespace crashpad {

//! \brief A reader for ELF images mapped into another process.
//!
//! This class is capable of reading both 32-bit and 64-bit images.
class ElfImageReader {
 private:
  class ProgramHeaderTable;

 public:
  //! \brief This class enables reading note segments from an ELF image.
  //!
  //! Objects of this class should be created by calling
  //! ElfImageReader::Notes() or ElfImageReader::NotesWithNameAndType().
  class NoteReader {
   public:
    ~NoteReader();

    //! \brief The return value for NextNote().
    enum class Result {
      //! \brief An error occurred. The NoteReader is invalidated and message is
      //!     logged.
      kError,

      //! \brief A note was found.
      kSuccess,

      //! \brief No more notes were found.
      kNoMoreNotes,
    };

    //! \brief A type large enough to hold a note type, potentially across
    //!     bitness.
    using NoteType = decltype(Elf64_Nhdr::n_type);

    //! \brief Searches for the next note in the image.
    //!
    //! \param[out] name The name of the note owner, if not `nullptr`.
    //! \param[out] type A type for the note, if not `nullptr`.
    //! \param[out] desc The note descriptor.
    //! \return a #Result value. \a name, \a type, and \a desc are only valid if
    //!     this method returns Result::kSuccess.
    Result NextNote(std::string* name, NoteType* type, std::string* desc);

    // private
    NoteReader(const ElfImageReader* elf_reader_,
               const ProcessMemoryRange* range,
               const ProgramHeaderTable* phdr_table,
               ssize_t max_note_size,
               const std::string& name_filter = std::string(),
               NoteType type_filter = 0,
               bool use_filter = false);

   private:
    // Reads the next note at the current segment address. Sets retry_ to true
    // and returns kError if use_filter_ is true and the note's name and type do
    // not match name_filter_ and type_filter_.
    template <typename T>
    Result ReadNote(std::string* name, NoteType* type, std::string* desc);

    VMAddress current_address_;
    VMAddress segment_end_address_;
    const ElfImageReader* elf_reader_;  // weak
    const ProcessMemoryRange* range_;  // weak
    const ProgramHeaderTable* phdr_table_;  // weak
    std::unique_ptr<ProcessMemoryRange> segment_range_;
    size_t phdr_index_;
    ssize_t max_note_size_;
    std::string name_filter_;
    NoteType type_filter_;
    bool use_filter_;
    bool is_valid_;
    bool retry_;

    DISALLOW_COPY_AND_ASSIGN(NoteReader);
  };

  ElfImageReader();
  ~ElfImageReader();

  //! \brief Initializes the reader.
  //!
  //! This method must be called once on an object and must be successfully
  //! called before any other method in this class may be called.
  //!
  //! \param[in] memory A memory reader for the remote process.
  //! \param[in] address The address in the remote process' address space where
  //!     the ELF image is loaded.
  //! \param[in] verbose `true` if this method should log error messages during
  //!     initialization. Setting this value to `false` will reduce the error
  //!     messages relating to verifying the ELF image, but may not suppress
  //!     logging entirely.
  bool Initialize(const ProcessMemoryRange& memory,
                  VMAddress address,
                  bool verbose = true);

  //! \brief Returns the base address of the image's memory range.
  //!
  //! This may differ from the address passed to Initialize() if the ELF header
  //! is not loaded at the start of the first `PT_LOAD` segment.
  VMAddress Address() const { return memory_.Base(); }

  //! \brief Returns the size of the range containing all loaded segments for
  //!     this image.
  //!
  //! The size may include memory that is unmapped or mapped to other objects if
  //! this image's `PT_LOAD` segments are not contiguous.
  VMSize Size() const { return memory_.Size(); }

  //! \brief Returns the file type for the image.
  //!
  //! Possible values include `ET_EXEC` or `ET_DYN` from `<elf.h>`.
  uint16_t FileType() const;

  //! \brief Returns the load bias for the image.
  //!
  //! The load bias is the actual load address minus the preferred load address.
  VMOffset GetLoadBias() const { return load_bias_; }

  //! \brief Determines the name of this object using `DT_SONAME`, if present.
  //!
  //! \param[out] name The name of this object, only valid if this method
  //!     returns `true`.
  //! \return `true` if a name was found for this object.
  bool SoName(std::string* name);

  //! \brief Reads information from the dynamic symbol table about the symbol
  //!     identified by \a name.
  //!
  //! \param[in] name The name of the symbol to search for.
  //! \param[out] address The address of the symbol in the target process'
  //!     address space, if found.
  //! \param[out] size The size of the symbol, if found.
  //! \return `true` if the symbol was found.
  bool GetDynamicSymbol(const std::string& name,
                        VMAddress* address,
                        VMSize* size);

  //! \brief Reads a `NUL`-terminated C string from this image's dynamic string
  //!     table.
  //!
  //! \param[in] offset the byte offset in the string table to start reading.
  //! \param[out] string the string read.
  //! \return `true` on success. Otherwise `false` with a message logged.
  bool ReadDynamicStringTableAtOffset(VMSize offset, std::string* string);

  //! \brief Determine the debug address.
  //!
  //! The debug address is a pointer to an `r_debug` struct defined in
  //! `<link.h>`.
  //!
  //! \param[out] debug the debug address, if found.
  //! \return `true` if the debug address was found.
  bool GetDebugAddress(VMAddress* debug);

  //! \brief Determine the address of `PT_DYNAMIC` segment.
  //!
  //! \param[out] address The address of the array, valid if this method returns
  //!     `true`.
  //! \return `true` on success. Otherwise `false` with a message logged.
  bool GetDynamicArrayAddress(VMAddress* address);

  //! \brief Return the address of the program header table.
  VMAddress GetProgramHeaderTableAddress();

  //! \brief Return a NoteReader for this image, which scans all PT_NOTE
  //!     segments in the image.
  //!
  //! The returned NoteReader is only valid for the lifetime of the
  //! ElfImageReader that created it.
  //!
  //! \param[in] max_note_size The maximum note size to read. Notes whose
  //!     combined name, descriptor, and padding size are greater than
  //!     \a max_note_size will be silently skipped. A \a max_note_size of -1
  //!     indicates infinite maximum note size.
  //! \return A NoteReader object capable of reading notes in this image.
  std::unique_ptr<NoteReader> Notes(ssize_t max_note_size);

  //! \brief Return a NoteReader for this image, which scans all PT_NOTE
  //!     segments in the image, filtering by name and type.
  //!
  //! The returned NoteReader is only valid for the lifetime of the
  //! ElfImageReader that created it.
  //!
  //! \param[in] name The note name to match.
  //! \param[in] type The note type to match.
  //! \param[in] max_note_size The maximum note size to read. Notes whose
  //!     combined name, descriptor, and padding size are greater than
  //!     \a max_note_size will be silently skipped. A \a max_note_size of -1
  //!     indicates infinite maximum note size.
  //! \return A NoteReader object capable of reading notes in this image.
  std::unique_ptr<NoteReader> NotesWithNameAndType(const std::string& name,
                                                   NoteReader::NoteType type,
                                                   ssize_t max_note_size);

  //! \brief Return a ProcessMemoryRange restricted to the range of this image.
  //!
  //! The caller does not take ownership of the returned object.
  const ProcessMemoryRange* Memory() const;

  //! \brief Retrieves the number of symbol table entries in `DT_SYMTAB`
  //!     according to the data in the `DT_HASH` section.
  //!
  //! \note Exposed for testing, not normally otherwise useful.
  //!
  //! \param[out] number_of_symbol_table_entries The number of entries expected
  //!     in `DT_SYMTAB`.
  //! \return `true` if a `DT_HASH` section was found, and was read
  //!     successfully, otherwise `false` with an error logged.
  bool GetNumberOfSymbolEntriesFromDtHash(
      VMSize* number_of_symbol_table_entries);

  //! \brief Retrieves the number of symbol table entries in `DT_SYMTAB`
  //!     according to the data in the `DT_GNU_HASH` section.
  //!
  //! \note Exposed for testing, not normally otherwise useful.
  //!
  //! \note Depending on the linker that generated the `DT_GNU_HASH` section,
  //!     this value may not be as expected if there are zero exported symbols.
  //!
  //! \param[out] number_of_symbol_table_entries The number of entries expected
  //!     in `DT_SYMTAB`.
  //! \return `true` if a `DT_GNU_HASH` section was found, and was read
  //!     successfully, otherwise `false` with an error logged.
  bool GetNumberOfSymbolEntriesFromDtGnuHash(
      VMSize* number_of_symbol_table_entries);

 private:
  template <typename PhdrType>
  class ProgramHeaderTableSpecific;

  bool InitializeProgramHeaders(bool verbose);
  bool InitializeDynamicArray();
  bool InitializeDynamicSymbolTable();
  bool GetAddressFromDynamicArray(uint64_t tag, bool log, VMAddress* address);

  union {
    Elf32_Ehdr header_32_;
    Elf64_Ehdr header_64_;
  };
  VMAddress ehdr_address_;
  VMOffset load_bias_;
  ProcessMemoryRange memory_;
  std::unique_ptr<ProgramHeaderTable> program_headers_;
  std::unique_ptr<ElfDynamicArrayReader> dynamic_array_;
  std::unique_ptr<ElfSymbolTableReader> symbol_table_;
  InitializationStateDcheck initialized_;
  InitializationState dynamic_array_initialized_;
  InitializationState symbol_table_initialized_;

  DISALLOW_COPY_AND_ASSIGN(ElfImageReader);
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_ELF_ELF_IMAGE_READER_H_
