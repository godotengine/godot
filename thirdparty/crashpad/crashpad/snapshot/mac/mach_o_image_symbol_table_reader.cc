// Copyright 2014 The Crashpad Authors. All rights reserved.
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

#include "snapshot/mac/mach_o_image_symbol_table_reader.h"

#include <mach-o/loader.h>
#include <mach-o/nlist.h>
#include <sys/types.h>

#include <memory>
#include <utility>

#include "base/strings/stringprintf.h"
#include "util/mac/checked_mach_address_range.h"
#include "util/mach/task_memory.h"

namespace crashpad {

namespace internal {

//! \brief The internal implementation for MachOImageSymbolTableReader.
//!
//! Initialization is broken into more than one function that needs to share
//! data, so member variables are used. However, much of this data is irrelevant
//! after initialization is completed, so rather than doing it in
//! MachOImageSymbolTableReader, it’s handled by this class, which is a “friend”
//! of MachOImageSymbolTableReader.
class MachOImageSymbolTableReaderInitializer {
 public:
  MachOImageSymbolTableReaderInitializer(
      ProcessReaderMac* process_reader,
      const MachOImageSegmentReader* linkedit_segment,
      const std::string& module_info)
      : module_info_(module_info),
        linkedit_range_(),
        process_reader_(process_reader),
        linkedit_segment_(linkedit_segment) {
    linkedit_range_.SetRange(process_reader_->Is64Bit(),
                             linkedit_segment->Address(),
                             linkedit_segment->Size());
    DCHECK(linkedit_range_.IsValid());
  }

  ~MachOImageSymbolTableReaderInitializer() {}

  //! \brief Reads the symbol table from another process.
  //!
  //! \sa MachOImageSymbolTableReader::Initialize()
  bool Initialize(const process_types::symtab_command* symtab_command,
                  const process_types::dysymtab_command* dysymtab_command,
                  MachOImageSymbolTableReader::SymbolInformationMap*
                      external_defined_symbols) {
    mach_vm_address_t symtab_address =
        AddressForLinkEditComponent(symtab_command->symoff);
    uint32_t symbol_count = symtab_command->nsyms;
    size_t nlist_size = process_types::nlist::ExpectedSize(process_reader_);
    mach_vm_size_t symtab_size = symbol_count * nlist_size;
    if (!IsInLinkEditSegment(symtab_address, symtab_size, "symtab")) {
      return false;
    }

    // If a dysymtab is present, use it to filter the symtab for just the
    // portion used for extdefsym. If no dysymtab is present, the entire symtab
    // will need to be consulted.
    uint32_t skip_count = 0;
    if (dysymtab_command) {
      if (dysymtab_command->iextdefsym >= symtab_command->nsyms ||
          dysymtab_command->iextdefsym + dysymtab_command->nextdefsym >
              symtab_command->nsyms) {
        LOG(WARNING) << base::StringPrintf(
                            "dysymtab extdefsym %u + %u > symtab nsyms %u",
                            dysymtab_command->iextdefsym,
                            dysymtab_command->nextdefsym,
                            symtab_command->nsyms) << module_info_;
        return false;
      }

      skip_count = dysymtab_command->iextdefsym;
      mach_vm_size_t skip_size = skip_count * nlist_size;
      symtab_address += skip_size;
      symtab_size -= skip_size;
      symbol_count = dysymtab_command->nextdefsym;
    }

    mach_vm_address_t strtab_address =
        AddressForLinkEditComponent(symtab_command->stroff);
    mach_vm_size_t strtab_size = symtab_command->strsize;
    if (!IsInLinkEditSegment(strtab_address, strtab_size, "strtab")) {
      return false;
    }

    std::unique_ptr<process_types::nlist[]> symbols(
        new process_types::nlist[symtab_command->nsyms]);
    if (!process_types::nlist::ReadArrayInto(
            process_reader_, symtab_address, symbol_count, &symbols[0])) {
      LOG(WARNING) << "could not read symbol table" << module_info_;
      return false;
    }

    std::unique_ptr<TaskMemory::MappedMemory> string_table;
    for (size_t symbol_index = 0; symbol_index < symbol_count; ++symbol_index) {
      const process_types::nlist& symbol = symbols[symbol_index];
      std::string symbol_info = base::StringPrintf(", symbol index %zu%s",
                                                   skip_count + symbol_index,
                                                   module_info_.c_str());
      bool valid_symbol = true;
      if ((symbol.n_type & N_STAB) == 0 && (symbol.n_type & N_PEXT) == 0 &&
          (symbol.n_type & N_EXT)) {
        uint8_t symbol_type = symbol.n_type & N_TYPE;
        if (symbol_type == N_ABS || symbol_type == N_SECT) {
          if (symbol.n_strx >= strtab_size) {
            LOG(WARNING) << base::StringPrintf(
                                "string at 0x%x out of bounds (0x%llx)",
                                symbol.n_strx,
                                strtab_size) << symbol_info;
            return false;
          }

          if (!string_table) {
            string_table = process_reader_->Memory()->ReadMapped(
                strtab_address, strtab_size);
            if (!string_table) {
              LOG(WARNING) << "could not read string table" << module_info_;
              return false;
            }
          }

          std::string name;
          if (!string_table->ReadCString(symbol.n_strx, &name)) {
            LOG(WARNING) << "could not read string" << symbol_info;
            return false;
          }

          if (symbol_type == N_ABS && symbol.n_sect != NO_SECT) {
            LOG(WARNING) << base::StringPrintf("N_ABS symbol %s in section %u",
                                               name.c_str(),
                                               symbol.n_sect) << symbol_info;
            return false;
          }

          if (symbol_type == N_SECT && symbol.n_sect == NO_SECT) {
            LOG(WARNING) << base::StringPrintf(
                                "N_SECT symbol %s in section NO_SECT",
                                name.c_str()) << symbol_info;
            return false;
          }

          MachOImageSymbolTableReader::SymbolInformation this_symbol_info;
          this_symbol_info.value = symbol.n_value;
          this_symbol_info.section = symbol.n_sect;
          if (!external_defined_symbols->insert(
                  std::make_pair(name, this_symbol_info)).second) {
            LOG(WARNING) << "duplicate symbol " << name << symbol_info;
            return false;
          }
        } else {
          // External indirect symbols may be found in the portion of the symbol
          // table used for external symbols as opposed to indirect symbols when
          // the indirect symbols are also external. These can be produced by
          // Xcode 5.1 ld64-236.3/src/ld/LinkEditClassic.hpp
          // ld::tool::SymbolTableAtom<>::addGlobal(). Indirect symbols are not
          // currently supported by this symbol table reader, so ignore them
          // without failing or logging a message when encountering them. See
          // https://groups.google.com/a/chromium.org/d/topic/crashpad-dev/k7QkLwO71Zo
          valid_symbol = symbol_type == N_INDR;
        }
      } else {
        valid_symbol = false;
      }
      if (!valid_symbol && dysymtab_command) {
        LOG(WARNING) << "non-external symbol with type " << symbol.n_type
                     << " in extdefsym" << symbol_info;
        return false;
      }
    }

    return true;
  }

 private:
  //! \brief Computes the address for data in the `__LINKEDIT` segment
  //!     identified by its file offset in a Mach-O image.
  //!
  //! \param[in] fileoff The file offset relative to the beginning of an image’s
  //!     `mach_header` or `mach_header_64` of the data in the `__LINKEDIT`
  //!     segment.
  //!
  //! \return The address, in the remote process’ address space, of the
  //!     requested data.
  mach_vm_address_t AddressForLinkEditComponent(uint32_t fileoff) const {
    return linkedit_range_.Base() + fileoff - linkedit_segment_->fileoff();
  }

  //! \brief Determines whether an address range is located within the
  //!     `__LINKEDIT` segment.
  //!
  //! \param[in] address The base address of the range to check.
  //! \param[in] size The size of the range to check.
  //! \param[in] tag A string that identifies the range being checked. This is
  //!     used only for logging.
  //!
  //! \return `true` if the range identified by \a address + \a size lies
  //!     entirely within the `__LINKEDIT` segment. `false` if that range is
  //!     invalid, or if that range is not contained by the `__LINKEDIT`
  //!     segment, with an appropriate message logged.
  bool IsInLinkEditSegment(mach_vm_address_t address,
                           mach_vm_size_t size,
                           const char* tag) const {
    CheckedMachAddressRange subrange(process_reader_->Is64Bit(), address, size);
    if (!subrange.IsValid()) {
      LOG(WARNING) << base::StringPrintf("invalid %s range (0x%llx + 0x%llx)",
                                         tag,
                                         address,
                                         size) << module_info_;
      return false;
    }

    if (!linkedit_range_.ContainsRange(subrange)) {
      LOG(WARNING) << base::StringPrintf(
                          "%s at 0x%llx + 0x%llx outside of " SEG_LINKEDIT
                          " segment at 0x%llx + 0x%llx",
                          tag,
                          address,
                          size,
                          linkedit_range_.Base(),
                          linkedit_range_.Size()) << module_info_;
      return false;
    }

    return true;
  }

  std::string module_info_;
  CheckedMachAddressRange linkedit_range_;
  ProcessReaderMac* process_reader_;  // weak
  const MachOImageSegmentReader* linkedit_segment_;  // weak

  DISALLOW_COPY_AND_ASSIGN(MachOImageSymbolTableReaderInitializer);
};

}  // namespace internal

MachOImageSymbolTableReader::MachOImageSymbolTableReader()
    : external_defined_symbols_(), initialized_() {
}

MachOImageSymbolTableReader::~MachOImageSymbolTableReader() {
}

bool MachOImageSymbolTableReader::Initialize(
    ProcessReaderMac* process_reader,
    const process_types::symtab_command* symtab_command,
    const process_types::dysymtab_command* dysymtab_command,
    const MachOImageSegmentReader* linkedit_segment,
    const std::string& module_info) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);

  internal::MachOImageSymbolTableReaderInitializer initializer(process_reader,
                                                               linkedit_segment,
                                                               module_info);
  if (!initializer.Initialize(
          symtab_command, dysymtab_command, &external_defined_symbols_)) {
    return false;
  }

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

const MachOImageSymbolTableReader::SymbolInformation*
MachOImageSymbolTableReader::LookUpExternalDefinedSymbol(
    const std::string& name) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  const auto& iterator = external_defined_symbols_.find(name);
  if (iterator == external_defined_symbols_.end()) {
    return nullptr;
  }
  return &iterator->second;
}

}  // namespace crashpad
