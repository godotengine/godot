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

#ifndef CRASHPAD_SNAPSHOT_ELF_ELF_SYMBOL_TABLE_READER_H_
#define CRASHPAD_SNAPSHOT_ELF_ELF_SYMBOL_TABLE_READER_H_

#include <stdint.h>

#include <string>

#include "base/macros.h"
#include "util/misc/address_types.h"
#include "util/process/process_memory_range.h"

namespace crashpad {

class ElfImageReader;

//! \brief A reader for symbol tables in ELF images mapped into another process.
class ElfSymbolTableReader {
 public:
  //! \brief Information about a symbol in a module's symbol table.
  struct SymbolInformation {
    //! \brief The address of the symbol as it exists in the symbol table, not
    //!     adjusted for any load bias.
    VMAddress address;

    //! \brief The size of the symbol.
    VMSize size;

    //! \brief The section index that the symbol definition is in relation to.
    uint16_t shndx;

    //! \brief Specifies the type of symbol. Possible values include
    //!     `STT_OBJECT`, `STT_FUNC`, etc.
    uint8_t type;

    //! \brief Specifies the default scope at which a symbol takes precedence.
    //!     Possible values include `STB_LOCAL`, `STB_GLOBAL`, `STB_WEAK`, or
    //!     OS/processor specific values.
    uint8_t binding;

    //! \brief Together with binding, can limit the visibility of a symbol to
    //!     the module that defines it. Possible values include `STV_DEFAULT`,
    //!     `STV_INTERNAL`, `STV_HIDDEN`, and `STV_PROTECTED`.
    uint8_t visibility;
  };

  // TODO(jperaza): Support using .hash and .gnu.hash sections to improve symbol
  // lookup.
  ElfSymbolTableReader(const ProcessMemoryRange* memory,
                       ElfImageReader* elf_reader,
                       VMAddress address,
                       VMSize num_entries);
  ~ElfSymbolTableReader();

  //! \brief Lookup information about a symbol.
  //!
  //! \param[in] name The name of the symbol to search for.
  //! \param[out] info The symbol information, if found.
  //! \return `true` if the symbol is found.
  bool GetSymbol(const std::string& name, SymbolInformation* info);

 private:
  template <typename SymEnt>
  bool ScanSymbolTable(const std::string& name, SymbolInformation* info);

  const ProcessMemoryRange* const memory_;  // weak
  ElfImageReader* const elf_reader_;  // weak
  const VMAddress base_address_;
  const VMSize num_entries_;

  DISALLOW_COPY_AND_ASSIGN(ElfSymbolTableReader);
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_ELF_ELF_SYMBOL_TABLE_READER_H_
