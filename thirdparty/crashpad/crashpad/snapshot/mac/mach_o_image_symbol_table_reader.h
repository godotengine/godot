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

#ifndef CRASHPAD_SNAPSHOT_MAC_MACH_O_IMAGE_SYMBOL_TABLE_READER_H_
#define CRASHPAD_SNAPSHOT_MAC_MACH_O_IMAGE_SYMBOL_TABLE_READER_H_

#include <map>
#include <string>

#include <mach/mach.h>
#include <stdint.h>

#include "base/macros.h"
#include "snapshot/mac/mach_o_image_segment_reader.h"
#include "snapshot/mac/process_reader_mac.h"
#include "snapshot/mac/process_types.h"
#include "util/misc/initialization_state_dcheck.h"

namespace crashpad {

//! \brief A reader for symbol tables in Mach-O images mapped into another
//!     process.
class MachOImageSymbolTableReader {
 public:
  //! \brief Information about a symbol in a module’s symbol table.
  //!
  //! This is a more minimal form of the `nlist` (or `nlist_64`) structure,
  //! only containing the equivalent of the `n_value` and `n_sect` fields.
  struct SymbolInformation {
    //! \brief The address of the symbol as it exists in the symbol table, not
    //!     adjusted for any “slide.”
    mach_vm_address_t value;

    //! \brief The 1-based section index in the module in which the symbol is
    //!     found.
    //!
    //! For symbols defined in a section (`N_SECT`), this is the section index
    //! that can be passed to MachOImageReader::GetSectionAtIndex(), and \a
    //! value will need to be adjusted for segment slide if the containing
    //! segment slid when loaded. For absolute symbols (`N_ABS`), this will be
    //! `NO_SECT` (`0`), and \a value must not be adjusted for segment slide.
    uint8_t section;
  };

  // TODO(mark): Use std::unordered_map or a similar hash-based map? For now,
  // std::map is fine because this map only stores external defined symbols, and
  // there aren’t expected to be very many of those that performance would
  // become a problem. In reality, std::unordered_map does not appear to provide
  // a performance advantage. It appears that the memory copies currently done
  // by TaskMemory::Read() have substantially more impact on symbol table
  // operations.
  //
  // This is public so that the type is available to
  // MachOImageSymbolTableReaderInitializer.
  using SymbolInformationMap = std::map<std::string, SymbolInformation>;

  MachOImageSymbolTableReader();
  ~MachOImageSymbolTableReader();

  //! \brief Reads the symbol table from another process.
  //!
  //! This method must only be called once on an object. This method must be
  //! called successfully before any other method in this class may be called.
  //!
  //! \param[in] process_reader The reader for the remote process.
  //! \param[in] symtab_command The `LC_SYMTAB` load command that identifies
  //!     the symbol table.
  //! \param[in] dysymtab_command The `LC_DYSYMTAB` load command that identifies
  //!     dynamic symbol information within the symbol table. This load command
  //!     is not present in all modules, and this parameter may be `nullptr` for
  //!     modules that do not have this information. When present, \a
  //!     dysymtab_command is an optimization that allows the symbol table
  //!     reader to only examine symbol table entries known to be relevant for
  //!     its purposes.
  //! \param[in] linkedit_segment The `__LINKEDIT` segment. This segment should
  //!     contain the data referenced by \a symtab_command and \a
  //!     dysymtab_command. This may be any segment in the module, but by
  //!     convention, the name `__LINKEDIT` is used for this purpose.
  //! \param[in] module_info A string to be used in logged messages. This string
  //!     is for diagnostic purposes only, and may be empty.
  //!
  //! \return `true` if the symbol table was read successfully. `false`
  //!     otherwise, with an appropriate message logged.
  bool Initialize(ProcessReaderMac* process_reader,
                  const process_types::symtab_command* symtab_command,
                  const process_types::dysymtab_command* dysymtab_command,
                  const MachOImageSegmentReader* linkedit_segment,
                  const std::string& module_info);

  //! \brief Looks up a symbol in the image’s symbol table.
  //!
  //! The returned information captures the symbol as it exists in the image’s
  //! symbol table, not adjusted for any “slide.”
  //!
  //! \param[in] name The name of the symbol to look up, “mangled” or
  //!     “decorated” appropriately. For example, use `"_main"` to look up the
  //!     symbol for the C `main()` function, and use `"__Z4Funcv"` to look up
  //!     the symbol for the C++ `Func()` function.
  //!
  //! \return A SymbolInformation* object with information about the symbol if
  //!     it was found, or `nullptr` if the symbol was not found or if an error
  //!     occurred. On error, a warning message will also be logged. The caller
  //!     does not take ownership; the lifetime of the returned object is scoped
  //!     to the lifetime of this MachOImageSymbolTableReader object.
  //!
  //! \note Symbol values returned via this interface are not adjusted for
  //!     “slide.” For slide-adjusted values, use the higher-level
  //!     MachOImageReader::LookUpExternalDefinedSymbol() interface.
  const SymbolInformation* LookUpExternalDefinedSymbol(
      const std::string& name) const;

 private:
  SymbolInformationMap external_defined_symbols_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(MachOImageSymbolTableReader);
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_MAC_MACH_O_IMAGE_SYMBOL_TABLE_READER_H_
