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

#ifndef CRASHPAD_SNAPSHOT_MAC_MACH_O_IMAGE_READER_H_
#define CRASHPAD_SNAPSHOT_MAC_MACH_O_IMAGE_READER_H_

#include <mach/mach.h>
#include <stdint.h>
#include <sys/types.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "base/macros.h"
#include "snapshot/mac/process_types.h"
#include "util/misc/initialization_state_dcheck.h"
#include "util/misc/uuid.h"

namespace crashpad {

class MachOImageSegmentReader;
class MachOImageSymbolTableReader;
class ProcessReaderMac;

//! \brief A reader for Mach-O images mapped into another process.
//!
//! This class is capable of reading both 32-bit (`mach_header`/`MH_MAGIC`) and
//! 64-bit (`mach_header_64`/`MH_MAGIC_64`) images based on the bitness of the
//! remote process.
//!
//! \sa MachOImageAnnotationsReader
class MachOImageReader {
 public:
  MachOImageReader();
  ~MachOImageReader();

  //! \brief Reads the Mach-O image file’s load commands from another process.
  //!
  //! This method must only be called once on an object. This method must be
  //! called successfully before any other method in this class may be called.
  //!
  //! \param[in] process_reader The reader for the remote process.
  //! \param[in] address The address, in the remote process’ address space,
  //!     where the `mach_header` or `mach_header_64` at the beginning of the
  //!     image to be read is located. This address can be determined by reading
  //!     the remote process’ dyld information (see
  //!     snapshot/mac/process_types/dyld_images.proctype).
  //! \param[in] name The module’s name, a string to be used in logged messages.
  //!     This string is for diagnostic purposes and to relax otherwise strict
  //!     parsing rules for common modules with known defects.
  //!
  //! \return `true` if the image was read successfully, including all load
  //!     commands. `false` otherwise, with an appropriate message logged.
  bool Initialize(ProcessReaderMac* process_reader,
                  mach_vm_address_t address,
                  const std::string& name);

  //! \brief Returns the Mach-O file type.
  //!
  //! This value comes from the `filetype` field of the `mach_header` or
  //! `mach_header_64`. Common values include `MH_EXECUTE`, `MH_DYLIB`,
  //! `MH_DYLINKER`, and `MH_BUNDLE`.
  uint32_t FileType() const { return file_type_; }

  //! \brief Returns the Mach-O image’s load address.
  //!
  //! This is the value passed as \a address to Initialize().
  mach_vm_address_t Address() const { return address_; }

  //! \brief Returns the mapped size of the Mach-O image’s `__TEXT` segment.
  //!
  //! Note that this is returns only the size of the `__TEXT` segment, not of
  //! any other segment. This is because the interface only allows one load
  //! address and size to be reported, but Mach-O image files may consist of
  //! multiple discontiguous segments. By convention, the `__TEXT` segment is
  //! always mapped at the beginning of a Mach-O image file, and it is the most
  //! useful for the expected intended purpose of collecting data to obtain
  //! stack backtraces. The implementation insists during initialization that
  //! the `__TEXT` segment be mapped at the beginning of the file.
  //!
  //! In practice, discontiguous segments are only found for images that have
  //! loaded out of the dyld shared cache, but the `__TEXT` segment’s size is
  //! returned for modules that loaded with contiguous segments as well for
  //! consistency.
  mach_vm_size_t Size() const { return size_; }

  //! \brief Returns the Mach-O image’s “slide,” the difference between its
  //!     actual load address and its preferred load address.
  //!
  //! “Slide” is computed by subtracting the `__TEXT` segment’s preferred load
  //! address from its actual load address. It will be reported as a positive
  //! offset when the actual load address is greater than the preferred load
  //! address. The preferred load address is taken to be the segment’s reported
  //! `vmaddr` value.
  mach_vm_size_t Slide() const { return slide_; }

  //! \brief Obtain segment information by segment name.
  //!
  //! \param[in] segment_name The name of the segment to search for, for
  //!     example, `"__TEXT"`.
  //!
  //! \return A pointer to the segment information if it was found, or `nullptr`
  //!     if it was not found. The caller does not take ownership; the lifetime
  //!     of the returned object is scoped to the lifetime of this
  //!     MachOImageReader object.
  const MachOImageSegmentReader* GetSegmentByName(
      const std::string& segment_name) const;

  //! \brief Obtain section information by segment and section name.
  //!
  //! \param[in] segment_name The name of the segment to search for, for
  //!     example, `"__TEXT"`.
  //! \param[in] section_name The name of the section within the segment to
  //!     search for, for example, `"__text"`.
  //! \param[out] address The actual address that the section was loaded at in
  //!     memory, taking any “slide” into account if the section did not load at
  //!     its preferred address as stored in the Mach-O image file. This
  //!     parameter can be `nullptr`.
  //!
  //! \return A pointer to the section information if it was found, or `nullptr`
  //!     if it was not found. The caller does not take ownership; the lifetime
  //!     of the returned object is scoped to the lifetime of this
  //!     MachOImageReader object.
  //!
  //! No parameter is provided for the section’s size, because it can be
  //! obtained from the returned process_types::section::size field.
  //!
  //! \note The process_types::section::addr field gives the section’s preferred
  //!     load address as stored in the Mach-O image file, and is not adjusted
  //!     for any “slide” that may have occurred when the image was loaded. Use
  //!     \a address to obtain the section’s actual load address.
  const process_types::section* GetSectionByName(
      const std::string& segment_name,
      const std::string& section_name,
      mach_vm_address_t* address) const;

  //! \brief Obtain section information by section index.
  //!
  //! \param[in] index The index of the section to return, in the order that it
  //!     appears in the segment load commands. This is a 1-based index,
  //!     matching the section number values used for `nlist::n_sect`.
  //! \param[out] containing_segment The segment that contains the section.
  //!     This parameter can be `nullptr`. The caller does not take ownership;
  //!     the lifetime of the returned object is scoped to the lifetime of this
  //!     MachOImageReader object.
  //! \param[out] address The actual address that the section was loaded at in
  //!     memory, taking any “slide” into account if the section did not load at
  //!     its preferred address as stored in the Mach-O image file. This
  //!     parameter can be `nullptr`.
  //!
  //! \return A pointer to the section information. If \a index is out of range,
  //!     logs a warning and returns `nullptr`. The caller does not take
  //!     ownership; the lifetime of the returned object is scoped to the
  //!     lifetime of this MachOImageReader object.
  //!
  //! No parameter is provided for the section’s size, because it can be
  //! obtained from the returned process_types::section::size field.
  //!
  //! \note The process_types::section::addr field gives the section’s preferred
  //!     load address as stored in the Mach-O image file, and is not adjusted
  //!     for any “slide” that may have occurred when the image was loaded. Use
  //!     \a address to obtain the section’s actual load address.
  //! \note Unlike MachOImageSegmentReader::GetSectionAtIndex(), this method
  //!     accepts out-of-range values for \a index, and returns `nullptr`
  //!     instead of aborting execution upon encountering an out-of-range value.
  //!     This is because a Mach-O image file’s symbol table refers to this
  //!     per-module section index, and an out-of-range index in that case
  //!     should be treated as a data error (where the data is beyond this
  //!     code’s control) and handled non-fatally by reporting the error to the
  //!     caller.
  const process_types::section* GetSectionAtIndex(
      size_t index,
      const MachOImageSegmentReader** containing_segment,
      mach_vm_address_t* address) const;

  //! \brief Looks up a symbol in the image’s symbol table.
  //!
  //! This method is capable of locating external defined symbols. Specifically,
  //! this method can look up symbols that have these charcteristics:
  //!  - `N_STAB` (debugging) and `N_PEXT` (private external) must not be set.
  //!  - `N_EXT` (external) must be set.
  //!  - The type must be `N_ABS` (absolute) or `N_SECT` (defined in section).
  //!
  //! `N_INDR` (indirect), `N_UNDF` (undefined), and `N_PBUD` (prebound
  //! undefined) symbols cannot be located through this mechanism.
  //!
  //! \param[in] name The name of the symbol to look up, “mangled” or
  //!     “decorated” appropriately. For example, use `"_main"` to look up the
  //!     symbol for the C `main()` function, and use `"__Z4Funcv"` to look up
  //!     the symbol for the C++ `Func()` function. Contrary to `dlsym()`, the
  //!     leading underscore must not be stripped when using this interface.
  //! \param[out] value If the lookup was successful, this will be set to the
  //!     value of the symbol, adjusted for any “slide” as needed. The value can
  //!     be used as an address in the remote process’ address space where the
  //!     pointee of the symbol exists in memory.
  //!
  //! \return `true` if the symbol lookup was successful and the symbol was
  //!     found. `false` otherwise, including error conditions (for which a
  //!     warning message will be logged), modules without symbol tables, and
  //!     symbol names not found in the symbol table.
  //!
  //! \note Symbol values returned via this interface are adjusted for “slide”
  //!     as appropriate, in contrast to the underlying implementation,
  //!     MachOImageSymbolTableReader::LookUpExternalDefinedSymbol().
  //!
  //! \warning Symbols that are resolved by running symbol resolvers
  //!     (`.symbol_resolver`) are not properly handled by this interface. The
  //!     address of the symbol resolver is returned because that’s what shows
  //!     up in the symbol table, rather than the effective address of the
  //!     resolved symbol as used by dyld after running the resolver. The only
  //!     way to detect this situation would be to read the `LC_DYLD_INFO` or
  //!     `LC_DYLD_INFO_ONLY` load command if present and looking for the
  //!     `EXPORT_SYMBOL_FLAGS_STUB_AND_RESOLVER` flag, but that would just be
  //!     able to detect symbols with a resolver, it would not be able to
  //!     resolve them from out-of-process, so it’s not currently done.
  bool LookUpExternalDefinedSymbol(const std::string& name,
                                   mach_vm_address_t* value) const;

  //! \brief Returns a Mach-O dylib image’s current version.
  //!
  //! This information comes from the `dylib_current_version` field of a dylib’s
  //! `LC_ID_DYLIB` load command. For dylibs without this load command, `0` will
  //! be returned.
  //!
  //! This method may only be called on Mach-O images for which FileType()
  //! returns `MH_DYLIB`.
  uint32_t DylibVersion() const;

  //! \brief Returns a Mach-O image’s source version.
  //!
  //! This information comes from a Mach-O image’s `LC_SOURCE_VERSION` load
  //! command. For Mach-O images without this load command, `0` will be
  //! returned.
  uint64_t SourceVersion() const { return source_version_; }

  //! \brief Returns a Mach-O image’s UUID.
  //!
  //! This information comes from a Mach-O image’s `LC_UUID` load command. For
  //! Mach-O images without this load command, a zeroed-out UUID value will be
  //! returned.
  //
  // UUID is a name in this scope (referring to this method), so the parameter’s
  // type needs to be qualified with |crashpad::|.
  void UUID(crashpad::UUID* uuid) const;

  //! \brief Returns the dynamic linker’s pathname.
  //!
  //! The dynamic linker is normally /usr/lib/dyld.
  //!
  //! For executable images (those with file type `MH_EXECUTE`), this is the
  //! name provided in the `LC_LOAD_DYLINKER` load command, if any. For dynamic
  //! linker images (those with file type `MH_DYLINKER`), this is the name
  //! provided in the `LC_ID_DYLINKER` load command. In other cases, this will
  //! be empty.
  std::string DylinkerName() const { return dylinker_name_; }

  //! \brief Obtains the module’s CrashpadInfo structure.
  //!
  //! \return `true` on success, `false` on failure. If the module does not have
  //!     a `__DATA,crashpad_info` section, this will return `false` without
  //!     logging any messages. Other failures will result in messages being
  //!     logged.
  bool GetCrashpadInfo(process_types::CrashpadInfo* crashpad_info) const;

 private:
  // A generic helper routine for the other Read*Command() methods.
  template <typename T>
  bool ReadLoadCommand(mach_vm_address_t load_command_address,
                       const std::string& load_command_info,
                       uint32_t expected_load_command_id,
                       T* load_command);

  // The Read*Command() methods are subroutines called by Initialize(). They are
  // responsible for reading a single load command. They may update the member
  // fields of their MachOImageReader object. If they can’t make sense of a load
  // command, they return false.
  bool ReadSegmentCommand(mach_vm_address_t load_command_address,
                          const std::string& load_command_info);
  bool ReadSymTabCommand(mach_vm_address_t load_command_address,
                         const std::string& load_command_info);
  bool ReadDySymTabCommand(mach_vm_address_t load_command_address,
                           const std::string& load_command_info);
  bool ReadIdDylibCommand(mach_vm_address_t load_command_address,
                          const std::string& load_command_info);
  bool ReadDylinkerCommand(mach_vm_address_t load_command_address,
                           const std::string& load_command_info);
  bool ReadUUIDCommand(mach_vm_address_t load_command_address,
                       const std::string& load_command_info);
  bool ReadSourceVersionCommand(mach_vm_address_t load_command_address,
                                const std::string& load_command_info);
  bool ReadUnexpectedCommand(mach_vm_address_t load_command_address,
                             const std::string& load_command_info);

  // Performs deferred initialization of the symbol table. Because a module’s
  // symbol table is often not needed, this is not handled in Initialize(), but
  // is done lazily, on-demand as needed.
  //
  // symbol_table_initialized_ will be transitioned to the appropriate state. If
  // initialization completes successfully, this will be the valid state.
  // Otherwise, it will be left in the invalid state and a warning message will
  // be logged.
  //
  // Note that if the object contains no symbol table, symbol_table_initialized_
  // will be set to the valid state, but symbol_table_ will be nullptr.
  void InitializeSymbolTable() const;

  std::vector<std::unique_ptr<MachOImageSegmentReader>> segments_;
  std::map<std::string, size_t> segment_map_;
  std::string module_name_;
  std::string module_info_;
  std::string dylinker_name_;
  crashpad::UUID uuid_;
  mach_vm_address_t address_;
  mach_vm_size_t size_;
  mach_vm_size_t slide_;
  uint64_t source_version_;
  std::unique_ptr<process_types::symtab_command> symtab_command_;
  std::unique_ptr<process_types::dysymtab_command> dysymtab_command_;

  // symbol_table_ (and symbol_table_initialized_) are mutable in order to
  // maintain LookUpExternalDefinedSymbol() as a const interface while allowing
  // lazy initialization via InitializeSymbolTable(). This is logical
  // const-ness, not physical const-ness.
  mutable std::unique_ptr<MachOImageSymbolTableReader> symbol_table_;

  std::unique_ptr<process_types::dylib_command> id_dylib_command_;
  ProcessReaderMac* process_reader_;  // weak
  uint32_t file_type_;
  InitializationStateDcheck initialized_;

  // symbol_table_initialized_ protects symbol_table_: symbol_table_ can only
  // be used when symbol_table_initialized_ is valid, although
  // symbol_table_initialized_ being valid doesn’t imply that symbol_table_ is
  // set. symbol_table_initialized_ will be valid without symbol_table_ being
  // set in modules that have no symbol table.
  mutable InitializationState symbol_table_initialized_;

  DISALLOW_COPY_AND_ASSIGN(MachOImageReader);
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_MAC_MACH_O_IMAGE_READER_H_
