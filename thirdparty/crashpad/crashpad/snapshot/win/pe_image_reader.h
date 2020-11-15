// Copyright 2015 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_SNAPSHOT_WIN_PE_IMAGE_READER_H_
#define CRASHPAD_SNAPSHOT_WIN_PE_IMAGE_READER_H_

#include <windows.h>
#include <stdint.h>
#include <sys/types.h>

#include <string>

#include "base/macros.h"
#include "snapshot/win/process_subrange_reader.h"
#include "util/misc/initialization_state_dcheck.h"
#include "util/misc/uuid.h"
#include "util/win/address_types.h"
#include "util/win/process_structs.h"

namespace crashpad {

class ProcessReaderWin;

namespace process_types {

template <class Traits>
struct CrashpadInfo {
  uint32_t signature;
  uint32_t size;
  uint32_t version;
  uint32_t indirectly_referenced_memory_cap;
  uint32_t padding_0;
  uint8_t crashpad_handler_behavior;  // TriState.
  uint8_t system_crash_reporter_forwarding;  // TriState.
  uint8_t gather_indirectly_referenced_memory;  // TriState.
  uint8_t padding_1;
  typename Traits::Pointer extra_address_ranges;
  typename Traits::Pointer simple_annotations;
  typename Traits::Pointer user_data_minidump_stream_head;
  typename Traits::Pointer annotations_list;
};

}  // namespace process_types

//! \brief A reader for PE images mapped into another process.
//!
//! This class is capable of reading both 32-bit and 64-bit images based on the
//! bitness of the remote process.
//!
//! \sa PEImageAnnotationsReader
//! \sa PEImageResourceReader
class PEImageReader {
 public:
  PEImageReader();
  ~PEImageReader();

  //! \brief Initializes the reader.
  //!
  //! This method must be called only once on an object. This method must be
  //! called successfully before any other method in this class may be called.
  //!
  //! \param[in] process_reader The reader for the remote process.
  //! \param[in] address The address, in the remote process' address space,
  //!     where the `IMAGE_DOS_HEADER` is located.
  //! \param[in] size The size of the image.
  //! \param[in] module_name The module's name, a string to be used in logged
  //!     messages. This string is for diagnostic purposes.
  //!
  //! \return `true` if the image was read successfully, `false` otherwise, with
  //!     an appropriate message logged.
  bool Initialize(ProcessReaderWin* process_reader,
                  WinVMAddress address,
                  WinVMSize size,
                  const std::string& module_name);

  //! \brief Returns the image's load address.
  //!
  //! This is the value passed as \a address to Initialize().
  WinVMAddress Address() const { return module_subrange_reader_.Base(); }

  //! \brief Returns the image's size.
  //!
  //! This is the value passed as \a size to Initialize().
  WinVMSize Size() const { return module_subrange_reader_.Size(); }

  //! \brief Obtains the module's CrashpadInfo structure.
  //!
  //! \return `true` on success, `false` on failure. If the module does not have
  //!     a `CPADinfo` section, this will return `false` without logging any
  //!     messages. Other failures will result in messages being logged.
  template <class Traits>
  bool GetCrashpadInfo(
      process_types::CrashpadInfo<Traits>* crashpad_info) const;

  //! \brief Obtains information from the module's debug directory, if any.
  //!
  //! \param[out] uuid The unique identifier of the executable/PDB.
  //! \param[out] age The age field for the pdb (the number of times it's been
  //!     relinked).
  //! \param[out] pdbname Name of the pdb file.
  //!
  //! \return `true` on success, with the parameters set appropriately. `false`
  //!     on failure. This method may return `false` without logging anything in
  //!     the case of a module that does not contain relevant debugging
  //!     information but is otherwise properly structured.
  bool DebugDirectoryInformation(UUID* uuid,
                                 DWORD* age,
                                 std::string* pdbname) const;

  //! \brief Obtains the module’s `VS_FIXEDFILEINFO`, containing its version and
  //!     type information.
  //!
  //! The data obtained from this method should be equivalent to what could be
  //! obtained by calling GetModuleVersionAndType(). Avoiding that function
  //! ensures that the data in the module loaded into the remote process will be
  //! used as-is, without the risks associated with loading the module into the
  //! reading process.
  //!
  //! \param[out] vs_fixed_file_info The VS_FIXEDFILEINFO on success.
  //!     VS_FIXEDFILEINFO::dwFileFlags will have been masked with
  //!     VS_FIXEDFILEINFO::dwFileFlagsMask already.
  //!
  //! \return `true` on success. `false` if the module does not contain this
  //!     information, without logging any messages. `false` on failure, with
  //!     a message logged.
  bool VSFixedFileInfo(VS_FIXEDFILEINFO* vs_fixed_file_info) const;

 private:
  //! \brief Reads the `IMAGE_NT_HEADERS` from the beginning of the image.
  //!
  //! \param[out] nt_headers The contents of the templated NtHeadersType
  //!     structure read from the remote process.
  //! \param[out] nt_headers_address The address of the templated NtHeadersType
  //!     structure in the remote process’ address space. If this information is
  //!     not needed, this parameter may be `nullptr`.
  //!
  //! \return `true` on success, with \a nt_headers and optionally \a
  //!     nt_headers_address set appropriately. `false` on failure, with a
  //!     message logged.
  template <class NtHeadersType>
  bool ReadNtHeaders(NtHeadersType* nt_headers,
                     WinVMAddress* nt_headers_address) const;

  //! \brief Finds a given section by name in the image.
  template <class NtHeadersType>
  bool GetSectionByName(const std::string& name,
                        IMAGE_SECTION_HEADER* section) const;

  //! \brief Finds the `IMAGE_DATA_DIRECTORY` in
  //!     `IMAGE_OPTIONAL_HEADER::DataDirectory` at the specified \a index.
  //!
  //! \param[in] index An `IMAGE_DIRECTORY_ENTRY_*` constant specifying the
  //!     data to be returned.
  //! \param[out] entry The `IMAGE_DATA_DIRECTORY` found within the module.
  //!
  //! \return `true` on success, with \a entry set appropriately. `false` if the
  //!     module does not contain the specified information, without logging a
  //!     message. `false` on failure, with a message logged.
  bool ImageDataDirectoryEntry(size_t index, IMAGE_DATA_DIRECTORY* entry) const;

  //! \brief A templatized helper for ImageDataDirectoryEntry() to account for
  //!     differences in \a NtHeadersType.
  template <class NtHeadersType>
  bool ImageDataDirectoryEntryT(size_t index,
                                IMAGE_DATA_DIRECTORY* entry) const;

  ProcessSubrangeReader module_subrange_reader_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(PEImageReader);
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_WIN_PE_IMAGE_READER_H_
