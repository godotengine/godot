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

#ifndef CRASHPAD_SNAPSHOT_WIN_PE_IMAGE_RESOURCE_READER_H_
#define CRASHPAD_SNAPSHOT_WIN_PE_IMAGE_RESOURCE_READER_H_

#include <windows.h>
#include <stdint.h>

#include <vector>

#include "base/macros.h"
#include "snapshot/win/process_subrange_reader.h"
#include "util/misc/initialization_state_dcheck.h"
#include "util/win/address_types.h"

namespace crashpad {

//! \brief A reader for resources stored in PE images mapped into another
//!     process.
//!
//! \sa PEImageReader
class PEImageResourceReader {
 public:
  PEImageResourceReader();
  ~PEImageResourceReader();

  //! \brief Initializes the resource reader.
  //!
  //! \param[in] module_subrange_reader The reader for the module.
  //! \param[in] resources_directory_entry The module’s `IMAGE_DATA_DIRECTORY`
  //!     for its resources area. This is taken from the module’s
  //!     `IMAGE_OPTIONAL_HEADER::DataDirectory` at index
  //!     `IMAGE_DIRECTORY_ENTRY_RESOURCE`.
  //!
  //! \return `true` on success, `false` on failure with a message logged.
  bool Initialize(const ProcessSubrangeReader& module_subrange_reader,
                  const IMAGE_DATA_DIRECTORY& resources_directory_entry);

  //! \brief Locates a resource in a module by its ID.
  //!
  //! This method is similar to `FindResourceEx()`, but it operates on modules
  //! loaded in a remote process’ address space. It is not necessary to
  //! `LoadLibrary()` a module into a process in order to use this method.
  //!
  //! No support is provided at present for locating resources by \a type or \a
  //! name using strings as opposed to integer identifiers.
  //!
  //! Languages are scanned in the order determined by
  //! GetEntryFromResourceDirectoryByLanguage().
  //!
  //! \param[in] type The integer identifier of the resource type, as in the
  //!     `lpType` parameter of `FindResourceEx()`.
  //! \param[in] name The integer identifier of the resource, as in the `lpName`
  //!     parameter of `FindResourceEx()`.
  //! \param[in] language The language of the resource, as in the `wLanguage`
  //!     parameter of `FindResourceEx()`.
  //! \param[out] address The address, in the remote process’ address space, of
  //!     the resource data.
  //! \param[out] size The size of the resource data.
  //! \param[out] code_page The code page used to encode textual resource data.
  //!     This parameter is optional.
  //!
  //! \return `true` on success, with the out parameters set appropriately.
  //!     `false` if the resource was not found, without logging any messages.
  //!     `false` on failure, with a message logged.
  bool FindResourceByID(uint16_t type,
                        uint16_t name,
                        uint16_t language,
                        WinVMAddress* address,
                        WinVMSize* size,
                        uint32_t* code_page) const;

 private:
  //! \brief Locates a resource directory entry within a resource directory by
  //!     integer ID.
  //!
  //! \param[in] resource_directory_offset The offset, in the module’s resources
  //!     area, of the resource directory to search.
  //! \param[in] id The integer identifier of the resource to search for.
  //! \param[in] want_subdirectory `true` if the resource directory entry is
  //!     expected to be a resource directory itself, `false` otherwise.
  //!
  //! \return The offset, in the module’s resources area, of the entry that was
  //!     found. On failure, `0`. `0` is technically a valid offset, but it
  //!     corresponds to the root resource directory, which should never be the
  //!     offset of another resource directory entry. If \a id was not found,
  //!     `0` will be returned without logging anything. For other failures, a
  //!     message will be logged.
  uint32_t GetEntryFromResourceDirectoryByID(uint32_t resource_directory_offset,
                                             uint16_t id,
                                             bool want_subdirectory) const;

  //! \brief Locates a resource directory entry within a resource directory by
  //!     language.
  //!
  //! This method is similar to GetEntryFromResourceDirectoryByID() with \a
  //! want_subdirectory set to `false`. Attempts are made to locate the resource
  //! by using these languages:
  //! <ul>
  //!     <li>If \a language is `LANG_NEUTRAL`:</li>
  //!     <ul>
  //!         <li>Unless `SUBLANG_SYS_DEFAULT` is specified, the language of the
  //!             thread’s locale, with its normal sublanguage and with
  //!             `SUBLANG_NEUTRAL`.</li>
  //!         <li>Unless `SUBLANG_SYS_DEFAULT` is specified, the language of the
  //!             user’s default locale, with its normal sublanguage and with
  //!             `SUBLANG_NEUTRAL`.</li>
  //!         <li>Unless `SUBLANG_DEFAULT` is specified, the language of the
  //!             system’s default locale, with its normal sublanguage and with
  //!             `SUBLANG_NEUTRAL`.</li>
  //!     </ul>
  //!     <li>If \a language is not `LANG_NEUTRAL`:</li>
  //!     <ul>
  //!         <li>\a language</li>
  //!         <li>\a language, with `SUBLANG_NEUTRAL`</li>
  //!     </ul>
  //!     <li>`LANG_NEUTRAL` with `SUBLANG_NEUTRAL`</li>
  //!     <li>`LANG_ENGLISH` with `SUBLANG_DEFAULT`</li>
  //!     <li>If none of the above match, the first language found</li>
  //! </ul>
  //!
  //! If only a specific language is desired without any fallbacks, call
  //! GetEntryFromResourceDirectoryByID() with the language directory’s offset
  //! instead, passing the desired language in the \a id parameter, and `false`
  //! for \a want_subdirectory.
  //!
  //! \param[in] language_directory_offset The offset, in the module’s resources
  //!     area, of the resource directory to search.
  //! \param[in] language The language of the resource to search for.
  //!
  //! \return The return value is as in GetEntryFromResourceDirectoryByID().
  uint32_t GetEntryFromResourceDirectoryByLanguage(
      uint32_t language_directory_offset,
      uint16_t language) const;

  //! \brief Reads a resource directory.
  //!
  //! \param[in] resource_directory_offset The offset, in the module’s resources
  //!     area, of the resource directory to read.
  //! \param[out] resource_directory The `IMAGE_RESOURCE_DIRECTORY` structure.
  //!     This parameter is optional.
  //! \param[out] named_entries A vector of \a
  //!     resource_directory->NumberOfNamedEntries
  //!     `IMAGE_RESOURCE_DIRECTORY_ENTRY` items that follow the resource
  //!     directory. This parameter is optional.
  //! \param[out] id_entries A vector of \a
  //!     resource_directory->NumberOfIdEntries `IMAGE_RESOURCE_DIRECTORY_ENTRY`
  //!     items that follow the named entries. This parameter is optional.
  //!
  //! \return `true` on success, with the out parameters set appropriately.
  //!     `false` on failure with a message logged.
  bool ReadResourceDirectory(
      uint32_t resource_directory_offset,
      IMAGE_RESOURCE_DIRECTORY* resource_directory,
      std::vector<IMAGE_RESOURCE_DIRECTORY_ENTRY>* named_entries,
      std::vector<IMAGE_RESOURCE_DIRECTORY_ENTRY>* id_entries) const;

  ProcessSubrangeReader resources_subrange_reader_;
  WinVMAddress module_base_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(PEImageResourceReader);
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_WIN_PE_IMAGE_RESOURCE_READER_H_
