// Copyright 2016 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_MINIDUMP_MINIDUMP_UNLOADED_MODULE_WRITER_H_
#define CRASHPAD_MINIDUMP_MINIDUMP_UNLOADED_MODULE_WRITER_H_

#include <windows.h>
#include <dbghelp.h>
#include <stdint.h>

#include <memory>
#include <string>
#include <vector>

#include "base/macros.h"
#include "minidump/minidump_stream_writer.h"
#include "minidump/minidump_string_writer.h"
#include "minidump/minidump_writable.h"
#include "snapshot/unloaded_module_snapshot.h"

namespace crashpad {

//! \brief The writer for a MINIDUMP_UNLOADED_MODULE object in a minidump file.
//!
//! Because MINIDUMP_UNLOADED_MODULE objects only appear as elements of
//! MINIDUMP_UNLOADED_MODULE_LIST objects, this class does not write any data on
//! its own. It makes its MINIDUMP_UNLOADED_MODULE data available to its
//! MinidumpUnloadedModuleListWriter parent, which writes it as part of a
//! MINIDUMP_UNLOADED_MODULE_LIST.
class MinidumpUnloadedModuleWriter final : public internal::MinidumpWritable {
 public:
  MinidumpUnloadedModuleWriter();
  ~MinidumpUnloadedModuleWriter() override;

  //! \brief Initializes the MINIDUMP_UNLOADED_MODULE based on \a
  //! unloaded_module_snapshot.
  //!
  //! \param[in] unloaded_module_snapshot The unloaded module snapshot to use as
  //!     source data.
  //!
  //! \note Valid in #kStateMutable. No mutator methods may be called before
  //!     this method, and it is not normally necessary to call any mutator
  //!     methods after this method.
  void InitializeFromSnapshot(
      const UnloadedModuleSnapshot& unloaded_module_snapshot);

  //! \brief Returns a MINIDUMP_UNLOADED_MODULE referencing this object’s data.
  //!
  //! This method is expected to be called by a MinidumpUnloadedModuleListWriter
  //! in order to obtain a MINIDUMP_UNLOADED_MODULE to include in its list.
  //!
  //! \note Valid in #kStateWritable.
  const MINIDUMP_UNLOADED_MODULE* MinidumpUnloadedModule() const;

  //! \brief Arranges for MINIDUMP_UNLOADED_MODULE::ModuleNameRva to point to a
  //!     MINIDUMP_STRING containing \a name.
  //!
  //! \note Valid in #kStateMutable.
  void SetName(const std::string& name);

  //! \brief Sets MINIDUMP_UNLOADED_MODULE::BaseOfImage.
  void SetImageBaseAddress(uint64_t image_base_address) {
    unloaded_module_.BaseOfImage = image_base_address;
  }

  //! \brief Sets MINIDUMP_UNLOADED_MODULE::SizeOfImage.
  void SetImageSize(uint32_t image_size) {
    unloaded_module_.SizeOfImage = image_size;
  }

  //! \brief Sets MINIDUMP_UNLOADED_MODULE::CheckSum.
  void SetChecksum(uint32_t checksum) { unloaded_module_.CheckSum = checksum; }

  //! \brief Sets MINIDUMP_UNLOADED_MODULE::TimeDateStamp.
  //!
  //! \note Valid in #kStateMutable.
  void SetTimestamp(time_t timestamp);

 protected:
  // MinidumpWritable:
  bool Freeze() override;
  size_t SizeOfObject() override;
  std::vector<MinidumpWritable*> Children() override;
  bool WriteObject(FileWriterInterface* file_writer) override;

 private:
  MINIDUMP_UNLOADED_MODULE unloaded_module_;
  std::unique_ptr<internal::MinidumpUTF16StringWriter> name_;

  DISALLOW_COPY_AND_ASSIGN(MinidumpUnloadedModuleWriter);
};

//! \brief The writer for a MINIDUMP_UNLOADED_MODULE_LIST stream in a minidump
//!     file, containing a list of MINIDUMP_UNLOADED_MODULE objects.
class MinidumpUnloadedModuleListWriter final
    : public internal::MinidumpStreamWriter {
 public:
  MinidumpUnloadedModuleListWriter();
  ~MinidumpUnloadedModuleListWriter() override;

  //! \brief Adds an initialized MINIDUMP_UNLOADED_MODULE for each unloaded
  //!     module in \a unloaded_module_snapshots to the
  //!     MINIDUMP_UNLOADED_MODULE_LIST.
  //!
  //! \param[in] unloaded_module_snapshots The unloaded module snapshots to use
  //!     as source data.
  //!
  //! \note Valid in #kStateMutable. AddUnloadedModule() may not be called
  //!     before this this method, and it is not normally necessary to call
  //!     AddUnloadedModule() after this method.
  void InitializeFromSnapshot(
      const std::vector<UnloadedModuleSnapshot>& unloaded_module_snapshots);

  //! \brief Adds a MinidumpUnloadedModuleWriter to the
  //!     MINIDUMP_UNLOADED_MODULE_LIST.
  //!
  //! This object takes ownership of \a unloaded_module and becomes its parent
  //! in the overall tree of internal::MinidumpWritable objects.
  //!
  //! \note Valid in #kStateMutable.
  void AddUnloadedModule(
      std::unique_ptr<MinidumpUnloadedModuleWriter> unloaded_module);

 protected:
  // MinidumpWritable:
  bool Freeze() override;
  size_t SizeOfObject() override;
  std::vector<MinidumpWritable*> Children() override;
  bool WriteObject(FileWriterInterface* file_writer) override;

  // MinidumpStreamWriter:
  MinidumpStreamType StreamType() const override;

 private:
  std::vector<std::unique_ptr<MinidumpUnloadedModuleWriter>> unloaded_modules_;
  MINIDUMP_UNLOADED_MODULE_LIST unloaded_module_list_base_;

  DISALLOW_COPY_AND_ASSIGN(MinidumpUnloadedModuleListWriter);
};

}  // namespace crashpad

#endif  // CRASHPAD_MINIDUMP_MINIDUMP_UNLOADED_MODULE_WRITER_H_
