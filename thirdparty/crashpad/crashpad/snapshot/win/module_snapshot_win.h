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

#ifndef CRASHPAD_SNAPSHOT_WIN_MODULE_SNAPSHOT_WIN_H_
#define CRASHPAD_SNAPSHOT_WIN_MODULE_SNAPSHOT_WIN_H_

#include <windows.h>
#include <stdint.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "base/macros.h"
#include "snapshot/crashpad_info_client_options.h"
#include "snapshot/module_snapshot.h"
#include "snapshot/win/process_reader_win.h"
#include "util/misc/initialization_state.h"
#include "util/misc/initialization_state_dcheck.h"
#include "util/win/process_info.h"

namespace crashpad {

class PEImageReader;
struct UUID;

namespace internal {

//! \brief A ModuleSnapshot of a code module (binary image) loaded into a
//!     running (or crashed) process on a Windows system.
class ModuleSnapshotWin final : public ModuleSnapshot {
 public:
  ModuleSnapshotWin();
  ~ModuleSnapshotWin() override;

  //! \brief Initializes the object.
  //!
  //! \param[in] process_reader A ProcessReaderWin for the process containing
  //!     the module.
  //! \param[in] process_reader_module The module within the ProcessReaderWin
  //!     for which the snapshot should be created.
  //!
  //! \return `true` if the snapshot could be created, `false` otherwise with
  //!     an appropriate message logged.
  bool Initialize(ProcessReaderWin* process_reader,
                  const ProcessInfo::Module& process_reader_module);

  //! \brief Returns options from the module's CrashpadInfo structure.
  //!
  //! \param[out] options Options set in the module's CrashpadInfo structure.
  void GetCrashpadOptions(CrashpadInfoClientOptions* options);

  //! \brief Returns the PEImageReader used to read this module. Only valid
  //!     after Initialize() is called.
  const PEImageReader& pe_image_reader() const { return *pe_image_reader_; }

  // ModuleSnapshot:

  std::string Name() const override;
  uint64_t Address() const override;
  uint64_t Size() const override;
  time_t Timestamp() const override;
  void FileVersion(uint16_t* version_0,
                   uint16_t* version_1,
                   uint16_t* version_2,
                   uint16_t* version_3) const override;
  void SourceVersion(uint16_t* version_0,
                     uint16_t* version_1,
                     uint16_t* version_2,
                     uint16_t* version_3) const override;
  ModuleType GetModuleType() const override;
  void UUIDAndAge(crashpad::UUID* uuid, uint32_t* age) const override;
  std::string DebugFileName() const override;
  std::vector<std::string> AnnotationsVector() const override;
  std::map<std::string, std::string> AnnotationsSimpleMap() const override;
  std::vector<AnnotationSnapshot> AnnotationObjects() const override;
  std::set<CheckedRange<uint64_t>> ExtraMemoryRanges() const override;
  std::vector<const UserMinidumpStream*> CustomMinidumpStreams() const override;

 private:
  template <class Traits>
  void GetCrashpadOptionsInternal(CrashpadInfoClientOptions* options);

  template <class Traits>
  void GetCrashpadExtraMemoryRanges(
      std::set<CheckedRange<uint64_t>>* ranges) const;

  template <class Traits>
  void GetCrashpadUserMinidumpStreams(
      std::vector<std::unique_ptr<const UserMinidumpStream>>* streams) const;

  // Initializes vs_fixed_file_info_ if it has not yet been initialized, and
  // returns a pointer to it. Returns nullptr on failure, with a message logged
  // on the first call.
  const VS_FIXEDFILEINFO* VSFixedFileInfo() const;

  std::wstring name_;
  std::string pdb_name_;
  UUID uuid_;
  std::unique_ptr<PEImageReader> pe_image_reader_;
  ProcessReaderWin* process_reader_;  // weak
  time_t timestamp_;
  uint32_t age_;
  // Too const-y: https://crashpad.chromium.org/bug/9.
  mutable std::vector<std::unique_ptr<const UserMinidumpStream>> streams_;
  InitializationStateDcheck initialized_;

  // VSFixedFileInfo() is logically const, but updates these members on the
  // call. See https://crashpad.chromium.org/bug/9.
  mutable VS_FIXEDFILEINFO vs_fixed_file_info_;
  mutable InitializationState initialized_vs_fixed_file_info_;

  DISALLOW_COPY_AND_ASSIGN(ModuleSnapshotWin);
};

}  // namespace internal
}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_WIN_MODULE_SNAPSHOT_WIN_H_
