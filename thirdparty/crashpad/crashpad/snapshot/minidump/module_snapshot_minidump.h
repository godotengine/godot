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

#ifndef CRASHPAD_SNAPSHOT_MINIDUMP_MODULE_SNAPSHOT_MINIDUMP_H_
#define CRASHPAD_SNAPSHOT_MINIDUMP_MODULE_SNAPSHOT_MINIDUMP_H_

#include <windows.h>
#include <dbghelp.h>
#include <stdint.h>
#include <sys/types.h>

#include <map>
#include <string>
#include <vector>

#include "base/macros.h"
#include "snapshot/annotation_snapshot.h"
#include "snapshot/module_snapshot.h"
#include "util/file/file_reader.h"
#include "util/misc/initialization_state_dcheck.h"

namespace crashpad {
namespace internal {

//! \brief A ModuleSnapshot based on a module in a minidump file.
class ModuleSnapshotMinidump final : public ModuleSnapshot {
 public:
  ModuleSnapshotMinidump();
  ~ModuleSnapshotMinidump() override;

  //! \brief Initializes the object.
  //!
  //! \param[in] file_reader A file reader corresponding to a minidump file.
  //!     The file reader must support seeking.
  //! \param[in] minidump_module_rva The file offset in \a file_reader at which
  //!     the module’s MINIDUMP_MODULE structure is located.
  //! \param[in] minidump_crashpad_module_info_location The location in \a
  //!     file_reader at which the module’s corresponding
  //!     MinidumpModuleCrashpadInfo structure is located. If no such
  //!     corresponding structure is available for a module, this may be
  //!     `nullptr`.
  //!
  //! \return `true` if the snapshot could be created, `false` otherwise with
  //!     an appropriate message logged.
  bool Initialize(FileReaderInterface* file_reader,
                  RVA minidump_module_rva,
                  const MINIDUMP_LOCATION_DESCRIPTOR*
                      minidump_crashpad_module_info_location);

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
  // Initializes data carried in a MinidumpModuleCrashpadInfo structure on
  // behalf of Initialize().
  bool InitializeModuleCrashpadInfo(FileReaderInterface* file_reader,
                                    const MINIDUMP_LOCATION_DESCRIPTOR*
                                        minidump_module_crashpad_info_location);

  MINIDUMP_MODULE minidump_module_;
  std::vector<std::string> annotations_vector_;
  std::map<std::string, std::string> annotations_simple_map_;
  std::vector<AnnotationSnapshot> annotation_objects_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(ModuleSnapshotMinidump);
};

}  // namespace internal
}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_MINIDUMP_MODULE_SNAPSHOT_MINIDUMP_H_
