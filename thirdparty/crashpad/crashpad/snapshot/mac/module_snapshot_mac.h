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

#ifndef CRASHPAD_SNAPSHOT_MAC_MODULE_SNAPSHOT_MAC_H_
#define CRASHPAD_SNAPSHOT_MAC_MODULE_SNAPSHOT_MAC_H_

#include <stdint.h>
#include <sys/types.h>

#include <map>
#include <string>
#include <vector>

#include "base/macros.h"
#include "client/crashpad_info.h"
#include "snapshot/crashpad_info_client_options.h"
#include "snapshot/mac/process_reader_mac.h"
#include "snapshot/module_snapshot.h"
#include "util/misc/initialization_state_dcheck.h"

namespace crashpad {

class MachOImageReader;
struct UUID;

namespace internal {

//! \brief A ModuleSnapshot of a code module (binary image) loaded into a
//!     running (or crashed) process on a macOS system.
class ModuleSnapshotMac final : public ModuleSnapshot {
 public:
  ModuleSnapshotMac();
  ~ModuleSnapshotMac() override;

  //! \brief Initializes the object.
  //!
  //! \param[in] process_reader A ProcessReaderMac for the task containing the
  //!     module.
  //! \param[in] process_reader_module The module within the ProcessReaderMac
  //!     for which the snapshot should be created.
  //!
  //! \return `true` if the snapshot could be created, `false` otherwise with
  //!     an appropriate message logged.
  bool Initialize(ProcessReaderMac* process_reader,
                  const ProcessReaderMac::Module& process_reader_module);

  //! \brief Returns options from the module’s CrashpadInfo structure.
  //!
  //! \param[out] options Options set in the module’s CrashpadInfo structure.
  void GetCrashpadOptions(CrashpadInfoClientOptions* options);

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
  std::string name_;
  time_t timestamp_;
  const MachOImageReader* mach_o_image_reader_;  // weak
  ProcessReaderMac* process_reader_;  // weak
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(ModuleSnapshotMac);
};

}  // namespace internal
}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_MAC_MODULE_SNAPSHOT_MAC_H_
