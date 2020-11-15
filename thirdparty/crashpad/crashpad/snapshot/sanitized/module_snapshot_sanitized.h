// Copyright 2018 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_SNAPSHOT_SANITIZED_MODULE_SNAPSHOT_SANITIZED_H_
#define CRASHPAD_SNAPSHOT_SANITIZED_MODULE_SNAPSHOT_SANITIZED_H_

#include <string>
#include <vector>

#include "base/macros.h"
#include "snapshot/module_snapshot.h"

namespace crashpad {
namespace internal {

//! \brief A ModuleSnapshot which wraps and filters sensitive information from
//!     another ModuleSnapshot.
class ModuleSnapshotSanitized final : public ModuleSnapshot {
 public:
  //! \brief Constructs this object.
  //!
  //! \param[in] snapshot The ModuleSnapshot to sanitize.
  //! \param[in] annotations_whitelist A list of annotation names to allow to be
  //!     returned by AnnotationsSimpleMap() or AnnotationObjects(). If
  //!     `nullptr`, all annotations will be returned.
  ModuleSnapshotSanitized(
      const ModuleSnapshot* snapshot,
      const std::vector<std::string>* annotations_whitelist);
  ~ModuleSnapshotSanitized() override;

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
  const ModuleSnapshot* snapshot_;
  const std::vector<std::string>* annotations_whitelist_;

  DISALLOW_COPY_AND_ASSIGN(ModuleSnapshotSanitized);
};

}  // namespace internal
}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_SANITIZED_MODULE_SNAPSHOT_SANITIZED_H_
