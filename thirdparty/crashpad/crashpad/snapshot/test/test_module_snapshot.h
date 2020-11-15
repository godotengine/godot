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

#ifndef CRASHPAD_SNAPSHOT_TEST_TEST_MODULE_SNAPSHOT_H_
#define CRASHPAD_SNAPSHOT_TEST_TEST_MODULE_SNAPSHOT_H_

#include <stdint.h>
#include <sys/types.h>

#include <map>
#include <string>
#include <vector>

#include "base/macros.h"
#include "snapshot/module_snapshot.h"

namespace crashpad {
namespace test {

//! \brief A test ModuleSnapshot that can carry arbitrary data for testing
//!     purposes.
class TestModuleSnapshot final : public ModuleSnapshot {
 public:
  TestModuleSnapshot();
  ~TestModuleSnapshot() override;

  void SetName(const std::string& name) { name_ = name; }
  void SetAddressAndSize(uint64_t address, uint64_t size) {
    address_ = address;
    size_ = size;
  }
  void SetTimestamp(time_t timestamp) { timestamp_ = timestamp; }
  void SetFileVersion(uint16_t file_version_0,
                      uint16_t file_version_1,
                      uint16_t file_version_2,
                      uint16_t file_version_3) {
    file_version_[0] = file_version_0;
    file_version_[1] = file_version_1;
    file_version_[2] = file_version_2;
    file_version_[3] = file_version_3;
  }
  void SetSourceVersion(uint16_t source_version_0,
                        uint16_t source_version_1,
                        uint16_t source_version_2,
                        uint16_t source_version_3) {
    source_version_[0] = source_version_0;
    source_version_[1] = source_version_1;
    source_version_[2] = source_version_2;
    source_version_[3] = source_version_3;
  }
  void SetModuleType(ModuleType module_type) { module_type_ = module_type; }
  void SetUUIDAndAge(const crashpad::UUID& uuid, uint32_t age) {
    uuid_ = uuid;
    age_ = age;
  }
  void SetDebugFileName(const std::string& debug_file_name) {
    debug_file_name_ = debug_file_name;
  }
  void SetAnnotationsVector(
      const std::vector<std::string>& annotations_vector) {
    annotations_vector_ = annotations_vector;
  }
  void SetAnnotationsSimpleMap(
      const std::map<std::string, std::string>& annotations_simple_map) {
    annotations_simple_map_ = annotations_simple_map;
  }
  void SetAnnotationObjects(
      const std::vector<AnnotationSnapshot>& annotations) {
    annotation_objects_ = annotations;
  }
  void SetExtraMemoryRanges(
      const std::set<CheckedRange<uint64_t>>& extra_memory_ranges) {
    extra_memory_ranges_ = extra_memory_ranges;
  }

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
  uint64_t address_;
  uint64_t size_;
  time_t timestamp_;
  uint16_t file_version_[4];
  uint16_t source_version_[4];
  ModuleType module_type_;
  uint32_t age_;
  crashpad::UUID uuid_;
  std::string debug_file_name_;
  std::vector<std::string> annotations_vector_;
  std::map<std::string, std::string> annotations_simple_map_;
  std::vector<AnnotationSnapshot> annotation_objects_;
  std::set<CheckedRange<uint64_t>> extra_memory_ranges_;

  DISALLOW_COPY_AND_ASSIGN(TestModuleSnapshot);
};

}  // namespace test
}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_TEST_TEST_MODULE_SNAPSHOT_H_
