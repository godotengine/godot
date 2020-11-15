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

#include "snapshot/test/test_module_snapshot.h"

namespace crashpad {
namespace test {

TestModuleSnapshot::TestModuleSnapshot()
    : name_(),
      address_(0),
      size_(0),
      timestamp_(0),
      file_version_(),
      source_version_(),
      module_type_(kModuleTypeUnknown),
      age_(0),
      uuid_(),
      debug_file_name_(),
      annotations_vector_(),
      annotations_simple_map_(),
      extra_memory_ranges_() {
}

TestModuleSnapshot::~TestModuleSnapshot() {
}

std::string TestModuleSnapshot::Name() const {
  return name_;
}

uint64_t TestModuleSnapshot::Address() const {
  return address_;
}

uint64_t TestModuleSnapshot::Size() const {
  return size_;
}

time_t TestModuleSnapshot::Timestamp() const {
  return timestamp_;
}

void TestModuleSnapshot::FileVersion(uint16_t* version_0,
                                     uint16_t* version_1,
                                     uint16_t* version_2,
                                     uint16_t* version_3) const {
  *version_0 = file_version_[0];
  *version_1 = file_version_[1];
  *version_2 = file_version_[2];
  *version_3 = file_version_[3];
}

void TestModuleSnapshot::SourceVersion(uint16_t* version_0,
                                       uint16_t* version_1,
                                       uint16_t* version_2,
                                       uint16_t* version_3) const {
  *version_0 = source_version_[0];
  *version_1 = source_version_[1];
  *version_2 = source_version_[2];
  *version_3 = source_version_[3];
}

ModuleSnapshot::ModuleType TestModuleSnapshot::GetModuleType() const {
  return module_type_;
}

void TestModuleSnapshot::UUIDAndAge(crashpad::UUID* uuid, uint32_t* age) const {
  *uuid = uuid_;
  *age = age_;
}

std::string TestModuleSnapshot::DebugFileName() const {
  return debug_file_name_;
}

std::vector<std::string> TestModuleSnapshot::AnnotationsVector() const {
  return annotations_vector_;
}

std::map<std::string, std::string> TestModuleSnapshot::AnnotationsSimpleMap()
    const {
  return annotations_simple_map_;
}

std::vector<AnnotationSnapshot> TestModuleSnapshot::AnnotationObjects() const {
  return annotation_objects_;
}

std::set<CheckedRange<uint64_t>> TestModuleSnapshot::ExtraMemoryRanges() const {
  return extra_memory_ranges_;
}

std::vector<const UserMinidumpStream*>
TestModuleSnapshot::CustomMinidumpStreams() const {
  return std::vector<const UserMinidumpStream*>();
}

}  // namespace test
}  // namespace crashpad
