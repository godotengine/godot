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

#include "snapshot/sanitized/module_snapshot_sanitized.h"

namespace crashpad {
namespace internal {

namespace {

bool KeyIsInWhitelist(const std::string& name,
                      const std::vector<std::string>& whitelist) {
  for (const auto& key : whitelist) {
    if (name == key) {
      return true;
    }
  }
  return false;
}

}  // namespace

ModuleSnapshotSanitized::ModuleSnapshotSanitized(
    const ModuleSnapshot* snapshot,
    const std::vector<std::string>* annotations_whitelist)
    : snapshot_(snapshot), annotations_whitelist_(annotations_whitelist) {}

ModuleSnapshotSanitized::~ModuleSnapshotSanitized() = default;

std::string ModuleSnapshotSanitized::Name() const {
  return snapshot_->Name();
}

uint64_t ModuleSnapshotSanitized::Address() const {
  return snapshot_->Address();
}

uint64_t ModuleSnapshotSanitized::Size() const {
  return snapshot_->Size();
}

time_t ModuleSnapshotSanitized::Timestamp() const {
  return snapshot_->Timestamp();
}

void ModuleSnapshotSanitized::FileVersion(uint16_t* version_0,
                                          uint16_t* version_1,
                                          uint16_t* version_2,
                                          uint16_t* version_3) const {
  snapshot_->FileVersion(version_0, version_1, version_2, version_3);
}

void ModuleSnapshotSanitized::SourceVersion(uint16_t* version_0,
                                            uint16_t* version_1,
                                            uint16_t* version_2,
                                            uint16_t* version_3) const {
  snapshot_->SourceVersion(version_0, version_1, version_2, version_3);
}

ModuleSnapshot::ModuleType ModuleSnapshotSanitized::GetModuleType() const {
  return snapshot_->GetModuleType();
}

void ModuleSnapshotSanitized::UUIDAndAge(crashpad::UUID* uuid,
                                         uint32_t* age) const {
  snapshot_->UUIDAndAge(uuid, age);
}

std::string ModuleSnapshotSanitized::DebugFileName() const {
  return snapshot_->DebugFileName();
}

std::vector<std::string> ModuleSnapshotSanitized::AnnotationsVector() const {
  // TODO(jperaza): If/when AnnotationsVector() begins to be used, determine
  // whether and how the content should be sanitized.
  DCHECK(snapshot_->AnnotationsVector().empty());
  return std::vector<std::string>();
}

std::map<std::string, std::string>
ModuleSnapshotSanitized::AnnotationsSimpleMap() const {
  std::map<std::string, std::string> annotations =
      snapshot_->AnnotationsSimpleMap();
  if (annotations_whitelist_) {
    for (auto kv = annotations.begin(); kv != annotations.end(); ++kv) {
      if (!KeyIsInWhitelist(kv->first, *annotations_whitelist_)) {
        annotations.erase(kv);
      }
    }
  }
  return annotations;
}

std::vector<AnnotationSnapshot> ModuleSnapshotSanitized::AnnotationObjects()
    const {
  std::vector<AnnotationSnapshot> annotations = snapshot_->AnnotationObjects();
  if (annotations_whitelist_) {
    std::vector<AnnotationSnapshot> whitelisted;
    for (const auto& anno : annotations) {
      if (KeyIsInWhitelist(anno.name, *annotations_whitelist_)) {
        whitelisted.push_back(anno);
      }
    }
    annotations.swap(whitelisted);
  }
  return annotations;
}

std::set<CheckedRange<uint64_t>> ModuleSnapshotSanitized::ExtraMemoryRanges()
    const {
  DCHECK(snapshot_->ExtraMemoryRanges().empty());
  return std::set<CheckedRange<uint64_t>>();
}

std::vector<const UserMinidumpStream*>
ModuleSnapshotSanitized::CustomMinidumpStreams() const {
  return snapshot_->CustomMinidumpStreams();
}

}  // namespace internal
}  // namespace crashpad
