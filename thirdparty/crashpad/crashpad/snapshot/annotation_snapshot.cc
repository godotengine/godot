// Copyright 2017 The Crashpad Authors. All rights reserved.
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

#include "snapshot/annotation_snapshot.h"

namespace crashpad {

AnnotationSnapshot::AnnotationSnapshot() : name(), type(0), value() {}

AnnotationSnapshot::AnnotationSnapshot(const std::string& name,
                                       uint16_t type,
                                       const std::vector<uint8_t>& value)
    : name(name), type(type), value(value) {}

AnnotationSnapshot::~AnnotationSnapshot() = default;

bool AnnotationSnapshot::operator==(const AnnotationSnapshot& other) const {
  return name == other.name && type == other.type && value == other.value;
}

}  // namespace crashpad
