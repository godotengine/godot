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

#ifndef CRASHPAD_SNAPSHOT_ANNOTATION_SNAPSHOT_H_
#define CRASHPAD_SNAPSHOT_ANNOTATION_SNAPSHOT_H_

#include <stdint.h>

#include <string>
#include <vector>

namespace crashpad {

// \!brief The snapshot representation of a client's Annotation.
struct AnnotationSnapshot {
  AnnotationSnapshot();
  AnnotationSnapshot(const std::string& name,
                     uint16_t type,
                     const std::vector<uint8_t>& value);
  ~AnnotationSnapshot();

  bool operator==(const AnnotationSnapshot& other) const;
  bool operator!=(const AnnotationSnapshot& other) const {
    return !(*this == other);
  }

  //! \brief A non-unique name by which this annotation can be identified.
  std::string name;

  //! \brief The Annotation::Type of data stored in the annotation. This value
  //!     may be client-supplied and need not correspond to a Crashpad-defined
  //!     type.
  uint16_t type;

  //! \brief The data for the annotation. Guranteed to be non-empty, since
  //!     empty annotations are skipped. The representation of the data should
  //!     be interpreted as \a #type.
  std::vector<uint8_t> value;
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_ANNOTATION_SNAPSHOT_H_
