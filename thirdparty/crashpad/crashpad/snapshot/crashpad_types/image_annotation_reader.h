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

#ifndef CRASHPAD_SNAPSHOT_CRASHPAD_TYPES_IMAGE_ANNOTATION_READER_H_
#define CRASHPAD_SNAPSHOT_CRASHPAD_TYPES_IMAGE_ANNOTATION_READER_H_

#include <map>
#include <string>
#include <vector>

#include "base/macros.h"
#include "snapshot/annotation_snapshot.h"
#include "util/misc/address_types.h"
#include "util/process/process_memory_range.h"

namespace crashpad {

//! \brief Reads Annotations from another process via a ProcessMemoryRange.
//!
//! These annotations are stored for the benefit of crash reporters, and provide
//! information thought to be potentially useful for crash analysis.
class ImageAnnotationReader {
 public:
  //! \brief Constructs the object.
  //!
  //! \param[in] memory A memory reader for the remote process.
  explicit ImageAnnotationReader(const ProcessMemoryRange* memory);

  ~ImageAnnotationReader();

  //! \brief Reads annotations that are organized as key-value pairs, where all
  //!     keys and values are strings.
  //!
  //! \param[in] address The address in the target process' address space of a
  //!     SimpleStringDictionary containing the annotations to read.
  //! \param[out] annotations The annotations read, valid if this method
  //!     returns `true`.
  //! \return `true` on success. `false` on failure with a message logged.
  bool SimpleMap(VMAddress address,
                 std::map<std::string, std::string>* annotations) const;

  //! \brief Reads the module's annotations that are organized as a list of
  //!     typed annotation objects.
  //!
  //! \param[in] address The address in the target process' address space of an
  //!     AnnotationList.
  //! \param[out] annotations The annotations read, valid if this method returns
  //!     `true`.
  //! \return `true` on success. `false` on failure with a message logged.
  bool AnnotationsList(VMAddress,
                       std::vector<AnnotationSnapshot>* annotations) const;

 private:
  template <class Traits>
  bool ReadAnnotationList(VMAddress address,
                          std::vector<AnnotationSnapshot>* annotations) const;

  const ProcessMemoryRange* memory_;  // weak

  DISALLOW_COPY_AND_ASSIGN(ImageAnnotationReader);
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_CRASHPAD_TYPES_IMAGE_ANNOTATION_READER_H_
