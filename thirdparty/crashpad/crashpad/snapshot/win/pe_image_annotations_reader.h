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

#ifndef CRASHPAD_SNAPSHOT_WIN_PE_IMAGE_ANNOTATIONS_READER_H_
#define CRASHPAD_SNAPSHOT_WIN_PE_IMAGE_ANNOTATIONS_READER_H_

#include <map>
#include <string>
#include <vector>

#include "base/macros.h"
#include "snapshot/annotation_snapshot.h"

namespace crashpad {

class PEImageReader;
class ProcessReaderWin;

//! \brief A reader of annotations stored in a PE image mapped into another
//!     process.
//!
//! These annotations are stored for the benefit of crash reporters, and provide
//! information thought to be potentially useful for crash analysis.
//!
//! Currently, this class can decode information stored only in the CrashpadInfo
//! structure. This format is used by Crashpad clients. The "simple annotations"
//! are recovered from any module with a compatible data section, and are
//! included in the annotations returned by SimpleMap().
class PEImageAnnotationsReader {
 public:
  //! \brief Constructs the object.
  //!
  //! \param[in] process_reader The reader for the remote process.
  //! \param[in] pe_image_reader The PEImageReader for the PE image file
  //!     contained within the remote process.
  //! \param[in] name The module's name, a string to be used in logged messages.
  //!     This string is for diagnostic purposes only, and may be empty.
  PEImageAnnotationsReader(ProcessReaderWin* process_reader,
                           const PEImageReader* pe_image_reader,
                           const std::wstring& name);
  ~PEImageAnnotationsReader() {}

  //! \brief Returns the module's annotations that are organized as key-value
  //!     pairs, where all keys and values are strings.
  std::map<std::string, std::string> SimpleMap() const;

  //! \brief Returns the module's annotations that are organized as a list of
  //!     typed annotation objects.
  std::vector<AnnotationSnapshot> AnnotationsList() const;

 private:
  // Reads CrashpadInfo::simple_annotations_ on behalf of SimpleMap().
  template <class Traits>
  void ReadCrashpadSimpleAnnotations(
      std::map<std::string, std::string>* simple_map_annotations) const;

  // Reads CrashpadInfo::annotations_list_ on behalf of AnnotationsList().
  template <class Traits>
  void ReadCrashpadAnnotationsList(
      std::vector<AnnotationSnapshot>* vector_annotations) const;

  std::wstring name_;
  ProcessReaderWin* process_reader_;  // weak
  const PEImageReader* pe_image_reader_;  // weak

  DISALLOW_COPY_AND_ASSIGN(PEImageAnnotationsReader);
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_WIN_PE_IMAGE_ANNOTATIONS_READER_H_
