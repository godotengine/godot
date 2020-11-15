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

#ifndef CRASHPAD_SNAPSHOT_MAC_MACH_O_IMAGE_ANNOTATIONS_READER_H_
#define CRASHPAD_SNAPSHOT_MAC_MACH_O_IMAGE_ANNOTATIONS_READER_H_

#include <map>
#include <string>
#include <vector>

#include "base/macros.h"
#include "snapshot/annotation_snapshot.h"
#include "snapshot/mac/process_types.h"

namespace crashpad {

class MachOImageReader;
class ProcessReaderMac;

//! \brief A reader for annotations stored in a Mach-O image mapped into another
//!     process.
//!
//! These annotations are stored for the benefit of crash reporters, and provide
//! information thought to be potentially useful for crash analysis. This class
//! can decode annotations stored in these formats:
//!  - CrashpadInfo. This format is used by Crashpad clients. The “simple
//!    annotations” are recovered from any module with a compatible data
//!    section, and are included in the annotations returned by SimpleMap().
//!  - `CrashReporterClient.h`’s `crashreporter_annotations_t`. This format is
//!    used by Apple code. The `message` and `message2` fields can be recovered
//!    from any module with a compatible data section, and are included in the
//!    annotations returned by Vector().
//!  - `dyld`’s `error_string`. This format is used exclusively by dyld,
//!    typically for fatal errors. This string can be recovered from any
//!    `MH_DYLINKER`-type module with this symbol, and is included in the
//!    annotations returned by Vector().
class MachOImageAnnotationsReader {
 public:
  //! \brief Constructs an object.
  //!
  //! \param[in] process_reader The reader for the remote process.
  //! \param[in] image_reader The MachOImageReader for the Mach-O image file
  //!     contained within the remote process.
  //! \param[in] name The module’s name, a string to be used in logged messages.
  //!     This string is for diagnostic purposes only, and may be empty.
  MachOImageAnnotationsReader(ProcessReaderMac* process_reader,
                              const MachOImageReader* image_reader,
                              const std::string& name);

  ~MachOImageAnnotationsReader() {}

  //! \brief Returns the module’s annotations that are organized as a vector of
  //!     strings.
  std::vector<std::string> Vector() const;

  //! \brief Returns the module’s annotations that are organized as key-value
  //!     pairs, where all keys and values are strings.
  std::map<std::string, std::string> SimpleMap() const;

  //! \brief Returns the module’s annotations that are organized as a list of
  //      typed annotation objects.
  std::vector<AnnotationSnapshot> AnnotationsList() const;

 private:
  // Reades crashreporter_annotations_t::message and
  // crashreporter_annotations_t::message2 on behalf of Vector().
  void ReadCrashReporterClientAnnotations(
      std::vector<std::string>* vector_annotations) const;

  // Reads dyld_error_string on behalf of Vector().
  void ReadDyldErrorStringAnnotation(
      std::vector<std::string>* vector_annotations) const;

  // Reads CrashpadInfo::simple_annotations_ on behalf of SimpleMap().
  void ReadCrashpadSimpleAnnotations(
      std::map<std::string, std::string>* simple_map_annotations) const;

  // Reads CrashpadInfo::annotations_list_ on behalf of AnnotationsList().
  void ReadCrashpadAnnotationsList(
      std::vector<AnnotationSnapshot>* vector_annotations) const;

  std::string name_;
  ProcessReaderMac* process_reader_;  // weak
  const MachOImageReader* image_reader_;  // weak

  DISALLOW_COPY_AND_ASSIGN(MachOImageAnnotationsReader);
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_MAC_MACH_O_IMAGE_ANNOTATIONS_READER_H_
