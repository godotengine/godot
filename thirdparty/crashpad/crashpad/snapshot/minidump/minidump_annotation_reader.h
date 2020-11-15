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

#ifndef SNAPSHOT_MINIDUMP_MINIDUMP_ANNOTATION_READER_H_
#define SNAPSHOT_MINIDUMP_MINIDUMP_ANNOTATION_READER_H_

#include <windows.h>
#include <dbghelp.h>

#include <vector>

#include "snapshot/annotation_snapshot.h"
#include "util/file/file_reader.h"

namespace crashpad {
namespace internal {

//! \brief Reads a MinidumpAnnotationList from a minidump file at \a location
//!     in \a file_reader, and returns it in \a list.
//!
//! \return `true` on success, with \a list set by replacing its contents.
//!     `false` on failure, with a message logged.
bool ReadMinidumpAnnotationList(FileReaderInterface* file_reader,
                                const MINIDUMP_LOCATION_DESCRIPTOR& location,
                                std::vector<AnnotationSnapshot>* list);

}  // namespace internal
}  // namespace crashpad

#endif  // SNAPSHOT_MINIDUMP_MINIDUMP_ANNOTATION_READER_H_
