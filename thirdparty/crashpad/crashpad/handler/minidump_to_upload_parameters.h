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

#ifndef HANDLER_MINIDUMP_TO_UPLOAD_PARAMETERS_H_
#define HANDLER_MINIDUMP_TO_UPLOAD_PARAMETERS_H_

#include <map>
#include <string>

#include "snapshot/process_snapshot.h"

namespace crashpad {

//! \brief Given a ProcessSnapshot, returns a map of key-value pairs to use as
//!     HTTP form parameters for upload to a Breakpad crash report colleciton
//!     server.
//!
//! The map is built by combining the process simple annotations map with
//! each module’s simple annotations map and annotation objects.
//!
//! In the case of duplicate simple map keys or annotation names, the map will
//! retain the first value found for any key, and will log a warning about
//! discarded values. The precedence rules for annotation names are: the two
//! reserved keys discussed below, process simple annotations, module simple
//! annotations, and module annotation objects.
//!
//! For annotation objects, only ones of that are Annotation::Type::kString are
//! included.
//!
//! Each module’s annotations vector is also examined and built into a single
//! string value, with distinct elements separated by newlines, and stored at
//! the key named “list_annotations”, which supersedes any other key found by
//! that name.
//!
//! The client ID stored in the minidump is converted to a string and stored at
//! the key named “guid”, which supersedes any other key found by that name.
//!
//! In the event of an error reading the minidump file, a message will be
//! logged.
//!
//! \param[in] process_snapshot The process snapshot from which annotations
//!     will be extracted.
//!
//! \returns A string map of the annotations.
std::map<std::string, std::string> BreakpadHTTPFormParametersFromMinidump(
    const ProcessSnapshot* process_snapshot);

}  // namespace crashpad

#endif  // HANDLER_MINIDUMP_TO_UPLOAD_PARAMETERS_H_
