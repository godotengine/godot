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

#ifndef CRASHPAD_HANDLER_MAC_FILE_LIMIT_ANNOTATION_H_
#define CRASHPAD_HANDLER_MAC_FILE_LIMIT_ANNOTATION_H_

namespace crashpad {

//! \brief Records a `"file-limits"` simple annotation for the process.
//!
//! This annotation will be used to confirm the theory that certain crashes are
//! caused by systems at or near their file descriptor table size limits.
//!
//! The format of the annotation is four comma-separated values: the system-wide
//! `kern.num_files` and `kern.maxfiles` values from `sysctl()`, and the
//! process-specific current and maximum file descriptor limits from
//! `getrlimit(RLIMIT_NOFILE, …)`.
//!
//! See https://crashpad.chromium.org/bug/180.
//!
//! TODO(mark): Remove this annotation after sufficient data has been collected
//! for analysis.
void RecordFileLimitAnnotation();

}  // namespace crashpad

#endif  // CRASHPAD_HANDLER_MAC_FILE_LIMIT_ANNOTATION_H_
