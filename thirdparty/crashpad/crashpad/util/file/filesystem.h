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

#ifndef CRASHPAD_UTIL_FILE_FILESYSTEM_H_
#define CRASHPAD_UTIL_FILE_FILESYSTEM_H_

#include <time.h>

#include "base/files/file_path.h"
#include "util/file/file_io.h"

namespace crashpad {

//! \brief Determines the modification time for a file, directory, or symbolic
//!     link, logging a message on failure.
//!
//! \param[in] path The file to get the modification time for.
//! \param[out] mtime The modification time as seconds since the POSIX Epoch.
//! \return `true` on success. `false` on failure with a message logged.
bool FileModificationTime(const base::FilePath& path, timespec* mtime);

//! \brief Creates a directory, logging a message on failure.
//!
//! \param[in] path The path to the directory to create.
//! \param[in] permissions The permissions to use if the directory is created.
//! \param[in] may_reuse If `true`, this function will return `true` if a
//!     directory or symbolic link to a directory with path \a path already
//!     exists. If the directory already exists, it's permissions may differ
//!     from \a permissions.
//! \return `true` if the directory is successfully created or it already
//!     existed and \a may_reuse is `true`. Otherwise, `false`.
bool LoggingCreateDirectory(const base::FilePath& path,
                            FilePermissions permissions,
                            bool may_reuse);

//! \brief Moves a file, symbolic link, or directory, logging a message on
//!     failure.
//!
//! \a source must exist and refer to a file, symbolic link, or directory.
//!
//! \a source and \a dest must be on the same filesystem.
//!
//! If \a dest exists, it may be overwritten:
//!
//! If \a dest exists and refers to a file or to a live or dangling symbolic
//! link to a file, it will be overwritten if \a source also refers to a file or
//! to a live or dangling symbolic link to a file or directory.
//!
//! On POSIX, if \a dest refers to a directory, it will be overwritten only if
//! it is empty and \a source also refers to a directory.
//!
//! On Windows, if \a dest refers to a directory or to a live or dangling
//! symbolic link to a directory, it will not be overwritten.
//!
//! \param[in] source The path to the file to be moved.
//! \param[in] dest The new path for the file.
//! \return `true` on success. `false` on failure with a message logged.
bool MoveFileOrDirectory(const base::FilePath& source,
                         const base::FilePath& dest);

//! \brief Determines if a path refers to a regular file, logging a message on
//!     failure.
//!
//! On POSIX, this function returns `true` if \a path refers to a file that is
//! not a symbolic link, directory, or other kind of special file.
//!
//! On Windows, this function returns `true` if \a path refers to a file that
//! is not a symbolic link or directory.
//!
//! \param[in] path The path to the file to check.
//! \return `true` if the file exists and is a regular file. Otherwise `false`.
bool IsRegularFile(const base::FilePath& path);

//! \brief Determines if a path refers to a directory, logging a message on
//!     failure.
//!
//! \param[in] path The path to check.
//! \param[in] allow_symlinks Whether to allow the final component in the path
//!     to be a symbolic link to a directory.
//! \return `true` if the path exists and is a directory. Otherwise `false`.
bool IsDirectory(const base::FilePath& path, bool allow_symlinks);

//! \brief Removes a file or a symbolic link to a file or directory, logging a
//!     message on failure.
//!
//! \param[in] path The path to the file to remove.
//! \return `true` on success. `false` on failure with a message logged.
bool LoggingRemoveFile(const base::FilePath& path);

//! \brief Non-recurseively removes an empty directory, logging a message on
//!     failure.
//!
//! This function will not remove symbolic links to directories.
//!
//! \param[in] path The to the directory to remove.
//! \return `true` if the directory was removed. Otherwise, `false`.
bool LoggingRemoveDirectory(const base::FilePath& path);

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_FILE_FILESYSTEM_H_
