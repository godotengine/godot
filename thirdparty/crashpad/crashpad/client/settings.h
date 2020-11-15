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

#ifndef CRASHPAD_CLIENT_SETTINGS_H_
#define CRASHPAD_CLIENT_SETTINGS_H_

#include <time.h>

#include <string>

#include "base/files/file_path.h"
#include "base/macros.h"
#include "base/scoped_generic.h"
#include "build/build_config.h"
#include "util/file/file_io.h"
#include "util/misc/initialization_state.h"
#include "util/misc/uuid.h"

namespace crashpad {

namespace internal {

struct ScopedLockedFileHandleTraits {
  static FileHandle InvalidValue() { return kInvalidFileHandle; }
  static void Free(FileHandle handle);
};

}  // namespace internal

//! \brief An interface for accessing and modifying the settings of a
//!     CrashReportDatabase.
//!
//! This class must not be instantiated directly, but rather an instance of it
//! should be retrieved via CrashReportDatabase::GetSettings().
class Settings {
 public:
  Settings();
  ~Settings();

  //! \brief Initializes the settings data store.
  //!
  //! This method must be called only once, and must be successfully called
  //! before any other method in this class may be called.
  //!
  //! \param[in] path The location to store the settings data.
  //! \return `true` if the data store was initialized successfully, otherwise
  //!     `false` with an error logged.
  bool Initialize(const base::FilePath& path);

  //! \brief Retrieves the immutable identifier for this client, which is used
  //!     on a server to locate all crash reports from a specific Crashpad
  //!     database.
  //!
  //! This is automatically initialized when the database is created.
  //!
  //! \param[out] client_id The unique client identifier.
  //!
  //! \return On success, returns `true`, otherwise returns `false` with an
  //!     error logged.
  bool GetClientID(UUID* client_id);

  //! \brief Retrieves the user’s preference for submitting crash reports to a
  //!     collection server.
  //!
  //! The default value is `false`.
  //!
  //! \param[out] enabled Whether crash reports should be uploaded.
  //!
  //! \return On success, returns `true`, otherwise returns `false` with an
  //!     error logged.
  bool GetUploadsEnabled(bool* enabled);

  //! \brief Sets the user’s preference for submitting crash reports to a
  //!     collection server.
  //!
  //! \param[in] enabled Whether crash reports should be uploaded.
  //!
  //! \return On success, returns `true`, otherwise returns `false` with an
  //!     error logged.
  bool SetUploadsEnabled(bool enabled);

  //! \brief Retrieves the last time at which a report was attempted to be
  //!     uploaded.
  //!
  //! The default value is `0` if it has never been set before.
  //!
  //! \param[out] time The last time at which a report was uploaded.
  //!
  //! \return On success, returns `true`, otherwise returns `false` with an
  //!     error logged.
  bool GetLastUploadAttemptTime(time_t* time);

  //! \brief Sets the last time at which a report was attempted to be uploaded.
  //!
  //! This is only meant to be used internally by the CrashReportDatabase.
  //!
  //! \param[in] time The last time at which a report was uploaded.
  //!
  //! \return On success, returns `true`, otherwise returns `false` with an
  //!     error logged.
  bool SetLastUploadAttemptTime(time_t time);

 private:
  struct Data;

  // This must be constructed with MakeScopedLockedFileHandle(). It both unlocks
  // and closes the file on destruction. Note that on Fuchsia, this handle DOES
  // NOT offer correct operation, only an attempt to DCHECK if racy behavior is
  // detected.
#if defined(OS_FUCHSIA)
  struct ScopedLockedFileHandle {
   public:
    ScopedLockedFileHandle();
    ScopedLockedFileHandle(FileHandle handle,
                           const base::FilePath& lockfile_path);
    ScopedLockedFileHandle(ScopedLockedFileHandle&& other);
    ScopedLockedFileHandle& operator=(ScopedLockedFileHandle&& other);
    ~ScopedLockedFileHandle();

    // These mirror the non-Fuchsia ScopedLockedFileHandle via ScopedGeneric so
    // that calling code can pretend this implementation is the same.
    bool is_valid() const { return handle_ != kInvalidFileHandle; }
    FileHandle get() { return handle_; }
    void reset() {
      Destroy();
      handle_ = kInvalidFileHandle;
      lockfile_path_ = base::FilePath();
    }

   private:
    void Destroy();

    FileHandle handle_;
    base::FilePath lockfile_path_;

    DISALLOW_COPY_AND_ASSIGN(ScopedLockedFileHandle);
  };
#else  // OS_FUCHSIA
  using ScopedLockedFileHandle =
      base::ScopedGeneric<FileHandle, internal::ScopedLockedFileHandleTraits>;
#endif  // OS_FUCHSIA
  static ScopedLockedFileHandle MakeScopedLockedFileHandle(
      FileHandle file,
      FileLocking locking,
      const base::FilePath& file_path);

  // Opens the settings file for reading. On error, logs a message and returns
  // the invalid handle.
  ScopedLockedFileHandle OpenForReading();

  // Opens the settings file for reading and writing. On error, logs a message
  // and returns the invalid handle. |mode| determines how the file will be
  // opened. |mode| must not be FileWriteMode::kTruncateOrCreate.
  //
  // If |log_open_error| is false, nothing will be logged for an error
  // encountered when attempting to open the file, but this method will still
  // return false. This is intended to be used to suppress error messages when
  // attempting to create a new settings file when multiple attempts are made.
  ScopedLockedFileHandle OpenForReadingAndWriting(FileWriteMode mode,
                                                  bool log_open_error);

  // Opens the settings file and reads the data. If that fails, an error will
  // be logged and the settings will be recovered and re-initialized. If that
  // also fails, returns false with additional log data from recovery.
  bool OpenAndReadSettings(Data* out_data);

  // Opens the settings file for writing and reads the data. If reading fails,
  // recovery is attempted. Returns the opened file handle on success, or the
  // invalid file handle on failure, with an error logged.
  ScopedLockedFileHandle OpenForWritingAndReadSettings(Data* out_data);

  // Reads the settings from |handle|. Logs an error and returns false on
  // failure. This does not perform recovery.
  //
  // |handle| must be the result of OpenForReading() or
  // OpenForReadingAndWriting().
  //
  // If |log_read_error| is false, nothing will be logged for a read error, but
  // this method will still return false. This is intended to be used to
  // suppress error messages when attempting to read a newly created settings
  // file.
  bool ReadSettings(FileHandle handle, Data* out_data, bool log_read_error);

  // Writes the settings to |handle|. Logs an error and returns false on
  // failure. This does not perform recovery.
  //
  // |handle| must be the result of OpenForReadingAndWriting().
  bool WriteSettings(FileHandle handle, const Data& data);

  // Recovers the settings file by re-initializing the data. If |handle| is the
  // invalid handle, this will open the file; if it is not, then it must be the
  // result of OpenForReadingAndWriting(). If the invalid handle is passed, the
  // caller must not be holding the handle. The new settings data are stored in
  // |out_data|. Returns true on success and false on failure, with an error
  // logged.
  bool RecoverSettings(FileHandle handle, Data* out_data);

  // Initializes a settings file and writes the data to |handle|. Returns true
  // on success and false on failure, with an error logged.
  //
  // |handle| must be the result of OpenForReadingAndWriting().
  bool InitializeSettings(FileHandle handle);

  const base::FilePath& file_path() const { return file_path_; }

  base::FilePath file_path_;

  InitializationState initialized_;

  DISALLOW_COPY_AND_ASSIGN(Settings);
};

}  // namespace crashpad

#endif  // CRASHPAD_CLIENT_SETTINGS_H_
