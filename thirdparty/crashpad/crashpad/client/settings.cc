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

#include "client/settings.h"

#include <stdint.h>

#include <limits>

#include "base/logging.h"
#include "base/posix/eintr_wrapper.h"
#include "util/file/filesystem.h"
#include "util/numeric/in_range_cast.h"

namespace crashpad {

#if defined(OS_FUCHSIA)

Settings::ScopedLockedFileHandle::ScopedLockedFileHandle()
    : handle_(kInvalidFileHandle), lockfile_path_() {
    }

Settings::ScopedLockedFileHandle::ScopedLockedFileHandle(
    FileHandle handle,
    const base::FilePath& lockfile_path)
    : handle_(handle), lockfile_path_(lockfile_path) {
}

Settings::ScopedLockedFileHandle::ScopedLockedFileHandle(
    ScopedLockedFileHandle&& other)
    : handle_(other.handle_), lockfile_path_(other.lockfile_path_) {
  other.handle_ = kInvalidFileHandle;
  other.lockfile_path_ = base::FilePath();
}

Settings::ScopedLockedFileHandle& Settings::ScopedLockedFileHandle::operator=(
    ScopedLockedFileHandle&& other) {
  handle_ = other.handle_;
  lockfile_path_ = other.lockfile_path_;

  other.handle_ = kInvalidFileHandle;
  other.lockfile_path_ = base::FilePath();
  return *this;
}

Settings::ScopedLockedFileHandle::~ScopedLockedFileHandle() {
  Destroy();
}

void Settings::ScopedLockedFileHandle::Destroy() {
  if (handle_ != kInvalidFileHandle) {
    CheckedCloseFile(handle_);
  }
  if (!lockfile_path_.empty()) {
    DCHECK(LoggingRemoveFile(lockfile_path_));
  }
}

#else // OS_FUCHSIA

namespace internal {

// static
void ScopedLockedFileHandleTraits::Free(FileHandle handle) {
  if (handle != kInvalidFileHandle) {
    LoggingUnlockFile(handle);
    CheckedCloseFile(handle);
  }
}

}  // namespace internal

#endif  // OS_FUCHSIA

struct Settings::Data {
  static const uint32_t kSettingsMagic = 'CPds';
  static const uint32_t kSettingsVersion = 1;

  enum Options : uint32_t {
    kUploadsEnabled = 1 << 0,
  };

  Data() : magic(kSettingsMagic),
           version(kSettingsVersion),
           options(0),
           padding_0(0),
           last_upload_attempt_time(0),
           client_id() {}

  uint32_t magic;
  uint32_t version;
  uint32_t options;
  uint32_t padding_0;
  int64_t last_upload_attempt_time;  // time_t
  UUID client_id;
};

Settings::Settings() = default;

Settings::~Settings() = default;

bool Settings::Initialize(const base::FilePath& file_path) {
  DCHECK(initialized_.is_uninitialized());
  initialized_.set_invalid();
  file_path_ = file_path;

  Data settings;
  if (!OpenForWritingAndReadSettings(&settings).is_valid())
    return false;

  initialized_.set_valid();
  return true;
}

bool Settings::GetClientID(UUID* client_id) {
  DCHECK(initialized_.is_valid());

  Data settings;
  if (!OpenAndReadSettings(&settings))
    return false;

  *client_id = settings.client_id;
  return true;
}

bool Settings::GetUploadsEnabled(bool* enabled) {
  DCHECK(initialized_.is_valid());

  Data settings;
  if (!OpenAndReadSettings(&settings))
    return false;

  *enabled = (settings.options & Data::Options::kUploadsEnabled) != 0;
  return true;
}

bool Settings::SetUploadsEnabled(bool enabled) {
  DCHECK(initialized_.is_valid());

  Data settings;
  ScopedLockedFileHandle handle = OpenForWritingAndReadSettings(&settings);
  if (!handle.is_valid())
    return false;

  if (enabled)
    settings.options |= Data::Options::kUploadsEnabled;
  else
    settings.options &= ~Data::Options::kUploadsEnabled;

  return WriteSettings(handle.get(), settings);
}

bool Settings::GetLastUploadAttemptTime(time_t* time) {
  DCHECK(initialized_.is_valid());

  Data settings;
  if (!OpenAndReadSettings(&settings))
    return false;

  *time = InRangeCast<time_t>(settings.last_upload_attempt_time,
                              std::numeric_limits<time_t>::max());
  return true;
}

bool Settings::SetLastUploadAttemptTime(time_t time) {
  DCHECK(initialized_.is_valid());

  Data settings;
  ScopedLockedFileHandle handle = OpenForWritingAndReadSettings(&settings);
  if (!handle.is_valid())
    return false;

  settings.last_upload_attempt_time = InRangeCast<int64_t>(time, 0);

  return WriteSettings(handle.get(), settings);
}

// static
Settings::ScopedLockedFileHandle Settings::MakeScopedLockedFileHandle(
    FileHandle file,
    FileLocking locking,
    const base::FilePath& file_path) {
  ScopedFileHandle scoped(file);
#if defined(OS_FUCHSIA)
  base::FilePath lockfile_path(file_path.value() + ".__lock__");
  if (scoped.is_valid()) {
    ScopedFileHandle lockfile_scoped(
        LoggingOpenFileForWrite(lockfile_path,
                                FileWriteMode::kCreateOrFail,
                                FilePermissions::kWorldReadable));
    // This is a lightweight attempt to try to catch racy behavior.
    DCHECK(lockfile_scoped.is_valid());
    return ScopedLockedFileHandle(scoped.release(), lockfile_path);
  }
  return ScopedLockedFileHandle(scoped.release(), base::FilePath());
#else
  if (scoped.is_valid()) {
    if (!LoggingLockFile(scoped.get(), locking))
      scoped.reset();
  }
  return ScopedLockedFileHandle(scoped.release());
#endif
}

Settings::ScopedLockedFileHandle Settings::OpenForReading() {
  return MakeScopedLockedFileHandle(
      LoggingOpenFileForRead(file_path()), FileLocking::kShared, file_path());
}

Settings::ScopedLockedFileHandle Settings::OpenForReadingAndWriting(
    FileWriteMode mode, bool log_open_error) {
  DCHECK(mode != FileWriteMode::kTruncateOrCreate);

  FileHandle handle;
  if (log_open_error) {
    handle = LoggingOpenFileForReadAndWrite(
        file_path(), mode, FilePermissions::kWorldReadable);
  } else {
    handle = OpenFileForReadAndWrite(
        file_path(), mode, FilePermissions::kWorldReadable);
  }

  return MakeScopedLockedFileHandle(
      handle, FileLocking::kExclusive, file_path());
}

bool Settings::OpenAndReadSettings(Data* out_data) {
  ScopedLockedFileHandle handle = OpenForReading();
  if (!handle.is_valid())
    return false;

  if (ReadSettings(handle.get(), out_data, true))
    return true;

  // The settings file is corrupt, so reinitialize it.
  handle.reset();

  // The settings failed to be read, so re-initialize them.
  return RecoverSettings(kInvalidFileHandle, out_data);
}

Settings::ScopedLockedFileHandle Settings::OpenForWritingAndReadSettings(
    Data* out_data) {
  ScopedLockedFileHandle handle;
  bool created = false;
  if (!initialized_.is_valid()) {
    // If this object is initializing, it hasn’t seen a settings file already,
    // so go easy on errors. Creating a new settings file for the first time
    // shouldn’t spew log messages.
    //
    // First, try to use an existing settings file.
    handle = OpenForReadingAndWriting(FileWriteMode::kReuseOrFail, false);

    if (!handle.is_valid()) {
      // Create a new settings file if it didn’t already exist.
      handle = OpenForReadingAndWriting(FileWriteMode::kCreateOrFail, false);

      if (handle.is_valid()) {
        created = true;
      }

      // There may have been a race to create the file, and something else may
      // have won. There will be one more attempt to try to open or create the
      // file below.
    }
  }

  if (!handle.is_valid()) {
    // Either the object is initialized, meaning it’s already seen a valid
    // settings file, or the object is initializing and none of the above
    // attempts to create the settings file succeeded. Either way, this is the
    // last chance for success, so if this fails, log a message.
    handle = OpenForReadingAndWriting(FileWriteMode::kReuseOrCreate, true);
  }

  if (!handle.is_valid())
    return ScopedLockedFileHandle();

  // Attempt reading the settings even if the file is known to have just been
  // created. The file-create and file-lock operations don’t occur atomically,
  // and something else may have written the settings before this invocation
  // took the lock. If the settings file was definitely just created, though,
  // don’t log any read errors. The expected non-race behavior in this case is a
  // zero-length read, with ReadSettings() failing.
  if (!ReadSettings(handle.get(), out_data, !created)) {
    if (!RecoverSettings(handle.get(), out_data))
      return ScopedLockedFileHandle();
  }

  return handle;
}

bool Settings::ReadSettings(FileHandle handle,
                            Data* out_data,
                            bool log_read_error) {
  if (LoggingSeekFile(handle, 0, SEEK_SET) != 0)
    return false;

  bool read_result =
      log_read_error
          ? LoggingReadFileExactly(handle, out_data, sizeof(*out_data))
          : ReadFileExactly(handle, out_data, sizeof(*out_data));

  if (!read_result)
    return false;

  if (out_data->magic != Data::kSettingsMagic) {
    LOG(ERROR) << "Settings magic is not " << Data::kSettingsMagic;
    return false;
  }

  if (out_data->version != Data::kSettingsVersion) {
    LOG(ERROR) << "Settings version is not " << Data::kSettingsVersion;
    return false;
  }

  return true;
}

bool Settings::WriteSettings(FileHandle handle, const Data& data) {
  if (LoggingSeekFile(handle, 0, SEEK_SET) != 0)
    return false;

  if (!LoggingTruncateFile(handle))
    return false;

  return LoggingWriteFile(handle, &data, sizeof(Data));
}

bool Settings::RecoverSettings(FileHandle handle, Data* out_data) {
  ScopedLockedFileHandle scoped_handle;
  if (handle == kInvalidFileHandle) {
    scoped_handle =
        OpenForReadingAndWriting(FileWriteMode::kReuseOrCreate, true);
    handle = scoped_handle.get();

    // Test if the file has already been recovered now that the exclusive lock
    // is held.
    if (ReadSettings(handle, out_data, true))
      return true;
  }

  if (handle == kInvalidFileHandle) {
    LOG(ERROR) << "Invalid file handle";
    return false;
  }

  if (!InitializeSettings(handle))
    return false;

  return ReadSettings(handle, out_data, true);
}

bool Settings::InitializeSettings(FileHandle handle) {
  Data settings;
  if (!settings.client_id.InitializeWithNew())
    return false;

  return WriteSettings(handle, settings);
}

}  // namespace crashpad
