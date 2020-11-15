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

#ifndef CRASHPAD_MINIDUMP_MINIDUMP_FILE_WRITER_H_
#define CRASHPAD_MINIDUMP_MINIDUMP_FILE_WRITER_H_

#include <windows.h>
#include <dbghelp.h>
#include <sys/types.h>

#include <memory>
#include <set>
#include <vector>

#include "base/macros.h"
#include "minidump/minidump_extensions.h"
#include "minidump/minidump_stream_writer.h"
#include "minidump/minidump_writable.h"
#include "util/file/file_io.h"

namespace crashpad {

class ProcessSnapshot;
class MinidumpUserExtensionStreamDataSource;

//! \brief The root-level object in a minidump file.
//!
//! This object writes a MINIDUMP_HEADER and list of MINIDUMP_DIRECTORY entries
//! to a minidump file.
class MinidumpFileWriter final : public internal::MinidumpWritable {
 public:
  MinidumpFileWriter();
  ~MinidumpFileWriter() override;

  //! \brief Initializes the MinidumpFileWriter and populates it with
  //!     appropriate child streams based on \a process_snapshot.
  //!
  //! This method will add additional streams to the minidump file as children
  //! of the MinidumpFileWriter object and as pointees of the top-level
  //! MINIDUMP_DIRECTORY. To do so, it will obtain other snapshot information
  //! from \a process_snapshot, such as a SystemSnapshot, lists of
  //! ThreadSnapshot and ModuleSnapshot objects, and, if available, an
  //! ExceptionSnapshot.
  //!
  //! The streams are added in the order that they are expected to be most
  //! useful to minidump readers, to improve data locality and minimize seeking.
  //! The streams are added in this order:
  //!  - kMinidumpStreamTypeSystemInfo
  //!  - kMinidumpStreamTypeMiscInfo
  //!  - kMinidumpStreamTypeThreadList
  //!  - kMinidumpStreamTypeException (if present)
  //!  - kMinidumpStreamTypeModuleList
  //!  - kMinidumpStreamTypeUnloadedModuleList (if present)
  //!  - kMinidumpStreamTypeCrashpadInfo (if present)
  //!  - kMinidumpStreamTypeMemoryInfoList (if present)
  //!  - kMinidumpStreamTypeHandleData (if present)
  //!  - User streams (if present)
  //!  - kMinidumpStreamTypeMemoryList
  //!
  //! \param[in] process_snapshot The process snapshot to use as source data.
  //!
  //! \note Valid in #kStateMutable. No mutator methods may be called before
  //!     this method, and it is not normally necessary to call any mutator
  //!     methods after this method.
  void InitializeFromSnapshot(const ProcessSnapshot* process_snapshot);

  //! \brief Sets MINIDUMP_HEADER::Timestamp.
  //!
  //! \note Valid in #kStateMutable.
  void SetTimestamp(time_t timestamp);

  //! \brief Adds a stream to the minidump file and arranges for a
  //!     MINIDUMP_DIRECTORY entry to point to it.
  //!
  //! This object takes ownership of \a stream and becomes its parent in the
  //! overall tree of internal::MinidumpWritable objects.
  //!
  //! At most one object of each stream type (as obtained from
  //! internal::MinidumpStreamWriter::StreamType()) may be added to a
  //! MinidumpFileWriter object. If an attempt is made to add a stream whose
  //! type matches an existing stream’s type, this method discards the new
  //! stream.
  //!
  //! \note Valid in #kStateMutable.
  //!
  //! \return `true` on success. `false` on failure, as occurs when an attempt
  //!     is made to add a stream whose type matches an existing stream’s type,
  //!     with a message logged.
  bool AddStream(std::unique_ptr<internal::MinidumpStreamWriter> stream);

  //! \brief Adds a user extension stream to the minidump file and arranges for
  //!     a MINIDUMP_DIRECTORY entry to point to it.
  //!
  //! This object takes ownership of \a user_extension_stream_data.
  //!
  //! At most one object of each stream type (as obtained from
  //! internal::MinidumpStreamWriter::StreamType()) may be added to a
  //! MinidumpFileWriter object. If an attempt is made to add a stream whose
  //! type matches an existing stream’s type, this method discards the new
  //! stream.
  //!
  //! \param[in] user_extension_stream_data The stream data to add to the
  //!    minidump file. Note that the buffer this object points to must be valid
  //!    through WriteEverything().
  //!
  //! \note Valid in #kStateMutable.
  //!
  //! \return `true` on success. `false` on failure, as occurs when an attempt
  //!     is made to add a stream whose type matches an existing stream’s type,
  //!     with a message logged.
  bool AddUserExtensionStream(
      std::unique_ptr<MinidumpUserExtensionStreamDataSource>
          user_extension_stream_data);

  // MinidumpWritable:

  //! \copydoc internal::MinidumpWritable::WriteEverything()
  //!
  //! This method does not initially write the final value for
  //! MINIDUMP_HEADER::Signature. After all child objects have been written, it
  //! rewinds to the beginning of the file and writes the correct value for this
  //! field. This prevents incompletely-written minidump files from being
  //! mistaken for valid ones.
  bool WriteEverything(FileWriterInterface* file_writer) override;

 protected:
  // MinidumpWritable:
  bool Freeze() override;
  size_t SizeOfObject() override;
  std::vector<MinidumpWritable*> Children() override;
  bool WillWriteAtOffsetImpl(FileOffset offset) override;
  bool WriteObject(FileWriterInterface* file_writer) override;

 private:
  MINIDUMP_HEADER header_;
  std::vector<std::unique_ptr<internal::MinidumpStreamWriter>> streams_;

  // Protects against multiple streams with the same ID being added.
  std::set<MinidumpStreamType> stream_types_;

  DISALLOW_COPY_AND_ASSIGN(MinidumpFileWriter);
};

}  // namespace crashpad

#endif  // CRASHPAD_MINIDUMP_MINIDUMP_WRITER_H_
