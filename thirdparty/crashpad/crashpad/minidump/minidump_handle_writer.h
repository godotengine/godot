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

#ifndef CRASHPAD_MINIDUMP_MINIDUMP_HANDLE_WRITER_H_
#define CRASHPAD_MINIDUMP_MINIDUMP_HANDLE_WRITER_H_

#include <windows.h>
#include <dbghelp.h>
#include <sys/types.h>

#include <map>
#include <string>
#include <vector>

#include "minidump/minidump_stream_writer.h"
#include "minidump/minidump_string_writer.h"
#include "minidump/minidump_writable.h"
#include "snapshot/handle_snapshot.h"

namespace crashpad {

//! \brief The writer for a MINIDUMP_HANDLE_DATA_STREAM stream in a minidump
//!     and its contained MINIDUMP_HANDLE_DESCRIPTOR s.
//!
//! As we currently do not track any data beyond what MINIDUMP_HANDLE_DESCRIPTOR
//! supports, we only write that type of record rather than the newer
//! MINIDUMP_HANDLE_DESCRIPTOR_2.
//!
//! Note that this writer writes both the header (MINIDUMP_HANDLE_DATA_STREAM)
//! and the list of objects (MINIDUMP_HANDLE_DESCRIPTOR), which is different
//! from some of the other list writers.
class MinidumpHandleDataWriter final : public internal::MinidumpStreamWriter {
 public:
  MinidumpHandleDataWriter();
  ~MinidumpHandleDataWriter() override;

  //! \brief Adds a MINIDUMP_HANDLE_DESCRIPTOR for each handle in \a
  //!     handle_snapshot to the MINIDUMP_HANDLE_DATA_STREAM.
  //!
  //! \param[in] handle_snapshots The handle snapshots to use as source data.
  //!
  //! \note Valid in #kStateMutable.
  void InitializeFromSnapshot(
      const std::vector<HandleSnapshot>& handle_snapshots);

 protected:
  // MinidumpWritable:
  bool Freeze() override;
  size_t SizeOfObject() override;
  std::vector<MinidumpWritable*> Children() override;
  bool WriteObject(FileWriterInterface* file_writer) override;

  // MinidumpStreamWriter:
  MinidumpStreamType StreamType() const override;

 private:
  MINIDUMP_HANDLE_DATA_STREAM handle_data_stream_base_;
  std::vector<MINIDUMP_HANDLE_DESCRIPTOR> handle_descriptors_;
  std::map<std::string, internal::MinidumpUTF16StringWriter*> strings_;

  DISALLOW_COPY_AND_ASSIGN(MinidumpHandleDataWriter);
};

}  // namespace crashpad

#endif  // CRASHPAD_MINIDUMP_MINIDUMP_HANDLE_WRITER_H_
