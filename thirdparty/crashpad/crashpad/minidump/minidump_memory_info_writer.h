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

#ifndef CRASHPAD_MINIDUMP_MINIDUMP_MEMORY_INFO_WRITER_H_
#define CRASHPAD_MINIDUMP_MINIDUMP_MEMORY_INFO_WRITER_H_

#include <windows.h>
#include <dbghelp.h>
#include <stdint.h>
#include <sys/types.h>

#include <vector>

#include "base/macros.h"
#include "minidump/minidump_stream_writer.h"
#include "minidump/minidump_writable.h"

namespace crashpad {

class MemoryMapRegionSnapshot;
class MinidumpContextWriter;
class MinidumpMemoryListWriter;
class MinidumpMemoryWriter;

//! \brief The writer for a MINIDUMP_MEMORY_INFO_LIST stream in a minidump file,
//!     containing a list of MINIDUMP_MEMORY_INFO objects.
class MinidumpMemoryInfoListWriter final
    : public internal::MinidumpStreamWriter {
 public:
  MinidumpMemoryInfoListWriter();
  ~MinidumpMemoryInfoListWriter() override;

  //! \brief Initializes a MINIDUMP_MEMORY_INFO_LIST based on \a memory_map.
  //!
  //! \param[in] memory_map The vector of memory map region snapshots to use as
  //!     source data.
  //!
  //! \note Valid in #kStateMutable.
  void InitializeFromSnapshot(
      const std::vector<const MemoryMapRegionSnapshot*>& memory_map);

 protected:
  // MinidumpWritable:
  bool Freeze() override;
  size_t SizeOfObject() override;
  std::vector<internal::MinidumpWritable*> Children() override;
  bool WriteObject(FileWriterInterface* file_writer) override;

  // MinidumpStreamWriter:
  MinidumpStreamType StreamType() const override;

 private:
  MINIDUMP_MEMORY_INFO_LIST memory_info_list_base_;
  std::vector<MINIDUMP_MEMORY_INFO> items_;

  DISALLOW_COPY_AND_ASSIGN(MinidumpMemoryInfoListWriter);
};

}  // namespace crashpad

#endif  // CRASHPAD_MINIDUMP_MINIDUMP_MEMORY_INFO_WRITER_H_
