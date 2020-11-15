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

#ifndef CRASHPAD_MINIDUMP_MINIDUMP_STREAM_WRITER_H_
#define CRASHPAD_MINIDUMP_MINIDUMP_STREAM_WRITER_H_

#include <windows.h>
#include <dbghelp.h>

#include "base/macros.h"
#include "minidump/minidump_extensions.h"
#include "minidump/minidump_writable.h"

namespace crashpad {
namespace internal {

//! \brief The base class for all second-level objects (“streams”) in a minidump
//!     file.
//!
//! Instances of subclasses of this class are children of the root-level
//! MinidumpFileWriter object.
class MinidumpStreamWriter : public MinidumpWritable {
 public:
  ~MinidumpStreamWriter() override;

  //! \brief Returns an object’s stream type.
  //!
  //! \note Valid in any state.
  virtual MinidumpStreamType StreamType() const = 0;

  //! \brief Returns a MINIDUMP_DIRECTORY entry that serves as a pointer to this
  //!     stream.
  //!
  //! This method is provided for MinidumpFileWriter, which calls it in order to
  //! obtain the directory entry for a stream.
  //!
  //! \note Valid only in #kStateWritable.
  const MINIDUMP_DIRECTORY* DirectoryListEntry() const;

 protected:
  MinidumpStreamWriter();

  // MinidumpWritable:
  bool Freeze() override;

 private:
  MINIDUMP_DIRECTORY directory_list_entry_;

  DISALLOW_COPY_AND_ASSIGN(MinidumpStreamWriter);
};

}  // namespace internal
}  // namespace crashpad

#endif  // CRASHPAD_MINIDUMP_STREAM_WRITER_H_
