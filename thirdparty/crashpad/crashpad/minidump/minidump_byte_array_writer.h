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

#ifndef CRASHPAD_MINIDUMP_MINIDUMP_BYTE_ARRAY_WRITER_H_
#define CRASHPAD_MINIDUMP_MINIDUMP_BYTE_ARRAY_WRITER_H_

#include <memory>
#include <vector>

#include "base/macros.h"
#include "minidump/minidump_extensions.h"
#include "minidump/minidump_writable.h"

namespace crashpad {

//! \brief Writes a variable-length byte array for a minidump into a
//!     \sa MinidumpByteArray.
class MinidumpByteArrayWriter final : public internal::MinidumpWritable {
 public:
  MinidumpByteArrayWriter();
  ~MinidumpByteArrayWriter() override;

  //! \brief Sets the data to be written.
  //!
  //! \note Valid in #kStateMutable.
  void set_data(const std::vector<uint8_t>& data) { data_ = data; }

  //! \brief Sets the data to be written.
  //!
  //! \note Valid in #kStateMutable.
  void set_data(const uint8_t* data, size_t size);

  //! \brief Gets the data to be written.
  //!
  //! \note Valid in any state.
  const std::vector<uint8_t>& data() const { return data_; }

 protected:
  // MinidumpWritable:

  bool Freeze() override;
  size_t SizeOfObject() override;
  bool WriteObject(FileWriterInterface* file_writer) override;

 private:
  std::unique_ptr<MinidumpByteArray> minidump_array_;
  std::vector<uint8_t> data_;

  DISALLOW_COPY_AND_ASSIGN(MinidumpByteArrayWriter);
};

}  // namespace crashpad

#endif  // CRASHPAD_MINIDUMP_MINIDUMP_BYTE_ARRAY_WRITER_H_
