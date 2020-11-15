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

#ifndef CRASHPAD_MINIDUMP_MINIDUMP_USER_EXTENSION_STREAM_DATA_SOURCE_H_
#define CRASHPAD_MINIDUMP_MINIDUMP_USER_EXTENSION_STREAM_DATA_SOURCE_H_

#include <stdint.h>
#include <sys/types.h>

#include "base/macros.h"

#include "minidump/minidump_extensions.h"

namespace crashpad {

//! \brief Describes a user extension data stream in a minidump.
class MinidumpUserExtensionStreamDataSource {
 public:
  //! \brief An interface implemented by readers of
  //!     MinidumpUserExtensionStreamDataSource.
  class Delegate {
   public:
    //! \brief Called by  MinidumpUserExtensionStreamDataSource::Read() to
    //!     provide data requested by a call to that method.
    //!
    //! \param[in] data A pointer to the data that was read. The callee does not
    //!     take ownership of this data. This data is only valid for the
    //!     duration of the call to this method. This parameter may be `nullptr`
    //!     if \a size is `0`.
    //! \param[in] size The size of the data that was read.
    //!
    //! \return `true` on success, `false` on failure.
    //!     MinidumpUserExtensionStreamDataSource::ReadStreamData() will use
    //!     this as its own return value.
    virtual bool ExtensionStreamDataSourceRead(const void* data,
                                               size_t size) = 0;

   protected:
    ~Delegate() {}
  };

  //! \brief Constructs a MinidumpUserExtensionStreamDataSource.
  //!
  //! \param[in] stream_type The type of the user extension stream.
  explicit MinidumpUserExtensionStreamDataSource(uint32_t stream_type);
  virtual ~MinidumpUserExtensionStreamDataSource();

  MinidumpStreamType stream_type() const { return stream_type_; }

  //! \brief The size of this data stream.
  virtual size_t StreamDataSize() = 0;

  //! \brief Calls Delegate::UserStreamDataSourceRead(), providing it with
  //!     the stream data.
  //!
  //! Implementations do not necessarily compute the stream data prior to
  //! this method being called. The stream data may be computed or loaded
  //! lazily and may be discarded after being passed to the delegate.
  //!
  //! \return `false` on failure, otherwise, the return value of
  //!     Delegate::ExtensionStreamDataSourceRead(), which should be `true` on
  //!     success and `false` on failure.
  virtual bool ReadStreamData(Delegate* delegate) = 0;

 private:
  MinidumpStreamType stream_type_;

  DISALLOW_COPY_AND_ASSIGN(MinidumpUserExtensionStreamDataSource);
};

}  // namespace crashpad

#endif  // CRASHPAD_MINIDUMP_MINIDUMP_USER_EXTENSION_STREAM_DATA_SOURCE_H_
