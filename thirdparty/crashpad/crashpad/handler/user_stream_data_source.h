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

#ifndef CRASHPAD_HANDLER_USER_STREAM_DATA_SOURCE_H_
#define CRASHPAD_HANDLER_USER_STREAM_DATA_SOURCE_H_

#include <memory>
#include <vector>

namespace crashpad {

class MinidumpFileWriter;
class MinidumpUserExtensionStreamDataSource;
class ProcessSnapshot;

//! \brief Extensibility interface for embedders who wish to add custom streams
//!     to minidumps.
class UserStreamDataSource {
 public:
  virtual ~UserStreamDataSource() {}

  //! \brief Produce the contents for an extension stream for a crashed program.
  //!
  //! Called after \a process_snapshot has been initialized for the crashed
  //! process to (optionally) produce the contents of a user extension stream
  //! that will be attached to the minidump.
  //!
  //! \param[in] process_snapshot An initialized snapshot for the crashed
  //!     process.
  //!
  //! \return A new data source for the stream to add to the minidump or
  //!      `nullptr` on failure or to opt out of adding a stream.
  virtual std::unique_ptr<MinidumpUserExtensionStreamDataSource>
  ProduceStreamData(ProcessSnapshot* process_snapshot) = 0;
};

using UserStreamDataSources =
    std::vector<std::unique_ptr<UserStreamDataSource>>;

//! \brief Adds user extension streams to a minidump.
//!
//! Dispatches to each source in \a user_stream_data_sources and adds returned
//! extension streams to \a minidump_file_writer.
//!
//! \param[in] user_stream_data_sources A pointer to the data sources, or
//!     `nullptr`.
//! \param[in] process_snapshot An initialized snapshot to the crashing process.
//! \param[in] minidump_file_writer Any extension streams will be added to this
//!     minidump.
void AddUserExtensionStreams(
    const UserStreamDataSources* user_stream_data_sources,
    ProcessSnapshot* process_snapshot,
    MinidumpFileWriter* minidump_file_writer);

}  // namespace crashpad

#endif  // CRASHPAD_HANDLER_USER_STREAM_DATA_SOURCE_H_
