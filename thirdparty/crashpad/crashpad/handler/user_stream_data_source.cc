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

#include "handler/user_stream_data_source.h"

#include "base/logging.h"
#include "minidump/minidump_file_writer.h"
#include "minidump/minidump_user_extension_stream_data_source.h"
#include "snapshot/process_snapshot.h"

namespace crashpad {

void AddUserExtensionStreams(
    const UserStreamDataSources* user_stream_data_sources,
    ProcessSnapshot* process_snapshot,
    MinidumpFileWriter* minidump_file_writer) {
  if (!user_stream_data_sources)
    return;
  for (const auto& source : *user_stream_data_sources) {
    std::unique_ptr<MinidumpUserExtensionStreamDataSource> data_source(
        source->ProduceStreamData(process_snapshot));
    if (data_source &&
        !minidump_file_writer->AddUserExtensionStream(std::move(data_source))) {
      // This should only happen if multiple user stream sources yield the
      // same stream type. It's the user's responsibility to make sure
      // sources don't collide on the same stream type.
      LOG(ERROR) << "AddUserExtensionStream failed";
    }
  }
}

}  // namespace crashpad
