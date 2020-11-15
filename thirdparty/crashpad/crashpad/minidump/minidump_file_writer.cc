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

#include "minidump/minidump_file_writer.h"

#include <utility>

#include "base/logging.h"
#include "minidump/minidump_crashpad_info_writer.h"
#include "minidump/minidump_exception_writer.h"
#include "minidump/minidump_handle_writer.h"
#include "minidump/minidump_memory_info_writer.h"
#include "minidump/minidump_memory_writer.h"
#include "minidump/minidump_misc_info_writer.h"
#include "minidump/minidump_module_writer.h"
#include "minidump/minidump_system_info_writer.h"
#include "minidump/minidump_thread_id_map.h"
#include "minidump/minidump_thread_writer.h"
#include "minidump/minidump_unloaded_module_writer.h"
#include "minidump/minidump_user_extension_stream_data_source.h"
#include "minidump/minidump_user_stream_writer.h"
#include "minidump/minidump_writer_util.h"
#include "snapshot/exception_snapshot.h"
#include "snapshot/module_snapshot.h"
#include "snapshot/process_snapshot.h"
#include "util/file/file_writer.h"
#include "util/numeric/safe_assignment.h"

namespace crashpad {

MinidumpFileWriter::MinidumpFileWriter()
    : MinidumpWritable(), header_(), streams_(), stream_types_() {
  // Don’t set the signature field right away. Leave it set to 0, so that a
  // partially-written minidump file isn’t confused for a complete and valid
  // one. The header will be rewritten in WriteToFile().
  header_.Signature = 0;

  header_.Version = MINIDUMP_VERSION;
  header_.CheckSum = 0;
  header_.Flags = MiniDumpNormal;
}

MinidumpFileWriter::~MinidumpFileWriter() {
}

void MinidumpFileWriter::InitializeFromSnapshot(
    const ProcessSnapshot* process_snapshot) {
  DCHECK_EQ(state(), kStateMutable);
  DCHECK_EQ(header_.Signature, 0u);
  DCHECK_EQ(header_.TimeDateStamp, 0u);
  DCHECK_EQ(static_cast<MINIDUMP_TYPE>(header_.Flags), MiniDumpNormal);
  DCHECK(streams_.empty());

  // This time is truncated to an integer number of seconds, not rounded, for
  // compatibility with the truncation of process_snapshot->ProcessStartTime()
  // done by MinidumpMiscInfoWriter::InitializeFromSnapshot(). Handling both
  // timestamps in the same way allows the highest-fidelity computation of
  // process uptime as the difference between the two values.
  timeval snapshot_time;
  process_snapshot->SnapshotTime(&snapshot_time);
  SetTimestamp(snapshot_time.tv_sec);

  const SystemSnapshot* system_snapshot = process_snapshot->System();
  auto system_info = std::make_unique<MinidumpSystemInfoWriter>();
  system_info->InitializeFromSnapshot(system_snapshot);
  bool add_stream_result = AddStream(std::move(system_info));
  DCHECK(add_stream_result);

  auto misc_info = std::make_unique<MinidumpMiscInfoWriter>();
  misc_info->InitializeFromSnapshot(process_snapshot);
  add_stream_result = AddStream(std::move(misc_info));
  DCHECK(add_stream_result);

  auto memory_list = std::make_unique<MinidumpMemoryListWriter>();
  auto thread_list = std::make_unique<MinidumpThreadListWriter>();
  thread_list->SetMemoryListWriter(memory_list.get());
  MinidumpThreadIDMap thread_id_map;
  thread_list->InitializeFromSnapshot(process_snapshot->Threads(),
                                      &thread_id_map);
  add_stream_result = AddStream(std::move(thread_list));
  DCHECK(add_stream_result);

  const ExceptionSnapshot* exception_snapshot = process_snapshot->Exception();
  if (exception_snapshot) {
    auto exception = std::make_unique<MinidumpExceptionWriter>();
    exception->InitializeFromSnapshot(exception_snapshot, thread_id_map);
    add_stream_result = AddStream(std::move(exception));
    DCHECK(add_stream_result);
  }

  auto module_list = std::make_unique<MinidumpModuleListWriter>();
  module_list->InitializeFromSnapshot(process_snapshot->Modules());
  add_stream_result = AddStream(std::move(module_list));
  DCHECK(add_stream_result);

  auto unloaded_modules = process_snapshot->UnloadedModules();
  if (!unloaded_modules.empty()) {
    auto unloaded_module_list =
        std::make_unique<MinidumpUnloadedModuleListWriter>();
    unloaded_module_list->InitializeFromSnapshot(unloaded_modules);
    add_stream_result = AddStream(std::move(unloaded_module_list));
    DCHECK(add_stream_result);
  }

  auto crashpad_info = std::make_unique<MinidumpCrashpadInfoWriter>();
  crashpad_info->InitializeFromSnapshot(process_snapshot);

  // Since the MinidumpCrashpadInfo stream is an extension, it’s safe to not add
  // it to the minidump file if it wouldn’t carry any useful information.
  if (crashpad_info->IsUseful()) {
    add_stream_result = AddStream(std::move(crashpad_info));
    DCHECK(add_stream_result);
  }

  std::vector<const MemoryMapRegionSnapshot*> memory_map_snapshot =
      process_snapshot->MemoryMap();
  if (!memory_map_snapshot.empty()) {
    auto memory_info_list = std::make_unique<MinidumpMemoryInfoListWriter>();
    memory_info_list->InitializeFromSnapshot(memory_map_snapshot);
    add_stream_result = AddStream(std::move(memory_info_list));
    DCHECK(add_stream_result);
  }

  std::vector<HandleSnapshot> handles_snapshot = process_snapshot->Handles();
  if (!handles_snapshot.empty()) {
    auto handle_data_writer = std::make_unique<MinidumpHandleDataWriter>();
    handle_data_writer->InitializeFromSnapshot(handles_snapshot);
    add_stream_result = AddStream(std::move(handle_data_writer));
    DCHECK(add_stream_result);
  }

  memory_list->AddFromSnapshot(process_snapshot->ExtraMemory());
  if (exception_snapshot) {
    memory_list->AddFromSnapshot(exception_snapshot->ExtraMemory());
  }

  // These user streams must be added last. Otherwise, a user stream with the
  // same type as a well-known stream could preempt the well-known stream. As it
  // stands now, earlier-discovered user streams can still preempt
  // later-discovered ones. The well-known memory list stream is added after
  // these user streams, but only with a check here to avoid adding a user
  // stream that would preempt the memory list stream.
  for (const auto& module : process_snapshot->Modules()) {
    for (const UserMinidumpStream* stream : module->CustomMinidumpStreams()) {
      if (stream->stream_type() == kMinidumpStreamTypeMemoryList) {
        LOG(WARNING) << "discarding duplicate stream of type "
                     << stream->stream_type();
        continue;
      }
      auto user_stream = std::make_unique<MinidumpUserStreamWriter>();
      user_stream->InitializeFromSnapshot(stream);
      AddStream(std::move(user_stream));
    }
  }

  // The memory list stream should be added last. This keeps the “extra memory”
  // at the end so that if the minidump file is truncated, other, more critical
  // data is more likely to be preserved. Note that non-“extra” memory regions
  // will not have to ride at the end of the file. Thread stack memory, for
  // example, exists as a children of threads, and appears alongside them in the
  // file, despite also being mentioned by the memory list stream.
  add_stream_result = AddStream(std::move(memory_list));
  DCHECK(add_stream_result);
}

void MinidumpFileWriter::SetTimestamp(time_t timestamp) {
  DCHECK_EQ(state(), kStateMutable);

  internal::MinidumpWriterUtil::AssignTimeT(&header_.TimeDateStamp, timestamp);
}

bool MinidumpFileWriter::AddStream(
    std::unique_ptr<internal::MinidumpStreamWriter> stream) {
  DCHECK_EQ(state(), kStateMutable);

  MinidumpStreamType stream_type = stream->StreamType();

  auto rv = stream_types_.insert(stream_type);
  if (!rv.second) {
    LOG(WARNING) << "discarding duplicate stream of type " << stream_type;
    return false;
  }

  streams_.push_back(std::move(stream));

  DCHECK_EQ(streams_.size(), stream_types_.size());
  return true;
}

bool MinidumpFileWriter::AddUserExtensionStream(
    std::unique_ptr<MinidumpUserExtensionStreamDataSource>
        user_extension_stream_data) {
  DCHECK_EQ(state(), kStateMutable);

  auto user_stream = std::make_unique<MinidumpUserStreamWriter>();
  user_stream->InitializeFromUserExtensionStream(
      std::move(user_extension_stream_data));

  return AddStream(std::move(user_stream));
}

bool MinidumpFileWriter::WriteEverything(FileWriterInterface* file_writer) {
  DCHECK_EQ(state(), kStateMutable);

  FileOffset start_offset = file_writer->Seek(0, SEEK_CUR);
  if (start_offset < 0) {
    return false;
  }

  if (!MinidumpWritable::WriteEverything(file_writer)) {
    return false;
  }

  FileOffset end_offset = file_writer->Seek(0, SEEK_CUR);
  if (end_offset < 0) {
    return false;
  }

  // Now that the entire minidump file has been completely written, go back to
  // the beginning and rewrite the header with the correct signature to identify
  // it as a valid minidump file.
  header_.Signature = MINIDUMP_SIGNATURE;

  if (file_writer->Seek(start_offset, SEEK_SET) != 0) {
    return false;
  }

  if (!file_writer->Write(&header_, sizeof(header_))) {
    return false;
  }

  // Seek back to the end of the file, in case some non-minidump content will be
  // written to the file after the minidump content.
  return file_writer->Seek(end_offset, SEEK_SET) >= 0;
}

bool MinidumpFileWriter::Freeze() {
  DCHECK_EQ(state(), kStateMutable);

  if (!MinidumpWritable::Freeze()) {
    return false;
  }

  size_t stream_count = streams_.size();
  CHECK_EQ(stream_count, stream_types_.size());

  if (!AssignIfInRange(&header_.NumberOfStreams, stream_count)) {
    LOG(ERROR) << "stream_count " << stream_count << " out of range";
    return false;
  }

  return true;
}

size_t MinidumpFileWriter::SizeOfObject() {
  DCHECK_GE(state(), kStateFrozen);
  DCHECK_EQ(streams_.size(), stream_types_.size());

  return sizeof(header_) + streams_.size() * sizeof(MINIDUMP_DIRECTORY);
}

std::vector<internal::MinidumpWritable*> MinidumpFileWriter::Children() {
  DCHECK_GE(state(), kStateFrozen);
  DCHECK_EQ(streams_.size(), stream_types_.size());

  std::vector<MinidumpWritable*> children;
  for (const auto& stream : streams_) {
    children.push_back(stream.get());
  }

  return children;
}

bool MinidumpFileWriter::WillWriteAtOffsetImpl(FileOffset offset) {
  DCHECK_EQ(state(), kStateFrozen);
  DCHECK_EQ(offset, 0);
  DCHECK_EQ(streams_.size(), stream_types_.size());

  auto directory_offset = streams_.empty() ? 0 : offset + sizeof(header_);
  if (!AssignIfInRange(&header_.StreamDirectoryRva, directory_offset)) {
    LOG(ERROR) << "offset " << directory_offset << " out of range";
    return false;
  }

  return MinidumpWritable::WillWriteAtOffsetImpl(offset);
}

bool MinidumpFileWriter::WriteObject(FileWriterInterface* file_writer) {
  DCHECK_EQ(state(), kStateWritable);
  DCHECK_EQ(streams_.size(), stream_types_.size());

  WritableIoVec iov;
  iov.iov_base = &header_;
  iov.iov_len = sizeof(header_);
  std::vector<WritableIoVec> iovecs(1, iov);

  for (const auto& stream : streams_) {
    iov.iov_base = stream->DirectoryListEntry();
    iov.iov_len = sizeof(MINIDUMP_DIRECTORY);
    iovecs.push_back(iov);
  }

  return file_writer->WriteIoVec(&iovecs);
}

}  // namespace crashpad
