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

#include "minidump/minidump_thread_writer.h"

#include <utility>

#include "base/logging.h"
#include "minidump/minidump_context_writer.h"
#include "minidump/minidump_memory_writer.h"
#include "snapshot/memory_snapshot.h"
#include "snapshot/thread_snapshot.h"
#include "util/file/file_writer.h"
#include "util/numeric/safe_assignment.h"

namespace crashpad {

MinidumpThreadWriter::MinidumpThreadWriter()
    : MinidumpWritable(), thread_(), stack_(nullptr), context_(nullptr) {
}

MinidumpThreadWriter::~MinidumpThreadWriter() {
}

void MinidumpThreadWriter::InitializeFromSnapshot(
    const ThreadSnapshot* thread_snapshot,
    const MinidumpThreadIDMap* thread_id_map) {
  DCHECK_EQ(state(), kStateMutable);
  DCHECK(!stack_);
  DCHECK(!context_);

  auto thread_id_it = thread_id_map->find(thread_snapshot->ThreadID());
  DCHECK(thread_id_it != thread_id_map->end());
  SetThreadID(thread_id_it->second);

  SetSuspendCount(thread_snapshot->SuspendCount());
  SetPriority(thread_snapshot->Priority());
  SetTEB(thread_snapshot->ThreadSpecificDataAddress());

  const MemorySnapshot* stack_snapshot = thread_snapshot->Stack();
  if (stack_snapshot && stack_snapshot->Size() > 0) {
    std::unique_ptr<SnapshotMinidumpMemoryWriter> stack(
        new SnapshotMinidumpMemoryWriter(stack_snapshot));
    SetStack(std::move(stack));
  }

  std::unique_ptr<MinidumpContextWriter> context =
      MinidumpContextWriter::CreateFromSnapshot(thread_snapshot->Context());
  SetContext(std::move(context));
}

const MINIDUMP_THREAD* MinidumpThreadWriter::MinidumpThread() const {
  DCHECK_EQ(state(), kStateWritable);

  return &thread_;
}

void MinidumpThreadWriter::SetStack(
    std::unique_ptr<SnapshotMinidumpMemoryWriter> stack) {
  DCHECK_EQ(state(), kStateMutable);

  stack_ = std::move(stack);
}

void MinidumpThreadWriter::SetContext(
    std::unique_ptr<MinidumpContextWriter> context) {
  DCHECK_EQ(state(), kStateMutable);

  context_ = std::move(context);
}

bool MinidumpThreadWriter::Freeze() {
  DCHECK_EQ(state(), kStateMutable);
  CHECK(context_);

  if (!MinidumpWritable::Freeze()) {
    return false;
  }

  if (stack_) {
    stack_->RegisterMemoryDescriptor(&thread_.Stack);
  }

  context_->RegisterLocationDescriptor(&thread_.ThreadContext);

  return true;
}

size_t MinidumpThreadWriter::SizeOfObject() {
  DCHECK_GE(state(), kStateFrozen);

  // This object doesn’t directly write anything itself. Its MINIDUMP_THREAD is
  // written by its parent as part of a MINIDUMP_THREAD_LIST, and its children
  // are responsible for writing themselves.
  return 0;
}

std::vector<internal::MinidumpWritable*> MinidumpThreadWriter::Children() {
  DCHECK_GE(state(), kStateFrozen);
  DCHECK(context_);

  std::vector<MinidumpWritable*> children;
  if (stack_) {
    children.push_back(stack_.get());
  }
  children.push_back(context_.get());

  return children;
}

bool MinidumpThreadWriter::WriteObject(FileWriterInterface* file_writer) {
  DCHECK_EQ(state(), kStateWritable);

  // This object doesn’t directly write anything itself. Its MINIDUMP_THREAD is
  // written by its parent as part of a MINIDUMP_THREAD_LIST, and its children
  // are responsible for writing themselves.
  return true;
}

MinidumpThreadListWriter::MinidumpThreadListWriter()
    : MinidumpStreamWriter(),
      threads_(),
      memory_list_writer_(nullptr),
      thread_list_base_() {
}

MinidumpThreadListWriter::~MinidumpThreadListWriter() {
}

void MinidumpThreadListWriter::InitializeFromSnapshot(
    const std::vector<const ThreadSnapshot*>& thread_snapshots,
    MinidumpThreadIDMap* thread_id_map) {
  DCHECK_EQ(state(), kStateMutable);
  DCHECK(threads_.empty());

  BuildMinidumpThreadIDMap(thread_snapshots, thread_id_map);

  for (const ThreadSnapshot* thread_snapshot : thread_snapshots) {
    auto thread = std::make_unique<MinidumpThreadWriter>();
    thread->InitializeFromSnapshot(thread_snapshot, thread_id_map);
    AddThread(std::move(thread));
  }

  // Do this in a separate loop to keep the thread stacks earlier in the dump,
  // and together.
  for (const ThreadSnapshot* thread_snapshot : thread_snapshots)
    memory_list_writer_->AddFromSnapshot(thread_snapshot->ExtraMemory());
}

void MinidumpThreadListWriter::SetMemoryListWriter(
    MinidumpMemoryListWriter* memory_list_writer) {
  DCHECK_EQ(state(), kStateMutable);
  DCHECK(threads_.empty());

  memory_list_writer_ = memory_list_writer;
}

void MinidumpThreadListWriter::AddThread(
    std::unique_ptr<MinidumpThreadWriter> thread) {
  DCHECK_EQ(state(), kStateMutable);

  if (memory_list_writer_) {
    SnapshotMinidumpMemoryWriter* stack = thread->Stack();
    if (stack) {
      memory_list_writer_->AddNonOwnedMemory(stack);
    }
  }

  threads_.push_back(std::move(thread));
}

bool MinidumpThreadListWriter::Freeze() {
  DCHECK_EQ(state(), kStateMutable);

  if (!MinidumpStreamWriter::Freeze()) {
    return false;
  }

  size_t thread_count = threads_.size();
  if (!AssignIfInRange(&thread_list_base_.NumberOfThreads, thread_count)) {
    LOG(ERROR) << "thread_count " << thread_count << " out of range";
    return false;
  }

  return true;
}

size_t MinidumpThreadListWriter::SizeOfObject() {
  DCHECK_GE(state(), kStateFrozen);

  return sizeof(thread_list_base_) + threads_.size() * sizeof(MINIDUMP_THREAD);
}

std::vector<internal::MinidumpWritable*> MinidumpThreadListWriter::Children() {
  DCHECK_GE(state(), kStateFrozen);

  std::vector<MinidumpWritable*> children;
  for (const auto& thread : threads_) {
    children.push_back(thread.get());
  }

  return children;
}

bool MinidumpThreadListWriter::WriteObject(FileWriterInterface* file_writer) {
  DCHECK_EQ(state(), kStateWritable);

  WritableIoVec iov;
  iov.iov_base = &thread_list_base_;
  iov.iov_len = sizeof(thread_list_base_);
  std::vector<WritableIoVec> iovecs(1, iov);

  for (const auto& thread : threads_) {
    iov.iov_base = thread->MinidumpThread();
    iov.iov_len = sizeof(MINIDUMP_THREAD);
    iovecs.push_back(iov);
  }

  return file_writer->WriteIoVec(&iovecs);
}

MinidumpStreamType MinidumpThreadListWriter::StreamType() const {
  return kMinidumpStreamTypeThreadList;
}

}  // namespace crashpad
