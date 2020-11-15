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

#include "minidump/minidump_exception_writer.h"

#include <utility>

#include "base/logging.h"
#include "base/numerics/safe_conversions.h"
#include "minidump/minidump_context_writer.h"
#include "snapshot/exception_snapshot.h"
#include "util/file/file_writer.h"
#include "util/misc/arraysize_unsafe.h"

namespace crashpad {

MinidumpExceptionWriter::MinidumpExceptionWriter()
    : MinidumpStreamWriter(), exception_(), context_(nullptr) {
}

MinidumpExceptionWriter::~MinidumpExceptionWriter() {
}

void MinidumpExceptionWriter::InitializeFromSnapshot(
    const ExceptionSnapshot* exception_snapshot,
    const MinidumpThreadIDMap& thread_id_map) {
  DCHECK_EQ(state(), kStateMutable);
  DCHECK(!context_);

  auto thread_id_it = thread_id_map.find(exception_snapshot->ThreadID());
  DCHECK(thread_id_it != thread_id_map.end());
  SetThreadID(thread_id_it->second);

  SetExceptionCode(exception_snapshot->Exception());
  SetExceptionFlags(exception_snapshot->ExceptionInfo());
  SetExceptionAddress(exception_snapshot->ExceptionAddress());
  SetExceptionInformation(exception_snapshot->Codes());

  std::unique_ptr<MinidumpContextWriter> context =
      MinidumpContextWriter::CreateFromSnapshot(exception_snapshot->Context());
  SetContext(std::move(context));
}

void MinidumpExceptionWriter::SetContext(
    std::unique_ptr<MinidumpContextWriter> context) {
  DCHECK_EQ(state(), kStateMutable);

  context_ = std::move(context);
}

void MinidumpExceptionWriter::SetExceptionInformation(
    const std::vector<uint64_t>& exception_information) {
  DCHECK_EQ(state(), kStateMutable);

  const size_t parameters = exception_information.size();
  constexpr size_t kMaxParameters =
      ARRAYSIZE_UNSAFE(exception_.ExceptionRecord.ExceptionInformation);
  CHECK_LE(parameters, kMaxParameters);

  exception_.ExceptionRecord.NumberParameters =
      base::checked_cast<uint32_t>(parameters);
  size_t parameter = 0;
  for (; parameter < parameters; ++parameter) {
    exception_.ExceptionRecord.ExceptionInformation[parameter] =
        exception_information[parameter];
  }
  for (; parameter < kMaxParameters; ++parameter) {
    exception_.ExceptionRecord.ExceptionInformation[parameter] = 0;
  }
}

bool MinidumpExceptionWriter::Freeze() {
  DCHECK_EQ(state(), kStateMutable);
  CHECK(context_);

  if (!MinidumpStreamWriter::Freeze()) {
    return false;
  }

  context_->RegisterLocationDescriptor(&exception_.ThreadContext);

  return true;
}

size_t MinidumpExceptionWriter::SizeOfObject() {
  DCHECK_GE(state(), kStateFrozen);

  return sizeof(exception_);
}

std::vector<internal::MinidumpWritable*> MinidumpExceptionWriter::Children() {
  DCHECK_GE(state(), kStateFrozen);
  DCHECK(context_);

  std::vector<MinidumpWritable*> children;
  children.push_back(context_.get());

  return children;
}

bool MinidumpExceptionWriter::WriteObject(FileWriterInterface* file_writer) {
  DCHECK_EQ(state(), kStateWritable);

  return file_writer->Write(&exception_, sizeof(exception_));
}

MinidumpStreamType MinidumpExceptionWriter::StreamType() const {
  return kMinidumpStreamTypeException;
}

}  // namespace crashpad
