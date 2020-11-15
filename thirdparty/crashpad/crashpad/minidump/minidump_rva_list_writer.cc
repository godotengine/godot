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

#include "minidump/minidump_rva_list_writer.h"

#include <utility>

#include "base/logging.h"
#include "util/file/file_writer.h"
#include "util/numeric/safe_assignment.h"

namespace crashpad {
namespace internal {

MinidumpRVAListWriter::MinidumpRVAListWriter()
    : MinidumpWritable(),
      rva_list_base_(new MinidumpRVAList()),
      children_(),
      child_rvas_() {
}

MinidumpRVAListWriter::~MinidumpRVAListWriter() {
}

void MinidumpRVAListWriter::AddChild(std::unique_ptr<MinidumpWritable> child) {
  DCHECK_EQ(state(), kStateMutable);

  children_.push_back(std::move(child));
}

bool MinidumpRVAListWriter::Freeze() {
  DCHECK_EQ(state(), kStateMutable);
  DCHECK(child_rvas_.empty());

  if (!MinidumpWritable::Freeze()) {
    return false;
  }

  size_t child_count = children_.size();
  if (!AssignIfInRange(&rva_list_base_->count, child_count)) {
    LOG(ERROR) << "child_count " << child_count << " out of range";
    return false;
  }

  child_rvas_.resize(child_count);
  for (size_t index = 0; index < child_count; ++index) {
    children_[index]->RegisterRVA(&child_rvas_[index]);
  }

  return true;
}

size_t MinidumpRVAListWriter::SizeOfObject() {
  DCHECK_GE(state(), kStateFrozen);

  return sizeof(*rva_list_base_) + children_.size() * sizeof(RVA);
}

std::vector<MinidumpWritable*> MinidumpRVAListWriter::Children() {
  DCHECK_GE(state(), kStateFrozen);

  std::vector<MinidumpWritable*> children;
  for (const auto& child : children_) {
    children.push_back(child.get());
  }

  return children;
}

bool MinidumpRVAListWriter::WriteObject(FileWriterInterface* file_writer) {
  DCHECK_EQ(state(), kStateWritable);
  DCHECK_EQ(children_.size(), child_rvas_.size());

  WritableIoVec iov;
  iov.iov_base = rva_list_base_.get();
  iov.iov_len = sizeof(*rva_list_base_);
  std::vector<WritableIoVec> iovecs(1, iov);

  if (!child_rvas_.empty()) {
    iov.iov_base = &child_rvas_[0];
    iov.iov_len = child_rvas_.size() * sizeof(RVA);
    iovecs.push_back(iov);
  }

  return file_writer->WriteIoVec(&iovecs);
}

}  // namespace internal
}  // namespace crashpad
