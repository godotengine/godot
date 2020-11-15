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

#include "minidump/minidump_simple_string_dictionary_writer.h"

#include <utility>

#include "base/logging.h"
#include "util/file/file_writer.h"
#include "util/numeric/safe_assignment.h"

namespace crashpad {

MinidumpSimpleStringDictionaryEntryWriter::
    MinidumpSimpleStringDictionaryEntryWriter()
    : MinidumpWritable(), entry_(), key_(), value_() {
}

MinidumpSimpleStringDictionaryEntryWriter::
    ~MinidumpSimpleStringDictionaryEntryWriter() {
}

const MinidumpSimpleStringDictionaryEntry*
MinidumpSimpleStringDictionaryEntryWriter::
    GetMinidumpSimpleStringDictionaryEntry() const {
  DCHECK_EQ(state(), kStateWritable);

  return &entry_;
}

void MinidumpSimpleStringDictionaryEntryWriter::SetKeyValue(
    const std::string& key,
    const std::string& value) {
  DCHECK_EQ(state(), kStateMutable);

  key_.SetUTF8(key);
  value_.SetUTF8(value);
}

bool MinidumpSimpleStringDictionaryEntryWriter::Freeze() {
  DCHECK_EQ(state(), kStateMutable);

  if (!MinidumpWritable::Freeze()) {
    return false;
  }

  key_.RegisterRVA(&entry_.key);
  value_.RegisterRVA(&entry_.value);

  return true;
}

size_t MinidumpSimpleStringDictionaryEntryWriter::SizeOfObject() {
  DCHECK_GE(state(), kStateFrozen);

  // This object doesn’t directly write anything itself. Its
  // MinidumpSimpleStringDictionaryEntry is written by its parent as part of a
  // MinidumpSimpleStringDictionary, and its children are responsible for
  // writing themselves.
  return 0;
}

std::vector<internal::MinidumpWritable*>
MinidumpSimpleStringDictionaryEntryWriter::Children() {
  DCHECK_GE(state(), kStateFrozen);

  std::vector<MinidumpWritable*> children(1, &key_);
  children.push_back(&value_);
  return children;
}

bool MinidumpSimpleStringDictionaryEntryWriter::WriteObject(
    FileWriterInterface* file_writer) {
  DCHECK_EQ(state(), kStateWritable);

  // This object doesn’t directly write anything itself. Its
  // MinidumpSimpleStringDictionaryEntry is written by its parent as part of a
  // MinidumpSimpleStringDictionary, and its children are responsible for
  // writing themselves.
  return true;
}

MinidumpSimpleStringDictionaryWriter::MinidumpSimpleStringDictionaryWriter()
    : MinidumpWritable(),
      entries_(),
      simple_string_dictionary_base_(new MinidumpSimpleStringDictionary()) {
}

MinidumpSimpleStringDictionaryWriter::~MinidumpSimpleStringDictionaryWriter() {
  for (auto& item : entries_)
    delete item.second;
}

void MinidumpSimpleStringDictionaryWriter::InitializeFromMap(
    const std::map<std::string, std::string>& map) {
  DCHECK_EQ(state(), kStateMutable);
  DCHECK(entries_.empty());

  for (const auto& iterator : map) {
    auto entry = std::make_unique<MinidumpSimpleStringDictionaryEntryWriter>();
    entry->SetKeyValue(iterator.first, iterator.second);
    AddEntry(std::move(entry));
  }
}

void MinidumpSimpleStringDictionaryWriter::AddEntry(
    std::unique_ptr<MinidumpSimpleStringDictionaryEntryWriter> entry) {
  DCHECK_EQ(state(), kStateMutable);

  const std::string& key = entry->Key();
  auto iterator = entries_.find(key);
  if (iterator != entries_.end()) {
    delete iterator->second;
    iterator->second = entry.release();
  } else {
    entries_[key] = entry.release();
  }
}

bool MinidumpSimpleStringDictionaryWriter::IsUseful() const {
  return !entries_.empty();
}

bool MinidumpSimpleStringDictionaryWriter::Freeze() {
  DCHECK_EQ(state(), kStateMutable);

  if (!MinidumpWritable::Freeze()) {
    return false;
  }

  size_t entry_count = entries_.size();
  if (!AssignIfInRange(&simple_string_dictionary_base_->count, entry_count)) {
    LOG(ERROR) << "entry_count " << entry_count << " out of range";
    return false;
  }

  return true;
}

size_t MinidumpSimpleStringDictionaryWriter::SizeOfObject() {
  DCHECK_GE(state(), kStateFrozen);

  return sizeof(*simple_string_dictionary_base_) +
         entries_.size() * sizeof(MinidumpSimpleStringDictionaryEntry);
}

std::vector<internal::MinidumpWritable*>
MinidumpSimpleStringDictionaryWriter::Children() {
  DCHECK_GE(state(), kStateMutable);

  std::vector<MinidumpWritable*> children;
  for (const auto& key_entry : entries_) {
    children.push_back(key_entry.second);
  }

  return children;
}

bool MinidumpSimpleStringDictionaryWriter::WriteObject(
    FileWriterInterface* file_writer) {
  DCHECK_GE(state(), kStateWritable);

  WritableIoVec iov;
  iov.iov_base = simple_string_dictionary_base_.get();
  iov.iov_len = sizeof(*simple_string_dictionary_base_);
  std::vector<WritableIoVec> iovecs(1, iov);

  for (const auto& key_entry : entries_) {
    iov.iov_base = key_entry.second->GetMinidumpSimpleStringDictionaryEntry();
    iov.iov_len = sizeof(MinidumpSimpleStringDictionaryEntry);
    iovecs.push_back(iov);
  }

  return file_writer->WriteIoVec(&iovecs);
}

}  // namespace crashpad
