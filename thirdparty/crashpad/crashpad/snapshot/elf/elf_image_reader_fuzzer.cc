// Copyright 2018 The Crashpad Authors. All rights reserved.
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

#include <inttypes.h>

#include "base/logging.h"
#include "snapshot/elf/elf_image_reader.h"
#include "util/process/process_memory.h"

using namespace crashpad;

class FakeProcessMemory : public ProcessMemory {
 public:
  FakeProcessMemory(const uint8_t* data, size_t size, VMAddress fake_base)
      : data_(data), size_(size), fake_base_(fake_base) {}

  ssize_t ReadUpTo(VMAddress address,
                   size_t size,
                   void* buffer) const override {
    VMAddress offset_in_data = address - fake_base_;
    if (offset_in_data > size_)
      return -1;
    ssize_t read_size = std::min(size_ - offset_in_data, size);
    memcpy(buffer, &data_[offset_in_data], read_size);
    return read_size;
  }

 private:
  const uint8_t* data_;
  size_t size_;
  VMAddress fake_base_;
};

extern "C" int LLVMFuzzerInitialize(int* argc, char*** argv) {
  // Swallow all logs to avoid spam.
  logging::SetLogMessageHandler(
      [](logging::LogSeverity, const char*, int, size_t, const std::string&) {
        return true;
      });
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  constexpr size_t kBase = 0x10000;
  FakeProcessMemory process_memory(data, size, kBase);
  ProcessMemoryRange process_memory_range;
  process_memory_range.Initialize(&process_memory, true, kBase, size);

  ElfImageReader reader;
  if (!reader.Initialize(process_memory_range, kBase))
    return 0;

  ElfImageReader::NoteReader::Result result;
  std::string note_name;
  std::string note_desc;
  ElfImageReader::NoteReader::NoteType note_type;
  auto notes = reader.Notes(-1);
  while ((result = notes->NextNote(&note_name, &note_type, &note_desc)) ==
         ElfImageReader::NoteReader::Result::kSuccess) {
    LOG(ERROR) << note_name << note_type << note_desc;
  }

  return 0;
}
