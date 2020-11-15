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

#include "util/mach/task_memory.h"

#include <mach/mach_vm.h>
#include <string.h>

#include <algorithm>

#include "base/logging.h"
#include "base/mac/mach_logging.h"
#include "base/strings/stringprintf.h"
#include "util/stdlib/strnlen.h"

namespace crashpad {

TaskMemory::MappedMemory::~MappedMemory() {
}

bool TaskMemory::MappedMemory::ReadCString(
    size_t offset, std::string* string) const {
  if (offset >= user_size_) {
    LOG(WARNING) << "offset out of range";
    return false;
  }

  const char* string_base = reinterpret_cast<const char*>(data_) + offset;
  size_t max_length = user_size_ - offset;
  size_t string_length = strnlen(string_base, max_length);
  if (string_length == max_length) {
    LOG(WARNING) << "unterminated string";
    return false;
  }

  string->assign(string_base, string_length);
  return true;
}

TaskMemory::MappedMemory::MappedMemory(vm_address_t vm_address,
                                       size_t vm_size,
                                       size_t user_offset,
                                       size_t user_size)
    : vm_(vm_address, vm_size),
      data_(reinterpret_cast<const void*>(vm_address + user_offset)),
      user_size_(user_size) {
  vm_address_t vm_end = vm_address + vm_size;
  vm_address_t user_address = reinterpret_cast<vm_address_t>(data_);
  vm_address_t user_end = user_address + user_size;
  DCHECK_GE(user_address, vm_address);
  DCHECK_LE(user_address, vm_end);
  DCHECK_GE(user_end, vm_address);
  DCHECK_LE(user_end, vm_end);
}

TaskMemory::TaskMemory(task_t task) : task_(task) {
}

bool TaskMemory::Read(mach_vm_address_t address, size_t size, void* buffer) {
  std::unique_ptr<MappedMemory> memory = ReadMapped(address, size);
  if (!memory) {
    return false;
  }

  memcpy(buffer, memory->data(), size);
  return true;
}

std::unique_ptr<TaskMemory::MappedMemory> TaskMemory::ReadMapped(
    mach_vm_address_t address,
    size_t size) {
  if (size == 0) {
    return std::unique_ptr<MappedMemory>(new MappedMemory(0, 0, 0, 0));
  }

  mach_vm_address_t region_address = mach_vm_trunc_page(address);
  mach_vm_size_t region_size =
      mach_vm_round_page(address - region_address + size);

  vm_offset_t region;
  mach_msg_type_number_t region_count;
  kern_return_t kr =
      mach_vm_read(task_, region_address, region_size, &region, &region_count);
  if (kr != KERN_SUCCESS) {
    MACH_LOG(WARNING, kr) << base::StringPrintf(
        "mach_vm_read(0x%llx, 0x%llx)", region_address, region_size);
    return std::unique_ptr<MappedMemory>();
  }

  DCHECK_EQ(region_count, region_size);
  return std::unique_ptr<MappedMemory>(
      new MappedMemory(region, region_size, address - region_address, size));
}

bool TaskMemory::ReadCString(mach_vm_address_t address, std::string* string) {
  return ReadCStringInternal(address, false, 0, string);
}

bool TaskMemory::ReadCStringSizeLimited(mach_vm_address_t address,
                                        mach_vm_size_t size,
                                        std::string* string) {
  return ReadCStringInternal(address, true, size, string);
}

bool TaskMemory::ReadCStringInternal(mach_vm_address_t address,
                                     bool has_size,
                                     mach_vm_size_t size,
                                     std::string* string) {
  if (has_size) {
    if (size == 0)  {
      string->clear();
      return true;
    }
  } else {
    size = PAGE_SIZE;
  }

  std::string local_string;
  mach_vm_address_t read_address = address;
  do {
    mach_vm_size_t read_length =
        std::min(size, PAGE_SIZE - (read_address % PAGE_SIZE));
    std::unique_ptr<MappedMemory> read_region =
        ReadMapped(read_address, read_length);
    if (!read_region) {
      return false;
    }

    const char* read_region_data =
        reinterpret_cast<const char*>(read_region->data());
    size_t read_region_data_length = strnlen(read_region_data, read_length);
    local_string.append(read_region_data, read_region_data_length);
    if (read_region_data_length < read_length) {
      string->swap(local_string);
      return true;
    }

    if (has_size) {
      size -= read_length;
    }
    read_address = mach_vm_trunc_page(read_address + read_length);
  } while ((!has_size || size > 0) && read_address > address);

  LOG(WARNING) << base::StringPrintf("unterminated string at 0x%llx", address);
  return false;
}

}  // namespace crashpad
