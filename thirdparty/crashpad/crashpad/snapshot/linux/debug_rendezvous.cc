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

#include "snapshot/linux/debug_rendezvous.h"

#include <stdint.h>

#include <set>

#include "base/logging.h"

namespace crashpad {

namespace {

struct Traits32 {
  using Integer = int32_t;
  using Address = uint32_t;
};

struct Traits64 {
  using Integer = int64_t;
  using Address = uint64_t;
};

template <typename Traits>
struct DebugRendezvousSpecific {
  typename Traits::Integer r_version;
  typename Traits::Address r_map;
  typename Traits::Address r_brk;
  typename Traits::Integer r_state;
  typename Traits::Address r_ldbase;
};

template <typename Traits>
struct LinkEntrySpecific {
  typename Traits::Address l_addr;
  typename Traits::Address l_name;
  typename Traits::Address l_ld;
  typename Traits::Address l_next;
  typename Traits::Address l_prev;
};

template <typename Traits>
bool ReadLinkEntry(const ProcessMemoryRange& memory,
                   LinuxVMAddress* address,
                   DebugRendezvous::LinkEntry* entry_out) {
  LinkEntrySpecific<Traits> entry;
  if (!memory.Read(*address, sizeof(entry), &entry)) {
    return false;
  }

  std::string name;
  if (!memory.ReadCStringSizeLimited(entry.l_name, 4096, &name)) {
    name.clear();
  }

  entry_out->load_bias = entry.l_addr;
  entry_out->dynamic_array = entry.l_ld;
  entry_out->name.swap(name);

  *address = entry.l_next;
  return true;
}

}  // namespace

DebugRendezvous::LinkEntry::LinkEntry()
    : name(), load_bias(0), dynamic_array(0) {}

DebugRendezvous::DebugRendezvous()
    : modules_(), executable_(), initialized_() {}

DebugRendezvous::~DebugRendezvous() {}

bool DebugRendezvous::Initialize(const ProcessMemoryRange& memory,
                                 LinuxVMAddress address) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);
  if (!(memory.Is64Bit() ? InitializeSpecific<Traits64>(memory, address)
                         : InitializeSpecific<Traits32>(memory, address))) {
    return false;
  }
  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

const DebugRendezvous::LinkEntry* DebugRendezvous::Executable() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return &executable_;
}

const std::vector<DebugRendezvous::LinkEntry>& DebugRendezvous::Modules()
    const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return modules_;
}

template <typename Traits>
bool DebugRendezvous::InitializeSpecific(const ProcessMemoryRange& memory,
                                         LinuxVMAddress address) {
  DebugRendezvousSpecific<Traits> debug;
  if (!memory.Read(address, sizeof(debug), &debug)) {
    return false;
  }
  if (debug.r_version != 1) {
    LOG(ERROR) << "unexpected version " << debug.r_version;
    return false;
  }

  LinuxVMAddress link_entry_address = debug.r_map;
  if (!ReadLinkEntry<Traits>(memory, &link_entry_address, &executable_)) {
    return false;
  }

  std::set<LinuxVMAddress> visited;
  while (link_entry_address) {
    if (!visited.insert(link_entry_address).second) {
      LOG(ERROR) << "cycle at address 0x" << std::hex << link_entry_address;
      return false;
    }

    LinkEntry entry;
    if (!ReadLinkEntry<Traits>(memory, &link_entry_address, &entry)) {
      return false;
    }
    modules_.push_back(entry);
  }

  return true;
}

}  // namespace crashpad
