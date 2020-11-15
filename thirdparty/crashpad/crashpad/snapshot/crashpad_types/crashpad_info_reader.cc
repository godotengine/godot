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

#include "snapshot/crashpad_types/crashpad_info_reader.h"

#include <type_traits>

#include "build/build_config.h"
#include "client/crashpad_info.h"
#include "util/linux/traits.h"
#include "util/misc/as_underlying_type.h"

namespace crashpad {

namespace {

void UnsetIfNotValidTriState(TriState* value) {
  switch (AsUnderlyingType(*value)) {
    case AsUnderlyingType(TriState::kUnset):
    case AsUnderlyingType(TriState::kEnabled):
    case AsUnderlyingType(TriState::kDisabled):
      return;
  }
  LOG(WARNING) << "Unsetting invalid TriState " << AsUnderlyingType(*value);
  *value = TriState::kUnset;
}

}  // namespace

class CrashpadInfoReader::InfoContainer {
 public:
  virtual ~InfoContainer() = default;

  virtual bool Read(const ProcessMemoryRange* memory, VMAddress address) = 0;

 protected:
  InfoContainer() = default;
};

template <class Traits>
class CrashpadInfoReader::InfoContainerSpecific : public InfoContainer {
 public:
  InfoContainerSpecific() : InfoContainer() {}
  ~InfoContainerSpecific() override = default;

  bool Read(const ProcessMemoryRange* memory, VMAddress address) override {
    if (!memory->Read(address,
                      offsetof(decltype(info), size) + sizeof(info.size),
                      &info)) {
      return false;
    }

    if (info.signature != CrashpadInfo::kSignature) {
      LOG(ERROR) << "invalid signature 0x" << std::hex << info.signature;
      return false;
    }

    if (!memory->Read(address,
                      std::min(VMSize{info.size}, VMSize{sizeof(info)}),
                      &info)) {
      return false;
    }

    if (info.size > sizeof(info)) {
      LOG(INFO) << "large crashpad info size " << info.size;
    }

    if (info.version != 1) {
      LOG(ERROR) << "unexpected version " << info.version;
      return false;
    }

    if (sizeof(info) > info.size) {
      memset(reinterpret_cast<char*>(&info) + info.size,
             0,
             sizeof(info) - info.size);
    }

    UnsetIfNotValidTriState(&info.crashpad_handler_behavior);
    UnsetIfNotValidTriState(&info.system_crash_reporter_forwarding);
    UnsetIfNotValidTriState(&info.gather_indirectly_referenced_memory);

    return true;
  }

  struct {
    uint32_t signature;
    uint32_t size;
    uint32_t version;
    uint32_t indirectly_referenced_memory_cap;
    uint32_t padding_0;
    TriState crashpad_handler_behavior;
    TriState system_crash_reporter_forwarding;
    TriState gather_indirectly_referenced_memory;
    uint8_t padding_1;
    typename Traits::Address extra_memory_ranges;
    typename Traits::Address simple_annotations;
    typename Traits::Address user_data_minidump_stream_head;
    typename Traits::Address annotations_list;
  } info;

#if defined(ARCH_CPU_64_BITS)
#define NATIVE_TRAITS Traits64
#else
#define NATIVE_TRAITS Traits32
#endif
  static_assert(!std::is_same<Traits, NATIVE_TRAITS>::value ||
                    sizeof(info) == sizeof(CrashpadInfo),
                "CrashpadInfo size mismtach");
#undef NATIVE_TRAITS
};

CrashpadInfoReader::CrashpadInfoReader()
    : container_(), is_64_bit_(false), initialized_() {}

CrashpadInfoReader::~CrashpadInfoReader() = default;

bool CrashpadInfoReader::Initialize(const ProcessMemoryRange* memory,
                                    VMAddress address) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);

  is_64_bit_ = memory->Is64Bit();

  std::unique_ptr<InfoContainer> new_container;
  if (is_64_bit_) {
    new_container = std::make_unique<InfoContainerSpecific<Traits64>>();
  } else {
    new_container = std::make_unique<InfoContainerSpecific<Traits32>>();
  }

  if (!new_container->Read(memory, address)) {
    return false;
  }
  container_ = std::move(new_container);

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

#define GET_MEMBER(name)                                                      \
  (is_64_bit_                                                                 \
       ? reinterpret_cast<InfoContainerSpecific<Traits64>*>(container_.get()) \
             ->info.name                                                      \
       : reinterpret_cast<InfoContainerSpecific<Traits32>*>(container_.get()) \
             ->info.name)

#define DEFINE_GETTER(type, method, member)          \
  type CrashpadInfoReader::method() {                \
    INITIALIZATION_STATE_DCHECK_VALID(initialized_); \
    return GET_MEMBER(member);                       \
  }

DEFINE_GETTER(TriState, CrashpadHandlerBehavior, crashpad_handler_behavior);

DEFINE_GETTER(TriState,
              SystemCrashReporterForwarding,
              system_crash_reporter_forwarding);

DEFINE_GETTER(TriState,
              GatherIndirectlyReferencedMemory,
              gather_indirectly_referenced_memory);

DEFINE_GETTER(uint32_t,
              IndirectlyReferencedMemoryCap,
              indirectly_referenced_memory_cap);

DEFINE_GETTER(VMAddress, ExtraMemoryRanges, extra_memory_ranges);

DEFINE_GETTER(VMAddress, SimpleAnnotations, simple_annotations);

DEFINE_GETTER(VMAddress, AnnotationsList, annotations_list);

DEFINE_GETTER(VMAddress,
              UserDataMinidumpStreamHead,
              user_data_minidump_stream_head);

#undef DEFINE_GETTER
#undef GET_MEMBER

}  // namespace crashpad
