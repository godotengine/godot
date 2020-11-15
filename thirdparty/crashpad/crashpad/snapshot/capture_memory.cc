// Copyright 2016 The Crashpad Authors. All rights reserved.
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

#include "snapshot/capture_memory.h"

#include <stdint.h>

#include <limits>
#include <memory>

#include "snapshot/memory_snapshot.h"

namespace crashpad {
namespace internal {

namespace {

void MaybeCaptureMemoryAround(CaptureMemory::Delegate* delegate,
                              uint64_t address) {
  constexpr uint64_t non_address_offset = 0x10000;
  if (address < non_address_offset)
    return;

  const uint64_t max_address = delegate->Is64Bit() ?
      std::numeric_limits<uint64_t>::max() :
      std::numeric_limits<uint32_t>::max();
  if (address > max_address - non_address_offset)
    return;

  constexpr uint64_t kRegisterByteOffset = 128;
  const uint64_t target = address - kRegisterByteOffset;
  constexpr uint64_t size = 512;
  static_assert(kRegisterByteOffset <= size / 2,
                "negative offset too large");
  auto ranges =
      delegate->GetReadableRanges(CheckedRange<uint64_t>(target, size));
  for (const auto& range : ranges) {
    delegate->AddNewMemorySnapshot(range);
  }
}

template <class T>
void CaptureAtPointersInRange(uint8_t* buffer,
                              uint64_t buffer_size,
                              CaptureMemory::Delegate* delegate) {
  for (uint64_t address_offset = 0; address_offset < buffer_size;
       address_offset += sizeof(T)) {
    uint64_t target_address = *reinterpret_cast<T*>(&buffer[address_offset]);
    MaybeCaptureMemoryAround(delegate, target_address);
  }
}

}  // namespace

// static
void CaptureMemory::PointedToByContext(const CPUContext& context,
                                       Delegate* delegate) {
#if defined(ARCH_CPU_X86_FAMILY)
  if (context.architecture == kCPUArchitectureX86_64) {
    MaybeCaptureMemoryAround(delegate, context.x86_64->rax);
    MaybeCaptureMemoryAround(delegate, context.x86_64->rbx);
    MaybeCaptureMemoryAround(delegate, context.x86_64->rcx);
    MaybeCaptureMemoryAround(delegate, context.x86_64->rdx);
    MaybeCaptureMemoryAround(delegate, context.x86_64->rdi);
    MaybeCaptureMemoryAround(delegate, context.x86_64->rsi);
    MaybeCaptureMemoryAround(delegate, context.x86_64->rbp);
    MaybeCaptureMemoryAround(delegate, context.x86_64->r8);
    MaybeCaptureMemoryAround(delegate, context.x86_64->r9);
    MaybeCaptureMemoryAround(delegate, context.x86_64->r10);
    MaybeCaptureMemoryAround(delegate, context.x86_64->r11);
    MaybeCaptureMemoryAround(delegate, context.x86_64->r12);
    MaybeCaptureMemoryAround(delegate, context.x86_64->r13);
    MaybeCaptureMemoryAround(delegate, context.x86_64->r14);
    MaybeCaptureMemoryAround(delegate, context.x86_64->r15);
    MaybeCaptureMemoryAround(delegate, context.x86_64->rip);
  } else {
    MaybeCaptureMemoryAround(delegate, context.x86->eax);
    MaybeCaptureMemoryAround(delegate, context.x86->ebx);
    MaybeCaptureMemoryAround(delegate, context.x86->ecx);
    MaybeCaptureMemoryAround(delegate, context.x86->edx);
    MaybeCaptureMemoryAround(delegate, context.x86->edi);
    MaybeCaptureMemoryAround(delegate, context.x86->esi);
    MaybeCaptureMemoryAround(delegate, context.x86->ebp);
    MaybeCaptureMemoryAround(delegate, context.x86->eip);
  }
#elif defined(ARCH_CPU_ARM_FAMILY)
  if (context.architecture == kCPUArchitectureARM64) {
    MaybeCaptureMemoryAround(delegate, context.arm64->pc);
    for (size_t i = 0; i < arraysize(context.arm64->regs); ++i) {
      MaybeCaptureMemoryAround(delegate, context.arm64->regs[i]);
    }
  } else {
    MaybeCaptureMemoryAround(delegate, context.arm->pc);
    for (size_t i = 0; i < arraysize(context.arm->regs); ++i) {
      MaybeCaptureMemoryAround(delegate, context.arm->regs[i]);
    }
  }
#elif defined(ARCH_CPU_MIPS_FAMILY)
  for (size_t i = 0; i < arraysize(context.mipsel->regs); ++i) {
    MaybeCaptureMemoryAround(delegate, context.mipsel->regs[i]);
  }
#else
#error Port.
#endif
}

// static
void CaptureMemory::PointedToByMemoryRange(const MemorySnapshot& memory,
                                           Delegate* delegate) {
  if (memory.Size() == 0)
    return;

  const size_t alignment =
      delegate->Is64Bit() ? sizeof(uint64_t) : sizeof(uint32_t);
  if (memory.Address() % alignment != 0 || memory.Size() % alignment != 0) {
    LOG(ERROR) << "unaligned range";
    return;
  }

  std::unique_ptr<uint8_t[]> buffer(new uint8_t[memory.Size()]);
  if (!delegate->ReadMemory(memory.Address(), memory.Size(), buffer.get())) {
    LOG(ERROR) << "ReadMemory";
    return;
  }

  if (delegate->Is64Bit())
    CaptureAtPointersInRange<uint64_t>(buffer.get(), memory.Size(), delegate);
  else
    CaptureAtPointersInRange<uint32_t>(buffer.get(), memory.Size(), delegate);
}

}  // namespace internal
}  // namespace crashpad
