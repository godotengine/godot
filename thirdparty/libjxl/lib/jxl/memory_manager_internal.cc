// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/memory_manager_internal.h"

#include <jxl/memory_manager.h>
#include <jxl/types.h>

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>     // memcpy
#include <hwy/base.h>  // kMaxVectorSize

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/simd_util.h"

namespace jxl {

namespace {

void* MemoryManagerDefaultAlloc(void* opaque, size_t size) {
  return malloc(size);
}

void MemoryManagerDefaultFree(void* opaque, void* address) { free(address); }

}  // namespace

void* MemoryManagerAlloc(const JxlMemoryManager* memory_manager, size_t size) {
  return memory_manager->alloc(memory_manager->opaque, size);
}

void MemoryManagerFree(const JxlMemoryManager* memory_manager, void* address) {
  memory_manager->free(memory_manager->opaque, address);
}

Status MemoryManagerInit(JxlMemoryManager* self,
                         const JxlMemoryManager* memory_manager) {
  if (memory_manager) {
    *self = *memory_manager;
  } else {
    memset(self, 0, sizeof(*self));
  }
  bool is_default_alloc = (self->alloc == nullptr);
  bool is_default_free = (self->free == nullptr);
  if (is_default_alloc != is_default_free) {
    return false;
  }
  if (is_default_alloc) self->alloc = jxl::MemoryManagerDefaultAlloc;
  if (is_default_free) self->free = jxl::MemoryManagerDefaultFree;

  return true;
}

size_t BytesPerRow(const size_t xsize, const size_t sizeof_t) {
  // Special case: we don't allow any ops -> don't need extra padding/
  if (xsize == 0) {
    return 0;
  }

  const size_t vec_size = MaxVectorSize();
  size_t valid_bytes = xsize * sizeof_t;

  // Allow unaligned accesses starting at the last valid value.
  // Skip for the scalar case because no extra lanes will be loaded.
  if (vec_size != 0) {
    valid_bytes += vec_size - sizeof_t;
  }

  // Round up to vector and cache line size.
  const size_t align = std::max(vec_size, memory_manager_internal::kAlignment);
  size_t bytes_per_row = RoundUpTo(valid_bytes, align);

  // During the lengthy window before writes are committed to memory, CPUs
  // guard against read after write hazards by checking the address, but
  // only the lower 11 bits. We avoid a false dependency between writes to
  // consecutive rows by ensuring their sizes are not multiples of 2 KiB.
  // Avoid2K prevents the same problem for the planes of an Image3.
  if (bytes_per_row % memory_manager_internal::kAlias == 0) {
    bytes_per_row += align;
  }

  JXL_DASSERT(bytes_per_row % align == 0);
  return bytes_per_row;
}

StatusOr<AlignedMemory> AlignedMemory::Create(JxlMemoryManager* memory_manager,
                                              size_t size, size_t pre_padding) {
  JXL_ENSURE(pre_padding <= memory_manager_internal::kAlias);
  size_t allocation_size = size + pre_padding + memory_manager_internal::kAlias;
  if (size > allocation_size) {
    return JXL_FAILURE("Requested allocation is too large");
  }
  JXL_ENSURE(memory_manager);
  void* allocated =
      memory_manager->alloc(memory_manager->opaque, allocation_size);
  if (allocated == nullptr) {
    return JXL_FAILURE("Allocation failed");
  }
  return AlignedMemory(memory_manager, allocated, pre_padding);
}

AlignedMemory::AlignedMemory(JxlMemoryManager* memory_manager, void* allocation,
                             size_t pre_padding)
    : allocation_(allocation), memory_manager_(memory_manager) {
  // Congruence to `offset` (mod kAlias) reduces cache conflicts and load/store
  // stalls, especially with large allocations that would otherwise have similar
  // alignments.
  static std::atomic<uint32_t> next_group{0};
  size_t group =
      static_cast<size_t>(next_group.fetch_add(1, std::memory_order_relaxed));
  group &= (memory_manager_internal::kNumAlignmentGroups - 1);
  size_t offset = memory_manager_internal::kAlignment * group;

  // Actual allocation.
  uintptr_t address = reinterpret_cast<uintptr_t>(allocation) + pre_padding;

  // Aligned address, but might land before allocation (50%/50%) or not have
  // enough pre-padding.
  uintptr_t aligned_address =
      (address & ~(memory_manager_internal::kAlias - 1)) + offset;
  if (aligned_address < address)
    aligned_address += memory_manager_internal::kAlias;

  address_ = reinterpret_cast<void*>(aligned_address);  // NOLINT
}

AlignedMemory::AlignedMemory(AlignedMemory&& other) noexcept {
  allocation_ = other.allocation_;
  memory_manager_ = other.memory_manager_;
  address_ = other.address_;
  other.memory_manager_ = nullptr;
}

AlignedMemory& AlignedMemory::operator=(AlignedMemory&& other) noexcept {
  if (this == &other) return *this;
  if (memory_manager_ && allocation_) {
    memory_manager_->free(memory_manager_->opaque, allocation_);
  }
  allocation_ = other.allocation_;
  memory_manager_ = other.memory_manager_;
  address_ = other.address_;
  other.memory_manager_ = nullptr;
  return *this;
}

AlignedMemory::~AlignedMemory() {
  if (memory_manager_ == nullptr) return;
  memory_manager_->free(memory_manager_->opaque, allocation_);
}

}  // namespace jxl
