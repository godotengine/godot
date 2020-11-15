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

#include "minidump/minidump_writable.h"

#include <stdint.h>

#include "base/logging.h"
#include "util/file/file_writer.h"
#include "util/numeric/safe_assignment.h"

namespace {

constexpr size_t kMaximumAlignment = 16;

}  // namespace

namespace crashpad {
namespace internal {

MinidumpWritable::~MinidumpWritable() {
}

bool MinidumpWritable::WriteEverything(FileWriterInterface* file_writer) {
  DCHECK_EQ(state_, kStateMutable);

  if (!Freeze()) {
    return false;
  }

  DCHECK_EQ(state_, kStateFrozen);

  FileOffset offset = 0;
  std::vector<MinidumpWritable*> write_sequence;
  size_t size = WillWriteAtOffset(kPhaseEarly, &offset, &write_sequence);
  if (size == kInvalidSize) {
    return false;
  }

  offset += size;
  if (WillWriteAtOffset(kPhaseLate, &offset, &write_sequence) == kInvalidSize) {
    return false;
  }

  DCHECK_EQ(state_, kStateWritable);
  DCHECK_EQ(write_sequence.front(), this);

  for (MinidumpWritable* writable : write_sequence) {
    if (!writable->WritePaddingAndObject(file_writer)) {
      return false;
    }
  }

  DCHECK_EQ(state_, kStateWritten);

  return true;
}

void MinidumpWritable::RegisterRVA(RVA* rva) {
  DCHECK_LE(state_, kStateFrozen);

  registered_rvas_.push_back(rva);
}

void MinidumpWritable::RegisterLocationDescriptor(
    MINIDUMP_LOCATION_DESCRIPTOR* location_descriptor) {
  DCHECK_LE(state_, kStateFrozen);

  registered_location_descriptors_.push_back(location_descriptor);
}

MinidumpWritable::MinidumpWritable()
    : registered_rvas_(),
      registered_location_descriptors_(),
      leading_pad_bytes_(0),
      state_(kStateMutable) {
}

bool MinidumpWritable::Freeze() {
  DCHECK_EQ(state_, kStateMutable);
  state_ = kStateFrozen;

  std::vector<MinidumpWritable*> children = Children();
  for (MinidumpWritable* child : children) {
    if (!child->Freeze()) {
      return false;
    }
  }

  return true;
}

size_t MinidumpWritable::Alignment() {
  DCHECK_GE(state_, kStateFrozen);

  return 4;
}

std::vector<MinidumpWritable*> MinidumpWritable::Children() {
  DCHECK_GE(state_, kStateFrozen);

  return std::vector<MinidumpWritable*>();
}

MinidumpWritable::Phase MinidumpWritable::WritePhase() {
  return kPhaseEarly;
}

size_t MinidumpWritable::WillWriteAtOffset(
    Phase phase,
    FileOffset* offset,
    std::vector<MinidumpWritable*>* write_sequence) {
  FileOffset local_offset = *offset;
  CHECK_GE(local_offset, 0);

  size_t leading_pad_bytes_this_phase;
  size_t size;
  if (phase == WritePhase()) {
    DCHECK_EQ(state_, kStateFrozen);

    // Add this object to the sequence of MinidumpWritable objects to be
    // written.
    write_sequence->push_back(this);

    size = SizeOfObject();

    if (size > 0) {
      // Honor this object’s request to be aligned to a specific byte boundary.
      // Once the alignment is corrected, this object knows exactly what file
      // offset it will be written at.
      size_t alignment = Alignment();
      CHECK_LE(alignment, kMaximumAlignment);

      leading_pad_bytes_this_phase =
          (alignment - (local_offset % alignment)) % alignment;
      local_offset += leading_pad_bytes_this_phase;
      *offset = local_offset;
    } else {
      // If the object is size 0, alignment is of no concern.
      leading_pad_bytes_this_phase = 0;
    }
    leading_pad_bytes_ = leading_pad_bytes_this_phase;

    // Now that the file offset that this object will be written at is known,
    // let the subclass implementation know in case it’s interested.
    if (!WillWriteAtOffsetImpl(local_offset)) {
      return kInvalidSize;
    }

    // Populate the RVA fields in other objects that have registered to point to
    // this one. Typically, a parent object will have registered to point to its
    // children, but this can also occur where no parent-child relationship
    // exists.
    if (!registered_rvas_.empty() ||
        !registered_location_descriptors_.empty()) {
      RVA local_rva;
      if (!AssignIfInRange(&local_rva, local_offset)) {
        LOG(ERROR) << "offset " << local_offset << " out of range";
        return kInvalidSize;
      }

      for (RVA* rva : registered_rvas_) {
        *rva = local_rva;
      }

      if (!registered_location_descriptors_.empty()) {
        decltype(registered_location_descriptors_[0]->DataSize) local_size;
        if (!AssignIfInRange(&local_size, size)) {
          LOG(ERROR) << "size " << size << " out of range";
          return kInvalidSize;
        }

        for (MINIDUMP_LOCATION_DESCRIPTOR* location_descriptor :
                 registered_location_descriptors_) {
          location_descriptor->DataSize = local_size;
          location_descriptor->Rva = local_rva;
        }
      }
    }

    // This object is now considered writable. However, if it contains RVA or
    // MINIDUMP_LOCATION_DESCRIPTOR fields, they may not be fully updated yet,
    // because it’s the repsonsibility of these fields’ pointees to update them.
    // Once WillWriteAtOffset has completed running for both phases on an entire
    // tree, and the entire tree has moved into kStateFrozen, all RVA and
    // MINIDUMP_LOCATION_DESCRIPTOR fields within that tree will be populated.
    state_ = kStateWritable;
  } else {
    if (phase == kPhaseEarly) {
      DCHECK_EQ(state_, kStateFrozen);
    } else {
      DCHECK_EQ(state_, kStateWritable);
    }

    size = 0;
    leading_pad_bytes_this_phase = 0;
  }

  // Loop over children regardless of whether this object itself will write
  // during this phase. An object’s children are not required to be written
  // during the same phase as their parent.
  std::vector<MinidumpWritable*> children = Children();
  for (MinidumpWritable* child : children) {
    // Use “auto” here because it’s impossible to know whether size_t (size) or
    // FileOffset (local_offset) is the wider type, and thus what type the
    // result of adding these two variables will have.
    auto unaligned_child_offset = local_offset + size;
    FileOffset child_offset;
    if (!AssignIfInRange(&child_offset, unaligned_child_offset)) {
      LOG(ERROR) << "offset " << unaligned_child_offset << " out of range";
      return kInvalidSize;
    }

    size_t child_size =
        child->WillWriteAtOffset(phase, &child_offset, write_sequence);
    if (child_size == kInvalidSize) {
      return kInvalidSize;
    }

    size += child_size;
  }

  return leading_pad_bytes_this_phase + size;
}

bool MinidumpWritable::WillWriteAtOffsetImpl(FileOffset offset) {
  return true;
}

bool MinidumpWritable::WritePaddingAndObject(FileWriterInterface* file_writer) {
  DCHECK_EQ(state_, kStateWritable);

  // The number of elements in kZeroes must be at least one less than the
  // maximum Alignment() ever encountered.
  static constexpr uint8_t kZeroes[kMaximumAlignment - 1] = {};
  DCHECK_LE(leading_pad_bytes_, arraysize(kZeroes));

  if (leading_pad_bytes_) {
    if (!file_writer->Write(&kZeroes, leading_pad_bytes_)) {
      return false;
    }
  }

  if (!WriteObject(file_writer)) {
    return false;
  }

  state_ = kStateWritten;
  return true;
}

}  // namespace internal
}  // namespace crashpad
