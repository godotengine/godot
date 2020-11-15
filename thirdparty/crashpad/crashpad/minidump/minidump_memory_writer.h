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

#ifndef CRASHPAD_MINIDUMP_MINIDUMP_MEMORY_WRITER_H_
#define CRASHPAD_MINIDUMP_MINIDUMP_MEMORY_WRITER_H_

#include <windows.h>
#include <dbghelp.h>
#include <stdint.h>
#include <sys/types.h>

#include <memory>
#include <vector>

#include "base/macros.h"
#include "minidump/minidump_stream_writer.h"
#include "minidump/minidump_writable.h"
#include "snapshot/memory_snapshot.h"
#include "util/file/file_io.h"

namespace crashpad {

//! \brief The base class for writers of memory ranges pointed to by
//!     MINIDUMP_MEMORY_DESCRIPTOR objects in a minidump file.
class SnapshotMinidumpMemoryWriter : public internal::MinidumpWritable,
                                     public MemorySnapshot::Delegate {
 public:
  explicit SnapshotMinidumpMemoryWriter(const MemorySnapshot* memory_snapshot);
  ~SnapshotMinidumpMemoryWriter() override;

  //! \brief Returns a MINIDUMP_MEMORY_DESCRIPTOR referencing the data that this
  //!     object writes.
  //!
  //! This method is expected to be called by a MinidumpMemoryListWriter in
  //! order to obtain a MINIDUMP_MEMORY_DESCRIPTOR to include in its list.
  //!
  //! \note Valid in #kStateWritable.
  const MINIDUMP_MEMORY_DESCRIPTOR* MinidumpMemoryDescriptor() const;

  //! \brief Registers a memory descriptor as one that should point to the
  //!     object on which this method is called.
  //!
  //! This method is expected to be called by objects of other classes, when
  //! those other classes have their own memory descriptors that need to point
  //! to memory ranges within a minidump file. MinidumpThreadWriter is one such
  //! class. This method is public for this reason, otherwise it would suffice
  //! to be private.
  //!
  //! \note Valid in #kStateFrozen or any preceding state.
  void RegisterMemoryDescriptor(MINIDUMP_MEMORY_DESCRIPTOR* memory_descriptor);

  //! \brief Sets the underlying memory snapshot. Does not take ownership of \a
  //!     memory_snapshot.
  void SetSnapshot(const MemorySnapshot* memory_snapshot) {
    memory_snapshot_ = memory_snapshot;
  }

 private:
  friend class MinidumpMemoryListWriter;

  // MemorySnapshot::Delegate:
  bool MemorySnapshotDelegateRead(void* data, size_t size) override;

  // MinidumpWritable:
  bool Freeze() override;
  size_t SizeOfObject() final;
  bool WriteObject(FileWriterInterface* file_writer) override;

  //! \brief Returns the object’s desired byte-boundary alignment.
  //!
  //! Memory regions are aligned to a 16-byte boundary. The actual alignment
  //! requirements of any data within the memory region are unknown, and may be
  //! more or less strict than this depending on the platform.
  //!
  //! \return `16`.
  //!
  //! \note Valid in #kStateFrozen or any subsequent state.
  size_t Alignment() override;

  bool WillWriteAtOffsetImpl(FileOffset offset) override;

  //! \brief Returns the object’s desired write phase.
  //!
  //! Memory regions are written at the end of minidump files, because it is
  //! expected that unlike most other data in a minidump file, the contents of
  //! memory regions will be accessed sparsely.
  //!
  //! \return #kPhaseLate.
  //!
  //! \note Valid in any state.
  Phase WritePhase() final;

  //! \brief Gets the underlying memory snapshot that the memory writer will
  //!     write to the minidump.
  const MemorySnapshot* UnderlyingSnapshot() const { return memory_snapshot_; }

  MINIDUMP_MEMORY_DESCRIPTOR memory_descriptor_;

  // weak
  std::vector<MINIDUMP_MEMORY_DESCRIPTOR*> registered_memory_descriptors_;
  const MemorySnapshot* memory_snapshot_;
  FileWriterInterface* file_writer_;

  DISALLOW_COPY_AND_ASSIGN(SnapshotMinidumpMemoryWriter);
};

//! \brief The writer for a MINIDUMP_MEMORY_LIST stream in a minidump file,
//!     containing a list of MINIDUMP_MEMORY_DESCRIPTOR objects.
class MinidumpMemoryListWriter final : public internal::MinidumpStreamWriter {
 public:
  MinidumpMemoryListWriter();
  ~MinidumpMemoryListWriter() override;

  //! \brief Adds a concrete initialized SnapshotMinidumpMemoryWriter for each
  //!     memory snapshot in \a memory_snapshots to the MINIDUMP_MEMORY_LIST.
  //!
  //! Memory snapshots are added in the fashion of AddMemory().
  //!
  //! \param[in] memory_snapshots The memory snapshots to use as source data.
  //!
  //! \note Valid in #kStateMutable.
  void AddFromSnapshot(
      const std::vector<const MemorySnapshot*>& memory_snapshots);

  //! \brief Adds a SnapshotMinidumpMemoryWriter to the MINIDUMP_MEMORY_LIST.
  //!
  //! This object takes ownership of \a memory_writer and becomes its parent in
  //! the overall tree of internal::MinidumpWritable objects.
  //!
  //! \note Valid in #kStateMutable.
  void AddMemory(std::unique_ptr<SnapshotMinidumpMemoryWriter> memory_writer);

  //! \brief Adds a SnapshotMinidumpMemoryWriter that’s a child of another
  //!     internal::MinidumpWritable object to the MINIDUMP_MEMORY_LIST.
  //!
  //! \a memory_writer does not become a child of this object, but the
  //! MINIDUMP_MEMORY_LIST will still contain a MINIDUMP_MEMORY_DESCRIPTOR for
  //! it. \a memory_writer must be a child of another object in the
  //! internal::MinidumpWritable tree.
  //!
  //! This method exists to be called by objects that have their own
  //! SnapshotMinidumpMemoryWriter children but wish for them to also appear in
  //! the minidump file’s MINIDUMP_MEMORY_LIST. MinidumpThreadWriter, which has
  //! a SnapshotMinidumpMemoryWriter for thread stack memory, is an example.
  //!
  //! \note Valid in #kStateMutable.
  void AddNonOwnedMemory(SnapshotMinidumpMemoryWriter* memory_writer);

 protected:
  // MinidumpWritable:
  bool Freeze() override;
  size_t SizeOfObject() override;
  std::vector<MinidumpWritable*> Children() override;
  bool WriteObject(FileWriterInterface* file_writer) override;

  // MinidumpStreamWriter:
  MinidumpStreamType StreamType() const override;

  //! \brief Merges any overlapping and abutting memory ranges that were added
  //!     via AddFromSnapshot() and AddMemory() into single entries.
  //!
  //! This is expected to be called once just before writing, generally from
  //! Freeze().
  //!
  //! This function has the side-effect of merging owned ranges, dropping any
  //! owned ranges that overlap with non-owned ranges, removing empty ranges,
  //! and sorting all ranges by address.
  //!
  //! Per its name, this coalesces owned memory, however, this is not a complete
  //! solution for ensuring that no overlapping memory ranges are emitted in the
  //! minidump. In particular, if AddNonOwnedMemory() is used to add multiple
  //! overlapping ranges, then overlapping ranges will still be emitted to the
  //! minidump. Currently, AddNonOwnedMemory() is used only for adding thread
  //! stacks, so overlapping shouldn't be a problem in practice. For more
  //! details see https://crashpad.chromium.org/bug/61 and
  //! https://crrev.com/c/374539.
  void CoalesceOwnedMemory();

 private:
  //! \brief Drops children_ ranges that overlap non_owned_memory_writers_.
  void DropRangesThatOverlapNonOwned();

  std::vector<SnapshotMinidumpMemoryWriter*> non_owned_memory_writers_;  // weak
  std::vector<std::unique_ptr<SnapshotMinidumpMemoryWriter>> children_;
  std::vector<std::unique_ptr<const MemorySnapshot>>
      snapshots_created_during_merge_;
  std::vector<SnapshotMinidumpMemoryWriter*> all_memory_writers_;  // weak
  MINIDUMP_MEMORY_LIST memory_list_base_;

  DISALLOW_COPY_AND_ASSIGN(MinidumpMemoryListWriter);
};

}  // namespace crashpad

#endif  // CRASHPAD_MINIDUMP_MINIDUMP_MEMORY_WRITER_H_
