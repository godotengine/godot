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

#ifndef CRASHPAD_MINIDUMP_MINIDUMP_WRITABLE_H_
#define CRASHPAD_MINIDUMP_MINIDUMP_WRITABLE_H_

#include <windows.h>
#include <dbghelp.h>
#include <sys/types.h>

#include <limits>
#include <vector>

#include "base/macros.h"
#include "util/file/file_io.h"

namespace crashpad {

class FileWriterInterface;

namespace internal {

//! \brief The base class for all content that might be written to a minidump
//!     file.
class MinidumpWritable {
 public:
  virtual ~MinidumpWritable();

  //! \brief Writes an object and all of its children to a minidump file.
  //!
  //! Use this on the root object of a tree of MinidumpWritable objects,
  //! typically on a MinidumpFileWriter object.
  //!
  //! \param[in] file_writer The file writer to receive the minidump file’s
  //!     content.
  //!
  //! \return `true` on success. `false` on failure, with an appropriate message
  //!     logged.
  //!
  //! \note Valid in #kStateMutable, and transitions the object and the entire
  //!     tree beneath it through all states to #kStateWritten.
  //!
  //! \note This method should rarely be overridden.
  virtual bool WriteEverything(FileWriterInterface* file_writer);

  //! \brief Registers a file offset pointer as one that should point to the
  //!     object on which this method is called.
  //!
  //! Once the file offset at which an object will be written is known (when it
  //! enters #kStateWritable), registered RVA pointers will be updated.
  //!
  //! \param[in] rva A pointer to storage for the file offset that should
  //!     contain this object’s writable file offset, once it is known.
  //!
  //! \note Valid in #kStateFrozen or any preceding state.
  //
  // This is public instead of protected because objects of derived classes need
  // to be able to register their own pointers with distinct objects.
  void RegisterRVA(RVA* rva);

  //! \brief Registers a location descriptor as one that should point to the
  //!     object on which this method is called.
  //!
  //! Once an object’s size and the file offset at it will be written is known
  //! (when it enters #kStateFrozen), the relevant data in registered location
  //! descriptors will be updated.
  //!
  //! \param[in] location_descriptor A pointer to a location descriptor that
  //!     should contain this object’s writable size and file offset, once they
  //!     are known.
  //!
  //! \note Valid in #kStateFrozen or any preceding state.
  //
  // This is public instead of protected because objects of derived classes need
  // to be able to register their own pointers with distinct objects.
  void RegisterLocationDescriptor(
      MINIDUMP_LOCATION_DESCRIPTOR* location_descriptor);

 protected:
  //! \brief Identifies the state of an object.
  //!
  //! Objects will normally transition through each of these states as they are
  //! created, populated with data, and then written to a minidump file.
  enum State {
    //! \brief The object’s properties can be modified.
    kStateMutable = 0,

    //! \brief The object is “frozen”.
    //!
    //! Its properties cannot be modified. Pointers to file offsets of other
    //! structures may not yet be valid.
    kStateFrozen,

    //! \brief The object is writable.
    //!
    //! The file offset at which it will be written is known. Pointers to file
    //! offsets of other structures are valid when all objects in a tree are in
    //! this state.
    kStateWritable,

    //! \brief The object has been written to a minidump file.
    kStateWritten,
  };

  //! \brief Identifies the phase during which an object will be written to a
  //!     minidump file.
  enum Phase {
    //! \brief Objects that are written to a minidump file “early”.
    //!
    //! The normal sequence is for an object to write itself and then write all
    //! of its children.
    kPhaseEarly = 0,

    //! \brief Objects that are written to a minidump file “late”.
    //!
    //! Some objects, such as those capturing memory region snapshots, are
    //! written to minidump files after all other objects. This “late” phase
    //! identifies such objects. This is useful to improve spatial locality in
    //! minidump files in accordance with expected access patterns: unlike most
    //! other data, memory snapshots are large and do not usually need to be
    //! consulted in their entirety in order to process a minidump file.
    kPhaseLate,
  };

  //! \brief A size value used to signal failure by methods that return
  //!     `size_t`.
  static constexpr size_t kInvalidSize = std::numeric_limits<size_t>::max();

  MinidumpWritable();

  //! \brief The state of the object.
  State state() const { return state_; }

  //! \brief Transitions the object from #kStateMutable to #kStateFrozen.
  //!
  //! The default implementation marks the object as frozen and recursively
  //! calls Freeze() on all of its children. Subclasses may override this method
  //! to perform processing that should only be done once callers have finished
  //! populating an object with data. Typically, a subclass implementation would
  //! call RegisterRVA() or RegisterLocationDescriptor() on other objects as
  //! appropriate, because at the time Freeze() runs, the in-memory locations of
  //! RVAs and location descriptors are known and will not change for the
  //! remaining duration of an object’s lifetime.
  //!
  //! \return `true` on success. `false` on failure, with an appropriate message
  //!     logged.
  virtual bool Freeze();

  //! \brief Returns the amount of space that this object will consume when
  //!     written to a minidump file, in bytes, not including any leading or
  //!     trailing padding necessary to maintain proper alignment.
  //!
  //! \note Valid in #kStateFrozen or any subsequent state.
  virtual size_t SizeOfObject() = 0;

  //! \brief Returns the object’s desired byte-boundary alignment.
  //!
  //! The default implementation returns `4`. Subclasses may override this as
  //! needed.
  //!
  //! \note Valid in #kStateFrozen or any subsequent state.
  virtual size_t Alignment();

  //! \brief Returns the object’s children.
  //!
  //! \note Valid in #kStateFrozen or any subsequent state.
  virtual std::vector<MinidumpWritable*> Children();

  //! \brief Returns the object’s desired write phase.
  //!
  //! The default implementation returns #kPhaseEarly. Subclasses may override
  //! this method to alter their write phase.
  //!
  //! \note Valid in any state.
  virtual Phase WritePhase();

  //! \brief Prepares the object to be written at a known file offset,
  //!     transitioning it from #kStateFrozen to #kStateWritable.
  //!
  //! This method is responsible for determining the final file offset of the
  //! object, which may be increased from \a offset to meet alignment
  //! requirements. It calls WillWriteAtOffsetImpl() for the benefit of
  //! subclasses. It populates all RVAs and location descriptors registered with
  //! it via RegisterRVA() and RegisterLocationDescriptor(). It also recurses
  //! into all known children.
  //!
  //! \param[in] phase The phase during which the object will be written. If
  //!     this does not match Phase(), processing is suppressed, although
  //!     recursive processing will still occur on all children. This addresses
  //!     the case where parents and children do not write in the same phase.
  //! \param[in] offset The file offset at which the object will be written. The
  //!     offset may need to be adjusted for alignment.
  //! \param[out] write_sequence This object will append itself to this list,
  //!     such that on return from a recursive tree of WillWriteAtOffset()
  //!     calls, elements of the vector will be organized in the sequence that
  //!     the objects will be written to the minidump file.
  //!
  //! \return The file size consumed by this object and all children, including
  //!     any padding inserted to meet alignment requirements. On failure,
  //!     #kInvalidSize, with an appropriate message logged.
  //!
  //! \note This method cannot be overridden. Subclasses that need to perform
  //!     processing when an object transitions to #kStateWritable should
  //!     implement WillWriteAtOffsetImpl(), which is called by this method.
  size_t WillWriteAtOffset(Phase phase,
                           FileOffset* offset,
                           std::vector<MinidumpWritable*>* write_sequence);

  //! \brief Called once an object’s writable file offset is determined, as it
  //!     transitions into #kStateWritable.
  //!
  //! Subclasses can override this method if they need to provide additional
  //! processing once their writable file offset is known. Typically, this will
  //! be done by subclasses that handle certain RVAs themselves instead of using
  //! the RegisterRVA() interface.
  //!
  //! \param[in] offset The file offset at which the object will be written. The
  //!     value passed to this method will already have been adjusted to meet
  //!     alignment requirements.
  //!
  //! \return `true` on success. `false` on error, indicating that the minidump
  //!     file should not be written.
  //!
  //! \note Valid in #kStateFrozen. The object will transition to
  //!     #kStateWritable after this method returns.
  virtual bool WillWriteAtOffsetImpl(FileOffset offset);

  //! \brief Writes the object, transitioning it from #kStateWritable to
  //!     #kStateWritten.
  //!
  //! Writes any padding necessary to meet alignment requirements, and then
  //! calls WriteObject() to write the object’s content.
  //!
  //! \param[in] file_writer The file writer to receive the object’s content.
  //!
  //! \return `true` on success. `false` on error with an appropriate message
  //!     logged.
  //!
  //! \note This method cannot be overridden. Subclasses must override
  //!     WriteObject().
  bool WritePaddingAndObject(FileWriterInterface* file_writer);

  //! \brief Writes the object’s content.
  //!
  //! \param[in] file_writer The file writer to receive the object’s content.
  //!
  //! \return `true` on success. `false` on error, indicating that the content
  //!     could not be written to the minidump file.
  //!
  //! \note Valid in #kStateWritable. The object will transition to
  //!     #kStateWritten after this method returns.
  virtual bool WriteObject(FileWriterInterface* file_writer) = 0;

 private:
  std::vector<RVA*> registered_rvas_;  // weak

  // weak
  std::vector<MINIDUMP_LOCATION_DESCRIPTOR*> registered_location_descriptors_;

  size_t leading_pad_bytes_;
  State state_;

  DISALLOW_COPY_AND_ASSIGN(MinidumpWritable);
};

}  // namespace internal
}  // namespace crashpad

#endif  // CRASHPAD_MINIDUMP_MINIDUMP_WRITABLE_H_
