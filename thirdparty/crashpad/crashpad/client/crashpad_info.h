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

#ifndef CRASHPAD_CLIENT_CRASHPAD_INFO_H_
#define CRASHPAD_CLIENT_CRASHPAD_INFO_H_

#include <stdint.h>

#include "base/macros.h"
#include "build/build_config.h"
#include "client/annotation_list.h"
#include "client/simple_address_range_bag.h"
#include "client/simple_string_dictionary.h"
#include "util/misc/tri_state.h"

#if defined(OS_WIN)
#include <windows.h>
#endif  // OS_WIN

namespace crashpad {

namespace internal {

//! \brief A linked list of blocks representing custom streams in the minidump,
//!     with addresses (and size) stored as uint64_t to simplify reading from
//!     the handler process.
struct UserDataMinidumpStreamListEntry {
  //! \brief The address of the next entry in the linked list.
  uint64_t next;

  //! \brief The base address of the memory block in the target process' address
  //!     space that represents the user data stream.
  uint64_t base_address;

  //! \brief The size of memory block in the target process' address space that
  //!     represents the user data stream.
  uint64_t size;

  //! \brief The stream type identifier.
  uint32_t stream_type;
};

}  // namespace internal

//! \brief A structure that can be used by a Crashpad-enabled program to
//!     provide information to the Crashpad crash handler.
//!
//! It is possible for one CrashpadInfo structure to appear in each loaded code
//! module in a process, but from the perspective of the user of the client
//! interface, there is only one global CrashpadInfo structure, located in the
//! module that contains the client interface code.
struct CrashpadInfo {
 public:
  //! \brief Returns the global CrashpadInfo structure.
  static CrashpadInfo* GetCrashpadInfo();

  CrashpadInfo();

  //! \brief Sets the bag of extra memory ranges to be included in the snapshot.
  //!
  //! Extra memory ranges may exist in \a address_range_bag at the time that
  //! this method is called, or they may be added, removed, or modified in \a
  //! address_range_bag after this method is called.
  //!
  //! TODO(scottmg) This is currently only supported on Windows.
  //!
  //! \param[in] address_range_bag A bag of address ranges. The CrashpadInfo
  //!     object does not take ownership of the SimpleAddressRangeBag object.
  //!     It is the caller’s responsibility to ensure that this pointer remains
  //!     valid while it is in effect for a CrashpadInfo object.
  void set_extra_memory_ranges(SimpleAddressRangeBag* address_range_bag) {
    extra_memory_ranges_ = address_range_bag;
  }

  //! \brief Sets the simple annotations dictionary.
  //!
  //! Simple annotations set on a CrashpadInfo structure are interpreted by
  //! Crashpad as module-level annotations.
  //!
  //! Annotations may exist in \a simple_annotations at the time that this
  //! method is called, or they may be added, removed, or modified in \a
  //! simple_annotations after this method is called.
  //!
  //! \param[in] simple_annotations A dictionary that maps string keys to string
  //!     values. The CrashpadInfo object does not take ownership of the
  //!     SimpleStringDictionary object. It is the caller’s responsibility to
  //!     ensure that this pointer remains valid while it is in effect for a
  //!     CrashpadInfo object.
  //!
  //! \sa simple_annotations()
  void set_simple_annotations(SimpleStringDictionary* simple_annotations) {
    simple_annotations_ = simple_annotations;
  }

  //! \return The simple annotations dictionary.
  //!
  //! \sa set_simple_annotations()
  SimpleStringDictionary* simple_annotations() const {
    return simple_annotations_;
  }

  //! \brief Sets the annotations list.
  //!
  //! Unlike the \a simple_annotations structure, the \a annotations can
  //! typed data and it is not limited to a dictionary form. Annotations are
  //! interpreted by Crashpad as module-level annotations.
  //!
  //! Annotations may exist in \a list at the time that this method is called,
  //! or they may be added, removed, or modified in \a list after this method is
  //! called.
  //!
  //! \param[in] list A list of set Annotation objects that maintain arbitrary,
  //!     typed key-value state. The CrashpadInfo object does not take ownership
  //!     of the AnnotationsList object. It is the caller’s responsibility to
  //!     ensure that this pointer remains valid while it is in effect for a
  //!     CrashpadInfo object.
  //!
  //! \sa annotations_list()
  //! \sa AnnotationList::Register()
  void set_annotations_list(AnnotationList* list) { annotations_list_ = list; }

  //! \return The annotations list.
  //!
  //! \sa set_annotations_list()
  //! \sa AnnotationList::Get()
  //! \sa AnnotationList::Register()
  AnnotationList* annotations_list() const { return annotations_list_; }

  //! \brief Enables or disables Crashpad handler processing.
  //!
  //! When handling an exception, the Crashpad handler will scan all modules in
  //! a process. The first one that has a CrashpadInfo structure populated with
  //! a value other than #kUnset for this field will dictate whether the handler
  //! is functional or not. If all modules with a CrashpadInfo structure specify
  //! #kUnset, the handler will be enabled. If disabled, the Crashpad handler
  //! will still run and receive exceptions, but will not take any action on an
  //! exception on its own behalf, except for the action necessary to determine
  //! that it has been disabled.
  //!
  //! The Crashpad handler should not normally be disabled. More commonly, it
  //! is appropriate to disable crash report upload by calling
  //! Settings::SetUploadsEnabled().
  void set_crashpad_handler_behavior(TriState crashpad_handler_behavior) {
    crashpad_handler_behavior_ = crashpad_handler_behavior;
  }

  //! \brief Enables or disables Crashpad forwarding of exceptions to the
  //!     system’s crash reporter.
  //!
  //! When handling an exception, the Crashpad handler will scan all modules in
  //! a process. The first one that has a CrashpadInfo structure populated with
  //! a value other than #kUnset for this field will dictate whether the
  //! exception is forwarded to the system’s crash reporter. If all modules with
  //! a CrashpadInfo structure specify #kUnset, forwarding will be enabled.
  //! Unless disabled, forwarding may still occur if the Crashpad handler is
  //! disabled by SetCrashpadHandlerState(). Even when forwarding is enabled,
  //! the Crashpad handler may choose not to forward all exceptions to the
  //! system’s crash reporter in cases where it has reason to believe that the
  //! system’s crash reporter would not normally have handled the exception in
  //! Crashpad’s absence.
  void set_system_crash_reporter_forwarding(
      TriState system_crash_reporter_forwarding) {
    system_crash_reporter_forwarding_ = system_crash_reporter_forwarding;
  }

  //! \brief Enables or disables Crashpad capturing indirectly referenced memory
  //!     in the minidump.
  //!
  //! When handling an exception, the Crashpad handler will scan all modules in
  //! a process. The first one that has a CrashpadInfo structure populated with
  //! a value other than #kUnset for this field will dictate whether the extra
  //! memory is captured.
  //!
  //! This causes Crashpad to include pages of data referenced by locals or
  //! other stack memory. Turning this on can increase the size of the minidump
  //! significantly.
  //!
  //! \param[in] gather_indirectly_referenced_memory Whether extra memory should
  //!     be gathered.
  //! \param[in] limit The amount of memory in bytes after which no more
  //!     indirectly gathered memory should be captured. This value is only used
  //!     when \a gather_indirectly_referenced_memory is TriState::kEnabled.
  void set_gather_indirectly_referenced_memory(
      TriState gather_indirectly_referenced_memory,
      uint32_t limit) {
    gather_indirectly_referenced_memory_ = gather_indirectly_referenced_memory;
    indirectly_referenced_memory_cap_ = limit;
  }

  //! \brief Adds a custom stream to the minidump.
  //!
  //! The memory block referenced by \a data and \a size will added to the
  //! minidump as separate stream with type \a stream_type. The memory referred
  //! to by \a data and \a size is owned by the caller and must remain valid
  //! while it is in effect for the CrashpadInfo object.
  //!
  //! Note that streams will appear in the minidump in the reverse order to
  //! which they are added.
  //!
  //! TODO(scottmg) This is currently only supported on Windows.
  //!
  //! \param[in] stream_type The stream type identifier to use. This should be
  //!     normally be larger than `MINIDUMP_STREAM_TYPE::LastReservedStream`
  //!     which is `0xffff`.
  //! \param[in] data The base pointer of the stream data.
  //! \param[in] size The size of the stream data.
  void AddUserDataMinidumpStream(uint32_t stream_type,
                                 const void* data,
                                 size_t size);

  enum : uint32_t {
    kSignature = 'CPad',
  };

 private:
  // The compiler won’t necessarily see anyone using these fields, but it
  // shouldn’t warn about that. These fields aren’t intended for use by the
  // process they’re found in, they’re supposed to be read by the crash
  // reporting process.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-private-field"
#endif

  // Fields present in version 1, subject to a check of the size_ field:
  uint32_t signature_;  // kSignature
  uint32_t size_;  // The size of the entire CrashpadInfo structure.
  uint32_t version_;  // kCrashpadInfoVersion
  uint32_t indirectly_referenced_memory_cap_;
  uint32_t padding_0_;
  TriState crashpad_handler_behavior_;
  TriState system_crash_reporter_forwarding_;
  TriState gather_indirectly_referenced_memory_;
  uint8_t padding_1_;
  SimpleAddressRangeBag* extra_memory_ranges_;  // weak
  SimpleStringDictionary* simple_annotations_;  // weak
  internal::UserDataMinidumpStreamListEntry* user_data_minidump_stream_head_;
  AnnotationList* annotations_list_;  // weak

  // It’s generally safe to add new fields without changing
  // kCrashpadInfoVersion, because readers should check size_ and ignore fields
  // that aren’t present, as well as unknown fields.
  //
  // Adding fields? Consider snapshot/crashpad_info_size_test_module.cc too.

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

  DISALLOW_COPY_AND_ASSIGN(CrashpadInfo);
};

}  // namespace crashpad

#endif  // CRASHPAD_CLIENT_CRASHPAD_INFO_H_
