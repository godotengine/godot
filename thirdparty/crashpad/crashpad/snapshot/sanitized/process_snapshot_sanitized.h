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

#ifndef CRASHPAD_SNAPSHOT_SANITIZED_PROCESS_SNAPSHOT_SANITIZED_H_
#define CRASHPAD_SNAPSHOT_SANITIZED_PROCESS_SNAPSHOT_SANITIZED_H_

#include <memory>
#include <string>
#include <vector>

#include "base/macros.h"
#include "snapshot/exception_snapshot.h"
#include "snapshot/process_snapshot.h"
#include "snapshot/sanitized/module_snapshot_sanitized.h"
#include "snapshot/sanitized/thread_snapshot_sanitized.h"
#include "snapshot/thread_snapshot.h"
#include "snapshot/unloaded_module_snapshot.h"
#include "util/misc/address_types.h"
#include "util/misc/initialization_state_dcheck.h"
#include "util/misc/range_set.h"

namespace crashpad {

//! \brief A ProcessSnapshot which wraps and filters sensitive information from
//!     another ProcessSnapshot.
class ProcessSnapshotSanitized final : public ProcessSnapshot {
 public:
  ProcessSnapshotSanitized();
  ~ProcessSnapshotSanitized() override;

  //! \brief Initializes this object.
  //!
  //! This method must be successfully called before calling any other method on
  //! this object.
  //!
  //! \param[in] snapshot The ProcessSnapshot to sanitize.
  //! \param[in] annotations_whitelist A list of annotations names to allow to
  //!     be returned by AnnotationsSimpleMap() or from this object's module
  //!     snapshots. If `nullptr`, all annotations will be returned.
  //! \param[in] target_module_address An address in the target process'
  //!     address space within the bounds of a module to target. If the
  //!     crashing thread's context and stack do not contain any pointers into
  //!     this module's address range, this method will return `false`. If this
  //!     value is 0, this method will not check the context or stack for
  //!     references to any particular module.
  //! \param[in] sanitize_stacks If `true`, the MemorySnapshots for each
  //!     thread's stack will be filtered using an
  //!     internal::StackSnapshotSanitized.
  //! \return `false` if \a snapshot does not meet sanitization requirements and
  //!     should be filtered entirely. Otherwise `true`.
  bool Initialize(const ProcessSnapshot* snapshot,
                  const std::vector<std::string>* annotations_whitelist,
                  VMAddress target_module_address,
                  bool sanitize_stacks);

  // ProcessSnapshot:

  pid_t ProcessID() const override;
  pid_t ParentProcessID() const override;
  void SnapshotTime(timeval* snapshot_time) const override;
  void ProcessStartTime(timeval* start_time) const override;
  void ProcessCPUTimes(timeval* user_time, timeval* system_time) const override;
  void ReportID(UUID* report_id) const override;
  void ClientID(UUID* client_id) const override;
  const std::map<std::string, std::string>& AnnotationsSimpleMap()
      const override;
  const SystemSnapshot* System() const override;
  std::vector<const ThreadSnapshot*> Threads() const override;
  std::vector<const ModuleSnapshot*> Modules() const override;
  std::vector<UnloadedModuleSnapshot> UnloadedModules() const override;
  const ExceptionSnapshot* Exception() const override;
  std::vector<const MemoryMapRegionSnapshot*> MemoryMap() const override;
  std::vector<HandleSnapshot> Handles() const override;
  std::vector<const MemorySnapshot*> ExtraMemory() const override;

 private:
  // Only used when annotations_whitelist_ != nullptr.
  std::vector<std::unique_ptr<internal::ModuleSnapshotSanitized>> modules_;

  // Only used when sanitize_stacks_ == true.
  std::vector<std::unique_ptr<internal::ThreadSnapshotSanitized>> threads_;

  RangeSet address_ranges_;
  const ProcessSnapshot* snapshot_;
  const std::vector<std::string>* annotations_whitelist_;
  bool sanitize_stacks_;
  InitializationStateDcheck initialized_;

  DISALLOW_COPY_AND_ASSIGN(ProcessSnapshotSanitized);
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_SANITIZED_PROCESS_SNAPSHOT_SANITIZED_H_
