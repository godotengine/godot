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

#ifndef CRASHPAD_SNAPSHOT_SNAPSHOT_CPU_ARCHITECTURE_H_
#define CRASHPAD_SNAPSHOT_SNAPSHOT_CPU_ARCHITECTURE_H_

namespace crashpad {

//! \brief A system’s CPU architecture.
//!
//! This can be used to represent the CPU architecture of an entire system
//! as in SystemSnapshot::CPUArchitecture(). It can also be used to represent
//! the architecture of a CPUContext structure in its CPUContext::architecture
//! field without reference to external data.
enum CPUArchitecture {
  //! \brief The CPU architecture is unknown.
  kCPUArchitectureUnknown = 0,

  //! \brief 32-bit x86.
  kCPUArchitectureX86,

  //! \brief x86_64.
  kCPUArchitectureX86_64,

  //! \brief 32-bit ARM.
  kCPUArchitectureARM,

  //! \brief 64-bit ARM.
  kCPUArchitectureARM64,

  //! \brief 32-bit MIPSEL.
  kCPUArchitectureMIPSEL,

  //! \brief 64-bit MIPSEL.
  kCPUArchitectureMIPS64EL
};

}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_SNAPSHOT_CPU_ARCHITECTURE_H_
