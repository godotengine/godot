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

#ifndef CRASHPAD_SNAPSHOT_X86_CPUID_READER_H_
#define CRASHPAD_SNAPSHOT_X86_CPUID_READER_H_

#include <stdint.h>

#include <string>

namespace crashpad {
namespace internal {

//! \brief Reads x86-family CPU information by calling `cpuid`.
class CpuidReader {
 public:
  CpuidReader();
  ~CpuidReader();

  //! \see SystemSnapshot::CPURevision
  uint32_t Revision() const;

  //! \see SystemSnapshot::CPUVendor
  std::string Vendor() const { return vendor_; }

  //! \see SystemSnapshot::CPUX86Signature
  uint32_t Signature() const { return signature_; }

  //! \see SystemSnapshot::CPUX86Features
  uint64_t Features() const { return features_; }

  //! \see SystemSnapshot::CPUX86ExtendedFeatures
  uint64_t ExtendedFeatures() const { return extended_features_; }

  //! \see SystemSnapshot::CPUX86Leaf7Features
  uint32_t Leaf7Features() const;

  //! \see SystemSnapshot::NXEnabled
  bool NXEnabled() const { return (ExtendedFeatures() & (1 << 20)) != 0; }

  //! \see SystemSnapshot::CPUX86SupportsDAZ
  bool SupportsDAZ() const;

 private:
  void Cpuid(uint32_t cpuinfo[4], uint32_t leaf) const;

  uint64_t features_;
  uint64_t extended_features_;
  std::string vendor_;
  uint32_t max_leaf_;
  uint32_t signature_;
};

}  // namespace internal
}  // namespace crashpad

#endif  // CRASHPAD_SNAPSHOT_X86_CPUID_READER_H_
