// Copyright (c) 2017 Pierre Moreau
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

#ifndef INCLUDE_SPIRV_TOOLS_LINKER_HPP_
#define INCLUDE_SPIRV_TOOLS_LINKER_HPP_

#include <cstdint>

#include <memory>
#include <vector>

#include "libspirv.hpp"

namespace spvtools {

class LinkerOptions {
 public:
  // Returns whether a library or an executable should be produced by the
  // linking phase.
  //
  // All exported symbols are kept when creating a library, whereas they will
  // be removed when creating an executable.
  // The returned value will be true if creating a library, and false if
  // creating an executable.
  bool GetCreateLibrary() const { return create_library_; }

  // Sets whether a library or an executable should be produced.
  void SetCreateLibrary(bool create_library) {
    create_library_ = create_library;
  }

  // Returns whether to verify the uniqueness of the unique ids in the merged
  // context.
  bool GetVerifyIds() const { return verify_ids_; }

  // Sets whether to verify the uniqueness of the unique ids in the merged
  // context.
  void SetVerifyIds(bool verify_ids) { verify_ids_ = verify_ids; }

  // Returns whether to allow for imported symbols to have no corresponding
  // exported symbols
  bool GetAllowPartialLinkage() const { return allow_partial_linkage_; }

  // Sets whether to allow for imported symbols to have no corresponding
  // exported symbols
  void SetAllowPartialLinkage(bool allow_partial_linkage) {
    allow_partial_linkage_ = allow_partial_linkage;
  }

  bool GetUseHighestVersion() const { return use_highest_version_; }
  void SetUseHighestVersion(bool use_highest_vers) {
    use_highest_version_ = use_highest_vers;
  }

 private:
  bool create_library_{false};
  bool verify_ids_{false};
  bool allow_partial_linkage_{false};
  bool use_highest_version_{false};
};

// Links one or more SPIR-V modules into a new SPIR-V module. That is, combine
// several SPIR-V modules into one, resolving link dependencies between them.
//
// At least one binary has to be provided in |binaries|. Those binaries do not
// have to be valid, but they should be at least parseable.
// The functions can fail due to the following:
// * The given context was not initialised using `spvContextCreate()`;
// * No input modules were given;
// * One or more of those modules were not parseable;
// * The input modules used different addressing or memory models;
// * The ID or global variable number limit were exceeded;
// * Some entry points were defined multiple times;
// * Some imported symbols did not have an exported counterpart;
// * Possibly other reasons.
spv_result_t Link(const Context& context,
                  const std::vector<std::vector<uint32_t>>& binaries,
                  std::vector<uint32_t>* linked_binary,
                  const LinkerOptions& options = LinkerOptions());
spv_result_t Link(const Context& context, const uint32_t* const* binaries,
                  const size_t* binary_sizes, size_t num_binaries,
                  std::vector<uint32_t>* linked_binary,
                  const LinkerOptions& options = LinkerOptions());

}  // namespace spvtools

#endif  // INCLUDE_SPIRV_TOOLS_LINKER_HPP_
