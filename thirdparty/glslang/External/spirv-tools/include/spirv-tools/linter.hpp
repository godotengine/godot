// Copyright (c) 2021 Google LLC.
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

#ifndef INCLUDE_SPIRV_TOOLS_LINTER_HPP_
#define INCLUDE_SPIRV_TOOLS_LINTER_HPP_

#include "libspirv.hpp"

namespace spvtools {

// C++ interface for SPIR-V linting functionalities. It wraps the context
// (including target environment and the corresponding SPIR-V grammar) and
// provides a method for linting.
//
// Instances of this class provides basic thread-safety guarantee.
class Linter {
 public:
  explicit Linter(spv_target_env env);

  ~Linter();

  // Sets the message consumer to the given |consumer|. The |consumer| will be
  // invoked once for each message communicated from the library.
  void SetMessageConsumer(MessageConsumer consumer);

  // Returns a reference to the registered message consumer.
  const MessageConsumer& Consumer() const;

  bool Run(const uint32_t* binary, size_t binary_size);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace spvtools

#endif  // INCLUDE_SPIRV_TOOLS_LINTER_HPP_
