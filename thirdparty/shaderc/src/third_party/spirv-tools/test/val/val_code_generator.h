// Copyright (c) 2019 Google LLC.
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

// Utility class used to generate SPIR-V code strings for tests

#include <string>
#include <vector>

namespace spvtools {
namespace val {

struct EntryPoint {
  std::string name;
  std::string execution_model;
  std::string execution_modes;
  std::string body;
  std::string interfaces;
};

class CodeGenerator {
 public:
  static CodeGenerator GetDefaultShaderCodeGenerator();
  static CodeGenerator GetWebGPUShaderCodeGenerator();

  std::string Build() const;

  std::vector<EntryPoint> entry_points_;
  std::string capabilities_;
  std::string extensions_;
  std::string memory_model_;
  std::string before_types_;
  std::string types_;
  std::string after_types_;
  std::string add_at_the_end_;
};

}  // namespace val
}  // namespace spvtools
