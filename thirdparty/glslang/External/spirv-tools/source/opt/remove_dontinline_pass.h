// Copyright (c) 2022 Google LLC
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

#ifndef SOURCE_OPT_REMOVE_DONTINLINE_PASS_H_
#define SOURCE_OPT_REMOVE_DONTINLINE_PASS_H_

#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class RemoveDontInline : public Pass {
 public:
  const char* name() const override { return "remove-dont-inline"; }
  Status Process() override;

 private:
  // Clears the DontInline function control from every function in the module.
  // Returns true of a change was made.
  bool ClearDontInlineFunctionControl();

  // Clears the DontInline function control from |function|.
  // Returns true of a change was made.
  bool ClearDontInlineFunctionControl(Function* function);
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_REMOVE_DONTINLINE_PASS_H_
