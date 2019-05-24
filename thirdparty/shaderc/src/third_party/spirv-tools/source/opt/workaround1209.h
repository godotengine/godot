// Copyright (c) 2018 Google Inc.
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

#ifndef SOURCE_OPT_WORKAROUND1209_H_
#define SOURCE_OPT_WORKAROUND1209_H_

#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class Workaround1209 : public Pass {
 public:
  const char* name() const override { return "workaround-1209"; }
  Status Process() override;

 private:
  // There is at least one driver where an OpUnreachable found in a loop is not
  // handled correctly.  Workaround that by changing the OpUnreachable into a
  // branch to the loop merge.
  //
  // Returns true if the code changed.
  bool RemoveOpUnreachableInLoops();
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_WORKAROUND1209_H_
