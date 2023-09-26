// Copyright (c) 2016 Google Inc.
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

#ifndef SOURCE_OPT_NULL_PASS_H_
#define SOURCE_OPT_NULL_PASS_H_

#include "source/opt/module.h"
#include "source/opt/pass.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class NullPass : public Pass {
 public:
  const char* name() const override { return "null"; }
  Status Process() override { return Status::SuccessWithoutChange; }
};

}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_NULL_PASS_H_
