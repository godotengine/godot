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

#ifndef SOURCE_LINT_LINTS_H_
#define SOURCE_LINT_LINTS_H_

#include "source/opt/ir_context.h"

namespace spvtools {
namespace lint {

// All of the functions in this namespace output to the error consumer in the
// |context| argument and return |true| if no errors are found. They do not
// modify the IR.
namespace lints {

bool CheckDivergentDerivatives(opt::IRContext* context);

}  // namespace lints
}  // namespace lint
}  // namespace spvtools

#endif  // SOURCE_LINT_LINTS_H_
