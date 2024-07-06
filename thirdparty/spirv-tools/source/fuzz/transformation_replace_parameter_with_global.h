// Copyright (c) 2020 Vasyl Teliman
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

#ifndef SOURCE_FUZZ_TRANSFORMATION_REPLACE_PARAMETER_WITH_GLOBAL_H_
#define SOURCE_FUZZ_TRANSFORMATION_REPLACE_PARAMETER_WITH_GLOBAL_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationReplaceParameterWithGlobal : public Transformation {
 public:
  explicit TransformationReplaceParameterWithGlobal(
      protobufs::TransformationReplaceParameterWithGlobal message);

  TransformationReplaceParameterWithGlobal(uint32_t function_type_fresh_id,
                                           uint32_t parameter_id,
                                           uint32_t global_variable_fresh_id);

  // - |function_type_fresh_id| is a fresh id.
  // - |parameter_id| is the result id of the parameter to replace.
  // - |global_variable_fresh_id| is a fresh id.
  // - |function_type_fresh_id| is not equal to |global_variable_fresh_id|.
  // - the function that contains |parameter_id| may not be an entry-point
  //   function.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // - Removes parameter with result id |parameter_id| from its function
  // - Adds a global variable to store the value for the parameter
  // - Add an OpStore instruction before each function call to
  //   store parameter's value into the variable
  // - Adds OpLoad at the beginning of the function to load the
  //   value from the variable into the old parameter's id
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

  // Returns true if the type of the parameter is supported by this
  // transformation.
  static bool IsParameterTypeSupported(opt::IRContext* ir_context,
                                       uint32_t param_type_id);

 private:
  protobufs::TransformationReplaceParameterWithGlobal message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_REPLACE_PARAMETER_WITH_GLOBAL_H_
