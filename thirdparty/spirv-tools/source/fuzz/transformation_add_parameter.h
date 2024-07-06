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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_PARAMETER_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADD_PARAMETER_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationAddParameter : public Transformation {
 public:
  explicit TransformationAddParameter(
      protobufs::TransformationAddParameter message);

  TransformationAddParameter(uint32_t function_id, uint32_t parameter_fresh_id,
                             uint32_t parameter_type_id,
                             std::map<uint32_t, uint32_t> call_parameter_ids,
                             uint32_t function_type_fresh_id);

  // - |function_id| must be a valid result id of some non-entry-point function
  //   in the module.
  // - |parameter_type_id| is a type id of the new parameter. The type must be
  //   supported by this transformation as specified by IsParameterTypeSupported
  //   function.
  // - |call_parameter_id| must map from every id of an OpFunctionCall
  //   instruction of this function to the id that will be passed as the new
  //   parameter at that call site. There could be no callers, therefore this
  //   map can be empty.
  // - |parameter_fresh_id| and |function_type_fresh_id| are fresh ids and are
  //   not equal.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // - Creates a new OpFunctionParameter instruction with result id
  //   |parameter_fresh_id| for the function with |function_id|.
  // - Adjusts function's type to include a new parameter.
  // - Adds an argument to every caller of the function to account for the added
  //   parameter. The argument is the value in |call_parameter_id| map.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

  // Returns true if the type of the parameter is supported by this
  // transformation.
  static bool IsParameterTypeSupported(opt::IRContext* ir_context,
                                       uint32_t type_id);

 private:
  protobufs::TransformationAddParameter message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_PARAMETER_H_
