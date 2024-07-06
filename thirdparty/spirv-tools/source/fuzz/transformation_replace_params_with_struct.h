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

#ifndef SOURCE_FUZZ_TRANSFORMATION_REPLACE_PARAMS_WITH_STRUCT_H_
#define SOURCE_FUZZ_TRANSFORMATION_REPLACE_PARAMS_WITH_STRUCT_H_

#include <map>

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationReplaceParamsWithStruct : public Transformation {
 public:
  explicit TransformationReplaceParamsWithStruct(
      protobufs::TransformationReplaceParamsWithStruct message);

  TransformationReplaceParamsWithStruct(
      const std::vector<uint32_t>& parameter_id,
      uint32_t fresh_function_type_id, uint32_t fresh_parameter_id,
      const std::map<uint32_t, uint32_t>& caller_id_to_fresh_composite_id);

  // - Each element of |parameter_id| is a valid result id of some
  //   OpFunctionParameter instruction. All parameter ids must correspond to
  //   parameters of the same function. That function may not be an entry-point
  //   function.
  // - Types of all parameters must be supported by this transformation (see
  //   IsParameterTypeSupported method).
  // - |parameter_id| may not be empty or contain duplicates.
  // - There must exist an OpTypeStruct instruction containing types of all
  //   replaced parameters. Type of the i'th component of the struct is equal
  //   to the type of the instruction with result id |parameter_id[i]|.
  // - |caller_id_to_fresh_composite_id| should contain a key for at least every
  //   result id of an OpFunctionCall instruction that calls the function.
  // - |fresh_function_type_id|, |fresh_parameter_id|,
  //   |caller_id_to_fresh_composite_id| are all fresh and unique ids.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // - Creates a new function parameter with result id |fresh_parameter_id|.
  //   Parameter's type is OpTypeStruct with each components type equal to the
  //   type of the replaced parameter.
  // - OpCompositeConstruct with result id from |fresh_composite_id| is inserted
  //   before each OpFunctionCall instruction.
  // - OpCompositeExtract with result id equal to the result id of the replaced
  //   parameter is created in the function.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

  // Returns true if parameter's type is supported by this transformation.
  static bool IsParameterTypeSupported(opt::IRContext* ir_context,
                                       uint32_t param_type_id);

 private:
  // Returns a result id of the OpTypeStruct instruction required by this
  // transformation (see docs on the IsApplicable method to learn more).
  uint32_t MaybeGetRequiredStructType(opt::IRContext* ir_context) const;

  // Returns a vector of indices of parameters to replace. Concretely, i'th
  // element is the index of the parameter with result id |parameter_id[i]| in
  // its function.
  std::vector<uint32_t> ComputeIndicesOfReplacedParameters(
      opt::IRContext* ir_context) const;

  protobufs::TransformationReplaceParamsWithStruct message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_REPLACE_PARAMS_WITH_STRUCT_H_
