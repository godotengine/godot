// Copyright (c) 2020 Google LLC
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

#ifndef SOURCE_FUZZ_TRANSFORMATION_COMPUTE_DATA_SYNONYM_FACT_CLOSURE_H_
#define SOURCE_FUZZ_TRANSFORMATION_COMPUTE_DATA_SYNONYM_FACT_CLOSURE_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

class TransformationComputeDataSynonymFactClosure : public Transformation {
 public:
  explicit TransformationComputeDataSynonymFactClosure(
      protobufs::TransformationComputeDataSynonymFactClosure message);

  explicit TransformationComputeDataSynonymFactClosure(
      uint32_t maximum_equivalence_class_size);

  // This transformation is trivially applicable.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Forces the fact manager to compute a closure of data synonym facts, so that
  // facts implied by existing facts are deduced.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

 private:
  protobufs::TransformationComputeDataSynonymFactClosure message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_COMPUTE_DATA_SYNONYM_FACT_CLOSURE_H_
