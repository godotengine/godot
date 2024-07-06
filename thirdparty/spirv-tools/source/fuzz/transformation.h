// Copyright (c) 2019 Google LLC
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

#ifndef SOURCE_FUZZ_TRANSFORMATION_H_
#define SOURCE_FUZZ_TRANSFORMATION_H_

#include <memory>
#include <unordered_set>

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

// Rules for transformations
// -------------------------
//
// - Immutability: a transformation must be immutable.
// - Ability to copy and serialize: to ensure that a copy of a transformation,
//     possibly saved out to disk and read back again, is indistinguishable
//     from the original transformation, thus a transformation must depend
//     only on well-defined pieces of state, such as instruction ids.  It must
//     not rely on state such as pointers to instructions and blocks.
// - Determinism: the effect of a transformation on a module be a deterministic
//     function of the module and the transformation.  Any randomization should
//     be applied before creating the transformation, not during its
//     application.
// - Well-defined and precondition: the 'IsApplicable' method should only
//     return true if the transformation can be cleanly applied to the given
//     module, to mutate it into a valid and semantically-equivalent module, as
//     long as the module is initially valid.
// - Ability to test precondition on any valid module: 'IsApplicable' should be
//     designed so that it is safe to ask whether a transformation is
//     applicable to an arbitrary valid module.  For example, if a
//     transformation involves a block id, 'IsApplicable' should check whether
//     the module indeed has a block with that id, and return false if not.  It
//     must not assume that there is such a block.
// - Documented precondition: while the implementation of 'IsApplicable' should
//     should codify the precondition, the method should be commented in the
//     header file for a transformation with a precise English description of
//     the precondition.
// - Documented effect: while the implementation of 'Apply' should codify the
//     effect of the transformation, the method should be commented in the
//     header file for a transformation with a precise English description of
//     the effect.

class Transformation {
 public:
  virtual ~Transformation();

  // Factory method to obtain a transformation object from the protobuf
  // representation of a transformation given by |message|.
  static std::unique_ptr<Transformation> FromMessage(
      const protobufs::Transformation& message);

  // A precondition that determines whether the transformation can be cleanly
  // applied in a semantics-preserving manner to the SPIR-V module given by
  // |ir_context|, in the presence of facts and other contextual information
  // captured by |transformation_context|.
  //
  // Preconditions for individual transformations must be documented in the
  // associated header file using precise English. The transformation context
  // provides access to facts about the module that are known to be true, on
  // which the precondition may depend.
  virtual bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const = 0;

  // Requires that IsApplicable(ir_context, *transformation_context) holds.
  // Applies the transformation, mutating |ir_context| and possibly updating
  // |transformation_context| with new facts established by the transformation.
  virtual void Apply(opt::IRContext* ir_context,
                     TransformationContext* transformation_context) const = 0;

  // Returns the set of fresh ids that appear in the transformation's protobuf
  // message.
  virtual std::unordered_set<uint32_t> GetFreshIds() const = 0;

  // Turns the transformation into a protobuf message for serialization.
  virtual protobufs::Transformation ToMessage() const = 0;

  // Helper that returns true if and only if (a) |id| is a fresh id for the
  // module, and (b) |id| is not in |ids_used_by_this_transformation|, a set of
  // ids already known to be in use by a transformation.  This is useful when
  // checking id freshness for a transformation that uses many ids, all of which
  // must be distinct.
  static bool CheckIdIsFreshAndNotUsedByThisTransformation(
      uint32_t id, opt::IRContext* ir_context,
      std::set<uint32_t>* ids_used_by_this_transformation);
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_H_
