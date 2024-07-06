// Copyright (c) 2021 Google LLC
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

#ifndef SOURCE_OPT_DESC_SROA_UTIL_H_
#define SOURCE_OPT_DESC_SROA_UTIL_H_

#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {

// Provides functions for the descriptor array SROA.
namespace descsroautil {

// Returns true if |var| is an OpVariable instruction that represents a
// descriptor array.
bool IsDescriptorArray(IRContext* context, Instruction* var);

// Returns true if |type| is a type that could be used for a structured buffer
// as opposed to a type that would be used for a structure of resource
// descriptors.
bool IsTypeOfStructuredBuffer(IRContext* context, const Instruction* type);

// Returns the first index of the OpAccessChain instruction |access_chain| as
// a constant. Returns nullptr if it is not a constant.
const analysis::Constant* GetAccessChainIndexAsConst(IRContext* context,
                                                     Instruction* access_chain);

// Returns the number of elements of an OpVariable instruction |var| whose type
// must be a pointer to an array or a struct.
uint32_t GetNumberOfElementsForArrayOrStruct(IRContext* context,
                                             Instruction* var);

// Returns the first Indexes operand id of the OpAccessChain or
// OpInBoundsAccessChain instruction |access_chain|. The access chain must have
// at least 1 index.
uint32_t GetFirstIndexOfAccessChain(Instruction* access_chain);

}  // namespace descsroautil
}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_DESC_SROA_UTIL_H_
