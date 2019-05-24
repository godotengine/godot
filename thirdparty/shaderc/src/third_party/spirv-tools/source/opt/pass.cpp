// Copyright (c) 2017 The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
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

#include "source/opt/pass.h"

#include "source/opt/iterator.h"

namespace spvtools {
namespace opt {

namespace {

const uint32_t kTypePointerTypeIdInIdx = 1;

}  // namespace

Pass::Pass() : consumer_(nullptr), context_(nullptr), already_run_(false) {}

Pass::Status Pass::Run(IRContext* ctx) {
  if (already_run_) {
    return Status::Failure;
  }
  already_run_ = true;

  context_ = ctx;
  Pass::Status status = Process();
  context_ = nullptr;

  if (status == Status::SuccessWithChange) {
    ctx->InvalidateAnalysesExceptFor(GetPreservedAnalyses());
  }
  assert(ctx->IsConsistent());
  return status;
}

uint32_t Pass::GetPointeeTypeId(const Instruction* ptrInst) const {
  const uint32_t ptrTypeId = ptrInst->type_id();
  const Instruction* ptrTypeInst = get_def_use_mgr()->GetDef(ptrTypeId);
  return ptrTypeInst->GetSingleWordInOperand(kTypePointerTypeIdInIdx);
}

}  // namespace opt
}  // namespace spvtools
